r"""
LangGraph scouting agent.

    START -> agent --(tool calls)--> tools --> agent      (research loop)
               \--(done)--> write_report --> verify
                                 ^              |
                                 \--(fail, <3)--/        (grounding retry loop)
                                                |
                                          (pass or 3 tries) -> END

* agent:        LLM with the API tools bound; gathers evidence.
* tools:        runs tool calls and records raw outputs for verification.
* write_report: LLM with structured output (ScoutingReport); sees the raw tool
                outputs and, on retries, the verifier's feedback.
* verify:       deterministic code (agent/grounding.py) — no LLM grades itself.
"""

from __future__ import annotations

import json
import operator
import os
import time
from typing import Annotated, Any, Optional, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from .grounding import tool_outputs_json, verify
from .schema import ScoutingReport
from .tools import TOOLS

MAX_REPORT_ATTEMPTS = 3
MAX_TOOL_ROUNDS = 6

AGENT_PROMPT = """You are a tennis scouting analyst for the US Open (men's singles: hard court, best of five sets).
Gather evidence with the tools before answering:
- If a name is partial, lowercase or misspelled, call search_players first to get the exact name.
- Always call predict_match for the matchup in question.
- Call player_profile for both players, and head_to_head when the rivalry is relevant.
Do not state any statistic yourself. When you have enough evidence, reply with a one-line note
saying so (no tool call); a separate step writes the final report."""

REPORT_PROMPT = """Write a scouting report as structured output using ONLY the tool outputs below.
Rules:
- predicted_winner and win_probability_pct must equal predict_match's winner and confidence.
- Every number you write must appear in a tool output, or be a simple difference/sum of two of them.
- Each key factor must cite the tool its evidence came from.
- If the user asked for something the tools don't provide (e.g. prize money, injuries), say it isn't
  available in the data instead of guessing.
- The summary must directly answer the user's question.

User question:
{question}

Tool outputs (JSON):
{tool_outputs}"""


class ScoutState(TypedDict, total=False):
    question: str
    messages: Annotated[list[BaseMessage], add_messages]
    tool_outputs: Annotated[list[dict], operator.add]
    usage: Annotated[list[dict], operator.add]
    tool_rounds: int
    report: Optional[dict]
    verification: Optional[dict]
    attempts: int
    feedback: Optional[str]
    status: str


def default_llm():
    """Pick the model from env vars:
    SCOUT_PROVIDER=anthropic (default) -> Claude, needs ANTHROPIC_API_KEY
    SCOUT_PROVIDER=ollama              -> a local model via Ollama, free, no key
    SCOUT_MODEL overrides the model name for either provider."""
    provider = os.environ.get("SCOUT_PROVIDER", "anthropic").lower()
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=os.environ.get("SCOUT_MODEL", "qwen3:8b"),
            temperature=0,
            # Thinking models (e.g. qwen3) otherwise write long hidden reasoning before
            # every reply, which is most of the latency on a laptop. SCOUT_THINK=1 re-enables.
            reasoning=os.environ.get("SCOUT_THINK", "0") == "1",
            num_ctx=int(os.environ.get("SCOUT_NUM_CTX", "8192")),  # fits tool outputs; smaller = faster
            base_url=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"),
        )
    from langchain_anthropic import ChatAnthropic

    # No temperature override: current Claude models reject non-default values.
    return ChatAnthropic(
        model=os.environ.get("SCOUT_MODEL", "claude-sonnet-5-5"),
        max_tokens=2000,
        timeout=60,
        max_retries=2,
    )


def _usage(msg: Any, node: str) -> list[dict]:
    meta = getattr(msg, "usage_metadata", None)
    if not meta:
        return []
    return [{"node": node, "input_tokens": meta.get("input_tokens", 0),
             "output_tokens": meta.get("output_tokens", 0)}]


def build_graph(llm=None, report_llm=None):
    """llm: chat model supporting bind_tools. report_llm: runnable returning
    {"parsed": ScoutingReport | None, "raw": AIMessage}. Both injectable for tests."""
    llm = llm or default_llm()
    agent_llm = llm.bind_tools(TOOLS)
    # json_schema = native structured outputs; forced tool calling isn't supported on current models
    report_llm = report_llm or llm.with_structured_output(ScoutingReport, method="json_schema", include_raw=True)
    tools_by_name = {t.name: t for t in TOOLS}

    def agent(state: ScoutState) -> dict:
        msg = agent_llm.invoke([SystemMessage(AGENT_PROMPT)] + state["messages"])
        return {"messages": [msg], "usage": _usage(msg, "agent")}

    def tools(state: ScoutState) -> dict:
        last: AIMessage = state["messages"][-1]
        msgs, outputs = [], []
        for call in last.tool_calls:
            try:
                out = tools_by_name[call["name"]].invoke(call["args"])
            except Exception as exc:  # surface tool failures to the model instead of crashing
                out = {"error": f"{type(exc).__name__}: {exc}"}
            outputs.append({"tool": call["name"], "args": call["args"], "output": out})
            msgs.append(ToolMessage(content=json.dumps(out), tool_call_id=call["id"], name=call["name"]))
        return {"messages": msgs, "tool_outputs": outputs,
                "tool_rounds": state.get("tool_rounds", 0) + 1}

    def route_after_agent(state: ScoutState) -> str:
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None) and state.get("tool_rounds", 0) < MAX_TOOL_ROUNDS:
            return "tools"
        return "write_report"

    def write_report(state: ScoutState) -> dict:
        prompt = REPORT_PROMPT.format(question=state["question"],
                                      tool_outputs=tool_outputs_json(state.get("tool_outputs", [])))
        msgs = [HumanMessage(prompt)]
        if state.get("feedback"):
            msgs.append(HumanMessage(
                "Your previous report FAILED verification. Fix every problem below and rewrite it:\n"
                + state["feedback"]))
        result = report_llm.invoke(msgs)
        parsed = result.get("parsed")
        report = parsed.model_dump() if parsed is not None else None
        return {"report": report, "usage": _usage(result.get("raw"), "write_report")}

    def verify_node(state: ScoutState) -> dict:
        attempts = state.get("attempts", 0) + 1
        if state.get("report") is None:
            v = {"passed": False, "problems": ["Output did not match the ScoutingReport schema"],
                 "ungrounded_numbers": [], "numbers_checked": 0}
        else:
            v = verify(state["report"], state.get("tool_outputs", []), state["question"])
        if v["passed"]:
            status = "verified"
        elif attempts >= MAX_REPORT_ATTEMPTS:
            status = "unverified"  # never silently return a failing report
        else:
            status = "retrying"
        return {"verification": v, "attempts": attempts, "status": status,
                "feedback": "\n".join(f"- {p}" for p in v["problems"]) or None}

    def route_after_verify(state: ScoutState) -> str:
        return "write_report" if state["status"] == "retrying" else END

    g = StateGraph(ScoutState)
    g.add_node("agent", agent)
    g.add_node("tools", tools)
    g.add_node("write_report", write_report)
    g.add_node("verify", verify_node)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route_after_agent, {"tools": "tools", "write_report": "write_report"})
    g.add_edge("tools", "agent")
    g.add_edge("write_report", "verify")
    g.add_conditional_edges("verify", route_after_verify, {"write_report": "write_report", END: END})
    return g.compile()


def scout(question: str, graph=None) -> dict:
    """Run the agent on one question. Returns the final state plus timing."""
    graph = graph or build_graph()
    t0 = time.perf_counter()
    final = graph.invoke({"question": question, "messages": [HumanMessage(question)],
                          "tool_outputs": [], "usage": [], "attempts": 0},
                         config={"recursion_limit": 40})
    final["latency_s"] = round(time.perf_counter() - t0, 2)
    return final
