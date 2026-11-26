"""
Agent tests that need no API key and no network: a scripted chat model stands
in for the LLM, and an httpx MockTransport stands in for the prediction API.

Run: pytest -q tests/
"""

import json

import httpx
import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

from agent import build_graph, scout
from agent.grounding import allowed_numbers, verify
from agent.schema import ScoutingReport
from agent.tools import set_client

PREDICT = {
    "winner": "Jannik Sinner", "confidence": 55.0, "elo_diff": 56.3, "h2h_diff": -4,
    "matchup_breakdown": {
        "pA_name": "Jannik Sinner", "pB_name": "Carlos Alcaraz",
        "pA_elo": 2336.6, "pB_elo": 2280.3, "pA_hard_elo": 2327.1, "pB_hard_elo": 2171.1,
        "pA_h2h_wins": 6, "pB_h2h_wins": 10,
        "pA_serve_pts_won": 0.737, "pB_serve_pts_won": 0.693,
        "deciding_factors": ["Jannik Sinner is rated higher on hard courts (2327 vs 2171)"],
    },
}
PROFILES = {
    "Jannik Sinner": {"name": "Jannik Sinner", "elo": 2336.6, "hard_court_elo": 2327.1, "atp_rank": 2,
                      "serve_points_won_pct": 73.7, "return_points_won_pct": 42.5,
                      "elo_change_last_90_days": 35.3, "wins_in_last_10": 10},
    "Carlos Alcaraz": {"name": "Carlos Alcaraz", "elo": 2280.3, "hard_court_elo": 2171.1, "atp_rank": 1,
                       "serve_points_won_pct": 69.3, "return_points_won_pct": 41.0,
                       "elo_change_last_90_days": -12.4, "wins_in_last_10": 8},
}
H2H = {"p1": "Jannik Sinner", "p2": "Carlos Alcaraz", "p1_wins": 6, "p2_wins": 10, "total_matches": 16}


def fake_api(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/predict":
        return httpx.Response(200, json=PREDICT)
    if path == "/players":
        return httpx.Response(200, json={"players": list(PROFILES)})
    if path.startswith("/players/") and path.endswith("/profile"):
        name = path.split("/")[2]
        return httpx.Response(200, json=PROFILES[name]) if name in PROFILES else \
            httpx.Response(404, json={"detail": f"Unknown player: {name}"})
    if path == "/h2h":
        return httpx.Response(200, json=H2H)
    return httpx.Response(404, json={"detail": "no route"})


@pytest.fixture(autouse=True)
def api():
    set_client(httpx.Client(base_url="http://test", transport=httpx.MockTransport(fake_api)))


class ScriptedChat(BaseChatModel):
    """Returns pre-scripted AIMessages in order; bind_tools is a no-op."""
    script: list

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        return ChatResult(generations=[ChatGeneration(message=self.script.pop(0))])


def call(name, args, i):
    return {"name": name, "args": args, "id": f"call_{i}", "type": "tool_call"}


RESEARCH = [
    AIMessage(content="", tool_calls=[
        call("predict_match", {"player_a": "Jannik Sinner", "player_b": "Carlos Alcaraz"}, 1),
        call("player_profile", {"name": "Jannik Sinner"}, 2),
        call("player_profile", {"name": "Carlos Alcaraz"}, 3),
        call("head_to_head", {"player_a": "Jannik Sinner", "player_b": "Carlos Alcaraz"}, 4),
    ]),
    AIMessage(content="I have enough evidence."),
]


def report(**overrides):
    base = dict(
        player_a="Jannik Sinner", player_b="Carlos Alcaraz",
        predicted_winner="Jannik Sinner", win_probability_pct=55.0,
        key_factors=[
            {"claim": "Sinner's hard-court Elo is 2327 vs 2171, a 156-point edge", "source": "player_profile"},
            {"claim": "Sinner wins 73.7% of service points to Alcaraz's 69.3%", "source": "player_profile"},
            {"claim": "Alcaraz leads the head-to-head 10-6", "source": "head_to_head"},
        ],
        upset_path="Alcaraz's 10-6 head-to-head lead shows he can solve Sinner's game.",
        summary="The model makes Sinner a 55% favourite on hard courts.",
    )
    base.update(overrides)
    return ScoutingReport(**base)


def scripted_reports(*reports):
    """Returns reports in order; repeats the last one if the agent retries more."""
    queue = list(reports)
    return RunnableLambda(lambda msgs: {"parsed": queue.pop(0) if len(queue) > 1 else queue[0],
                                        "raw": AIMessage(content="")})


def run(*reports):
    g = build_graph(llm=ScriptedChat(script=list(RESEARCH)), report_llm=scripted_reports(*reports))
    return scout("Sinner vs Alcaraz at the US Open, who wins and why?", graph=g)


# ---------------------------------------------------------------- graph behaviour
def test_grounded_report_passes_first_try():
    r = run(report())
    assert r["status"] == "verified" and r["attempts"] == 1
    assert [t["tool"] for t in r["tool_outputs"]] == ["predict_match", "player_profile", "player_profile", "head_to_head"]


def test_hallucinated_number_triggers_retry_then_passes():
    bad = report(summary="Sinner has won 87% of his US Open matches and is a 55% favourite.")
    r = run(bad, report())
    assert r["status"] == "verified" and r["attempts"] == 2


def test_wrong_winner_is_caught():
    r = run(report(predicted_winner="Carlos Alcaraz"), report())
    assert r["attempts"] == 2 and r["status"] == "verified"


def test_persistent_hallucination_is_flagged_not_returned_as_verified():
    bad = report(summary="Sinner earned 23 million dollars this year.")
    r = run(bad, bad, bad)
    assert r["status"] == "unverified" and r["attempts"] == 3
    assert any(u["number"] == "23" for u in r["verification"]["ungrounded_numbers"])


def test_schema_failure_counts_as_failed_attempt():
    g = build_graph(llm=ScriptedChat(script=list(RESEARCH)),
                    report_llm=RunnableLambda(lambda m: {"parsed": None, "raw": AIMessage(content="")}))
    r = scout("Sinner vs Alcaraz?", graph=g)
    assert r["status"] == "unverified" and r["report"] is None


def test_unknown_player_returns_error_to_model_not_crash():
    script = [AIMessage(content="", tool_calls=[call("player_profile", {"name": "Nobody"}, 1)]),
              AIMessage(content="done")]
    g = build_graph(llm=ScriptedChat(script=script), report_llm=scripted_reports(report()))
    r = scout("Who is Nobody?", graph=g)
    assert "error" in r["tool_outputs"][0]["output"]
    assert r["status"] != "verified"  # predict_match never called -> cannot verify


# ---------------------------------------------------------------- grounding rules
def outputs():
    return [{"tool": "predict_match", "args": {}, "output": PREDICT},
            {"tool": "player_profile", "args": {}, "output": PROFILES["Jannik Sinner"]},
            {"tool": "head_to_head", "args": {}, "output": H2H}]


@pytest.mark.parametrize("n,ok", [
    (2327.1, True), (2327, True),      # exact and rounded
    (73.7, True),                      # fraction 0.737 shown as percent
    (45.0, True),                      # complement of 55% win probability
    (156, True),                       # derived difference 2327.1 - 2171.1
    (87, False), (2400, False),        # invented
])
def test_allowed_numbers(n, ok):
    from agent.grounding import _matches
    assert _matches(str(n), allowed_numbers(outputs(), "")) is ok


def test_factor_citing_uncalled_tool_fails():
    rep = report(key_factors=[{"claim": "Elo 2336.6", "source": "predict_match"},
                              {"claim": "10-6 lead", "source": "head_to_head"}]).model_dump()
    only_predict = [o for o in outputs() if o["tool"] == "predict_match"]
    v = verify(rep, only_predict, "")
    assert not v["passed"] and any("head_to_head" in p for p in v["problems"])


def test_signed_gap_between_profiles():
    from agent.grounding import _matches
    both = outputs() + [{"tool": "player_profile", "args": {}, "output": PROFILES["Carlos Alcaraz"]}]
    A = allowed_numbers(both, "")
    assert _matches("47.7", A)       # +35.3 vs -12.4 form gap
    assert not _matches("22.9", A)   # the wrong gap from subtracting absolute values
