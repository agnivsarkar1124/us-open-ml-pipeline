"""
Runs the scouting agent over evals/questions.jsonl and reports:

  * verified rate         share of answers that passed every grounding check
  * first-try rate        share that passed without a retry
  * ungrounded-number rate share of numbers in FINAL answers that are untraceable
  * tool recall           share of questions where every expected tool was called
  * entity accuracy       share where predict_match was called on the right two players
  * latency and tokens    mean and p95 per question

    export ANTHROPIC_API_KEY=...
    python -m evals.run_eval --api-url http://127.0.0.1:8000 [--limit 10 | --per-category 2] [--model claude-haiku-5-5]
    python -m evals.run_eval --provider ollama --model qwen3:8b      # free, local
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import httpx

from agent.graph import build_graph, scout
from agent.tools import set_client


def p95(xs):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(0.95 * (len(xs) - 1))))] if xs else None


def evaluate_one(q: dict, graph) -> dict:
    try:
        r = scout(q["question"], graph=graph)
    except Exception as exc:
        return {"id": q["id"], "category": q["category"], "error": f"{type(exc).__name__}: {exc}"}
    called = [t["tool"] for t in r.get("tool_outputs", [])]
    predict_pairs = [frozenset((t["args"].get("player_a", ""), t["args"].get("player_b", "")))
                     for t in r.get("tool_outputs", []) if t["tool"] == "predict_match"
                     and "winner" in t["output"]]
    v = r.get("verification") or {}
    usage = r.get("usage", [])
    return {
        "id": q["id"],
        "category": q["category"],
        "question": q["question"],
        "status": r.get("status"),
        "attempts": r.get("attempts"),
        "tools_called": called,
        "tool_recall": all(t in called for t in q["expected_tools"]),
        "entity_correct": frozenset(q["players"]) in predict_pairs,
        "numbers_checked": v.get("numbers_checked", 0),
        "ungrounded_numbers": len(v.get("ungrounded_numbers", [])),
        "problems": v.get("problems", []),
        "latency_s": r.get("latency_s"),
        "input_tokens": sum(u["input_tokens"] for u in usage),
        "output_tokens": sum(u["output_tokens"] for u in usage),
        "report": r.get("report"),
    }


def summarize(rows: list[dict], model: str) -> dict:
    ok = [r for r in rows if "error" not in r]
    n = len(ok)

    def rate(key, pred=lambda v: bool(v)):
        return round(sum(pred(r[key]) for r in ok) / n, 3) if n else None

    numbers = sum(r["numbers_checked"] for r in ok)
    by_cat = {}
    for r in ok:
        c = by_cat.setdefault(r["category"], {"n": 0, "verified": 0})
        c["n"] += 1
        c["verified"] += r["status"] == "verified"
    return {
        "model": model,
        "questions": len(rows),
        "errors": len(rows) - n,
        "verified_rate": rate("status", lambda s: s == "verified"),
        "first_try_rate": round(sum(r["status"] == "verified" and r["attempts"] == 1 for r in ok) / n, 3) if n else None,
        "ungrounded_number_rate": round(sum(r["ungrounded_numbers"] for r in ok) / numbers, 4) if numbers else None,
        "numbers_checked": numbers,
        "tool_recall": rate("tool_recall"),
        "entity_accuracy": rate("entity_correct"),
        "latency_mean_s": round(statistics.mean(r["latency_s"] for r in ok), 2) if n else None,
        "latency_p95_s": p95([r["latency_s"] for r in ok]),
        "tokens_mean": round(statistics.mean(r["input_tokens"] + r["output_tokens"] for r in ok)) if n else None,
        "by_category": {k: {"n": v["n"], "verified_rate": round(v["verified"] / v["n"], 3)} for k, v in by_cat.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-url", default=os.environ.get("US_OPEN_API_URL", "http://127.0.0.1:8000"))
    ap.add_argument("--questions", default="evals/questions.jsonl")
    ap.add_argument("--limit", type=int, help="first N questions")
    ap.add_argument("--per-category", type=int, help="first N questions from EACH category (stratified sample)")
    ap.add_argument("--provider", default=os.environ.get("SCOUT_PROVIDER", "anthropic"),
                    choices=["anthropic", "ollama"])
    ap.add_argument("--model", default=os.environ.get("SCOUT_MODEL"))
    ap.add_argument("--out", default="eval")
    args = ap.parse_args()

    os.environ["SCOUT_PROVIDER"] = args.provider
    args.model = args.model or ("qwen3:8b" if args.provider == "ollama" else "claude-sonnet-5-5")
    os.environ["SCOUT_MODEL"] = args.model
    set_client(httpx.Client(base_url=args.api_url, timeout=120))
    graph = build_graph()
    questions = [json.loads(l) for l in Path(args.questions).read_text().splitlines() if l.strip()]
    if args.per_category:
        seen = {}
        questions = [q for q in questions
                     if seen.setdefault(q["category"], 0) < args.per_category
                     and not seen.__setitem__(q["category"], seen[q["category"]] + 1)]
    questions = questions[: args.limit] if args.limit else questions

    rows = []
    for i, q in enumerate(questions, 1):
        t0 = time.perf_counter()
        row = evaluate_one(q, graph)
        rows.append(row)
        print(f"[{i}/{len(questions)}] {q['id']:<18} {row.get('status', 'ERROR'):<10} "
              f"attempts={row.get('attempts')} {time.perf_counter() - t0:.1f}s")

    summary = summarize(rows, f"{args.provider}:{args.model}")
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    (out / "agent_results.json").write_text(json.dumps(summary, indent=2))
    (out / "agent_runs.jsonl").write_text("\n".join(json.dumps(r, default=str) for r in rows) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
