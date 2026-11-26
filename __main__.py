"""
CLI:  python -m agent "Sinner vs Alcaraz at the US Open — who wins and why?"

Env: ANTHROPIC_API_KEY (required), US_OPEN_API_URL (default: the Render API),
     SCOUT_MODEL (default: claude-sonnet-5-5).
"""

import argparse
import json

from .graph import scout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--json", action="store_true", help="print the full result as JSON")
    args = ap.parse_args()

    r = scout(args.question)
    if args.json:
        print(json.dumps({k: r.get(k) for k in ("report", "verification", "status", "attempts",
                                                 "usage", "latency_s")}, indent=2, default=str))
        return

    rep, v = r.get("report"), r["verification"]
    print(f"\nStatus: {r['status'].upper()}  (attempts: {r['attempts']}, "
          f"{v['numbers_checked']} numbers checked, {r['latency_s']}s)")
    print("Tools called:", ", ".join(t["tool"] for t in r["tool_outputs"]) or "none")
    if rep:
        print(f"\n{rep['predicted_winner']} to win ({rep['win_probability_pct']}%)\n")
        for f in rep["key_factors"]:
            print(f"  • {f['claim']}  [{f['source']}]")
        print(f"\nUpset path: {rep['upset_path']}\n\n{rep['summary']}")
    if v["problems"]:
        print("\nVerification problems:\n" + "\n".join(f"  - {p}" for p in v["problems"]))


if __name__ == "__main__":
    main()
