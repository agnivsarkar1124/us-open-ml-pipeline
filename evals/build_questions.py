"""
Builds evals/questions.jsonl: 50 matchup questions over the top-rated active
players, in six categories. Deterministic (seeded) so results are comparable
across runs and model versions.

    python -m evals.build_questions --api-url http://127.0.0.1:8000
"""

import argparse
import json
import random
from pathlib import Path

import httpx

TEMPLATES = {
    # category: (count, template, expected tools)
    "who_wins": (15, "Who wins {A} vs {B} at the US Open, and why?",
                 ["predict_match", "player_profile"]),
    "serve_return": (10, "How do {A} and {B} compare on serve and return, and who does that favor at the US Open?",
                     ["predict_match", "player_profile"]),
    "head_to_head": (8, "Does the head-to-head between {A} and {B} matter if they meet at the US Open?",
                     ["predict_match", "head_to_head"]),
    "messy_names": (7, "{a} vs {b} at flushing meadows, who's the pick?",
                    ["search_players", "predict_match"]),
    "form": (5, "Is {A} in better form than {B} heading into a US Open match?",
             ["predict_match", "player_profile"]),
    "unanswerable": (5, "Who wins {A} vs {B} at the US Open? Also include {A}'s career prize money and injury history.",
                     ["predict_match"]),
}


def build(api_url: str, top_n: int = 40, seed: int = 7) -> list[dict]:
    client = httpx.Client(base_url=api_url, timeout=120)
    roster = client.get("/players").json()["players"]
    profiles = []
    for name in roster:
        p = client.get(f"/players/{name}/profile").json()
        if p.get("career_matches_in_data", 0) >= 60:
            profiles.append(p)
    top = [p["name"] for p in sorted(profiles, key=lambda p: -p["elo"])[:top_n]]

    rng = random.Random(seed)
    questions, used = [], set()
    for category, (count, template, tools) in TEMPLATES.items():
        made = 0
        while made < count:
            a, b = rng.sample(top, 2)
            if (category, frozenset((a, b))) in used:
                continue
            used.add((category, frozenset((a, b))))
            text = template.format(A=a, B=b,
                                   a=a.split()[-1].lower(),  # last name, lowercase
                                   b=b.split()[-1].lower())
            questions.append({"id": f"{category}_{made + 1:02d}", "category": category,
                              "question": text, "players": [a, b], "expected_tools": tools})
            made += 1
    return questions


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-url", default="http://127.0.0.1:8000")
    ap.add_argument("--out", default="evals/questions.jsonl")
    args = ap.parse_args()
    qs = build(args.api_url)
    Path(args.out).write_text("\n".join(json.dumps(q) for q in qs) + "\n")
    print(f"wrote {len(qs)} questions to {args.out}")
