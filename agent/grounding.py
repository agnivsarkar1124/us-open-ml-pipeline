"""
Grounding checks: every number in the report must be traceable to a tool output.

A number counts as grounded if it matches (at the precision written) any of:
  * a number in a tool output (including numbers inside strings like "73.7%"),
    its rounding, or its percentage form (0.737 -> 73.7)
  * the underdog's win probability (100 - predict_match confidence)
  * the difference between two like-for-like numbers (Player A's vs Player B's
    value of the same stat), e.g. an Elo gap
  * a number that appears in the user's question
  * small counting integers (0-5) and data-range years (2010-2027)

Beyond numbers, the report's predicted winner and win probability must match
predict_match exactly, and each key factor must cite a tool that was called.
"""

from __future__ import annotations

import itertools
import json
import re

NUMBER_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
ALWAYS_OK = set(range(0, 6)) | set(range(2010, 2028))
PROB_TOLERANCE = 0.5  # percentage points


def extract_numbers(text: str) -> list[str]:
    return [m.replace(",", "") for m in NUMBER_RE.findall(text)]


def _numbers_in(obj) -> list[float]:
    """All numbers anywhere in a JSON-like object, including inside strings."""
    out = []
    if isinstance(obj, bool) or obj is None:
        return out
    if isinstance(obj, (int, float)):
        out.append(abs(float(obj)))
    elif isinstance(obj, str):
        out.extend(float(n) for n in extract_numbers(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(_numbers_in(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out.extend(_numbers_in(v))
    return out


def _comparable_pairs(tool_outputs: list[dict]) -> list[tuple[float, float]]:
    """Pairs of like-for-like numbers whose difference is a legitimate derived stat:
    pA_x vs pB_x and p1_x vs p2_x inside one output, and the same field across two
    player profiles. Deliberately narrow: allowing differences of *any* two numbers
    makes almost every small integer 'grounded' by coincidence."""
    pairs = []

    def walk(d: dict):
        for k, v in d.items():
            if isinstance(v, dict):
                walk(v)
            for a_pre, b_pre in (("pA_", "pB_"), ("p1_", "p2_")):
                if k.startswith(a_pre):
                    other = d.get(b_pre + k[len(a_pre):])
                    if isinstance(v, (int, float)) and isinstance(other, (int, float)):
                        pairs.append((float(v), float(other)))

    for t in tool_outputs:
        if isinstance(t["output"], dict):
            walk(t["output"])
    profiles = [t["output"] for t in tool_outputs
                if t["tool"] == "player_profile" and isinstance(t["output"], dict) and "error" not in t["output"]]
    for p, q in itertools.combinations(profiles, 2):
        for k, v in p.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and isinstance(q.get(k), (int, float)):
                pairs.append((float(v), float(q[k])))
    return pairs


def allowed_numbers(tool_outputs: list[dict], question: str) -> set[float]:
    base = set(_numbers_in([t["output"] for t in tool_outputs]))
    variants = set(base)
    for v in base:
        if v <= 1:
            variants.add(v * 100)  # rates shown as percentages
    for t in tool_outputs:  # the underdog's win probability
        if t["tool"] == "predict_match" and isinstance(t["output"].get("confidence"), (int, float)):
            variants.add(100 - t["output"]["confidence"])
    for a, b in _comparable_pairs(tool_outputs):
        gap = abs(a - b)  # signs kept until here: +35.3 vs -12.4 is a 47.7 gap, not 22.9
        variants.add(gap * 100 if max(abs(a), abs(b)) <= 1 else gap)  # gaps between rates as points
    variants |= {float(n) for n in extract_numbers(question)}
    return variants


def _matches(written: str, allowed: set[float]) -> bool:
    x = float(written)
    if x.is_integer() and int(x) in ALWAYS_OK:
        return True
    decimals = len(written.split(".")[1]) if "." in written else 0
    tol = 0.5 * 10 ** -decimals + 1e-9
    return any(abs(a - x) <= tol for a in allowed)


def report_text_fields(report: dict) -> list[tuple[str, str]]:
    fields = [(f"key_factors[{i}]", f["claim"]) for i, f in enumerate(report["key_factors"])]
    fields += [("upset_path", report["upset_path"]), ("summary", report["summary"])]
    return fields


def verify(report: dict, tool_outputs: list[dict], question: str) -> dict:
    problems: list[str] = []
    called = {t["tool"] for t in tool_outputs}
    allowed = allowed_numbers(tool_outputs, question)

    # 1. Numbers in free text must be grounded
    ungrounded = []
    for field, text in report_text_fields(report):
        for n in extract_numbers(text):
            if not _matches(n, allowed):
                ungrounded.append({"field": field, "number": n, "text": text})
    for u in ungrounded:
        problems.append(f"{u['field']}: the number {u['number']} does not appear in any tool output")

    # 2. Winner and probability must match the model exactly
    preds = [t["output"] for t in tool_outputs if t["tool"] == "predict_match" and "winner" in t["output"]]
    if not preds:
        problems.append("predict_match was never called successfully; the prediction is unsupported")
    else:
        p = preds[-1]
        if report["predicted_winner"].strip().lower() != p["winner"].strip().lower():
            problems.append(f"predicted_winner is {report['predicted_winner']!r} but predict_match says {p['winner']!r}")
        if abs(report["win_probability_pct"] - p["confidence"]) > PROB_TOLERANCE:
            problems.append(f"win_probability_pct is {report['win_probability_pct']} but predict_match says {p['confidence']}")

    # 3. Each factor must cite a tool that was actually called
    for i, f in enumerate(report["key_factors"]):
        if f["source"] not in called:
            problems.append(f"key_factors[{i}] cites {f['source']}, which was never called")

    # 4. Free-text fields must actually be filled in (an empty string passes schema validation)
    for field in ("upset_path", "summary"):
        if len(report.get(field, "").strip()) < 20:
            problems.append(f"{field} is empty or too short; write at least one full sentence")

    n_numbers = sum(len(extract_numbers(t)) for _, t in report_text_fields(report))
    return {
        "passed": not problems,
        "problems": problems,
        "ungrounded_numbers": ungrounded,
        "numbers_checked": n_numbers,
    }


def tool_outputs_json(tool_outputs: list[dict]) -> str:
    return json.dumps([{"tool": t["tool"], "args": t["args"], "output": t["output"]} for t in tool_outputs], indent=1)
