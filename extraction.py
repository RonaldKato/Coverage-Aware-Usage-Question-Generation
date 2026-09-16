"""
extraction.py
-------------
Two candidate-sentence extractors over the same raw corpus:

  * `extract_baseline`  -- reproduces the original paper's heuristic
    (Section 4.1): match the pattern "for + [verb in progressive tense]"
    (regex: `for\\s+\\w+ing`). This is a *precision-oriented, single-pattern*
    rule and is the root cause of the recall limitation we address.

  * `extract_broadened` -- a multi-pattern, regex-ensemble extractor that
    recognizes seven common linguistic realizations of usage statements
    (P1-P7 from data_generation.py) instead of one. It trades a small
    amount of precision for a large gain in recall, which is exactly the
    trade-off the limitation paragraph asks us to study.

Both extractors return the same schema so they can be plugged into the
rest of the pipeline interchangeably.
"""
import re
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List

ROOT = Path(__file__).resolve().parents[1]

BASELINE_RE = re.compile(r"\bfor\s+([a-z0-9 ,'-]+?ing\b[a-z0-9 ,'-]*)", re.IGNORECASE)

BROADENED_PATTERNS = {
    "P1_for_gerund": BASELINE_RE,
    "P2_when_clause": re.compile(r"\bwhen\s+(?:I|we)\s+([a-z0-9 ,'-]+)", re.IGNORECASE),
    "P3_ideal_adj": re.compile(r"\bideal\s+(?:for|choice if you'?re into)\s+([a-z0-9 ,'-]+)", re.IGNORECASE),
    "P4_during_prep": re.compile(r"\bduring\s+([a-z0-9 ,'-]+)", re.IGNORECASE),
    "P5_comparative": re.compile(r"\bbetter for\s+([a-z0-9 ,'-]+?)\s+because", re.IGNORECASE),
    "P6_bare_noun": re.compile(r"\bfor\s+([a-z0-9 ,'-]+?);", re.IGNORECASE),
    "P7_first_person_narrative": re.compile(r"\bfor\s+([a-z0-9 ,'-]+?),\s+I'?m glad|after a season of\s+([a-z0-9 ,'-]+)", re.IGNORECASE),
}


@dataclass
class Candidate:
    category: str
    sentence: str
    matched_pattern: str
    facet_id: int  # gold facet, carried through for evaluation only
    usage_gold: str
    aspect_gold: str
    is_applicable: bool


def _load_corpus(path: Path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def extract_baseline(records) -> List[Candidate]:
    out = []
    for r in records:
        if BASELINE_RE.search(r["sentence"]):
            out.append(Candidate(r["category"], r["sentence"], "P1_for_gerund",
                                  r["facet_id"], r["usage_gold"], r["aspect_gold"], r["is_applicable"]))
    return out


def extract_broadened(records) -> List[Candidate]:
    out = []
    for r in records:
        matched = None
        for name, pat in BROADENED_PATTERNS.items():
            if pat.search(r["sentence"]):
                matched = name
                break
        if matched:
            out.append(Candidate(r["category"], r["sentence"], matched,
                                  r["facet_id"], r["usage_gold"], r["aspect_gold"], r["is_applicable"]))
    return out


if __name__ == "__main__":
    records = _load_corpus(ROOT / "data" / "synthetic_reviews.jsonl")
    base = extract_baseline(records)
    broad = extract_broadened(records)
    print(f"Total sentences:      {len(records)}")
    print(f"Baseline (P1 only):   {len(base)} candidates")
    print(f"Broadened (P1-P7):    {len(broad)} candidates")

    Path(ROOT / "data" / "candidates_baseline.jsonl").write_text(
        "\n".join(json.dumps(asdict(c)) for c in base))
    Path(ROOT / "data" / "candidates_broadened.jsonl").write_text(
        "\n".join(json.dumps(asdict(c)) for c in broad))
