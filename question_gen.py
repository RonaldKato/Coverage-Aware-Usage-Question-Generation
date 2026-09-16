"""
question_gen.py
----------------
Turns a selected candidate sentence into a yes/no usage-elicitation
question, following the original paper's best-performing surface
template ("Are you looking for a [category] that is great for
[usage]?", Section 3.1.2), plus two paraphrase templates (mirrors the
paper's Step-3 paraphrase-expansion protocol) so each surviving
candidate yields up to three question variants.

Applicability filtering (`is_generic`) is a transparent, rule-based
stand-in for the paper's RoBERTa N/A-classifier (TQG+CLS / NSQG): it
flags sentences that do not contain any extractable usage span (too
short, generic-praise phrasing, no usage noun phrase captured by the
extractor). This keeps the demo pipeline dependency-free while
preserving the same role in the architecture -- swap in a fine-tuned
classifier or an LLM call here without touching any other module.
"""
import re

GENERIC_MARKERS = [
    "exactly as described", "fast shipping", "five stars", "would buy again",
    "premium for the price", "thank you so much",
]

QUESTION_TEMPLATES = [
    "Are you looking for a {category} that is great for {usage}?",
    "Would you be interested in a {category} that is good for {usage}?",
    "Do you need a {category} for {usage}?",
]


def is_generic(sentence: str, usage_span: str) -> bool:
    s = sentence.lower()
    if not usage_span or len(usage_span.strip()) < 3:
        return True
    if any(m in s for m in GENERIC_MARKERS):
        return True
    return False


def clean_usage_span(usage_span: str) -> str:
    span = usage_span.strip().rstrip(".,;")
    span = re.sub(r"^(a|an|the)\s+", "", span, flags=re.IGNORECASE)
    return span


def generate_questions(category: str, usage_span: str, n_variants: int = 3):
    usage = clean_usage_span(usage_span)
    cat_singular = category.rstrip("s") if category.endswith("s") and not category.endswith("ss") else category
    templates = QUESTION_TEMPLATES[:n_variants]
    return [t.format(category=cat_singular.lower(), usage=usage) for t in templates]
