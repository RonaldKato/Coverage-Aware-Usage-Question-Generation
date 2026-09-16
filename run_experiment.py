"""
run_experiment.py
------------------
End-to-end driver that reproduces every number and figure reported in
the accompanying paper. Usage:

    python src/run_experiment.py

Outputs:
    tables/table1_dataset_stats.csv
    tables/table2_main_results.csv
    tables/table3_taxonomy_ari.csv
    tables/table4_budget_sweep.csv
    data/selected_questions_sample.csv   (qualitative examples)
    figures/*.png                        (see make_figures.py)
"""
import json
import itertools
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from data_generation import generate_corpus, TAXONOMY
from extraction import extract_baseline, extract_broadened, Candidate
from embeddings import SentenceEmbedder
from clustering import induce_taxonomy, taxonomy_ari
from selection import select_frequency, select_mmr, select_cluster_stratified
from question_gen import generate_questions, is_generic
from metrics import facet_recall, tail_coverage, vendi_score

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
DATA = ROOT / "data"
TABLES.mkdir(exist_ok=True)

RANDOM_STATE = 42
BUDGET_MAIN = 8            # per-category budget used for the headline comparison
BUDGET_SWEEP = [2, 4, 6, 8, 10, 15, 20]

CONFIGS = {
    # name                         : (extractor, selector)
    "Baseline (TQG-style)":          ("baseline", "frequency"),
    "Extraction-only (P1-P7 + freq)": ("broadened", "frequency"),
    "Selection-only (P1 + MMR)":     ("baseline", "mmr"),
    "Cluster-stratified":            ("broadened", "cluster"),
    "CA-UQG (proposed)":             ("broadened", "mmr"),
}


def build_records():
    corpus = generate_corpus()
    records = [r.__dict__ for r in corpus]
    return records


def run_category_pipeline(category, cat_records, embedder_cache):
    base_cands = [c for c in extract_baseline(cat_records)]
    broad_cands = [c for c in extract_broadened(cat_records)]

    pools = {"baseline": base_cands, "broadened": broad_cands}

    # embeddings + induced taxonomy are computed once on the larger
    # (broadened) pool so every config is scored against the *same*
    # reference space -- this is important for a fair comparison.
    sentences = [c.sentence for c in broad_cands]
    embedder = SentenceEmbedder(n_components=24, random_state=RANDOM_STATE).fit(sentences)
    embedder_cache[category] = embedder
    broad_emb = embedder.transform(sentences)

    km_model, best_k, sil = induce_taxonomy(broad_emb, k_min=3, k_max=min(10, max(3, len(broad_cands) // 4)))
    gold_facets = [c.facet_id for c in broad_cands]
    ari = taxonomy_ari(km_model.labels_, gold_facets)

    # counts of gold facets in the *full applicable pool* define the
    # denominator for recall/tail-coverage (population of true uses).
    all_facets_applicable = [c.facet_id for c in broad_cands if c.is_applicable]
    facet_counts = Counter(all_facets_applicable)

    return {
        "base_cands": base_cands,
        "broad_cands": broad_cands,
        "broad_emb": broad_emb,
        "cluster_labels": km_model.labels_,
        "best_k": best_k,
        "silhouette": sil,
        "ari": ari,
        "facet_counts": dict(facet_counts),
    }


def score_config(cat_state, extractor_key, selector_key, budget, embedder):
    if extractor_key == "baseline":
        cands = cat_state["base_cands"]
        sentences = [c.sentence for c in cands]
        emb = embedder.transform(sentences) if sentences else np.zeros((0, 1))
    else:
        cands = cat_state["broad_cands"]
        emb = cat_state["broad_emb"]

    if len(cands) == 0:
        return dict(facet_recall=0.0, tail_coverage=0.0, vendi=0.0, n_selected=0,
                     n_generated_questions=0)

    if selector_key == "frequency":
        idx = select_frequency(cands, budget)
    elif selector_key == "mmr":
        idx = select_mmr(emb, budget, lambda_=0.5)
    elif selector_key == "cluster":
        # stratify by cluster labels computed on the broadened pool; if the
        # config uses the baseline extractor we fall back to frequency,
        # since cluster labels are only defined over the broadened pool.
        if extractor_key == "broadened":
            idx = select_cluster_stratified(cat_state["cluster_labels"], budget, random_state=RANDOM_STATE)
        else:
            idx = select_frequency(cands, budget)
    else:
        raise ValueError(selector_key)

    selected = [cands[i] for i in idx]
    selected_emb = emb[idx]

    # applicability filter + question generation, exactly as it would run
    # in production (rule-based N/A filter from question_gen.py)
    n_questions = 0
    generated_examples = []
    for c in selected:
        if is_generic(c.sentence, c.usage_gold):
            continue
        qs = generate_questions(c.category, c.usage_gold, n_variants=3)
        n_questions += len(qs)
        generated_examples.append((c.sentence, c.usage_gold, qs[0]))

    selected_facet_ids = [c.facet_id for c in selected if c.is_applicable]
    fr = facet_recall(selected_facet_ids, list(cat_state["facet_counts"].keys()))
    tc = tail_coverage(selected_facet_ids, cat_state["facet_counts"])
    vds = vendi_score(selected_emb) if len(selected_emb) > 0 else 0.0

    return dict(facet_recall=fr, tail_coverage=tc, vendi=vds,
                n_selected=len(selected), n_generated_questions=n_questions,
                examples=generated_examples)


def main():
    records = build_records()
    df_records = pd.DataFrame(records)

    # ---------------------------------------------------------------
    # Table 1: dataset statistics
    # ---------------------------------------------------------------
    stats_rows = []
    cat_states = {}
    embedder_cache = {}
    for cat in TAXONOMY:
        cat_records = [r for r in records if r["category"] == cat]
        state = run_category_pipeline(cat, cat_records, embedder_cache)
        cat_states[cat] = state
        n_applicable = sum(1 for r in cat_records if r["is_applicable"])
        stats_rows.append({
            "category": cat,
            "n_sentences": len(cat_records),
            "n_applicable_gold": n_applicable,
            "n_gold_facets": len(TAXONOMY[cat]),
            "n_baseline_candidates": len(state["base_cands"]),
            "n_broadened_candidates": len(state["broad_cands"]),
            "induced_k": state["best_k"],
            "silhouette": round(state["silhouette"], 3) if state["silhouette"] else None,
            "taxonomy_ari": round(state["ari"], 3),
        })
    table1 = pd.DataFrame(stats_rows)
    table1.to_csv(TABLES / "table1_dataset_stats.csv", index=False)

    # ---------------------------------------------------------------
    # Table 2: main results @ BUDGET_MAIN, averaged over categories
    # ---------------------------------------------------------------
    main_rows = []
    examples_all = []
    for cfg_name, (extractor_key, selector_key) in CONFIGS.items():
        per_cat = []
        for cat in TAXONOMY:
            state = cat_states[cat]
            res = score_config(state, extractor_key, selector_key, BUDGET_MAIN, embedder_cache[cat])
            per_cat.append(res)
            for ex in res.get("examples", [])[:1]:
                examples_all.append({"config": cfg_name, "category": cat,
                                      "review_sentence": ex[0], "usage_span": ex[1],
                                      "generated_question": ex[2]})
        main_rows.append({
            "config": cfg_name,
            "facet_recall_mean": np.mean([r["facet_recall"] for r in per_cat]),
            "tail_coverage_mean": np.mean([r["tail_coverage"] for r in per_cat]),
            "vendi_score_mean": np.mean([r["vendi"] for r in per_cat]),
            "avg_questions_generated": np.mean([r["n_generated_questions"] for r in per_cat]),
        })
    table2 = pd.DataFrame(main_rows).round(4)
    table2.to_csv(TABLES / "table2_main_results.csv", index=False)
    pd.DataFrame(examples_all).to_csv(DATA / "selected_questions_sample.csv", index=False)

    # per-category breakdown @ BUDGET_MAIN (needed for Figure 3)
    per_cat_rows = []
    for cfg_name, (extractor_key, selector_key) in CONFIGS.items():
        for cat in TAXONOMY:
            state = cat_states[cat]
            res = score_config(state, extractor_key, selector_key, BUDGET_MAIN, embedder_cache[cat])
            per_cat_rows.append({"config": cfg_name, "category": cat,
                                  "facet_recall": res["facet_recall"],
                                  "tail_coverage": res["tail_coverage"],
                                  "vendi": res["vendi"]})
    pd.DataFrame(per_cat_rows).to_csv(DATA / "per_category_budget8.csv", index=False)

    # ---------------------------------------------------------------
    # Table 3: taxonomy induction diagnostic (already partly in table1)
    # ---------------------------------------------------------------
    table3 = table1[["category", "induced_k", "n_gold_facets", "silhouette", "taxonomy_ari"]]
    table3.to_csv(TABLES / "table3_taxonomy_ari.csv", index=False)

    # ---------------------------------------------------------------
    # Table 4: budget sweep for baseline vs proposed (for the coverage curve)
    # ---------------------------------------------------------------
    sweep_rows = []
    for cfg_name, (extractor_key, selector_key) in CONFIGS.items():
        for b in BUDGET_SWEEP:
            fr_list, tc_list, vds_list = [], [], []
            for cat in TAXONOMY:
                state = cat_states[cat]
                res = score_config(state, extractor_key, selector_key, b, embedder_cache[cat])
                fr_list.append(res["facet_recall"])
                tc_list.append(res["tail_coverage"])
                vds_list.append(res["vendi"])
            sweep_rows.append({
                "config": cfg_name, "budget": b,
                "facet_recall_mean": np.mean(fr_list),
                "tail_coverage_mean": np.mean(tc_list),
                "vendi_score_mean": np.mean(vds_list),
            })
    table4 = pd.DataFrame(sweep_rows).round(4)
    table4.to_csv(TABLES / "table4_budget_sweep.csv", index=False)

    print("Table 1 (dataset stats):\n", table1.to_string(index=False))
    print("\nTable 2 (main results @ budget=%d):\n" % BUDGET_MAIN, table2.to_string(index=False))
    print("\nSaved tables to", TABLES)

    # persist category states needed for the qualitative 2D figure
    np.savez(DATA / "bikes_embedding_cache.npz",
             emb=cat_states["Bikes"]["broad_emb"],
             facet_ids=np.array([c.facet_id for c in cat_states["Bikes"]["broad_cands"]]),
             cluster_labels=cat_states["Bikes"]["cluster_labels"])


if __name__ == "__main__":
    main()
