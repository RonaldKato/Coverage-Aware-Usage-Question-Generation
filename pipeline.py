"""
pipeline.py
-----------
A technical, auditable pipeline runner built on top of the research
modules (extraction.py, embeddings.py, clustering.py, selection.py,
question_gen.py). Where run_experiment.py runs the *experiment grid*
(many configs x many budgets, for the results tables), this module runs
the pipeline *once, end-to-end, as a production system would*: each
stage is an explicit `Stage` object that consumes and emits the typed
objects defined in schema.py, and every run is captured in a
`PipelineManifest` -- the object-count / timing provenance record
written to `data/pipeline_manifest.json`.

Run standalone:
    python pipeline.py
"""
import json
import time
from pathlib import Path
from dataclasses import asdict

import numpy as np

from data_generation import TAXONOMY
from extraction import extract_baseline, extract_broadened
from embeddings import SentenceEmbedder
from clustering import induce_taxonomy
from selection import select_frequency, select_mmr
from question_gen import generate_questions, is_generic
from schema import ReviewRecord, Candidate, EmbeddingRecord, ClusterAssignment, \
    SelectedCandidate, GeneratedQuestion, PipelineManifest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TABLES = ROOT / "tables"


class Stage:
    """Base class for one pipeline stage. Subclasses implement `run`,
    which must return (output_objects, count). Timing and counting are
    handled uniformly here so every stage is comparably instrumented."""
    name = "stage"

    def run(self, context):
        raise NotImplementedError

    def __call__(self, context, manifest):
        t0 = time.perf_counter()
        context = self.run(context)
        dt = time.perf_counter() - t0
        manifest["stage_seconds"].setdefault(self.name, {})[context["category"]] = round(dt, 5)
        manifest["stage_counts"].setdefault(self.name, {})[context["category"]] = context["count"]
        return context


class ExtractionStage(Stage):
    name = "1_extraction"

    def __init__(self, extractor_key: str):
        self.extractor_key = extractor_key  # "baseline" | "broadened"

    def run(self, context):
        fn = extract_baseline if self.extractor_key == "baseline" else extract_broadened
        raw_cands = fn(context["records"])  # list[extraction.Candidate]
        candidates = [
            Candidate(
                source=ReviewRecord(
                    category=c.category, sentence=c.sentence, pattern=c.matched_pattern,
                    facet_id=c.facet_id, usage_gold=c.usage_gold, aspect_gold=c.aspect_gold,
                    is_applicable=c.is_applicable,
                ),
                matched_pattern=c.matched_pattern,
                extracted_span=c.usage_gold,  # regex-recovered span == gold span in this simulator
            )
            for c in raw_cands
        ]
        context["candidates"] = candidates
        context["count"] = len(candidates)
        return context


class EmbeddingStage(Stage):
    name = "2_embedding"

    def __init__(self, random_state=42):
        self.random_state = random_state

    def run(self, context):
        sentences = [c.source.sentence for c in context["candidates"]]
        embedder = SentenceEmbedder(n_components=24, random_state=self.random_state)
        if sentences:
            vecs = embedder.fit_transform(sentences)
        else:
            vecs = np.zeros((0, 24))
        context["embedder"] = embedder
        context["embeddings"] = vecs
        context["embedding_records"] = [
            EmbeddingRecord(candidate_idx=i, vector=vecs[i].tolist()) for i in range(len(vecs))
        ]
        context["count"] = len(context["embedding_records"])
        return context


class ClusteringStage(Stage):
    name = "3_clustering"

    def __init__(self, k_min=3, k_max=10, random_state=42):
        self.k_min, self.k_max, self.random_state = k_min, k_max, random_state

    def run(self, context):
        emb = context["embeddings"]
        if len(emb) < self.k_min + 1:
            context["cluster_assignments"] = []
            context["count"] = 0
            return context
        km, k, sil = induce_taxonomy(emb, k_min=self.k_min,
                                      k_max=min(self.k_max, max(self.k_min, len(emb) // 4)),
                                      random_state=self.random_state)
        labels = km.labels_
        sizes = {c: int((labels == c).sum()) for c in set(labels)}
        context["cluster_assignments"] = [
            ClusterAssignment(candidate_idx=i, cluster_id=int(labels[i]),
                               cluster_size=sizes[labels[i]], is_singleton=sizes[labels[i]] == 1)
            for i in range(len(labels))
        ]
        context["induced_k"] = k
        context["silhouette"] = sil
        context["count"] = len(context["cluster_assignments"])
        return context


class SelectionStage(Stage):
    name = "4_selection"

    def __init__(self, selector_key: str, budget: int, lambda_: float = 0.5):
        self.selector_key, self.budget, self.lambda_ = selector_key, budget, lambda_

    def run(self, context):
        cands = context["candidates"]
        emb = context["embeddings"]
        if len(cands) == 0:
            context["selected"] = []
            context["count"] = 0
            return context
        if self.selector_key == "frequency":
            idx = select_frequency(cands, self.budget)
            scores = list(range(len(idx), 0, -1))
        else:
            idx = select_mmr(emb, self.budget, lambda_=self.lambda_)
            centroid = emb.mean(axis=0, keepdims=True)
            scores = (emb[idx] @ centroid.T).flatten().tolist()
        context["selected"] = [
            SelectedCandidate(candidate_idx=i, rank=r + 1, selection_score=float(s),
                               strategy=self.selector_key)
            for r, (i, s) in enumerate(zip(idx, scores))
        ]
        context["count"] = len(context["selected"])
        return context


class GenerationStage(Stage):
    name = "5_generation"

    def run(self, context):
        cands = context["candidates"]
        questions = []
        for sel in context["selected"]:
            c = cands[sel.candidate_idx]
            if is_generic(c.source.sentence, c.source.usage_gold):
                questions.append(GeneratedQuestion(selected_from=sel, template_id=None,
                                                     question_text=None, is_na=True))
                continue
            qs = generate_questions(c.source.category, c.source.usage_gold, n_variants=3)
            for t_id, q in enumerate(qs):
                questions.append(GeneratedQuestion(selected_from=sel, template_id=t_id,
                                                     question_text=q, is_na=False))
        context["questions"] = questions
        context["count"] = len(questions)
        return context


class Pipeline:
    """Chains stages in order and produces a PipelineManifest across all
    categories for one (extractor, selector, budget) configuration."""

    def __init__(self, extractor_key, selector_key, budget, random_state=42):
        self.stages = [
            ExtractionStage(extractor_key),
            EmbeddingStage(random_state),
            ClusteringStage(random_state=random_state),
            SelectionStage(selector_key, budget),
            GenerationStage(),
        ]
        self.config = {"extractor": extractor_key, "selector": selector_key, "budget": budget}
        self.random_state = random_state

    def run_all_categories(self, records_by_category):
        manifest = {"stage_counts": {}, "stage_seconds": {}}
        outputs = {}
        for cat, records in records_by_category.items():
            context = {"category": cat, "records": records}
            for stage in self.stages:
                context = stage(context, manifest)
            outputs[cat] = context
        pm = PipelineManifest(
            run_id=time.strftime("%Y%m%dT%H%M%S"),
            config=self.config,
            stage_counts=manifest["stage_counts"],
            stage_seconds=manifest["stage_seconds"],
            random_state=self.random_state,
        )
        return outputs, pm


def _records_by_category():
    import json as _json
    records = [_json.loads(l) for l in open(DATA / "synthetic_reviews.jsonl")]
    by_cat = {cat: [] for cat in TAXONOMY}
    for r in records:
        by_cat[r["category"]].append(r)
    return by_cat


def main():
    records_by_category = _records_by_category()

    runs = {
        "Baseline (TQG-style)": Pipeline("baseline", "frequency", budget=8),
        "CA-UQG (proposed)": Pipeline("broadened", "mmr", budget=8),
    }

    all_manifests = {}
    flat_rows = []
    for name, pipe in runs.items():
        outputs, pm = pipe.run_all_categories(records_by_category)
        all_manifests[name] = asdict(pm)
        for stage_name, per_cat in pm.stage_counts.items():
            for cat, count in per_cat.items():
                flat_rows.append({
                    "config": name, "stage": stage_name, "category": cat,
                    "object_count": count,
                    "seconds": pm.stage_seconds[stage_name][cat],
                })
        # qualitative dump of generated questions for one config
        if name == "CA-UQG (proposed)":
            sample = []
            for cat, ctx in outputs.items():
                for q in ctx["questions"][:2]:
                    if not q.is_na:
                        sample.append({"category": cat, "question": q.question_text,
                                        "template_id": q.template_id,
                                        "selection_rank": q.selected_from.rank})
            (DATA / "pipeline_generated_sample.json").write_text(json.dumps(sample, indent=2))

    (DATA / "pipeline_manifest.json").write_text(json.dumps(all_manifests, indent=2))

    import pandas as pd
    df = pd.DataFrame(flat_rows)
    df.to_csv(TABLES / "table5_pipeline_manifest.csv", index=False)

    funnel = (df.groupby(["config", "stage"])["object_count"].sum()
                .reset_index().pivot(index="stage", columns="config", values="object_count"))
    funnel = funnel.reindex(["1_extraction", "2_embedding", "3_clustering", "4_selection", "5_generation"])
    funnel.to_csv(TABLES / "table6_funnel.csv")

    print("Pipeline manifest written to", DATA / "pipeline_manifest.json")
    print("\nObject-count funnel (summed across categories):\n", funnel.to_string())


if __name__ == "__main__":
    main()
