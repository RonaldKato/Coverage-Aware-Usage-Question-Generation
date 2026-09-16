"""
schema.py
---------
Canonical, typed definitions of every object that flows through the
CA-UQG pipeline. Every other module constructs / consumes these types;
this file is the single source of truth for field names, so the object
diagram (make_figures.py::fig6_object_schema) and the runtime pipeline
(pipeline.py) can never drift apart.

Object lineage (one object produces the next):

    ReviewRecord  --[extraction.py]-->  Candidate
    Candidate     --[embeddings.py]-->  EmbeddingRecord
    EmbeddingRecord --[clustering.py]--> ClusterAssignment
    Candidate + ClusterAssignment --[selection.py]--> SelectedCandidate
    SelectedCandidate --[question_gen.py]--> GeneratedQuestion

Each dataclass below states its stage, its upstream type, and every
field with a one-line description -- this is the docstring content
rendered verbatim into Figure 6 of the paper.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ReviewRecord:
    """Stage 0 -- raw input. One sentence-level unit from the review corpus.

    category      : product category, one of the 12 taxonomy categories
    sentence      : raw sentence text
    pattern       : generator-internal ground-truth template family (P1-P7, or N0 for distractors)
    facet_id      : gold usage-facet id within `category` (-1 if not applicable) -- EVAL ONLY
    usage_gold    : gold usage noun/gerund phrase -- EVAL ONLY
    aspect_gold   : gold product aspect mentioned alongside the usage -- EVAL ONLY
    is_applicable : whether this sentence expresses a genuine, question-worthy usage
    """
    category: str
    sentence: str
    pattern: str
    facet_id: int
    usage_gold: str
    aspect_gold: str
    is_applicable: bool


@dataclass
class Candidate:
    """Stage 1 -- output of extraction.py. A ReviewRecord that matched at
    least one extraction pattern, with the matched usage span isolated.

    source          : the ReviewRecord this candidate was extracted from
    matched_pattern : which of P1-P7 fired (first match wins)
    extracted_span  : the regex-captured usage span (extractor's belief;
                       compare to `source.usage_gold` for extraction accuracy)
    """
    source: ReviewRecord
    matched_pattern: str
    extracted_span: str


@dataclass
class EmbeddingRecord:
    """Stage 2 -- output of embeddings.py. A dense representation of a
    Candidate's sentence in the shared category-level vector space.

    candidate_idx : index of the Candidate this embedding belongs to
                    (within the category's candidate pool)
    vector        : L2-normalized, 24-dim TF-IDF+SVD embedding
    """
    candidate_idx: int
    vector: List[float]


@dataclass
class ClusterAssignment:
    """Stage 3 -- output of clustering.py. Which induced usage-taxonomy
    bucket a candidate falls into, plus the bucket's own descriptive stats.

    candidate_idx  : index of the Candidate this assignment belongs to
    cluster_id     : induced cluster id (0..k-1), unsupervised
    cluster_size   : number of candidates sharing this cluster_id
    is_singleton   : True if cluster_size == 1 (long-tail usage warning)
    """
    candidate_idx: int
    cluster_id: int
    cluster_size: int
    is_singleton: bool


@dataclass
class SelectedCandidate:
    """Stage 4 -- output of selection.py. A Candidate chosen to spend the
    per-category question-generation budget on, plus why it was chosen.

    candidate_idx   : index into the category's candidate pool
    rank            : 1-indexed position in the selection order
    selection_score : the score that justified selection
                       (MMR marginal-relevance score, or raw frequency rank)
    strategy        : "frequency" | "mmr" | "cluster_stratified"
    """
    candidate_idx: int
    rank: int
    selection_score: float
    strategy: str


@dataclass
class GeneratedQuestion:
    """Stage 5 -- output of question_gen.py. The final, user-facing artifact.

    selected_from   : the SelectedCandidate this question was generated from
    template_id     : which surface template produced this variant (0-2)
    question_text   : final natural-language yes/no elicitation question
    is_na           : True if the applicability filter rejected this candidate
                       (no question_text is produced in that case)
    """
    selected_from: SelectedCandidate
    template_id: Optional[int]
    question_text: Optional[str]
    is_na: bool


@dataclass
class PipelineManifest:
    """Execution-provenance record written once per pipeline run. This is
    what turns CA-UQG from a notebook experiment into an auditable
    pipeline: every stage's input/output object counts, the configuration
    used, and per-category timings are captured here and serialized to
    `data/pipeline_manifest.json`.

    run_id            : timestamp-based identifier
    config            : the full CONFIGS entry used (extractor, selector, budget)
    stage_counts      : {stage_name: {category: object_count}}
    stage_seconds     : {stage_name: wall_clock_seconds}
    random_state      : seed used for every stochastic component
    """
    run_id: str
    config: dict
    stage_counts: dict
    stage_seconds: dict
    random_state: int
