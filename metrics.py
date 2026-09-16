"""
metrics.py
----------
Three complementary metrics operationalizing "coverage" and "diversity",
the two axes the original paper leaves open.

1. Facet Recall (FR): using the synthetic corpus's known gold usage
   facets (see data_generation.py), the fraction of a category's true
   usage facets that are represented by >=1 selected/generated question.
   This is the most direct, literal measurement of the limitation
   paragraph's recall concern. In a real (non-synthetic) deployment,
   gold facets are unavailable, which is exactly why we also report the
   two label-free metrics below.

2. Tail Coverage (TC): Facet Recall re-weighted to upweight rare facets
   (inverse frequency weighting). A system can have high plain recall by
   only ever covering the popular uses; TC specifically rewards reaching
   the long tail, which is where the paper's limitation bites hardest.

3. Vendi Diversity Score (VDS): a label-free, embedding-only diversity
   metric (Friedman & Dieng, 2023) computed as the effective rank
   (exponential of Shannon entropy of the normalized eigenvalues) of the
   similarity matrix among the *selected* questions' embeddings. Unlike
   FR/TC it needs no gold taxonomy at all, so it is the metric we would
   keep monitoring after deployment on a real, unlabeled corpus.
"""
import numpy as np


def facet_recall(selected_facet_ids, all_facet_ids):
    selected_facet_ids = set(f for f in selected_facet_ids if f != -1)
    all_facet_ids = set(f for f in all_facet_ids if f != -1)
    if not all_facet_ids:
        return 0.0
    return len(selected_facet_ids & all_facet_ids) / len(all_facet_ids)


def tail_coverage(selected_facet_ids, all_facet_ids_with_counts):
    """all_facet_ids_with_counts: dict facet_id -> frequency in the full pool."""
    facets = {f: c for f, c in all_facet_ids_with_counts.items() if f != -1}
    if not facets:
        return 0.0
    inv_weights = {f: 1.0 / c for f, c in facets.items()}
    total_weight = sum(inv_weights.values())
    selected = set(f for f in selected_facet_ids if f != -1)
    covered_weight = sum(inv_weights[f] for f in selected if f in inv_weights)
    return covered_weight / total_weight


def vendi_score(embeddings: np.ndarray) -> float:
    """Effective number of 'distinct directions' spanned by the selected
    embeddings. Higher = more diverse. Computed from the eigenvalues of
    the (row-normalized) similarity matrix K = X X^T / n."""
    n = embeddings.shape[0]
    if n <= 1:
        return float(n)
    K = embeddings @ embeddings.T
    K = K / n
    eigvals = np.linalg.eigvalsh(K)
    eigvals = np.clip(eigvals, 1e-12, None)
    eigvals = eigvals / eigvals.sum()
    entropy = -np.sum(eigvals * np.log(eigvals))
    return float(np.exp(entropy))
