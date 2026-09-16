"""
clustering.py
-------------
Induces a *reference usage taxonomy* per category directly from data, with
no access to the gold facet labels. This operationalizes the paper's
open problem: "determining whether the coverage of usage-related
questions is sufficient for a given category". We approximate "the space
of possible uses for a category" with KMeans clusters over sentence
embeddings (choosing k per category via silhouette score), then treat
each cluster as one distinct "usage bucket".

Coverage of a set of generated/selected questions is then defined
(metrics.py) as the fraction of these induced buckets that are hit by at
least one output, optionally weighted by bucket size (mirrors how
frequently that usage type appears in the underlying review population).

We additionally report agreement between the *induced* clusters and the
*gold* facet_id (Adjusted Rand Index) purely as a sanity check that the
unsupervised taxonomy is meaningful -- this is diagnostic only and is
never used by the selection/generation pipeline itself.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score


def induce_taxonomy(embeddings: np.ndarray, k_min=3, k_max=10, random_state=42):
    """Choose k by best silhouette score in [k_min, k_max] and fit KMeans."""
    n = embeddings.shape[0]
    k_max = min(k_max, n - 1)
    if k_max < k_min:
        k_max = k_min = max(2, min(2, n - 1))
    best_k, best_score, best_model = None, -1, None
    for k in range(k_min, max(k_min, k_max) + 1):
        if k >= n:
            break
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(embeddings)
        if len(set(km.labels_)) < 2:
            continue
        score = silhouette_score(embeddings, km.labels_)
        if score > best_score:
            best_k, best_score, best_model = k, score, km
    if best_model is None:
        best_model = KMeans(n_clusters=min(2, n), n_init=10, random_state=random_state).fit(embeddings)
        best_k = best_model.n_clusters
    return best_model, best_k, best_score


def taxonomy_ari(cluster_labels, gold_facet_ids):
    """Diagnostic-only: agreement between induced clusters and gold facets."""
    return adjusted_rand_score(gold_facet_ids, cluster_labels)
