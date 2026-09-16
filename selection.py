"""
selection.py
------------
Given a pool of candidate sentences (already extracted) and a fixed
"budget" B of sentences that will be turned into elicitation questions
per category (question generation, downstream crowdsourcing, or serving
cost all scale with B), which B sentences should we pick?

  * `select_frequency`  -- the implicit strategy of the original pipeline:
    keep whatever the heuristic returns, in encounter order / by raw
    frequency. Popular uses (mentioned in many reviews) dominate the
    budget; rare-but-valid uses are crowded out. This is the mechanism
    behind the recall/coverage limitation.

  * `select_mmr` -- Maximal Marginal Relevance (Carbonell & Goldstein,
    1998) over the sentence-embedding space: greedily pick the sentence
    that is representative of *still-uncovered* regions of the usage
    space, penalizing similarity to sentences already chosen. This
    directly optimizes for spread (diversity) rather than raw frequency,
    at a fixed budget.

  * `select_cluster_stratified` -- a simple, strong diversity baseline:
    round-robin one item per induced cluster (clustering.py) until the
    budget is exhausted. Included as an ablation between "no diversity
    control" (frequency) and "similarity-based diversity control" (MMR).
"""
import numpy as np


def select_frequency(candidates, budget):
    """No diversity control -- first-come selection (mirrors default
    pipeline behaviour when no re-ranking / dedup step is applied)."""
    return list(range(min(budget, len(candidates))))


def select_mmr(embeddings: np.ndarray, budget: int, lambda_: float = 0.5):
    """Standard MMR. lambda_=1 -> pure relevance (here: pure centrality),
    lambda_=0 -> pure diversity (max spread)."""
    n = embeddings.shape[0]
    budget = min(budget, n)
    centroid = embeddings.mean(axis=0, keepdims=True)
    relevance = (embeddings @ centroid.T).flatten()

    selected, remaining = [], list(range(n))
    # seed with the most central (most "typical") sentence
    first = int(np.argmax(relevance))
    selected.append(first)
    remaining.remove(first)

    while len(selected) < budget and remaining:
        sel_emb = embeddings[selected]
        sims_to_selected = embeddings[remaining] @ sel_emb.T
        max_sim = sims_to_selected.max(axis=1)
        rel = relevance[remaining]
        mmr_score = lambda_ * rel - (1 - lambda_) * max_sim
        best_local = int(np.argmax(mmr_score))
        selected.append(remaining[best_local])
        remaining.pop(best_local)
    return selected


def select_cluster_stratified(cluster_labels, budget, random_state=42):
    rng = np.random.RandomState(random_state)
    clusters = {}
    for idx, c in enumerate(cluster_labels):
        clusters.setdefault(c, []).append(idx)
    for c in clusters:
        rng.shuffle(clusters[c])

    order = list(clusters.keys())
    rng.shuffle(order)
    selected = []
    pointer = {c: 0 for c in order}
    while len(selected) < budget and any(pointer[c] < len(clusters[c]) for c in order):
        for c in order:
            if len(selected) >= budget:
                break
            if pointer[c] < len(clusters[c]):
                selected.append(clusters[c][pointer[c]])
                pointer[c] += 1
    return selected
