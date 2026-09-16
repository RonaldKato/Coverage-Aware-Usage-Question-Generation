"""
make_figures.py
----------------
Renders every figure used in the paper from the CSV tables / npz cache
produced by run_experiment.py. Run after run_experiment.py.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
DATA = ROOT / "data"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Sans",
})

PALETTE = {
    "Baseline (TQG-style)": "#9aa0a6",
    "Extraction-only (P1-P7 + freq)": "#4e79a7",
    "Selection-only (P1 + MMR)": "#f28e2b",
    "Cluster-stratified": "#59a14f",
    "CA-UQG (proposed)": "#e15759",
}


# ---------------------------------------------------------------------------
# Figure 1: pipeline architecture diagram
# ---------------------------------------------------------------------------
def fig1_pipeline():
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.set_xlim(0, 11); ax.set_ylim(0, 3.4); ax.axis("off")

    boxes = [
        ("Review\ncorpus", 0.3, "#e8eaf0"),
        ("Multi-pattern\nextraction\n(P1\u2013P7)", 2.05, "#cfe2f3"),
        ("Sentence\nembedding", 3.8, "#cfe2f3"),
        ("Taxonomy\ninduction\n(clustering)", 5.55, "#d9ead3"),
        ("Coverage-aware\nselection\n(MMR)", 7.3, "#fce5cd"),
        ("Question\ngeneration\n+ N/A filter", 9.05, "#f4cccc"),
    ]
    w, h, y = 1.55, 1.5, 0.95
    centers = []
    for label, x, color in boxes:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                              linewidth=1.2, edgecolor="#333333", facecolor=color)
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9.5, weight="bold")
        centers.append((x + w, y + h / 2))

    for i in range(len(centers) - 1):
        x0, yc = centers[i]
        x1 = boxes[i + 1][1]
        ax.add_patch(FancyArrowPatch((x0, yc), (x1, yc), arrowstyle="-|>", mutation_scale=14,
                                      linewidth=1.3, color="#333333"))

    # feedback loop: taxonomy informs selection (coverage signal)
    ax.annotate("", xy=(7.3 + 0.0, y + h + 0.55), xytext=(5.55 + w / 2, y + h + 0.05),
                arrowprops=dict(arrowstyle="-|>", color="#888888", lw=1.1,
                                 connectionstyle="arc3,rad=-0.25", linestyle="dashed"))
    ax.text(6.6, y + h + 0.62, "coverage signal", fontsize=8.3, color="#666666", style="italic")

    ax.text(0.3 + w / 2, y - 0.35, "Sec. 4.1", ha="center", fontsize=8, color="#777777")
    ax.text(5.5, 3.15, "Coverage-Aware Usage Question Generation (CA-UQG) Pipeline",
            ha="center", fontsize=12.5, weight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "fig1_pipeline_architecture.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: coverage vs. budget curves (Facet Recall & Tail Coverage)
# ---------------------------------------------------------------------------
def fig2_budget_sweep():
    df = pd.read_csv(TABLES / "table4_budget_sweep.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for metric, ax, title in [
        ("facet_recall_mean", axes[0], "Facet Recall vs. selection budget"),
        ("tail_coverage_mean", axes[1], "Tail Coverage vs. selection budget"),
    ]:
        for cfg, color in PALETTE.items():
            sub = df[df.config == cfg].sort_values("budget")
            lw, ls, marker = (2.6, "-", "o") if cfg == "CA-UQG (proposed)" else (1.6, "--" if cfg == "Baseline (TQG-style)" else "-", "s")
            ax.plot(sub.budget, sub[metric], label=cfg, color=color, linewidth=lw, linestyle=ls, marker=marker, markersize=4)
        ax.set_xlabel("Sentences selected per category (budget)")
        ax.set_ylabel(title.split(" vs.")[0])
        ax.set_title(title, fontsize=10.5)
        ax.set_ylim(0, 1.0)
        ax.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.08), fontsize=8.7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_coverage_vs_budget.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: per-category tail coverage, baseline vs proposed, at fixed budget
# ---------------------------------------------------------------------------
def fig3_per_category_bars():
    df = pd.read_csv(TABLES / "table4_budget_sweep.csv")
    df8 = df[df.budget == 8]
    # need per-category; recompute quickly by re-reading table1 categories via table2 is avg only.
    # We instead reuse run_experiment's per-category detail saved separately.
    per_cat = pd.read_csv(DATA / "per_category_budget8.csv")
    cats = per_cat.category.unique()
    x = np.arange(len(cats))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11, 4.6))
    base = per_cat[per_cat.config == "Baseline (TQG-style)"].set_index("category").loc[cats, "tail_coverage"]
    prop = per_cat[per_cat.config == "CA-UQG (proposed)"].set_index("category").loc[cats, "tail_coverage"]
    ax.bar(x - width / 2, base, width, label="Baseline (TQG-style)", color=PALETTE["Baseline (TQG-style)"])
    ax.bar(x + width / 2, prop, width, label="CA-UQG (proposed)", color=PALETTE["CA-UQG (proposed)"])
    ax.set_xticks(x); ax.set_xticklabels(cats, rotation=35, ha="right")
    ax.set_ylabel("Tail Coverage @ budget = 8")
    ax.set_title("Per-category tail-use coverage: baseline vs. CA-UQG")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_per_category_tail_coverage.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: diversity (Vendi score) comparison across configs
# ---------------------------------------------------------------------------
def fig4_vendi_bars():
    df = pd.read_csv(TABLES / "table2_main_results.csv")
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    colors = [PALETTE[c] for c in df.config]
    bars = ax.bar(df.config, df.vendi_score_mean, color=colors)
    for b, v in zip(bars, df.vendi_score_mean):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_ylabel("Vendi Diversity Score (mean across categories)")
    ax.set_title("Label-free diversity of selected questions @ budget = 8")
    ax.set_xticklabels(df.config, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_vendi_diversity.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5: qualitative 2D usage-space map for one category (Bikes)
# ---------------------------------------------------------------------------
def fig5_embedding_map():
    cache = np.load(DATA / "bikes_embedding_cache.npz")
    emb, facet_ids, cluster_labels = cache["emb"], cache["facet_ids"], cache["cluster_labels"]
    pca = PCA(n_components=2, random_state=42)
    xy = pca.fit_transform(emb)

    from extraction import extract_baseline, extract_broadened
    import json
    records = [json.loads(l) for l in open(DATA / "synthetic_reviews.jsonl")]
    bike_records = [r for r in records if r["category"] == "Bikes"]
    broad_cands = extract_broadened(bike_records)
    base_sentence_set = set(c.sentence for c in extract_baseline(bike_records))

    from embeddings import SentenceEmbedder
    from selection import select_mmr, select_frequency
    sentences = [c.sentence for c in broad_cands]
    is_baseline_reachable = np.array([s in base_sentence_set for s in sentences])

    base_idx_within_broad = [i for i, r in enumerate(is_baseline_reachable) if r]
    freq_selected = set(select_frequency([c for i, c in enumerate(broad_cands) if is_baseline_reachable[i]], 8))
    freq_selected_global = set(base_idx_within_broad[i] for i in freq_selected)
    mmr_selected = set(select_mmr(emb, 8, lambda_=0.5))

    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=facet_ids, cmap="tab10", s=42, alpha=0.55,
                     edgecolor="white", linewidth=0.3, label="_nolegend_")
    ax.scatter(xy[list(freq_selected_global), 0], xy[list(freq_selected_global), 1],
               facecolor="none", edgecolor="black", s=190, linewidth=1.8,
               marker="o", label="Selected by Baseline (freq., budget=8)")
    ax.scatter(xy[list(mmr_selected), 0], xy[list(mmr_selected), 1],
               facecolor="none", edgecolor="#e15759", s=320, linewidth=2.3,
               marker="D", label="Selected by CA-UQG (MMR, budget=8)")
    ax.set_title("Usage-space map for \"Bikes\" (PCA of sentence embeddings)\ncolor = gold usage facet, budget = 8")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_bikes_usage_space_map.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6: object / data-schema diagram (technical pipeline objects)
# ---------------------------------------------------------------------------
def fig6_object_schema():
    fig, ax = plt.subplots(figsize=(13, 8.6))
    ax.set_xlim(0, 13); ax.set_ylim(0, 8.6); ax.axis("off")

    def draw_box(title, fields, x, top_y, w, color):
        rh = 0.34
        h = 0.62 + rh * len(fields)
        box = FancyBboxPatch((x, top_y - h), w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
                              linewidth=1.3, edgecolor="#333333", facecolor=color)
        ax.add_patch(box)
        ax.text(x + w / 2, top_y - 0.28, title, ha="center", va="top", fontsize=10.8, weight="bold")
        ax.plot([x + 0.12, x + w - 0.12], [top_y - 0.5, top_y - 0.5], color="#333333", linewidth=0.8)
        for i, f in enumerate(fields):
            ax.text(x + 0.16, top_y - 0.74 - rh * i, f, ha="left", va="top", fontsize=8.4, family="monospace")
        return (x, top_y - h, w, h)  # bbox

    row1_top = 7.55
    b_review = draw_box("ReviewRecord",
                         ["category: str", "sentence: str", "pattern: str", "facet_id: int  (eval-only)",
                          "usage_gold: str  (eval-only)", "aspect_gold: str  (eval-only)", "is_applicable: bool"],
                         0.3, row1_top, 3.0, "#e8eaf0")
    b_cand = draw_box("Candidate",
                       ["source: ReviewRecord", "matched_pattern: str", "extracted_span: str"],
                       3.7, row1_top, 3.0, "#cfe2f3")
    b_emb = draw_box("EmbeddingRecord",
                      ["candidate_idx: int", "vector: float[24]"],
                      7.1, row1_top, 2.7, "#d9ead3")
    b_clus = draw_box("ClusterAssignment",
                       ["candidate_idx: int", "cluster_id: int", "cluster_size: int", "is_singleton: bool"],
                       10.2, row1_top, 2.5, "#fce5cd")

    row2_top = 2.55
    b_sel = draw_box("SelectedCandidate",
                      ["candidate_idx: int", "rank: int", "selection_score: float", "strategy: str"],
                      0.3, row2_top, 3.1, "#f4cccc")
    b_gen = draw_box("GeneratedQuestion",
                      ["selected_from: SelectedCandidate", "template_id: int | None",
                       "question_text: str | None", "is_na: bool"],
                      3.8, row2_top, 3.6, "#f9e79f")
    b_man = draw_box("PipelineManifest",
                      ["run_id: str", "config: dict", "stage_counts: dict", "stage_seconds: dict",
                       "random_state: int"],
                      7.8, row2_top, 3.4, "#d5c6e0")

    arrow_kw = dict(arrowstyle="-|>", mutation_scale=15, linewidth=1.4, color="#333333")

    def right_mid(b):
        x, y, w, h = b
        return (x + w, y + h / 2)

    def left_mid(b):
        x, y, w, h = b
        return (x, y + h / 2)

    def bottom_mid(b):
        x, y, w, h = b
        return (x + w / 2, y)

    def top_mid(b):
        x, y, w, h = b
        return (x + w / 2, y + h)

    ax.add_patch(FancyArrowPatch(right_mid(b_review), left_mid(b_cand), **arrow_kw))
    ax.add_patch(FancyArrowPatch(right_mid(b_cand), left_mid(b_emb), **arrow_kw))
    ax.add_patch(FancyArrowPatch(right_mid(b_emb), left_mid(b_clus), **arrow_kw))

    # row1 -> row2 (Candidate + ClusterAssignment -> SelectedCandidate)
    ax.add_patch(FancyArrowPatch(bottom_mid(b_cand), top_mid(b_sel), connectionstyle="arc3,rad=0.15", **arrow_kw))
    ax.add_patch(FancyArrowPatch(bottom_mid(b_clus), top_mid(b_sel), connectionstyle="arc3,rad=-0.35", **arrow_kw))
    # SelectedCandidate -> GeneratedQuestion
    ax.add_patch(FancyArrowPatch(right_mid(b_sel), left_mid(b_gen), **arrow_kw))
    # every stage -> PipelineManifest (provenance), drawn as one dashed arrow from GeneratedQuestion
    ax.add_patch(FancyArrowPatch(right_mid(b_gen), left_mid(b_man),
                                  arrowstyle="-|>", mutation_scale=15, linewidth=1.2,
                                  color="#888888", linestyle="dashed"))

    ax.text(6.5, 8.4, "CA-UQG Object / Data Schema", ha="center", fontsize=14.5, weight="bold")
    ax.text(6.5, 8.08, "Boxes = typed objects (schema.py)  \u2022  solid arrows = produced-by lineage  \u2022  dashed = logged into",
            ha="center", fontsize=9.3, style="italic", color="#555555")
    fig.tight_layout()
    fig.savefig(FIG / "fig6_object_schema.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 7: annotated extraction examples (labeled spans, P1-P7)
# ---------------------------------------------------------------------------
def fig7_annotated_extraction():
    import re
    from extraction import BROADENED_PATTERNS

    examples = [
        ("Bikes", "The low gear ratio is great for climbing hills."),
        ("Tents", "When I go backpacking trips, the packed weight really shows its value."),
        ("Grills", "Ideal for anyone who does a lot of weekend barbecues."),
        ("Jackets", "During ski trips, the waterproofing held up really well."),
        ("Blenders", "Compared to my old one, this is much better for crushing ice for cocktails because of the motor power."),
        ("Vacuums", "Bought it for cleaning pet hair; the suction power was the deciding factor."),
        ("Feeders", "Every time we head out for attracting cardinals, I'm glad we have the perch design."),
        ("Snow Shovels", "Great product, exactly as described, fast shipping."),
    ]

    fig, ax = plt.subplots(figsize=(12, 6.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, len(examples) + 0.5); ax.axis("off")
    colors = {"P1_for_gerund": "#a8d5ff", "P2_when_clause": "#c5e8b7", "P3_ideal_adj": "#ffe0a3",
              "P4_during_prep": "#f7b7c2", "P5_comparative": "#d9c2f0", "P6_bare_noun": "#b7e8e0",
              "P7_first_person_narrative": "#f0d9b5", None: "#e0e0e0"}

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()

    def text_width_data(s, **kw):
        t = ax.text(0, -100, s, **kw)  # off-screen probe
        bbox = t.get_window_extent(renderer=renderer)
        (x0, _), (x1, _) = inv.transform((bbox.x0, bbox.y0)), inv.transform((bbox.x1, bbox.y1))
        t.remove()
        return x1 - x0

    y = len(examples)
    for cat, sent in examples:
        matched, span = None, None
        for name, pat in BROADENED_PATTERNS.items():
            m = pat.search(sent)
            if m:
                matched, span = name, m.group(0)
                break
        ax.text(0.05, y, f"[{cat}]", fontsize=8.7, family="monospace", color="#555555", va="center")
        x = 1.7
        if matched and span:
            start = sent.find(span)
            pre, hit, post = sent[:start], span, sent[start + len(span):]
            fs = 9.3
            if pre:
                ax.text(x, y, pre, fontsize=fs, va="center", ha="left")
                x += text_width_data(pre, fontsize=fs)
            hit_w = text_width_data(hit, fontsize=fs, weight="bold")
            ax.text(x + hit_w / 2, y, hit, fontsize=fs, va="center", ha="center", weight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[matched], edgecolor="#666666", linewidth=0.8))
            x += hit_w + 0.15
            if post:
                ax.text(x, y, post, fontsize=fs, va="center", ha="left")
            label = matched
        else:
            ax.text(x, y, sent, fontsize=9.3, va="center", ha="left", color="#777777", style="italic")
            label = "N/A (no pattern match \u2192 rejected)"
        ax.text(11.95, y, label, fontsize=7.8, va="center", ha="right", color="#444444", family="monospace")
        y -= 1

    ax.text(6, len(examples) + 0.35, "Annotated extraction examples: matched usage span \u2192 pattern label",
            ha="center", fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "fig7_annotated_extraction.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 8: induced-cluster vs. gold-facet confusion matrix (Bikes)
# ---------------------------------------------------------------------------
def fig8_cluster_confusion():
    import json
    from data_generation import TAXONOMY
    from extraction import extract_broadened
    from embeddings import SentenceEmbedder
    from clustering import induce_taxonomy

    records = [json.loads(l) for l in open(DATA / "synthetic_reviews.jsonl")]
    cat = "Bikes"
    cat_records = [r for r in records if r["category"] == cat]
    cands = extract_broadened(cat_records)
    sentences = [c.sentence for c in cands]
    emb = SentenceEmbedder(n_components=24, random_state=42).fit_transform(sentences)
    km, k, sil = induce_taxonomy(emb, k_min=3, k_max=10, random_state=42)

    facet_names = [u for u, a in TAXONOMY[cat]]
    n_facets = len(facet_names)
    mat = np.zeros((n_facets, k), dtype=int)
    for c, cl in zip(cands, km.labels_):
        if c.facet_id != -1:
            mat[c.facet_id, cl] += 1

    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    im = ax.imshow(mat, cmap="Blues", aspect="auto")
    ax.set_xticks(range(k)); ax.set_xticklabels([f"C{i}" for i in range(k)])
    ax.set_yticks(range(n_facets)); ax.set_yticklabels(facet_names, fontsize=8.5)
    ax.set_xlabel("Induced cluster id (unsupervised)")
    ax.set_ylabel("Gold usage facet (Bikes category)")
    ax.set_title("Induced-cluster \u00d7 gold-facet contingency table (Bikes)\ndarker = more candidate sentences of that facet fall in that cluster")
    for i in range(n_facets):
        for j in range(k):
            if mat[i, j] > 0:
                ax.text(j, i, mat[i, j], ha="center", va="center",
                        color="white" if mat[i, j] > mat.max() / 2 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8, label="# candidate sentences")
    fig.tight_layout()
    fig.savefig(FIG / "fig8_cluster_confusion_bikes.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 9: usage-space map with facet NAME labels at cluster centroids
# ---------------------------------------------------------------------------
def fig9_labeled_embedding_space():
    import json
    from data_generation import TAXONOMY
    from extraction import extract_broadened

    cache = np.load(DATA / "bikes_embedding_cache.npz")
    emb, facet_ids = cache["emb"], cache["facet_ids"]
    pca = PCA(n_components=2, random_state=42)
    xy = pca.fit_transform(emb)

    facet_names = [u for u, a in TAXONOMY["Bikes"]]
    fig, ax = plt.subplots(figsize=(11.2, 7.2))
    cmap = plt.get_cmap("tab10")

    present = [fid for fid in sorted(set(facet_ids)) if fid != -1]
    centroids = {}
    for fid in present:
        mask = facet_ids == fid
        ax.scatter(xy[mask, 0], xy[mask, 1], color=cmap(fid % 10), s=55, alpha=0.8,
                   edgecolor="white", linewidth=0.4, zorder=3)
        centroids[fid] = (xy[mask, 0].mean(), xy[mask, 1].mean())

    # leader-line label layout: stack labels evenly down the right margin,
    # ordered by centroid y, and connect each to its centroid with a thin line
    x_right = xy[:, 0].max() + 0.45
    y_top, y_bot = xy[:, 1].max(), xy[:, 1].min()
    ordered = sorted(present, key=lambda f: -centroids[f][1])
    n = len(ordered)
    for i, fid in enumerate(ordered):
        ly = y_top - (y_top - y_bot) * (i / max(1, n - 1))
        cx, cy = centroids[fid]
        color = cmap(fid % 10)
        ax.plot([cx, x_right - 0.05], [cy, ly], color=color, linewidth=0.8, alpha=0.6, zorder=1)
        ax.text(x_right, ly, facet_names[fid], fontsize=9, weight="bold", va="center", ha="left",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=color, linewidth=1.2))

    ax.set_xlim(xy[:, 0].min() - 0.15, x_right + 2.2)
    ax.set_title("Labeled usage-space map \u2014 \"Bikes\"\neach point = one extracted candidate sentence; leader lines connect each facet's centroid to its name")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    fig.tight_layout()
    fig.savefig(FIG / "fig9_labeled_usage_space_bikes.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 10: object-count funnel through pipeline stages
# ---------------------------------------------------------------------------
def fig10_funnel():
    df = pd.read_csv(TABLES / "table6_funnel.csv")
    stages = df["stage"].tolist()
    stage_labels = ["1. Extraction\n(ReviewRecord\u2192Candidate)", "2. Embedding\n(\u2192EmbeddingRecord)",
                    "3. Clustering\n(\u2192ClusterAssignment)", "4. Selection\n(\u2192SelectedCandidate)",
                    "5. Generation\n(\u2192GeneratedQuestion)"]
    base = df["Baseline (TQG-style)"].tolist()
    prop = df["CA-UQG (proposed)"].tolist()

    x = np.arange(len(stages))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    b1 = ax.bar(x - width / 2, base, width, label="Baseline (TQG-style)", color=PALETTE["Baseline (TQG-style)"])
    b2 = ax.bar(x + width / 2, prop, width, label="CA-UQG (proposed)", color=PALETTE["CA-UQG (proposed)"])
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 5, int(b.get_height()),
                    ha="center", fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(stage_labels, fontsize=8.6)
    ax.set_ylabel("Object count (summed across 12 categories)")
    ax.set_title("Object-count funnel through the CA-UQG pipeline stages")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "fig10_pipeline_funnel.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig1_pipeline()
    fig2_budget_sweep()
    fig3_per_category_bars()
    fig4_vendi_bars()
    fig5_embedding_map()
    fig6_object_schema()
    fig7_annotated_extraction()
    fig8_cluster_confusion()
    fig9_labeled_embedding_space()
    fig10_funnel()
    print("Figures written to", FIG)
