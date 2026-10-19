"""An interactive map of the restorations in the word2vec topic space.

Every document of the analytic model is one point: a walk snapshot (one
chain at one step, all seventeen verses), Henry's seventeen verses, and
every window of seventeen consecutive extant trimeters of Sophocles.  A
document's position comes from its topic profile, the share of its lemmata
in each of the 34 word2vec topics (square-root shares, so that distances are
Hellinger distances between profiles), laid out in two dimensions by UMAP;
its colour is its dominant topic, the topic with the most lemmata in it.
With --space word2vec the position is instead the mean of the document's
word2vec lemma vectors, which for documents of twenty lemmata averages out
to a featureless cloud.  Hovering shows the Greek text, searching finds a
word in it, and a second colouring separates the walk, Henry and Sophocles.

  python scripts/topic_map.py --data data --out results/analytic

writes topic_map.html (the interactive map, self-contained) and
topic_map.png (a static version for the report).
"""
import argparse, html, json, os
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq

FIRST, LAST = 13, 29
WINDOW = LAST - FIRST + 1


def documents(data_dir, max_walk, rng):
    """one record per document: cell, name, chain, step, lemma counts (vocabulary index -> count), Greek text"""
    corpus = pq.read_table(os.path.join(data_dir, "corpus.parquet")).to_pandas()
    imp = pq.read_table(os.path.join(data_dir, "imputations.parquet")).to_pandas()
    docs = []

    sophocles = corpus[corpus.source == "sophocles"].sort_values("line")
    n_windows = len(sophocles) // WINDOW
    for w in range(n_windows):
        rows = sophocles.iloc[w * WINDOW:(w + 1) * WINDOW]
        docs.append({"cell": "sophocles", "name": f"Sophocles, window {w + 1} (lines {int(rows.line.iloc[0]) + 1}-{int(rows.line.iloc[-1]) + 1})",
                     "chain": "", "step": -1, "counts": bag(rows), "text": " / ".join(rows.text)})

    henry = corpus[corpus.source == "henry"].sort_values("line")
    docs.append({"cell": "henry", "name": "Henry, vv. 13-29", "chain": "", "step": -1, "counts": bag(henry), "text": " / ".join(henry.text)})

    walk = imp[(imp.restorer == "walk") & (imp.style_tilted == False)].copy()  # noqa: E712  (untilted snapshots only)
    walk["chain_id"] = np.where(walk.chain.isna(), walk.chain_set.astype(str), walk.chain_set.astype(str) + " " + walk.chain.astype(str))
    groups = walk.groupby(["chain_id", "iteration"], sort=True)
    keys = list(groups.groups)
    if len(keys) > max_walk:
        keys = [keys[i] for i in sorted(rng.choice(len(keys), max_walk, replace=False))]
    for chain_id, step in keys:
        rows = groups.get_group((chain_id, step)).sort_values("verse")
        docs.append({"cell": "walk", "name": f"walk, chain {chain_id}, step {int(step)}", "chain": chain_id, "step": int(step),
                     "counts": bag(rows), "text": " / ".join(rows.text)})
    return docs


def bag(rows):
    counts = defaultdict(int)
    for idx, cnt in zip(rows.lemma_idx, rows["count"]):
        for i, c in zip(idx, cnt):
            counts[int(i)] += int(c)
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results/analytic")
    ap.add_argument("--max-walk", type=int, default=4000, help="walk snapshots drawn at random (all 12,722 make a 13 MB page)")
    ap.add_argument("--neighbors", type=int, default=15)
    ap.add_argument("--min-dist", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=917)
    ap.add_argument("--space", choices=["topics", "word2vec"], default="topics", help="what a document's position is computed from")
    ap.add_argument("--reuse-coords", action="store_true", help="reuse the saved layout instead of running UMAP again")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    vocab = json.load(open(os.path.join(a.data, "vocab.json"), encoding="utf-8"))
    vectors = np.load(os.path.join(a.data, "vocab_vectors.npy"))
    topic_of = {r["index"]: r["cluster_K40"] for r in vocab if r["cluster_K40"]}
    docs = documents(a.data, a.max_walk, rng)
    print(f"{len(docs)} documents: " + ", ".join(f"{c} {sum(d['cell'] == c for d in docs)}" for c in ("sophocles", "henry", "walk")), flush=True)

    # a document's label is the topic with most lemmata in it; its position comes from its topic profile
    # (square-root shares) or, with --space word2vec, from the count-weighted mean of its lemma vectors
    topics = sorted(set(topic_of.values())); column = {t: k for k, t in enumerate(topics)}
    profile = np.zeros((len(docs), len(topics)), dtype=np.float32)
    mean_vector = np.zeros((len(docs), vectors.shape[1]), dtype=np.float32)
    labels, keep = [], []
    for n, d in enumerate(docs):
        weight = 0.0
        for i, c in d["counts"].items():
            if i in topic_of and np.any(vectors[i]):
                profile[n, column[topic_of[i]]] += c; mean_vector[n] += c * vectors[i]; weight += c
        if weight > 0:
            mean_vector[n] /= weight; keep.append(n)
            labels.append(topics[int(np.argmax(profile[n]))])
    docs = [docs[n] for n in keep]
    profile = profile[keep]; profile = np.sqrt(profile / profile.sum(1, keepdims=True))
    X = profile if a.space == "topics" else mean_vector[keep]
    labels = np.array(labels)
    # a topic that dominates fewer than ten documents cannot be outlined; those documents are left unlabelled
    NOISE = "minor dominant topic"
    for topic in set(labels):
        if (labels == topic).sum() < 10:
            labels[labels == topic] = NOISE
    print(f"{len(docs)} documents with topic lemmata; {len(set(labels) - {NOISE})} dominant topics outlined", flush=True)

    import pandas as pd
    os.makedirs(a.out, exist_ok=True)
    coords_path = os.path.join(a.out, "topic_map_coords.csv")
    if a.reuse_coords and os.path.exists(coords_path):
        coords = pd.read_csv(coords_path)[["x", "y"]].to_numpy()
        print(f"reused the layout in {coords_path}", flush=True)
    else:
        import umap
        metric = "euclidean" if a.space == "topics" else "cosine"       # euclidean on square-root shares = Hellinger
        reducer = umap.UMAP(n_neighbors=a.neighbors, min_dist=a.min_dist, metric=metric, random_state=a.seed)
        coords = reducer.fit_transform(X)
        print("UMAP done", flush=True)
    pd.DataFrame({"name": [d["name"] for d in docs], "cell": [d["cell"] for d in docs], "topic": labels,
                  "x": coords[:, 0], "y": coords[:, 1]}).to_csv(coords_path, index=False)

    import datamapplot
    cells = np.array([d["cell"] for d in docs])
    sizes = np.where(cells == "henry", 8.0, np.where(cells == "sophocles", 3.0, 1.5))
    hover = [html.escape(d["text"]) for d in docs]
    extra = {"name": [d["name"] for d in docs], "cell": cells, "chain": [d["chain"] for d in docs], "step": [d["step"] for d in docs],
             "topic": labels, "n_lemmata": [sum(d["counts"].values()) for d in docs]}
    extra = pd.DataFrame(extra)
    template = ("<div style='max-width:420px'><b>{name}</b><br>dominant topic: <i>{topic}</i> ({n_lemmata} lemmata)<br>"
                "<span style='font-size:0.9em'>{hover_text}</span></div>")
    cell_colors = {"walk": "#4e79a7", "henry": "#e15759", "sophocles": "#59a14f"}
    plot = datamapplot.create_interactive_plot(
        coords, labels, hover_text=hover, extra_point_data=extra, hover_text_html_template=template,
        marker_size_array=sizes, enable_search=True, search_field="hover_text",
        cluster_boundary_polygons=True, color_cluster_boundaries=True, polygon_alpha=4.0,  # in the 30-unit space datamapplot rescales the layout to
        title="Restorations of P.Oxy. 5292 in the topic space",
        sub_title=f"{int((cells == 'walk').sum())} walk snapshots, Henry, and {int((cells == 'sophocles').sum())} Sophoclean windows of seventeen trimeters; "
                  "a point is a document's profile over the 34 word2vec topics, coloured by its dominant topic",
        font_family="Georgia", tooltip_font_family="Georgia", label_wrap_width=14, text_min_pixel_size=12, text_max_pixel_size=28,
        colormap_rawdata=[cells], colormap_metadata=[{"field": "cell", "description": "Source", "kind": "categorical",
                                                     "color_mapping": cell_colors, "show_legend": True}],
        point_radius_max_pixels=12, initial_zoom_fraction=0.95, noise_label=NOISE)
    plot.save(os.path.join(a.out, "topic_map.html"))
    print(f"wrote {a.out}/topic_map.html ({os.path.getsize(os.path.join(a.out, 'topic_map.html')) / 1e6:.1f} MB)", flush=True)

    fig, ax = datamapplot.create_plot(coords, labels, noise_label=NOISE, label_font_size=7, label_wrap_width=14, marker_size_array=sizes * 2,
                                      title="Restorations in the topic space",
                                      sub_title="walk snapshots (small), Sophoclean windows (medium), Henry (large)", figsize=(12, 12))
    fig.savefig(os.path.join(a.out, "topic_map.png"), dpi=150, bbox_inches="tight")
    print(f"wrote {a.out}/topic_map.png", flush=True)


if __name__ == "__main__":
    main()
