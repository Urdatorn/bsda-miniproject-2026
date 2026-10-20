"""An interactive map of the topic vocabulary, with the restorations drawn on it.

Every topic-bearing lemma of the vocabulary is one point, placed by UMAP
from its skip-gram word2vec vector (the space the 34 topics were clustered
in, so the topics are the regions of the map) and coloured by its topic.
A point's size is how often the walk uses the lemma, and a topic label's
size (and so the order in which labels give way as one zooms out) is the
walk's share of that topic; hovering shows the
lemma's rate per thousand topic lemmata in Sophocles and in the walk; the
alternative colouring shows the walk's rate against Sophocles' (log2 ratio,
blue where the walk under-uses a word, red where it over-uses it).  The
search box finds a lemma.

  python scripts/lemma_map.py --data data --out figures

writes hard_lemma_map.html (self-contained) and hard_lemma_map.png.
"""
import argparse, html, json, os
from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def lemma_counts(table):
    counts = defaultdict(int)
    for idx, cnt in zip(table["lemma_idx"].to_pylist(), table["count"].to_pylist()):
        for i, c in zip(idx, cnt):
            counts[int(i)] += int(c)
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--neighbors", type=int, default=15)
    ap.add_argument("--min-dist", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=917)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    vocab = json.load(open(os.path.join(a.data, "vocab.json"), encoding="utf-8"))
    vectors = np.load(os.path.join(a.data, "vocab_vectors.npy"))
    corpus = pq.read_table(os.path.join(a.data, "corpus.parquet"))
    imp = pq.read_table(os.path.join(a.data, "imputations.parquet"))
    source = np.array(corpus["source"].to_pylist())
    restorer = np.array(imp["restorer"].to_pylist()); tilted = np.array([bool(t) for t in imp["style_tilted"].to_pylist()])
    counts = {"sophocles": lemma_counts(corpus.filter(source == "sophocles")),
              "walk": lemma_counts(imp.filter((restorer == "walk") & ~tilted))}

    rows = [r for r in vocab if r["cluster_K40"] and np.any(vectors[r["index"]])]
    index = np.array([r["index"] for r in rows])
    labels = np.array([r["cluster_K40"] for r in rows])
    lemmata = [r["lemma"] for r in rows]
    total = {cell: sum(c for i, c in counts[cell].items() if i in set(index)) for cell in counts}
    rate = {cell: np.array([counts[cell].get(i, 0) for i in index]) / total[cell] * 1000 for cell in counts}
    print(f"{len(rows)} topic-bearing lemmata with a vector; topic lemmata counted: " +
          ", ".join(f"{cell} {total[cell]}" for cell in total), flush=True)

    import umap
    reducer = umap.UMAP(n_neighbors=a.neighbors, min_dist=a.min_dist, metric="cosine", random_state=a.seed)
    coords = reducer.fit_transform(vectors[index])
    print("UMAP done", flush=True)
    log_ratio = np.log2((rate["walk"] + 0.05) / (rate["sophocles"] + 0.05))

    import datamapplot
    import datamapplot.create_plots
    sizes = 1.0 + 3.0 * np.sqrt(rate["walk"])

    # datamapplot sizes a label (and ranks it when labels collide) by its cluster's point count;
    # here a label is sized by the walk's share of the topic instead, so the labels that survive
    # zooming out are the topics the walk uses most.  The function that builds the label table is
    # wrapped to overwrite its size column (square root, so the small topics stay legible).
    walk_share = pd.Series(rate["walk"]).groupby(labels).sum()
    build_labels = datamapplot.create_plots.label_text_and_polygon_dataframes

    def labels_sized_by_walk(*args, **kwargs):
        table = build_labels(*args, **kwargs)
        table["size"] = np.sqrt(table["label"].map(walk_share).fillna(0.0).to_numpy())
        return table
    datamapplot.create_plots.label_text_and_polygon_dataframes = labels_sized_by_walk
    extra = pd.DataFrame({"topic": labels, "rate_sophocles": np.round(rate["sophocles"], 2), "rate_walk": np.round(rate["walk"], 2),
                          "ratio": np.round(log_ratio, 2)})
    template = ("<div><b style='font-size:1.3em'>{hover_text}</b><br>topic: <i>{topic}</i><br>"
                "per 1000 topic lemmata: Sophocles {rate_sophocles}, walk {rate_walk} (log2 ratio {ratio})</div>")
    plot = datamapplot.create_interactive_plot(
        coords, labels, hover_text=[html.escape(w) for w in lemmata], extra_point_data=extra, hover_text_html_template=template,
        marker_size_array=sizes, enable_search=True, search_field="hover_text",
        title="The topic vocabulary, with the walk drawn on it",
        sub_title=f"{len(rows)} lemmata placed by their word2vec vectors and coloured by topic; a point's size is the walk's use of the word, "
                  "a label's size the walk's share of the topic",
        font_family="Georgia", tooltip_font_family="Georgia", label_wrap_width=14, text_min_pixel_size=12, text_max_pixel_size=28,
        colormap_rawdata=[log_ratio],
        colormap_metadata=[{"field": "ratio", "description": "walk against Sophocles (log2 rate ratio)", "kind": "continuous", "cmap": "RdBu_r"}],
        point_radius_max_pixels=14, initial_zoom_fraction=0.95)
    plot.save(os.path.join(a.out, "hard_lemma_map.html"))
    print(f"wrote {a.out}/hard_lemma_map.html ({os.path.getsize(os.path.join(a.out, 'hard_lemma_map.html')) / 1e6:.1f} MB)", flush=True)

    fig, ax = datamapplot.create_plot(coords, labels, label_font_size=7, label_wrap_width=14, marker_size_array=sizes * 2,
                                      title="The topic vocabulary", sub_title="lemmata placed by word2vec, coloured by topic, sized by the walk's use",
                                      figsize=(12, 12))
    fig.savefig(os.path.join(a.out, "hard_lemma_map.png"), dpi=150, bbox_inches="tight")
    print(f"wrote {a.out}/hard_lemma_map.png", flush=True)


if __name__ == "__main__":
    main()
