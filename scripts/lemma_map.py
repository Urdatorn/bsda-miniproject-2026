"""An interactive map of the topic vocabulary, with the restorations drawn on it.

Every topic-bearing lemma of the vocabulary is one point, placed by UMAP
from its skip-gram word2vec vector (the space the 34 topics were clustered
in, so the topics are the regions of the map) and coloured by its topic.
A point's size is how often the walk uses the lemma.

  python scripts/lemma_map.py --data data --out figures

writes hard_lemma_map.png and, when the soft model's results/soft/phi_matrix.csv.gz
exists, soft_lemma_map.png on the same layout: every lemma coloured by the
topic that gives it the most probability under the soft model's phi (the
seeded topic with the largest phi[k, w], or a free topic), so that the
lemmata the soft model moves out of their cluster show.  Text-width figures
in the ICML style (figure_style.py).
"""
import argparse, json, os
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
    ap.add_argument("--soft", default="results/soft", help="the soft model's results (phi.json, phi_matrix.csv.gz); skipped if absent")
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
    from figure_style import apply, font_family, TEXT_WIDTH
    apply()
    sizes = 1.0 + 3.0 * np.sqrt(rate["walk"])
    fig, ax = datamapplot.create_plot(coords, labels, label_font_size=6, label_wrap_width=14, marker_size_array=sizes,
                                      font_family=font_family(), title="The topic vocabulary",
                                      sub_title="lemmata placed by word2vec, coloured by topic, sized by the walk's use",
                                      title_keywords={"fontsize": 9}, sub_title_keywords={"fontsize": 7},
                                      figsize=(TEXT_WIDTH, TEXT_WIDTH), dpi=300)
    fig.savefig(os.path.join(a.out, "hard_lemma_map.png"), dpi=300, bbox_inches="tight")
    print(f"wrote {a.out}/hard_lemma_map.png", flush=True)

    # the soft map: the same layout, every lemma coloured by its soft topic
    phi_path = os.path.join(a.soft, "phi_matrix.csv.gz")
    if not os.path.exists(phi_path):
        print(f"no {phi_path}: the soft map is skipped", flush=True); return
    phi_table = pd.read_csv(phi_path, index_col=0)                           # (K, V), rows sum to one
    names = list(phi_table.index); phi = phi_table.to_numpy()
    responsibility = phi[:, index] / phi[:, index].sum(0, keepdims=True)      # P(topic | lemma) with a flat prior over topics
    soft_labels = np.array([names[k] for k in responsibility.argmax(0)])
    moved = soft_labels != labels
    print(f"soft topics: {moved.sum()} of {len(rows)} lemmata ({100 * moved.mean():.1f} %) get a topic other than their cluster; "
          + f"{np.isin(soft_labels, names[-(len(names) - len(set(labels))):]).sum()} go to a free topic", flush=True)
    fig, ax = datamapplot.create_plot(coords, soft_labels, label_font_size=6, label_wrap_width=14, marker_size_array=sizes,
                                      font_family=font_family(), title="The topic vocabulary under the soft model",
                                      sub_title=f"same layout, coloured by the soft topic; {moved.sum()} of {len(rows)} lemmata leave their cluster",
                                      title_keywords={"fontsize": 9}, sub_title_keywords={"fontsize": 7},
                                      figsize=(TEXT_WIDTH, TEXT_WIDTH), dpi=300)
    fig.savefig(os.path.join(a.out, "soft_lemma_map.png"), dpi=300, bbox_inches="tight")
    print(f"wrote {a.out}/soft_lemma_map.png", flush=True)


if __name__ == "__main__":
    main()
