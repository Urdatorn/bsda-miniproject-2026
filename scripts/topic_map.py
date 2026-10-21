"""An interactive map of the restorations in the word2vec topic space.

Every document of the hard model is one point: a imputation (one
chain at one step, all seventeen verses) and every window of seventeen
consecutive extant trimeters of Sophocles.  A
document's position comes from its topic profile, the share of its lemmata
in each of the 34 word2vec topics (square-root shares, so that distances are
Hellinger distances between profiles), laid out in two dimensions by UMAP;
its colour is its dominant topic, the topic with the most lemmata in it.
With --space word2vec the position is instead the mean of the document's
word2vec lemma vectors, which for documents of twenty lemmata averages out
to a featureless cloud.

  python scripts/topic_map.py --data data --out figures

writes hard_topic_map.png (text width, ICML style, figure_style.py).
"""
import argparse, json, os
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq

FIRST, LAST = 13, 29
WINDOW = LAST - FIRST + 1


def documents(data_dir, max_imputations, rng):
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

    imputations = imp[(imp.restorer == "walk") & (imp.style_tilted == False)].copy()  # noqa: E712  (untilted snapshots only)
    imputations["chain_id"] = np.where(imputations.chain.isna(), imputations.chain_set.astype(str), imputations.chain_set.astype(str) + " " + imputations.chain.astype(str))
    groups = imputations.groupby(["chain_id", "iteration"], sort=True)
    keys = list(groups.groups)
    if len(keys) > max_imputations:
        keys = [keys[i] for i in sorted(rng.choice(len(keys), max_imputations, replace=False))]
    for chain_id, step in keys:
        rows = groups.get_group((chain_id, step)).sort_values("verse")
        docs.append({"cell": "imputations", "name": f"imputation, chain {chain_id}, step {int(step)}", "chain": chain_id, "step": int(step),
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
    ap.add_argument("--out", default="figures")
    ap.add_argument("--max-imputations", type=int, default=4000, help="imputations drawn at random (all 12,722 make a 13 MB page)")
    ap.add_argument("--neighbors", type=int, default=15)
    ap.add_argument("--min-dist", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=917)
    ap.add_argument("--space", choices=["topics", "word2vec"], default="topics", help="what a document's position is computed from")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    vocab = json.load(open(os.path.join(a.data, "vocab.json"), encoding="utf-8"))
    vectors = np.load(os.path.join(a.data, "vocab_vectors.npy"))
    topic_of = {r["index"]: r["cluster_K40"] for r in vocab if r["cluster_K40"]}
    docs = documents(a.data, a.max_imputations, rng)
    print(f"{len(docs)} documents: " + ", ".join(f"{c} {sum(d['cell'] == c for d in docs)}" for c in ("sophocles", "imputations")), flush=True)

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
    import umap
    metric = "euclidean" if a.space == "topics" else "cosine"       # euclidean on square-root shares = Hellinger
    reducer = umap.UMAP(n_neighbors=a.neighbors, min_dist=a.min_dist, metric=metric, random_state=a.seed)
    coords = reducer.fit_transform(X)
    print("UMAP done", flush=True)

    import datamapplot
    from figure_style import apply, font_family, TEXT_WIDTH
    apply()
    cells = np.array([d["cell"] for d in docs])
    sizes = np.where(cells == "sophocles", 2.0, 1.0)
    fig, ax = datamapplot.create_plot(coords, labels, noise_label=NOISE, label_font_size=6, label_wrap_width=14, marker_size_array=sizes,
                                      font_family=font_family(), title="Restorations in the topic space",
                                      sub_title="imputations (small), Sophoclean windows (larger), coloured by dominant topic",
                                      title_keywords={"fontsize": 9}, sub_title_keywords={"fontsize": 7},
                                      figsize=(TEXT_WIDTH, TEXT_WIDTH), dpi=300)
    fig.savefig(os.path.join(a.out, "hard_topic_map.png"), dpi=300, bbox_inches="tight")
    print(f"wrote {a.out}/hard_topic_map.png", flush=True)


if __name__ == "__main__":
    main()
