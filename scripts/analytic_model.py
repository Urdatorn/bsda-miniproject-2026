#!/usr/bin/env python3
"""The analytic model: unsupervised topics (hard clusters), exact conjugate posterior.

Topics.  Each vocabulary lemma carries a cluster label in vocab.json: the K = 40
Ward cut of a word2vec space of Ancient Greek lemmata (skip-gram on the
lemmatised OGA corpus, validated against the AGREE benchmark), minus one
cluster of rare lemmata and four clusters of function words.  A topic is a hard cluster; its label is the two
non-cognate lemmata nearest its centroid.  The clustering is upstream of this
repository (Urdatorn/tereus-imputation, scripts/semantic_clusters.py and
scripts/cluster_topics.py); here the labels are data.

Model.  A document of N = 64 topic-bearing lemmata has topic counts
n = (n_1 ... n_K) ~ Multinomial(N, theta), theta ~ Dirichlet(alpha 1).  The
posterior is conjugate: theta | n ~ Dirichlet(alpha + n), exactly.  A cell
(one lacuna, one restorer) is the equal mixture of its documents' posteriors,
so its spread is the spread between restorations plus the Dirichlet
uncertainty of each.  Each cell is compared with Sophocles' extant trimeters
treated the same way: P(theta_k^cell > theta_k^Sophocles) over paired draws,
and the lift of the posterior means.

Usage: python scripts/analytic_model.py --data data --out results/analytic
Writes posterior.csv (cell x topic) and posterior.md.
"""
import argparse, json, os, sys
from collections import Counter
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from bsda.data import load, documents, FIRST, LAST, RESTORERS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data"); ap.add_argument("--out", default="results/analytic")
    ap.add_argument("--n", type=int, default=64, help="lemmata per document")
    ap.add_argument("--alpha", type=float, default=1.0, help="symmetric Dirichlet prior")
    ap.add_argument("--max-docs", type=int, default=1000, help="documents per lacuna cell (Sophocles 4x, walk 2x)")
    ap.add_argument("--draws", type=int, default=100, help="posterior draws per document")
    ap.add_argument("--seed", type=int, default=917)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); rng = np.random.default_rng(a.seed)
    D = load(a.data)
    # topics in order of their mass in the tragic corpus
    mass = Counter(); [mass.update({D.cluster[w]: 1}) for w in D.streams["tragic"] if D.cluster[w]]
    names = [t for t, _ in mass.most_common()]; K = len(names); tid = {t: k for k, t in enumerate(names)}
    topic_of = np.array([tid.get(c, -1) if c else -1 for c in D.cluster])            # lemma index -> topic, -1 = noise / no topic
    covered = np.mean([topic_of[w] >= 0 for w in D.streams["real"]])
    print(f"{K} topics; {covered:.1%} of Sophocles' vocabulary lemmata carry a topic", flush=True)

    def to_topics(stream): return [int(topic_of[w]) for w in stream if topic_of[w] >= 0]

    def posterior(docs):
        N = np.stack([np.bincount(d, minlength=K) for d in docs])                            # (docs, K) topic counts
        return np.concatenate([rng.dirichlet(a.alpha + n, a.draws) for n in N]), N          # the exact posterior, mixed over documents

    def summarise(docs, label, base=None):
        if not docs: return None, None
        P, N = posterior(docs); pooled = a.alpha + N.sum(0)
        s = {"label": label, "n_docs": len(docs), "n_tokens": int(N.sum()), "mean": P.mean(0).tolist(), "q05": np.quantile(P, 0.05, 0).tolist(), "q95": np.quantile(P, 0.95, 0).tolist(),
             "p_largest": (np.bincount(P.argmax(1), minlength=K) / len(P)).tolist(), "pooled_mean": (pooled / pooled.sum()).tolist(), "counts": N.sum(0).tolist()}
        if base is not None:
            B = base[rng.integers(0, len(base), len(P))]                                      # paired draws from the Sophocles mixture
            s["p_above_sophocles"] = (P > B).mean(0).tolist(); s["lift"] = (P.mean(0) / B.mean(0)).tolist()
        return s, P

    real, P_real = summarise(documents(to_topics(D.streams["real"]), a.n, a.max_docs * 4, rng=rng), "Sophocles, extant trimeters")
    res = {"names": names, "K": K, "n": a.n, "alpha": a.alpha, "real": real,
           "walk": summarise(documents(to_topics(D.streams["walk"]), a.n, a.max_docs * 2, rng=rng), "walk passages", P_real)[0],
           "henry": summarise([to_topics(D.henry)], "intact vv. 1-12 + Henry's vv. 13-29", P_real)[0], "lacunae": {}}
    for v in range(FIRST, LAST + 1):
        res["lacunae"][v] = {r: summarise(documents(to_topics(D.streams[(v, r)]), a.n, a.max_docs, shuffle=True, rng=rng), f"v. {v}, {r}", P_real)[0] for r in RESTORERS}

    cells = [(str(v), r, res["lacunae"][v][r]) for v in range(FIRST, LAST + 1) for r in RESTORERS] + [("walk", "", res["walk"]), ("henry", "", res["henry"]), ("sophocles", "", res["real"])]
    rows = ["cell,restorer,topic,n_docs,n_tokens,count,mean,q05,q95,p_largest,p_above_sophocles,lift"]
    for v, r, s in cells:
        if s:
            for k in range(K):
                rows.append(f"{v},{r},{names[k]},{s['n_docs']},{s['n_tokens']},{s['counts'][k]},{s['mean'][k]:.4f},{s['q05'][k]:.4f},{s['q95'][k]:.4f},{s['p_largest'][k]:.4f},"
                            + (f"{s['p_above_sophocles'][k]:.4f},{s['lift'][k]:.3f}" if "lift" in s else ","))
    open(f"{a.out}/posterior.csv", "w", encoding="utf-8").write("\n".join(rows) + "\n")
    md = ["# Analytic model: unsupervised topics, closed-form posterior\n",
          f"{K} topics (hard clusters of the vocabulary, labels from vocab.json); documents of {a.n} topic-bearing lemmata; theta ~ Dirichlet({a.alpha:g}), "
          f"posterior Dirichlet({a.alpha:g} + n) per document with {a.draws} draws, mixed over the cell's documents; Sophocles baseline of {real['n_docs']} documents.\n",
          "## Topics above the Sophocles baseline\n", "Topics with P(theta_k > theta_k of Sophocles) >= 0.9, by lift of the posterior mean (lift, posterior mean).\n",
          "| cell | restorer | docs | topics raised |", "|---|---|---|---|"]
    for v, r, s in cells:
        if s and "lift" in s:
            up = sorted((k for k in range(K) if s["p_above_sophocles"][k] >= 0.9), key=lambda k: -s["lift"][k])
            md.append(f"| {v} | {r} | {s['n_docs']} | " + ", ".join(f"**{names[k]}** ({s['lift'][k]:.1f}×, {s['mean'][k]:.2f})" for k in up) + " |")
    md += ["\n## Posterior mean per topic\n", "| cell | " + " | ".join(names) + " |", "|---|" + "---|" * K]
    for v, r, s in cells:
        if s: md.append(f"| {v} {r} | " + " | ".join(f"{m:.2f}" for m in s["mean"]) + " |")
    open(f"{a.out}/posterior.md", "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md[4:4 + 56])); print(f"wrote {a.out}/posterior.{{csv,md}}", flush=True)


if __name__ == "__main__":
    main()
