#!/usr/bin/env python3
"""The amortized mixed model: seeded topics with soft topic-word distributions, amortized posterior.

Generative model (a seeded LDA, BDA3 §22.3 as a Dirichlet-multinomial mixture):
the topic-word distributions phi (K x V) are fitted once with tomotopy on the
three tragedians, with one topic seeded per unsupervised cluster of the
vocabulary (the 35 clusters of the analytic model, each seeded with all its
lemmata), plus --free unseeded topics.  A document is then n content lemmata
drawn from the mixture sum_k theta_k phi_k with theta ~ Dirichlet(1) on the
simplex.  The quantity of interest is p(theta | document) for every lacuna
cell; "the lost text of v. 21 is about marriage" is the posterior statement
that theta_marriage-bed is the largest of the seeded topics.

Amortization (BayesFlow).  A word enters the likelihood only through its K
log-likelihoods log phi[:, w], so a document is a set of K-vectors and a
DeepSet is the right summary network.  A coupling flow learns theta in
additive log-ratio coordinates from --sims simulated documents and is checked
on 1000 held-out simulations: recovery R², 90 % interval coverage and
simulation-based calibration (rank histograms).  The trained network then
reads the real documents: Sophocles' extant trimeters, the fills of every
lacuna per restorer, the walk passages and Henry's text.  Every cell is also
compared with Sophocles by P(theta_k^cell > theta_k^Sophocles) over paired
posterior draws and by the lift of the posterior means, as in the analytic model.

Usage (GPU recommended; KERAS_BACKEND=torch):
  python scripts/amortized_mixed_model.py --data data --out results/amortized_mixed
Writes phi.json (topics), sbc.csv, recovery.csv, cells.{json,md,csv}.
"""
import argparse, json, os, time, unicodedata
from collections import Counter
from dataclasses import dataclass
import numpy as np, pyarrow.parquet as pq

FIRST, LAST = 13, 29
RESTORERS = ("stoicheia", "candidates", "walk")


# --- the data: bags of lemmata from the Hugging Face dataset, read into streams ---
# A bag is a sparse count vector over the V content lemmata of vocab.json; a
# document for the model is n lemmata, cut from a stream of bags.

def expand(table):
    """every bag of a table, as the flat list of its lemma indices (counts expanded)"""
    out = []
    for ids, cnt in zip(table["lemma_idx"].to_pylist(), table["count"].to_pylist()):
        out.extend(np.repeat(ids, cnt).tolist())
    return out


@dataclass
class Data:
    """vocab: the lemmata (index = column of a bag); cluster: the unsupervised topic of
    each lemma or None; streams: "real" (Sophocles' extant trimeters), "tragic" (the three
    tragedians), "walk" (the untilted walk passages) and (verse, restorer) (the fills of one
    lacuna cell), each a list of lemma indices in reading order; henry: the papyrus as Henry
    reads it"""
    vocab: list; cluster: list; streams: dict; henry: list

    @property
    def V(self): return len(self.vocab)


def load(data_dir="data"):
    vocab = json.load(open(os.path.join(data_dir, "vocab.json"), encoding="utf-8"))
    corpus = pq.read_table(os.path.join(data_dir, "corpus.parquet"))
    imp = pq.read_table(os.path.join(data_dir, "imputations.parquet"))
    src = np.array(corpus["source"].to_pylist())
    streams = {"real": expand(corpus.filter(src == "sophocles")),
               "tragic": expand(corpus.filter(np.isin(src, ["sophocles", "aeschylus", "euripides"])))}
    restorer = np.array(imp["restorer"].to_pylist()); verse = np.array(imp["verse"].to_pylist())
    tilted = np.array([bool(t) for t in imp["style_tilted"].to_pylist()])
    streams["walk"] = expand(imp.filter((restorer == "walk") & ~tilted))
    for v in range(FIRST, LAST + 1):
        for r in RESTORERS:
            streams[(v, r)] = expand(imp.filter((verse == v) & (restorer == r) & ~tilted))
    intact = corpus.filter(src == "intact"); henry = corpus.filter(src == "henry")
    henry_doc = expand(intact.take(np.argsort(intact["line"].to_pylist()))) + expand(henry.take(np.argsort(henry["line"].to_pylist())))
    return Data([r["lemma"] for r in vocab], [r["cluster_K40"] for r in vocab], streams, henry_doc)


def documents(stream, n=64, limit=None, shuffle=False, rng=None):
    """cut a stream into documents of exactly n lemmata; at most `limit`, sampled; shuffled first if asked"""
    ids = list(stream)
    if shuffle: ids = list(rng.permutation(ids))
    docs = [ids[i:i + n] for i in range(0, len(ids) - n + 1, n)]
    if limit and len(docs) > limit: docs = [docs[i] for i in sorted(rng.choice(len(docs), limit, replace=False))]
    return docs


# --- the model ---


def alr(theta): return np.log(theta[..., :-1] / theta[..., -1:])


def alr_inv(z):
    z = np.concatenate([z, np.zeros_like(z[..., :1])], -1); z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data"); ap.add_argument("--out", default="results/amortized_mixed")
    ap.add_argument("--free", type=int, default=4, help="unseeded topics beside the seeded topics")
    ap.add_argument("--seed-strength", type=float, default=200.0, help="prior weight of a seed word on its topic")
    ap.add_argument("--n", type=int, default=64, help="lemmata per document")
    ap.add_argument("--iters", type=int, default=1000, help="Gibbs iterations for phi")
    ap.add_argument("--sims", type=int, default=200000, help="simulated documents for training the flow")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--post-draws", type=int, default=300)
    ap.add_argument("--max-docs", type=int, default=1000, help="documents per lacuna cell (Sophocles 4x, walk 2x)")
    ap.add_argument("--seed", type=int, default=917)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    os.environ.setdefault("KERAS_BACKEND", "torch"); rng = np.random.default_rng(a.seed)
    D = load(a.data); V = D.V; wid = {w: i for i, w in enumerate(D.vocab)}
    print(f"vocabulary {V} lemmata; streams: Sophocles {len(D.streams['real'])}, tragic {len(D.streams['tragic'])}, walk {len(D.streams['walk'])}, "
          f"fills per cell median {int(np.median([len(D.streams[(v, r)]) for v in range(FIRST, LAST + 1) for r in RESTORERS]))}", flush=True)

    # ---- seeded LDA for phi, fitted on real tragedy only; the restorations are test documents
    import tomotopy as tp
    # one seeded topic per cluster of vocab.json, in order of tragic mass
    mass = Counter(D.cluster[w] for w in D.streams["tragic"] if D.cluster[w])
    seeds = {c: [D.vocab[i] for i in range(V) if D.cluster[i] == c] for c, _ in mass.most_common()}
    H = len(seeds); K = H + a.free; names = list(seeds) + [f"free {j}" for j in range(a.free)]
    print(f"{H} seeded topics (the clusters) + {a.free} free", flush=True)
    lda = tp.LDAModel(tw=tp.TermWeight.ONE, k=K, alpha=0.1, eta=0.01, seed=a.seed)
    for d in documents(D.streams["tragic"], a.n): lda.add_doc([D.vocab[i] for i in d])
    found = {}
    for j, (name, words) in enumerate(seeds.items()):
        found[name] = []
        for w in (unicodedata.normalize("NFC", w.lower()) for w in words):
            if w in wid:
                prior = [0.01] * K; prior[j] = a.seed_strength; lda.set_word_prior(w, prior); found[name].append(w)
    print("seed lemmata in the vocabulary: " + "; ".join(f"{n}: {len(v)}/{len(seeds[n])}" for n, v in list(found.items())[:12]) + (" ..." if H > 12 else ""), flush=True)
    lda.train(a.iters, workers=8)
    order = {w: i for i, w in enumerate(lda.used_vocabs)}
    phi = np.full((K, V), 1e-8)
    for k in range(K):
        d = lda.get_topic_word_dist(k)
        for w, i in wid.items():
            if w in order: phi[k, i] = d[order[w]]
    phi /= phi.sum(1, keepdims=True); logphi = np.log(phi).T.astype(np.float32)          # (V, K)
    top = {names[k]: [w for w, _ in lda.get_topic_words(k, top_n=15)] for k in range(K)}
    json.dump({"names": names, "top_words": top, "ll_per_word": lda.ll_per_word, "seeds_found": found, "vocab_size": V}, open(f"{a.out}/phi.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for k in range(K): print(f"  {names[k]:18s} " + " ".join(top[names[k]][:10]), flush=True)

    # ---- simulator and amortized posterior
    import keras, bayesflow as bf
    keras.utils.set_random_seed(a.seed)

    def simulate(S):
        theta = rng.dirichlet(np.ones(K), S)
        words = np.stack([rng.choice(V, a.n, p=theta[s] @ phi) for s in range(S)])
        return {"inference_variables": alr(theta).astype(np.float32), "summary_variables": logphi[words]}
    sims = simulate(a.sims); held = simulate(1000)
    approx = bf.approximators.ContinuousApproximator(
        inference_network=bf.networks.CouplingFlow(depth=6, subnet_kwargs={"widths": (128, 128)}),
        summary_network=bf.networks.DeepSet(summary_dim=32, embed_dim=64, mlp_widths=(64, 64)),
        adapter=bf.approximators.ContinuousApproximator.build_adapter(inference_variables="inference_variables", summary_variables="summary_variables"))
    ds = bf.datasets.OfflineDataset(sims, batch_size=256, adapter=approx.adapter)
    vs = bf.datasets.OfflineDataset(held, batch_size=256, adapter=approx.adapter, shuffle=False)
    approx.compile(optimizer=keras.optimizers.AdamW(learning_rate=keras.optimizers.schedules.CosineDecay(1e-3, a.epochs * max(1, a.sims // 256)), weight_decay=1e-4))
    t0 = time.time()
    log = keras.callbacks.LambdaCallback(on_epoch_end=lambda ep, lg: print(f"  epoch {ep + 1}/{a.epochs}: loss {lg.get('loss', float('nan')):.3f} validation {lg.get('val_loss', float('nan')):.3f} ({time.time() - t0:.0f}s)", flush=True) if ep % 5 == 4 or ep == 0 else None)
    print(f"simulated {a.sims} documents; training the flow", flush=True)
    hist = approx.fit(dataset=ds, validation_data=vs, epochs=a.epochs, verbose=0, callbacks=[log, keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)])
    print(f"flow trained: {len(hist.history['loss'])} epochs, validation loss {np.nanmin(hist.history.get('val_loss', [np.nan])):.3f}", flush=True)

    def posterior(X, draws=a.post_draws):
        """X (docs, n, K) -> theta draws (docs, draws, K)"""
        out = approx.sample(conditions={"summary_variables": X.astype(np.float32)}, num_samples=draws)
        z = out["inference_variables"] if isinstance(out, dict) else out
        return alr_inv(np.asarray(z, dtype=np.float64))

    # ---- checks on held-out simulations: SBC ranks, recovery, coverage
    th_true = alr_inv(held["inference_variables"].astype(np.float64)); post = posterior(held["summary_variables"])
    ranks = (post < th_true[:, None, :]).sum(1)
    with open(f"{a.out}/sbc.csv", "w") as f:
        f.write("topic,rank\n"); f.writelines(f"{names[k]},{r}\n" for k in range(K) for r in ranks[:, k])
    pm = post.mean(1); r2 = [1 - ((pm[:, k] - th_true[:, k]) ** 2).sum() / ((th_true[:, k] - th_true[:, k].mean()) ** 2).sum() for k in range(K)]
    cover = [float(((np.quantile(post[:, :, k], 0.05, 1) <= th_true[:, k]) & (th_true[:, k] <= np.quantile(post[:, :, k], 0.95, 1))).mean()) for k in range(K)]
    with open(f"{a.out}/recovery.csv", "w") as f:
        f.write("topic,true,post_mean,q05,q95\n")
        for k in range(K):
            for i in range(300): f.write(f"{names[k]},{th_true[i, k]:.4f},{pm[i, k]:.4f},{np.quantile(post[i, :, k], 0.05):.4f},{np.quantile(post[i, :, k], 0.95):.4f}\n")
    from scipy.stats import chisquare
    sbc_p = [float(chisquare(np.histogram(ranks[:, k], bins=10, range=(0, a.post_draws + 1))[0]).pvalue) for k in range(K)]
    print("recovery R² by topic: " + " ".join(f"{names[k]} {r2[k]:.2f}" for k in range(K)), flush=True)
    print("90 % interval coverage: " + " ".join(f"{names[k]} {cover[k]:.2f}" for k in range(K)), flush=True)
    print("SBC uniformity p-values: " + " ".join(f"{names[k]} {sbc_p[k]:.2f}" for k in range(K)), flush=True)

    # ---- real documents
    def summarise(docs, label, base=None):
        if not docs: return None, None
        X = logphi[np.array(docs)]; P3 = posterior(X); P = P3.reshape(-1, K)          # pooled over documents: a mixture over samples of fills
        seeded = P[:, :H]; dom = np.bincount(seeded.argmax(1), minlength=H) / len(P)
        s = {"label": label, "n_docs": len(docs), "mean": P.mean(0).tolist(), "q05": np.quantile(P, 0.05, 0).tolist(), "q95": np.quantile(P, 0.95, 0).tolist(),
             "p_dominant_seeded": dom.tolist(), "p_above_025": (seeded > 0.25).mean(0).tolist(), "free_mass": float(P[:, H:].sum(1).mean())}
        if base is not None:                                                            # paired draws against the Sophocles mixture, as in the analytic model
            B = base[rng.integers(0, len(base), len(P))]; s["p_above_sophocles"] = (P > B).mean(0).tolist(); s["lift"] = (P.mean(0) / B.mean(0)).tolist()
        return s, P
    real, P_real = summarise(documents(D.streams["real"], a.n, a.max_docs * 4, rng=rng), "Sophocles, extant trimeters")
    res = {"names": names, "n": a.n, "real": real,
           "walk": summarise(documents(D.streams["walk"], a.n, a.max_docs * 2, rng=rng), "walk passages", P_real)[0],
           "henry": summarise([D.henry[:a.n]], "intact vv. 1-12 + Henry's vv. 13-29", P_real)[0] if len(D.henry) >= a.n else None, "lacunae": {}}
    for v in range(FIRST, LAST + 1):
        res["lacunae"][v] = {r: summarise(documents(D.streams[(v, r)], a.n, a.max_docs, shuffle=True, rng=rng), f"v. {v}, {r}", P_real)[0] for r in RESTORERS}
        print(f"  v. {v} read", flush=True)
    res["checks"] = {"recovery_r2": r2, "coverage90": cover, "sbc_p": sbc_p}
    json.dump(res, open(f"{a.out}/cells.json", "w"), indent=1)
    cells = [(str(v), r, res["lacunae"][v][r]) for v in range(FIRST, LAST + 1) for r in RESTORERS] + [("walk", "", res["walk"]), ("henry", "", res["henry"]), ("sophocles", "", res["real"])]
    with open(f"{a.out}/cells.csv", "w") as f:
        f.write("cell,restorer,topic,n_docs,mean,q05,q95,p_dominant,p_above_025,p_above_sophocles,lift\n")
        for v, r, s in cells:
            if s:
                for j in range(H): f.write(f"{v},{r},{names[j]},{s['n_docs']},{s['mean'][j]:.4f},{s['q05'][j]:.4f},{s['q95'][j]:.4f},{s['p_dominant_seeded'][j]:.4f},{s['p_above_025'][j]:.4f},"
                                           + (f"{s['p_above_sophocles'][j]:.4f},{s['lift'][j]:.3f}" if "lift" in s else ",") + "\n")
    md = [f"# Amortized mixed model: seeded LDA, amortized posterior (BayesFlow)\n",
          f"Seeded LDA with {H} seeded topics (the clusters) and {a.free} free topics on a vocabulary of {V} content lemmata; documents of {a.n} lemmata; prior theta ~ Dirichlet(1). "
          f"Posterior from a coupling flow trained on {a.sims} simulated documents; on 1000 held-out simulations: recovery R² "
          + ", ".join(f"{names[k]} {r2[k]:.2f}" for k in range(H)) + "; 90 % coverage " + ", ".join(f"{cover[k]:.2f}" for k in range(H)) + "; SBC p " + ", ".join(f"{sbc_p[k]:.2f}" for k in range(H)) + ".\n",
          "## Topics above the Sophocles baseline\n", "Seeded topics with P(theta_k > theta_k of Sophocles) >= 0.9 over paired posterior draws, by lift of the posterior mean (lift, posterior mean).\n",
          "| cell | restorer | docs | topics raised |", "|---|---|---|---|"]
    for v, r, s in cells:
        if s and "lift" in s:
            up = sorted((k for k in range(H) if s["p_above_sophocles"][k] >= 0.9), key=lambda k: -s["lift"][k])
            md.append(f"| {v} | {r} | {s['n_docs']} | " + ", ".join(f"**{names[k]}** ({s['lift'][k]:.1f}×, {s['mean'][k]:.2f})" for k in up) + " |")
    md += ["\n## Dominant seeded topic\n", "Each cell: P(this seeded topic is the largest of the seeded topics) over the cell's documents and posterior draws; last column the mean mass on the free topics.\n",
          "| cell | restorer | " + " | ".join(names[:H]) + " | free |", "|---|---|" + "---|" * (H + 1)]
    for v, r, s in cells:
        if s: md.append(f"| {v} | {r} | " + " | ".join(f"{p:.2f}" for p in s["p_dominant_seeded"]) + f" | {s['free_mass']:.2f} |")
    md += ["\nPosterior mean mass per seeded topic [5 %, 95 %]:\n", "| cell | restorer | " + " | ".join(names[:H]) + " |", "|---|---|" + "---|" * H]
    for v, r, s in cells:
        if s: md.append(f"| {v} | {r} | " + " | ".join(f"{m:.2f} [{lo:.2f}, {hi:.2f}]" for m, lo, hi in zip(s["mean"][:H], s["q05"][:H], s["q95"][:H])) + " |")
    md += ["\nTopics (top lemmata):\n"] + [f"- **{k}**: " + ", ".join(w) for k, w in top.items()]
    open(f"{a.out}/cells.md", "w", encoding="utf-8").write("\n".join(md) + "\n"); print("\n".join(md[:60]), flush=True)
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
