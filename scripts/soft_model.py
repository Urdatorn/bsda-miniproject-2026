#!/usr/bin/env python3
"""The soft model: seeded topics with soft topic-word distributions, amortized posterior.

Generative model (a seeded LDA, BDA3 §22.3 as a Dirichlet-multinomial mixture):
the topic-word distributions phi (K x V) are fitted once with tomotopy on
tragedy, with one topic seeded per unsupervised cluster of the
vocabulary (the clusters of the hard model, each seeded with all its
lemmata), plus --free unseeded topics.  A document is one whole restoration of
the seventeen lost verses: its N content lemmata, however many there are, drawn
from the mixture sum_k theta_k phi_k with theta ~ Dirichlet(1) on the simplex.
The quantity of interest is p(theta | document) for every cell.  phi is
fitted on Aeschylus and Euripides only, so that Sophocles' windows, the
baseline, are test documents like the restorations and phi is not fitted on
the very text it is then compared with.

The cells.  The walk is the object: one document per untilted snapshot (one
chain at one step, all seventeen verses).  Sophocles' extant trimeters in
windows of seventeen are the reference, the same shape as the lacuna.  Only restored text enters a
document; the surviving letters of a damaged verse are never counted.

Amortization (BayesFlow).  A word enters the likelihood only through its K
log-likelihoods log phi[:, w], so a document is a set of K-vectors and a
DeepSet is the right summary network; documents differ in length, so the set
is padded and the DeepSet pools over the real elements only, with log N as an
extra summary.  A coupling flow learns theta in additive log-ratio coordinates
from --sims simulated documents whose lengths are drawn from the real
documents' lengths, and is checked on 1000 held-out simulations: recovery R²,
90 % interval coverage and simulation-based calibration (rank histograms).
Every cell is read on two levels, as in the hard model: its mean profile
(the documents' posteriors averaged draw by draw), compared with Sophocles'
by P(mean_k^cell > mean_k^Soph) and the lift of the means, and a random
document against a random window of Sophocles, P(theta_k^doc > theta_k^window).

Usage (GPU recommended; KERAS_BACKEND=torch):
  python scripts/soft_model.py --data data --out results/soft
Writes phi.json (topics) and phi_matrix.npy (K x V), sbc.csv, recovery.csv,
cells.{json,csv}, and comparisons.csv: the walk against Sophocles, against
Aeschylus and Euripides and against Classical Attic (data/attic_windows.parquet,
scripts/window_baselines.py), and Sophocles against the latter two, with the
lift's 90 % interval over the posterior draws of the two mean profiles.
"""
import argparse, json, os, time, unicodedata
from collections import Counter, defaultdict
import numpy as np, pyarrow.parquet as pq

FIRST, LAST = 13, 29
WINDOW = LAST - FIRST + 1          # seventeen verses
TRAGEDIANS = ("sophocles", "aeschylus", "euripides")


# --- the data: bags of lemmata from the Hugging Face dataset, read into documents ---
# A bag is a sparse count vector over the V content lemmata of vocab.json; a
# document is the concatenation of the bags of one whole restoration.

def row_lemmata(table):
    """one list of lemma indices per row of a table of bags (counts expanded)"""
    return [np.repeat(ids, cnt).tolist() for ids, cnt in zip(table["lemma_idx"].to_pylist(), table["count"].to_pylist())]


def in_line_order(table):
    return table.take(np.argsort(table["line"].to_pylist(), kind="stable"))


def windows(rows, size=WINDOW):
    """consecutive windows of `size` rows, each one document"""
    return [sum(rows[i:i + size], []) for i in range(0, len(rows) - size + 1, size)]


def load(data_dir):
    """vocab (the lemmata, index = column of a bag), cluster (the topic of each lemma or None),
    docs: {"sophocles": windows, "walk": snapshots, "tragedians": the windows of Aeschylus and Euripides (also the LDA's
    corpus), "attic": the Classical Attic windows of 28 lemmata}, and the windows of Aeschylus and Euripides for the LDA"""
    vocab = json.load(open(os.path.join(data_dir, "vocab.json"), encoding="utf-8"))
    corpus = pq.read_table(os.path.join(data_dir, "corpus.parquet"))
    imp = pq.read_table(os.path.join(data_dir, "imputations.parquet"))
    src = np.array(corpus["source"].to_pylist())
    docs, tragic = {}, []
    for author in TRAGEDIANS:
        author_windows = windows(row_lemmata(in_line_order(corpus.filter(src == author))))
        if author == "sophocles": docs["sophocles"] = author_windows
        if author != "sophocles": tragic += author_windows
    restorer = np.array(imp["restorer"].to_pylist())
    tilted = np.array([bool(t) for t in imp["style_tilted"].to_pylist()])
    walk = imp.filter((restorer == "walk") & ~tilted)
    snapshot = defaultdict(list)
    for key_set, key_chain, key_step, lemmata in zip(walk["chain_set"].to_pylist(), walk["chain"].to_pylist(), walk["iteration"].to_pylist(), row_lemmata(walk)):
        snapshot[(key_set, key_chain, key_step)] += lemmata
    docs["walk"] = [snapshot[k] for k in sorted(snapshot, key=str)]
    docs["tragedians"] = list(tragic)
    attic_path = os.path.join(data_dir, "attic_windows.parquet")
    if os.path.exists(attic_path):
        docs["attic"] = row_lemmata(pq.read_table(attic_path))
    else:
        print(f"no {attic_path}: the Classical Attic cell is skipped (scripts/window_baselines.py writes it)", flush=True)
    docs = {name: [d for d in ds if d] for name, ds in docs.items()}                  # a document needs at least one lemma
    return [r["lemma"] for r in vocab], [r["cluster_K40"] for r in vocab], docs, tragic


# --- the model ---

def alr(theta): return np.log(theta[..., :-1] / theta[..., -1:])


def alr_inv(z):
    z = np.concatenate([z, np.zeros_like(z[..., :1])], -1); z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data"); ap.add_argument("--out", default="results/soft")
    ap.add_argument("--free", type=int, default=4, help="unseeded topics beside the seeded topics")
    ap.add_argument("--seed-strength", type=float, default=200.0, help="prior weight of a seed word on its topic")
    ap.add_argument("--iters", type=int, default=1000, help="Gibbs iterations for phi")
    ap.add_argument("--sims", type=int, default=200000, help="simulated documents for training the flow")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--post-draws", type=int, default=300)
    ap.add_argument("--max-docs", type=int, default=0, help="walk snapshots read, sampled at random if more (0 = every snapshot)")
    ap.add_argument("--seed", type=int, default=917)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    os.environ.setdefault("KERAS_BACKEND", "torch"); rng = np.random.default_rng(a.seed)
    vocab, cluster, docs, tragic = load(a.data); V = len(vocab); wid = {w: i for i, w in enumerate(vocab)}
    if a.max_docs and len(docs["walk"]) > a.max_docs: docs["walk"] = [docs["walk"][i] for i in sorted(rng.choice(len(docs["walk"]), a.max_docs, replace=False))]
    lengths = {name: np.array([len(d) for d in ds]) for name, ds in docs.items()}
    print(f"vocabulary {V} lemmata; documents: " + "; ".join(f"{name} {len(ds)} (mean {lengths[name].mean():.1f} lemmata, max {lengths[name].max()})" for name, ds in docs.items())
          + f"; {len(tragic)} windows of Aeschylus and Euripides for the LDA", flush=True)
    n_max = int(max(l.max() for l in lengths.values()))
    length_pool = np.concatenate(list(lengths.values()))                              # the simulator draws N from the real documents' lengths

    # ---- seeded LDA for phi, fitted on real tragedy only; the restorations (and, by default, Sophocles) are test documents
    import tomotopy as tp
    mass = Counter(cluster[w] for d in tragic for w in d if cluster[w])            # one seeded topic per cluster, in order of tragic mass
    seeds = {c: [vocab[i] for i in range(V) if cluster[i] == c] for c, _ in mass.most_common()}
    H = len(seeds); K = H + a.free; names = list(seeds) + [f"free {j}" for j in range(a.free)]
    print(f"{H} seeded topics (the clusters) + {a.free} free", flush=True)
    lda = tp.LDAModel(tw=tp.TermWeight.ONE, k=K, alpha=0.1, eta=0.01, seed=a.seed)
    for d in tragic: lda.add_doc([vocab[i] for i in d])
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
    np.save(f"{a.out}/phi_matrix.npy", phi.astype(np.float32))                            # for the soft lemma map
    top = {names[k]: [w for w, _ in lda.get_topic_words(k, top_n=15)] for k in range(K)}
    json.dump({"names": names, "top_words": top, "ll_per_word": lda.ll_per_word, "seeds_found": found, "vocab_size": V}, open(f"{a.out}/phi.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for k in range(K): print(f"  {names[k]:18s} " + " ".join(top[names[k]][:10]), flush=True)

    # ---- simulator and amortized posterior
    import keras, bayesflow as bf
    keras.utils.set_random_seed(a.seed)

    def pad(word_lists):
        """documents of different lengths -> (docs, n_max, K) log-likelihoods and (docs, n_max) mask"""
        X = np.zeros((len(word_lists), n_max, K), np.float32); M = np.zeros((len(word_lists), n_max), np.float32)
        for i, words in enumerate(word_lists):
            words = words[:n_max]; X[i, :len(words)] = logphi[words]; M[i, :len(words)] = 1.0
        return X, M

    def simulate(S):
        theta = rng.dirichlet(np.ones(K), S); N = rng.choice(length_pool, S)
        X, M = pad([rng.choice(V, N[s], p=theta[s] @ phi).tolist() for s in range(S)])
        return {"inference_variables": alr(theta).astype(np.float32), "summary_variables": X, "summary_mask": M}

    class MaskedDeepSet(bf.networks.SummaryNetwork):
        """a DeepSet that pools over the real elements of a padded set, with log N as an extra summary"""
        def __init__(self, summary_dim=32, embed_dim=64, **kw):
            super().__init__(**kw)
            self.embed = keras.Sequential([keras.layers.Dense(embed_dim, activation="silu"), keras.layers.Dense(embed_dim, activation="silu")])
            self.head = keras.Sequential([keras.layers.Dense(64, activation="silu"), keras.layers.Dense(summary_dim)])

        def call(self, x, mask=None, **kwargs):
            if mask is None: mask = keras.ops.ones_like(x[..., :1])
            mask = keras.ops.reshape(keras.ops.cast(mask, x.dtype), (keras.ops.shape(x)[0], keras.ops.shape(x)[1], 1))
            h = self.embed(x) * mask
            n = keras.ops.maximum(keras.ops.sum(mask, axis=1), 1.0)                     # (batch, 1)
            mean = keras.ops.sum(h, axis=1) / n
            largest = keras.ops.max(h + (mask - 1.0) * 1e4, axis=1)
            return self.head(keras.ops.concatenate([mean, largest, keras.ops.log(n)], axis=-1))

    sims = simulate(a.sims); held = simulate(1000)
    approx = bf.approximators.ContinuousApproximator(
        inference_network=bf.networks.CouplingFlow(depth=6, subnet_kwargs={"widths": (128, 128)}),
        summary_network=MaskedDeepSet(summary_dim=32, embed_dim=64),
        adapter=bf.approximators.ContinuousApproximator.build_adapter(inference_variables="inference_variables", summary_variables="summary_variables", summary_mask="summary_mask"))
    ds = bf.datasets.OfflineDataset(sims, batch_size=256, adapter=approx.adapter)
    vs = bf.datasets.OfflineDataset(held, batch_size=256, adapter=approx.adapter, shuffle=False)
    approx.compile(optimizer=keras.optimizers.AdamW(learning_rate=keras.optimizers.schedules.CosineDecay(1e-3, a.epochs * max(1, a.sims // 256)), weight_decay=1e-4))
    t0 = time.time()
    log = keras.callbacks.LambdaCallback(on_epoch_end=lambda ep, lg: print(f"  epoch {ep + 1}/{a.epochs}: loss {lg.get('loss', float('nan')):.3f} validation {lg.get('val_loss', float('nan')):.3f} ({time.time() - t0:.0f}s)", flush=True) if ep % 5 == 4 or ep == 0 else None)
    print(f"simulated {a.sims} documents of {length_pool.min()}-{n_max} lemmata; training the flow", flush=True)
    hist = approx.fit(dataset=ds, validation_data=vs, epochs=a.epochs, verbose=0, callbacks=[log, keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)])
    print(f"flow trained: {len(hist.history['loss'])} epochs, validation loss {np.nanmin(hist.history.get('val_loss', [np.nan])):.3f}", flush=True)

    def posterior(X, M, draws=a.post_draws):
        """padded documents -> theta draws (docs, draws, K)"""
        out = approx.sample(conditions={"summary_variables": X, "summary_mask": M}, num_samples=draws)
        z = out["inference_variables"] if isinstance(out, dict) else out
        return alr_inv(np.asarray(z, dtype=np.float64))

    # ---- checks on held-out simulations: SBC ranks, recovery, coverage
    th_true = alr_inv(held["inference_variables"].astype(np.float64)); post = posterior(held["summary_variables"], held["summary_mask"])
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

    # ---- the real documents, cell by cell
    def summarise(word_lists, base=None):
        """base: the Sophocles summary. Returns the cell's summary with its draws kept for comparisons."""
        X, M = pad(word_lists); P3 = posterior(X, M)                                   # (docs, draws, K)
        mean_draws = P3.mean(0); P = P3.reshape(-1, K); seeded = P[:, :H]
        s = {"n_docs": len(word_lists), "n_lemmata": float(np.mean([len(w) for w in word_lists])),
             "mean": mean_draws.mean(0).tolist(), "q05": np.quantile(mean_draws, 0.05, 0).tolist(), "q95": np.quantile(mean_draws, 0.95, 0).tolist(),
             "p_dominant_seeded": (np.bincount(seeded.argmax(1), minlength=H) / len(P)).tolist(), "free_mass": float(P[:, H:].sum(1).mean()),
             "_mean_draws": mean_draws, "_document_draws": P}
        if base is not None:
            s["p_above_sophocles"] = (mean_draws > base["_mean_draws"]).mean(0).tolist()         # paired by draw index
            s["lift"] = (mean_draws.mean(0) / base["_mean_draws"].mean(0)).tolist()
            B = base["_document_draws"][rng.integers(0, len(base["_document_draws"]), len(P))]
            s["p_document"] = (P > B).mean(0).tolist()
        return s
    cells = {"sophocles": summarise(docs["sophocles"])}
    for name in ("walk", "tragedians", "attic"):
        if name in docs:
            cells[name] = summarise(docs[name], cells["sophocles"]); print(f"  {name} read: {cells[name]['n_docs']} documents", flush=True)

    # the comparisons behind the circle grids: focus against baseline, seeded topics only, the lift's interval over
    # the posterior draws of the two mean profiles (paired by draw index) and the document-level probability
    def compare(focus, base, name):
        F, Bm = cells[focus]["_mean_draws"][:, :H], cells[base]["_mean_draws"][:, :H]
        ratio = F / Bm
        P, Q = cells[focus]["_document_draws"][:, :H], cells[base]["_document_draws"][:, :H]
        Q = Q[rng.integers(0, len(Q), len(P))]
        return [{"topic": names[k], "comparison": name, "rate_focus": F.mean(0)[k], "rate_baseline": Bm.mean(0)[k],
                 "lift": F.mean(0)[k] / Bm.mean(0)[k], "lift_q05": np.quantile(ratio[:, k], 0.05), "lift_q95": np.quantile(ratio[:, k], 0.95),
                 "p_above": (F[:, k] > Bm[:, k]).mean(), "p_document": (P[:, k] > Q[:, k]).mean()} for k in range(H)]
    comparisons = []
    for focus, base in (("walk", "sophocles"), ("walk", "tragedians"), ("walk", "attic"), ("sophocles", "tragedians"), ("sophocles", "attic")):
        if focus in cells and base in cells: comparisons += compare(focus, base, f"{focus} vs {base}")
    with open(f"{a.out}/comparisons.csv", "w") as f:
        f.write("topic,comparison,rate_focus,rate_baseline,lift,lift_q05,lift_q95,p_above,p_document\n")
        for r in comparisons:
            f.write(f"{r['topic']},{r['comparison']},{r['rate_focus']:.5f},{r['rate_baseline']:.5f},{r['lift']:.4f},{r['lift_q05']:.4f},{r['lift_q95']:.4f},{r['p_above']:.4f},{r['p_document']:.4f}\n")
    for name in dict.fromkeys(r["comparison"] for r in comparisons):
        rows = sorted((r for r in comparisons if r["comparison"] == name and (r["lift_q05"] > 1 or r["lift_q95"] < 1)), key=lambda r: -r["lift"])
        print(f"  {name}: {len(rows)} topics with an interval off 1; largest " + ", ".join(f"{r['topic']} {r['lift']:.2f}" for r in rows[:3])
              + "; smallest " + ", ".join(f"{r['topic']} {r['lift']:.2f}" for r in rows[-3:]), flush=True)
    res = {"names": names, "cells": {name: {k: v for k, v in s.items() if not k.startswith("_")} for name, s in cells.items()},
           "checks": {"recovery_r2": r2, "coverage90": cover, "sbc_p": sbc_p}}
    json.dump(res, open(f"{a.out}/cells.json", "w"), indent=1)
    with open(f"{a.out}/cells.csv", "w") as f:
        f.write("cell,topic,n_docs,n_lemmata,mean,q05,q95,p_dominant,p_above_sophocles,lift,p_document\n")
        for name, s in cells.items():
            for j in range(H):
                f.write(f"{name},{names[j]},{s['n_docs']},{s['n_lemmata']:.1f},{s['mean'][j]:.4f},{s['q05'][j]:.4f},{s['q95'][j]:.4f},{s['p_dominant_seeded'][j]:.4f},"
                        + (f"{s['p_above_sophocles'][j]:.4f},{s['lift'][j]:.3f},{s['p_document'][j]:.4f}" if "lift" in s else ",,") + "\n")
    raised = {name: [names[k] for k in sorted((k for k in range(H) if s["p_above_sophocles"][k] >= 0.9 and s["lift"][k] >= 1.5),
                                              key=lambda k: -s["lift"][k])] for name, s in cells.items() if "lift" in s}
    print(f"{H} seeded and {a.free} free topics on {V} lemmata; phi on {len(tragic)} windows of Aeschylus and Euripides; "
          f"recovery R² {min(r2):.2f}-{max(r2):.2f}, coverage {min(cover):.2f}-{max(cover):.2f}, SBC p {min(sbc_p):.2f}-{max(sbc_p):.2f}", flush=True)
    for name, s in cells.items():
        if "lift" in s:
            top3 = sorted(range(H), key=lambda k: -s["lift"][k])[:3]
            print(f"  {name}: raised {raised[name] or 'none'}; largest lifts " + ", ".join(f"{names[k]} {s['lift'][k]:.2f}" for k in top3), flush=True)
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
