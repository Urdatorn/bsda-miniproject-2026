#!/usr/bin/env python3
"""The soft model: seeded topics with soft topic-word distributions, exact posterior by NUTS.

Generative model (a seeded LDA, BDA3 §22.3 as a Dirichlet-multinomial mixture):
the topic-word distributions phi (K x V) are fitted once with tomotopy on
tragedy, with one topic seeded per unsupervised cluster of the vocabulary (the
clusters of the hard model, each seeded with all its lemmata), plus --free
unseeded topics.  A document is one whole restoration of the seventeen lost
verses: its N content lemmata, however many there are, drawn from the mixture
sum_k theta_k phi_k with theta ~ Dirichlet(1) on the simplex.  phi is fitted on
Aeschylus and Euripides only, so that Sophocles' windows, the baseline, are
test documents like the restorations and phi is not fitted on the very text it
is then compared with.

The posterior.  With phi fixed, p(theta | document) of one document is the
(K-1)-dimensional density prod_n sum_k theta_k phi[k, w_n] on the simplex, and
Stan's NUTS samples it exactly in about a second: the program is
lda_given_phi.stan (data: the N x K matrix of log phi for the document's
lemmata).  Every document of every set (the imputations' snapshots, Sophocles'
windows, the windows of Aeschylus and Euripides and of Classical Attic) is
sampled with --chains chains of --warmup + --samples iterations by a pool of
--workers CmdStan processes, --keep draws of each are kept, and Stan's own
diagnostics (split R-hat, bulk ESS, divergent transitions, tree-depth hits) are
recorded for every document.

The documents.  The imputations are the object: one document per untilted snapshot
(one chain at one step, all seventeen verses).  Sophocles' extant trimeters in
windows of seventeen are the baseline, the same shape as the lacuna.  Only
restored text enters a document; the surviving letters of a damaged verse are
never counted.  The imputations are read on two levels, as in the hard model: its mean
profile (the documents' posteriors averaged draw by draw), compared with
Sophocles' by P(mean_k^imp > mean_k^Soph) and the lift of the means, and a
random document against a random window of Sophocles,
P(theta_k^doc > theta_k^window).

Usage (a GH200 node for its cores; CMDSTAN must point at the aarch64 CmdStan):
  python scripts/soft_model.py --data data --out results/soft
Writes phi.json (topics; a free topic is named by the glosses of its three top lemmata) and phi_matrix.csv.gz (K x V, read by the report and
the lemma map), diagnostics.csv.gz (one row per document), check_documents.json
(two documents with their posterior summary, refitted with rstan in the
report), profiles.csv (the mean profiles of the imputations and of Sophocles, with the
imputations' lift), comparisons.csv (the imputations against Sophocles, against Aeschylus
and Euripides and against Classical Attic, and Sophocles against the latter
two, with the lift's 90 % interval over the posterior draws of the two mean
profiles), and theta_draws.npz (the kept draws of every document, for the
dataset, not for git).
"""
import argparse, gzip, json, os, shutil, tempfile, time, unicodedata
from collections import Counter, defaultdict
from multiprocessing import Pool
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
    docs: {"sophocles": windows, "imputations": snapshots, "tragedians": the windows of Aeschylus and Euripides (also the LDA's
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
    imputations = imp.filter((restorer == "walk") & ~tilted)
    snapshot = defaultdict(list)
    for key_set, key_chain, key_step, lemmata in zip(imputations["chain_set"].to_pylist(), imputations["chain"].to_pylist(), imputations["iteration"].to_pylist(), row_lemmata(walk)):
        snapshot[(key_set, key_chain, key_step)] += lemmata
    docs["imputations"] = [snapshot[k] for k in sorted(snapshot, key=str)]
    docs["tragedians"] = list(tragic)
    attic_path = os.path.join(data_dir, "attic_windows.parquet")
    if os.path.exists(attic_path):
        docs["attic"] = row_lemmata(pq.read_table(attic_path))
    else:
        print(f"no {attic_path}: the Classical Attic baseline is skipped (scripts/window_baselines.py writes it)", flush=True)
    docs = {name: [d for d in ds if d] for name, ds in docs.items()}                  # a document needs at least one lemma
    return [r["lemma"] for r in vocab], [r["cluster_K40"] for r in vocab], docs, tragic


# --- the exact posterior of one document: NUTS on lda_given_phi.stan, in a pool of CmdStan processes ---
# English glosses for naming the free topics (extend when a new lemma turns up)
GLOSS = {"παῖς": "child", "πατήρ": "father", "δόμος": "house", "θεός": "god", "βροτός": "mortal", "πρῶτος": "first",
         "κακός": "evil", "λόγος": "word", "φίλος": "friend", "χείρ": "hand", "πόλις": "city", "χθών": "earth",
         "θνῄσκω": "die", "γυνή": "woman", "τέκνον": "child", "μήτηρ": "mother", "κτείνω": "kill", "οἶκος": "home",
         "σοφός": "wise", "χάρις": "grace", "πείθω": "persuade", "ἄγω": "lead", "δίκη": "justice", "ὄμμα": "eye",
         "ἀνήρ": "man", "καλός": "fine", "δράω": "do", "ξένος": "stranger", "τύχη": "fortune", "ὁράω": "see",
         "γῆ": "land", "νεκρός": "corpse", "στρατός": "army", "δόρυ": "spear", "πούς": "foot"}

WORKER = {}


def worker_init(exe_file, logphi_path, chains, warmup, samples, keep):
    """each worker holds the compiled program and the log phi matrix once"""
    import cmdstanpy, logging
    cmdstanpy.utils.get_logger().setLevel(logging.WARNING)                              # no line per chain in the log
    WORKER.update(model=cmdstanpy.CmdStanModel(exe_file=exe_file), logphi=np.load(logphi_path),
                  chains=chains, warmup=warmup, samples=samples, keep=keep)


def fit_one(job):
    """(index, lemma indices, seed) -> (index, the kept theta draws (keep, K), Stan's diagnostics for the document)"""
    i, words, seed = job
    logphi = WORKER["logphi"]; K = logphi.shape[1]; t0 = time.time()
    tmp = tempfile.mkdtemp(prefix="soft-")                                             # CmdStan's csv files, removed at once
    try:
        fit = WORKER["model"].sample(data={"N": len(words), "K": K, "logphi": logphi[words].tolist()},
                                     chains=WORKER["chains"], parallel_chains=1, iter_warmup=WORKER["warmup"],
                                     iter_sampling=WORKER["samples"], seed=int(seed), output_dir=tmp,
                                     show_progress=False, show_console=False)
        theta = fit.stan_variable("theta")                                             # (chains * samples, K)
        summary = fit.summary(percentiles=(5, 95)).loc[[f"theta[{k + 1}]" for k in range(K)]]
        divergences = fit.divergences; depth = fit.max_treedepths
        diagnostics = {"max_rhat": float(summary["R_hat"].max()), "min_ess_bulk": float(summary["ESS_bulk"].min()),
                       "divergences": int(np.sum(divergences)) if divergences is not None else 0,
                       "treedepth_hits": int(np.sum(depth)) if depth is not None else 0, "seconds": time.time() - t0}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    step = max(1, len(theta) // WORKER["keep"])
    return i, theta[::step][:WORKER["keep"]].astype(np.float32), diagnostics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data"); ap.add_argument("--out", default="results/soft")
    ap.add_argument("--stan", default="lda_given_phi.stan", help="the Stan program of one document's posterior")
    ap.add_argument("--free", type=int, default=4, help="unseeded topics beside the seeded topics")
    ap.add_argument("--seed-strength", type=float, default=200.0, help="prior weight of a seed word on its topic")
    ap.add_argument("--iters", type=int, default=1000, help="Gibbs iterations for phi")
    ap.add_argument("--chains", type=int, default=4); ap.add_argument("--warmup", type=int, default=500); ap.add_argument("--samples", type=int, default=500)
    ap.add_argument("--keep", type=int, default=100, help="posterior draws kept per document (thinned from chains x samples)")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4)), help="CmdStan processes side by side")
    ap.add_argument("--max-docs", type=int, default=0, help="documents read per set, sampled at random if more (0 = all); for dry runs")
    ap.add_argument("--seed", type=int, default=917)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    vocab, cluster, docs, tragic = load(a.data); V = len(vocab); wid = {w: i for i, w in enumerate(vocab)}
    if a.max_docs:
        docs = {name: [ds[i] for i in sorted(rng.choice(len(ds), a.max_docs, replace=False))] if len(ds) > a.max_docs else ds for name, ds in docs.items()}
    lengths = {name: np.array([len(d) for d in ds]) for name, ds in docs.items()}
    print(f"vocabulary {V} lemmata; documents: " + "; ".join(f"{name} {len(ds)} (mean {lengths[name].mean():.1f} lemmata, max {lengths[name].max()})" for name, ds in docs.items())
          + f"; {len(tragic)} windows of Aeschylus and Euripides for the LDA", flush=True)

    # ---- the Stan program, compiled once
    import cmdstanpy, logging
    cmdstanpy.utils.get_logger().setLevel(logging.WARNING)
    model = cmdstanpy.CmdStanModel(stan_file=a.stan)
    print(f"CmdStan {cmdstanpy.cmdstan_version()} at {cmdstanpy.cmdstan_path()}; compiled {model.exe_file}", flush=True)

    # ---- seeded LDA for phi, fitted on real tragedy only; the restorations (and Sophocles) are test documents
    import tomotopy as tp
    mass = Counter(cluster[w] for d in tragic for w in d if cluster[w])            # one seeded topic per cluster, in order of tragic mass
    seeds = {c: [vocab[i] for i in range(V) if cluster[i] == c] for c, _ in mass.most_common()}
    H = len(seeds); K = H + a.free; names = list(seeds) + [f"free {j}" for j in range(a.free)]   # renamed below
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
    phi /= phi.sum(1, keepdims=True); logphi = np.log(phi).T                               # (V, K)
    # a free topic is named by the English glosses of its three most probable
    # lemmata, as a seeded topic is named by its cluster's label lemmata
    for j in range(a.free):
        words = [w for w, _ in lda.get_topic_words(H + j, top_n=3)]
        missing = [w for w in words if w not in GLOSS]
        if missing: print(f"  no English gloss for {' '.join(missing)}: add it to GLOSS in soft_model.py", flush=True)
        names[H + j] = "-".join(GLOSS.get(w, w) for w in words)
    assert len(set(names)) == K, "two topics got the same name"
    with gzip.open(f"{a.out}/phi_matrix.csv.gz", "wt", encoding="utf-8") as f:          # K rows, one column per lemma
        f.write("topic," + ",".join(vocab) + "\n")
        for k in range(K): f.write(names[k] + "," + ",".join(f"{v:.6g}" for v in phi[k]) + "\n")
    top = {names[k]: [w for w, _ in lda.get_topic_words(k, top_n=15)] for k in range(K)}
    json.dump({"names": names, "top_words": top, "ll_per_word": lda.ll_per_word, "seeds_found": found, "vocab_size": V}, open(f"{a.out}/phi.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for k in range(K): print(f"  {names[k]:18s} " + " ".join(top[names[k]][:10]), flush=True)

    # ---- every document of every set through NUTS
    scratch = tempfile.mkdtemp(prefix="soft-logphi-"); logphi_path = os.path.join(scratch, "logphi.npy"); np.save(logphi_path, logphi)
    draws, diagnostics = {}, []
    sets = [name for name in ("sophocles", "imputations", "tragedians", "attic") if name in docs]
    print(f"sampling {sum(len(docs[n]) for n in sets)} documents with {a.workers} workers: {a.chains} chains x ({a.warmup} + {a.samples}) iterations each, {a.keep} draws kept", flush=True)
    with Pool(a.workers, initializer=worker_init, initargs=(model.exe_file, logphi_path, a.chains, a.warmup, a.samples, a.keep)) as pool:
        for offset, name in enumerate(sets):
            word_lists = docs[name]; jobs = [(i, w, a.seed * 1000 + offset * 100000 + i) for i, w in enumerate(word_lists)]
            P3 = np.zeros((len(jobs), a.keep, K), np.float32); rows = [None] * len(jobs); t0 = time.time(); done = 0; last_log = t0
            for i, kept, d in pool.imap_unordered(fit_one, jobs, chunksize=4):
                P3[i] = kept; rows[i] = d; done += 1
                if done == len(jobs) or done % 1000 == 0 or time.time() - last_log > 120:
                    elapsed = time.time() - t0; rate = done / elapsed; seen = [r for r in rows if r]
                    print(f"  {name}: {done}/{len(jobs)} documents in {elapsed:.0f} s ({rate:.1f}/s, about {(len(jobs) - done) / rate:.0f} s left); "
                          f"largest R-hat so far {max(r['max_rhat'] for r in seen):.3f}, smallest bulk ESS {min(r['min_ess_bulk'] for r in seen):.0f}, "
                          f"documents with divergences {sum(r['divergences'] > 0 for r in seen)}", flush=True)
                    last_log = time.time()
            draws[name] = P3
            diagnostics += [{"source": name, "document": i, "n_lemmata": len(word_lists[i]), **rows[i]} for i in range(len(jobs))]
    shutil.rmtree(scratch, ignore_errors=True)
    with gzip.open(f"{a.out}/diagnostics.csv.gz", "wt") as f:
        f.write("source,document,n_lemmata,max_rhat,min_ess_bulk,divergences,treedepth_hits,seconds\n")
        for r in diagnostics:
            f.write(f"{r['source']},{r['document']},{r['n_lemmata']},{r['max_rhat']:.4f},{r['min_ess_bulk']:.1f},{r['divergences']},{r['treedepth_hits']},{r['seconds']:.2f}\n")
    for name in sets:
        rs = [r for r in diagnostics if r["source"] == name]
        print(f"  {name}: {len(rs)} documents; R-hat > 1.01 in {sum(r['max_rhat'] > 1.01 for r in rs)}, bulk ESS < 400 in {sum(r['min_ess_bulk'] < 400 for r in rs)}, "
              f"divergences in {sum(r['divergences'] > 0 for r in rs)}, tree-depth hits in {sum(r['treedepth_hits'] > 0 for r in rs)}; median {np.median([r['seconds'] for r in rs]):.1f} s per document", flush=True)
    np.savez_compressed(f"{a.out}/theta_draws.npz", names=np.array(names), **{name: draws[name].astype(np.float16) for name in sets})

    # ---- two documents with their posterior summary, refitted with rstan in the report
    check = []
    for name in ("imputations", "sophocles"):
        i = int(rng.integers(len(docs[name]))); P = draws[name][i].astype(np.float64)
        check.append({"source": name, "index": i, "lemma_idx": [int(w) for w in docs[name][i]], "mean": P.mean(0).round(5).tolist(),
                      "sd": P.std(0).round(5).tolist(), "q05": np.quantile(P, 0.05, 0).round(5).tolist(), "q95": np.quantile(P, 0.95, 0).round(5).tolist()})
    json.dump({"names": names, "draws": a.keep, "documents": check}, open(f"{a.out}/check_documents.json", "w"), indent=None)

    # ---- the sets, each read on two levels
    def summarise(name, base=None):
        """base: the Sophocles summary. Returns the set's summary with its draws kept for comparisons."""
        P3 = draws[name]; mean_draws = P3.mean(0).astype(np.float64); P = P3.reshape(-1, K).astype(np.float64); seeded = P[:, :H]
        s = {"n_docs": len(P3), "n_lemmata": float(lengths[name].mean()),
             "mean": mean_draws.mean(0).tolist(), "q05": np.quantile(mean_draws, 0.05, 0).tolist(), "q95": np.quantile(mean_draws, 0.95, 0).tolist(),
             "p_dominant_seeded": (np.bincount(seeded.argmax(1), minlength=H) / len(P)).tolist(), "free_mass": float(P[:, H:].sum(1).mean()),
             "_mean_draws": mean_draws, "_document_draws": P}
        if base is not None:
            s["p_above_sophocles"] = (mean_draws > base["_mean_draws"]).mean(0).tolist()         # paired by draw index
            s["lift"] = (mean_draws.mean(0) / base["_mean_draws"].mean(0)).tolist()
            B = base["_document_draws"][rng.integers(0, len(base["_document_draws"]), len(P))]
            s["p_document"] = (P > B).mean(0).tolist()
        return s
    profiles = {"sophocles": summarise("sophocles")}
    for name in ("imputations", "tragedians", "attic"):
        if name in docs: profiles[name] = summarise(name, profiles["sophocles"])

    # the comparisons behind the circle grids: focus against baseline, seeded topics only, the lift's interval over
    # the posterior draws of the two mean profiles (paired by draw index) and the document-level probability
    def compare(focus, base, name):
        F, Bm = profiles[focus]["_mean_draws"][:, :H], profiles[base]["_mean_draws"][:, :H]
        ratio = F / Bm
        P, Q = profiles[focus]["_document_draws"][:, :H], profiles[base]["_document_draws"][:, :H]
        Q = Q[rng.integers(0, len(Q), len(P))]
        return [{"topic": names[k], "comparison": name, "rate_focus": F.mean(0)[k], "rate_baseline": Bm.mean(0)[k],
                 "lift": F.mean(0)[k] / Bm.mean(0)[k], "lift_q05": np.quantile(ratio[:, k], 0.05), "lift_q95": np.quantile(ratio[:, k], 0.95),
                 "p_above": (F[:, k] > Bm[:, k]).mean(), "p_document": (P[:, k] > Q[:, k]).mean()} for k in range(H)]
    comparisons = []
    for focus, base in (("imputations", "sophocles"), ("imputations", "tragedians"), ("imputations", "attic"), ("sophocles", "tragedians"), ("sophocles", "attic")):
        if focus in profiles and base in profiles: comparisons += compare(focus, base, f"{focus} vs {base}")
    with open(f"{a.out}/comparisons.csv", "w") as f:
        f.write("topic,comparison,rate_focus,rate_baseline,lift,lift_q05,lift_q95,p_above,p_document\n")
        for r in comparisons:
            f.write(f"{r['topic']},{r['comparison']},{r['rate_focus']:.5f},{r['rate_baseline']:.5f},{r['lift']:.4f},{r['lift_q05']:.4f},{r['lift_q95']:.4f},{r['p_above']:.4f},{r['p_document']:.4f}\n")
    for name in dict.fromkeys(r["comparison"] for r in comparisons):
        rows = sorted((r for r in comparisons if r["comparison"] == name and (r["lift_q05"] > 1 or r["lift_q95"] < 1)), key=lambda r: -r["lift"])
        print(f"  {name}: {len(rows)} topics with an interval off 1; largest " + ", ".join(f"{r['topic']} {r['lift']:.2f}" for r in rows[:3])
              + "; smallest " + ", ".join(f"{r['topic']} {r['lift']:.2f}" for r in rows[-3:]), flush=True)
    with open(f"{a.out}/profiles.csv", "w") as f:
        f.write("source,topic,n_docs,n_lemmata,mean,q05,q95,p_dominant,p_above_sophocles,lift,p_document\n")
        for name, s in profiles.items():
            for j in range(H):
                f.write(f"{name},{names[j]},{s['n_docs']},{s['n_lemmata']:.1f},{s['mean'][j]:.4f},{s['q05'][j]:.4f},{s['q95'][j]:.4f},{s['p_dominant_seeded'][j]:.4f},"
                        + (f"{s['p_above_sophocles'][j]:.4f},{s['lift'][j]:.3f},{s['p_document'][j]:.4f}" if "lift" in s else ",,") + "\n")
    raised = {name: [names[k] for k in sorted((k for k in range(H) if s["p_above_sophocles"][k] >= 0.9 and s["lift"][k] >= 1.5),
                                              key=lambda k: -s["lift"][k])] for name, s in profiles.items() if "lift" in s}
    print(f"{H} seeded and {a.free} free topics on {V} lemmata; phi on {len(tragic)} windows of Aeschylus and Euripides; "
          f"{len(diagnostics)} documents sampled by NUTS", flush=True)
    for name, s in profiles.items():
        if "lift" in s:
            top3 = sorted(range(H), key=lambda k: -s["lift"][k])[:3]
            print(f"  {name}: raised {raised[name] or 'none'}; largest lifts " + ", ".join(f"{names[k]} {s['lift'][k]:.2f}" for k in top3), flush=True)
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
