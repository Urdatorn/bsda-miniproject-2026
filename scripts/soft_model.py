#!/usr/bin/env python3
"""The soft model: seeded topics with soft topic-word distributions, exact posterior by NUTS.

Generative model (a seeded LDA, BDA3 §22.3 as a Dirichlet-multinomial mixture):
the topic-word distributions phi (K x V) are fitted with tomotopy on tragedy,
with one topic seeded per unsupervised cluster of the vocabulary (the clusters
of the hard model, each seeded with all its lemmata), plus --free unseeded
topics.  A document is one whole restoration of the seventeen lost verses: its
N content lemmata, however many there are, drawn from the mixture
sum_k theta_k phi_k with theta ~ Dirichlet(alpha) on the simplex.  phi and
alpha are fitted on Aeschylus and Euripides only, so that Sophocles' windows,
the baseline, are test documents like the restorations and phi is not fitted
on the very text it is then compared with.

Two stages, the model cut between them (phi and alpha are not informed by the
test documents):

  1. The LDA (--lda DIR, fitted when DIR/phi_draws.npz is missing or with
     --refit-lda; --fit-lda-only stops here).  Collapsed Gibbs sampling,
     single-threaded so that the seed reproduces it, --iters sweeps of burn-in,
     then --phi-draws states of the chain --draw-spacing sweeps apart.  Each
     state gives phi (the smoothed topic-word distributions given the current
     assignments) and alpha (tomotopy re-estimates the Dirichlet concentration
     of every topic by maximum marginal likelihood every ten sweeps, so alpha
     is an empirical-Bayes estimate, not a posterior draw).  Writes
     phi_draws.npz (every state), phi_matrix.csv.gz and phi.json (the mean
     over states, with alpha, read by the report).

  2. The posterior of every document.  With phi fixed, p(theta | document) is
     the (K-1)-dimensional density prod_n sum_k theta_k phi[k, w_n] times the
     Dirichlet prior on the simplex, and Stan's NUTS samples it exactly in a
     few seconds: the program is lda_given_phi.stan (data: the N x K matrix of
     log phi for the document's lemmata and the vector alpha) at --adapt-delta
     0.99: with alpha_k < 1 the density is unbounded at the corners of the
     simplex, and 0.8 leaves a few divergent transitions per document where
     0.99 leaves none (sampling theta through log-gammas did no better).  --phi picks the
     state ('mean', the default, or the index of a draw), --alpha the prior
     ('lda', the LDA's own alpha, or a number for a symmetric Dirichlet).
     Every document of every set (the imputations' snapshots, Sophocles'
     windows, the windows of Aeschylus and Euripides and of Classical Attic) is
     sampled with --chains chains of --warmup + --samples iterations by a pool
     of --workers CmdStan processes, --keep draws of each are kept, and Stan's
     own diagnostics (split R-hat, bulk ESS, divergent transitions, tree-depth
     hits) are recorded for every document.

Running stage 2 once per state of stage 1 (slurm/soft_draws.slurm, one array
task per draw) and pooling the runs (scripts/soft_pool.py) carries the
uncertainty in phi into the report's intervals; running it under the mean phi
with --alpha 1, 0.1 and 2 beside 'lda' is the prior check.

The documents.  The imputations are the object: one document per untilted
snapshot (one chain at one step, all seventeen verses).  Sophocles' extant
trimeters in windows of seventeen are the baseline, the same shape as the
lacuna.  Only restored text enters a document; the surviving letters of a
damaged verse are never counted.  The imputations are read on two levels, as
in the hard model: their mean profile (the documents' posteriors averaged draw
by draw), compared with Sophocles' by P(mean_k^imp > mean_k^Soph) and the lift
of the means, and a random document against a random window of Sophocles,
P(theta_k^doc > theta_k^window).

Usage (a GH200 node for its cores; CMDSTAN must point at the aarch64 CmdStan):
  python scripts/soft_model.py --data data --lda results/soft/lda --fit-lda-only
  python scripts/soft_model.py --data data --lda results/soft/lda --phi 3 --out results/soft/draws/d03
  python scripts/soft_model.py --data data --lda results/soft/lda --phi mean --alpha 1 --out results/soft/mean_alpha_1
Stage 2 writes to --out: diagnostics.csv.gz (one row per document),
check_documents.json (two documents with their data and posterior summary,
refitted with rstan in the report), profiles.csv (the mean profiles of the
sets with the imputations' lift), comparisons.csv (the imputations against
Sophocles, against Aeschylus and Euripides and against Classical Attic, and
Sophocles against the latter two, with the lift's 90 % interval over the
posterior draws of the two mean profiles), set_draws.npz (the draws of every
set's mean profile, every document's posterior mean, a subsample of documents
with all their draws, and the chain and step of every imputation: what
soft_pool.py pools) and, unless --no-draws, theta_draws.npz (the kept draws
of every document, for the dataset, not for git).
"""
import argparse, gzip, json, os, shutil, tempfile, time, unicodedata
from collections import Counter, defaultdict
from multiprocessing import Pool
import numpy as np, pyarrow.parquet as pq

FIRST, LAST = 13, 29
WINDOW = LAST - FIRST + 1          # seventeen verses
TRAGEDIANS = ("sophocles", "aeschylus", "euripides")
SETS = ("sophocles", "imputations", "tragedians", "attic")


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


def load(data_dir, chain_set=None):
    """vocab (the lemmata, index = column of a bag), cluster (the topic of each lemma or None),
    docs: {"sophocles": windows, "imputations": snapshots, "tragedians": the windows of Aeschylus and Euripides (also the LDA's
    corpus), "attic": the Classical Attic windows of 30 lemmata}, the windows of Aeschylus and Euripides for the LDA, and
    keys: (chain_set, chain, step) of every imputation, in the order of docs["imputations"]"""
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
    walk_set = np.array([c or "" for c in imp["chain_set"].to_pylist()])
    imputations = imp.filter((restorer == "walk") & ~tilted & ((walk_set == chain_set) if chain_set else True))
    snapshot = defaultdict(list)
    for key_set, key_chain, key_step, lemmata in zip(imputations["chain_set"].to_pylist(), imputations["chain"].to_pylist(), imputations["iteration"].to_pylist(), row_lemmata(imputations)):
        snapshot[(key_set, key_chain, key_step)] += lemmata
    keys = [k for k in sorted(snapshot, key=str) if snapshot[k]]                      # a document needs at least one lemma
    docs["imputations"] = [snapshot[k] for k in keys]
    docs["tragedians"] = list(tragic)
    attic_path = os.path.join(data_dir, "attic_windows.parquet")
    if os.path.exists(attic_path):
        docs["attic"] = row_lemmata(pq.read_table(attic_path))
    else:
        print(f"no {attic_path}: the Classical Attic baseline is skipped (scripts/window_baselines.py writes it)", flush=True)
    docs = {name: [d for d in ds if d] for name, ds in docs.items()}
    return [r["lemma"] for r in vocab], [r["cluster_K40"] for r in vocab], docs, tragic, keys


# --- stage 1: the seeded LDA on Aeschylus and Euripides, several states of its Gibbs chain ---
# English glosses for naming the free topics (extend when a new lemma turns up)
GLOSS = {"παῖς": "child", "πατήρ": "father", "δόμος": "house", "θεός": "god", "βροτός": "mortal", "πρῶτος": "first",
         "κακός": "evil", "λόγος": "word", "φίλος": "friend", "χείρ": "hand", "πόλις": "city", "χθών": "earth",
         "θνῄσκω": "die", "γυνή": "woman", "τέκνον": "child", "μήτηρ": "mother", "κτείνω": "kill", "οἶκος": "home",
         "σοφός": "wise", "χάρις": "grace", "πείθω": "persuade", "ἄγω": "lead", "δίκη": "justice", "ὄμμα": "eye",
         "ἀνήρ": "man", "καλός": "fine", "δράω": "do", "ξένος": "stranger", "τύχη": "fortune", "ὁράω": "see",
         "γῆ": "land", "νεκρός": "corpse", "στρατός": "army", "δόρυ": "spear", "πούς": "foot", "ἔχω": "have",
         "λέγω": "say", "φέρω": "carry", "μέγας": "great", "πᾶς": "all", "εἶδον": "see", "δίδωμι": "give"}


def seeded_topics(vocab, cluster, tragic):
    """one seeded topic per cluster of the hard model, in order of tragic mass -> {label: lemmata}"""
    mass = Counter(cluster[w] for d in tragic for w in d if cluster[w])
    return {c: [vocab[i] for i in range(len(vocab)) if cluster[i] == c] for c, _ in mass.most_common()}


def fit_lda(a, vocab, cluster, tragic, out):
    """the seeded LDA on Aeschylus and Euripides; writes phi_draws.npz, phi_matrix.csv.gz and phi.json to out"""
    import tomotopy as tp
    os.makedirs(out, exist_ok=True)
    V = len(vocab); wid = {w: i for i, w in enumerate(vocab)}
    seeds = seeded_topics(vocab, cluster, tragic)
    H = len(seeds); K = H + a.free; names = list(seeds) + [f"free {j}" for j in range(a.free)]   # renamed below
    print(f"{H} seeded topics (the clusters) + {a.free} free; {len(tragic)} windows, {sum(len(d) for d in tragic)} lemmata", flush=True)
    # alpha starts at 0.1 for every topic and is re-estimated every optim_interval sweeps
    lda = tp.LDAModel(tw=tp.TermWeight.ONE, k=K, alpha=0.1, eta=0.01, seed=a.seed)
    lda.optim_interval = 10
    for d in tragic: lda.add_doc([vocab[i] for i in d])
    found = {}
    for j, (name, words) in enumerate(seeds.items()):
        found[name] = []
        for w in (unicodedata.normalize("NFC", w.lower()) for w in words):
            if w in wid:
                prior = [0.01] * K; prior[j] = a.seed_strength; lda.set_word_prior(w, prior); found[name].append(w)
    print("seed lemmata in the vocabulary: " + "; ".join(f"{n}: {len(v)}/{len(seeds[n])}" for n, v in list(found.items())[:12]) + (" ..." if H > 12 else ""), flush=True)
    t0 = time.time()
    lda.train(a.iters, workers=a.lda_workers)                                           # burn-in
    print(f"burn-in of {a.iters} sweeps in {time.time() - t0:.0f} s; ll/word {lda.ll_per_word:.4f}; alpha: seeded mean {np.mean(lda.alpha[:H]):.3f}, free mean {np.mean(lda.alpha[H:]):.3f}", flush=True)
    order = {w: i for i, w in enumerate(lda.used_vocabs)}
    column = np.array([order.get(w, -1) for w in vocab])                                # -1: not in the LDA's vocabulary

    def state():
        """phi at the current state of the chain: the smoothed topic-word distributions given the assignments"""
        phi = np.full((K, V), 1e-8)
        for k in range(K):
            dist = np.asarray(lda.get_topic_word_dist(k))
            phi[k, column >= 0] = dist[column[column >= 0]]
        return phi / phi.sum(1, keepdims=True)

    phi_draws, alpha_draws, ll = [], [], []
    for j in range(a.phi_draws):
        if j: lda.train(a.draw_spacing, workers=a.lda_workers)
        phi_draws.append(state()); alpha_draws.append(np.array(lda.alpha, dtype=np.float64)); ll.append(lda.ll_per_word)
        print(f"  state {j}: sweep {a.iters + j * a.draw_spacing}, ll/word {ll[-1]:.4f}, alpha seeded {alpha_draws[-1][:H].mean():.3f} free {alpha_draws[-1][H:].mean():.3f} ({time.time() - t0:.0f} s)", flush=True)
    phi_draws = np.array(phi_draws); alpha_draws = np.array(alpha_draws)
    phi = phi_draws.mean(0); phi /= phi.sum(1, keepdims=True)
    # a free topic is named by the English glosses of its three most probable
    # lemmata (in the mean phi), as a seeded topic is named by its cluster's label lemmata
    for j in range(a.free):
        words = [vocab[i] for i in np.argsort(-phi[H + j])[:3]]
        missing = [w for w in words if w not in GLOSS]
        if missing: print(f"  no English gloss for {' '.join(missing)}: add it to GLOSS in soft_model.py", flush=True)
        names[H + j] = "-".join(GLOSS.get(w, w) for w in words)
    assert len(set(names)) == K, "two topics got the same name"
    np.savez_compressed(f"{out}/phi_draws.npz", phi=phi_draws.astype(np.float32), alpha=alpha_draws, names=np.array(names),
                        ll_per_word=np.array(ll), burnin=a.iters, spacing=a.draw_spacing, seeded=H)
    with gzip.open(f"{out}/phi_matrix.csv.gz", "wt", encoding="utf-8") as f:            # K rows, one column per lemma: the mean phi
        f.write("topic," + ",".join(vocab) + "\n")
        for k in range(K): f.write(names[k] + "," + ",".join(f"{v:.6g}" for v in phi[k]) + "\n")
    top = {names[k]: [vocab[i] for i in np.argsort(-phi[k])[:15]] for k in range(K)}
    json.dump({"names": names, "seeded": H, "top_words": top, "ll_per_word": float(np.mean(ll)), "seeds_found": found, "vocab_size": V,
               "alpha": alpha_draws.mean(0).round(5).tolist(), "alpha_draws": alpha_draws.round(5).tolist(),
               "phi_draws": a.phi_draws, "burnin": a.iters, "draw_spacing": a.draw_spacing, "lda_workers": a.lda_workers, "seed": a.seed},
              open(f"{out}/phi.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for k in range(K): print(f"  {names[k]:18s} alpha {alpha_draws.mean(0)[k]:6.3f}  " + " ".join(top[names[k]][:10]), flush=True)
    print(f"wrote {out}: {a.phi_draws} states of phi, alpha and the mean", flush=True)


def choose_phi(lda_dir, phi_spec, alpha_spec):
    """the phi and alpha of stage 2: phi_spec 'mean' or a draw index, alpha_spec 'lda' or a number -> names, H, phi, alpha, label"""
    z = np.load(os.path.join(lda_dir, "phi_draws.npz"))
    names = [str(n) for n in z["names"]]; H = int(z["seeded"])
    if phi_spec == "mean":
        phi = z["phi"].astype(np.float64).mean(0); alpha = z["alpha"].mean(0); label = f"mean of {len(z['phi'])} states"
    else:
        j = int(phi_spec); phi = z["phi"][j].astype(np.float64); alpha = z["alpha"][j]; label = f"state {j} of {len(z['phi'])}"
    phi /= phi.sum(1, keepdims=True)
    if alpha_spec != "lda":
        alpha = np.full(len(names), float(alpha_spec))
    return names, H, phi, alpha, label


# --- stage 2: the exact posterior of one document, NUTS on lda_given_phi.stan, in a pool of CmdStan processes ---
WORKER = {}


def worker_init(exe_file, logphi_path, chains, warmup, samples, keep, alpha, adapt_delta):
    """each worker holds the compiled program, the log phi matrix and the prior once"""
    import cmdstanpy, logging
    cmdstanpy.utils.get_logger().setLevel(logging.WARNING)                              # no line per chain in the log
    WORKER.update(model=cmdstanpy.CmdStanModel(exe_file=exe_file), logphi=np.load(logphi_path),
                  chains=chains, warmup=warmup, samples=samples, keep=keep, alpha=list(map(float, alpha)), adapt_delta=adapt_delta)


def fit_one(job):
    """(index, lemma indices, seed) -> (index, the kept theta draws (keep, K), Stan's diagnostics for the document)"""
    i, words, seed = job
    logphi = WORKER["logphi"]; K = logphi.shape[1]; t0 = time.time()
    tmp = tempfile.mkdtemp(prefix="soft-")                                             # CmdStan's csv files, removed at once
    try:
        fit = WORKER["model"].sample(data={"N": len(words), "K": K, "logphi": logphi[words].tolist(), "alpha": WORKER["alpha"]},
                                     chains=WORKER["chains"], parallel_chains=1, iter_warmup=WORKER["warmup"],
                                     iter_sampling=WORKER["samples"], seed=int(seed), output_dir=tmp, adapt_delta=WORKER["adapt_delta"],
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
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data"); ap.add_argument("--out", default="results/soft/run", help="stage 2's output directory")
    ap.add_argument("--stan", default="lda_given_phi.stan", help="the Stan program of one document's posterior")
    ap.add_argument("--lda", default="results/soft/lda", help="stage 1's directory (phi_draws.npz, phi_matrix.csv.gz, phi.json)")
    ap.add_argument("--refit-lda", action="store_true", help="fit the LDA again even if --lda holds one")
    ap.add_argument("--fit-lda-only", action="store_true", help="stop after stage 1")
    ap.add_argument("--free", type=int, default=4, help="unseeded topics beside the seeded topics")
    ap.add_argument("--seed-strength", type=float, default=200.0, help="prior weight of a seed word on its topic")
    ap.add_argument("--iters", type=int, default=10000, help="Gibbs sweeps of burn-in for the LDA (alpha has settled by then; 3 ms a sweep)")
    ap.add_argument("--phi-draws", type=int, default=20, help="states of the LDA's chain kept after burn-in")
    ap.add_argument("--draw-spacing", type=int, default=500, help="sweeps between kept states")
    ap.add_argument("--lda-workers", type=int, default=1, help="threads for tomotopy; 1 reproduces the fit from the seed")
    ap.add_argument("--phi", default="mean", help="the state of phi for stage 2: 'mean' or the index of a kept state")
    ap.add_argument("--alpha", default="lda", help="the Dirichlet prior on theta: 'lda' (the LDA's alpha) or a number for a symmetric prior")
    ap.add_argument("--chains", type=int, default=4); ap.add_argument("--warmup", type=int, default=500); ap.add_argument("--samples", type=int, default=500)
    ap.add_argument("--adapt-delta", type=float, default=0.99, help="NUTS's target acceptance; 0.8 leaves divergences under the LDA's sparse alpha, 0.99 none")
    ap.add_argument("--keep", type=int, default=100, help="posterior draws kept per document (thinned from chains x samples)")
    ap.add_argument("--subsample", type=int, default=2000, help="documents per set whose kept draws go to set_draws.npz")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4)), help="CmdStan processes side by side")
    ap.add_argument("--max-docs", type=int, default=0, help="documents read per set, sampled at random if more (0 = all); for dry runs")
    ap.add_argument("--no-draws", action="store_true", help="do not write theta_draws.npz")
    ap.add_argument("--seed", type=int, default=917)
    ap.add_argument("--chain-set", default="chains_judge235_t4", help="the walk set the report analyses (chain_set of imputations.parquet); '' = every untilted walk state")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    vocab, cluster, docs, tragic, keys = load(a.data, a.chain_set); V = len(vocab)
    if a.max_docs:
        pick = {name: sorted(rng.choice(len(ds), a.max_docs, replace=False)) if len(ds) > a.max_docs else range(len(ds)) for name, ds in docs.items()}
        docs = {name: [ds[i] for i in pick[name]] for name, ds in docs.items()}
        keys = [keys[i] for i in pick["imputations"]]
    lengths = {name: np.array([len(d) for d in ds]) for name, ds in docs.items()}
    print(f"vocabulary {V} lemmata; documents: " + "; ".join(f"{name} {len(ds)} (mean {lengths[name].mean():.1f} lemmata, max {lengths[name].max()})" for name, ds in docs.items())
          + f"; {len(tragic)} windows of Aeschylus and Euripides for the LDA", flush=True)

    # ---- stage 1
    if a.refit_lda or not os.path.exists(os.path.join(a.lda, "phi_draws.npz")):
        fit_lda(a, vocab, cluster, tragic, a.lda)
    if a.fit_lda_only:
        return
    names, H, phi, alpha, label = choose_phi(a.lda, a.phi, a.alpha)
    K = len(names); logphi = np.log(phi).T                                                # (V, K)
    assert phi.shape[1] == V, f"{a.lda} is from another vocabulary ({phi.shape[1]} lemmata, the dataset has {V})"
    prior = f"Dirichlet({float(a.alpha):g})" if a.alpha != "lda" else f"the LDA's alpha (seeded {alpha[:H].mean():.3f}, free {alpha[H:].mean():.2f}, sum {alpha.sum():.1f})"
    print(f"phi: {label} from {a.lda}; prior {prior}", flush=True)
    os.makedirs(a.out, exist_ok=True)

    # ---- the Stan program, compiled once
    import cmdstanpy, logging
    cmdstanpy.utils.get_logger().setLevel(logging.WARNING)
    model = cmdstanpy.CmdStanModel(stan_file=a.stan)
    print(f"CmdStan {cmdstanpy.cmdstan_version()} at {cmdstanpy.cmdstan_path()}; compiled {model.exe_file}", flush=True)

    # ---- every document of every set through NUTS
    scratch = tempfile.mkdtemp(prefix="soft-logphi-"); logphi_path = os.path.join(scratch, "logphi.npy"); np.save(logphi_path, logphi)
    draws, diagnostics = {}, []
    sets = [name for name in SETS if name in docs]
    print(f"sampling {sum(len(docs[n]) for n in sets)} documents with {a.workers} workers: {a.chains} chains x ({a.warmup} + {a.samples}) iterations each at adapt_delta {a.adapt_delta}, {a.keep} draws kept", flush=True)
    with Pool(a.workers, initializer=worker_init, initargs=(model.exe_file, logphi_path, a.chains, a.warmup, a.samples, a.keep, alpha, a.adapt_delta)) as pool:
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
    if not a.no_draws:
        np.savez_compressed(f"{a.out}/theta_draws.npz", names=np.array(names), **{name: draws[name].astype(np.float16) for name in sets})

    # ---- what soft_pool.py pools: the draws of every set's mean profile, every document's posterior mean, a subsample of
    # documents with all their draws, and the chain and step of every imputation (for resampling by chain)
    pooled = {"names": np.array(names), "alpha": alpha, "label": np.array(label), "seeded": H,
              "imputation_chain": np.array([f"{k[0]}/{k[1]}" for k in keys]), "imputation_step": np.array([k[2] for k in keys])}
    for name in sets:
        P3 = draws[name]; sub = np.sort(rng.choice(len(P3), min(len(P3), a.subsample), replace=False))
        pooled[f"mean_{name}"] = P3.mean(0); pooled[f"doc_mean_{name}"] = P3.mean(1).astype(np.float16)
        pooled[f"sub_{name}"] = P3[sub].astype(np.float16); pooled[f"sub_index_{name}"] = sub; pooled[f"n_lemmata_{name}"] = lengths[name]
    np.savez_compressed(f"{a.out}/set_draws.npz", **pooled)

    # ---- two documents with their data and posterior summary, refitted with rstan in the report
    check = []
    for name in ("imputations", "sophocles"):
        i = int(rng.integers(len(docs[name]))); P = draws[name][i].astype(np.float64); words = docs[name][i]
        check.append({"source": name, "index": i, "lemma_idx": [int(w) for w in words], "logphi": logphi[words].round(5).tolist(),
                      "mean": P.mean(0).round(5).tolist(), "sd": P.std(0).round(5).tolist(),
                      "q05": np.quantile(P, 0.05, 0).round(5).tolist(), "q95": np.quantile(P, 0.95, 0).round(5).tolist()})
    json.dump({"names": names, "phi": label, "alpha": alpha.round(5).tolist(), "draws": a.keep, "documents": check}, open(f"{a.out}/check_documents.json", "w"), indent=None)

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
    print(f"{H} seeded and {K - H} free topics on {V} lemmata; phi {label}; {len(diagnostics)} documents sampled by NUTS; "
          f"free topics hold {profiles['imputations']['free_mass']:.1%} of the imputations' mass", flush=True)
    for name, s in profiles.items():
        if "lift" in s:
            top3 = sorted(range(H), key=lambda k: -s["lift"][k])[:3]
            print(f"  {name}: raised {raised[name] or 'none'}; largest lifts " + ", ".join(f"{names[k]} {s['lift'][k]:.2f}" for k in top3), flush=True)
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
