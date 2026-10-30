#!/usr/bin/env python3
"""The soft model along the walk: every document's posterior mean profile at
every --every-th recorded state of every chain, so that the report can follow
a topic's lift over the steps of the walk.

Stage 2 of scripts/soft_model.py on a thinned set of the imputations, with its
own loading, its choice of phi and alpha and its pool of CmdStan processes:
the mean phi of the twenty states and the LDA's mean alpha by default, the
plug-in setting of results/soft/mean_alpha_lda, whose Sophocles profile is the
baseline the report divides by.  Writes trajectory.csv: one row per document
(chain set, chain, step, number of lemmata), the posterior mean of every
topic's share, and Stan's diagnostics for the fit.

  python scripts/soft_trajectory.py --data data --lda results/soft/lda --out results/soft/trajectory
  sbatch slurm/soft_trajectory.slurm
"""
import argparse, os, shutil, tempfile, time
from multiprocessing import Pool

import numpy as np

from soft_model import load, choose_phi, worker_init, fit_one


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data"); ap.add_argument("--out", default="results/soft/trajectory")
    ap.add_argument("--stan", default="lda_given_phi.stan", help="the Stan program of one document's posterior")
    ap.add_argument("--lda", default="results/soft/lda", help="stage 1's directory (phi_draws.npz)")
    ap.add_argument("--phi", default="mean", help="the state of phi: 'mean' or the index of a kept state")
    ap.add_argument("--alpha", default="lda", help="the Dirichlet prior on theta: 'lda' or a number for a symmetric prior")
    ap.add_argument("--chain-set", default="chains_judge235_t4_spread,chains_judge235_t4_spread2,chains_judge235_t4_topic4", help="the walk sets, comma-separated (chain_set of imputations.parquet)")
    ap.add_argument("--every", type=int, default=5, help="every n-th recorded state of each chain (a state is recorded every 10 steps)")
    ap.add_argument("--chains", type=int, default=4); ap.add_argument("--warmup", type=int, default=500); ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--adapt-delta", type=float, default=0.99, help="as in soft_model.py: the LDA's alpha is below 1 on the seeded topics")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4)), help="CmdStan processes side by side")
    ap.add_argument("--seed", type=int, default=917)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

    vocab, cluster, docs, tragic, keys = load(a.data, a.chain_set)
    by_chain = {}
    for i, (chain_set, chain, step) in enumerate(keys):
        by_chain.setdefault((chain_set, chain), []).append((step, i))
    picked = [i for chain in sorted(by_chain) for _, i in sorted(by_chain[chain])[::a.every]]
    names, H, phi, alpha, label = choose_phi(a.lda, a.phi, a.alpha)
    K = len(names); logphi = np.log(phi).T                                                # (V, K)
    assert phi.shape[1] == len(vocab), f"{a.lda} is from another vocabulary"
    print(f"{len(picked)} of {len(keys)} imputations (every {a.every}th state of {len(by_chain)} chains); phi: {label}; alpha: {a.alpha}", flush=True)

    import cmdstanpy, logging
    cmdstanpy.utils.get_logger().setLevel(logging.WARNING)
    model = cmdstanpy.CmdStanModel(stan_file=a.stan)
    scratch = tempfile.mkdtemp(prefix="soft-logphi-"); logphi_path = os.path.join(scratch, "logphi.npy"); np.save(logphi_path, logphi)
    keep = a.chains * a.samples                                                            # every draw: the mean is all we keep
    jobs = [(i, docs["imputations"][i], a.seed * 1000 + i) for i in picked]
    rows, t0 = {}, time.time()
    with Pool(a.workers, initializer=worker_init, initargs=(model.exe_file, logphi_path, a.chains, a.warmup, a.samples, keep, alpha, a.adapt_delta)) as pool:
        for i, draws, d in pool.imap_unordered(fit_one, jobs, chunksize=2):
            rows[i] = (draws.mean(0), d)
            if len(rows) % 500 == 0 or len(rows) == len(jobs):
                print(f"  {len(rows)}/{len(jobs)} documents in {time.time() - t0:.0f} s", flush=True)
    shutil.rmtree(scratch, ignore_errors=True)

    with open(os.path.join(a.out, "trajectory.csv"), "w", encoding="utf-8") as f:
        f.write(",".join(["chain_set", "chain", "step", "n_lemmata"] + names + ["max_rhat", "min_ess_bulk", "divergences", "treedepth_hits"]) + "\n")
        for i in picked:
            mean, d = rows[i]; chain_set, chain, step = keys[i]
            f.write(",".join([chain_set, str(chain), str(step), str(len(docs["imputations"][i]))] + [f"{m:.6f}" for m in mean]
                             + [f"{d['max_rhat']:.4f}", f"{d['min_ess_bulk']:.1f}", str(d["divergences"]), str(d["treedepth_hits"])]) + "\n")
    print(f"wrote {a.out}/trajectory.csv: {len(picked)} documents, R-hat > 1.01 in {sum(rows[i][1]['max_rhat'] > 1.01 for i in picked)}, "
          f"divergences in {sum(rows[i][1]['divergences'] > 0 for i in picked)}", flush=True)


if __name__ == "__main__":
    main()
