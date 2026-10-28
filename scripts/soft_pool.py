#!/usr/bin/env python3
"""Pool the soft model's stage-2 runs over the states of the LDA's Gibbs chain.

soft_model.py --phi j fits every document under the j-th kept state (phi_j,
alpha_j) of the LDA and writes set_draws.npz; one such run per state
(slurm/soft_draws.slurm) gives D runs.  This script pools them into one
results directory with the same files the report reads from a single run, so
that the report can be knitted against it unchanged (params$results):

  profiles.csv      the mean profile of every set, its 90 % interval and the
                    imputations' lift against Sophocles, over all D x keep
                    draws of the mean profile (the posterior of theta given
                    phi_j, over the D states of phi: the uncertainty in phi
                    enters here); p_dominant and p_document averaged over the
                    states
  comparisons.csv   the lifts behind the circle grids with their 90 % interval
                    over the pooled draws (focus and baseline paired within a
                    state), plus lift_phi_min and lift_phi_max, the lift's
                    point estimate at the single states (how much phi alone
                    moves it), and for the imputations against each baseline
                    lift_chain_q05, lift_chain_q95 and p_chain_above from a
                    bootstrap that resamples the imputations by walk chain and
                    the baseline by window, with a state of phi drawn at random
                    per replicate and the posterior noise of the two mean
                    profiles added back (a random draw's deviation from the
                    mean of its state): the dependence between the snapshots of
                    one chain, which the posterior intervals ignore, on top of
                    what the posterior intervals carry
  lifts_by_state.csv  the lift of every comparison at every single state, with that state's own 90 % interval
  diagnostics.csv.gz  one row per document with the worst of Stan's
                    diagnostics over the states (largest R-hat, smallest bulk
                    ESS, divergences summed)
  diagnostics_by_draw.csv  the counts per state and set
  phi.json, phi_matrix.csv.gz  the mean phi and alpha, copied from --lda
  check_documents.json  copied from the first state's run (it carries its own
                    log phi and alpha, so the report's refit reproduces it)
  pooled.json       what was pooled

  python scripts/soft_pool.py --draws results/soft/draws --lda results/soft/lda --out results/soft/pooled
"""
import argparse, glob, gzip, json, os, shutil
import numpy as np, pandas as pd

BASELINES = ("sophocles", "tragedians", "attic")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--draws", default="results/soft/draws", help="the directory of the per-state runs (d00, d01, ...)")
    ap.add_argument("--lda", default="results/soft/lda", help="stage 1's directory")
    ap.add_argument("--out", default="results/soft/pooled")
    ap.add_argument("--bootstrap", type=int, default=2000, help="replicates of the chain bootstrap")
    ap.add_argument("--seed", type=int, default=917)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); rng = np.random.default_rng(a.seed)
    runs = sorted(d for d in glob.glob(os.path.join(a.draws, "d[0-9]*")) if os.path.exists(os.path.join(d, "set_draws.npz")))
    assert runs, f"no finished runs under {a.draws}"
    draws = [np.load(os.path.join(d, "set_draws.npz")) for d in runs]
    names = [str(n) for n in draws[0]["names"]]; K = len(names); H = int(draws[0]["seeded"]); D = len(runs)
    for z in draws: assert [str(n) for n in z["names"]] == names, "the runs have different topics"
    sets = [s for s in ("sophocles", "imputations", "tragedians", "attic") if f"mean_{s}" in draws[0].files]
    keep = draws[0]["mean_sophocles"].shape[0]
    print(f"{D} states of phi, {keep} draws of each mean profile per state; sets {', '.join(sets)}; {H} seeded + {K - H} free topics", flush=True)

    # ---- the pooled draws of every set's mean profile: (D * keep, K), state index alongside
    mean_draws = {s: np.concatenate([z[f"mean_{s}"].astype(np.float64) for z in draws]) for s in sets}
    state_of_draw = np.repeat(np.arange(D), keep)
    n_docs = {s: int(len(draws[0][f"doc_mean_{s}"])) for s in sets}
    n_lemmata = {s: float(draws[0][f"n_lemmata_{s}"].mean()) for s in sets}
    per_state = [pd.read_csv(os.path.join(d, "profiles.csv"), encoding="utf-8") for d in runs]
    p_dominant = {s: np.mean([t.loc[t.source == s, "p_dominant"].to_numpy() for t in per_state], axis=0) for s in sets}

    def p_document(focus, base):
        """P(theta_k^doc > theta_k^window), a random document of focus against a random one of base, within a state, averaged over states"""
        out = []
        for z in draws:
            P = z[f"sub_{focus}"].astype(np.float64).reshape(-1, K); Q = z[f"sub_{base}"].astype(np.float64).reshape(-1, K)
            out.append((P > Q[rng.integers(0, len(Q), len(P))]).mean(0))
        return np.mean(out, axis=0)

    soph = mean_draws["sophocles"]
    with open(f"{a.out}/profiles.csv", "w", encoding="utf-8") as f:
        f.write("source,topic,n_docs,n_lemmata,mean,q05,q95,p_dominant,p_above_sophocles,lift,p_document\n")
        for s in sets:
            M = mean_draws[s]; mean = M.mean(0); q05 = np.quantile(M, 0.05, 0); q95 = np.quantile(M, 0.95, 0)
            if s != "sophocles":
                p_above = (M > soph).mean(0); lift = mean / soph.mean(0); p_doc = p_document(s, "sophocles")
            for j in range(H):
                f.write(f"{s},{names[j]},{n_docs[s]},{n_lemmata[s]:.1f},{mean[j]:.4f},{q05[j]:.4f},{q95[j]:.4f},{p_dominant[s][j]:.4f},"
                        + (f"{p_above[j]:.4f},{lift[j]:.3f},{p_doc[j]:.4f}" if s != "sophocles" else ",,") + "\n")

    # ---- the chain bootstrap: the imputations resampled by walk chain, the baseline by window, phi by state
    chains = np.array([str(c) for c in draws[0]["imputation_chain"]]); chain_ids = np.unique(chains)
    members = {c: np.flatnonzero(chains == c) for c in chain_ids}
    doc_means = {s: np.stack([z[f"doc_mean_{s}"].astype(np.float64) for z in draws]) for s in sets}      # (D, n_docs, K)
    print(f"chain bootstrap: {len(chain_ids)} chains of the imputations, {a.bootstrap} replicates", flush=True)

    # the posterior noise of a set's mean profile at a state: a draw of the mean profile minus the mean of the
    # documents' posterior means; the bootstrap adds it back to its resampled means (the resampled set has as
    # many chains or windows as the real one, so the noise has about the right size)
    noise = {s: np.stack([z[f"mean_{s}"].astype(np.float64) - doc_means[s][i].mean(0) for i, z in enumerate(draws)]) for s in sets}  # (D, keep, K)

    def chain_bootstrap(base):
        ratios = np.empty((a.bootstrap, H))
        for b in range(a.bootstrap):
            d = rng.integers(D); i = rng.integers(keep)
            picked = np.concatenate([members[c] for c in rng.choice(chain_ids, len(chain_ids), replace=True)])
            imp = doc_means["imputations"][d][picked].mean(0) + noise["imputations"][d, i]
            windows = doc_means[base][d][rng.integers(0, n_docs[base], n_docs[base])].mean(0) + noise[base][d, i]
            ratios[b] = imp[:H] / windows[:H]
        return ratios

    # ---- the comparisons, paired within a state; lift at the single states; the chain bootstrap for the imputations
    rows = []; by_state = []
    for focus, base in (("imputations", "sophocles"), ("imputations", "tragedians"), ("imputations", "attic"), ("sophocles", "tragedians"), ("sophocles", "attic")):
        if focus not in sets or base not in sets: continue
        F, B = mean_draws[focus][:, :H], mean_draws[base][:, :H]; ratio = F / B
        per_state_lift = np.stack([F[state_of_draw == d].mean(0) / B[state_of_draw == d].mean(0) for d in range(D)])
        per_state_q05 = np.stack([np.quantile(ratio[state_of_draw == d], 0.05, axis=0) for d in range(D)])   # the single state's own 90 % interval
        per_state_q95 = np.stack([np.quantile(ratio[state_of_draw == d], 0.95, axis=0) for d in range(D)])
        p_doc = p_document(focus, base)
        boot = chain_bootstrap(base) if focus == "imputations" else None
        by_state += [{"topic": names[k], "comparison": f"{focus} vs {base}", "state": d, "lift": per_state_lift[d, k],
                      "lift_q05": per_state_q05[d, k], "lift_q95": per_state_q95[d, k]} for d in range(D) for k in range(H)]
        for k in range(H):
            rows.append({"topic": names[k], "comparison": f"{focus} vs {base}", "rate_focus": F.mean(0)[k], "rate_baseline": B.mean(0)[k],
                         "lift": F.mean(0)[k] / B.mean(0)[k], "lift_q05": np.quantile(ratio[:, k], 0.05), "lift_q95": np.quantile(ratio[:, k], 0.95),
                         "p_above": (F[:, k] > B[:, k]).mean(), "p_document": p_doc[k],
                         "lift_phi_min": per_state_lift[:, k].min(), "lift_phi_max": per_state_lift[:, k].max(),
                         "lift_chain_q05": np.quantile(boot[:, k], 0.05) if boot is not None else np.nan,
                         "lift_chain_q95": np.quantile(boot[:, k], 0.95) if boot is not None else np.nan,
                         "p_chain_above": (boot[:, k] > 1).mean() if boot is not None else np.nan})
    comparisons = pd.DataFrame(rows)
    comparisons.to_csv(f"{a.out}/comparisons.csv", index=False, float_format="%.4f")
    pd.DataFrame(by_state).to_csv(f"{a.out}/lifts_by_state.csv", index=False, float_format="%.4f")
    for name, t in comparisons.groupby("comparison", sort=False):
        off = t[(t.lift_q05 > 1) | (t.lift_q95 < 1)].sort_values("lift", ascending=False)
        line = f"  {name}: {len(off)} topics with a posterior interval off 1"
        if name.startswith("imputations"):
            off_chain = t[(t.lift_chain_q05 > 1) | (t.lift_chain_q95 < 1)]
            line += f", {len(off_chain)} with the chain-bootstrap interval off 1"
        print(line + "; largest " + ", ".join(f"{r.topic} {r.lift:.2f} [{r.lift_q05:.2f}, {r.lift_q95:.2f}]" for r in off.head(3).itertuples()), flush=True)

    # ---- diagnostics: the worst over the states for every document, and the counts per state
    diag = pd.concat([pd.read_csv(os.path.join(d, "diagnostics.csv.gz")).assign(state=i) for i, d in enumerate(runs)])
    worst = diag.groupby(["source", "document"], sort=False).agg(n_lemmata=("n_lemmata", "first"), max_rhat=("max_rhat", "max"),
                                                                  min_ess_bulk=("min_ess_bulk", "min"), divergences=("divergences", "sum"),
                                                                  treedepth_hits=("treedepth_hits", "sum"), seconds=("seconds", "mean")).reset_index()
    with gzip.open(f"{a.out}/diagnostics.csv.gz", "wt") as f:
        worst.to_csv(f, index=False, float_format="%.4f")
    by_state_diag = diag.groupby(["state", "source"]).agg(documents=("document", "size"), rhat_over_1_01=("max_rhat", lambda x: int((x > 1.01).sum())),
                                                     ess_under_400=("min_ess_bulk", lambda x: int((x < 400).sum())),
                                                     divergent=("divergences", lambda x: int((x > 0).sum())),
                                                     treedepth=("treedepth_hits", lambda x: int((x > 0).sum())), worst_rhat=("max_rhat", "max"),
                                                     least_ess=("min_ess_bulk", "min"), seconds=("seconds", "median")).reset_index()
    by_state_diag.to_csv(f"{a.out}/diagnostics_by_draw.csv", index=False, float_format="%.4f")
    print(f"  diagnostics over {D} states: R-hat > 1.01 in {(worst.max_rhat > 1.01).sum()} of {len(worst)} documents at their worst state, "
          f"bulk ESS < 400 in {(worst.min_ess_bulk < 400).sum()}, divergences in {(worst.divergences > 0).sum()}; "
          f"per state (median): R-hat > 1.01 in {by_state_diag.groupby('state').rhat_over_1_01.sum().median():.0f}, "
          f"ESS < 400 in {by_state_diag.groupby('state').ess_under_400.sum().median():.0f}, divergent in {by_state_diag.groupby('state').divergent.sum().median():.0f}, "
          f"tree depth hit in {by_state_diag.groupby('state').treedepth.sum().median():.0f}", flush=True)

    # ---- phi, alpha and the refit documents
    for name in ("phi.json", "phi_matrix.csv.gz"): shutil.copy(os.path.join(a.lda, name), a.out)
    shutil.copy(os.path.join(runs[0], "check_documents.json"), a.out)
    json.dump({"states": D, "runs": [os.path.basename(d) for d in runs], "keep_per_state": int(keep), "bootstrap": a.bootstrap,
               "chains": int(len(chain_ids)), "alpha_mean": np.mean([z["alpha"] for z in draws], axis=0).round(5).tolist()},
              open(f"{a.out}/pooled.json", "w"), indent=1)
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
