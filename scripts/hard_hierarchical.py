"""The hard lift with the chain as unit: a hierarchical Dirichlet-multinomial
model (hard_hierarchical.stan) that gives the lift of the imputations over
Sophocles a posterior of its own, as a check on the chain-resampled intervals
of the hard report.

Each of the 28 chains (the three rounds of starts) is one document, its topic
counts summed over its kept states; each window of seventeen Sophoclean
trimeters is another, cut as the hard report cuts them (444 windows, the last
one short). Chains and windows are Dirichlet-multinomial around their
corpus's profile mu, with one concentration kappa per corpus; the lift is
mu_imp / mu_soph. One kappa for all topics understates the spread of bursty
topics such as fire-flame-flash, whose intervals here are too narrow.

Writes results/hard/hierarchical_lift.csv: topic, lift (posterior mean),
q05, q95, p_above (posterior probability that the lift is above 1).

Usage: python scripts/hard_hierarchical.py --data data --out results/hard
"""
import argparse, json, os

import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel

SETS = ["chains_judge235_t4_spread", "chains_judge235_t4_spread2", "chains_judge235_t4_topic4"]
STAN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hard_hierarchical.stan")


def topic_counts(table, group, topic_of, column):
    """summed topic counts of each group of rows"""
    out = {}
    for g, idx, cnt in zip(group, table.lemma_idx, table["count"]):
        row = out.setdefault(g, np.zeros(len(column), int))
        for i, c in zip(idx, cnt):
            t = topic_of[int(i)]
            if t: row[column[t]] += int(c)
    return np.array(list(out.values()))


def main(data, out):
    vocab = json.load(open(os.path.join(data, "vocab.json"), encoding="utf-8"))
    topic_of = {v["index"]: v["cluster_K40"] for v in vocab}
    topics = sorted({t for t in topic_of.values() if t})
    column = {t: k for k, t in enumerate(topics)}

    imp = pd.read_parquet(os.path.join(data, "imputations.parquet"),
                          columns=["restorer", "chain_set", "chain", "style_tilted", "lemma_idx", "count"])
    imp = imp[(imp.restorer == "walk") & imp.chain_set.isin(SETS) & ~imp.style_tilted.fillna(False).astype(bool)]
    n_chain = topic_counts(imp, list(zip(imp.chain_set, imp.chain)), topic_of, column)
    corpus = pd.read_parquet(os.path.join(data, "corpus.parquet"))
    sophocles = corpus[corpus.source == "sophocles"].sort_values("line")
    n_window = topic_counts(sophocles, np.arange(len(sophocles)) // 17, topic_of, column)

    # start at the observed profiles: from a random start kappa can wander where the sampler stalls
    fit = CmdStanModel(stan_file=STAN).sample(
        data=dict(K=len(topics), C=len(n_chain), W=len(n_window), n_chain=n_chain.tolist(), n_window=n_window.tolist()),
        chains=4, iter_warmup=1000, iter_sampling=1000, seed=1, show_progress=False,
        inits=dict(mu_imp=((n_chain.sum(0) + 1) / (n_chain.sum() + len(topics))).tolist(),
                   mu_soph=((n_window.sum(0) + 1) / (n_window.sum() + len(topics))).tolist(),
                   log_kappa_imp=5.0, log_kappa_soph=3.0))
    print(fit.diagnose())
    lift = fit.stan_variable("lift")
    os.makedirs(out, exist_ok=True)
    pd.DataFrame({"topic": topics, "lift": lift.mean(0), "q05": np.quantile(lift, 0.05, 0),
                  "q95": np.quantile(lift, 0.95, 0), "p_above": (lift > 1).mean(0)}).round(4).to_csv(
        os.path.join(out, "hierarchical_lift.csv"), index=False)
    print(f"{len(n_chain)} chains, {len(n_window)} windows -> {out}/hierarchical_lift.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="results/hard")
    main(**vars(parser.parse_args()))
