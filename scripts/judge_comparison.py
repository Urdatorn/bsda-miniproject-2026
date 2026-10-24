"""Three judges side by side: the lift of every topic under each judge-steered walk set.

The dataset holds the untilted states of three judge-steered sets, one per
judge: Qwen3-235B-A22B-Instruct-2507 (the set the reports read),
Qwen2.5-72B-Instruct, and the Claude Sonnet 5.5 pilot of 2026-10-23 (eight
chains capped at 3.10 USD of judge calls each, about 1,500 steps, states
recorded by distance travelled).  For each set this script computes exactly
what topic_circles.py computes for the report's set: the pooled rate per
thousand topic lemmata of every topic, its lift against Sophocles, against
Aeschylus and Euripides and against Classical Attic, the 90 % interval from
resampling the set by chain and the baseline by window, and p_document.  One
table, results/hard/judge_comparison.csv, one row per set, topic and baseline.

  python scripts/judge_comparison.py --data data --out results/hard
"""
import argparse, json, os, sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from topic_circles import ATTIC, bootstrap_by_group, bootstrap_by_row, compare, imputation_documents  # noqa: E402

SETS = {"chains_judge235_t4": "Qwen3-235B", "chains_judge72_t4_v3e": "Qwen2.5-72B",
        "chains_judge_claude-sonnet-5-5_t4": "Claude Sonnet 5.5"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results/hard", help="where the table goes")
    ap.add_argument("--baselines", default=None, help="baseline_windows.parquet (default: in --data)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    vocab = json.load(open(os.path.join(a.data, "vocab.json"), encoding="utf-8"))
    topics = sorted({r["cluster_K40"] for r in vocab if r["cluster_K40"]})
    topic_id = {t: k for k, t in enumerate(topics)}
    topic_of = {r["index"]: topic_id[r["cluster_K40"]] for r in vocab if r["cluster_K40"]}

    windows = pq.read_table(a.baselines or os.path.join(a.data, "baseline_windows.parquet")).to_pandas()
    assert list(windows.columns[5:]) == topics, "topic columns of the windows differ from vocab.json"
    base = {"sophocles": windows.loc[windows.author == "sophocles", topics].to_numpy(dtype=np.int32),
            "tragedians": windows.loc[windows.author.isin(["aeschylus", "euripides"]), topics].to_numpy(dtype=np.int32),
            "attic": windows.loc[(windows.source == "oga") & windows.author.isin(ATTIC), topics].to_numpy(dtype=np.int32)}
    base_boot = {cell: bootstrap_by_row(base[cell]) for cell in base}
    print("baseline bootstraps done", flush=True)

    rows = []
    for chain_set, judge in SETS.items():
        counts, chain_of = imputation_documents(a.data, topic_of, len(topics), chain_set)
        boot = bootstrap_by_group(counts, chain_of)
        print(f"{judge} ({chain_set}): {len(counts)} documents in {len(set(chain_of))} chains, {counts.sum()} topic lemmata", flush=True)
        for cell in base:
            table = compare(f"imputations vs {cell}", counts, base[cell], boot, base_boot[cell], topics)
            table.insert(0, "judge", judge); table.insert(1, "chain_set", chain_set); table.insert(2, "documents", len(counts))
            rows.append(table)
    out = pd.concat(rows)
    out.to_csv(os.path.join(a.out, "judge_comparison.csv"), index=False, float_format="%.4f")
    soph = out[out.comparison == "imputations vs sophocles"].pivot(index="topic", columns="judge", values="lift")
    print("lift against Sophocles, the five largest under each judge:")
    for judge in SETS.values():
        print(f"  {judge}: " + ", ".join(f"{t} {v:.2f}" for t, v in soph[judge].sort_values(ascending=False).head(5).items()))


if __name__ == "__main__":
    main()
