"""One topic, every lemma: who uses which words, and which words carry the lift.

For one topic (default wrong-vengeance-err, the walk's clearest lift that
survives every baseline) the figure lists all its lemmata, ordered by the
walk's use, with each lemma's rate per thousand topic lemmata in the walk,
in Sophocles, in Aeschylus and Euripides and in Classical Attic (the OGA
works of the Attic authors before Alexander, ATTIC in
topic_circles.py), on a log scale; the line from Sophocles' rate to the walk's is red
where the walk uses the word more and blue where less.  The right-hand
panel is each lemma's contribution to the walk's excess over Sophocles on
the topic (walk rate minus Sophocles rate, in lemmata per thousand), so
that the lift of the topic is read as a list of words.  The counts come
from imputations.parquet (the walk) and baseline_lemma_counts.parquet
(scripts/window_baselines.py).

  python scripts/topic_lemmata.py --data data --out results/hard --figures figures --topic wrong-vengeance-err

writes topic_lemmata_<topic>.csv there and hard_topic_lemmata_<topic>.png to the
figures folder.
"""
import argparse, json, os
from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from topic_circles import ATTIC

CELLS = {"walk": "the walk", "sophocles": "Sophocles", "tragedians": "Aeschylus and Euripides", "attic": "Classical Attic"}
COLOURS = {"walk": "#4e79a7", "sophocles": "#59a14f", "tragedians": "#f28e2b", "attic": "#7f7f7f"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results/hard", help="where the table goes")
    ap.add_argument("--figures", default="figures", help="where the figure goes")
    ap.add_argument("--topic", default="wrong-vengeance-err")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); os.makedirs(a.figures, exist_ok=True)

    vocab = json.load(open(os.path.join(a.data, "vocab.json"), encoding="utf-8"))
    topic_of = {r["index"]: r["cluster_K40"] for r in vocab if r["cluster_K40"]}
    lemma_of = {r["index"]: r["lemma"] for r in vocab}
    members = sorted(i for i, t in topic_of.items() if t == a.topic)
    if not members:
        raise SystemExit(f"no topic {a.topic}")

    # counts over the whole vocabulary per cell; the rate is per thousand topic-bearing lemmata of the cell
    counts = {cell: np.zeros(len(vocab), dtype=np.int64) for cell in CELLS}
    imp = pq.read_table(os.path.join(a.data, "imputations.parquet"), columns=["restorer", "style_tilted", "lemma_idx", "count"])
    restorer = np.array(imp["restorer"].to_pylist()); tilted = np.array([bool(t) for t in imp["style_tilted"].to_pylist()])
    walk = imp.filter((restorer == "walk") & ~tilted)
    for idx, cnt in zip(walk["lemma_idx"].to_pylist(), walk["count"].to_pylist()):
        for i, c in zip(idx, cnt):
            counts["walk"][int(i)] += int(c)
    base = pq.read_table(os.path.join(a.data, "baseline_lemma_counts.parquet")).to_pandas()
    for cell, rows in (("sophocles", base[base.author == "sophocles"]),
                       ("tragedians", base[base.author.isin(["aeschylus", "euripides"])]),
                       ("attic", base[(base.source == "oga") & base.author.isin(ATTIC)])):
        np.add.at(counts[cell], rows.lemma_idx.to_numpy(), rows["count"].to_numpy())
    topic_bearing = np.array(sorted(topic_of))
    rate = {cell: counts[cell] / counts[cell][topic_bearing].sum() * 1000 for cell in CELLS}

    table = pd.DataFrame({"lemma": [lemma_of[i] for i in members], "lemma_idx": members})
    for cell in CELLS:
        table[f"rate_{cell}"] = rate[cell][members]
        table[f"count_{cell}"] = counts[cell][members]
    table["excess_over_sophocles"] = table.rate_walk - table.rate_sophocles
    table = table.sort_values("rate_walk", ascending=False).reset_index(drop=True)
    table.to_csv(os.path.join(a.out, f"topic_lemmata_{a.topic}.csv"), index=False, float_format="%.4f")
    total = {cell: table[f"rate_{cell}"].sum() for cell in CELLS}
    print(f"{a.topic}: {len(table)} lemmata; topic rate per thousand: " + ", ".join(f"{CELLS[c]} {total[c]:.1f}" for c in CELLS), flush=True)
    excess = table.sort_values("excess_over_sophocles", ascending=False)
    cum = excess.excess_over_sophocles.clip(lower=0).cumsum() / excess.excess_over_sophocles.clip(lower=0).sum()
    print("lemmata carrying the first three quarters of the positive excess over Sophocles: "
          + ", ".join(f"{r.lemma} {r.excess_over_sophocles:+.2f}" for r in excess[cum <= 0.75].itertuples()), flush=True)

    import matplotlib.pyplot as plt
    from figure_style import apply, TEXT_WIDTH
    apply()
    n = len(table)
    half = int(np.ceil(n / 2))
    fig = plt.figure(figsize=(TEXT_WIDTH, 0.112 * half + 1.0), layout="constrained")
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.9])
    floor = 0.02                                             # a rate of zero is drawn at the floor
    for part, ax in enumerate((fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]))):
        rows = table.iloc[part * half:(part + 1) * half]
        y = np.arange(len(rows))[::-1]
        for k, (_, r) in enumerate(rows.iterrows()):
            up = r.rate_walk >= r.rate_sophocles
            ax.plot([max(r.rate_sophocles, floor), max(r.rate_walk, floor)], [y[k], y[k]],
                    color="#d62728" if up else "#1f77b4", lw=1.0, alpha=0.75, zorder=1)
        for cell, marker, size in (("attic", "x", 9), ("tragedians", "D", 6), ("sophocles", "o", 11), ("walk", "o", 11)):
            ax.scatter(np.maximum(rows[f"rate_{cell}"], floor), y, s=size, marker=marker, color=COLOURS[cell],
                       linewidth=0.5, label=CELLS[cell], zorder=3, **({"edgecolor": "white"} if marker == "o" else {}))
        ax.set_yticks(y); ax.set_yticklabels(rows.lemma, fontsize=6)
        ax.set_xscale("log"); ax.set_xlim(floor * 0.8, max(table.rate_walk.max(), table.rate_sophocles.max(), table.rate_attic.max()) * 1.6)
        ax.set_xlabel("per thousand topic lemmata (log; zero at the left edge)", fontsize=6.5)
        ax.tick_params(axis="x", labelsize=6); ax.grid(axis="x", lw=0.3, alpha=0.5)
        ax.set_ylim(-1, len(rows))
        if part == 0:
            ax.legend(fontsize=6, loc="lower right", framealpha=0.9)
    ax = fig.add_subplot(grid[0, 2])
    shown = excess.head(30)[::-1]
    ax.barh(np.arange(len(shown)), shown.excess_over_sophocles,
            color=np.where(shown.excess_over_sophocles >= 0, "#d62728", "#1f77b4"))
    ax.set_yticks(np.arange(len(shown))); ax.set_yticklabels(shown.lemma, fontsize=6)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("walk minus Sophocles (per thousand)", fontsize=6.5)
    ax.set_title("thirty largest contributions to the walk's excess", fontsize=7)
    ax.tick_params(axis="x", labelsize=6); ax.grid(axis="x", lw=0.3, alpha=0.5)
    fig.suptitle(f"The lemmata of {a.topic}: {CELLS['walk']} {total['walk']:.1f}, Sophocles {total['sophocles']:.1f}, "
                 f"Aeschylus and Euripides {total['tragedians']:.1f}, Classical Attic {total['attic']:.1f} per thousand topic lemmata",
                 fontsize=8)
    fig.savefig(os.path.join(a.figures, f"hard_topic_lemmata_{a.topic}.png"), bbox_inches="tight")
    print(f"wrote {a.figures}/hard_topic_lemmata_{a.topic}.png and {a.out}/topic_lemmata_{a.topic}.csv", flush=True)


if __name__ == "__main__":
    main()
