"""Topic circle maps: is a set of documents' use of each topic unusual, and against whom?

One circle per topic in a grid ordered by the focus cell's share of the
topic, largest first.  A circle's area is that share, its colour the log2
lift of the share against a baseline, and it is drawn in full colour with a
black edge when the lift's 90 % interval excludes 1 (otherwise faded, the
lift in brackets).  Five figures:

  hard_circles_imputations_vs_{sophocles,tragedians,attic}.png   the imputations against
      Sophocles, against Aeschylus and Euripides, against Classical Attic;
  hard_circles_sophocles_vs_{tragedians,attic}.png   Sophocles against the
      other two: his own topics.
One text-width figure (ICML) per comparison, figure_style.py.

Shares are rates per thousand topic lemmata, pooled over the cell's
documents.  The baselines are windows of 28 vocabulary lemmata
(baseline_windows.parquet, scripts/window_baselines.py): Sophocles' and the
other tragedians' extant trimeters, and Classical Attic, the OGA works of
the Attic authors before Alexander (ATTIC below: the three tragedians with
their lyrics, Aristophanes, Thucydides, Xenophon, Plato, Aristotle, the
Attic orators and the sophists writing Attic).  The
interval of a lift comes from two bootstraps: the imputations resampled by chain
(eight chains), a baseline resampled by window.  p_document is the chance that
a random document of the focus set has a larger share than a random window
of the baseline.  Everything is written to topic_circles.csv.

  python scripts/topic_circles.py --data data --out results/hard --figures figures
"""
import argparse, json, os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

RNG = np.random.default_rng(917)
BOOT = 2000
# Classical Attic: Attic-dialect authors before Alexander, as OGA names them (Ionic and Doric authors, and
# everything Hellenistic or later, are out: no Herodotus, Hippocrates, Pindar, Theocritus, Menander, Theophrastus)
ATTIC = ["Aeschylus", "Sophocles", "Euripides", "Aristophanes", "Thucydides", "Xenophon", "Plato", "Aristotle",
         "Antiphon", "Andocides", "Lysias", "Isocrates", "Isaeus", "Demosthenes", "Aeschines", "Lycurgus", "Hyperides",
         "Dinarchus", "Gorgias of Leontini", "Critias", "Alcidamas", "Antiphon of Athens"]


def imputation_documents(data_dir, topic_of, K):
    """topic counts per imputation (documents x K) and the chain of each document"""
    imp = pq.read_table(os.path.join(data_dir, "imputations.parquet")).to_pandas()
    imputations = imp[(imp.restorer == "walk") & (imp.style_tilted == False)].copy()  # noqa: E712
    imputations["chain_id"] = np.where(imputations.chain.isna(), imputations.chain_set.astype(str),
                                imputations.chain_set.astype(str) + " " + imputations.chain.astype(str))
    keys = imputations.groupby(["chain_id", "iteration"], sort=True).ngroup().to_numpy()
    counts = np.zeros((keys.max() + 1, K), dtype=np.int32)
    chain_of = np.empty(keys.max() + 1, dtype=object)
    for key, chain_id, idx, cnt in zip(keys, imputations.chain_id, imputations.lemma_idx, imputations["count"]):
        chain_of[key] = chain_id
        for i, c in zip(idx, cnt):
            t = topic_of.get(int(i), -1)
            if t >= 0:
                counts[key, t] += int(c)
    return counts, chain_of


def rate(counts):
    """pooled rate per thousand topic lemmata"""
    return counts.sum(0) / counts.sum() * 1000


def bootstrap_by_group(counts, groups):
    """rates of BOOT resamples of the groups (chains) with replacement"""
    names, index = np.unique(groups, return_inverse=True)
    totals = np.zeros((len(names), counts.shape[1]))
    np.add.at(totals, index, counts)
    picks = RNG.integers(0, len(names), size=(BOOT, len(names)))
    sums = totals[picks].sum(1)
    return sums / sums.sum(1, keepdims=True) * 1000


def bootstrap_by_row(counts):
    """rates of BOOT resamples of the rows (windows) with replacement, by multinomial weights in chunks"""
    out = np.empty((BOOT, counts.shape[1]))
    n = counts.shape[0]
    chunk = max(1, int(2e8 // n))
    for start in range(0, BOOT, chunk):
        weights = RNG.multinomial(n, np.full(n, 1 / n), size=min(chunk, BOOT - start))
        sums = weights @ counts
        out[start:start + len(sums)] = sums / sums.sum(1, keepdims=True) * 1000
    return out


def p_document(focus_counts, base_counts):
    """per topic: P(share in a random focus document > share in a random baseline window), ties half"""
    focus_share = focus_counts / np.maximum(focus_counts.sum(1, keepdims=True), 1)
    base_share = base_counts / np.maximum(base_counts.sum(1, keepdims=True), 1)
    p = np.empty(focus_counts.shape[1])
    for k in range(focus_counts.shape[1]):
        base_sorted = np.sort(base_share[:, k])
        below = np.searchsorted(base_sorted, focus_share[:, k], side="left")
        equal = np.searchsorted(base_sorted, focus_share[:, k], side="right") - below
        p[k] = np.mean((below + equal / 2) / len(base_sorted))
    return p


def compare(name, focus, base, focus_boot, base_boot, topics):
    """one row per topic: rates, lift with its 90 % interval, and the document-level probability"""
    lift_boot = focus_boot / base_boot
    low, high = np.percentile(lift_boot, [5, 95], axis=0)
    return pd.DataFrame({"topic": topics, "comparison": name,
                         "rate_focus": rate(focus), "rate_baseline": rate(base),
                         "lift": rate(focus) / rate(base), "lift_q05": low, "lift_q95": high,
                         "p_document": p_document(focus, base)})


def draw(table, title, topics, path, columns=9):
    """one circle per topic in a grid ordered by the focus set's share, largest first (row by row), one text-width figure"""
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from figure_style import apply, TEXT_WIDTH
    apply()
    cmap = plt.get_cmap("RdBu_r")
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-1.5, vmax=1.5)
    share = table["rate_focus"].to_numpy()
    order = np.argsort(-share)
    rows = int(np.ceil(len(topics) / columns))
    # the area runs linearly from the smallest share (a small circle) to the largest (0.4 radius, in a 1 x 1.5 grid
    # cell with the label under the circle): the shares of 34-38 topics lie within a factor of two or three of each
    # other, so circles proportional to the share itself would all look alike
    spread = (share - share.min()) / max(share.max() - share.min(), 1e-12)
    radius = np.sqrt(0.12 ** 2 + (0.4 ** 2 - 0.12 ** 2) * spread)
    significant = ((table["lift_q05"] > 1) | (table["lift_q95"] < 1)).to_numpy()
    lift = table["lift"].to_numpy()
    width_units, height_units = columns, 1.5 * rows + 0.3
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, TEXT_WIDTH * height_units / width_units + 0.55), layout="constrained")
    for rank, k in enumerate(order):
        x, y = rank % columns, -1.5 * (rank // columns)
        colour = cmap(norm(np.log2(lift[k])))
        if significant[k]:
            ax.add_patch(plt.Circle((x, y), radius[k], facecolor=colour, edgecolor="black", linewidth=0.7, zorder=2))
        else:
            ax.add_patch(plt.Circle((x, y), radius[k], facecolor=colour, alpha=0.3, edgecolor="none", zorder=1))
        name = topics[k].replace("-", "\n", 1) if len(topics[k]) > 14 else topics[k]
        label = f"{name}\n{lift[k]:.2f}×" if significant[k] else f"{name}\n({lift[k]:.2f}×)"
        ax.text(x, y - 0.44, label, ha="center", va="top", fontsize=6, linespacing=0.95, zorder=3,
                fontweight="bold" if significant[k] else "normal", color="black" if significant[k] else "#666666")
    ax.set_title(title, fontsize=8, pad=4)
    ax.set_xlim(-0.55, columns - 0.45); ax.set_ylim(-1.5 * (rows - 1) - 1.05, 0.5)
    ax.set_aspect("equal"); ax.axis("off")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    bar = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.05, pad=0.01, aspect=70, shrink=0.55)
    bar.set_label(f"log2 lift against the baseline (full colour, black edge, lift printed: 90 % interval off 1); "
                  f"area from the smallest share, {100 * share.min():.1f} %, to the largest, {100 * share.max():.1f} %", fontsize=6.5)
    bar.ax.tick_params(labelsize=6)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results/hard", help="where the tables go")
    ap.add_argument("--figures", default="figures", help="where the figures go")
    ap.add_argument("--baselines", default=None, help="baseline_windows.parquet (default: in --data)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); os.makedirs(a.figures, exist_ok=True)

    vocab = json.load(open(os.path.join(a.data, "vocab.json"), encoding="utf-8"))
    topics = sorted({r["cluster_K40"] for r in vocab if r["cluster_K40"]})
    topic_id = {t: k for k, t in enumerate(topics)}
    topic_of = {r["index"]: topic_id[r["cluster_K40"]] for r in vocab if r["cluster_K40"]}
    K = len(topics)

    windows = pq.read_table(a.baselines or os.path.join(a.data, "baseline_windows.parquet")).to_pandas()
    assert list(windows.columns[5:]) == topics, "topic columns of the windows differ from vocab.json"
    counts = {}
    counts["sophocles"] = windows.loc[windows.author == "sophocles", topics].to_numpy(dtype=np.int32)
    counts["tragedians"] = windows.loc[windows.author.isin(["aeschylus", "euripides"]), topics].to_numpy(dtype=np.int32)
    counts["attic"] = windows.loc[(windows.source == "oga") & windows.author.isin(ATTIC), topics].to_numpy(dtype=np.int32)
    counts["imputations"], chain_of = imputation_documents(a.data, topic_of, K)
    for cell in counts:
        print(f"{cell}: {len(counts[cell])} documents, {counts[cell].sum()} topic lemmata", flush=True)
    attic_authors = windows[(windows.source == "oga") & windows.author.isin(ATTIC)].author.value_counts()
    print("Classical Attic, windows by author: " + ", ".join(f"{k} {v}" for k, v in attic_authors.items()), flush=True)
    missing = [x for x in ATTIC if x not in attic_authors.index]
    if missing:
        print("  not in the windows: " + ", ".join(missing), flush=True)

    boot = {cell: bootstrap_by_row(counts[cell]) for cell in ("sophocles", "tragedians", "attic")}
    boot["imputations"] = bootstrap_by_group(counts["imputations"], chain_of)
    print("bootstraps done", flush=True)

    baseline_name = {"sophocles": "Sophocles", "tragedians": "Aeschylus and Euripides", "attic": "Classical Attic"}
    tables = {}
    for focus, bases in (("imputations", ("sophocles", "tragedians", "attic")), ("sophocles", ("tragedians", "attic"))):
        for base in bases:
            tables[(focus, base)] = compare(f"{focus} vs {base}", counts[focus], counts[base], boot[focus], boot[base], topics)
    pd.concat(tables.values()).to_csv(os.path.join(a.out, "topic_circles.csv"), index=False, float_format="%.4f")

    focus_name = {"imputations": "the imputations", "sophocles": "Sophocles"}
    for (focus, base), table in tables.items():
        title = f"The topics of {focus_name[focus]} against {baseline_name[base]}: area the share (smallest to largest), colour the lift"
        draw(table, title, topics, os.path.join(a.figures, f"hard_circles_{focus}_vs_{base}.png"))
    for (focus, base), t in tables.items():
        sig = t[(t.lift_q05 > 1) | (t.lift_q95 < 1)].sort_values("lift", ascending=False)
        print(f"{focus} vs {base}: {len(sig)} topics with an interval off 1; largest lifts "
              + ", ".join(f"{r.topic} {r.lift:.2f}" for r in sig.head(4).itertuples())
              + "; smallest " + ", ".join(f"{r.topic} {r.lift:.2f}" for r in sig.tail(3).itertuples()), flush=True)
    print(f"wrote {a.figures}/hard_circles_<focus>_vs_<baseline>.png ({len(tables)} figures) and {a.out}/topic_circles.csv", flush=True)


if __name__ == "__main__":
    main()
