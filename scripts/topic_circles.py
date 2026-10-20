"""Topic circle maps: is a cell's use of each topic unusual, and against whom?

One circle per topic, placed at the centre of its lemmata in the lemma map's
layout (lemma_map_coords.csv, UMAP of the word2vec vectors), so the circles
sit where the topics sit on the lemma map.  A circle's area is the focus
cell's share of the topic, its colour the log2 lift of that share against a
baseline, and it is drawn in full colour with a black edge when the lift's
90 % interval excludes 1 (otherwise faded).  Two figures:

  topic_circles_walk.png       the walk against Sophocles, against Aeschylus
                               and Euripides, and against Greek at large;
  topic_circles_sophocles.png  Sophocles against Aeschylus and Euripides and
                               against Greek at large: his own topics.

Shares are rates per thousand topic lemmata, pooled over the cell's
documents.  The baselines are windows of 28 vocabulary lemmata
(baseline_windows.parquet, scripts/window_baselines.py): Sophocles' and the
other tragedians' extant trimeters, and every work of the OGA corpus.  The
interval of a lift comes from two bootstraps: the walk resampled by chain
(48 chains), a baseline resampled by window.  p_document is the chance that
a random document of the focus cell has a larger share than a random window
of the baseline.  Everything is written to topic_circles.csv.

  python scripts/topic_circles.py --data data --out results/analytic
"""
import argparse, json, os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

RNG = np.random.default_rng(917)
BOOT = 2000


def walk_documents(data_dir, topic_of, K):
    """topic counts per walk snapshot (documents x K) and the chain of each document"""
    imp = pq.read_table(os.path.join(data_dir, "imputations.parquet")).to_pandas()
    walk = imp[(imp.restorer == "walk") & (imp.style_tilted == False)].copy()  # noqa: E712
    walk["chain_id"] = np.where(walk.chain.isna(), walk.chain_set.astype(str),
                                walk.chain_set.astype(str) + " " + walk.chain.astype(str))
    keys = walk.groupby(["chain_id", "iteration"], sort=True).ngroup().to_numpy()
    counts = np.zeros((keys.max() + 1, K), dtype=np.int32)
    chain_of = np.empty(keys.max() + 1, dtype=object)
    for key, chain_id, idx, cnt in zip(keys, walk.chain_id, walk.lemma_idx, walk["count"]):
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


def spread(centres, radius, span, steps=400):
    """nudge overlapping circles apart (and their labels' room below them), keeping each near its place"""
    pos = centres.copy()
    room = radius + 0.055 * span               # the label sits under the circle
    for _ in range(steps):
        moved = False
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                d = pos[j] - pos[i]
                dist = np.hypot(*d)
                need = max(radius[i] + radius[j], 0.82 * (room[i] + room[j]))
                if dist < need:
                    push = (need - dist) / 2 * (d / dist if dist > 0 else np.array([1.0, 0.0]))
                    pos[i] -= push; pos[j] += push; moved = True
        if not moved:
            break
    return pos


def draw(panels, focus_name, centres, topics, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    fig, axes = plt.subplots(1, len(panels), figsize=(7.4 * len(panels), 7.8), squeeze=False,
                             layout="constrained")
    cmap = plt.get_cmap("RdBu_r")
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-1.5, vmax=1.5)
    span = max(np.ptp(centres[:, 0]), np.ptp(centres[:, 1]))
    share = panels[0][1]["rate_focus"].to_numpy()           # the focus cell's share, the same in every panel
    radius = 0.07 * span * np.sqrt(share / share.max())
    pos = spread(centres, radius, span)
    for ax, (title, table) in zip(axes[0], panels):
        significant = ((table["lift_q05"] > 1) | (table["lift_q95"] < 1)).to_numpy()
        lift = table["lift"].to_numpy()
        for k in np.argsort(-radius):
            colour = cmap(norm(np.log2(lift[k])))
            if significant[k]:
                ax.add_patch(plt.Circle(pos[k], radius[k], facecolor=colour, edgecolor="black", linewidth=1.1, zorder=2))
            else:
                ax.add_patch(plt.Circle(pos[k], radius[k], facecolor=colour, alpha=0.3, edgecolor="none", zorder=1))
        for k in range(len(topics)):
            name = topics[k].replace("-", "\n", 1) if len(topics[k]) > 16 else topics[k]
            label = f"{name}\n{lift[k]:.2f}×" if significant[k] else name
            ax.text(pos[k, 0], pos[k, 1] - radius[k] - 0.008 * span, label, ha="center", va="top", fontsize=6.3,
                    fontweight="bold" if significant[k] else "normal", color="black" if significant[k] else "#555555",
                    linespacing=0.95, zorder=3)
        ax.set_title(title, fontsize=11, pad=14)
        ax.set_xlim(pos[:, 0].min() - 0.12 * span, pos[:, 0].max() + 0.12 * span)
        ax.set_ylim(pos[:, 1].min() - 0.16 * span, pos[:, 1].max() + 0.12 * span)
        ax.set_aspect("equal"); ax.axis("off")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    bar = fig.colorbar(sm, ax=axes[0].tolist(), orientation="horizontal", fraction=0.04, pad=0.01, aspect=60, shrink=0.6)
    bar.set_label("log2 lift of the topic's share against the baseline; full colour, black edge and the lift printed: "
                  "its 90 % interval excludes 1", fontsize=8)
    bar.ax.tick_params(labelsize=8)
    fig.suptitle(f"The topics of {focus_name}: circle area the share of the topic, placed as on the lemma map", fontsize=12, y=1.04)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results/analytic")
    ap.add_argument("--baselines", default=None, help="baseline_windows.parquet (default: in --data)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

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
    counts["oga"] = windows.loc[windows.source == "oga", topics].to_numpy(dtype=np.int32)
    counts["walk"], chain_of = walk_documents(a.data, topic_of, K)
    for cell in counts:
        print(f"{cell}: {len(counts[cell])} documents, {counts[cell].sum()} topic lemmata", flush=True)
    top_authors = windows[windows.source == "oga"].author.value_counts().head(8)
    print("Greek at large, largest authors by windows: " + ", ".join(f"{k} {v}" for k, v in top_authors.items()), flush=True)

    boot = {cell: bootstrap_by_row(counts[cell]) for cell in ("sophocles", "tragedians", "oga")}
    boot["walk"] = bootstrap_by_group(counts["walk"], chain_of)
    print("bootstraps done", flush=True)

    baseline_name = {"sophocles": "Sophocles", "tragedians": "Aeschylus and Euripides", "oga": "Greek at large (OGA)"}
    tables = {}
    for focus, bases in (("walk", ("sophocles", "tragedians", "oga")), ("sophocles", ("tragedians", "oga"))):
        for base in bases:
            tables[(focus, base)] = compare(f"{focus} vs {base}", counts[focus], counts[base], boot[focus], boot[base], topics)
    pd.concat(tables.values()).to_csv(os.path.join(a.out, "topic_circles.csv"), index=False, float_format="%.4f")

    coords = pd.read_csv(os.path.join(a.out, "lemma_map_coords.csv"))
    centres = coords.groupby("topic")[["x", "y"]].median().loc[topics].to_numpy()
    draw([(f"against {baseline_name[b]}", tables[("walk", b)]) for b in ("sophocles", "tragedians", "oga")],
         "the walk", centres, topics, os.path.join(a.out, "topic_circles_walk.png"))
    draw([(f"against {baseline_name[b]}", tables[("sophocles", b)]) for b in ("tragedians", "oga")],
         "Sophocles", centres, topics, os.path.join(a.out, "topic_circles_sophocles.png"))
    for (focus, base), t in tables.items():
        sig = t[(t.lift_q05 > 1) | (t.lift_q95 < 1)].sort_values("lift", ascending=False)
        print(f"{focus} vs {base}: {len(sig)} topics with an interval off 1; largest lifts "
              + ", ".join(f"{r.topic} {r.lift:.2f}" for r in sig.head(4).itertuples())
              + "; smallest " + ", ".join(f"{r.topic} {r.lift:.2f}" for r in sig.tail(3).itertuples()), flush=True)
    print(f"wrote {a.out}/topic_circles_walk.png, topic_circles_sophocles.png, topic_circles.csv", flush=True)


if __name__ == "__main__":
    main()
