"""The paper's results figure: the imputations' lift against Sophocles for every
topic, under the hard and the soft model, one column wide, text set like the body text (figure_style.paper).

Hard: rates per thousand topic lemmata, 90 % interval from resampling the
imputations by chain and Sophocles by window (results/hard/topic_circles.csv).
Soft: ratio of the two mean profiles pooled over twenty states of phi, 90 %
interval from the same resampling by chain, with a state of phi drawn in each
replicate (results/soft/pooled/comparisons.csv). Only the topics that stand
out under at least one model (90 % interval entirely above or below 1) are
drawn. Topics ordered by the soft lift; a topic whose soft interval lies entirely above 1 is
set in bold, one whose soft interval lies entirely below 1 in italic. A dagger
marks a topic whose soft lift against Attic stands out in the same
direction as against Sophocles, a double dagger one whose hard lift does
(results/hard/topic_circles.csv and results/soft/pooled/comparisons.csv,
imputations vs attic). The number after a topic is the rank, in its soft word
distribution phi_k (data/soft/phi_matrix.csv.gz), of its highest-ranking
preserved lemma: a lemma of the papyrus's surviving words, those of the intact
verses 1-12 (their bags in data/corpus.parquet, with the three the tagger
missed) and the whole words before the lacunae of verses 13-29.

Usage: python scripts/paper_lift_figure.py latex/overleaf/figures/lift_hard_soft.pdf
"""
import csv, gzip, sys

import pandas as pd

from figure_style import paper, save, COLUMN_WIDTH, OKABE_ITO


# lemmata of the surviving words that the intact verses' bags lack: δώματα (v. 10), φυσάντων (v. 8),
# χρεών (v. 12), which the tagger missed, and the whole words before the lacunae of vv. 13-29
PRESERVED_EXTRA = ["δῶμα", "φύω", "χρεών", "νόμος", "ὁράω", "χρηστός", "δέσποινα", "λόγος", "ὅρκος", "φράζω",
                   "κοινός", "ἕρπω", "ἄγρα"]


def preserved_ranks():
    """Each topic's best rank (1 = its most probable lemma) among the preserved lemmata."""
    rows = csv.reader(gzip.open("data/soft/phi_matrix.csv.gz", "rt", encoding="utf-8"))
    vocab = next(rows)[1:]
    corpus = pd.read_parquet("data/corpus.parquet")
    preserved = {vocab[i] for bag in corpus[corpus.source == "intact"].lemma_idx for i in bag} | set(PRESERVED_EXTRA)
    ranks = {}
    for r in rows:
        order = sorted(range(len(vocab)), key=lambda j: -float(r[j + 1]))
        ranks[r[0]] = next(k + 1 for k, j in enumerate(order) if vocab[j] in preserved)
    return ranks


def lifts(path, columns=("lift", "lift_q05", "lift_q95"), comparison="imputations vs sophocles"):
    rows = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r["comparison"] == comparison:
            rows[r["topic"]] = tuple(float(r[c]) for c in columns)
    return rows


def main(out_path):
    paper()
    import matplotlib.pyplot as plt

    hard = lifts("results/hard/topic_circles.csv")
    soft = lifts("results/soft/pooled/comparisons.csv", ("lift", "lift_chain_q05", "lift_chain_q95"))
    hard_attic = lifts("results/hard/topic_circles.csv", comparison="imputations vs attic")
    soft_attic = lifts("results/soft/pooled/comparisons.csv", ("lift", "lift_chain_q05", "lift_chain_q95"), "imputations vs attic")
    ranks = preserved_ranks()
    stands_out = lambda v: v[1] > 1 or v[2] < 1
    side = lambda v: 1 if v[1] > 1 else -1 if v[2] < 1 else 0
    # the same side of 1 against Attic as against Sophocles
    also_attic = lambda sophocles, attic: side(sophocles) != 0 and side(attic) == side(sophocles)
    topics = sorted((t for t in soft if stands_out(soft[t]) or stands_out(hard[t])),
                    key=lambda t: soft[t][0])               # largest soft lift on top
    floor = 0.002                                           # a lift of zero sits at the left edge

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 4.5), layout="constrained")
    for offset, (name, values, colour) in zip((0.17, -0.17), (("hard", hard, OKABE_ITO[0]), ("soft", soft, OKABE_ITO[1]))):
        y = [i + offset for i in range(len(topics))]
        mid = [max(values[t][0], floor) for t in topics]
        lo = [max(values[t][1], floor) for t in topics]
        hi = [max(values[t][2], floor) for t in topics]
        ax.hlines(y, lo, hi, colors=colour, linewidth=0.8)
        ax.plot(mid, y, "o", color=colour, markersize=3, label=name)
    ax.axvline(1, color="grey", linestyle="--", linewidth=0.6)
    ax.set_xscale("log")
    ax.set_xlim(floor, 4)
    ax.set_xticks([0.01, 0.1, 1, 4])
    ax.set_xticklabels(["0.01", "0.1", "1", "4"])
    ax.minorticks_off()
    ax.set_yticks(range(len(topics)))
    # a topic stands out when its soft 90 % interval lies entirely above 1 (bold) or below 1 (italic)
    def label(t):
        marks = (r"\dag" if also_attic(soft[t], soft_attic[t]) else "") + (r"\ddag" if also_attic(hard[t], hard_attic[t]) else "")
        name = t + (r"\textsuperscript{%s}" % marks if marks else "") + " (%d)" % ranks[t]
        if soft[t][1] > 1: return r"\textbf{%s}" % name
        if soft[t][2] < 1: return r"\textit{%s}" % name
        return name
    ax.set_yticklabels([label(t) for t in topics])
    ax.set_ylim(-0.7, len(topics) - 0.3)
    ax.set_xlabel("lift (log scale)")
    fig.legend(loc="outside upper center", ncol=2, frameon=False, handletextpad=0.2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    save(fig, out_path)
    print(f"{len(topics)} of {len(soft)} topics -> {out_path}")


if __name__ == "__main__":
    main(sys.argv[1])
