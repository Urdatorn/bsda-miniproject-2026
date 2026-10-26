"""The paper's results figure: the imputations' lift against Sophocles for every
topic, under the hard and the soft model, one column wide, text set like the body text (figure_style.paper).

Hard: rates per thousand topic lemmata, 90 % interval from resampling the
imputations by chain and Sophocles by window (results/hard/topic_circles.csv).
Soft: ratio of the two mean profiles, 90 % interval over the posterior draws
(results/soft/comparisons.csv). Topics ordered by the hard lift.

Usage: python scripts/paper_lift_figure.py latex/overleaf/figures/lift_hard_soft.pdf
"""
import csv, sys

from figure_style import paper, save, COLUMN_WIDTH, OKABE_ITO


def lifts(path, comparison="imputations vs sophocles"):
    rows = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r["comparison"] == comparison:
            rows[r["topic"]] = tuple(float(r[c]) for c in ("lift", "lift_q05", "lift_q95"))
    return rows


def main(out_path):
    paper()
    import matplotlib.pyplot as plt

    hard = lifts("results/hard/topic_circles.csv")
    soft = lifts("results/soft/comparisons.csv")
    topics = sorted(hard, key=lambda t: hard[t][0])          # largest hard lift on top
    floor = 0.02                                            # a lift of zero sits at the left edge

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 6.6), layout="constrained")
    for offset, (name, values, colour) in zip((0.17, -0.17), (("hard", hard, OKABE_ITO[0]), ("soft", soft, OKABE_ITO[1]))):
        y = [i + offset for i in range(len(topics))]
        mid = [max(values[t][0], floor) for t in topics]
        lo = [max(values[t][1], floor) for t in topics]
        hi = [max(values[t][2], floor) for t in topics]
        ax.hlines(y, lo, hi, colors=colour, linewidth=0.8)
        ax.plot(mid, y, "o", color=colour, markersize=3, label=name)
    ax.axvline(1, color="grey", linestyle="--", linewidth=0.6)
    ax.axvline(1.5, color="grey", linestyle=":", linewidth=0.6)
    ax.set_xscale("log")
    ax.set_xlim(floor, 4)
    ax.set_xticks([0.05, 0.25, 1, 4])
    ax.set_xticklabels(["0.05", "0.25", "1", "4"])
    ax.minorticks_off()
    ax.set_yticks(range(len(topics)))
    ax.set_yticklabels(topics)
    ax.set_ylim(-0.7, len(topics) - 0.3)
    ax.set_xlabel("lift (log scale)")
    fig.legend(loc="outside upper center", ncol=2, frameon=False, handletextpad=0.2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    save(fig, out_path)
    print(f"{len(topics)} topics -> {out_path}")


if __name__ == "__main__":
    main(sys.argv[1])
