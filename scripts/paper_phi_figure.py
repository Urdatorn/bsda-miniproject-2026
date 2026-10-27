"""The paper's illustration of a soft topic: part of phi_k for
marriage-couch-bed, one column wide, text set like the body text (figure_style.paper).

The x axis follows the vocabulary's own (alphabetical) order: its first five
lemmata, then five of the topic's own lemmata where they fall, with gaps
marked by an ellipsis, so the plot shows both the near-zero floor that most
of the 3,128 lemmata sit on and the few lemmata that carry the topic.

Usage: python scripts/paper_phi_figure.py latex/overleaf/figures/phi_marriage.pdf
"""
import csv, gzip, sys

from figure_style import paper, save, COLUMN_WIDTH, OKABE_ITO

TOPIC = "marriage-couch-bed"
HEAD = 5                                                    # first lemmata of the vocabulary
OWN = ["γάμος", "λέκτρον", "νύμφη", "παρθένος", "πόσις"]    # lemmata of the topic


def phi_row(path, topic):
    rows = csv.reader(gzip.open(path, "rt", encoding="utf-8"))
    vocab = next(rows)[1:]
    for r in rows:
        if r[0] == topic:
            return vocab, [float(x) for x in r[1:]]
    raise KeyError(topic)


def main(out_path):
    paper()
    import matplotlib.pyplot as plt

    vocab, phi = phi_row("results/soft/pooled/phi_matrix.csv.gz", TOPIC)
    index = list(range(HEAD)) + sorted(vocab.index(w) for w in OWN)
    labels, heights, xs, gaps = [], [], [], []
    x = 0
    for i, j in enumerate(index):
        if i and j > index[i - 1] + 1:                      # skipped lemmata: leave a slot for "…"
            gaps.append(x)
            x += 1
        labels.append(r"\textgreek{%s}" % vocab[j])
        heights.append(phi[j])
        xs.append(x)
        x += 1
    gaps.append(x)                                          # the vocabulary trails off

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.1), layout="constrained")
    ax.bar(xs, heights, width=0.7, color=OKABE_ITO[1])
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=55, ha="right", rotation_mode="anchor")
    for g in gaps:                                          # skipped lemmata, upright on the axis
        ax.text(g, -0.04, r"\dots", transform=ax.get_xaxis_transform(), ha="center", va="top")
    ax.set_xlim(-0.6, x + 0.4)
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 0.1)
    ax.set_ylabel(r"$\phi_{k,w}$")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    save(fig, out_path)
    print(f"{TOPIC}: {len(labels)} lemmata -> {out_path}")


if __name__ == "__main__":
    main(sys.argv[1])
