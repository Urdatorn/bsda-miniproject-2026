"""The paper's illustration of a soft topic: part of phi_k for
marriage-couch-bed, one column wide, text set like the body text (figure_style.paper).

The x axis follows the vocabulary in alphabetical order: its first lemma, five
of the topic's own lemmata where they fall and the last lemma, with gaps
marked by an ellipsis, so the plot shows both the near-zero floor that most
of the 3,128 lemmata sit on and the few lemmata that carry the topic.

Usage: python scripts/paper_phi_figure.py latex/overleaf/figures/phi_marriage.pdf
"""
import csv, gzip, sys, unicodedata

from figure_style import paper, save, COLUMN_WIDTH, OKABE_ITO

TOPIC = "marriage-couch-bed"
HEAD = 1                                                    # the first lemma of the vocabulary
OWN = ["γάμος", "λέκτρον", "νύμφη", "παρθένος", "πόσις"]    # lemmata of the topic
SPELLING = {"ἀατός": "ἀάατος"}                             # the lemmatiser's form in data/vocab.json, corrected for display


def phi_row(path, topic):
    rows = csv.reader(gzip.open(path, "rt", encoding="utf-8"))
    vocab = next(rows)[1:]
    for r in rows:
        if r[0] == topic:
            return vocab, [float(x) for x in r[1:]]
    raise KeyError(topic)


def alphabetical(word):
    """dictionary order: the base letters, without accents and breathings, final sigma as sigma"""
    base = "".join(c for c in unicodedata.normalize("NFD", word.lower()) if not unicodedata.combining(c))
    return base.replace("ς", "σ"), word


def main(out_path):
    paper()
    import matplotlib.pyplot as plt

    vocab, phi = phi_row("results/soft/pooled/phi_matrix.csv.gz", TOPIC)          # the mean of the twenty states
    order = sorted(range(len(vocab)), key=lambda i: alphabetical(vocab[i]))     # from the first alpha word to the last omega word
    rank = {i: r for r, i in enumerate(order)}
    index = order[:HEAD] + sorted((vocab.index(w) for w in OWN), key=rank.get) + order[-1:]
    labels, heights, xs, gaps = [], [], [], []
    x = 0
    for n, j in enumerate(index):
        if n and rank[j] > rank[index[n - 1]] + 1:          # skipped lemmata: leave a slot for "…"
            gaps.append(x)
            x += 1
        labels.append(r"\textgreek{%s}" % SPELLING.get(vocab[j], vocab[j]))
        heights.append(phi[j])
        xs.append(x)
        x += 1

    # a broken y axis: the upper panel shows the topic's own lemmata on a linear scale,
    # the lower one the floor of about 4e-5 that most of the vocabulary sits on
    fig, (top, low) = plt.subplots(2, 1, sharex=True, figsize=(COLUMN_WIDTH, 2.2), layout="constrained",
                                   gridspec_kw={"height_ratios": [2.4, 1]})
    fig.get_layout_engine().set(hspace=0.16)                                     # room around the break
    for ax in (top, low):
        ax.bar(xs, heights, width=0.7, color=OKABE_ITO[1])
        ax.spines["right"].set_visible(False)
    top.set_ylim(0.016, 0.038)                                                  # ticks clear of the break
    top.set_yticks([0.02, 0.03])
    low.set_ylim(0, 8e-5)
    low.set_yticks([0, 4e-5])
    low.set_yticklabels(["0", r"$4\cdot10^{-5}$"])
    top.spines["bottom"].set_visible(False)
    top.spines["top"].set_visible(False)
    top.tick_params(axis="x", bottom=False)
    low.spines["top"].set_visible(False)
    mark = dict(marker=[(-1, -0.5), (1, 0.5)], markersize=6, linestyle="none", color="k", mec="k", mew=0.6, clip_on=False)
    top.plot([0], [0], transform=top.transAxes, **mark)                         # the break, on the y axis
    low.plot([0], [1], transform=low.transAxes, **mark)
    low.set_xticks(xs)
    low.set_xticklabels(labels, rotation=55, ha="right", rotation_mode="anchor")
    for g in gaps:                                          # skipped lemmata, upright on the axis
        low.text(g, -0.12, r"\dots", transform=low.get_xaxis_transform(), ha="center", va="top")
    low.set_xlim(-0.6, x - 0.4)
    fig.supylabel(r"$\phi_{k,w}$")
    save(fig, out_path)
    print(f"{TOPIC}: {', '.join(vocab[j] for j in index)} -> {out_path}")


if __name__ == "__main__":
    main(sys.argv[1])
