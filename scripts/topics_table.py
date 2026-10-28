#!/usr/bin/env python3
"""The topics as a LaTeX table: English gloss, the three label lemmata, and the
cluster's share of the topic-bearing lemmata and of
their tokens in the three tragedians.

The topics are read from the vocabulary of the dataset (data/vocab.json: every
lemma with its cluster label, the cluster's three label lemmata and the lemma's
token count in the three tragedians), the same file the hard report builds its
own topics table from.

Usage: python scripts/topics_table.py data/vocab.json latex/topics_table.tex
"""
import json, sys

def main(vocab_path, out_path):
    vocab = json.load(open(vocab_path, encoding="utf-8"))
    by_label = {}
    for entry in vocab:
        label = entry["cluster_K40"]
        if label is None: continue
        t = by_label.setdefault(label, {"label": label, "lemmata": entry["cluster_K40_lemmata"], "size": 0, "tragic_tokens": 0})
        t["size"] += 1; t["tragic_tokens"] += entry["tragic_count"]
    topics = sorted(by_label.values(), key=lambda t: -t["tragic_tokens"])     # in order of their mass in the tragedians
    n_lemmata, n_tokens = sum(t["size"] for t in topics), sum(t["tragic_tokens"] for t in topics)
    lines = ["% The topics of the hard model: English gloss, the three label lemmata, and the cluster's shares.",
             "% Needs xelatex (or lualatex) with a main font covering polytonic Greek, and \\usepackage{booktabs}; spans both columns of a two-column layout.",
             "\\begin{table*}[htbp]", "\\centering", "\\footnotesize", "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}llrr}", "\\toprule",
             "Topic & Label lemmata & Share of vocabulary (\\%) & Share of tragic tokens (\\%) \\\\", "\\midrule"]
    for t in topics:
        lines.append(f"{t['label']} & {t['lemmata']} & {100 * t['size'] / n_lemmata:.1f} & {100 * t['tragic_tokens'] / n_tokens:.1f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular*}",
              f"\\caption{{The {len(topics)} unsupervised topics, in order of their mass in the three tragedians. A topic is named by the three",
              "non-cognate lemmata nearest its centroid in the Word2Vec space; the English glosses are the labels used in the",
              f"analysis. The last columns give the cluster's share of the {n_lemmata} topic-bearing lemmata of the vocabulary and of their {n_tokens}",
              "tokens in the three tragedians; each column sums to 100.}", "\\label{tab:topics}", "\\end{table*}"]
    open(out_path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"{len(topics)} topics -> {out_path}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
