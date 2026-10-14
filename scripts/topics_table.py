#!/usr/bin/env python3
"""The topics as a LaTeX table: English gloss, the two label lemmata, their
transliteration, and the cluster's share of the topic-bearing lemmata and of
their tokens in the three tragedians.

Usage: python scripts/topics_table.py results/analytic/topics_K40.json results/analytic/topics_table.tex
"""
import json, sys, unicodedata

LETTER = {"α": "a", "β": "b", "γ": "g", "δ": "d", "ε": "e", "ζ": "z", "η": "ē", "θ": "th", "ι": "i", "κ": "k", "λ": "l", "μ": "m",
          "ν": "n", "ξ": "x", "ο": "o", "π": "p", "ρ": "r", "σ": "s", "ς": "s", "τ": "t", "υ": "y", "φ": "ph", "χ": "ch", "ψ": "ps", "ω": "ō"}
ROUGH, SUBSCRIPT = "̔", "ͅ"


def transliterate(word):
    """scholarly transliteration: ē ō, h for the rough breathing, y alone and u in diphthongs, n for gamma before a velar"""
    decomposed = unicodedata.normalize("NFD", word.lower())
    letters, rough, subscript, k = [], set(), set(), -1
    for c in decomposed:
        if not unicodedata.combining(c): letters.append(c); k += 1
        elif c == ROUGH: rough.add(k)
        elif c == SUBSCRIPT: subscript.add(k)
    out, j = [], 0
    while j < len(letters):
        c, nxt = letters[j], letters[j + 1] if j + 1 < len(letters) else ""
        if c == "γ" and nxt in "γκξχ": out.append("n"); j += 1; continue
        if c == "ρ" and j in rough: out.append("rh"); j += 1; continue
        if c in "αεηιουω":
            diphthong = c in "αεηου" and nxt in ("ι", "υ") and nxt != c
            h = "h" if j in rough or (diphthong and j + 1 in rough) else ""
            if diphthong: out.append(h + LETTER[c] + ("u" if nxt == "υ" else "i")); j += 2
            else: out.append(h + LETTER[c] + ("i" if j in subscript else "")); j += 1
            continue
        out.append(LETTER.get(c, c)); j += 1
    return "".join(out)


def main(topics_path, out_path):
    topics = json.load(open(topics_path, encoding="utf-8"))["topics"]
    n_lemmata, n_tokens = sum(t["size"] for t in topics), sum(t["tragic_tokens"] for t in topics)
    lines = ["% The topics of the analytic model: English gloss, the two label lemmata, their transliteration.",
             "% Needs xelatex (or lualatex) with a main font covering polytonic Greek, and \\usepackage{booktabs}.",
             "\\begin{table}[htbp]", "\\centering", "\\small", "\\begin{tabular}{lllrr}", "\\toprule",
             "Topic & Label lemmata & Transliteration & Lemmata (\\%) & Tragic tokens (\\%) \\\\", "\\midrule"]
    for t in topics:
        transliteration = "-".join(transliterate(w) for w in t["lemmata"].split("-"))
        lines.append(f"{t['label']} & {t['lemmata']} & {transliteration} & {100 * t['size'] / n_lemmata:.1f} & {100 * t['tragic_tokens'] / n_tokens:.1f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              f"\\caption{{The {len(topics)} unsupervised topics, in order of their mass in the three tragedians. A topic is named by the two",
              "non-cognate lemmata nearest its centroid in the word2vec space; the English glosses are the labels used in the",
              f"analysis. The last columns give the cluster's share of the {n_lemmata} topic-bearing lemmata of the vocabulary and of their {n_tokens}",
              "tokens in the three tragedians; each column sums to 100.}", "\\label{tab:topics}", "\\end{table}"]
    open(out_path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"{len(topics)} topics -> {out_path}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
