"""The soft model's circle grids, the twins of the hard model's.

One circle per seeded topic in a grid ordered by the focus cell's mean
profile under the soft model (the circle's area), coloured by the log2 lift
of that mean against a baseline, in full colour with the lift printed where
the lift's 90 % interval over the posterior draws of the two mean profiles
excludes 1.  The comparisons come from results/soft/comparisons.csv
(scripts/soft_model.py): the imputations against Sophocles, against Aeschylus and
Euripides and against Classical Attic; Sophocles against the latter two.

  python scripts/soft_circles.py --soft results/soft --figures figures

writes soft_circles_<focus>_vs_<baseline>.png, one text-width figure each.
"""
import argparse, os

import pandas as pd

from topic_circles import draw

BASELINE = {"sophocles": "Sophocles", "tragedians": "Aeschylus and Euripides", "attic": "Classical Attic"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--soft", default="results/soft")
    ap.add_argument("--figures", default="figures")
    a = ap.parse_args()
    os.makedirs(a.figures, exist_ok=True)
    table = pd.read_csv(os.path.join(a.soft, "comparisons.csv"))
    topics = sorted(table.topic.unique())

    focus_name = {"imputations": "the imputations", "sophocles": "Sophocles"}
    n = 0
    for focus, base in (("imputations", "sophocles"), ("imputations", "tragedians"), ("imputations", "attic"), ("sophocles", "tragedians"), ("sophocles", "attic")):
        rows = table[table.comparison == f"{focus} vs {base}"]
        if rows.empty:
            continue
        rows = rows.set_index("topic").loc[topics].reset_index()
        title = f"The soft model: {focus_name[focus]} against {BASELINE[base]}: area the mean share (smallest to largest), colour the lift"
        draw(rows, title, topics, os.path.join(a.figures, f"soft_circles_{focus}_vs_{base}.png")); n += 1
    print(f"wrote {n} figures soft_circles_<focus>_vs_<baseline>.png to {a.figures}", flush=True)


if __name__ == "__main__":
    main()
