"""The paper's figure of the main claim along the walk: the soft lift of
persuade-man-king against Sophocles over the steps of a few chains, one
column wide, text set like the body text (figure_style.paper).

Each line is one of four chains: the posterior mean share of the topic in
every fifth recorded state (results/soft/trajectory/trajectory.csv, from
scripts/soft_trajectory.py; the report's section on the lift along the walk
shows all sixteen), averaged over a moving window of 1,000 steps and divided
by Sophocles' mean share under the same plug-in model
(results/soft/mean_alpha_lda). The dotted line is the lift of all sixteen
chains under that model.

Usage: python scripts/paper_trajectory_figure.py latex/overleaf/figures/trajectory_persuade.pdf
"""
import csv, sys
from collections import defaultdict

from figure_style import paper, save, COLUMN_WIDTH, OKABE_ITO

TOPIC = "persuade-man-king"
WINDOW = 20                        # states in the moving window: 20 states 50 steps apart, 1,000 steps
CHAINS = (0, 5, 10, 15)            # two of the 12,000-step chains, two of the 10,000-step ones


def plug_in(path, column):
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r["topic"] == TOPIC and r.get("source", "sophocles") == "sophocles" and r.get("comparison", "imputations vs sophocles") == "imputations vs sophocles":
            return float(r[column])


def main(out_path):
    paper()
    import matplotlib.pyplot as plt

    sophocles = plug_in("results/soft/mean_alpha_lda/profiles.csv", "mean")
    pooled = plug_in("results/soft/mean_alpha_lda/comparisons.csv", "lift")
    chains = defaultdict(list)
    for r in csv.DictReader(open("results/soft/trajectory/trajectory.csv", encoding="utf-8")):
        if int(r["chain"]) in CHAINS:
            chains[int(r["chain"])].append((int(r["step"]), float(r[TOPIC])))

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 1.6), layout="constrained")
    for colour, (chain, states) in zip(OKABE_ITO[2:], sorted(chains.items())):
        states.sort()
        steps = [s for s, _ in states][WINDOW - 1:]
        means = [sum(t for _, t in states[i - WINDOW + 1:i + 1]) / WINDOW for i in range(WINDOW - 1, len(states))]
        ax.plot([s / 1000 for s in steps], [m / sophocles for m in means], color=colour, linewidth=0.9,
                label=f"chain {chain + 1}")
    ax.axhline(1, color="grey", linestyle="--", linewidth=0.6)
    ax.axhline(pooled, color="black", linestyle=":", linewidth=0.8)
    ax.set_xlabel("step of the walk (thousands)")
    ax.set_ylabel("lift")
    ax.set_ylim(bottom=0)
    ax.set_yticks([0, 2, 4, 6])
    fig.legend(loc="outside upper center", ncol=4, frameon=False, handlelength=1.0,
               handletextpad=0.3, columnspacing=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    save(fig, out_path)
    print(f"{TOPIC}: {len(chains)} chains, Sophocles {sophocles:.4f}, pooled lift {pooled:.2f} -> {out_path}")


if __name__ == "__main__":
    main(sys.argv[1])
