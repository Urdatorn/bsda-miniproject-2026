"""Which genre does the walk's topic profile resemble: epic or tragedy?

The walk's generator was trained on Greek at large, and the walk writes
tragic trimeters; its topic profile could lean either way.  Every genre is
a set of OGA authors (tragedy is the three tragedians' extant trimeters of
corpus.parquet, the register the walk is meant to write), pooled into one
topic profile from its baseline windows (baseline_windows.parquet,
scripts/window_baselines.py).  The distance from a focus cell's profile to
each genre is the Jensen-Shannon divergence in bits (0 identical, 1
disjoint) and the Hellinger distance, with a 90 % interval from two
bootstraps: the walk by chain, a genre by window.  Per topic, the table
also says whether the walk's rate is closer (in log ratio) to epic's or to
tragedy's.  Sophocles is run as a second focus, against the genres with
tragedy reduced to Aeschylus and Euripides, as the reference of a text that
is tragedy.

  python scripts/genre_distance.py --data data --out results/hard --figures figures

writes genre_distance.csv and genre_topics.csv there, and hard_genre_distance.png
to the figures folder.
"""
import argparse, json, os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from topic_circles import walk_documents, bootstrap_by_group, bootstrap_by_row, rate

GENRES = {
    "epic": ["Homer", "Hesiod", "Homeric Hymns", "Apollonius Rhodius", "Quintus Smyrnaeus", "Nonnus of Panopolis",
             "Oppian", "Oppian of Apamea", "Tryphiodorus", "Colluthus of Lycopolis", "Aratus Solensis",
             "Nicander of Colophon", "Batrachomyomachia"],
    "tragedy": ["sophocles", "aeschylus", "euripides"],          # the trimeters of corpus.parquet
    "comedy": ["Aristophanes", "Menander"],
    "lyric": ["Pindar", "Theocritus", "Callimachus", "Moschus", "Bion of Phlossa"],
    "oratory": ["Demosthenes", "Isocrates", "Lysias", "Aeschines", "Isaeus", "Andocides", "Antiphon", "Lycurgus",
                "Dinarchus", "Hyperides", "Demades", "Aristides, Aelius", "Libanius", "Dio Chrysostom"],
    "history": ["Herodotus", "Thucydides", "Xenophon", "Polybius", "Diodorus Siculus", "Dionysius of Halicarnassus",
                "Appianus of Alexandria", "Arrian", "Cassius Dio Cocceianus", "Herodian", "Procopius", "Zosimus",
                "Flavius Josephus", "Plutarch", "Pausanias"],
    "philosophy": ["Plato", "Aristotle", "Plotinus", "Epictetus", "Marcus Aurelius", "Sextus Empiricus", "Theophrastus",
                   "Porphyrius", "Proclus", "Iamblichus", "Alexander of Aphrodisias", "Simplicius", "John Philoponus"],
    "medicine": ["Galen", "Hippocrates", "Pseudo-Galen", "Dioscurides Pedianus", "Soranus", "Aretaeus of Cappadocia",
                 "Rufus of Ephesus", "Philumenus"],
    "scripture": ["Septuaginta", "Bible", "New Testament", "Origen", "Eusebius of Caesarea", "Clement of Alexandria",
                  "Basil, Saint, Bishop of Caesarea", "Gregory of Nazianzus", "John of Damascus, Saint",
                  "Cyrill of Alexandria", "Cyril of Alexandria", "Athanasius", "Athanasius of Alexandria"],
}


def js_divergence(p, q):
    """Jensen-Shannon divergence in bits between rows of p and rows of q (both rates, any scale)"""
    p = p / p.sum(-1, keepdims=True)
    q = q / q.sum(-1, keepdims=True)
    m = (p + q) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_p = np.nansum(np.where(p > 0, p * np.log2(p / m), 0.0), axis=-1)
        kl_q = np.nansum(np.where(q > 0, q * np.log2(q / m), 0.0), axis=-1)
    return (kl_p + kl_q) / 2


def hellinger(p, q):
    p = p / p.sum(-1, keepdims=True)
    q = q / q.sum(-1, keepdims=True)
    return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2, axis=-1) / 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results/hard", help="where the tables go")
    ap.add_argument("--figures", default="figures", help="where the figure goes")
    ap.add_argument("--baselines", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); os.makedirs(a.figures, exist_ok=True)

    vocab = json.load(open(os.path.join(a.data, "vocab.json"), encoding="utf-8"))
    topics = sorted({r["cluster_K40"] for r in vocab if r["cluster_K40"]})
    topic_id = {t: k for k, t in enumerate(topics)}
    topic_of = {r["index"]: topic_id[r["cluster_K40"]] for r in vocab if r["cluster_K40"]}
    windows = pq.read_table(a.baselines or os.path.join(a.data, "baseline_windows.parquet")).to_pandas()
    known = set(windows.author)
    for genre, authors in GENRES.items():
        missing = [x for x in authors if x not in known]
        if missing:
            print(f"{genre}: not in the windows: {', '.join(missing)}", flush=True)

    walk_counts, chain_of = walk_documents(a.data, topic_of, len(topics))
    counts = {"walk": walk_counts}
    counts["sophocles"] = windows.loc[windows.author == "sophocles", topics].to_numpy(dtype=np.int32)
    for genre, authors in GENRES.items():
        counts[genre] = windows.loc[windows.author.isin(authors), topics].to_numpy(dtype=np.int32)
    counts["tragedy without Sophocles"] = windows.loc[windows.author.isin(["aeschylus", "euripides"]), topics].to_numpy(dtype=np.int32)
    for cell, c in list(counts.items()):
        print(f"{cell}: {len(c)} documents, {c.sum()} topic lemmata", flush=True)
        if len(c) < 20:                      # too few windows to be a profile
            print(f"  {cell} dropped: fewer than 20 windows", flush=True)
            del counts[cell]
    genres = [g for g in GENRES if g in counts]

    boot = {cell: bootstrap_by_row(c) for cell, c in counts.items() if cell != "walk"}
    boot["walk"] = bootstrap_by_group(counts["walk"], chain_of)
    print("bootstraps done", flush=True)

    rows = []
    for focus in ("walk", "sophocles"):
        for genre in genres:
            base = genre
            if focus == "sophocles" and genre == "tragedy":
                base = "tragedy without Sophocles"
            js = js_divergence(boot[focus], boot[base])
            hd = hellinger(boot[focus], boot[base])
            rows.append({"focus": focus, "genre": genre, "windows": len(counts[base]),
                         "js_bits": js_divergence(rate(counts[focus]), rate(counts[base])),
                         "js_q05": np.percentile(js, 5), "js_q95": np.percentile(js, 95),
                         "hellinger": hellinger(rate(counts[focus]), rate(counts[base])),
                         "hellinger_q05": np.percentile(hd, 5), "hellinger_q95": np.percentile(hd, 95)})
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(a.out, "genre_distance.csv"), index=False, float_format="%.4f")
    for focus in ("walk", "sophocles"):
        t = table[table.focus == focus].sort_values("js_bits")
        print(f"{focus}, nearest genres by JS: " + ", ".join(f"{r.genre} {r.js_bits:.4f} [{r.js_q05:.4f}, {r.js_q95:.4f}]" for r in t.itertuples()), flush=True)

    # per topic: is the walk's rate closer to epic's or to tragedy's (log ratio)?
    walk_rate, epic_rate, tragedy_rate = rate(counts["walk"]), rate(counts["epic"]), rate(counts["tragedy"])
    per_topic = pd.DataFrame({"topic": topics, "rate_walk": walk_rate, "rate_epic": epic_rate, "rate_tragedy": tragedy_rate,
                              "log2_walk_over_epic": np.log2(walk_rate / epic_rate),
                              "log2_walk_over_tragedy": np.log2(walk_rate / tragedy_rate)})
    per_topic["closer_to"] = np.where(per_topic.log2_walk_over_epic.abs() < per_topic.log2_walk_over_tragedy.abs(), "epic", "tragedy")
    # the share of chain-bootstrap draws in which the topic is closer to epic
    closer_epic = np.abs(np.log2(boot["walk"] / epic_rate)) < np.abs(np.log2(boot["walk"] / tragedy_rate))
    per_topic["p_closer_to_epic"] = closer_epic.mean(0)
    per_topic.to_csv(os.path.join(a.out, "genre_topics.csv"), index=False, float_format="%.4f")
    n_epic = int((per_topic.closer_to == "epic").sum())
    print(f"per topic, the walk's rate is closer to epic's for {n_epic} of {len(topics)} topics and to tragedy's for "
          f"{len(topics) - n_epic}: " + ", ".join(f"{r.topic} ({r.closer_to}, P(epic) {r.p_closer_to_epic:.2f})" for r in per_topic.itertuples()), flush=True)

    import matplotlib.pyplot as plt
    from figure_style import apply, TEXT_WIDTH
    apply()
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.6), sharey=False)
    for ax, focus in zip(axes, ("walk", "sophocles")):
        t = table[table.focus == focus].sort_values("js_bits", ascending=False)
        y = np.arange(len(t))
        ax.barh(y, t.js_bits, color=np.where(t.genre.isin(["epic", "tragedy"]), "#e15759", "#9aa0a6"))
        ax.errorbar(t.js_bits, y, xerr=[t.js_bits - t.js_q05, t.js_q95 - t.js_bits], fmt="none", ecolor="black", capsize=2, lw=0.8)
        ax.set_yticks(y); ax.set_yticklabels([f"{g} ({w})" for g, w in zip(t.genre, t.windows)], fontsize=7)
        ax.set_xlabel("Jensen–Shannon divergence (bits)", fontsize=7.5)
        ax.set_title(f"from {'the walk' if focus == 'walk' else 'Sophocles'} to each genre (windows)", fontsize=8)
        ax.tick_params(axis="x", labelsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(a.figures, "hard_genre_distance.png"), bbox_inches="tight")
    print(f"wrote {a.out}/genre_distance.csv, genre_topics.csv and {a.figures}/hard_genre_distance.png", flush=True)


if __name__ == "__main__":
    main()
