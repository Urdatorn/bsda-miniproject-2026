"""Baseline windows of content lemmata: Greek at large, and the tragedians.

A walk document is one restoration of the seventeen verses, on average 28
vocabulary lemmata.  To ask whether the walk's topics are unusual for Greek
in general, and whether Sophocles' are, every text is cut into windows of
the same length: consecutive vocabulary lemmata (the V content lemmata of
vocab.json, the same tagger and the same filter as the tragic corpus),
--window of them per window, within one work.  Two sources:

  oga         every work of Urdatorn/oga-conllu-stoicheia (the Open Greek
              and Latin texts parsed by the Stoicheia tagger-parser), one
              window set per author;
  tragedians  Sophocles, Aeschylus and Euripides from corpus.parquet, the
              extant trimeters in order (not the restorations).

Every window is stored as its counts over the topics (cluster_K40 of
vocab.json): baseline_windows.parquet with `source` (oga or tragedians),
`author`, `work`, `window` and one uint8 column per topic, plus
`n_lemmata`.  Lemmata without a topic are counted in `n_lemmata` only.
Beside it, baseline_lemma_counts.parquet counts every vocabulary lemma per
author (`source`, `author`, `lemma_idx`, `count`), for lemma-level plots.

  python scripts/window_baselines.py --data data --oga <dir of parquet shards> --out data

Logs one line per shard.
"""
import argparse, glob, json, os, sys, time, unicodedata
from collections import defaultdict

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CONTENT = {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}        # as in the dataset builder
TRAGEDIANS = ("sophocles", "aeschylus", "euripides")


def bare(s):
    """letters only, lowercased, final sigma unified (the dataset builder's rule)"""
    return "".join(c.lower() for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c).startswith("L")).replace("ς", "σ")


def lemma_of(lemma, form):
    return unicodedata.normalize("NFC", (lemma or form or "").lower())


def content_lemmata(conllu):
    """the content lemmata of one CoNLL-U sentence, in text order"""
    out = []
    for line in conllu.split("\n"):
        if not line or line[0] == "#":
            continue
        f = line.split("\t")
        if len(f) < 4 or "-" in f[0] or "." in f[0]:
            continue
        if f[3] in CONTENT and len(bare(f[1])) >= 3:
            out.append(lemma_of(f[2], f[1]))
    return out


def windows_of(topic_stream, window):
    """non-overlapping windows of `window` vocabulary lemmata; each is the list of topic ids (-1: no topic)"""
    n = len(topic_stream) // window
    return [topic_stream[i * window:(i + 1) * window] for i in range(n)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--oga", required=True, help="directory of the oga-conllu-stoicheia parquet shards")
    ap.add_argument("--out", default="data")
    ap.add_argument("--window", type=int, default=28, help="vocabulary lemmata per window (the walk's mean document)")
    a = ap.parse_args()
    t0 = time.time()

    vocab = json.load(open(os.path.join(a.data, "vocab.json"), encoding="utf-8"))
    topics = sorted({r["cluster_K40"] for r in vocab if r["cluster_K40"]})
    topic_id = {t: k for k, t in enumerate(topics)}
    lemma_index = {r["lemma"]: r["index"] for r in vocab}
    index_topic = {r["index"]: (topic_id[r["cluster_K40"]] if r["cluster_K40"] else -1) for r in vocab}
    lemma_counts = defaultdict(lambda: np.zeros(len(vocab), dtype=np.int64))   # (source, author) -> counts over the vocabulary
    print(f"vocabulary {len(vocab)} lemmata, {len(topics)} topics, windows of {a.window}", flush=True)

    rows = []            # (source, author, work, window number, topic stream)

    # the tragedians, from the modelling corpus (bags per trimeter, in line order)
    corpus = pq.read_table(os.path.join(a.data, "corpus.parquet")).to_pandas()
    for author in TRAGEDIANS:
        lines = corpus[corpus.source == author].sort_values("line")
        stream = []
        for idx, cnt in zip(lines.lemma_idx, lines["count"]):
            for i, c in zip(idx, cnt):
                stream.extend([index_topic[int(i)]] * int(c))
                lemma_counts[("tragedians", author)][int(i)] += int(c)
        for w, win in enumerate(windows_of(stream, a.window)):
            rows.append(("tragedians", author, author, w, win))
        print(f"tragedians: {author} {len(stream)} vocabulary lemmata, {len(stream) // a.window} windows", flush=True)

    # Greek at large: every shard, every work; a work's sentences are contiguous within a shard
    shards = sorted(glob.glob(os.path.join(a.oga, "*.parquet")))
    if not shards:
        sys.exit(f"no parquet shards in {a.oga}")
    streams = defaultdict(list)          # (author, work id) -> topic stream
    n_sent = n_tok = 0
    for s, shard in enumerate(shards):
        table = pq.read_table(shard, columns=["author", "id", "conllu"])
        for author, work, conllu in zip(table["author"].to_pylist(), table["id"].to_pylist(), table["conllu"].to_pylist()):
            lemmata = content_lemmata(conllu or "")
            n_sent += 1; n_tok += len(lemmata)
            indices = [lemma_index[l] for l in lemmata if l in lemma_index]
            streams[(author, work)].extend(index_topic[i] for i in indices)
            np.add.at(lemma_counts[("oga", author)], indices, 1)
        print(f"[{time.time() - t0:7.1f}s] shard {s + 1}/{len(shards)} {os.path.basename(shard)}: {n_sent} sentences, "
              f"{n_tok} content lemmata, {len(streams)} works so far", flush=True)
    n_oga = 0
    for (author, work), stream in streams.items():
        for w, win in enumerate(windows_of(stream, a.window)):
            rows.append(("oga", author, work, w, win)); n_oga += 1
    print(f"oga: {len(streams)} works, {sum(len(s) for s in streams.values())} vocabulary lemmata, {n_oga} windows", flush=True)

    # one uint8 column per topic
    K = len(topics)
    counts = np.zeros((len(rows), K), dtype=np.uint8)
    n_lemmata = np.zeros(len(rows), dtype=np.uint8)
    for r, (_, _, _, _, win) in enumerate(rows):
        for t in win:
            if t >= 0:
                counts[r, t] += 1
        n_lemmata[r] = len(win)
    table = {"source": pa.array([r[0] for r in rows]), "author": pa.array([r[1] for r in rows]),
             "work": pa.array([r[2] for r in rows]), "window": pa.array([r[3] for r in rows], pa.int32()),
             "n_lemmata": pa.array(n_lemmata)}
    for k, t in enumerate(topics):
        table[t] = pa.array(counts[:, k])
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "baseline_windows.parquet")
    pq.write_table(pa.table(table), path, compression="zstd")
    print(f"wrote {path}: {len(rows)} windows ({os.path.getsize(path) / 1e6:.1f} MB) in {time.time() - t0:.0f}s", flush=True)

    # the lemma counts per author, sparse
    source_col, author_col, idx_col, count_col = [], [], [], []
    for (source, author), c in lemma_counts.items():
        present = np.nonzero(c)[0]
        source_col += [source] * len(present); author_col += [author] * len(present)
        idx_col += present.tolist(); count_col += c[present].tolist()
    path = os.path.join(a.out, "baseline_lemma_counts.parquet")
    pq.write_table(pa.table({"source": pa.array(source_col), "author": pa.array(author_col),
                             "lemma_idx": pa.array(idx_col, pa.int32()), "count": pa.array(count_col, pa.int64())}),
                   path, compression="zstd")
    print(f"wrote {path}: {len(idx_col)} author-lemma rows ({os.path.getsize(path) / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
