"""The prepared data: bags of lemmata from the Hugging Face dataset
Urdatorn/bsda-miniproject-2026, read into the streams the two models consume.

Everything upstream of the bags (candidate generation, Markov walks, parsing) is a
black box here.  A bag is a sparse count vector over the V content lemmata of
vocab.json; a document for the models is 64 lemmata, cut from a stream of bags.

    data = load("data")
    data.vocab          list of lemmata, index = column of the bag
    data.cluster        list, the unsupervised topic (label) of each lemma or None
    data.streams        {"real": Sophocles' extant trimeters, "tragic": all three
                        tragedians, "walk": the untilted walk passages,
                        (verse, restorer): the fills of one lacuna cell}
                        each a list of lemma indices in reading order
    data.henry          the papyrus as Henry reads it (intact vv. 1-12, his 13-29)
    data.hypotheses     the five seed lists of the supervised topics
"""
import json, os
from dataclasses import dataclass
import numpy as np, pyarrow.parquet as pq

FIRST, LAST = 13, 29
RESTORERS = ("stoicheia", "candidates", "walk")


def expand(table):
    """every bag of a table, as the flat list of its lemma indices (counts expanded)"""
    out = []
    for ids, cnt in zip(table["lemma_idx"].to_pylist(), table["count"].to_pylist()):
        out.extend(np.repeat(ids, cnt).tolist())
    return out


@dataclass
class Data:
    vocab: list; cluster: list; streams: dict; henry: list; hypotheses: dict

    @property
    def V(self): return len(self.vocab)


def load(data_dir="data"):
    vocab = json.load(open(os.path.join(data_dir, "vocab.json"), encoding="utf-8"))
    corpus = pq.read_table(os.path.join(data_dir, "corpus.parquet"))
    imp = pq.read_table(os.path.join(data_dir, "imputations.parquet"))
    src = np.array(corpus["source"].to_pylist()); line = np.array(corpus["line"].to_pylist())
    streams = {"real": expand(corpus.filter(src == "sophocles")),
               "tragic": expand(corpus.filter(np.isin(src, ["sophocles", "aeschylus", "euripides"])))}
    restorer = np.array(imp["restorer"].to_pylist()); verse = np.array(imp["verse"].to_pylist())
    tilted = np.array([bool(t) for t in imp["style_tilted"].to_pylist()])
    streams["walk"] = expand(imp.filter((restorer == "walk") & ~tilted))
    for v in range(FIRST, LAST + 1):
        for r in RESTORERS:
            streams[(v, r)] = expand(imp.filter((verse == v) & (restorer == r) & ~tilted))
    order = np.argsort(line)
    intact = corpus.filter(src == "intact"); henry = corpus.filter(src == "henry")
    henry_doc = expand(intact.take(np.argsort(intact["line"].to_pylist()))) + expand(henry.take(np.argsort(henry["line"].to_pylist())))
    hyps = json.load(open(os.path.join(data_dir, "hypotheses.json"), encoding="utf-8"))
    return Data([r["lemma"] for r in vocab], [r["cluster_K40"] for r in vocab], streams, henry_doc, hyps)


def documents(stream, n=64, limit=None, shuffle=False, rng=None):
    """cut a stream into documents of exactly n lemmata; at most `limit`, sampled; shuffled first if asked"""
    ids = list(stream)
    if shuffle: ids = list(rng.permutation(ids))
    docs = [ids[i:i + n] for i in range(0, len(ids) - n + 1, n)]
    if limit and len(docs) > limit: docs = [docs[i] for i in sorted(rng.choice(len(docs), limit, replace=False))]
    return docs
