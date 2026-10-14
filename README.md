# BSDA mini-project 2026: what did the lost verses of Sophocles' *Tereus* talk about?

Albin Thörn Cleland, Lund University. Bayesian Statistics and Data Analysis, mini-project.

P.Oxy. 3013 preserves twelve intact iambic trimeters of Sophocles' *Tereus* and seventeen damaged ones (vv. 13–29), each with a few letters
surviving at the start of the line. Many restorations of the damaged verses exist: a philologist's (W. B. Henry), the Stoicheia fill-in model's
candidates, and the states of judge-steered Markov walks over metrically valid Greek. This repository asks one question of all of them: **which
topics does each restorer put into each lost verse, and does that differ from what Sophocles writes elsewhere?**

The generation of the restorations is a black box here. Its output, one bag of content lemmata per restored verse, is the dataset
[Urdatorn/bsda-miniproject-2026](https://huggingface.co/datasets/Urdatorn/bsda-miniproject-2026) (see `Data` below). The generation itself,
with all its code and records, is [Urdatorn/tereus-imputation](https://github.com/Urdatorn/tereus-imputation) on GitHub and
[Urdatorn/tereus-imputation](https://huggingface.co/datasets/Urdatorn/tereus-imputation) on the Hub.

## Data

```sh
pip install -r requirements.txt
hf download Urdatorn/bsda-miniproject-2026 --repo-type dataset --local-dir data
```

- `vocab.json`: the $V = 4180$ content lemmata (nouns, verbs, adjectives, adverbs, proper names of three or more letters, three or more
  occurrences in the three tragedians, minus particles, light verbs and lemmata present in more than 25 % of Sophocles' 17-line windows).
  Each lemma carries its unsupervised topic label (`cluster_K40`, see Round 3).
- `imputations.parquet`: 631,974 rows, one per restored verse. `verse` 13–29, `restorer` ∈ {stoicheia, candidates, walk}, the walk's
  `chain_set`, `chain`, `iteration` and `style_tilted` flag, the `text`, and the bag as `lemma_idx` + `count` over `vocab.json`.
- `corpus.parquet`: one row per extant trimeter of Sophocles, Aeschylus and Euripides, plus the intact papyrus lines and Henry's text.
- `hypotheses.json`: five seed lists for the supervised topics of Round 2.

`bsda/data.py` reads these into streams of lemma indices: Sophocles (the baseline), the three tragedians (training text for the topic-word
distributions), the untilted walk passages, and one stream per lacuna cell (verse × restorer). A **document** is 64 consecutive lemmata of a
stream; a cell's documents are cut from its shuffled stream, at most 1000 per cell.

## Round 2: supervised topics, amortized posterior (`scripts/round2_seeded_lda_bayesflow.py`)

Topics are the five philological hypotheses about the lost text (oath and disclosure, countryside and hunt, violence and death, women and
marriage, kin and names), each seeded into a latent Dirichlet allocation with strong word priors, plus four free topics. The topic-word
distributions $\phi$ ($K \times V$) are fitted once on the three tragedians with tomotopy.

A document is then $n = 64$ lemmata drawn from the mixture $\sum_k \theta_k \phi_k$ with
$\theta \sim \mathrm{Dirichlet}(\mathbf 1)$, and the quantity of interest is $p(\theta \mid \text{document})$. Because a word enters the
likelihood only through its $K$ log-likelihoods $\log \phi_{\cdot w}$, a document is a set of $K$-vectors and a DeepSet summary network is
the natural compression; a coupling flow (BayesFlow) learns $\theta$ in additive log-ratio coordinates from 200,000 simulated documents. The
network is checked on 1000 held-out simulations (recovery $R^2$, 90 % interval coverage, simulation-based calibration) before it reads the
real documents. Each cell reports $P(\theta_h \text{ is the largest seeded topic})$ and the posterior mean of $\theta_h$ with a 90 % interval,
pooled over the cell's documents and draws. Results: `results/round2/`.

## Round 3: unsupervised topics, exact posterior (`scripts/round3_cluster_posterior.py`)

The supervised topics are biased towards existing readings. Round 3 replaces them with hard clusters of the vocabulary: lemmata are embedded
by word2vec (skip-gram over the whole lemmatised Greek corpus, Spearman 0.53 with the AGREE relatedness benchmark), the vocabulary is clustered
by Ward's method, and the cut $K = 40$ is chosen where AGREE's related pairs still co-cluster while random pairs no longer do. One cluster of
tagger noise is dropped, leaving 39 topics, each labelled by the two non-cognate lemmata nearest its centroid (ἀπόλλυμι-ἀναιρέω,
γάμος-λέχος, ἀνόσιος-μίασμα, κτύπος-γόος, ὄρνις-θήρ, ...). That construction lives upstream; here the labels arrive in `vocab.json`.

With hard clusters a document is a bag of topic labels, $\mathbf n \sim \mathrm{Multinomial}(N, \theta)$, and with
$\theta \sim \mathrm{Dirichlet}(\alpha \mathbf 1)$ the posterior is conjugate and exact:

$$\theta \mid \mathbf n \sim \mathrm{Dirichlet}(\alpha \mathbf 1 + \mathbf n).$$

A cell's posterior is the equal mixture of its documents' posteriors; it is compared with Sophocles' extant trimeters by
$P(\theta_k^{\text{cell}} > \theta_k^{\text{Sophocles}})$ over paired draws and by the lift of posterior means. Results: `results/round3/`.

## Run

```sh
python scripts/round3_cluster_posterior.py --data data --out results/round3        # seconds on a laptop
KERAS_BACKEND=torch python scripts/round2_seeded_lda_bayesflow.py --data data --out results/round2   # a GPU hour; CPU dry run with --sims 3000 --epochs 2
```

## Layout

```
bsda/data.py        the data loader (bags -> streams -> documents)
scripts/            one script per round
results/round2/     phi.json (topics), hypotheses.{md,csv,json}, sbc.csv, recovery.csv
results/round3/     topics_K40.json (the clusters and their labels), posterior.{md,csv,json}
```
