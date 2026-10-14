# BSDA mini-project 2026: what did the lost verses of Sophocles' *Tereus* talk about?

Albin Thörn Cleland, Lund University. Bayesian Statistics and Data Analysis, mini-project.

P.Oxy. 5292 preserves twelve intact iambic trimeters of Sophocles' *Tereus* and seventeen damaged ones (vv. 13–29), each with a few letters
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

The R Markdown reports download these files themselves if `data/` is missing.

- `vocab.json`: the $V = 3504$ content lemmata (nouns, verbs, adjectives, adverbs, proper names of three or more letters, three or more
  occurrences in the three tragedians, minus particles, light verbs and lemmata present in more than 25 % of Sophocles' 17-line windows).
  The lemmata come from a neural tagger, which damages some (λήσομενανθάνω, γείγνομαι, παήρ), so a lemma is kept only if it is
  well formed (Greek letters, exactly one accent, a breathing where Greek needs one, a final letter a Greek word can end in) and attested
  (an LSJ headword, or a word form of the Greek corpus for proper names and for the common words the Perseus LSJ lacks). The 674 lemmata
  that failed are listed with their reason in `vocab_dropped.json`. Each kept lemma carries its unsupervised topic label (`cluster_K40`, see the analytic model).
- `imputations.parquet`: 631,974 rows, one per restored verse. `verse` 13–29, `restorer` ∈ {stoicheia, candidates, walk}, the walk's
  `chain_set`, `chain`, `iteration` and `style_tilted` flag, the `text`, and the bag as `lemma_idx` + `count` over `vocab.json`.
- `corpus.parquet`: one row per extant trimeter of Sophocles, Aeschylus and Euripides, plus the intact papyrus lines and Henry's text.
- `hypotheses.json`: five seed lists for the supervised topics of the amortized mixed model.

`bsda/data.py` reads these into streams of lemma indices: Sophocles (the baseline), the three tragedians (training text for the topic-word
distributions), the untilted walk passages, and one stream per lacuna cell (verse × restorer). A **document** is 64 consecutive lemmata of a
stream; a cell's documents are cut from its shuffled stream, at most 1000 per cell.

## The amortized mixed model: soft topics, amortized posterior (`scripts/amortized_mixed_model.py`)

Topics are seeded into a latent Dirichlet allocation with strong word priors, plus four free topics. With `--topics hypotheses` the seeds are
the five philological hypotheses about the lost text (oath and disclosure, countryside and hunt, violence and death, women and marriage,
kin and names); with `--topics clusters` they are the 35 unsupervised clusters of the analytic model, each seeded with all its lemmata, so that the
same amortized workflow runs on unsupervised topics and the corpus decides how much a lemma leaks between clusters. The topic-word
distributions $\phi$ ($K \times V$) are fitted once on the three tragedians with tomotopy.

A document is then $n = 64$ lemmata drawn from the mixture $\sum_k \theta_k \phi_k$ with
$\theta \sim \mathrm{Dirichlet}(\mathbf 1)$, and the quantity of interest is $p(\theta \mid \text{document})$. Because a word enters the
likelihood only through its $K$ log-likelihoods $\log \phi_{\cdot w}$, a document is a set of $K$-vectors and a DeepSet summary network is
the natural compression; a coupling flow (BayesFlow) learns $\theta$ in additive log-ratio coordinates from 200,000 simulated documents. The
network is checked on 1000 held-out simulations (recovery $R^2$, 90 % interval coverage, simulation-based calibration) before it reads the
real documents. Each cell reports $P(\theta_h \text{ is the largest seeded topic})$ and the posterior mean of $\theta_h$ with a 90 % interval,
pooled over the cell's documents and draws, and, as in the analytic model, $P(\theta_k^{\text{cell}} > \theta_k^{\text{Sophocles}})$ over paired draws with the lift of
the posterior means. Results: `results/amortized_mixed/` (hypotheses) and `results/amortized_mixed_clusters/` (clusters).

## The analytic model: hard topics, exact posterior (`scripts/analytic_model.py`)

The supervised topics are biased towards existing readings. The analytic model replaces them with hard clusters of the vocabulary: lemmata are embedded
by word2vec (skip-gram over the whole lemmatised Greek corpus, Spearman 0.53 with the AGREE relatedness benchmark), the vocabulary is clustered
by Ward's method, and the cut $K = 40$ is chosen where AGREE's related pairs still co-cluster while random pairs no longer do. One cluster of
rare lemmata with unreliable vectors and four clusters of function words (particles, quantifiers, adverbs of time) are dropped, leaving
35 topics, each labelled by the English glosses of the two non-cognate lemmata nearest its centroid
(kill-destroy, marriage-bed, unholy-pollution, crash-wail, bird-beast, ...). That construction lives upstream; here the labels arrive in
`vocab.json` as `cluster_K40`, with the Greek pair beside them as `cluster_K40_lemmata` (wretched-miserable = δύστηνος-τλήμων).

With hard clusters a document is a bag of topic labels, $\mathbf n \sim \mathrm{Multinomial}(N, \theta)$, and with
$\theta \sim \mathrm{Dirichlet}(\alpha \mathbf 1)$ the posterior is conjugate and exact:

$$\theta \mid \mathbf n \sim \mathrm{Dirichlet}(\alpha \mathbf 1 + \mathbf n).$$

A cell's posterior is the equal mixture of its documents' posteriors; it is compared with Sophocles' extant trimeters by
$P(\theta_k^{\text{cell}} > \theta_k^{\text{Sophocles}})$ over paired draws and by the lift of posterior means. Results: `results/analytic/`.

## Run

```sh
python scripts/analytic_model.py --data data --out results/analytic        # seconds on a laptop
Rscript -e 'rmarkdown::render("analytic_model.Rmd")'                                  # the same model as a PDF report (xelatex); knit from the repo root
Rscript -e 'rmarkdown::render("amortized_mixed_model.Rmd")'                           # the GPU job's output as a PDF report
KERAS_BACKEND=torch python scripts/amortized_mixed_model.py --data data --out results/amortized_mixed   # a GPU hour; CPU dry run with --sims 3000 --epochs 2
KERAS_BACKEND=torch python scripts/amortized_mixed_model.py --topics clusters --out results/amortized_mixed_clusters
sbatch slurm/amortized_mixed_model.slurm --topics clusters --out results/amortized_mixed_clusters                                # the same on a GH200 (Arrhenius)
```

## Layout

```
bsda/data.py        the data loader (bags -> streams -> documents)
analytic_model.Rmd, amortized_mixed_model.Rmd   one PDF report per model (xelatex, a font with polytonic Greek)
scripts/            one Python script per model
slurm/              the GPU job for the amortized mixed model
results/amortized_mixed/     phi.json (topics), hypotheses.{md,csv,json}, sbc.csv, recovery.csv (five supervised topics)
results/amortized_mixed_clusters/  the same with the 35 clusters as seeded topics
results/analytic/     topics_K40.json (the clusters and their labels), posterior.{md,csv}
results/analytic_r/   the R Markdown report's posterior.csv (the PDF is compiled locally)
```
