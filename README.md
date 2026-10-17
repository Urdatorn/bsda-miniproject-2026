# BSDA mini-project 2026: what did the lost verses of Sophocles' *Tereus* talk about?

Albin Thörn Cleland, Lund University. Bayesian Statistics and Data Analysis, mini-project.

P.Oxy. 5292 preserves twelve intact iambic trimeters of Sophocles' *Tereus* and seventeen damaged ones (vv. 13–29), each with a few letters
surviving at the start of the line. Two restorations of the whole passage exist: a philologist's (W. B. Henry) and the
states of judge-steered Markov walks over metrically valid Greek. This repository asks one question of them: **which topics does the walk put
into the lost passage, and does that differ from what Sophocles writes elsewhere and from what Henry wrote?** (The dataset also holds the
one-verse fills of two older generators, Stoicheia and the candidates; they are not modelled here.)

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

- `vocab.json`: the $V = 3228$ content lemmata (nouns, verbs, adjectives and adverbs of three or more letters, three or more
  occurrences in the three tragedians, minus a stop list of particles, pronouns and light verbs). There is no frequency cut: a word in
  half of all scenes is still informative through its rate, and function words belong on the stop list. Proper names (a lemma
  capitalised in most of its tragic tokens: gods, heroes, places, ethnics) are dropped, since a name is a topic only in the sense that
  Frodo is the topic of a novel, and names cluster by play, not by theme; the one exception is Ἕλλην with its cognates, which in Greek
  means "not barbarian" rather than naming anyone.
  The lemmata come from a neural tagger, which damages some (λήσομενανθάνω, γείγνομαι, παήρ), so a lemma is kept only if it is
  well formed (Greek letters, exactly one accent, a breathing where Greek needs one, a final letter a Greek word can end in) and attested
  (an LSJ headword, or a word form of the Greek corpus for the common words the Perseus LSJ lacks). The 963 lemmata that failed
  one of the tests (227 malformed, 306 names, 430 unattested) are listed with their reason in `vocab_dropped.json`. Each kept lemma carries its unsupervised topic label (`cluster_K40`, see the analytic model).
- `imputations.parquet`: 631,974 rows, one per restored verse. `verse` 13–29, `restorer` ∈ {stoicheia, candidates, walk}, the walk's
  `chain_set`, `chain`, `iteration` and `style_tilted` flag, the `text`, and the bag as `lemma_idx` + `count` over `vocab.json`.
- `corpus.parquet`: one row per extant trimeter of Sophocles, Aeschylus and Euripides, plus the intact papyrus lines and Henry's text.

## Documents and cells

The walk is the object of study; the two one-verse generators are baselines. A **document** is one whole restoration of the seventeen lost
verses, all its lemmata, however many there are and nothing cut off. The cells are

- **walk**: one document per untilted snapshot (one chain at one step, all seventeen verses), 19,922 of them;
- **henry**: Henry's seventeen restored verses, one document;
- **sophocles**, the reference: every window of seventeen consecutive extant trimeters of Sophocles, 444 of them, the same shape as the lacuna.

Only restored text enters a document: the surviving letters of a damaged verse are never counted, for Henry as for the generators. Every cell is
read on two levels. Its **mean profile** is the documents' posteriors averaged draw by draw, compared with Sophocles' by
$P(\bar\theta_k^{\text{cell}} > \bar\theta_k^{\text{Soph}})$ and by the lift of the means; with thousands of restorations it is sharp, so the lift
is the effect size. A **random document** against a random window, $P(\theta_k^{\text{doc}} > \theta_k^{\text{window}})$, says how much the two
populations overlap. A topic counts as raised when the mean-profile probability is at least 0.9 and the lift at least 1.5.

## The amortized mixed model: soft topics, amortized posterior (`scripts/amortized_mixed_model.py`)

The 34 unsupervised clusters of the analytic model are seeded into a latent Dirichlet allocation with strong word priors, each cluster
seeding one topic with all its lemmata, plus four free topics, so that the corpus decides how much a lemma leaks between clusters. The topic-word
distributions $\phi$ ($K \times V$) are fitted once on the three tragedians with tomotopy.

A document's $N$ lemmata are drawn from the mixture $\sum_k \theta_k \phi_k$ with
$\theta \sim \mathrm{Dirichlet}(\mathbf 1)$, and the quantity of interest is $p(\theta \mid \text{document})$. Because a word enters the
likelihood only through its $K$ log-likelihoods $\log \phi_{\cdot w}$, a document is a set of $K$-vectors and a DeepSet summary network is
the natural compression; documents differ in length, so the set is padded and the network pools over the real elements only, with $\log N$ as
an extra summary. A coupling flow (BayesFlow) learns $\theta$ in additive log-ratio coordinates from 200,000 simulated documents whose lengths
are drawn from the real documents' lengths. The network is checked on 1000 held-out simulations (recovery $R^2$, 90 % interval coverage,
simulation-based calibration) before it reads the real documents. Each cell reports the mean profile with a 90 % interval, the comparison with
Sophocles on both levels, $P(\theta_h \text{ is the largest seeded topic})$ for a random document, and the mass on the free topics.
Results: `results/amortized_mixed/`.

## The analytic model: hard topics, exact posterior (`analytic_model.Rmd`)

The topics are hard clusters of the vocabulary: lemmata are embedded
by word2vec (skip-gram over the whole lemmatised Greek corpus, Spearman 0.53 with the AGREE relatedness benchmark), the vocabulary is clustered
by Ward's method, and the cut $K = 40$ is chosen where AGREE's related pairs still co-cluster while random pairs no longer do. Two clusters of
rare lemmata with unreliable vectors (median count in the whole Greek corpus below 20) and four clusters of function words (particles,
adverbs of time, position words, ordinals) are dropped, leaving 34 topics, each labelled by the English glosses of the three non-cognate
lemmata nearest its centroid (lament-pitiable-groan, marriage-couch-bed, temple-offer-precinct, ship-sail-hull, Greek-sea-barbarian, ...). That construction lives upstream; here the labels arrive in
`vocab.json` as `cluster_K40`, with the Greek triple beside them as `cluster_K40_lemmata` (man-friend-citizen = ἀνήρ-φίλος-πολίτης).

With hard clusters a document is a bag of topic labels, $\mathbf n \sim \mathrm{Multinomial}(N, \theta)$, and with
$\theta \sim \mathrm{Dirichlet}(\kappa \mathbf m)$, $\mathbf m$ Sophocles' own topic profile and $\kappa = 1$ one lemma's worth of it, the
posterior is conjugate and exact:

$$\theta \mid \mathbf n \sim \mathrm{Dirichlet}(\kappa \mathbf m + \mathbf n).$$

The prior is centred on Sophocles rather than symmetric because documents differ in length, and a symmetric prior pulls short documents harder
towards the uniform profile, which fabricates lift for topics that are simply absent. Results: `results/analytic/`.

Is the walk's topic profile an artefact of where the walk went? The report checks it two ways: the lift of every topic in windows of steps
(all 120 chains over steps 525–3000, the 24 long chains to 12000) with its slope per 1000 steps, and the Gelman–Rubin $\hat R$ of each topic's
share across the chains, written to `results/analytic/walk_stability.csv`. Drift is negligible and every topic but *ship-sail-hull* has
$\hat R \le 1.04$; *ship-sail-hull* ($\hat R = 1.61$) lives in some chains only.

Both reports also draw the posterior on the 2-simplex of the three topics the walk lifts most (the view of Blei, Ng and Jordan's
figure 2): for the walk, Henry and Sophocles' windows, the posterior of the cell's mean profile restricted to that triangle, drawn as
a kernel density of the posterior draws (the analytic model's own, the saved `profile_draws.csv` in the amortized one) with each cell
scaled to a peak of one. The walk and Sophocles are sharp peaks close together, because similar lifts leave the ratios of the three
topics unchanged; Henry, one document, is a broad hump.

## Maps

Two interactive maps (`datamapplot`, self-contained HTML files in `results/analytic/`, a static PNG beside each):

- `lemma_map.html` (`scripts/lemma_map.py`): the 2,531 topic lemmata placed by UMAP from their word2vec vectors, the space the topics
  were clustered in, so the topics are the regions of the map; a point's size is the walk's use of the word and a topic label's size
  is the walk's share of the topic (so the labels that outlast zooming out are the walk's largest topics), hovering gives its rate per
  thousand topic lemmata in Sophocles and in the walk, the search box finds a lemma, and the alternative
  colouring shows the walk's rate against Sophocles' (log2 ratio).
- `topic_map.html` (`scripts/topic_map.py`): 4,000 walk snapshots, Henry and the 443 Sophoclean windows placed by UMAP from their
  profiles over the 34 topics (Hellinger distance), coloured by dominant topic, the Greek text on hover and searchable. Documents of
  twenty lemmata are mixtures, so this map has no clear regions: it is kept for browsing the restorations, not as evidence.

## Run

```sh
Rscript -e 'rmarkdown::render("analytic_model.Rmd")'                                  # the whole analytic model, seconds on a laptop; knit from the repo root
Rscript -e 'rmarkdown::render("amortized_mixed_model.Rmd")'                           # the GPU job's output as a PDF report
KERAS_BACKEND=torch python scripts/amortized_mixed_model.py --data data --out results/amortized_mixed   # a GPU hour; CPU dry run with --sims 3000 --epochs 2
sbatch slurm/amortized_mixed_model.slurm                                                        # the same on a GH200 (Arrhenius)
python scripts/lemma_map.py --data data --out results/analytic      # the interactive maps (datamapplot, umap-learn); topic_map.py likewise
```

## Layout

```
analytic_model.Rmd, amortized_mixed_model.Rmd   one PDF report per model (xelatex, a font with polytonic Greek)
scripts/            the GPU script of the amortized mixed model; topics_table.py writes the LaTeX table of the topics;
                    lemma_map.py and topic_map.py draw the interactive maps
slurm/              the GPU job for the amortized mixed model
results/amortized_mixed/     phi.json (topics), cells.{md,csv,json}, sbc.csv, recovery.csv, profile_draws.csv and theta_samples.csv
                             (posterior samples for the simplex plots)
results/analytic/     topics_K40.json (the clusters and their labels), topics_table.tex, posterior.csv, walk_stability.csv and
                      topic_variability.csv from the Rmd; lemma_map.{html,png,_coords.csv} and topic_map.{html,png,_coords.csv}
```
