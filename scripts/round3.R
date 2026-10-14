# Round 3 in R: unsupervised topics (hard clusters of the vocabulary), exact conjugate posterior.
# A document is n = 64 topic-bearing lemmata; counts n ~ Multinomial(N, theta), theta ~ Dirichlet(alpha 1),
# so theta | n ~ Dirichlet(alpha 1 + n) exactly.  A cell (lacuna x restorer) is the equal mixture of its
# documents' posteriors and is compared with Sophocles by P(theta_k^cell > theta_k^Soph) over paired draws.
#   Rscript scripts/round3.R data results/round3_r
suppressMessages({library(arrow); library(dplyr); library(jsonlite)})
a <- commandArgs(TRUE); data <- if (length(a) >= 1) a[1] else "data"; out <- if (length(a) >= 2) a[2] else "results/round3_r"
n <- 64; alpha <- 1; draws <- 100; max_docs <- 1000; set.seed(917); dir.create(out, showWarnings = FALSE, recursive = TRUE)

vocab  <- fromJSON(file.path(data, "vocab.json"))                       # index, lemma, tragic_count, cluster_K40 (NA = no topic)
corpus <- read_parquet(file.path(data, "corpus.parquet")); imp <- read_parquet(file.path(data, "imputations.parquet"))
stream <- function(tab) rep(unlist(tab$lemma_idx), unlist(tab$count))    # bags -> lemma indices (0-based), counts expanded
topics <- function(s) { t <- vocab$cluster_K40[s + 1]; t[!is.na(t)] }    # lemma indices -> topic labels, dropping topicless lemmata
tragic <- filter(corpus, source %in% c("sophocles", "aeschylus", "euripides"))
names_ <- names(sort(table(topics(stream(tragic))), decreasing = TRUE)); K <- length(names_)   # topics by mass in the tragic corpus

docs <- function(s, limit, shuffle = FALSE) {                             # cut a topic stream into documents of exactly n labels
  x <- topics(s); if (shuffle) x <- sample(x); m <- length(x) %/% n; if (m == 0) return(list())
  d <- split(x[seq_len(m * n)], rep(seq_len(m), each = n)); if (m > limit) d <- d[sort(sample(m, limit))]; d }
rdirichlet <- function(S, par) { g <- matrix(rgamma(S * length(par), par), S, byrow = TRUE); g / rowSums(g) }
posterior <- function(d) do.call(rbind, lapply(d, function(x) rdirichlet(draws, alpha + tabulate(match(x, names_), K))))   # the exact posterior, mixed over documents
summarise_cell <- function(d, base = NULL) {
  if (!length(d)) return(NULL); P <- posterior(d); s <- list(n_docs = length(d), mean = colMeans(P), P = P)
  if (!is.null(base)) { B <- base[sample(nrow(base), nrow(P), replace = TRUE), ]; s$p_above <- colMeans(P > B); s$lift <- colMeans(P) / colMeans(base) }
  s }

soph <- summarise_cell(docs(stream(filter(corpus, source == "sophocles")), 4 * max_docs))
cells <- list(walk = summarise_cell(docs(stream(filter(imp, restorer == "walk", !coalesce(style_tilted, FALSE))), 2 * max_docs), soph$P),
              henry = summarise_cell(list(topics(stream(arrange(filter(corpus, source %in% c("intact", "henry")), line)))), soph$P))
for (v in 13:29) for (r in c("stoicheia", "candidates", "walk"))
  cells[[paste(v, r)]] <- summarise_cell(docs(stream(filter(imp, verse == v, restorer == r, !coalesce(style_tilted, FALSE))), max_docs, shuffle = TRUE), soph$P)

rows <- bind_rows(lapply(names(cells), function(cl) { s <- cells[[cl]]; if (is.null(s)) return(NULL)
  tibble(cell = cl, topic = names_, n_docs = s$n_docs, mean = s$mean, p_above_sophocles = s$p_above, lift = s$lift) }))
write.csv(rows, file.path(out, "posterior.csv"), row.names = FALSE)
raised <- rows %>% filter(p_above_sophocles >= 0.9) %>% arrange(cell, desc(lift)) %>% group_by(cell, n_docs) %>%
  summarise(raised = paste0("**", topic, "** (", sprintf("%.1f", lift), "×, ", sprintf("%.2f", mean), ")", collapse = ", "), .groups = "drop")
md <- c("# Round 3 (R): unsupervised topics, closed-form posterior", "", sprintf("%d topics; documents of %d lemmata; theta ~ Dirichlet(%g); %d draws per document; Sophocles baseline of %d documents.", K, n, alpha, draws, soph$n_docs), "",
        "| cell | docs | topics raised (P >= 0.9 above Sophocles; lift, posterior mean) |", "|---|---|---|", sprintf("| %s | %d | %s |", raised$cell, raised$n_docs, raised$raised))
writeLines(md, file.path(out, "posterior.md")); cat(md, sep = "\n")
