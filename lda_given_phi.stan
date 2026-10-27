data {
  int<lower=1> N;
  int<lower=1> K;
  matrix[N, K] logphi;      // log phi[:, w_n] for each lemma
  vector<lower=0>[K] alpha; // the Dirichlet prior on theta: rep_vector(1, K) is flat,
                            // the LDA's own estimate is sparse on the seeded topics
}
parameters {
  simplex[K] theta; // lives in R^{K-1}
}
model {
  // prior (sampled at adapt_delta 0.99: with alpha_k < 1 the density is
  // unbounded at the corners of the simplex and 0.8 leaves divergences)
  theta ~ dirichlet(alpha);

  // likelihood
  vector[K] log_theta = log(theta);
  for (n in 1:N)
    target += log_sum_exp(log_theta + logphi[n]');
}
