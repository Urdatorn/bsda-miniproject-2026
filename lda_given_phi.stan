data {
  int<lower=1> N;
  int<lower=1> K; 
  matrix[N, K] logphi; // log phi[:, w_n] for each lemma
  real<lower=0> alpha; // concentration of the Dirichlet prior on theta: 1 is flat
}
parameters { 
  simplex[K] theta; // lives in R^{K-1}
}
model {
  // prior: flat at alpha = 1; 0.1 (sparse) and 2 (even) in the prior check
  theta ~ dirichlet(rep_vector(alpha, K));

  // likelihood
  vector[K] log_theta = log(theta);
  for (n in 1:N) 
    target += log_sum_exp(log_theta + logphi[n]');
}
