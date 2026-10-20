data {
    int<lower=1> N;
  int<lower=1> K; 
  matrix[N, K] logphi; // log phi[:, w_n] for each lemma
}
parameters { 
    simplex[K] theta; // lives in R^{K-1}
}
model {
  // prior
  theta ~ dirichlet(rep_vector(1, K)); // not necessary

  // likelihood
  vector[K] log_theta = log(theta);
  for (n in 1:N) 
    target += log_sum_exp(log_theta + logphi[n]');
}
