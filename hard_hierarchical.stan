// The lift of the imputations over Sophocles with the chain as unit: each chain's
// summed topic counts, and each Sophoclean window's, are Dirichlet-multinomial
// around a corpus-level profile mu, with a concentration kappa for how much
// chains (or windows) vary around it.
functions {
  real dirmult_lpmf(array[] int n, vector a) {
    real a0 = sum(a);
    return lgamma(a0) - lgamma(sum(n) + a0) + sum(lgamma(to_vector(n) + a) - lgamma(a));
  }
}
data {
  int<lower=1> K;
  int<lower=1> C;
  int<lower=1> W;
  array[C, K] int<lower=0> n_chain;
  array[W, K] int<lower=0> n_window;
}
parameters {
  simplex[K] mu_imp;
  simplex[K] mu_soph;
  real log_kappa_imp;
  real log_kappa_soph;
}
model {
  mu_imp ~ dirichlet(rep_vector(1, K));
  mu_soph ~ dirichlet(rep_vector(1, K));
  log_kappa_imp ~ normal(5, 2);
  log_kappa_soph ~ normal(3, 2);
  for (c in 1:C) n_chain[c] ~ dirmult(exp(log_kappa_imp) * mu_imp);
  for (w in 1:W) n_window[w] ~ dirmult(exp(log_kappa_soph) * mu_soph);
}
generated quantities {
  vector[K] lift = mu_imp ./ mu_soph;
}
