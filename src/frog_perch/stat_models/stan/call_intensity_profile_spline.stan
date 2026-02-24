data {
  // --- Dimensions ---
  int<lower=1> T;                 // Number of observations
  int<lower=1> N;                 // Max count (bins per window)
  
  // --- Observations ---
  matrix[T, N + 1] w_obs;         
  
  // --- Spline Basis Matrices ---
  int<lower=1> K_season;          
  int<lower=1> K_diel;
  
  matrix[T, K_season] B_season;   
  matrix[T, K_diel]   B_diel;     
}

transformed data {
  array[N + 1] int k_seq;
  for (i in 1:(N + 1)) {
    k_seq[i] = i - 1;
  }
}

parameters {
  real beta_0;
  vector[K_season] z_season;
  vector[K_diel]   z_diel;
  real<lower=0> sigma_season;
  real<lower=0> sigma_diel;
  real<lower=0> phi;
}

transformed parameters {
  vector[K_season] beta_season;
  vector[K_diel]   beta_diel;

  // 1. Season (RW1)
  beta_season[1] = z_season[1] * sigma_season; 
  for (k in 2:K_season) {
      beta_season[k] = beta_season[k-1] + z_season[k] * sigma_season;
  }
  
  // 2. Diel (RW1)
  beta_diel[1] = z_diel[1] * sigma_diel; 
  for (k in 2:K_diel) {
      beta_diel[k] = beta_diel[k-1] + z_diel[k] * sigma_diel;
  }

  // Calculate Rates
  vector[T] log_lambda = beta_0 + B_season * beta_season + B_diel * beta_diel;
  vector[T] lambda = exp(log_lambda);
}

model {
  beta_0 ~ normal(0, 2);
  sigma_season ~ exponential(1); 
  sigma_diel   ~ exponential(1);
  phi ~ normal(0, 5);
  z_season ~ std_normal();
  z_diel   ~ std_normal();

  for (t in 1:T) {
    vector[N + 1] log_probs;
    for (i in 1:(N + 1)) {
       real log_bio = neg_binomial_2_lpmf(k_seq[i] | lambda[t], phi);
       real log_det = log(w_obs[t, i] + 1e-15);
       log_probs[i] = log_bio + log_det;
    }
    target += log_sum_exp(log_probs);
  }
}

generated quantities {
  // We need these explicitly saved to plot the components separately
  vector[T] trend_season = B_season * beta_season;
  vector[T] trend_diel   = B_diel * beta_diel;
}