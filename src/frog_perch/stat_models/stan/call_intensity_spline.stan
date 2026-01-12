data {
  int<lower=1> B;            // number of bins
  int<lower=1> M_obs;        // number of observed (day, minute) unique combos
  int<lower=1> D_obs;        // number of observed unique days

  // --- Spline Configuration ---
  int<lower=1> K_season;     // number of seasonal basis functions (e.g., 20)
  int<lower=1> K_diel;       // number of diel basis functions (e.g., 20)
  
  // Design Matrices: These map the basis weights to the OBSERVED times only.
  // Generate these in Python using patsy: dmatrix("cc(x, df=K)", ...)
  matrix[D_obs, K_season] X_season; 
  matrix[M_obs, K_diel]   X_diel;

  // --- Indices ---
  // We map bins directly to the ROWS of the design matrices above
  array[B] int<lower=1, upper=D_obs> day_idx;    // maps bin -> row in X_season
  array[B] int<lower=1, upper=M_obs> diel_idx;   // maps bin -> row in X_diel

  vector[B] ell_bin;         // mean ell in each bin
  array[B] int<lower=0> n_bin; // count in each bin
}

parameters {
  // --- Seasonal P-Spline ---
  real mu_season;
  real<lower=0> tau_season;
  vector[K_season] beta_season;  // The spline weights (small vector!)

  // --- Diel P-Spline ---
  real mu_diel;
  real<lower=0> tau_diel;
  vector[K_diel] beta_diel;      // The spline weights (small vector!)

  // --- Day-level nugget (Only for observed days) ---
  real<lower=0> sigma_day_proc;
  vector[D_obs] u_day;
}

transformed parameters {
  // Reconstruct the curve ONLY at observed locations
  vector[D_obs] s_obs = mu_season + X_season * beta_season;
  vector[M_obs] g_obs = mu_diel   + X_diel   * beta_diel;
  
  vector[D_obs] p_season;
  vector[M_obs] p_diel;
  vector[B] q_bin;

  // 1. Seasonal Probabilities (Base + Nugget)
  for (d in 1:D_obs) {
    // Note: s_obs is already on logit scale. Add nugget, then inverse logit.
    p_season[d] = inv_logit(s_obs[d] + u_day[d]);
  }

  // 2. Diel Probabilities
  p_diel = inv_logit(g_obs);

  // 3. Combine into Bins
  for (b in 1:B) {
    // We use the indices to look up the pre-calculated probabilities
    q_bin[b] = p_season[day_idx[b]] * p_diel[diel_idx[b]];
  }
}

model {
  // --- Priors ---
  mu_season ~ normal(-3, 1);
  mu_diel   ~ normal(-3, 1);
  
  // Smoothing Priors (Controls wiggle of the spline)
  tau_season ~ normal(0, 1); // Scale for weights
  tau_diel   ~ normal(0, 1);
  
  // --- P-Spline RW2 Penalties ---
  // Instead of penalizing adjacent time points, we penalize adjacent spline weights.
  // This enforces smoothness on the coefficients.
  
  // Seasonal Weights RW2 (Cyclic)
  for (k in 1:K_season) {
    int km1 = (k == 1) ? K_season : k - 1;
    int kp1 = (k == K_season) ? 1 : k + 1;
    // The weights are penalized, effectively smoothing the resulting curve
    beta_season[k] ~ normal((beta_season[km1] + beta_season[kp1]) / 2.0, tau_season);
  }
  
  // Diel Weights RW2 (Cyclic)
  for (k in 1:K_diel) {
    int km1 = (k == 1) ? K_diel : k - 1;
    int kp1 = (k == K_diel) ? 1 : k + 1;
    beta_diel[k] ~ normal((beta_diel[km1] + beta_diel[kp1]) / 2.0, tau_diel);
  }
  
  // Sum-to-zero constraint to separate intercept (mu) from shape (beta)
  sum(beta_season) ~ normal(0, 0.001 * K_season);
  sum(beta_diel)   ~ normal(0, 0.001 * K_diel);

  // --- Nugget ---
  sigma_day_proc ~ normal(0, 0.5);
  u_day ~ normal(0, sigma_day_proc);

  // --- Likelihood ---
  for (b in 1:B) {
    target += n_bin[b] * log_sum_exp(
      log(q_bin[b]) + ell_bin[b],
      log1m(q_bin[b])
    );
  }
}