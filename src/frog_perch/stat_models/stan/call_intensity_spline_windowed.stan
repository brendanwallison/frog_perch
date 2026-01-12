data {
  int<lower=1> B;            
  int<lower=1> M_obs;        
  int<lower=1> D_obs;        

  int<lower=1> K_season;     
  int<lower=1> K_diel;       
  
  matrix[D_obs, K_season] X_season; 
  matrix[M_obs, K_diel]   X_diel;

  array[B] int<lower=1, upper=D_obs> day_idx;
  array[B] int<lower=1, upper=M_obs> diel_idx;

  vector[B] ell_bin;         
  array[B] int<lower=0> n_bin; 
}

parameters {
  // --- Seasonal Spline ---
  real mu_season;
  real<lower=0> tau_season;
  vector[K_season] beta_season_std;  // non-centered

  // --- Diel Spline ---
  real mu_diel;
  real<lower=0> tau_diel;
  vector[K_diel] beta_diel_std;      // non-centered

  // --- Day-level biological nugget ---
  real<lower=0> sigma_day_proc;
  vector[D_obs] u_day_std;           // non-centered

  // --- Minute-level nugget (small) ---
  real<lower=0> sigma_minute;
  vector[M_obs] u_minute_std;        // non-centered
}

transformed parameters {
  // --- Scale the non-centered splines ---
  vector[K_season] beta_season_raw = tau_season * beta_season_std;
  vector[K_diel]   beta_diel_raw   = tau_diel   * beta_diel_std;

  // --- Center spline weights ---
  vector[K_season] beta_season = beta_season_raw - mean(beta_season_raw);
  vector[K_diel]   beta_diel   = beta_diel_raw   - mean(beta_diel_raw);

  // --- Scale the non-centered nuggets ---
  vector[D_obs] u_day = sigma_day_proc * u_day_std;
  vector[M_obs] u_minute = sigma_minute * u_minute_std;

  // --- Reconstruct curves ---
  vector[D_obs] s_obs = mu_season + X_season * beta_season;
  vector[M_obs] g_obs = mu_diel   + X_diel   * beta_diel;

  vector[D_obs] p_season;
  vector[M_obs] p_diel;
  vector[B] q_bin;

  // Add day-level biological nugget (centered)
  vector[D_obs] u_day_centered = u_day - mean(u_day);
  for (d in 1:D_obs)
    p_season[d] = inv_logit(s_obs[d] + u_day_centered[d]);

  // Add minute-level nugget (centered)
  vector[M_obs] u_minute_centered = u_minute - mean(u_minute);
  p_diel = inv_logit(g_obs + u_minute_centered);

  // Combine seasonal & diel for bins
  for (b in 1:B)
    q_bin[b] = p_season[day_idx[b]] * p_diel[diel_idx[b]];
}

model {
  // --- Priors ---
  mu_season ~ normal(-3, 1);
  mu_diel   ~ normal(-3, 1);

  tau_season ~ normal(0, 0.5);  // moderate smoothing
  tau_diel   ~ normal(0, 0.5);

  sigma_day_proc ~ normal(0, 0.3);
  sigma_minute   ~ normal(0, 0.05);  // tight prior on minute-level

  // Standard normals for non-centered parameters
  beta_season_std ~ std_normal();
  beta_diel_std   ~ std_normal();
  u_day_std       ~ std_normal();
  u_minute_std    ~ std_normal();

  // --- RW2 spline penalties (non-centered)
  for (k in 3:K_season)
    target += normal_lpdf(beta_season_std[k] | 2*beta_season_std[k-1] - beta_season_std[k-2], 1);
  beta_season_std[1] ~ normal(0, 1);
  beta_season_std[2] ~ normal(0, 1);

  for (k in 3:K_diel)
    target += normal_lpdf(beta_diel_std[k] | 2*beta_diel_std[k-1] - beta_diel_std[k-2], 1);
  beta_diel_std[1] ~ normal(0, 1);
  beta_diel_std[2] ~ normal(0, 1);

  // --- Likelihood ---
  for (b in 1:B)
    target += n_bin[b] * log_sum_exp(
      log(q_bin[b]) + ell_bin[b],
      log1m(q_bin[b])
    );
}
