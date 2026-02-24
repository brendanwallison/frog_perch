data {
  int<lower=1> B;            // number of bin observations
  int<lower=1> M_obs;        // number of unique minutes
  int<lower=1> D_obs;        // number of unique days

  int<lower=1> K_season;     
  int<lower=1> K_diel;       
  
  matrix[D_obs, K_season] X_season; // ignored (for drop-in)
  matrix[M_obs, K_diel]   X_diel;   // ignored (for drop-in)

  array[B] int<lower=1, upper=D_obs> day_idx;
  array[B] int<lower=1, upper=M_obs> diel_idx;

  vector[B] ell_bin;
  array[B] int<lower=0> n_bin;

  // New: input vectors for GP locations
  vector[D_obs] day_vals;      // e.g., day-of-year for each day
  vector[M_obs] min_vals;      // e.g., minute-of-day for each minute
}

parameters {
  real mu_season;
  real mu_diel;

  real<lower=0> sigma_day_proc;
  real<lower=0> rho_day;        // GP lengthscale for seasonal
  vector[D_obs] u_day_std;

  real<lower=0> sigma_minute;
  real<lower=0> rho_minute;     // GP lengthscale for diel
  vector[M_obs] u_minute_std;
}

transformed parameters {
  vector[D_obs] u_day;
  vector[M_obs] u_minute;
  vector[D_obs] p_season;
  vector[M_obs] p_diel;
  vector[B] q_bin;

  // --- GP Covariance for days ---
  matrix[D_obs, D_obs] K_day;
  for (i in 1:D_obs) {
    for (j in 1:D_obs) {
      K_day[i,j] = exp(-square(day_vals[i] - day_vals[j]) / (2 * square(rho_day)));
    }
  }
  K_day = K_day + diag_matrix(rep_vector(1e-6, D_obs)); // jitter
  u_day = sigma_day_proc * cholesky_decompose(K_day) * u_day_std;

  // --- GP Covariance for minutes ---
  matrix[M_obs, M_obs] K_min;
  for (i in 1:M_obs) {
    for (j in 1:M_obs) {
      K_min[i,j] = exp(-square(min_vals[i] - min_vals[j]) / (2 * square(rho_minute)));
    }
  }
  K_min = K_min + diag_matrix(rep_vector(1e-6, M_obs));
  u_minute = sigma_minute * cholesky_decompose(K_min) * u_minute_std;

  // --- Posterior probabilities ---
  for (d in 1:D_obs)
    p_season[d] = inv_logit(mu_season + u_day[d]);

  for (m in 1:M_obs)
    p_diel[m] = inv_logit(mu_diel + u_minute[m]);

  for (b in 1:B)
    q_bin[b] = p_season[day_idx[b]] * p_diel[diel_idx[b]];
}

model {
  // Priors
  mu_season ~ normal(-3, 1);
  mu_diel   ~ normal(-3, 1);

  sigma_day_proc ~ normal(0, 0.3);
  rho_day       ~ normal(0, 20);     // informative lengthscale prior
  u_day_std     ~ std_normal();

  sigma_minute ~ normal(0, 0.05);
  rho_minute   ~ normal(0, 10);      // informative lengthscale prior
  u_minute_std ~ std_normal();

  // Likelihood
  for (b in 1:B)
    target += n_bin[b] * log_sum_exp(
      log(q_bin[b]) + ell_bin[b],
      log1m(q_bin[b])
    );
}
