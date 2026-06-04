data {
  int<lower=1> B;
  int<lower=1> M_obs;
  int<lower=1> D_obs;

  int<lower=1> M_season;
  int<lower=1> M_diel;

  matrix[D_obs, 2*M_season] X_season;
  matrix[M_obs, 2*M_diel]   X_diel;

  array[B] int<lower=1, upper=D_obs> day_idx;
  array[B] int<lower=1, upper=M_obs> diel_idx;

  vector[B] ell_bin;
  array[B] int<lower=0> n_bin;
}

parameters {
  // Seasonal HSGP
  real mu_season;
  real<lower=0> alpha_season;
  real<lower=0> rho_season;
  vector[2*M_season] beta_season;

  // Diel HSGP
  real mu_diel;
  real<lower=0> alpha_diel;
  real<lower=0> rho_diel;
  vector[2*M_diel] beta_diel;

  // Observation noise
  real<lower=0> sigma_det;
}

transformed parameters {
  vector[2*M_season] diagSPD_season;
  vector[2*M_diel]   diagSPD_diel;

  {
    real term1 = square(alpha_season) * sqrt(2*pi()) * rho_season;
    for (m in 1:M_season) {
      real w = m;
      real S = term1 * exp(-0.5 * square(rho_season * w));
      diagSPD_season[2*m-1] = sqrt(S);
      diagSPD_season[2*m]   = sqrt(S);
    }
  }

  {
    real term1 = square(alpha_diel) * sqrt(2*pi()) * rho_diel;
    for (m in 1:M_diel) {
      real w = m;
      real S = term1 * exp(-0.5 * square(rho_diel * w));
      diagSPD_diel[2*m-1] = sqrt(S);
      diagSPD_diel[2*m]   = sqrt(S);
    }
  }

  vector[D_obs] s_smooth = mu_season + X_season * (beta_season .* diagSPD_season);
  vector[M_obs] g_smooth = mu_diel   + X_diel   * (beta_diel   .* diagSPD_diel);

  vector[D_obs] p_season = inv_logit(s_smooth);
  vector[M_obs] p_diel   = inv_logit(g_smooth);

  vector[B] q_bin_logit;
  for (b in 1:B)
    q_bin_logit[b] = logit(p_season[day_idx[b]] * p_diel[diel_idx[b]]);
}

model {
  // Priors
  mu_season ~ normal(0, 1);
  mu_diel   ~ normal(0, 1);

  alpha_season ~ normal(0, 1);
  alpha_diel   ~ normal(0, 1);

  rho_season ~ normal(0.3, 0.1);
  rho_diel   ~ normal(0.05, 0.1);

  beta_season ~ std_normal();
  beta_diel ~ std_normal();

  sigma_det ~ exponential(1);

  // Likelihood
  ell_bin ~ normal(q_bin_logit, sigma_det);
}
