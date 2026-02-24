functions {
  // Harmonic seasonal logit component: time-of-year s (in days, e.g. 1..365)
  // J = 1 harmonic
  real seasonal_logit_harmonic(
    real s,
    real beta0_year,
    vector beta_cos_year,
    vector beta_sin_year
  ) {
    real angle1 = 2 * pi() * s / 365;
    return beta0_year
           + beta_cos_year[1] * cos(angle1)
           + beta_sin_year[1] * sin(angle1);
  }

  // Harmonic diel logit component: time-of-day in hours (0..24)
  // K = 1 harmonic
  real diel_logit_harmonic(
    real t_hours,
    real alpha0_tod,
    vector alpha_cos_tod,
    vector alpha_sin_tod
  ) {
    real angle1 = 2 * pi() * t_hours / 24;
    return alpha0_tod
           + alpha_cos_tod[1] * cos(angle1)
           + alpha_sin_tod[1] * sin(angle1);
  }
}

data {
  int<lower=1> B;                 // number of bins
  int<lower=1> D;                 // number of days
  int<lower=1> M_obs;             // number of observed (day, minute) combos

  array[B] int<lower=1, upper=D> day_id;
  array[B] int<lower=1, upper=M_obs> dm_index;
  vector[B] ell_bin;
  array[B] int<lower=1> n_bin;

  vector[D] day_of_year;
  vector[M_obs] t_minutes;        // minute-of-day (0..1439)
}

parameters {
  // --- Harmonic coefficients for seasonal logit ---
  real beta0_year;
  vector[1] beta_cos_year;
  vector[1] beta_sin_year;

  // --- Harmonic coefficients for diel logit ---
  real alpha0_tod;
  vector[1] alpha_cos_tod;
  vector[1] alpha_sin_tod;

  // --- Day-level deviations (non-centered) ---
  vector[D] eta_day_raw;
  real<lower=0> sigma_day_dev;

  // --- Minute-of-day deviations (non-centered, invariant across days) ---
  vector[1440] delta_minute_raw;   // one per minute-of-day
  real<lower=0> sigma_min_dev;
}

transformed parameters {
  vector[D] seasonal_logit;
  vector[D] p_season;

  vector[M_obs] diel_logit;
  vector[M_obs] p_diel;

  vector[B] q_bin;

  vector[D] eta_day;
  vector[1440] delta_minute;       // centered minute-of-day effects
  vector[M_obs] delta_min;         // mapped to each observation

  // Non-centered random effects
  eta_day     = sigma_day_dev  * eta_day_raw;
  delta_minute = sigma_min_dev * delta_minute_raw;

  // Seasonal component
  for (d in 1:D) {
    real s = day_of_year[d];
    real s_harm = seasonal_logit_harmonic(
      s,
      beta0_year,
      beta_cos_year,
      beta_sin_year
    );
    seasonal_logit[d] = s_harm + eta_day[d];
    p_season[d]       = inv_logit(seasonal_logit[d]);
  }

  // Diel component (minute-of-day invariant across days)
  for (m in 1:M_obs) {
    int minute = to_int(t_minutes[m]) + 1;
    real t_hours = t_minutes[m] / 60.0;

    real g_harm = diel_logit_harmonic(
      t_hours,
      alpha0_tod,
      alpha_cos_tod,
      alpha_sin_tod
    );

    diel_logit[m] = g_harm + delta_minute[minute];
    p_diel[m]     = inv_logit(diel_logit[m]);
  }

  // Product structure
  for (b in 1:B) {
    int d = day_id[b];
    int idx_dm = dm_index[b];
    q_bin[b] = p_season[d] * p_diel[idx_dm];
  }
}

model {
  // Seasonal priors
  beta0_year    ~ normal(-2, 2);
  beta_cos_year ~ normal(0, 0.7);
  beta_sin_year ~ normal(0, 0.7);

  // Diel priors
  alpha0_tod    ~ normal(-2, 2);
  alpha_cos_tod ~ normal(0, 0.5);
  alpha_sin_tod ~ normal(0, 0.5);

  // Day deviations
  eta_day_raw   ~ normal(0, 1);
  sigma_day_dev ~ normal(0, 0.1) T[0,];

  // Minute-of-day deviations (invariant across days)
  delta_minute_raw ~ normal(0, 1);
  sigma_min_dev    ~ normal(0, 0.1) T[0,];

  // Likelihood
  for (b in 1:B) {
    target += n_bin[b] * log(q_bin[b] * exp(ell_bin[b]) + (1 - q_bin[b]));
  }
}

generated quantities {
  // Empty — curves reconstructed in Python
}