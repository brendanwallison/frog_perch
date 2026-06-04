data {
  int<lower=1> N;                          // number of slices
  int<lower=1> D;                          // number of days
  array[N] int<lower=1, upper=D> day;      // day index for each slice
  array[N] real t;                         // time-of-day covariate
  array[N] real<lower=0, upper=1> p;       // detector scores (p_i)

  // ✅ NEW: precomputed log-odds offsets
  vector[N] ell;
}

parameters {
  // Day-level logit intensities (random walk)
  vector[D] u;
  real<lower=0> sigma_day;

  // Time-of-day effect
  real gamma_0;
  real gamma_1;

  // Score distribution parameters (no longer used in likelihood)
  // but kept for backward compatibility
  real<lower=0> a_bg;
  real<lower=0> b_bg;
  real<lower=0> a_call;
  real<lower=0> b_call;
}

transformed parameters {
  vector[N] eta;
  vector[N] q;

  for (i in 1:N) {
    eta[i] = u[day[i]] + gamma_0 + gamma_1 * t[i];
    q[i] = inv_logit(eta[i]);
  }
}

model {
  // Priors -----------------------------------------------------

  // Day-level random walk
  u[1] ~ normal(-5, 1);
  for (d in 2:D)
    u[d] ~ normal(u[d - 1], sigma_day);

  sigma_day ~ exponential(1);

  // Time-of-day coefficients
  gamma_0 ~ normal(0, 2);
  gamma_1 ~ normal(0, 2);

  // Beta parameters (kept for compatibility)
  a_bg   ~ lognormal(-3, 0.5);
  b_bg   ~ lognormal(0, 0.5);
  a_call ~ lognormal(1.0, 0.5);
  b_call ~ lognormal(0, 0.5);

  // ------------------------------------------------------------
  // fast likelihood using precomputed log-odds ell[i]
  // ------------------------------------------------------------
  for (i in 1:N) {
    target += log(q[i] * exp(ell[i]) + (1 - q[i]));
  }

  // ------------------------------------------------------------
  // OLD mixture likelihood (commented out)
  // ------------------------------------------------------------
  /*
  for (i in 1:N) {
    target += log_mix(
      q[i],
      beta_lpdf(p[i] | a_call, b_call),
      beta_lpdf(p[i] | a_bg, b_bg)
    );
  }
  */
}

generated quantities {
  vector[N] z_prob;
  vector[D] day_intensity;

  // ✅ NEW: posterior P(Z_i=1 | p_i, q_i, ell_i)
  for (i in 1:N) {
    real odds = q[i] * exp(ell[i]);
    z_prob[i] = odds / (odds + (1 - q[i]));
  }

  // OLD method (commented out)
  /*
  for (i in 1:N) {
    real call_density = exp(beta_lpdf(p[i] | a_call, b_call));
    real bg_density   = exp(beta_lpdf(p[i] | a_bg, b_bg));
    z_prob[i] = (q[i] * call_density) /
                (q[i] * call_density + (1 - q[i]) * bg_density);
  }
  */

  // Daily average intensity (at t = 0)
  for (d in 1:D) {
    day_intensity[d] = inv_logit(u[d] + gamma_0);
  }
}