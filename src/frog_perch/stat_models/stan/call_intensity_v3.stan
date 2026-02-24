functions {
  // No helper functions needed — RW2 penalties written explicitly.
}

data {
  int<lower=1> B;                      // number of bins
  int<lower=1> M_obs;                  // number of observed (day, minute) combos

  int<lower=1> D_period;               // full seasonal period (365)
  int<lower=1> N_diel_period;          // full diel period (1440)

  // For each bin: true day-of-year (1..365)
  array[B] int<lower=1, upper=D_period> day_id;

  // For each observed combo: true minute-of-day (0..1439)
  array[M_obs] int<lower=0, upper=N_diel_period-1> t_minutes;

  // For each bin: which observed (day, minute) combo?
  array[B] int<lower=1, upper=M_obs> dm_index;

  vector[B] ell_bin;                   // mean ell in each bin
  array[B] int<lower=0> n_bin;         // count in each bin
}

parameters {
  // --- Seasonal RW2 (periodic, intrinsic) over full 365-day cycle ---
  real mu_season;
  real<lower=0> tau_season;
  vector[D_period] s_raw;

  // --- Diel RW2 (periodic, intrinsic) over full 1440-minute cycle ---
  real mu_diel;
  real<lower=0> tau_diel;
  vector[N_diel_period] g_raw;

  // --- Day-level nugget (over full 365-day cycle) ---
  real<lower=0> sigma_day_proc;
  vector[D_period] u_day;
}

transformed parameters {
  vector[D_period] s;                  // seasonal logits
  vector[N_diel_period] g;             // diel logits

  vector[D_period] p_season_base;      // seasonal probs before nugget
  vector[D_period] p_season;           // seasonal probs after nugget
  vector[M_obs] p_diel;                // diel probs for observed minutes
  vector[B] q_bin;                     // final product probs per bin

  // Remove RW2 nullspace (constant mode)
  vector[D_period] s_centered = s_raw - mean(s_raw);
  vector[N_diel_period] g_centered = g_raw - mean(g_raw);

  s = mu_season + tau_season * s_centered;
  g = mu_diel   + tau_diel   * g_centered;

  p_season_base = inv_logit(s);

  // Map diel logits to observed minutes
  for (m in 1:M_obs) {
    int minute = t_minutes[m] + 1;     // 0..1439 → 1..1440
    p_diel[m] = inv_logit(g[minute]);
  }

  // Apply day-level nugget on logit scale
  for (d in 1:D_period) {
    p_season[d] = inv_logit(logit(p_season_base[d]) + u_day[d]);
  }

  // Product structure at bin level
  for (b in 1:B) {
    int d  = day_id[b];                // true day-of-year
    int jm = dm_index[b];              // index into observed combos
    q_bin[b] = p_season[d] * p_diel[jm];
  }
}

model {
  // Global means
  mu_season ~ normal(-3, 1);
  mu_diel   ~ normal(-3, 1);

  // Curvature scales
  tau_season ~ normal(0, 0.05) T[0,];
  tau_diel   ~ normal(0, 0.01) T[0,];

  // --- Periodic intrinsic RW2 penalties ---
  // Seasonal (365-day cycle)
  for (d in 1:D_period) {
    int dm1 = (d == 1) ? D_period : d - 1;
    int dp1 = (d == D_period) ? 1 : d + 1;
    target += -0.5 * square(s_raw[dm1] - 2 * s_raw[d] + s_raw[dp1]);
  }

  // Diel (1440-minute cycle)
  for (m in 1:N_diel_period) {
    int mm1 = (m == 1) ? N_diel_period : m - 1;
    int mp1 = (m == N_diel_period) ? 1 : m + 1;
    target += -0.5 * square(g_raw[mm1] - 2 * g_raw[m] + g_raw[mp1]);
  }

  // Day-level nugget
  sigma_day_proc ~ normal(0, 0.05) T[0,];
  u_day          ~ normal(0, sigma_day_proc);

  // Likelihood
  for (b in 1:B) {
    target += n_bin[b] * log_sum_exp(
      log(q_bin[b]) + ell_bin[b],
      log1m(q_bin[b])
    );
  }
}

generated quantities {
  // empty — reconstruct curves in Python
}