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
  // --- Seasonal HSGP ---
  real mu_season;
  real<lower=0> alpha_season;
  real<lower=0> rho_season; 
  vector[2*M_season] beta_season;

  // --- Diel HSGP ---
  real mu_diel;
  real<lower=0> alpha_diel;
  real<lower=0> rho_diel;
  vector[2*M_diel] beta_diel;

  // --- UNIFIED NOISE (Day Level) ---
  // Absorbs both biological chaos AND detector error
  // lower=1e-6 prevents "Zero Variance" crash
  real<lower=1e-6> sigma_ell; 
  vector[D_obs] z_ell_day; 
}

transformed parameters {
  vector[2*M_season] diagSPD_season;
  vector[2*M_diel]   diagSPD_diel;

  // --- 1. RBF Spectral Densities ---
  {
    real period = 365.0;
    real term1 = (alpha_season^2) * sqrt(2.0 * pi()) * rho_season;
    for (m in 1:M_season) {
      real w = 2.0 * pi() * m / period;
      real S = term1 * exp(-0.5 * square(rho_season * w));
      
      diagSPD_season[2*(m-1) + 1] = sqrt(S);
      diagSPD_season[2*(m-1) + 2] = sqrt(S);
    }
  }

  {
    real period = 1440.0;
    real term1 = (alpha_diel^2) * sqrt(2.0 * pi()) * rho_diel;
    for (m in 1:M_diel) {
      real w = 2.0 * pi() * m / period;
      real S = term1 * exp(-0.5 * square(rho_diel * w));
      
      diagSPD_diel[2*(m-1) + 1] = sqrt(S);
      diagSPD_diel[2*(m-1) + 2] = sqrt(S);
    }
  }

  // --- 2. Reconstruct Curves ---
  // No "+ u_day" here anymore. s_obs is pure smooth curve.
  vector[D_obs] s_obs = mu_season + X_season * (beta_season .* diagSPD_season);
  vector[M_obs] g_obs = mu_diel   + X_diel   * (beta_diel   .* diagSPD_diel);
  
  vector[D_obs] p_season;
  vector[M_obs] p_diel;
  vector[B] q_bin;
  vector[B] ell_adj;

  // Pure smooth probability
  for (d in 1:D_obs) {
    p_season[d] = inv_logit(s_obs[d]);
  }
  p_diel = inv_logit(g_obs);

  // --- 3. Apply Unified Day-Level Noise ---
  // All deviation is applied here, to the Log-Odds.
  for (b in 1:B) {
      ell_adj[b] = ell_bin[b] + z_ell_day[day_idx[b]] * sigma_ell;
  }

  // --- 4. Combine ---
  for (b in 1:B) {
    q_bin[b] = p_season[day_idx[b]] * p_diel[diel_idx[b]];
  }
}

model {
  // --- Priors ---
  mu_season ~ normal(-3, 1);
  mu_diel   ~ normal(-3, 1);
  
  alpha_season ~ normal(0, 2); 
  alpha_diel   ~ normal(0, 2);

  rho_season   ~ lognormal(4.0, 0.5);  
  rho_diel     ~ lognormal(4.0, 0.5);

  beta_season ~ std_normal();
  beta_diel   ~ std_normal();

  // Unified Noise Priors
  sigma_ell ~ normal(0, 0.5); 
  z_ell_day ~ std_normal(); 

  // --- Likelihood ---
  for (b in 1:B) {
    target += n_bin[b] * log_sum_exp(
      log(q_bin[b]) + ell_adj[b],
      log1m(q_bin[b])
    );
  }
}