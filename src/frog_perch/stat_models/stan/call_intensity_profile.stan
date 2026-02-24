data {
  // --- Dimensions ---
  int<lower=1> T;                // Number of analysis windows
  int<lower=1> N;                // Max count (bins per window)
  
  // --- Observations ---
  // Likelihood Profile: P(k calls | detector) for k in 0..N
  matrix[T, N + 1] w_obs;        
  
  // --- HSGP Basis Functions ---
  int<lower=1> M_season;
  int<lower=1> M_diel;
  
  matrix[T, M_season] X_season;
  matrix[T, M_diel]   X_diel;
  
  // --- Domain Boundaries ---
  real L_season;
  real L_diel;
}

transformed data {
  // Helper sequence for counts [0, 1, ..., N]
  array[N + 1] int k_seq;
  for (i in 1:(N + 1)) {
    k_seq[i] = i - 1;
  }
}

parameters {
  // --- Intercept ---
  real beta_0;
  
  // --- HSGP Coefficients ---
  vector[M_season] z_season;
  vector[M_diel]   z_diel;
  
  // --- GP Hyperparameters ---
  real<lower=0> lengthscale_season;
  real<lower=0> sigma_season;
  
  real<lower=0> lengthscale_diel;
  real<lower=0> sigma_diel;
  
  // --- Overdispersion ---
  real<lower=0> phi;
}

transformed parameters {
  // 1. Compute Spectral Densities (Prior Variances)
  vector[M_season] sd_season;
  vector[M_diel]   sd_diel;
  
  // Season SPD
  for (m in 1:M_season) {
    real omega = m * pi() / (2 * L_season);
    real S = square(sigma_season) * sqrt(2 * pi()) * lengthscale_season * exp(-0.5 * square(omega * lengthscale_season));
    sd_season[m] = sqrt(S);
  }
  
  // Diel SPD
  for (m in 1:M_diel) {
    real omega = m * pi() / (2 * L_diel);
    real S = square(sigma_diel) * sqrt(2 * pi()) * lengthscale_diel * exp(-0.5 * square(omega * lengthscale_diel));
    sd_diel[m] = sqrt(S);
  }
  
  // 2. Scale the coefficients
  vector[M_season] beta_season = z_season .* sd_season;
  vector[M_diel]   beta_diel   = z_diel .* sd_diel;

  // 3. Calculate Latent Rates (calls per window)
  vector[T] log_lambda = beta_0 + X_season * beta_season + X_diel * beta_diel;
  vector[T] lambda = exp(log_lambda);
}

model {
  // --- Priors ---
  beta_0 ~ normal(0, 2);
  
  // GP Hyperparameters
  lengthscale_diel   ~ lognormal(-1, 1);   // median ~0.36, allows down to ~0.05
  lengthscale_season ~ lognormal(-1, 1); // median ~0.36, allows down to ~0.05
  
  sigma_diel         ~ exponential(1); 
  sigma_season       ~ exponential(1);
  
  phi ~ normal(0, 5);
  
  z_season ~ std_normal();
  z_diel   ~ std_normal();

  // --- Likelihood (The Mixture Model) ---
  for (t in 1:T) {
    vector[N + 1] log_probs;
    
    // We must loop here because neg_binomial_2_lpmf returns a scalar sum
    // if passed an array, but we need the individual log-probs for the mixture.
    for (i in 1:(N + 1)) {
       // 1. Biological Probability: P(k | lambda, phi)
       real log_bio = neg_binomial_2_lpmf(k_seq[i] | lambda[t], phi);
       
       // 2. Detector Evidence: P(k | detector)
       // w_obs is linear probability, so we take log()
       // Add epsilon to avoid log(0) if detector says 0 probability
       real log_det = log(w_obs[t, i] + 1e-15); 
       
       log_probs[i] = log_bio + log_det;
    }
    
    // Marginalize over k
    target += log_sum_exp(log_probs);
  }
}

generated quantities {
  vector[T] trend_season = X_season * beta_season;
  vector[T] trend_diel   = X_diel * beta_diel;
}