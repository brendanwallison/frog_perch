import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_median

# Enable 64-bit precision for stable gradients
jax.config.update("jax_enable_x64", True)

def model(data_dict):
    """
    NumPyro implementation of the Call Intensity P-Spline model.
    v5: Hybrid Parameterization (The "Goldilocks" Model).
    
    Changes:
      1. FIXED: Added 'trend_diel' to deterministic trackers.
      2. FIXED: Switched 'alpha_day' to Centered Parameterization (CP).
         - Why? With ~4000 windows per day, the data is too strong for NCP.
         - Result: Faster mixing, no stiffness (ESS should rise > 400).
    """
    w_obs    = data_dict["w_obs"]
    B_season = data_dict["B_season"]
    B_diel   = data_dict["B_diel"]
    N        = data_dict["N"]
    
    day_idx  = data_dict["day_idx"]   # Vector [T]
    num_days = data_dict["num_days"]  # Scalar
    
    T, _ = w_obs.shape
    _, K_season = B_season.shape
    _, K_diel   = B_diel.shape
    
    k_seq = jnp.arange(N + 1)

    # --- 1. Global Parameters ---
    beta_0 = numpyro.sample("beta_0", dist.Normal(-2.0, 2.0))
    phi    = numpyro.sample("phi",    dist.HalfNormal(10.0))

    # --- 2. Trend Smoothness (Keep NCP for Splines) ---
    # Splines always need NCP because they rely heavily on the prior (sigma)
    sigma_season = numpyro.sample("sigma_season", dist.HalfNormal(0.5))
    sigma_diel   = numpyro.sample("sigma_diel",   dist.HalfNormal(0.5))

    # --- 3. Day-Level Volatility (SWITCH TO CENTERED) ---
    sigma_day = numpyro.sample("sigma_day", dist.HalfNormal(1.0))
    
    # [CENTERED] We sample alpha directly from Normal(0, sigma_day)
    # The sampler sees the connection between alpha and sigma directly.
    # With 4,000 data points, this is vastly more efficient than NCP.
    alpha_day = numpyro.sample("alpha_day", dist.Normal(0.0, sigma_day).expand([num_days]))

    # --- 4. Spline Construction (Keep NCP) ---
    z_season_raw = numpyro.sample("z_season_raw", dist.Normal(0.0, 1.0).expand([K_season - 1]))
    z_diel_raw   = numpyro.sample("z_diel_raw",   dist.Normal(0.0, 1.0).expand([K_diel - 1]))
    
    beta_season = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(z_season_raw * sigma_season)])
    beta_diel   = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(z_diel_raw * sigma_diel)])

    # --- 5. Rate Calculation ---
    trend_season = jnp.dot(B_season, beta_season)
    trend_diel   = jnp.dot(B_diel,   beta_diel)
    
    # Hierarchy
    log_lambda = beta_0 + \
                 trend_season + trend_diel + \
                 alpha_day[day_idx]
                 
    lambd = jnp.exp(log_lambda)
    
    # --- Tracking ---
    numpyro.deterministic("lambda", lambd)
    numpyro.deterministic("trend_season", trend_season)
    numpyro.deterministic("trend_diel", trend_diel) # <--- FIXED: Added missing tracker

    # --- 6. Likelihood ---
    nb_dist = dist.NegativeBinomial2(mean=lambd[:, None], concentration=phi)
    log_bio = nb_dist.log_prob(k_seq[None, :]) 
    log_det = jnp.log(w_obs + 1e-15)
    
    log_mix = log_bio + log_det
    log_lik_per_window = jax.scipy.special.logsumexp(log_mix, axis=1)
    
    numpyro.factor("obs_log_prob", log_lik_per_window.sum())


def compile_and_run(stan_data, num_warmup=1000, num_samples=1000, num_chains=4, seed=0):
    print("🚀 Compiling JAX model (Hybrid Parameterization)...")
    
    data_dict = {
        "w_obs":    jnp.array(stan_data["w_obs"]),
        "B_season": jnp.array(stan_data["B_season"]),
        "B_diel":   jnp.array(stan_data["B_diel"]),
        "N":        stan_data["N"],
        "day_idx":  jnp.array(stan_data["day_idx"]),
        "num_days": stan_data["num_days"]
    }

    nuts_kernel = NUTS(
        model, 
        target_accept_prob=0.85, # Hybrid model is friendly, 0.85 is plenty
        init_strategy=init_to_median
    )
    
    mcmc = MCMC(
        nuts_kernel, 
        num_warmup=num_warmup, 
        num_samples=num_samples, 
        num_chains=num_chains,
        chain_method='parallel', 
        progress_bar=True
    )
    
    print("🔥 Sampling...")
    rng_key = jax.random.PRNGKey(seed)
    mcmc.run(rng_key, data_dict)
    
    return mcmc