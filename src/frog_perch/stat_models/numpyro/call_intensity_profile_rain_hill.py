import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_median

# Ensure float64 precision for stable NUTS sampling
jax.config.update("jax_enable_x64", True)

def model(data_dict):
    w_obs         = data_dict["w_obs"]
    precip_daily  = data_dict["precip_daily"]
    B_diel        = data_dict["B_diel"]
    N             = data_dict["N"]
    day_idx       = data_dict["day_idx"]
    num_days      = data_dict["num_days"]
    X_climate     = data_dict["X_climate"]  # Shape: (T, K_climate)
    K_climate     = data_dict["K_climate"]
    
    T, _ = w_obs.shape
    _, K_diel = B_diel.shape
    k_seq = jnp.arange(N + 1)

    # --- 0. Aggregate High-Res Climate to Daily Level (Plan 2) ---
    # MODERN JAX API: Count the number of observations per day
    day_counts = jnp.zeros(num_days).at[day_idx].add(1.0)
    safe_day_counts = jnp.maximum(day_counts, 1.0) # Prevent division by zero
    
    # MODERN JAX API: Sum the climate variables per day, then divide by counts
    X_day_sum = jnp.zeros((num_days, K_climate)).at[day_idx].add(X_climate)
    X_day = X_day_sum / safe_day_counts[:, None]

    # SAFETY NET: Explicitly center the daily aggregates to prevent intercept drift
    X_day_centered = X_day - jnp.mean(X_day, axis=0)

    # --- 1. Global Parameters ---
    beta_0 = numpyro.sample("beta_0", dist.Normal(-2.0, 1.0))
    phi_nb = numpyro.sample("phi", dist.HalfNormal(10.0)) 

    # --- 2. Ordered Half-Lives (Hydrological Decay) ---
    half_life_fast = numpyro.sample("half_life_fast", dist.LogNormal(jnp.log(0.5), 0.5))
    hl_diff        = numpyro.sample("hl_diff", dist.LogNormal(jnp.log(10.0), 0.5))
    half_life_slow = numpyro.deterministic("half_life_slow_val", half_life_fast + hl_diff)

    phi_f = 2 ** (-1.0 / half_life_fast)
    phi_s = 2 ** (-1.0 / half_life_slow)

    # --- 3. Latent Wetness States ---
    def wetness_update(phi, carry, p_t):
        w_t = phi * carry + (1.0 - phi) * p_t
        return w_t, w_t

    _, w_f_raw = jax.lax.scan(lambda c, p: wetness_update(phi_f, c, p), 0.0, precip_daily)
    _, w_s_raw = jax.lax.scan(lambda c, p: wetness_update(phi_s, c, p), 0.0, precip_daily)

    # --- 4. Rain Effects & Scaling ---
    k_slow = numpyro.sample("k_slow", dist.LogNormal(jnp.log(20.0), 0.5))
    n_slow = numpyro.sample("n_slow", dist.LogNormal(jnp.log(2.0), 0.5))

    hill_slow_full = (w_s_raw ** n_slow) / (k_slow ** n_slow + w_s_raw ** n_slow)
    saturating_slow_raw = hill_slow_full[day_idx]
    
    w_f_std = jnp.std(w_f_raw) + 1e-6
    linear_fast_raw = (w_f_raw / w_f_std)[day_idx]
    
    gamma_fast = numpyro.sample("gamma_fast", dist.Normal(0.0, 2.0))
    gamma_slow = numpyro.sample("gamma_slow", dist.Normal(0.0, 2.0))

    eff_f = gamma_fast * (linear_fast_raw - jnp.mean(linear_fast_raw))
    eff_s = gamma_slow * (saturating_slow_raw - jnp.mean(saturating_slow_raw))
    
    total_rain_effect = eff_f + eff_s

    # --- 5. Diel Trend ---
    sigma_diel = numpyro.sample("sigma_diel", dist.HalfNormal(0.2))
    z_diel_raw = numpyro.sample("z_diel_raw", dist.Normal(0, 1).expand([K_diel - 1]))
    beta_diel = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(z_diel_raw * sigma_diel)])
    trend_diel = jnp.dot(B_diel, beta_diel)

    # --- 6. PLAN 2: Daily Climate Drives Day-to-Day Variance ---
    # Tighter prior (0.3) because inputs are standardized
    beta_day_climate = numpyro.sample("beta_day_climate", dist.Normal(0, 0.3).expand([K_climate]))
    
    # Baseline shift for the day based strictly on centered daily climate
    mu_day = jnp.dot(X_day_centered, beta_day_climate)

    # Unexplained daily variance (noise)
    sigma_day  = numpyro.sample("sigma_day", dist.HalfNormal(0.2))
    alpha_day_raw = numpyro.sample("alpha_day_raw", dist.Normal(0, 1).expand([num_days]))
    
    # Non-centered parameterization 
    alpha_day = mu_day + (alpha_day_raw * sigma_day)

    # --- 7. Final Rate Calculation ---
    log_lambda = beta_0 + total_rain_effect + trend_diel + alpha_day[day_idx]
    lambd = jnp.exp(log_lambda)

    # --- 8. Deterministic tracking ---
    numpyro.deterministic("gamma_fast_val", gamma_fast)
    numpyro.deterministic("gamma_slow_val", gamma_slow)
    numpyro.deterministic("half_life_fast_val", half_life_fast)
    numpyro.deterministic("beta_climate_val", beta_day_climate)
    numpyro.deterministic("sigma_day_val", sigma_day)
    numpyro.deterministic("daily_wetness_fast", w_f_raw)
    numpyro.deterministic("daily_wetness_slow", w_s_raw)

    # --- 9. Likelihood ---
    nb_dist = dist.NegativeBinomial2(mean=lambd[:, None], concentration=phi_nb)
    log_bio = nb_dist.log_prob(k_seq[None, :])
    log_det = jnp.log(w_obs + 1e-15)
    log_mix = log_bio + log_det
    log_lik_per_window = jax.scipy.special.logsumexp(log_mix, axis=1)
    numpyro.factor("obs_log_prob", log_lik_per_window.sum())


def compile_and_run(stan_data, num_warmup=1000, num_samples=1000, num_chains=4, seed=0):
    print("🚀 Compiling JAX model (Plan 2: Climate as Daily Driver)...")
    
    data_dict = {
        "w_obs":         jnp.array(stan_data["w_obs"]),
        "precip_daily":  jnp.array(stan_data["precip_daily"]),
        "B_diel":        jnp.array(stan_data["B_diel"]),
        "N":             stan_data["N"],
        "day_idx":       jnp.array(stan_data["day_idx"]),
        "num_days":      stan_data["num_days"],
        "X_climate":     jnp.array(stan_data["X_climate"]),
        "K_climate":     stan_data["K_climate"]
    }

    nuts_kernel = NUTS(model, 
                       target_accept_prob=0.90, 
                       init_strategy=init_to_median,
                       max_tree_depth=10) 
                       
    mcmc = MCMC(nuts_kernel, num_warmup=num_warmup, num_samples=num_samples, 
                num_chains=num_chains, chain_method='parallel', progress_bar=True)
    
    print("🔥 Sampling...")
    rng_key = jax.random.PRNGKey(seed)
    mcmc.run(rng_key, data_dict)
    return mcmc