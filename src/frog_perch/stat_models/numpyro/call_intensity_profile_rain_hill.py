import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_median

jax.config.update("jax_enable_x64", True)

# The Canonical JAX-Native Vectorized Cubic B-Spline Basis Generator
# The Canonical JAX-Native Vectorized Cubic B-Spline Basis Generator
def jax_b3_spline_basis(x, knots, low_bound=17.0, up_bound=23.0):
    # Fix #1: Apply 4 repetitions (degree + 1) for boundary clamping
    all_knots = jnp.concatenate([
        jnp.repeat(low_bound, 4), 
        knots, 
        jnp.repeat(up_bound, 4)
    ])
    x_clip = jnp.clip(x, low_bound, up_bound)
    
    n_knots = len(all_knots)
    n_bases = n_knots - 1
    
    # Fix #2: Handle the right-edge closed interval
    is_last = (x_clip[:, None] >= all_knots[-2]) & (x_clip[:, None] <= all_knots[-1])
    is_others = (x_clip[:, None] >= all_knots[:-1]) & (x_clip[:, None] < all_knots[1:])
    
    # Mask for the 0th degree basis
    mask = jnp.where(is_last, True, is_others)
    B = [jnp.where(mask, 1.0, 0.0)]
    
    for deg in range(1, 4):
        B_next = []
        for i in range(n_bases - deg):
            denom1 = all_knots[i + deg] - all_knots[i]
            denom2 = all_knots[i + deg + 1] - all_knots[i + 1]
            
            d1_safe = jnp.where(denom1 > 0, denom1, 1.0)
            d2_safe = jnp.where(denom2 > 0, denom2, 1.0)
            
            term1 = jnp.where(denom1 > 0, 1.0, 0.0) * ((x_clip - all_knots[i]) / d1_safe) * B[-1][:, i]
            term2 = jnp.where(denom2 > 0, 1.0, 0.0) * ((all_knots[i + deg + 1] - x_clip) / d2_safe) * B[-1][:, i + 1]
            
            B_next.append((term1 + term2)[:, None])
            
        B.append(jnp.hstack(B_next))
        
    return B[-1]

def model(data_dict):
    w_obs         = data_dict["w_obs"]
    precip_daily  = data_dict["precip_daily"]
    N             = data_dict["N"]
    day_idx       = data_dict["day_idx"]
    num_days      = data_dict["num_days"]
    w_fraction    = data_dict.get("w_fraction", 0.0167)
    rms_obs       = data_dict["rms_obs"]  
    
    time_of_day   = data_dict["time_of_day"]
    knots_grid    = data_dict["knots_grid"]
    doy_smooth    = data_dict["doy_smooth"]
    K_diel_static = data_dict["K_diel_static"] 
   
    k_seq = jnp.arange(N + 1)

    # --- 1. Global Parameters ---
    beta_0 = numpyro.sample("beta_0", dist.Normal(-4.0, 1.0))
    phi_nb = numpyro.sample("phi", dist.Exponential(1.0))

    # --- 2. Slow Hydrology (Integrator) ---
    half_life_slow = numpyro.sample("half_life_slow", dist.LogNormal(jnp.log(5.0), 1.0))
    phi_s = 2 ** (-1.0 / half_life_slow)

    def wetness_update(phi, carry, p_t):
        w_t = phi * carry + (1.0 - phi) * p_t
        return w_t, w_t

    _, w_s_raw = jax.lax.scan(lambda c, p: wetness_update(phi_s, c, p), 0.0, precip_daily)

    # --- 3. Fast Hydrology (3-Day Matrix) ---
    p0 = precip_daily
    p1 = jnp.concatenate([jnp.array([0.0]), precip_daily[:-1]])
    p2 = jnp.concatenate([jnp.array([0.0, 0.0]), precip_daily[:-2]])

    b_p0 = numpyro.sample("b_p0", dist.Normal(0.0, 1.0))
    b_p1 = numpyro.sample("b_p1", dist.Normal(0.0, 1.0))
    b_p2 = numpyro.sample("b_p2", dist.Normal(0.0, 1.0))
    fast_rain_raw = (b_p0 * p0) + (b_p1 * p1) + (b_p2 * p2) 

    # --- 4. Slow Seasonal Plateau Scaling ---
    tau_pool = numpyro.sample("tau_pool", dist.HalfNormal(10.0))
    b_shape = numpyro.sample("b_shape", dist.HalfNormal(5.0))
    trigger = 1.0 / (1.0 + jnp.exp(-b_shape * (w_s_raw - tau_pool)))
    gamma_plateau = numpyro.sample("gamma_plateau", dist.Normal(0.0, 1.0))
    eff_slow_raw = gamma_plateau * trigger
    total_rain_effect = (fast_rain_raw + eff_slow_raw)[day_idx]

    # --- 5. Linear Decoupled Climate Slopes ---
    b_rms         = numpyro.sample("b_rms", dist.Normal(0.0, 1.0))
    b_temp_inter  = numpyro.sample("b_temp_inter", dist.Normal(0.0, 1.0))
    b_temp_intra  = numpyro.sample("b_temp_intra", dist.Normal(0.0, 1.0))
    b_rh_inter    = numpyro.sample("b_rh_inter", dist.Normal(0.0, 1.0))
    b_rh_intra    = numpyro.sample("b_rh_intra", dist.Normal(0.0, 1.0))
    b_light_inter = numpyro.sample("b_light_inter", dist.Normal(0.0, 1.0))
    b_light_intra = numpyro.sample("b_light_intra", dist.Normal(0.0, 1.0))

    # --- 6. Latent Time-Warping Phase Shift ---
    delta_seasonal = numpyro.sample("delta_seasonal", dist.Normal(0.0, 0.5))
    warped_time = time_of_day + (delta_seasonal * doy_smooth)
    
    B_warped = jax_b3_spline_basis(warped_time, knots_grid, low_bound=17.0, up_bound=23.0)

    sigma_diel = numpyro.sample("sigma_diel", dist.HalfNormal(1.0))
    z_diel_raw = numpyro.sample("z_diel_raw", dist.Normal(0, 1).expand([K_diel_static - 1]))
    beta_diel = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(z_diel_raw * sigma_diel)])
    trend_diel = jnp.dot(B_warped, beta_diel)

    # --- 7. Noise Suppression & Day Random Effects ---
    b_day = numpyro.sample("b_day", dist.HalfNormal(0.1))
    alpha_day_raw = numpyro.sample("alpha_day_raw", dist.Normal(0.0, 1.0).expand([num_days]))
    alpha_day = alpha_day_raw * b_day

    # --- 8. Rate Calculation ---
    log_lambda = (beta_0 + total_rain_effect + trend_diel + (b_rms * rms_obs) + 
                  (data_dict["temp_inter"] * b_temp_inter)  + (data_dict["temp_intra"] * b_temp_intra) + 
                  (data_dict["rh_inter"]   * b_rh_inter)    + (data_dict["rh_intra"]   * b_rh_intra) + 
                  (data_dict["light_inter"]* b_light_inter) + (data_dict["light_intra"]* b_light_intra) + 
                  alpha_day[day_idx])
    lambd = jnp.exp(log_lambda)

    # --- 9. Deterministic Tracking ---
    numpyro.deterministic("delta_seasonal_val", delta_seasonal)
    numpyro.deterministic("half_life_slow_val", half_life_slow)
    numpyro.deterministic("b_day_val", b_day)
    numpyro.deterministic("b_rms_val", b_rms)
    numpyro.deterministic("b_temp_inter_val", b_temp_inter)
    numpyro.deterministic("b_temp_intra_val", b_temp_intra)
    numpyro.deterministic("b_rh_inter_val", b_rh_inter)
    numpyro.deterministic("b_rh_intra_val", b_rh_intra)
    numpyro.deterministic("b_light_inter_val", b_light_inter)
    numpyro.deterministic("b_light_intra_val", b_light_intra)

    # --- 10. Likelihood ---
    nb_dist = dist.NegativeBinomial2(mean=lambd[:, None], concentration=phi_nb)
    log_bio = nb_dist.log_prob(k_seq[None, :])
    log_det = jnp.log(w_obs + 1e-15)
    log_mix = log_bio + log_det
    log_lik_per_window = jax.scipy.special.logsumexp(log_mix, axis=1)
    scaled_log_prob = log_lik_per_window.sum() * w_fraction
    numpyro.factor("obs_log_prob", scaled_log_prob)

def compile_and_run(stan_data, num_warmup=1000, num_samples=1000, num_chains=4, seed=0):
    print(f"🚀 Initializing NUTS MCMC Engine with {num_chains} Parallel Chains...")
    
    data_dict = {
        "w_obs":         jnp.array(stan_data["w_obs"]),
        "precip_daily":  jnp.array(stan_data["precip_daily"]),
        # 🌟 REMOVED: "B_diel" - the model handles this now
        "knots_grid":    jnp.array(stan_data["knots_grid"]), # ADDED
        "time_of_day":   jnp.array(stan_data["time_of_day"]), # ADDED
        "doy_smooth":    jnp.array(stan_data["doy_smooth"]),   # ADDED
        "K_diel_static": stan_data["K_diel_static"],          # ADDED
        "N":             stan_data["N"],
        "day_idx":       jnp.array(stan_data["day_idx"]),
        "num_days":      stan_data["num_days"],
        "w_fraction":    jnp.array(stan_data.get("w_fraction", 0.0167), dtype=jnp.float64),
        "rms_obs":       jnp.array(stan_data["rms_obs"]),
        "temp_inter":    jnp.array(stan_data["temp_inter"]),
        "temp_intra":    jnp.array(stan_data["temp_intra"]),
        "rh_inter":      jnp.array(stan_data["rh_inter"]),
        "rh_intra":      jnp.array(stan_data["rh_intra"]),
        "light_inter":   jnp.array(stan_data["light_inter"]),
        "light_intra":   jnp.array(stan_data["light_intra"]),
    }
   
    nuts_kernel = NUTS(model, target_accept_prob=0.92, init_strategy=init_to_median, max_tree_depth=10)
    
    mcmc = MCMC(
        nuts_kernel, 
        num_warmup=num_warmup, 
        num_samples=num_samples, 
        num_chains=num_chains, 
        chain_method='parallel', 
        progress_bar=True
    )
    
    rng_key = jax.random.PRNGKey(seed)
    mcmc.run(rng_key, data_dict)
    
    print("\n📊 --- NUTS Sampling Post-Processing Complete ---")
    mcmc.print_summary()
    
    return mcmc