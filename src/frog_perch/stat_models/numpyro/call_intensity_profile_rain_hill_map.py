import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta
import numpyro.optim as optim

jax.config.update("jax_enable_x64", True)

def model(data_dict):
    w_obs         = data_dict["w_obs"]
    precip_daily  = data_dict["precip_daily"]
    B_diel        = data_dict["B_diel"]
    N             = data_dict["N"]
    day_idx       = data_dict["day_idx"]
    num_days      = data_dict["num_days"]
    w_fraction    = data_dict.get("w_fraction", 0.0167)
   
    T, _ = w_obs.shape
    _, K_diel = B_diel.shape
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

    # --- 3. Fast Hydrology (3-Day Matrix - Bounded Geometric Interaction) ---
    p0 = precip_daily
    p1 = jnp.concatenate([jnp.array([0.0]), precip_daily[:-1]])
    p2 = jnp.concatenate([jnp.array([0.0, 0.0]), precip_daily[:-2]])

    b_p0 = numpyro.sample("b_p0", dist.Normal(0.0, 1.0))
    b_p1 = numpyro.sample("b_p1", dist.Normal(0.0, 1.0))
    b_p2 = numpyro.sample("b_p2", dist.Normal(0.0, 1.0))
    
    # Standard normal prior now perfectly matches the physical scale of the others
    b_p01 = numpyro.sample("b_p01", dist.Normal(0.0, 1.0))

    # Bounded square root interaction prevents product compounding
    p01_geom = jnp.sqrt(p0 * p1 + 1e-5)

    fast_rain_raw = (b_p0 * p0) + (b_p1 * p1) + (b_p2 * p2) + (b_p01 * p01_geom)

    # --- 4. Slow Effects & Scaling (Simple Plateau / No Habituation Model) ---
    tau_pool = numpyro.sample("tau_pool", dist.HalfNormal(10.0))
    b_shape = numpyro.sample("b_shape", dist.HalfNormal(5.0))
   
    trigger = 1.0 / (1.0 + jnp.exp(-b_shape * (w_s_raw - tau_pool)))

    gamma_plateau = numpyro.sample("gamma_plateau", dist.Normal(0.0, 2.0))
    eff_slow_raw = gamma_plateau * trigger

    # --- 5. Cross-Scale Interaction (Bounded Nonlinear) ---
    b_fast_slow = numpyro.sample("b_fast_slow", dist.Normal(0.0, 1.0))
    
    # jnp.tanh caps the daily vector magnitude, preventing runaway compounding
    eff_interact = b_fast_slow * jnp.tanh(fast_rain_raw) * trigger
    
    total_rain_effect = fast_rain_raw + eff_slow_raw + eff_interact

    # # Ridge penalty factor: Slopes the flat wet-season non-identifiability ridge.
    # # Evaluates to a completely negligible penalty at your baseline values (~0.007),
    # # but strictly prevents NUTS chains from drifting into wild triple-digit valleys.
    # ridge_penalty = 0.001 * (jnp.square(b_p0) + jnp.square(b_p1) + jnp.square(b_p2))
    # numpyro.factor("ridge_penalty", -ridge_penalty)

    total_rain_effect = (fast_rain_raw + eff_slow_raw + eff_interact)[day_idx]

    # --- 6. Diel & Day-Level Effects (Laplace) ---
    sigma_diel = numpyro.sample("sigma_diel", dist.HalfNormal(0.2))
    b_day = numpyro.sample("b_day", dist.HalfNormal(0.1))
   
    alpha_day_raw = numpyro.sample("alpha_day_raw", dist.Laplace(0.0, 1.0).expand([num_days]))
    alpha_day = alpha_day_raw * b_day

    z_diel_raw = numpyro.sample("z_diel_raw", dist.Normal(0, 1).expand([K_diel - 1]))
    beta_diel = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(z_diel_raw * sigma_diel)])
    trend_diel = jnp.dot(B_diel, beta_diel)

    # --- 7. Final Rate ---
    log_lambda = beta_0 + total_rain_effect + trend_diel + alpha_day[day_idx]
    lambd = jnp.exp(log_lambda)

    # --- 8. Deterministic tracking ---
    numpyro.deterministic("half_life_slow_val", half_life_slow)
    numpyro.deterministic("tau_pool_val", tau_pool)
    numpyro.deterministic("b_shape_val", b_shape)
    numpyro.deterministic("gamma_plateau_val", gamma_plateau)
    numpyro.deterministic("b_day_val", b_day)

    # --- 9. Likelihood ---
    nb_dist = dist.NegativeBinomial2(mean=lambd[:, None], concentration=phi_nb)
    log_bio = nb_dist.log_prob(k_seq[None, :])
    log_det = jnp.log(w_obs + 1e-15)
    log_mix = log_bio + log_det
    log_lik_per_window = jax.scipy.special.logsumexp(log_mix, axis=1)
    scaled_log_prob = log_lik_per_window.sum() * w_fraction
    numpyro.factor("obs_log_prob", scaled_log_prob)


def compile_and_run(stan_data, num_steps=5000, seed=0, use_burst=1.0):
    print("🚀 Compiling JAX model for Baseline Unscaled MAP Optimization...")
    data_dict = {
        "w_obs":         jnp.array(stan_data["w_obs"]),
        "precip_daily":  jnp.array(stan_data["precip_daily"]),
        "B_diel":        jnp.array(stan_data["B_diel"]),
        "N":             stan_data["N"],
        "day_idx":       jnp.array(stan_data["day_idx"]),
        "num_days":      stan_data["num_days"],
        "w_fraction":    jnp.array(stan_data["w_fraction"]),
        "use_burst":     jnp.array(use_burst, dtype=jnp.float64)
    }

    guide = AutoDelta(model)
    optimizer = optim.Adam(step_size=0.01)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    rng_key = jax.random.PRNGKey(seed)
    svi_result = svi.run(rng_key, num_steps, data_dict)
   
    # Extract point estimates
    map_params = svi_result.params
    final_loss = svi_result.losses[-1]
   
    # --- Instant Diagnostics ---
    b_day_val = float(map_params.get("b_day_auto_loc", 0.0))
   
    # Calculate Random Effect Variance
    alpha_raw = map_params.get("alpha_day_raw_auto_loc", jnp.zeros(stan_data["num_days"]))
    actual_alphas = alpha_raw * b_day_val
    alpha_var = float(jnp.var(actual_alphas))
    alpha_abs_mean = float(jnp.mean(jnp.abs(actual_alphas)))
   
    # Approximate BIC
    k = sum(v.size for k, v in map_params.items())
    n = stan_data["w_obs"].shape[0] * stan_data["w_obs"].shape[1]
    bic = k * jnp.log(n) + 2 * final_loss

    print("\n📊 --- MAP Convergence & Structural Diagnostics ---")
    print(f"📉 Final Loss (Neg Log-Post):  {final_loss:.2f}")
    print(f"⚖️ Approx BIC (k={k}):         {bic:.2f}")
    print(f"🎛️ b_day (Hierarch. Shrink):   {b_day_val:.4f}")
    print(f"📈 Alpha_day (Abs Mean):       {alpha_abs_mean:.4f}")
    print(f"📈 Alpha_day (Variance):       {alpha_var:.4f}")
    print("\n🌧️  --- Optimized 3-Day Pure Linear Rain Weights ---")
    print(f"💧 b_p0  (Day-Of Pulse):         {float(map_params['b_p0_auto_loc']):.4f}")
    print(f"⏱️  b_p1  (1-Day Lag Recovery):   {float(map_params['b_p1_auto_loc']):.4f}")
    print(f"⏳ b_p2  (2-Day Lag Recovery):   {float(map_params['b_p2_auto_loc']):.4f}")
    print(f"🔄 b_p01 (Consecutive Day Mult): {float(map_params['b_p01_auto_loc']):.4f}")
    print(f"⚡ b_fast_slow (Seasonal Gate): {float(map_params['b_fast_slow_auto_loc']):.4f}")
    print("-------------------------------------------\n")

    return svi_result, guide