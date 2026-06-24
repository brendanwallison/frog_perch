# """
# frog_perch/calibration/sensor_model.py
# Core generative math for the acoustic sensor calibration.
# """
# import numpy as np
# from scipy.special import expit as sigmoid
# from scipy.stats import beta

# import numpy as np
# from scipy.special import expit as sigmoid
# from scipy.stats import beta

# def calculate_likelihood_vector(y_mu_norm, y_var_norm, x_interf, p, k_max=16.0, return_log_likelihoods=False):
#     """
#     Computes the normalized probability vector P(k | y_mu_norm, y_var_norm, x_interf)
#     """
#     k_vec = np.arange(int(k_max) + 1)
    
#     # 1. Apply numerical guards directly to the pre-normalized inputs
#     y_mu = np.clip(y_mu_norm, 1e-4, 1.0 - 1e-4)
#     y_v = np.clip(y_var_norm, 1e-5, (y_mu * (1.0 - y_mu)) - 1e-5)
    
#     # 2. Derive Internal Machine Precision with a strict numerical floor > 0
#     phi_nn = (y_mu * (1.0 - y_mu) / y_v) - 1.0
#     phi_nn_safe = np.maximum(1e-3, phi_nn)
    
#     # 3. Mean Channel: Saturation and Masking 
#     noise_floor = sigmoid(p["b0"] + p["b_i"] * x_interf)
#     gamma_scale = np.exp(p["g0"] + p["g_i"] * x_interf) 
    
#     m_y_norm = noise_floor + (1.0 - noise_floor) * np.tanh(gamma_scale * (k_vec / k_max))
#     m_y_norm = np.clip(m_y_norm, 1e-4, 1.0 - 1e-4)
    
#     # 4. Dispersion Channel: Log-Linked Precision and Machine Trust
#     alpha_trust = p.get("alpha_v", 1.0)
#     base_precision = np.exp(p["a0"] + p["a_f"] * k_vec + p["a_i"] * x_interf)
    
#     phi_y = base_precision * (phi_nn_safe ** alpha_trust)
#     phi_y = np.clip(phi_y, 1.0, 1e4) # Bound to prevent beta function overflow
    
#     # 5. Compute Beta Likelihood Profiles
#     log_l_y = beta.logpdf(y_mu, m_y_norm * phi_y, (1.0 - m_y_norm) * phi_y)
#     combined = np.nan_to_num(log_l_y, nan=-50.0)
    
#     if return_log_likelihoods:
#         return combined
        
#     lik = np.exp(combined - np.max(combined))
#     return lik / (np.sum(lik) + 1e-12)

"""
frog_perch/calibration/sensor_model.py

Core generative math for the acoustic sensor calibration.

Revision:
- Replaces the fixed tanh saturation curve with a two-parameter Weibull CDF
  saturation function. This introduces a single additional shape parameter
  (h0) while preserving the Beta observation model, noise floor, precision
  channel, and overall generative structure.
"""

import numpy as np
from scipy.special import expit as sigmoid
from scipy.stats import beta


def calculate_likelihood_vector(
    y_mu_norm,
    y_var_norm,
    x_interf,
    p,
    k_max=16.0,
    return_log_likelihoods=False,
):
    """
    Computes the normalized likelihood vector

        P(k | y_mu_norm, y_var_norm, x_interf)

    Parameters
    ----------
    y_mu_norm : float
        Normalized NN predicted mean.

    y_var_norm : float
        Normalized NN predicted variance.

    x_interf : float
        Acoustic interference covariate.

    p : dict
        Learned calibration parameters.

        Existing parameters
        -------------------
        b0, b_i
            Noise-floor coefficients.

        g0, g_i
            Weibull scale coefficients (formerly tanh scale).

        a0, a_f, a_i
            Precision regression coefficients.

        alpha_v (optional)
            Machine-trust exponent.

        New optional parameter
        ----------------------
        h0
            Log-Weibull shape parameter.
            Defaults to 0.0 (shape = 1.0) if omitted.

    k_max : float
        Maximum count represented.

    return_log_likelihoods : bool
        If True, return unnormalized log likelihoods.
    """

    k_vec = np.arange(int(k_max) + 1)

    # ------------------------------------------------------------------
    # 1. Numerical guards on observed NN summaries
    # ------------------------------------------------------------------

    y_mu = np.clip(y_mu_norm, 1e-4, 1.0 - 1e-4)

    y_v = np.clip(
        y_var_norm,
        1e-5,
        (y_mu * (1.0 - y_mu)) - 1e-5,
    )

    # ------------------------------------------------------------------
    # 2. Recover internal NN precision
    # ------------------------------------------------------------------

    phi_nn = (y_mu * (1.0 - y_mu) / y_v) - 1.0
    phi_nn_safe = np.maximum(1e-3, phi_nn)

    # ------------------------------------------------------------------
    # 3. Mean channel
    # ------------------------------------------------------------------

    noise_floor = sigmoid(
        p["b0"] + p["b_i"] * x_interf
    )

    # Scale parameter (same role as previous gamma)
    scale = np.exp(
        p["g0"] + p["g_i"] * x_interf
    )

    # New Weibull shape parameter
    shape = np.exp(
        p.get("h0", 0.0)
    )

    z = k_vec / k_max

    # Weibull CDF saturation:
    #
    #   S(z) = 1 - exp(-(scale*z)^shape)
    #
    # shape = 1:
    #   exponential saturation
    #
    # shape > 1:
    #   delayed onset, steeper transition
    #
    # shape < 1:
    #   faster initial rise
    #

    exponent = np.power(scale * z, shape)

    # Numerical stability
    exponent = np.clip(exponent, 0.0, 60.0)

    saturation = 1.0 - np.exp(-exponent)

    m_y_norm = (
        noise_floor
        + (1.0 - noise_floor) * saturation
    )

    m_y_norm = np.clip(
        m_y_norm,
        1e-4,
        1.0 - 1e-4,
    )

    # ------------------------------------------------------------------
    # 4. Dispersion channel
    # ------------------------------------------------------------------

    alpha_trust = p.get("alpha_v", 1.0)

    base_precision = np.exp(
        p["a0"]
        + p["a_f"] * k_vec
        + p["a_i"] * x_interf
    )

    phi_y = (
        base_precision
        * (phi_nn_safe ** alpha_trust)
    )

    phi_y = np.clip(
        phi_y,
        1.0,
        1e4,
    )

    # ------------------------------------------------------------------
    # 5. Beta likelihood
    # ------------------------------------------------------------------

    log_l_y = beta.logpdf(
        y_mu,
        m_y_norm * phi_y,
        (1.0 - m_y_norm) * phi_y,
    )

    combined = np.nan_to_num(
        log_l_y,
        nan=-50.0,
    )

    if return_log_likelihoods:
        return combined

    lik = np.exp(
        combined - np.max(combined)
    )

    return lik / (np.sum(lik) + 1e-12)