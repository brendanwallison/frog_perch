"""
frog_perch/calibration/sensor_model.py
Core generative math for the acoustic sensor calibration.
"""
import numpy as np
from scipy.special import expit as sigmoid
from scipy.stats import beta

import numpy as np
from scipy.special import expit as sigmoid
from scipy.stats import beta

def calculate_likelihood_vector(y_raw, nu_obs, x_interf, p, k_max=16.0, return_log_likelihoods=False):
    """
    Computes the normalized probability vector P(k | y_raw, nu_obs, x_interf)
    conforming exactly to the formal LaTeX document specification.
    """
    k_vec = np.arange(int(k_max) + 1)
    
    # 1. Normalize network inputs safely away from hard 0/1 boundaries
    y_mu = np.clip(y_raw / k_max, 1e-4, 1.0 - 1e-4)
    y_v = np.clip(nu_obs, 1e-5, (y_mu * (1.0 - y_mu)) - 1e-5)
    
    # 2. Derive Internal Machine Precision with a strict numerical floor > 0
    # Insulates against negative bases when raised to fractional alpha_v exponents
    phi_nn = (y_mu * (1.0 - y_mu) / y_v) - 1.0
    phi_nn_safe = np.maximum(1e-3, phi_nn)
    
    # 3. Mean Channel: Saturation and Masking (Preserving Exact Document Math)
    # rho(x) = Sigmoid(b0 + b_i * x)
    # gamma(x) = Exp(g0 + g_i * x)
    noise_floor = sigmoid(p["b0"] + p["b_i"] * x_interf)
    gamma_scale = np.exp(p["g0"] + p["g_i"] * x_interf) 
    
    # Restored: gamma(x) * (k / K_max) exactly as intended
    m_y_norm = noise_floor + (1.0 - noise_floor) * np.tanh(gamma_scale * (k_vec / k_max))
    m_y_norm = np.clip(m_y_norm, 1e-4, 1.0 - 1e-4)
    
    # 4. Dispersion Channel: Log-Linked Precision and Machine Trust
    # phi_y = exp(a0 + a_f * k + a_i * x) * (phi_nn)^alpha_v
    alpha_trust = p.get("alpha_v", 1.0)
    base_precision = np.exp(p["a0"] + p["a_f"] * k_vec + p["a_i"] * x_interf)
    
    phi_y = base_precision * (phi_nn_safe ** alpha_trust)
    phi_y = np.clip(phi_y, 1.0, 1e4) # Bound to prevent beta function overflow
    
    # 5. Compute Beta Likelihood Profiles
    log_l_y = beta.logpdf(y_mu, m_y_norm * phi_y, (1.0 - m_y_norm) * phi_y)
    combined = np.nan_to_num(log_l_y, nan=-50.0)
    
    if return_log_likelihoods:
        return combined
        
    lik = np.exp(combined - np.max(combined))
    return lik / (np.sum(lik) + 1e-12)