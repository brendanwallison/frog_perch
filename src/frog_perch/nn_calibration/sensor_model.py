"""
frog_perch/calibration/sensor_model.py
Core generative math for the acoustic sensor calibration.
"""
import numpy as np
from scipy.special import expit as sigmoid
from scipy.stats import beta

def calculate_likelihood_vector(y_raw, nu_obs, x_interf, p, k_max=16.0, return_log_likelihoods=False):
    """
    Computes the normalized probability vector P(k | y_raw, nu_obs, x_interf).
    Used uniformly across calibration, visualization, and data prep.
    
    If return_log_likelihoods=True, returns the raw un-normalized log values.
    """
    k_vec = np.arange(int(k_max) + 1)
    phi_nu = np.clip(np.exp(p["ln_phi_nu"]), 1.0, 100.0) # Added clip for optimizer stability
    
    y_norm = np.clip(y_raw / k_max, 0.001, 0.999)
    nu_obs_safe = np.clip(nu_obs, 0.01, 0.99)
    
    # Generative Means
    noise_floor = sigmoid(p["b0"] + p["b_i"] * x_interf)
    delta = np.exp(p["g0"] - p["g_i"] * x_interf)
    m_y_norm = noise_floor + (1 - noise_floor) * np.tanh(k_vec / delta)
    m_y_norm = np.clip(m_y_norm, 1e-6, 1.0 - 1e-6)
    
    m_nu = sigmoid(p["a0"] - p["a_f"] * k_vec + p["a_i"] * x_interf)
    m_nu = np.clip(m_nu, 1e-6, 1.0 - 1e-6)
    
    # Precision Modulation
    phi_y = np.clip(np.exp(p["c0"] - p["c_u"] * np.log(nu_obs_safe + 1e-6)), 1.0, 100.0)
    
    # Log Likelihoods
    log_l_y = beta.logpdf(y_norm, m_y_norm * phi_y, (1 - m_y_norm) * phi_y)
    log_l_nu = beta.logpdf(nu_obs_safe, m_nu * phi_nu, (1 - m_nu) * phi_nu)
    
    combined = np.nan_to_num(log_l_y + log_l_nu, nan=-50.0)
    
    if return_log_likelihoods:
        return combined
        
    lik = np.exp(combined - np.max(combined))
    return lik / (np.sum(lik) + 1e-12)