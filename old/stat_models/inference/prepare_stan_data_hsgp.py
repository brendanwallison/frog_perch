from __future__ import annotations
import pandas as pd
import numpy as np
from scipy.stats import beta

# No longer need patsy
# from patsy import dmatrix

REQUIRED_COLUMNS = {
    "prob",
    "day_index",
    "time_of_day_hours",
}

def compute_log_odds(p, a_call, b_call, a_bg, b_bg):
    """Compute ell_i = log f_call(p_i) - log f_bg(p_i)."""
    log_f_call = beta.logpdf(p, a_call, b_call)
    log_f_bg   = beta.logpdf(p, a_bg, b_bg)
    return log_f_call - log_f_bg


def make_hsgp_basis(x_vals: np.ndarray, M: int) -> np.ndarray:
    """
    Non-periodic HSGP basis on a bounded interval [L, R].
    Produces shape (N, 2*M):
      [sin(pi*x), cos(pi*x), sin(2*pi*x), cos(2*pi*x), ...]
    after mapping x to [0,1].
    """
    x_vals = np.asarray(x_vals)
    N = len(x_vals)
    X = np.zeros((N, 2*M))
    
    L = float(np.min(x_vals))
    R = float(np.max(x_vals))
    x_scaled = (x_vals - L) / (R - L)
    
    for m in range(1, M+1):
        X[:, 2*(m-1)]     = np.sin(np.pi * m * x_scaled)
        X[:, 2*(m-1) + 1] = np.cos(np.pi * m * x_scaled)
    
    return X


def prepare_stan_data_hsgp(
    df: pd.DataFrame,
    *,
    a_call: float,
    b_call: float,
    a_bg: float,
    b_bg: float,
    n_ell_bins: int = 10,
    M_season: int = 10,  # Number of frequencies (replaces K_season)
    M_diel: int = 10,    # Number of frequencies (replaces K_diel)
) -> dict:

    df = df.copy()

    # 1. Compute ell_i
    df["ell"] = compute_log_odds(
        df["prob"].to_numpy(),
        a_call=a_call,
        b_call=b_call,
        a_bg=a_bg,
        b_bg=b_bg,
    )

    # 2. Compute Time Indices
    # Minute of day (0-1439)
    df["minute_of_day"] = (df["time_of_day_hours"] * 60).astype(int)
    
    # Day of year (1-365)
    df["date"] = pd.to_datetime(df["date"])
    df["day_of_year"] = df["date"].dt.dayofyear.astype(int)

    # 3. Create ell quantile bins
    df["ell_bin"] = pd.qcut(
        df["ell"],
        q=n_ell_bins,
        labels=False,
        duplicates="drop",
    )
    df = df.dropna(subset=["ell_bin"])
    df["ell_bin"] = df["ell_bin"].astype(int)

    # ------------------------------------------------------------
    # 4. Generate HSGP Fourier Design Matrices
    # ------------------------------------------------------------
    obs_days = np.sort(df["day_of_year"].unique())
    obs_mins = np.sort(df["minute_of_day"].unique())

    X_season = make_hsgp_basis(obs_days, M=M_season)
    X_diel   = make_hsgp_basis(obs_mins, M=M_diel)


    # 5. Create Index Maps (Value -> Matrix Row Index 1..N)
    # Stan uses 1-based indexing
    day_map = {val: i + 1 for i, val in enumerate(obs_days)}
    min_map = {val: i + 1 for i, val in enumerate(obs_mins)}

    df["day_idx"] = df["day_of_year"].map(day_map)
    df["diel_idx"] = df["minute_of_day"].map(min_map)

    # ------------------------------------------------------------
    # 6. Group into Bins for Likelihood
    # ------------------------------------------------------------
    grouped = df.groupby(["day_idx", "diel_idx", "ell_bin"])

    B = len(grouped)
    
    day_idx_vec = []
    diel_idx_vec = []
    ell_bin_vals = []
    n_bin = []

    for (d_idx, m_idx, ebin), g in grouped:
        day_idx_vec.append(int(d_idx))
        diel_idx_vec.append(int(m_idx))
        ell_bin_vals.append(float(g["ell"].mean()))
        n_bin.append(int(len(g)))

    # ------------------------------------------------------------
    # 7. Return Final Data
    # ------------------------------------------------------------
    print(f"[prepare_stan_data] Generated HSGP: Season (D_obs={len(obs_days)}, M={M_season}), Diel (M_obs={len(obs_mins)}, M={M_diel})")

    return {
        "B": B,
        "M_obs": len(obs_mins),
        "D_obs": len(obs_days),
        
        # HSGP Config
        "M_season": M_season,
        "M_diel": M_diel,
        
        # Design Matrices (Fourier)
        "X_season": X_season,
        "X_diel": X_diel,

        # Data Bins
        "day_idx": day_idx_vec,   # Maps bin -> row in X_season
        "diel_idx": diel_idx_vec, # Maps bin -> row in X_diel
        "ell_bin": ell_bin_vals,
        "n_bin": n_bin,
    }


def prepare_stan_data(
    df: pd.DataFrame,
    *,
    a_call: float,
    b_call: float,
    a_bg: float,
    b_bg: float,
    use_binning: bool = True,
    # Renamed arguments for clarity, but you can map K->M if you want to keep CLI same
    M_season: int = 10, 
    M_diel: int = 10,
) -> dict:

    if use_binning:
        print("[prepare_stan_data] Using HSGP/BINNED mode.")
        return prepare_stan_data_hsgp(
            df,
            a_call=a_call,
            b_call=b_call,
            a_bg=a_bg,
            b_bg=b_bg,
            # Map the "K" arguments from CLI to "M" for HSGP
            M_season=M_season,
            M_diel=M_diel,
        )
    else:
        raise NotImplementedError("Slice mode is not yet compatible with the HSGP Stan model.")