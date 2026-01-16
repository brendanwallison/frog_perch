"""
Stan fitting utilities for the call-intensity model (Likelihood Profile / HSGP).

This module provides:
    - low-level wrappers for compiling and sampling
    - a high-level pipeline that loads CSVs, prepares aggregated likelihood profiles,
      compiles the model, and runs sampling.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

from cmdstanpy import CmdStanModel, CmdStanMCMC
import pandas as pd

from .data_loading import load_detector_csvs
# Ensure this filename matches where you saved the preprocessing script
from .prepare_stan_data_likelihood_profile import prepare_stan_data_vectorized as prepare_stan_data


# ------------------------------------------------------------
# Low-level utilities (Standard CmdStanPy wrappers)
# ------------------------------------------------------------

def compile_call_intensity_model(stan_path: Path) -> CmdStanModel:
    if not stan_path.exists():
        raise FileNotFoundError(f"Stan model not found: {stan_path}")
    return CmdStanModel(stan_file=str(stan_path))


def fit_call_intensity(
    stan_model: CmdStanModel,
    stan_data: Dict[str, Any],
    *,
    iter_warmup: int = 100,
    iter_sampling: int = 100,
    chains: int = 4,
    parallel_chains: Optional[int] = None,
    seed: int = 12345,
    show_progress: bool = True,
    refresh: int = 1,  # Print progress every N iterations
) -> CmdStanMCMC:

    if parallel_chains is None:
        parallel_chains = chains

    fit = stan_model.sample(
        data=stan_data,
        iter_warmup=iter_warmup,
        iter_sampling=iter_sampling,
        chains=chains,
        parallel_chains=parallel_chains,
        seed=seed,
        show_progress=show_progress,
        refresh=refresh, 
    )

    return fit


def print_diagnostics(fit: CmdStanMCMC) -> None:
    print("\n=== Sampler Diagnostics ===")
    print(fit.diagnose())

    print("\n=== Summary Statistics ===")
    # Print summary of key parameters (exclude huge vectors like w_obs if they appear)
    summary = fit.summary()
    # Filter to show main latent parameters
    vars_to_show = [v for v in summary.index if not v.startswith("w_obs")]
    print(summary.loc[vars_to_show].to_string())


# ------------------------------------------------------------
# High-level pipeline (Likelihood Profile Version)
# ------------------------------------------------------------

def run_call_intensity_pipeline(
    csv_dir: Path,
    stan_model_path: Path,
    *,
    # MCMC settings
    iter_warmup: int = 1000,
    iter_sampling: int = 1000,
    chains: int = 4,
    parallel_chains: Optional[int] = None,
    seed: int = 12345,
    show_progress: bool = True,

    # Calibration parameters (Beta distribution params)
    a_call: float,
    b_call: float,
    a_bg: float,
    b_bg: float,

    # Aggregation settings (Likelihood Profile)
    bin_duration_sec: float = 0.2,  # e.g., 200ms sub-second bins
    window_length_sec: float = 5.0, # e.g., 5s analysis window
    
    # HSGP settings
    M_season: int = 10,
    M_diel: int = 10,
    
    cache_file: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, CmdStanMCMC]: 
    
    print(f"Loading data from {csv_dir}...")
    df = load_detector_csvs(csv_dir)

    print("Preparing Likelihood Profile data (aggregating sub-second bins)...")
    
    stan_data, window_metadata,hsgsp_params = prepare_stan_data(
        df,
        a_call=a_call,
        b_call=b_call,
        a_bg=a_bg,
        b_bg=b_bg,
        bin_duration_sec=bin_duration_sec,
        window_length_sec=window_length_sec,
        cache_file=cache_file,
    )
  
    # Optional: Verify data shapes before fitting
    print(f"  > Windows (T): {stan_data['T']}")
    print(f"  > Bins per window (N): {stan_data['N'] if 'N' in stan_data else 'Variable'}")

    print(f"Compiling Stan model from {stan_model_path}...")
    model = compile_call_intensity_model(stan_model_path)

    print("Starting Sampling...")
    fit = fit_call_intensity(
        model,
        stan_data,
        iter_warmup=iter_warmup,
        iter_sampling=iter_sampling,
        chains=chains,
        parallel_chains=parallel_chains,
        seed=seed,
        show_progress=show_progress,
    )

    # Return: (Raw Bins, Window Metadata, Stan Fit)
    return df, window_metadata, hsgsp_params, fit