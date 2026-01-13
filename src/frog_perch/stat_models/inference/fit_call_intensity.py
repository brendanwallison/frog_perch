"""
Stan fitting utilities for the call-intensity model.

This module provides:
    - low-level wrappers for compiling and sampling
    - a high-level pipeline that loads CSVs, prepares Stan data,
      compiles the model, and runs sampling
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any

from cmdstanpy import CmdStanModel, CmdStanMCMC

from .data_loading import load_detector_csvs
# Ensure this matches your actual filename. 
# If you created a new file for HSGP, update this import (e.g. .prepare_stan_data_hsgp).
# If you edited the existing file, keep it as is.
from .prepare_stan_data_hsgp import prepare_stan_data
import pandas as pd


# ------------------------------------------------------------
# Low-level utilities (unchanged)
# ------------------------------------------------------------

def compile_call_intensity_model(stan_path: Path) -> CmdStanModel:
    if not stan_path.exists():
        raise FileNotFoundError(f"Stan model not found: {stan_path}")
    return CmdStanModel(stan_file=str(stan_path))


def fit_call_intensity(
    stan_model: CmdStanModel,
    stan_data: Dict[str, Any],
    *,
    iter_warmup: int = 1000,
    iter_sampling: int = 1000,
    chains: int = 4,
    parallel_chains: Optional[int] = None,
    seed: int = 12345,
    show_progress: bool = True,
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
        refresh=1,   # print every iteration
    )

    return fit


def print_diagnostics(fit: CmdStanMCMC) -> None:
    print("\n=== Sampler Diagnostics ===")
    print(fit.diagnose())

    print("\n=== Summary Statistics ===")
    print(fit.summary().to_string())


# ------------------------------------------------------------
# ✅ High-level pipeline (UPDATED)
# ------------------------------------------------------------

def run_call_intensity_pipeline(
    csv_dir: Path,
    stan_model_path: Path,
    *,
    iter_warmup: int = 1000,
    iter_sampling: int = 1000,
    chains: int = 4,
    parallel_chains: Optional[int] = None,
    seed: int = 12345,
    show_progress: bool = True,

    a_call: float,
    b_call: float,
    a_bg: float,
    b_bg: float,

    use_binning: bool = True,
    
    # NEW ARGS: Use M (frequencies) instead of K (knots)
    M_season: int = 10,
    M_diel: int = 10,
) -> tuple[pd.DataFrame, CmdStanMCMC]:

    df = load_detector_csvs(csv_dir)

    # Note: If your prepare_stan_data wrapper still uses "K_season" arguments
    # (as defined in the previous step for backward compatibility), 
    # we map M -> K here.
    stan_data = prepare_stan_data(
        df,
        a_call=a_call,
        b_call=b_call,
        a_bg=a_bg,
        b_bg=b_bg,
        
        # Pass M_season to the data prep function
        # (Assuming the wrapper function signature is: K_season=...)
        M_season=M_season,
        M_diel=M_diel,
    )

    model = compile_call_intensity_model(stan_model_path)

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

    return df, fit