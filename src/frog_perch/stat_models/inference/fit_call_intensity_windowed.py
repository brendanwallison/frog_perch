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
# Point this to wherever you saved the windowed data prep logic.
# If you overwrote prepare_stan_data.py, this is correct.
from .prepare_stan_data_windowed import prepare_stan_data
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

    use_binning: bool = False,
    
    # Spline Knots (Degrees of Freedom)
    K_season: int = 20,
    K_diel: int = 20,
) -> tuple[pd.DataFrame, CmdStanMCMC]:

    df = load_detector_csvs(csv_dir)

    # Pass the K arguments to the data prep function
    stan_data = prepare_stan_data(
        df,
        a_call=a_call,
        b_call=b_call,
        a_bg=a_bg,
        b_bg=b_bg,
        use_binning=use_binning,
        K_season=K_season,
        K_diel=K_diel,
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