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
from .prepare_stan_data import prepare_stan_data


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
    )

    return fit


def print_diagnostics(fit: CmdStanMCMC) -> None:
    print("\n=== Sampler Diagnostics ===")
    print(fit.diagnose())

    print("\n=== Summary Statistics ===")
    print(fit.summary().to_string())


# ------------------------------------------------------------
# ✅ High-level pipeline
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
) -> tuple[pd.DataFrame, CmdStanMCMC]:
    """
    Full end-to-end pipeline:
        1. Load detector CSVs
        2. Prepare Stan data
        3. Compile Stan model
        4. Run sampling

    Returns
    -------
    df : pd.DataFrame
        The merged detector data with timestamps and covariates.
    fit : CmdStanMCMC
        The fitted Stan model.
    """

    # Step 1: Load CSVs
    df = load_detector_csvs(csv_dir)

    # Step 2: Prepare Stan data
    stan_data = prepare_stan_data(df)

    # Step 3: Compile model
    model = compile_call_intensity_model(stan_model_path)

    # Step 4: Fit model
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