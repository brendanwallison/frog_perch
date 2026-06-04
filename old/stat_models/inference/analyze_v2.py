#!/usr/bin/env python3
"""
Preliminary analysis for call_intensity_v2 (binned harmonic model).

Assumes:
- CmdStan CSV outputs are in:   stat_results/call_intensity_v2
- Merged detector data is in:   stat_results/call_intensity_v2/merged_detector_data.csv
- Stan model is the harmonic binned model from call_intensity_v2.stan.

This script:
  * Loads the fit via CmdStanPy (from_csv)
  * Summarizes sampler diagnostics (divergences, ESS, R-hat, treedepth, rough BFMI)
  * Reconstructs seasonal and diel curves on grids
  * Saves basic plots into the same results directory
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cmdstanpy import CmdStanMCMC, from_csv


# ---------------------------------------------------------------------
# Paths and basic config
# ---------------------------------------------------------------------

RESULTS_DIR = Path("/home/breallis/dev/frog_perch/stat_results/call_intensity_v2")
MERGED_CSV = RESULTS_DIR / "merged_detector_data.csv"

CSV_GLOB_PATTERN = "call_intensity_v2-*.csv"

N_SEASONAL_POINTS = 365
N_DIEL_POINTS = 1440


# ---------------------------------------------------------------------
# Load fit
# ---------------------------------------------------------------------

def load_fit_from_dir(results_dir: Path, csv_pattern: str) -> CmdStanMCMC:
    csv_files = sorted(results_dir.glob(csv_pattern))
    if not csv_files:
        raise RuntimeError(f"No CSV files matching {csv_pattern} in {results_dir}")
    fit = from_csv([str(f) for f in csv_files])
    return fit


# ---------------------------------------------------------------------
# Diagnostics using only attributes visible in dir(fit)
# ---------------------------------------------------------------------

def print_basic_diagnostics(fit: CmdStanMCMC) -> None:
    # Diagnose text (truncate for sanity)
    diag = fit.diagnose()
    diag_lines = diag.splitlines()
    print("\n=== Sampler diagnostics (first 40 lines) ===")
    print("\n".join(diag_lines[:40]))
    if len(diag_lines) > 40:
        print(f"... (truncated {len(diag_lines) - 40} lines)")

    # Summary table
    summary = fit.summary()
    print("\n=== Summary (head) ===")
    print(summary.head(20).to_string())
    print("\n=== Summary (tail) ===")
    print(summary.tail(10).to_string())
    print(f"\nTotal parameters: {summary.shape[0]}")

    # Divergences: fit.divergences is (chains, draws) of booleans
    div = fit.divergences
    print("\n=== Divergences (per chain) ===")
    for c in range(div.shape[0]):
        print(f"Chain {c+1}: {int(div[c].sum())} divergences")

    # Treedepth: fit.max_treedepths is (chains, draws) of ints
    td = fit.max_treedepths
    max_td = int(td.max())
    print("\n=== Treedepth saturation (per chain) ===")
    for c in range(td.shape[0]):
        hits = int((td[c] == max_td).sum())
        print(f"Chain {c+1}: {hits} transitions hit treedepth {max_td}")

    # Rough BFMI using energy__ from draws() if present
    # draws() returns array (chains, draws, num_cols)
    # column_names() gives names; we look for 'energy__'
    cols = fit.column_names
    # --- BFMI computation (one value per chain) ---
    try:
        # CmdStanPy stores per-draw kinetic energy in fit.bfmi with shape (chains, draws)
        bfmi_raw = fit.bfmi  # shape (chains, draws)

        print("\n=== BFMI (per chain) ===")
        for c in range(bfmi_raw.shape[0]):
            e = bfmi_raw[c, :]              # kinetic energy per draw
            var_e = np.var(e)

            if var_e == 0:
                bfmi = np.nan
            else:
                bfmi = np.var(np.diff(e)) / var_e

            print(f"Chain {c+1}: BFMI ≈ {bfmi:.3f}")

    except Exception:
        print("\n=== BFMI ===")
        print("Could not compute BFMI (no energy__ column or unexpected structure).")

    # R-hat / ESS for key parameters
    print("\n=== R-hat and ESS (selected parameters) ===")
    interesting = [
        "beta0_year",
        "beta_cos_year[1]",
        "beta_cos_year[2]",
        "beta_sin_year[1]",
        "beta_sin_year[2]",
        "alpha0_tod",
        "alpha_cos_tod[1]",
        "alpha_cos_tod[2]",
        "alpha_sin_tod[1]",
        "alpha_sin_tod[2]",
        "sigma_day_dev",
        "sigma_min_dev",
    ]
    for param in interesting:
        if param in summary.index:
            row = summary.loc[param]
            print(
                f"{param:20s}  "
                f"Rhat={row['R_hat']:.3f}  "
                f"ESS_bulk={row['ESS_bulk']:.1f}  "
                f"ESS_tail={row['ESS_tail']:.1f}"
            )
        else:
            print(f"{param:20s}  (not found in summary index)")


# ---------------------------------------------------------------------
# Extract parameter draws
# ---------------------------------------------------------------------

def extract_param_draws(fit: CmdStanMCMC) -> dict:
    """
    Use the supported stan_variables() API instead of stan_variable_names.
    Returns a dict: name -> draws array (draws, ...).
    """
    return fit.stan_variables()


# ---------------------------------------------------------------------
# Harmonic reconstructions (mirror Stan logic)
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Harmonic reconstructions (mirror Stan logic, 1 harmonic)
# ---------------------------------------------------------------------

def seasonal_harmonic_py(s, beta0_year, beta_cos_year, beta_sin_year):
    s = np.asarray(s)
    angle1 = 2 * np.pi * s / 365.0
    return (
        beta0_year
        + beta_cos_year[0] * np.cos(angle1)
        + beta_sin_year[0] * np.sin(angle1)
    )


def diel_harmonic_py(t_hours, alpha0_tod, alpha_cos_tod, alpha_sin_tod):
    t_hours = np.asarray(t_hours)
    angle1 = 2 * np.pi * t_hours / 24.0
    return (
        alpha0_tod
        + alpha_cos_tod[0] * np.cos(angle1)
        + alpha_sin_tod[0] * np.sin(angle1)
    )

def summarize_curve(draws_curve: np.ndarray) -> dict:
    """
    draws_curve: array of shape (n_draws, n_points)
    Returns dict with mean, median, and 50/95% bands.
    """
    q = np.quantile(draws_curve, [0.025, 0.25, 0.5, 0.75, 0.975], axis=0)
    return {
        "mean": draws_curve.mean(axis=0),
        "q025": q[0],
        "q25": q[1],
        "median": q[2],
        "q75": q[3],
        "q975": q[4],
    }


def plot_with_bands(x, summary: dict, title: str, xlabel: str, ylabel: str, out_path: Path):
    plt.figure(figsize=(8, 4))
    plt.fill_between(x, summary["q025"], summary["q975"],
                     color="lightgray", alpha=0.7, label="95% CI")
    plt.fill_between(x, summary["q25"], summary["q75"],
                     color="gray", alpha=0.7, label="50% CI")
    plt.plot(x, summary["median"], color="black", lw=1.5, label="median")
    plt.plot(x, summary["mean"], color="blue", lw=1, alpha=0.7, label="mean")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

# ---------------------------------------------------------------------
# Reconstruct seasonal and diel curves from posterior draws (1 harmonic)
# ---------------------------------------------------------------------

def reconstruct_seasonal_components(draws: dict, day_of_year: np.ndarray):
    """
    Returns three summaries over a seasonal grid:
      1) harmonic-only seasonal probability
      2) day-effect-only probability (no intercept)
      3) harmonic + day-effect probability
    """

    beta0_year    = draws["beta0_year"]        # (draws,)
    beta_cos_year = draws["beta_cos_year"]     # (draws, 1)
    beta_sin_year = draws["beta_sin_year"]     # (draws, 1)
    eta_day       = draws["eta_day"]           # (draws, D)

    n_draws, D = eta_day.shape

    # --- 1) Harmonic-only seasonal probability on a smooth grid ---
    s_grid = np.linspace(1, 365, N_SEASONAL_POINTS)
    seasonal_logit_harm = np.empty((n_draws, N_SEASONAL_POINTS))

    for i in range(n_draws):
        seasonal_logit_harm[i, :] = seasonal_harmonic_py(
            s_grid,
            beta0_year[i],
            beta_cos_year[i, :],
            beta_sin_year[i, :],
        )

    seasonal_prob_harm = 1 / (1 + np.exp(-seasonal_logit_harm))
    seasonal_harm_summary = summarize_curve(seasonal_prob_harm)
    seasonal_harm_summary["x"] = s_grid

    # --- 2) Day-effect-only probability (mapped to DOY) ---
    day_of_year = np.asarray(day_of_year)
    if day_of_year.shape[0] != D:
        raise ValueError("day_of_year length mismatch with eta_day")

    s_unique = np.sort(np.unique(day_of_year))
    eta_day_on_doy = np.empty((n_draws, s_unique.shape[0]))

    for j, s in enumerate(s_unique):
        idx = np.where(day_of_year == s)[0]
        if idx.size == 0:
            eta_day_on_doy[:, j] = 0.0
        else:
            eta_day_on_doy[:, j] = eta_day[:, idx].mean(axis=1)

    day_prob_only = 1 / (1 + np.exp(-eta_day_on_doy))
    day_only_summary = summarize_curve(day_prob_only)
    day_only_summary["x"] = s_unique

    # --- 3) Combined seasonal probability ---
    seasonal_logit_harm_at_unique = np.empty_like(eta_day_on_doy)
    for i in range(n_draws):
        seasonal_logit_harm_at_unique[i, :] = seasonal_harmonic_py(
            s_unique,
            beta0_year[i],
            beta_cos_year[i, :],
            beta_sin_year[i, :],
        )

    combined_logit = seasonal_logit_harm_at_unique + eta_day_on_doy
    combined_prob = 1 / (1 + np.exp(-combined_logit))

    combined_summary = summarize_curve(combined_prob)
    combined_summary["x"] = s_unique

    return seasonal_harm_summary, day_only_summary, combined_summary

def reconstruct_diel_components(draws: dict, t_minutes: np.ndarray) -> tuple[dict, dict, dict]:
    """
    Returns three summaries over minute-of-day:
      1) harmonic-only diel probability
      2) minute-of-day-effect-only probability
      3) harmonic + minute-of-day effect probability

    New model: minute-of-day deviations are invariant across days.
    We work directly on the 0..1439 grid, not per-observation.
    """

    alpha0_tod    = draws["alpha0_tod"]        # (draws,)
    alpha_cos_tod = draws["alpha_cos_tod"]     # (draws, 1)
    alpha_sin_tod = draws["alpha_sin_tod"]     # (draws, 1)

    delta_minute_raw = draws["delta_minute_raw"]  # (draws, 1440)
    sigma_min_dev    = draws["sigma_min_dev"]     # (draws,)

    n_draws, n_minutes = delta_minute_raw.shape
    assert n_minutes == 1440, f"Expected 1440 minute-of-day effects, got {n_minutes}"

    # 1) Harmonic-only curve on a fine 0..1439 grid
    minutes_grid = np.arange(n_minutes)          # 0..1439
    t_hours_grid = minutes_grid / 60.0

    diel_logits_harm = np.empty((n_draws, n_minutes))
    for i in range(n_draws):
        diel_logits_harm[i, :] = diel_harmonic_py(
            t_hours_grid,
            alpha0_tod[i],
            alpha_cos_tod[i, :],
            alpha_sin_tod[i, :],
        )

    diel_probs_harm = 1.0 / (1.0 + np.exp(-diel_logits_harm))
    diel_harm_summary = summarize_curve(diel_probs_harm)
    diel_harm_summary["x_minutes"] = minutes_grid
    diel_harm_summary["x_hours"]   = t_hours_grid

    # 2) Minute-of-day-only effect (invariant across days)
    #    delta_minute_raw is non-centered; center it:
    #    delta_minute = sigma_min_dev * delta_minute_raw
    delta_minute = sigma_min_dev[:, None] * delta_minute_raw    # (draws, 1440)

    minute_probs_only = 1.0 / (1.0 + np.exp(-delta_minute))
    minute_only_summary = summarize_curve(minute_probs_only)
    minute_only_summary["x_minutes"] = minutes_grid
    minute_only_summary["x_hours"]   = t_hours_grid

    # 3) Combined logits: harmonic + minute-of-day effect
    diel_logits_combined = diel_logits_harm + delta_minute
    diel_probs_combined  = 1.0 / (1.0 + np.exp(-diel_logits_combined))

    combined_summary = summarize_curve(diel_probs_combined)
    combined_summary["x_minutes"] = minutes_grid
    combined_summary["x_hours"]   = t_hours_grid

    return diel_harm_summary, minute_only_summary, combined_summary


# ---------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------

def plot_with_bands(x, summary: dict, title: str, xlabel: str, ylabel: str, out_path: Path):
    """
    Generic uncertainty-band plotter for any 1D posterior summary.
    Expects summary to contain: mean, median, q025, q25, q75, q975.
    """
    plt.figure(figsize=(8, 4))

    # 95% band
    plt.fill_between(
        x, summary["q025"], summary["q975"],
        color="lightgray", alpha=0.7, label="95% CI"
    )

    # 50% band
    plt.fill_between(
        x, summary["q25"], summary["q75"],
        color="gray", alpha=0.7, label="50% CI"
    )

    # Median + mean
    plt.plot(x, summary["median"], color="black", lw=1.5, label="median")
    plt.plot(x, summary["mean"], color="blue", lw=1, alpha=0.7, label="mean")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def run_plots(draws: dict, results_dir: Path, day_of_year: np.ndarray, t_minutes: np.ndarray):
    """
    Produces all six decomposition plots:
      Seasonal:
        - harmonic only
        - day effect only
        - harmonic + day effect
      Diel:
        - harmonic only
        - minute effect only
        - harmonic + minute effect
    Plus sigma posterior histograms.
    """

    # ============================================================
    # Seasonal decomposition
    # ============================================================
    seasonal_harm, day_only, seasonal_combined = reconstruct_seasonal_components(
        draws, day_of_year
    )

    plot_with_bands(
        x=seasonal_harm["x"],
        summary=seasonal_harm,
        title="Seasonal call probability – harmonic only",
        xlabel="Day of year",
        ylabel="P(call)",
        out_path=results_dir / "seasonal_harmonic_only.png",
    )

    plot_with_bands(
        x=day_only["x"],
        summary=day_only,
        title="Seasonal call probability – day effect only",
        xlabel="Day of year",
        ylabel="P(call)",
        out_path=results_dir / "seasonal_day_only.png",
    )

    plot_with_bands(
        x=seasonal_combined["x"],
        summary=seasonal_combined,
        title="Seasonal call probability – harmonic + day effect",
        xlabel="Day of year",
        ylabel="P(call)",
        out_path=results_dir / "seasonal_harmonic_plus_day.png",
    )

    # ============================================================
    # Diel decomposition
    # ============================================================
    diel_harm, minute_only, diel_combined = reconstruct_diel_components(
        draws, t_minutes
    )

    plot_with_bands(
        x=diel_harm["x_hours"],
        summary=diel_harm,
        title="Diel call probability – harmonic only",
        xlabel="Time of day (hours)",
        ylabel="P(call)",
        out_path=results_dir / "diel_harmonic_only.png",
    )

    plot_with_bands(
        x=minute_only["x_hours"],
        summary=minute_only,
        title="Diel call probability – minute effect only",
        xlabel="Time of day (hours)",
        ylabel="P(call)",
        out_path=results_dir / "diel_minute_only.png",
    )

    plot_with_bands(
        x=diel_combined["x_hours"],
        summary=diel_combined,
        title="Diel call probability – harmonic + minute effect",
        xlabel="Time of day (hours)",
        ylabel="P(call)",
        out_path=results_dir / "diel_harmonic_plus_minute.png",
    )

    # ============================================================
    # Sigma posteriors
    # ============================================================
    for param in ["sigma_day_dev", "sigma_min_dev"]:
        if param in draws:
            plt.figure(figsize=(6, 4))
            plt.hist(draws[param], bins=50, density=True, alpha=0.7)
            plt.title(f"Posterior of {param}")
            plt.xlabel(param)
            plt.ylabel("Density")
            plt.tight_layout()
            plt.savefig(results_dir / f"{param}_hist.png", dpi=200)
            plt.close()

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    print(f"Loading fit from {RESULTS_DIR} ...")
    fit = load_fit_from_dir(RESULTS_DIR, CSV_GLOB_PATTERN)

    print("Loading merged detector data (if needed) ...")
    if MERGED_CSV.exists():
        df_merged = pd.read_csv(MERGED_CSV)
        print(f"Merged data shape: {df_merged.shape}")
    else:
        df_merged = None
        print("No merged_detector_data.csv found.")

    print_basic_diagnostics(fit)

    draws = extract_param_draws(fit)

    print("\nReconstructing seasonal and diel curves and writing plots...")
    D = draws["eta_day"].shape[1]
    M_obs = draws["delta_min"].shape[1]

    day_of_year = np.linspace(1, 365, D)
    t_minutes = np.linspace(0, 1439, M_obs)   # synthetic minute-of-day index

    run_plots(draws, RESULTS_DIR, day_of_year, t_minutes)

    print("\nDone. Plots and diagnostics written to:")
    print(f"  {RESULTS_DIR}")


if __name__ == "__main__":
    main()