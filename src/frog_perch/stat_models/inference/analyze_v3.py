#!/usr/bin/env python3
"""
Analysis script for call_intensity_v3 (periodic seasonal × diel RW2 model).

- Seasonal period = 365 (true day-of-year)
- Diel period = 1440 (true minute-of-day)
- Reconstructs seasonal and diel probability curves
- Produces unified diagnostics: ESS/Rhat summaries + divergence-based pairs plots
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cmdstanpy import CmdStanMCMC, from_csv

# ---------------------------------------------------------------------
# Paths and config
# ---------------------------------------------------------------------

RESULTS_DIR = Path("/home/breallis/dev/frog_perch/stat_results/call_intensity_v3")
MERGED_CSV = RESULTS_DIR / "merged_detector_data.csv"
CSV_GLOB_PATTERN = "call_intensity_v3-*.csv"

# Diel and seasonal indices for high-data / no-data regions
MINUTE_HIGH = 19 * 60   # hour 19
MINUTE_LOW  = 5 * 60    # hour 5
DAY_HIGH    = 275
DAY_LOW     = 150

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def inv_logit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def load_fit(results_dir: Path, pattern: str) -> CmdStanMCMC:
    csvs = sorted(results_dir.glob(pattern))
    if not csvs:
        raise RuntimeError(f"No CmdStan CSV files found in {results_dir} matching {pattern}")
    return from_csv([str(f) for f in csvs])


def summarize_curve(draws_curve: np.ndarray) -> dict:
    """
    draws_curve: (n_draws, dim)
    """
    q = np.quantile(draws_curve, [0.025, 0.25, 0.5, 0.75, 0.975], axis=0)
    return {
        "mean":   draws_curve.mean(axis=0),
        "q025":   q[0],
        "q25":    q[1],
        "median": q[2],
        "q75":    q[3],
        "q975":   q[4],
    }


def plot_with_bands(
    x: np.ndarray,
    summary: dict,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
) -> None:
    plt.figure(figsize=(8, 4))
    plt.fill_between(x, summary["q025"], summary["q975"], color="lightgray", alpha=0.7)
    plt.fill_between(x, summary["q25"], summary["q75"], color="gray", alpha=0.7)
    plt.plot(x, summary["median"], color="black", lw=1.5, label="median")
    plt.plot(x, summary["mean"], color="blue", lw=1, alpha=0.7, label="mean")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ---------------------------------------------------------------------
# Divergence helpers
# ---------------------------------------------------------------------

def get_divergent_mask(fit):
    """
    Return a boolean mask of shape (n_draws_total,) indicating which draws diverged.
    Uses fit.method_variables(), which contains sampler diagnostics.
    """
    div = fit.method_variables()["divergent__"]   # shape (draws, chains)
    return (div == 1).reshape(-1)


def extract_scalar(fit: CmdStanMCMC, name: str) -> np.ndarray:
    """
    Extract a scalar parameter (possibly per-draw, per-chain) and flatten to (n_draws_total,).
    """
    arr = fit.stan_variable(name)
    return arr.reshape(-1)


def extract_element(fit: CmdStanMCMC, name: str, index: int) -> np.ndarray:
    """
    Extract element 'index' (0-based) from a vector parameter and flatten to (n_draws_total,).
    """
    arr = fit.stan_variable(name)  # shape (draws_total, dim) in CmdStanPy
    return arr[:, index]


def plot_divergence_pairs(
    x: np.ndarray,
    y: np.ndarray,
    div_mask: np.ndarray,
    xlabel: str,
    ylabel: str,
    outpath: Path,
) -> None:
    plt.figure(figsize=(6, 6))
    plt.scatter(x[~div_mask], y[~div_mask], s=6, alpha=0.3, color="gray", label="non-divergent")
    plt.scatter(x[div_mask],  y[div_mask],  s=12, alpha=0.9, color="red",  label="divergent")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_sum_vs_components(fit: CmdStanMCMC, outdir: Path,
                           day: int, minute: int, label: str) -> None:
    div = get_divergent_mask(fit)

    s_d = extract_element_flat(fit, "s", day)
    u_d = extract_element_flat(fit, "u_day", day)
    g_m = extract_element_flat(fit, "g", minute)

    eta = s_d + u_d + g_m

    # eta vs s
    plot_divergence_pairs(
        eta, s_d, div,
        xlabel=f"eta[{day},{minute}] = s+u+g",
        ylabel=f"s[{day}]",
        outpath=outdir / f"div_eta_vs_s_{label}.png",
    )

    # eta vs u
    plot_divergence_pairs(
        eta, u_d, div,
        xlabel=f"eta[{day},{minute}] = s+u+g",
        ylabel=f"u_day[{day}]",
        outpath=outdir / f"div_eta_vs_u_{label}.png",
    )

    # eta vs g
    plot_divergence_pairs(
        eta, g_m, div,
        xlabel=f"eta[{day},{minute}] = s+u+g",
        ylabel=f"g[{minute}]",
        outpath=outdir / f"div_eta_vs_g_{label}.png",
    )

def extract_element_flat(fit: CmdStanMCMC, name: str, index: int) -> np.ndarray:
    # CmdStanPy gives per-draw arrays; just take column and flatten across chains.
    arr = fit.stan_variable(name)  # shape (draws_total, dim)
    return arr[:, index]


def plot_nonident_triplet(fit: CmdStanMCMC, outdir: Path,
                          day: int, minute: int, label: str) -> None:
    div = get_divergent_mask(fit)

    s_d = extract_element_flat(fit, "s", day)       # 0-based day index
    u_d = extract_element_flat(fit, "u_day", day)
    g_m = extract_element_flat(fit, "g", minute)

    # s vs u
    plot_divergence_pairs(
        s_d, u_d, div,
        xlabel=f"s[{day}]",
        ylabel=f"u_day[{day}]",
        outpath=outdir / f"div_s_vs_u_day_{label}.png",
    )

    # s vs g
    plot_divergence_pairs(
        s_d, g_m, div,
        xlabel=f"s[{day}]",
        ylabel=f"g[{minute}]",
        outpath=outdir / f"div_s_vs_g_{label}.png",
    )

    # u vs g
    plot_divergence_pairs(
        u_d, g_m, div,
        xlabel=f"u_day[{day}]",
        ylabel=f"g[{minute}]",
        outpath=outdir / f"div_u_vs_g_{label}.png",
    )

# ---------------------------------------------------------------------
# Unified diagnostics (ESS/Rhat + divergences)
# ---------------------------------------------------------------------

def plot_diagnostics(fit: CmdStanMCMC, outdir: Path) -> None:
    """
    Unified diagnostic suite:
    - ESS bulk / tail histograms
    - Rhat histogram
    - Worst-Rhat parameter histogram per chain
    - Divergence-based pairs plots for RW2 scales and latent fields
    - Scale parameter posteriors
    """
    outdir.mkdir(parents=True, exist_ok=True)
    summ = fit.summary()
    ess_bulk = summ["ESS_bulk"].to_numpy()
    ess_tail = summ["ESS_tail"].to_numpy()
    rhat = summ["R_hat"].to_numpy()
    param_names = summ.index.to_list()

    # -------------------------
    # ESS bulk histogram
    # -------------------------
    plt.figure(figsize=(6, 4))
    plt.hist(ess_bulk, bins=40)
    plt.title("ESS bulk across parameters")
    plt.tight_layout()
    plt.savefig(outdir / "ess_bulk_hist.png", dpi=200)
    plt.close()

    # -------------------------
    # ESS tail histogram
    # -------------------------
    plt.figure(figsize=(6, 4))
    plt.hist(ess_tail, bins=40)
    plt.title("ESS tail across parameters")
    plt.tight_layout()
    plt.savefig(outdir / "ess_tail_hist.png", dpi=200)
    plt.close()

    # -------------------------
    # Rhat histogram
    # -------------------------
    plt.figure(figsize=(6, 4))
    plt.hist(rhat, bins=40)
    plt.title("Rhat across parameters")
    plt.tight_layout()
    plt.savefig(outdir / "rhat_hist.png", dpi=200)
    plt.close()

    # -------------------------
    # Worst-Rhat parameter histograms by chain
    # -------------------------
    worst_name = param_names[np.argmax(rhat)]
    base = worst_name.split("[")[0]
    arr = fit.stan_variable(base)

    # If vector, pull the specific index
    if "[" in worst_name:
        idx = int(worst_name.split("[")[1].split("]")[0]) - 1
        arr = arr[:, idx]  # shape (draws_total,)

    # Reshape to (draws_per_chain, n_chains)
    if arr.ndim == 1:
        n_total = arr.shape[0]
        n_chains = fit.chains
        arr = arr.reshape(n_chains, n_total // n_chains).T

    n_chains = arr.shape[1]
    fig, axs = plt.subplots(n_chains, 1, figsize=(8, 2.5 * n_chains), sharex=True)
    if n_chains == 1:
        axs = [axs]

    for c in range(n_chains):
        axs[c].hist(arr[:, c], bins=40)
        axs[c].set_title(f"{worst_name} – chain {c+1}")

    plt.tight_layout()
    fig.savefig(outdir / f"worst_rhat_{base}.png", dpi=200)
    plt.close(fig)

    # -----------------------------------------------------------------
    # Divergence-based geometry diagnostics
    # -----------------------------------------------------------------
    div = get_divergent_mask(fit)
    if div.size == 0:
        print("No divergence information found in sampler diagnostics.")
        return

    # Core parameters for RW2 and process noise
    tau_diel   = extract_scalar(fit, "tau_diel")
    tau_season = extract_scalar(fit, "tau_season")
    mu_diel    = extract_scalar(fit, "mu_diel")
    mu_season  = extract_scalar(fit, "mu_season")
    sigma_day  = extract_scalar(fit, "sigma_day_proc")

    g_high = extract_element(fit, "g_centered", MINUTE_HIGH)
    g_low  = extract_element(fit, "g_centered", MINUTE_LOW)
    s_high = extract_element(fit, "s_centered", DAY_HIGH)
    s_low  = extract_element(fit, "s_centered", DAY_LOW)

    # ---- DIEL FIELD ----
    plot_divergence_pairs(
        tau_diel, g_high, div,
        "tau_diel", f"g_centered[{MINUTE_HIGH}] (high-data minute)",
        outdir / "div_tau_diel_vs_g_high.png",
    )
    plot_divergence_pairs(
        tau_diel, g_low, div,
        "tau_diel", f"g_centered[{MINUTE_LOW}] (no-data minute)",
        outdir / "div_tau_diel_vs_g_low.png",
    )
    plot_divergence_pairs(
        tau_diel, mu_diel, div,
        "tau_diel", "mu_diel",
        outdir / "div_tau_diel_vs_mu_diel.png",
    )

    # ---- SEASONAL FIELD ----
    plot_divergence_pairs(
        tau_season, s_high, div,
        "tau_season", f"s_centered[{DAY_HIGH}] (high-data day)",
        outdir / "div_tau_season_vs_s_high.png",
    )
    plot_divergence_pairs(
        tau_season, s_low, div,
        "tau_season", f"s_centered[{DAY_LOW}] (no-data day)",
        outdir / "div_tau_season_vs_s_low.png",
    )
    plot_divergence_pairs(
        tau_season, mu_season, div,
        "tau_season", "mu_season",
        outdir / "div_tau_season_vs_mu_season.png",
    )

    # ---- PROCESS NOISE ----
    # Representative day for u_day (use DAY_HIGH to anchor in data-rich region)
    u_day_high = extract_element(fit, "u_day", DAY_HIGH)
    plot_divergence_pairs(
        sigma_day, u_day_high, div,
        "sigma_day_proc", f"u_day[{DAY_HIGH}]",
        outdir / "div_sigma_day_vs_u_day_high.png",
    )
    plot_divergence_pairs(
        sigma_day, tau_season, div,
        "sigma_day_proc", "tau_season",
        outdir / "div_sigma_day_vs_tau_season.png",
    )

    # ---- MIXTURE LIKELIHOOD ----
    # These are per-bin parameters; we just look at their joint behavior.
    try:
        ell_bin = extract_scalar(fit, "ell_bin")   # if present
        q_bin   = extract_scalar(fit, "q_bin")
        plot_divergence_pairs(
            q_bin, ell_bin, div,
            "q_bin", "ell_bin",
            outdir / "div_q_bin_vs_ell_bin.png",
        )
    except ValueError:
        # ell_bin may not be stored as a parameter; skip if absent
        pass

    plot_nonident_triplet(fit, RESULTS_DIR, day=DAY_LOW, minute=MINUTE_LOW,  label="lowday_lowmin")
    plot_nonident_triplet(fit, RESULTS_DIR, day=DAY_LOW, minute=MINUTE_HIGH, label="lowday_highmin")
    plot_nonident_triplet(fit, RESULTS_DIR, day=DAY_HIGH, minute=MINUTE_LOW, label="highday_lowmin")
    plot_nonident_triplet(fit, RESULTS_DIR, day=DAY_HIGH, minute=MINUTE_HIGH,label="highday_highmin")

    # High‑data day × High‑data minute
    plot_sum_vs_components(
        fit,
        outdir=RESULTS_DIR,
        day=DAY_HIGH,
        minute=MINUTE_HIGH,
        label="highday_highmin"
    )

    # High‑data day × Low‑data minute
    plot_sum_vs_components(
        fit,
        outdir=RESULTS_DIR,
        day=DAY_HIGH,
        minute=MINUTE_LOW,
        label="highday_lowmin"
    )

    # Low‑data day × High‑data minute
    plot_sum_vs_components(
        fit,
        outdir=RESULTS_DIR,
        day=DAY_LOW,
        minute=MINUTE_HIGH,
        label="lowday_highmin"
    )

    # Low‑data day × Low‑data minute
    plot_sum_vs_components(
        fit,
        outdir=RESULTS_DIR,
        day=DAY_LOW,
        minute=MINUTE_LOW,
        label="lowday_lowmin"
    )

    # ---- Scale parameters histograms (posterior) ----
    for name in ["tau_season", "tau_diel", "sigma_day_proc"]:
        arr = extract_scalar(fit, name)
        plt.figure(figsize=(6, 4))
        plt.hist(arr, bins=50, density=True)
        plt.title(f"Posterior of {name}")
        plt.tight_layout()
        plt.savefig(outdir / f"{name}_hist.png", dpi=200)
        plt.close()


# ---------------------------------------------------------------------
# Seasonal reconstruction
# ---------------------------------------------------------------------

def reconstruct_seasonal(draws: dict) -> tuple[dict, dict, dict]:
    """
    Reconstruct seasonal probabilities from:
    - s           (smooth seasonal latent)
    - u_day       (day-level nugget)
    Uses the logit scale directly from the model:
    - p_season_base = inv_logit(s)
    - p_season      = inv_logit(s + u_day)
    """
    s = draws["s"]      # (n_draws, 365)
    u = draws["u_day"]  # (n_draws, 365)

    base_prob     = inv_logit(s)
    nugget_prob   = inv_logit(u)
    combined_prob = inv_logit(s + u)

    return (
        summarize_curve(base_prob),
        summarize_curve(nugget_prob),
        summarize_curve(combined_prob),
    )


# ---------------------------------------------------------------------
# Diel reconstruction
# ---------------------------------------------------------------------

def reconstruct_diel(draws: dict) -> dict:
    """
    Reconstruct diel probabilities from g (latent logit).
    """
    g = draws["g"]  # (n_draws, 1440)
    diel_probs = inv_logit(g)
    return summarize_curve(diel_probs)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    fit = load_fit(RESULTS_DIR, CSV_GLOB_PATTERN)
    draws = fit.stan_variables()

    # ---------------- Unified diagnostics ----------------
    plot_diagnostics(fit, RESULTS_DIR)

    # ---------------- Seasonal ----------------
    base_s, nugget_s, comb_s = reconstruct_seasonal(draws)
    D_period = draws["s"].shape[1]      # should be 365
    doy = np.arange(1, D_period + 1)

    plot_with_bands(
        doy, base_s,
        "Seasonal base probability",
        "Day of year", "P(call)",
        RESULTS_DIR / "seasonal_base.png",
    )
    plot_with_bands(
        doy, nugget_s,
        "Day-level nugget effect (on probability scale)",
        "Day of year", "P(call)",
        RESULTS_DIR / "seasonal_nugget.png",
    )
    plot_with_bands(
        doy, comb_s,
        "Seasonal probability (base + nugget)",
        "Day of year", "P(call)",
        RESULTS_DIR / "seasonal_combined.png",
    )

    # ---------------- Diel ----------------
    diel_s = reconstruct_diel(draws)
    N_diel = draws["g"].shape[1]        # should be 1440
    hours = np.arange(N_diel) / 60.0

    plot_with_bands(
        hours, diel_s,
        "Diel probability",
        "Hour of day", "P(call)",
        RESULTS_DIR / "diel.png",
    )
    
    print("Analysis complete. Outputs written to", RESULTS_DIR)

if __name__ == "__main__":
    main()