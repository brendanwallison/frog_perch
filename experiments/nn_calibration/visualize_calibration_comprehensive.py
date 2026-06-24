#!/usr/bin/env python3
"""
visualize_calibration.py

Comprehensive publication-style diagnostics for the frog sensor calibration.

Produces one PDF (or PNG series) containing the following figure groups:

  Figure 1  — Raw NN Output Behavior
  Figure 2  — Ground Truth Coverage & Distributions
  Figure 3  — Learned Mean Channel (noise floor + saturation)
  Figure 4  — Learned Precision / Dispersion Channel
  Figure 5  — Model Fit vs Empirical Saturation (mean domain)
  Figure 6  — Precision-Domain Fit Diagnostics
  Figure 7  — Posterior Likelihood Surfaces
  Figure 8  — Residual & Calibration Error Analysis
  Figure 9  — Interference Sensitivity
  Figure 10 — Parameter Summary

Each figure is saved individually as <base>_figN_<title>.png and the script
also saves a combined multi-page PDF.

Column expectations in calibration CSV
---------------------------------------
  nn_mu                   – unnormalised NN mean output  (0–1)
  nn_var                  – unnormalised NN variance output
  log_mean_rms_1000_1500  – acoustic interference proxy
  gt_mu                   – continuous ground-truth mean count (= true count)
  gt_var                  – continuous ground-truth variance   (optional but used)
  q_k                     – ground truth weights for k=0, k=1, ...  (list encoded)

Edit TRUE_COUNT_COLUMN / GT_VAR_COLUMN below if your CSV uses other names.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from scipy.special import expit as sigmoid
from scipy.stats import beta as beta_dist, pearsonr, spearmanr

import configs.nn_config as config
from frog_perch.nn_calibration.sensor_model import calculate_likelihood_vector

# ---------------------------------------------------------------------------
# User-editable column names
# ---------------------------------------------------------------------------
TRUE_COUNT_COLUMN = "gt_mu"   # continuous ground-truth mean calls
GT_VAR_COLUMN     = "gt_var"  # continuous ground-truth variance (set None to skip)

# ---------------------------------------------------------------------------
# Global aesthetics
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "figure.dpi":          150,
    "savefig.dpi":         250,
    "font.size":           11,
    "axes.titlesize":      12,
    "axes.labelsize":      11,
    "legend.fontsize":     9,
    "xtick.labelsize":     9,
    "ytick.labelsize":     9,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "figure.facecolor":    "white",
    "axes.facecolor":      "#f9f9f9",
    "grid.color":          "white",
    "grid.linewidth":      0.8,
    "axes.grid":           True,
})

CMAP_DIV   = "RdBu_r"
CMAP_SEQ   = "viridis"
CMAP_HEAT  = "magma"
SCATTER_S  = 6
SCATTER_A  = 0.30

# Interference percentiles used for conditional curves
INTERF_PERCENTILES = [5, 25, 50, 75, 95]
INTERF_LABELS      = ["p05 (quiet)", "p25", "p50 (median)", "p75", "p95 (noisy)"]
INTERF_COLORS      = ["#1a9641", "#a6d96a", "#fdae61", "#d7191c", "#7b2d8b"]


# ============================================================================
# Helper utilities
# ============================================================================

def running_stat(x: np.ndarray, y: np.ndarray, stat: str = "median"):
    """Return (unique_x, stat_y) using exact discrete x values."""
    xs = np.unique(x)
    if stat == "median":
        ys = np.array([np.median(y[x == k]) for k in xs])
    elif stat == "mean":
        ys = np.array([np.mean(y[x == k]) for k in xs])
    elif stat == "std":
        ys = np.array([np.std(y[x == k]) for k in xs])
    else:
        raise ValueError(stat)
    return xs, ys


def mu_var_to_precision(mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    """
    Synchronized with sensor_model.py:
    1. Clip mu and var to ensure valid domain.
    2. Derived phi must be >= 1e-3.
    """
    # Normalize inputs for the calculation (must be [0, 1])
    # Ensure clipping matches sensor_model.py
    mu_c = np.clip(mu, 1e-4, 1.0 - 1e-4)
    # Variance must be less than mu*(1-mu)
    max_var = mu_c * (1.0 - mu_c)
    var_c = np.clip(var, 1e-5, max_var - 1e-5)
    
    phi = (mu_c * (1.0 - mu_c) / var_c) - 1.0
    
    # Return 0 or negative for invalid values so the 'ok' mask can filter them
    return np.where(phi < 0, -1.0, phi)


def model_mean_curve(p: dict, x_interf: float, kmax: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (k_vec, predicted_mu_unnorm) for the mean channel."""
    k = np.arange(int(kmax) + 1, dtype=float)
    floor = sigmoid(p["b0"] + p["b_i"] * x_interf)
    gamma = np.exp(p["g0"] + p["g_i"] * x_interf)
    mu    = floor + (1.0 - floor) * np.tanh(gamma * (k / kmax))
    return k, mu * kmax          # un-normalise to raw NN mean scale


def model_precision_curve(p: dict, x_interf: float, kmax: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (k_vec, base_precision phi_y) for the dispersion channel."""
    k   = np.arange(int(kmax) + 1, dtype=float)
    phi = np.exp(p["a0"] + p["a_f"] * k + p["a_i"] * x_interf)
    return k, phi


def savefig(fig: plt.Figure, pdf: PdfPages, path: str):
    """Save figure to the PDF and as a standalone PNG."""
    pdf.savefig(fig, bbox_inches="tight")
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# CalibrationReport
# ============================================================================

class CalibrationReport:

    def __init__(self, params_path: str, csv_path: str):
        with open(params_path) as f:
            self.p = json.load(f)

        self.df   = pd.read_csv(csv_path)
        self.kmax = float(self.p["K_MAX"])

        # Resolve ground-truth columns
        self.truth = self._resolve_column(
            TRUE_COUNT_COLUMN,
            ["true_count", "k_true", "k", "gt_mu"],
            "ground-truth mean count",
        )
        self.truth_var = self._resolve_column(
            GT_VAR_COLUMN,
            ["gt_var"],
            "ground-truth variance",
            required=False,
        )

        # Centered interference covariate
        self.df["x"] = (
            self.df["log_mean_rms_1000_1500"]
            - self.p["x_interf_center_mean"]
        )

        # Convenience un-normalised columns
        self.df["nn_mu"] = self.df["nn_mu"] / self.kmax
        self.df["nn_var"] = self.df["nn_var"] / (self.kmax ** 2)

        self.df["nn_mu_raw"] = self.df["nn_mu"] * self.kmax
        self.df["nn_var_raw"] = self.df["nn_var"] * (self.kmax ** 2)

        # NN-reported precision (internal to the sensor model)
        self.df["nn_phi"] = mu_var_to_precision(
            self.df["nn_mu"].values,
            self.df["nn_var"].values,
        )

        # Percentile thresholds for interference conditioning
        self.x_pcts = np.percentile(self.df["x"].values, INTERF_PERCENTILES)

        # Parse q_k if present
        if "q_k" in self.df.columns:
            self.df["q_k_parsed"] = self.df["q_k"].apply(
                lambda v: np.array(eval(v) if isinstance(v, str) else v)
            )
            self.has_qk = True
        else:
            self.has_qk = False

        # Pre-compute model-predicted mean at zero interference for every sample
        _, m_curve_0 = model_mean_curve(self.p, 0.0, self.kmax)
        self.df["model_mu_at_gt"] = np.interp(
            self.df[self.truth].values, np.arange(int(self.kmax) + 1), m_curve_0
        )
        self.df["residual_mu"] = self.df["nn_mu_raw"] - self.df["model_mu_at_gt"]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_column(
        self,
        preferred: str | None,
        fallbacks: list[str],
        description: str,
        required: bool = True,
    ) -> str | None:
        if preferred is not None and preferred in self.df.columns:
            return preferred
        for col in fallbacks:
            if col in self.df.columns:
                return col
        if required:
            raise RuntimeError(
                f"Could not locate {description} column. "
                f"Tried: {[preferred] + fallbacks}"
            )
        return None

    # ------------------------------------------------------------------
    # Figure 1 — Raw NN Output Behavior
    # ------------------------------------------------------------------

    def fig_nn_outputs(self, base: str, pdf: PdfPages):
        fig = plt.figure(figsize=(20, 14))
        gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)
        fig.suptitle("Figure 1 — Raw Neural-Network Output Behavior", fontsize=14, fontweight="bold")

        gt  = self.df[self.truth].values
        mu  = self.df["nn_mu_raw"].values
        var = self.df["nn_var_raw"].values
        phi = self.df["nn_phi"].values
        x   = self.df["x"].values

        # 1a  NN mean vs true count (scatter + running median)
        ax = fig.add_subplot(gs[0, 0])
        ax.scatter(gt, mu, s=SCATTER_S, alpha=SCATTER_A, color="#4477AA")
        xs, ys = running_stat(np.round(gt).astype(int), mu, "median")
        ax.plot(xs, ys, lw=2.5, color="#EE6677", label="Running median")
        ax.plot([0, self.kmax], [0, self.kmax], "k--", lw=1, label="Identity")
        ax.set(title="NN mean vs true count", xlabel="True count (gt_mu)",
               ylabel="NN mean (raw)")
        ax.legend()

        # 1b  NN variance vs true count
        ax = fig.add_subplot(gs[0, 1])
        ax.scatter(gt, var, s=SCATTER_S, alpha=SCATTER_A, color="#4477AA")
        xs, ys = running_stat(np.round(gt).astype(int), var, "median")
        ax.plot(xs, ys, lw=2.5, color="#EE6677", label="Running median")
        ax.set(title="NN variance vs true count", xlabel="True count (gt_mu)",
               ylabel="NN variance (raw)")
        ax.legend()

        # 1c  NN precision vs true count
        ax = fig.add_subplot(gs[0, 2])
        valid = np.isfinite(phi)
        ax.scatter(gt[valid], phi[valid], s=SCATTER_S, alpha=SCATTER_A, color="#4477AA")
        xs, ys = running_stat(np.round(gt[valid]).astype(int), phi[valid], "median")
        ax.plot(xs, ys, lw=2.5, color="#EE6677", label="Running median")
        ax.set(title="NN precision (φ) vs true count",
               xlabel="True count (gt_mu)", ylabel="NN precision φ = µ(1-µ)/σ² − 1")
        ax.legend()

        # 1d  NN mean vs NN variance coloured by interference
        ax = fig.add_subplot(gs[1, 0])
        sc = ax.scatter(mu, var, c=x, s=SCATTER_S, alpha=SCATTER_A, cmap=CMAP_DIV)
        plt.colorbar(sc, ax=ax, label="Interference (centred)")
        ax.set(title="NN mean vs NN variance\n(coloured by interference)",
               xlabel="NN mean (raw)", ylabel="NN variance (raw)")

        # 1e  NN precision distribution
        ax = fig.add_subplot(gs[1, 1])
        phi_clipped = phi[np.isfinite(phi) & (phi > 0)]
        ax.hist(np.log10(phi_clipped + 1e-3), bins=60, color="#4477AA", edgecolor="white", lw=0.3)
        ax.set(title="Distribution of NN precision (log₁₀)",
               xlabel="log₁₀(φ_nn)", ylabel="Count")

        # 1f  NN mean error vs interference
        ax = fig.add_subplot(gs[1, 2])
        err = mu - gt
        ax.scatter(x, err, s=SCATTER_S, alpha=SCATTER_A, color="#4477AA")
        xs_s = np.sort(np.unique(np.round(x, 1)))
        ax.axhline(0, color="k", lw=1, ls="--")
        ax.set(title="NN mean error vs interference",
               xlabel="Interference (centred)", ylabel="NN mean − true count")

        savefig(fig, pdf, f"{base}_fig1_nn_outputs.png")

    # ------------------------------------------------------------------
    # Figure 2 — Ground Truth Coverage & Distributions
    # ------------------------------------------------------------------

    def fig_ground_truth(self, base: str, pdf: PdfPages):
        has_var = self.truth_var is not None
        ncols   = 3 if has_var else 2
        fig, axes = plt.subplots(2, ncols, figsize=(7 * ncols, 12))
        fig.suptitle("Figure 2 — Ground Truth Coverage & Distributions", fontsize=14, fontweight="bold")

        gt   = self.df[self.truth].values
        gtv  = self.df[self.truth_var].values if has_var else None
        x    = self.df["x"].values

        # 2a  gt_mu histogram
        ax = axes[0, 0]
        ax.hist(gt, bins=np.arange(0, self.kmax + 2) - 0.5, color="#4477AA",
                edgecolor="white", lw=0.4, rwidth=0.85)
        ax.set(title="Distribution of gt_mu", xlabel="True count (gt_mu)", ylabel="Count")

        # 2b  gt_mu vs interference
        ax = axes[0, 1]
        ax.scatter(x, gt, s=SCATTER_S, alpha=SCATTER_A, color="#4477AA")
        ax.set(title="gt_mu vs interference", xlabel="Interference (centred)",
               ylabel="True count (gt_mu)")

        if has_var:
            # 2c  gt_var histogram
            ax = axes[0, 2]
            ax.hist(gtv, bins=60, color="#4477AA", edgecolor="white", lw=0.3)
            ax.set(title="Distribution of gt_var", xlabel="True variance (gt_var)", ylabel="Count")

        # 2d  Joint density gt_mu vs gt_var (if available), else nn_mu vs gt_mu
        ax = axes[1, 0]
        if has_var:
            h, xedge, yedge = np.histogram2d(gt, gtv, bins=40)
            ax.imshow(h.T, origin="lower",
                      extent=[xedge[0], xedge[-1], yedge[0], yedge[-1]],
                      aspect="auto", cmap=CMAP_HEAT)
            ax.set(title="Joint density: gt_mu vs gt_var",
                   xlabel="gt_mu", ylabel="gt_var")
        else:
            ax.scatter(gt, self.df["nn_mu_raw"].values, s=SCATTER_S, alpha=SCATTER_A)
            ax.set(title="NN mean vs gt_mu", xlabel="gt_mu", ylabel="NN mean")

        # 2e  gt_mu precision (gt_mu(1-gt_mu/kmax)/(gt_var/kmax²) - 1)
        if has_var:
            ax = axes[1, 1]
            gt_mu_norm = np.clip(gt / self.kmax, 1e-4, 1 - 1e-4)
            gt_var_norm = np.clip(gtv / (self.kmax ** 2), 1e-6, 1)
            gt_phi = mu_var_to_precision(gt_mu_norm, gt_var_norm)
            valid  = np.isfinite(gt_phi) & (gt_phi > 0)
            ax.scatter(gt[valid], np.log10(gt_phi[valid]),
                       s=SCATTER_S, alpha=SCATTER_A, color="#4477AA")
            xs, ys = running_stat(np.round(gt[valid]).astype(int), np.log10(gt_phi[valid]), "median")
            ax.plot(xs, ys, lw=2, color="#EE6677", label="Running median")
            ax.set(title="GT precision (log₁₀) vs true count",
                   xlabel="gt_mu", ylabel="log₁₀(φ_gt)")
            ax.legend()

            # 2f  gt_phi vs nn_phi scatter
            ax = axes[1, 2]
            nn_phi = self.df["nn_phi"].values
            ok     = valid & np.isfinite(nn_phi) & (nn_phi > 0)
            ax.scatter(np.log10(gt_phi[ok]), np.log10(nn_phi[ok]),
                       s=SCATTER_S, alpha=SCATTER_A, color="#4477AA")
            lims = (min(np.log10(gt_phi[ok]).min(), np.log10(nn_phi[ok]).min()),
                    max(np.log10(gt_phi[ok]).max(), np.log10(nn_phi[ok]).max()))
            ax.plot(lims, lims, "k--", lw=1, label="Identity")
            r, _ = pearsonr(np.log10(gt_phi[ok]), np.log10(nn_phi[ok]))
            ax.set(title=f"GT precision vs NN precision (log₁₀)\nr = {r:.3f}",
                   xlabel="log₁₀(φ_gt)", ylabel="log₁₀(φ_nn)")
            ax.legend()
        else:
            for ax in axes[1, 1:]:
                ax.set_visible(False)

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        savefig(fig, pdf, f"{base}_fig2_ground_truth.png")

    # ------------------------------------------------------------------
    # Figure 3 — Learned Mean Channel
    # ------------------------------------------------------------------

    def fig_mean_channel(self, base: str, pdf: PdfPages):
        fig = plt.figure(figsize=(20, 12))
        gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.33)
        fig.suptitle("Figure 3 — Learned Mean Channel (Noise Floor & Saturation)", fontsize=14, fontweight="bold")

        x_grid = np.linspace(self.df["x"].min(), self.df["x"].max(), 300)
        k_grid  = np.arange(int(self.kmax) + 1, dtype=float)

        # 3a  Noise floor vs interference
        ax = fig.add_subplot(gs[0, 0])
        floor = sigmoid(self.p["b0"] + self.p["b_i"] * x_grid)
        ax.plot(x_grid, floor * self.kmax, lw=2.5, color="#EE6677")
        ax.set(title="Noise floor f(x) × K_MAX",
               xlabel="Interference (centred)", ylabel="Floor (in count units)")

        # 3b  Saturation scale gamma vs interference
        ax = fig.add_subplot(gs[0, 1])
        gamma = np.exp(self.p["g0"] + self.p["g_i"] * x_grid)
        ax.plot(x_grid, gamma, lw=2.5, color="#4477AA")
        ax.set(title="Saturation scale γ(x)",
               xlabel="Interference (centred)", ylabel="γ")

        # 3c  Model mean curves for several interference percentiles
        ax = fig.add_subplot(gs[0, 2])
        for xi, lbl, col in zip(self.x_pcts, INTERF_LABELS, INTERF_COLORS):
            k, m = model_mean_curve(self.p, xi, self.kmax)
            ax.plot(k, m, lw=2, color=col, label=lbl)
        ax.plot([0, self.kmax], [0, self.kmax], "k--", lw=1, label="Identity")
        ax.set(title="Predicted NN mean by interference percentile",
               xlabel="True count k", ylabel="Predicted NN mean")
        ax.legend(fontsize=8)

        # 3d  Saturation ratio (predicted_mu / k) as heatmap over k × interference
        ax = fig.add_subplot(gs[1, 0])
        X, K = np.meshgrid(x_grid, k_grid)
        floor_2d = sigmoid(self.p["b0"] + self.p["b_i"] * X)
        gamma_2d = np.exp(self.p["g0"] + self.p["g_i"] * X)
        mu_norm  = floor_2d + (1 - floor_2d) * np.tanh(gamma_2d * (K / self.kmax))
        im = ax.pcolormesh(x_grid, k_grid, mu_norm, cmap=CMAP_SEQ, shading="auto")
        plt.colorbar(im, ax=ax, label="µ̂ (normalised)")
        ax.set(title="Mean channel surface µ̂(k, x)",
               xlabel="Interference (centred)", ylabel="True count k")

        # 3e  Bias (predicted_mu - k/kmax) heatmap — where does the model over/underestimate?
        ax = fig.add_subplot(gs[1, 1])
        bias = mu_norm - (K / self.kmax)
        vmax = np.abs(bias).max()
        im = ax.pcolormesh(x_grid, k_grid, bias, cmap=CMAP_DIV,
                           norm=Normalize(-vmax, vmax), shading="auto")
        plt.colorbar(im, ax=ax, label="µ̂ − k/K_MAX")
        ax.set(title="Mean channel bias surface",
               xlabel="Interference (centred)", ylabel="True count k")

        # 3f  Sensitivity: d(floor)/dx and d(gamma)/dx
        ax = fig.add_subplot(gs[1, 2])
        dfloor_dx = self.p["b_i"] * floor * (1 - floor)        # sigmoid derivative
        dgamma_dx = self.p["g_i"] * gamma
        ax.plot(x_grid, dfloor_dx, lw=2, color="#EE6677", label="d(floor)/dx")
        ax.plot(x_grid, dgamma_dx, lw=2, color="#4477AA", label="d(γ)/dx")
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set(title="Interference sensitivities of floor & γ",
               xlabel="Interference (centred)", ylabel="Derivative")
        ax.legend()

        savefig(fig, pdf, f"{base}_fig3_mean_channel.png")

    # ------------------------------------------------------------------
    # Figure 4 — Learned Precision / Dispersion Channel
    # ------------------------------------------------------------------

    def fig_precision_channel(self, base: str, pdf: PdfPages):
        fig = plt.figure(figsize=(20, 12))
        gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.33)
        fig.suptitle("Figure 4 — Learned Precision / Dispersion Channel", fontsize=14, fontweight="bold")

        x_grid = np.linspace(self.df["x"].min(), self.df["x"].max(), 300)
        k_grid  = np.arange(int(self.kmax) + 1, dtype=float)

        # 4a  Base precision curve vs k at median interference
        ax = fig.add_subplot(gs[0, 0])
        for xi, lbl, col in zip(self.x_pcts, INTERF_LABELS, INTERF_COLORS):
            k, phi = model_precision_curve(self.p, xi, self.kmax)
            ax.plot(k, np.log10(phi), lw=2, color=col, label=lbl)
        ax.set(title="Model base precision log₁₀(φ_base) by interference",
               xlabel="True count k", ylabel="log₁₀(φ_base)")
        ax.legend(fontsize=8)

        # 4b  Precision surface φ_base(k, x) as heatmap
        ax = fig.add_subplot(gs[0, 1])
        X, K = np.meshgrid(x_grid, k_grid)
        phi_surf = np.exp(self.p["a0"] + self.p["a_f"] * K + self.p["a_i"] * X)
        im = ax.pcolormesh(x_grid, k_grid, np.log10(phi_surf), cmap=CMAP_HEAT, shading="auto")
        plt.colorbar(im, ax=ax, label="log₁₀(φ_base)")
        ax.set(title="Base precision surface log₁₀(φ_base)(k, x)",
               xlabel="Interference (centred)", ylabel="True count k")

        # 4c  Trust parameter alpha_v and its effect:  φ_eff = φ_base * phi_nn^alpha_v
        ax = fig.add_subplot(gs[0, 2])
        alpha_v = self.p.get("alpha_v", 1.0)
        phi_nn_grid = np.logspace(-1, 3, 300)
        for phi_b in [1, 10, 100, 1000]:
            phi_eff = phi_b * (phi_nn_grid ** alpha_v)
            ax.plot(np.log10(phi_nn_grid), np.log10(phi_eff), lw=2,
                    label=f"φ_base={phi_b}")
        ax.set(title=f"Effective precision φ_eff = φ_base × φ_nn^α_v\n(α_v = {alpha_v:.3f})",
               xlabel="log₁₀(φ_nn)", ylabel="log₁₀(φ_eff)")
        ax.legend(fontsize=8)

        # 4d  Scatter: empirical NN precision vs expected base precision at ground truth k
        ax = fig.add_subplot(gs[1, 0])
        nn_phi = self.df["nn_phi"].values
        gt     = self.df[self.truth].values
        x      = self.df["x"].values
        model_phi_base = np.exp(
            self.p["a0"]
            + self.p["a_f"] * np.clip(gt, 0, self.kmax)
            + self.p["a_i"] * x
        )
        ok = np.isfinite(nn_phi) & (nn_phi > 0)
        sc = ax.scatter(np.log10(model_phi_base[ok]), np.log10(nn_phi[ok]),
                        c=gt[ok], s=SCATTER_S, alpha=SCATTER_A, cmap=CMAP_SEQ)
        plt.colorbar(sc, ax=ax, label="gt_mu")
        mn = min(np.log10(model_phi_base[ok]).min(), np.log10(nn_phi[ok]).min())
        mx = max(np.log10(model_phi_base[ok]).max(), np.log10(nn_phi[ok]).max())
        ax.plot([mn, mx], [mn, mx], "k--", lw=1)
        ax.set(title="Model base precision vs NN precision (log₁₀)",
               xlabel="log₁₀(φ_base) @ gt_mu", ylabel="log₁₀(φ_nn)")

        # 4e  Effective precision predicted by model vs NN (full pipeline)
        ax = fig.add_subplot(gs[1, 1])
        phi_eff = np.clip(model_phi_base * (np.maximum(1e-3, nn_phi) ** alpha_v), 1, 1e4)
        ax.scatter(gt[ok], np.log10(phi_eff[ok]),
                   s=SCATTER_S, alpha=SCATTER_A, c=x[ok], cmap=CMAP_DIV)
        xs, ys = running_stat(np.round(gt[ok]).astype(int), np.log10(phi_eff[ok]), "median")
        ax.plot(xs, ys, lw=2, color="#EE6677", label="Running median")
        ax.set(title="Effective model precision log₁₀(φ_eff) vs gt_mu",
               xlabel="gt_mu", ylabel="log₁₀(φ_eff)")
        ax.legend()

        # 4f  Histogram of log10(phi_eff) stratified by count bucket
        ax = fig.add_subplot(gs[1, 2])
        buckets = [(0, 2), (2, 5), (5, self.kmax + 1)]
        colors  = ["#4477AA", "#EE6677", "#228833"]
        for (lo, hi), col in zip(buckets, colors):
            mask = (gt >= lo) & (gt < hi) & ok
            if mask.sum() > 5:
                ax.hist(np.log10(phi_eff[mask]), bins=40, alpha=0.5, color=col,
                        edgecolor="white", lw=0.3, label=f"k ∈ [{int(lo)},{int(hi)})")
        ax.set(title="Effective precision distribution by count bucket",
               xlabel="log₁₀(φ_eff)", ylabel="Count")
        ax.legend()

        savefig(fig, pdf, f"{base}_fig4_precision_channel.png")

    # ------------------------------------------------------------------
    # Figure 5 — Model Fit vs Empirical Saturation (mean domain)
    # ------------------------------------------------------------------

    def fig_mean_fit(self, base: str, pdf: PdfPages):
        fig = plt.figure(figsize=(20, 14))
        gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.33)
        fig.suptitle("Figure 5 — Model Fit: Mean Domain", fontsize=14, fontweight="bold")

        gt  = self.df[self.truth].values
        mu  = self.df["nn_mu_raw"].values
        x   = self.df["x"].values
        res = self.df["residual_mu"].values

        # 5a  Overlay: empirical running median vs model curves
        ax = fig.add_subplot(gs[0, 0])
        ax.scatter(gt, mu, s=SCATTER_S, alpha=SCATTER_A * 0.5, color="#4477AA")
        xs, ys = running_stat(np.round(gt).astype(int), mu, "median")
        ax.plot(xs, ys, lw=3, color="#000000", label="Empirical median", zorder=5)
        for xi, lbl, col in zip(self.x_pcts[[1, 2, 3]], INTERF_LABELS[1:4], INTERF_COLORS[1:4]):
            k, m = model_mean_curve(self.p, xi, self.kmax)
            ax.plot(k, m, "--", lw=2, color=col, label=f"Model {lbl}")
        ax.plot([0, self.kmax], [0, self.kmax], "k:", lw=1, label="Identity")
        ax.set(title="NN mean vs gt_mu: empirical vs model", xlabel="gt_mu", ylabel="NN mean (raw)")
        ax.legend(fontsize=8)

        # 5b  Residuals vs gt_mu
        ax = fig.add_subplot(gs[0, 1])
        sc = ax.scatter(gt, res, c=x, s=SCATTER_S, alpha=SCATTER_A, cmap=CMAP_DIV)
        plt.colorbar(sc, ax=ax, label="Interference")
        xs, ys = running_stat(np.round(gt).astype(int), res, "median")
        ax.plot(xs, ys, lw=2.5, color="#EE6677", label="Running median")
        ax.axhline(0, color="k", lw=1, ls="--")
        ax.set(title="Residuals (NN mean − model) vs gt_mu",
               xlabel="gt_mu", ylabel="Residual (count units)")
        ax.legend()

        # 5c  Residuals vs interference
        ax = fig.add_subplot(gs[0, 2])
        ax.scatter(x, res, c=gt, s=SCATTER_S, alpha=SCATTER_A, cmap=CMAP_SEQ)
        ax.axhline(0, color="k", lw=1, ls="--")
        ax.set(title="Residuals vs interference",
               xlabel="Interference (centred)", ylabel="Residual (count units)")

        # 5d  Residual histogram
        ax = fig.add_subplot(gs[1, 0])
        ax.hist(res, bins=60, color="#4477AA", edgecolor="white", lw=0.3)
        ax.axvline(np.nanmedian(res), color="#EE6677", lw=2, ls="--",
                   label=f"Median = {np.nanmedian(res):.3f}")
        ax.set(title="Residual distribution", xlabel="Residual (count units)", ylabel="Count")
        ax.legend()

        # 5e  Per-count-bin residual violin / box
        ax = fig.add_subplot(gs[1, 1])
        k_bins = np.arange(int(self.kmax) + 1)
        data   = [res[np.round(gt).astype(int) == k] for k in k_bins]
        data   = [d for d in data if len(d) > 3]
        positions = [k for k, d in zip(k_bins, data) if len(d) > 3]
        bp = ax.boxplot(data, positions=positions, widths=0.6,
                        patch_artist=True,
                        boxprops=dict(facecolor="#4477AA", alpha=0.5),
                        medianprops=dict(color="#EE6677", lw=2),
                        showfliers=False)
        ax.axhline(0, color="k", lw=1, ls="--")
        ax.set(title="Residual distribution by true count bin",
               xlabel="gt_mu (rounded)", ylabel="Residual (count units)")

        # 5f  QQ-plot of residuals vs normal
        ax = fig.add_subplot(gs[1, 2])
        r_sorted = np.sort(res[np.isfinite(res)])
        n = len(r_sorted)
        theoretical = np.array(
            [np.interp(i / (n + 1), np.linspace(0, 1, 1000),
                        np.sort(np.random.randn(1000))) for i in range(1, n + 1)]
        )
        # Use proper normal quantiles instead of simulation
        from scipy.stats import norm as norm_dist
        theoretical = norm_dist.ppf(np.linspace(0.5 / n, 1 - 0.5 / n, n))
        r_std = (r_sorted - r_sorted.mean()) / (r_sorted.std() + 1e-12)
        ax.scatter(theoretical, r_std, s=3, alpha=0.3, color="#4477AA")
        mn_v, mx_v = theoretical.min(), theoretical.max()
        ax.plot([mn_v, mx_v], [mn_v, mx_v], "k--", lw=1.5, label="Normal")
        ax.set(title="QQ-plot of residuals vs normal",
               xlabel="Theoretical quantiles", ylabel="Standardised residuals")
        ax.legend()

        savefig(fig, pdf, f"{base}_fig5_mean_fit.png")

    # ------------------------------------------------------------------
    # Figure 6 — Precision-Domain Fit Diagnostics
    # ------------------------------------------------------------------

    def fig_precision_fit(self, base: str, pdf: PdfPages):
        fig = plt.figure(figsize=(20, 12))
        gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.33)
        fig.suptitle("Figure 6 — Precision-Domain Fit Diagnostics", fontsize=14, fontweight="bold")

        gt      = self.df[self.truth].values
        x       = self.df["x"].values
        nn_phi  = self.df["nn_phi"].values
        alpha_v = self.p.get("alpha_v", 1.0)

        phi_base = np.exp(
            self.p["a0"]
            + self.p["a_f"] * np.clip(gt, 0, self.kmax)
            + self.p["a_i"] * x
        )
        phi_nn_safe = np.maximum(1e-3, nn_phi)
        phi_eff     = np.clip(phi_base * (phi_nn_safe ** alpha_v), 1, 1e4)
        ok          = np.isfinite(nn_phi) & (nn_phi > 0)

        # 6a  Predicted phi_eff vs NN phi (should track if trust is calibrated)
        ax = fig.add_subplot(gs[0, 0])
        ax.scatter(np.log10(phi_nn_safe[ok]), np.log10(phi_eff[ok]),
                   c=gt[ok], s=SCATTER_S, alpha=SCATTER_A, cmap=CMAP_SEQ)
        mn = np.log10(phi_nn_safe[ok]).min(); mx = np.log10(phi_nn_safe[ok]).max()
        ax.plot([mn, mx], [mn, mx], "k--", lw=1)
        ax.set(title="φ_eff vs φ_nn (log₁₀)", xlabel="log₁₀(φ_nn)", ylabel="log₁₀(φ_eff)")

        # 6b  If gt_var available: compare implied gt precision vs phi_eff
        if self.truth_var is not None:
            gtv     = self.df[self.truth_var].values
            gt_mu_n = np.clip(gt / self.kmax, 1e-4, 1 - 1e-4)
            gt_v_n  = np.clip(gtv / self.kmax ** 2, 1e-6, 1)
            gt_phi  = mu_var_to_precision(gt_mu_n, gt_v_n)
            ok2     = ok & np.isfinite(gt_phi) & (gt_phi > 0)

            ax = fig.add_subplot(gs[0, 1])
            ax.scatter(np.log10(gt_phi[ok2]), np.log10(phi_eff[ok2]),
                       c=gt[ok2], s=SCATTER_S, alpha=SCATTER_A, cmap=CMAP_SEQ)
            mn = min(np.log10(gt_phi[ok2]).min(), np.log10(phi_eff[ok2]).min())
            mx = max(np.log10(gt_phi[ok2]).max(), np.log10(phi_eff[ok2]).max())
            ax.plot([mn, mx], [mn, mx], "k--", lw=1)
            r, _ = pearsonr(np.log10(gt_phi[ok2]), np.log10(phi_eff[ok2]))
            ax.set(title=f"φ_eff vs φ_gt (log₁₀)  r = {r:.3f}",
                   xlabel="log₁₀(φ_gt)", ylabel="log₁₀(φ_eff)")
        else:
            ax = fig.add_subplot(gs[0, 1])
            ax.text(0.5, 0.5, "gt_var not available", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="grey")
            ax.set(title="φ_eff vs φ_gt (unavailable)")

        # 6c  Precision residual vs interference
        ax = fig.add_subplot(gs[0, 2])
        log_res = np.log10(phi_eff[ok]) - np.log10(phi_nn_safe[ok])
        sc = ax.scatter(x[ok], log_res, c=gt[ok], s=SCATTER_S, alpha=SCATTER_A, cmap=CMAP_SEQ)
        plt.colorbar(sc, ax=ax, label="gt_mu")
        ax.axhline(0, color="k", lw=1, ls="--")
        ax.set(title="Precision residual vs interference\nlog₁₀(φ_eff) − log₁₀(φ_nn)",
               xlabel="Interference (centred)", ylabel="log₁₀ precision residual")

        # 6d  Expected beta variance from model vs NN variance
        ax = fig.add_subplot(gs[1, 0])
        mu_mod = self.df["model_mu_at_gt"].values / self.kmax          # normalised predicted mu
        mu_mod_c = np.clip(mu_mod, 1e-4, 1 - 1e-4)
        var_from_phi = mu_mod_c * (1 - mu_mod_c) / (phi_eff + 1.0)    # Beta variance formula
        nn_var_n = self.df["nn_var"].values
        sc = ax.scatter(var_from_phi[ok], nn_var_n[ok],
                        c=gt[ok], s=SCATTER_S, alpha=SCATTER_A, cmap=CMAP_SEQ)
        plt.colorbar(sc, ax=ax, label="gt_mu")
        mx = max(var_from_phi[ok].max(), nn_var_n[ok].max())
        ax.plot([0, mx], [0, mx], "k--", lw=1)
        ax.set(title="Model-implied variance vs NN variance (normalised)",
               xlabel="Var from model (µ(1-µ)/(φ+1))", ylabel="NN variance (norm.)")

        # 6e  Distribution of phi_eff by interference quartile
        ax = fig.add_subplot(gs[1, 1])
        quartiles = np.percentile(x, [0, 25, 50, 75, 100])
        for i in range(4):
            mask = (x >= quartiles[i]) & (x < quartiles[i + 1]) & ok
            if mask.sum() > 5:
                ax.hist(np.log10(phi_eff[mask]), bins=40, alpha=0.4,
                        label=f"x Q{i+1}", edgecolor="white", lw=0.3)
        ax.set(title="Effective precision distribution by interference quartile",
               xlabel="log₁₀(φ_eff)", ylabel="Count")
        ax.legend(fontsize=8)

        # 6f  Scatter: predicted beta log-pdf vs interference (calibration quality)
        ax = fig.add_subplot(gs[1, 2])
        mu_n  = self.df["nn_mu"].values
        var_n = self.df["nn_var"].values
        mu_n_c  = np.clip(mu_n, 1e-4, 1 - 1e-4)
        var_n_c = np.clip(var_n, 1e-6, mu_n_c * (1 - mu_n_c) - 1e-5)
        phi_nn2 = (mu_n_c * (1 - mu_n_c) / var_n_c) - 1.0
        phi_nn2_safe = np.maximum(1e-3, phi_nn2)

        log_pdfs = beta_dist.logpdf(
            mu_n_c,
            mu_mod_c * phi_eff + 1e-6,
            (1 - mu_mod_c) * phi_eff + 1e-6,
        )
        log_pdfs = np.clip(log_pdfs, -50, None)
        sc = ax.scatter(x, log_pdfs, c=gt, s=SCATTER_S, alpha=SCATTER_A, cmap=CMAP_SEQ)
        plt.colorbar(sc, ax=ax, label="gt_mu")
        xs, ys = running_stat(np.round(x, 1), log_pdfs, "median")
        ax.plot(xs, ys, lw=2, color="#EE6677")
        ax.set(title="Log-likelihood (Beta) vs interference",
               xlabel="Interference (centred)", ylabel="log p(y | model, k=gt_mu)")

        savefig(fig, pdf, f"{base}_fig6_precision_fit.png")

    # ------------------------------------------------------------------
    # Figure 7 — Posterior Likelihood Surfaces
    # ------------------------------------------------------------------

    def fig_posterior_surfaces(self, base: str, pdf: PdfPages):
        fig = plt.figure(figsize=(20, 14))
        gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.33)
        fig.suptitle("Figure 7 — Posterior Likelihood Surfaces", fontsize=14, fontweight="bold")

        y_grid = np.linspace(0.02, 0.98, 100)
        k_max  = int(self.kmax)

        # Build likelihood surface at three variance levels
        for col_idx, (var_val, var_label) in enumerate(
            [(0.01, "Low var (0.01)"), (0.05, "Med var (0.05)"), (0.15, "High var (0.15)")]
        ):
            ax = fig.add_subplot(gs[0, col_idx])
            heat = np.zeros((len(y_grid), k_max + 1))
            for i, mu in enumerate(y_grid):
                var_clipped = min(var_val, mu * (1 - mu) - 1e-5)
                heat[i] = calculate_likelihood_vector(
                    mu, max(var_clipped, 1e-5), 0.0, self.p, self.kmax
                )
            im = ax.imshow(
                heat, origin="lower", aspect="auto",
                extent=[0, self.kmax, 0, self.kmax],
                cmap=CMAP_HEAT
            )
            plt.colorbar(im, ax=ax)
            ax.set(title=f"Posterior p(k|y, x=0) – {var_label}",
                   xlabel="Latent count k", ylabel="NN mean (raw)")

        # 7d  MAP estimate from posterior vs gt_mu
        ax = fig.add_subplot(gs[1, 0])
        gt  = self.df[self.truth].values
        mu  = self.df["nn_mu"].values
        var = self.df["nn_var"].values
        x_c = self.df["x"].values

        map_k = np.zeros(len(gt))
        for i in range(len(gt)):
            lik = calculate_likelihood_vector(
                np.clip(mu[i], 1e-4, 1 - 1e-4),
                np.clip(var[i], 1e-5, mu[i] * (1 - mu[i]) - 1e-5),
                x_c[i], self.p, self.kmax
            )
            map_k[i] = np.argmax(lik)

        ax.scatter(gt, map_k, s=SCATTER_S, alpha=SCATTER_A, color="#4477AA")
        xs, ys = running_stat(np.round(gt).astype(int), map_k, "median")
        ax.plot(xs, ys, lw=2.5, color="#EE6677", label="Running median")
        ax.plot([0, self.kmax], [0, self.kmax], "k--", lw=1, label="Identity")
        ax.set(title="MAP count estimate vs gt_mu",
               xlabel="gt_mu", ylabel="MAP k̂")
        ax.legend()

        # 7e  Posterior mean vs gt_mu
        ax = fig.add_subplot(gs[1, 1])
        post_mean = np.zeros(len(gt))
        k_vec = np.arange(k_max + 1, dtype=float)
        for i in range(min(len(gt), 5000)):     # cap for speed
            lik = calculate_likelihood_vector(
                np.clip(mu[i], 1e-4, 1 - 1e-4),
                np.clip(var[i], 1e-5, mu[i] * (1 - mu[i]) - 1e-5),
                x_c[i], self.p, self.kmax
            )
            post_mean[i] = np.dot(lik, k_vec)

        n_used = min(len(gt), 5000)
        ax.scatter(gt[:n_used], post_mean[:n_used], s=SCATTER_S, alpha=SCATTER_A, color="#4477AA")
        xs, ys = running_stat(np.round(gt[:n_used]).astype(int), post_mean[:n_used], "median")
        ax.plot(xs, ys, lw=2.5, color="#EE6677", label="Running median")
        ax.plot([0, self.kmax], [0, self.kmax], "k--", lw=1)
        ax.set(title=f"Posterior mean count vs gt_mu (n={n_used})",
               xlabel="gt_mu", ylabel="E[k | y, x]")
        ax.legend()

        # 7f  Posterior uncertainty (std) vs gt_mu
        ax = fig.add_subplot(gs[1, 2])
        post_std = np.zeros(n_used)
        for i in range(n_used):
            lik = calculate_likelihood_vector(
                np.clip(mu[i], 1e-4, 1 - 1e-4),
                np.clip(var[i], 1e-5, mu[i] * (1 - mu[i]) - 1e-5),
                x_c[i], self.p, self.kmax
            )
            pm = np.dot(lik, k_vec)
            post_std[i] = np.sqrt(np.dot(lik, (k_vec - pm) ** 2))

        sc = ax.scatter(gt[:n_used], post_std, c=x_c[:n_used],
                        s=SCATTER_S, alpha=SCATTER_A, cmap=CMAP_DIV)
        plt.colorbar(sc, ax=ax, label="Interference")
        xs, ys = running_stat(np.round(gt[:n_used]).astype(int), post_std, "median")
        ax.plot(xs, ys, lw=2.5, color="#EE6677", label="Running median")
        ax.set(title=f"Posterior std vs gt_mu (n={n_used})",
               xlabel="gt_mu", ylabel="Posterior std")
        ax.legend()

        savefig(fig, pdf, f"{base}_fig7_posterior_surfaces.png")

    # ------------------------------------------------------------------
    # Figure 8 — Residual & Calibration Error Analysis
    # ------------------------------------------------------------------

    def fig_residual_analysis(self, base: str, pdf: PdfPages):
        fig = plt.figure(figsize=(20, 14))
        gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.33)
        fig.suptitle("Figure 8 — Residual & Calibration Error Analysis", fontsize=14, fontweight="bold")

        gt    = self.df[self.truth].values
        res   = self.df["residual_mu"].values
        x     = self.df["x"].values
        nn_mu = self.df["nn_mu_raw"].values

        # 8a  Mean absolute error by count bin
        ax = fig.add_subplot(gs[0, 0])
        k_bins = np.arange(int(self.kmax) + 1)
        maes   = [np.mean(np.abs(res[np.round(gt).astype(int) == k]))
                  for k in k_bins if (np.round(gt).astype(int) == k).sum() > 0]
        ks_valid = [k for k in k_bins if (np.round(gt).astype(int) == k).sum() > 0]
        ax.bar(ks_valid, maes, color="#4477AA", edgecolor="white", lw=0.4)
        ax.set(title="Mean absolute residual by true count bin",
               xlabel="gt_mu (rounded)", ylabel="|Residual| mean")

        # 8b  Bias vs interference (running mean)
        ax = fig.add_subplot(gs[0, 1])
        x_sort = np.sort(np.unique(np.round(x, 1)))
        ax.scatter(x, res, s=SCATTER_S, alpha=SCATTER_A * 0.5, color="#4477AA")
        xs_s = np.linspace(x.min(), x.max(), 50)
        from scipy.ndimage import uniform_filter1d
        # bin-wise mean
        bins    = np.linspace(x.min(), x.max(), 30)
        bin_idx = np.digitize(x, bins)
        bx = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
        by = [np.mean(res[bin_idx == i + 1]) if (bin_idx == i + 1).sum() > 0 else np.nan
              for i in range(len(bins) - 1)]
        ax.plot(bx, by, lw=2.5, color="#EE6677", label="Binned mean")
        ax.axhline(0, color="k", lw=1, ls="--")
        ax.set(title="Model bias vs interference",
               xlabel="Interference (centred)", ylabel="Residual mean")
        ax.legend()

        # 8c  Calibration curve: empirical quantiles vs model (RMSE decomposition)
        ax = fig.add_subplot(gs[0, 2])
        pct_range = np.arange(5, 100, 5)
        emp_pcts  = np.percentile(np.abs(res), pct_range)
        ax.plot(pct_range, emp_pcts, "o-", lw=2, color="#4477AA", label="|Residual| percentiles")
        ax.set(title="Cumulative error profile",
               xlabel="Percentile", ylabel="|Residual| (count units)")
        ax.axhline(0.5, color="#EE6677", ls="--", lw=1.5, label="0.5 count threshold")
        ax.axhline(1.0, color="#228833", ls="--", lw=1.5, label="1.0 count threshold")
        ax.legend()

        # 8d  Log-likelihood per sample as a proxy calibration score
        ax = fig.add_subplot(gs[1, 0])
        mu_n    = self.df["nn_mu"].values
        var_n   = self.df["nn_var"].values
        mu_mod  = np.clip(self.df["model_mu_at_gt"].values / self.kmax, 1e-4, 1 - 1e-4)
        nn_phi  = self.df["nn_phi"].values
        alpha_v = self.p.get("alpha_v", 1.0)
        phi_base = np.exp(
            self.p["a0"]
            + self.p["a_f"] * np.clip(gt, 0, self.kmax)
            + self.p["a_i"] * x
        )
        phi_nn_safe = np.maximum(1e-3, nn_phi)
        phi_eff = np.clip(phi_base * (phi_nn_safe ** alpha_v), 1, 1e4)
        a_beta  = np.clip(mu_mod * phi_eff, 1e-4, None)
        b_beta  = np.clip((1 - mu_mod) * phi_eff, 1e-4, None)
        mu_n_c  = np.clip(mu_n, 1e-4, 1 - 1e-4)
        ll      = beta_dist.logpdf(mu_n_c, a_beta, b_beta)
        ll      = np.clip(ll, -50, None)

        sc = ax.scatter(gt, ll, c=x, s=SCATTER_S, alpha=SCATTER_A, cmap=CMAP_DIV)
        plt.colorbar(sc, ax=ax, label="Interference")
        xs, ys = running_stat(np.round(gt).astype(int), ll, "median")
        ax.plot(xs, ys, lw=2.5, color="#EE6677")
        ax.set(title="Per-sample log-likelihood vs gt_mu",
               xlabel="gt_mu", ylabel="log p(y | model, k)")

        # 8e  RMSE by interference bin
        ax = fig.add_subplot(gs[1, 1])
        n_bins_x = 10
        xbins  = np.percentile(x, np.linspace(0, 100, n_bins_x + 1))
        rmses  = []
        x_mids = []
        for i in range(n_bins_x):
            mask = (x >= xbins[i]) & (x < xbins[i + 1])
            if mask.sum() > 2:
                rmses.append(np.sqrt(np.mean(res[mask] ** 2)))
                x_mids.append((xbins[i] + xbins[i + 1]) / 2)
        ax.bar(x_mids, rmses, width=np.diff(xbins)[:len(rmses)],
               color="#4477AA", edgecolor="white", lw=0.4)
        ax.set(title="RMSE of NN mean vs model by interference bin",
               xlabel="Interference (centred)", ylabel="RMSE (count units)")

        # 8f  NLL vs training NLL baseline (histogram of per-sample NLL)
        ax = fig.add_subplot(gs[1, 2])
        ax.hist(-ll, bins=60, color="#4477AA", edgecolor="white", lw=0.3)
        ax.axvline(np.nanmedian(-ll), color="#EE6677", lw=2, ls="--",
                   label=f"Median NLL = {np.nanmedian(-ll):.2f}")
        ax.set(title="Per-sample NLL distribution", xlabel="NLL", ylabel="Count")
        ax.legend()

        savefig(fig, pdf, f"{base}_fig8_residuals.png")

    # ------------------------------------------------------------------
    # Figure 9 — Interference Sensitivity
    # ------------------------------------------------------------------

    def fig_interference(self, base: str, pdf: PdfPages):
        fig = plt.figure(figsize=(20, 12))
        gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.33)
        fig.suptitle("Figure 9 — Interference Sensitivity", fontsize=14, fontweight="bold")

        x   = self.df["x"].values
        gt  = self.df[self.truth].values
        mu  = self.df["nn_mu_raw"].values
        res = self.df["residual_mu"].values

        # 9a  Raw interference distribution
        ax = fig.add_subplot(gs[0, 0])
        ax.hist(x, bins=60, color="#4477AA", edgecolor="white", lw=0.3)
        ax.set(title="Distribution of interference covariate",
               xlabel="Interference (centred)", ylabel="Count")

        # 9b  NN mean error as a function of interference, stratified by count
        ax = fig.add_subplot(gs[0, 1])
        buckets = [(0, 2, "#4477AA"), (2, 5, "#EE6677"), (5, self.kmax + 1, "#228833")]
        for lo, hi, col in buckets:
            mask = (gt >= lo) & (gt < hi)
            if mask.sum() > 10:
                bins = np.linspace(x.min(), x.max(), 20)
                idx  = np.digitize(x[mask], bins)
                bx   = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
                by   = [np.mean(res[mask][idx == i + 1])
                        if (idx == i + 1).sum() > 0 else np.nan
                        for i in range(len(bins) - 1)]
                ax.plot(bx, by, lw=2, color=col, label=f"k ∈ [{int(lo)},{int(hi)})")
        ax.axhline(0, color="k", lw=1, ls="--")
        ax.set(title="Residual vs interference by count bucket",
               xlabel="Interference (centred)", ylabel="Mean residual")
        ax.legend()

        # 9c  NN mean saturation effect conditioned on interference quartile
        ax = fig.add_subplot(gs[0, 2])
        qtiles = np.percentile(x, [0, 25, 50, 75, 100])
        cols   = ["#1a9641", "#a6d96a", "#fdae61", "#d7191c"]
        for i in range(4):
            mask = (x >= qtiles[i]) & (x < qtiles[i + 1])
            if mask.sum() > 10:
                xs, ys = running_stat(np.round(gt[mask]).astype(int), mu[mask], "median")
                ax.plot(xs, ys, lw=2, color=cols[i], label=f"x Q{i+1}")
        ax.plot([0, self.kmax], [0, self.kmax], "k--", lw=1)
        ax.set(title="NN mean saturation by interference quartile",
               xlabel="gt_mu", ylabel="NN mean (raw) – running median")
        ax.legend(fontsize=8)

        # 9d  Interference vs NN precision
        ax = fig.add_subplot(gs[1, 0])
        nn_phi = self.df["nn_phi"].values
        ok = np.isfinite(nn_phi) & (nn_phi > 0)
        sc = ax.scatter(x[ok], np.log10(nn_phi[ok]),
                        c=gt[ok], s=SCATTER_S, alpha=SCATTER_A, cmap=CMAP_SEQ)
        plt.colorbar(sc, ax=ax, label="gt_mu")
        ax.set(title="NN precision vs interference",
               xlabel="Interference (centred)", ylabel="log₁₀(φ_nn)")

        # 9e  Model floor and gamma as fn of raw interference value
        ax = fig.add_subplot(gs[1, 1])
        x_raw = np.linspace(self.df["log_mean_rms_1000_1500"].min(),
                             self.df["log_mean_rms_1000_1500"].max(), 300)
        x_c   = x_raw - self.p["x_interf_center_mean"]
        floor  = sigmoid(self.p["b0"] + self.p["b_i"] * x_c)
        gamma  = np.exp(self.p["g0"] + self.p["g_i"] * x_c)
        ax2 = ax.twinx()
        ax.plot(x_raw, floor, lw=2.5, color="#EE6677", label="Noise floor")
        ax2.plot(x_raw, gamma, lw=2.5, color="#4477AA", ls="--", label="γ")
        ax.set(title="Model parameters vs raw interference",
               xlabel="log mean RMS (1000–1500 Hz)", ylabel="Noise floor")
        ax2.set_ylabel("γ", color="#4477AA")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

        # 9f  Joint histogram: interference × gt_mu (data coverage)
        ax = fig.add_subplot(gs[1, 2])
        h, xedge, yedge = np.histogram2d(x, gt, bins=30)
        ax.imshow(h.T, origin="lower",
                  extent=[xedge[0], xedge[-1], yedge[0], yedge[-1]],
                  aspect="auto", cmap=CMAP_HEAT)
        ax.set(title="Joint coverage: interference × gt_mu",
               xlabel="Interference (centred)", ylabel="gt_mu")

        savefig(fig, pdf, f"{base}_fig9_interference.png")

    # ------------------------------------------------------------------
    # Figure 10 — Parameter Summary
    # ------------------------------------------------------------------

    def fig_parameters(self, base: str, pdf: PdfPages):
        fig = plt.figure(figsize=(18, 10))
        gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.35)
        fig.suptitle("Figure 10 — Calibrated Parameter Summary", fontsize=14, fontweight="bold")

        p = self.p

        # 10a  Parameter bar chart
        ax = fig.add_subplot(gs[0, 0])
        names  = ["a0", "a_f", "a_i", "alpha_v", "b0", "b_i", "g0", "g_i"]
        values = [p.get(n, np.nan) for n in names]
        colors = ["#4477AA" if v >= 0 else "#EE6677" for v in values]
        ax.barh(names, values, color=colors, edgecolor="white", lw=0.4)
        ax.axvline(0, color="k", lw=0.8)
        ax.set(title="Fitted parameter values", xlabel="Value")

        # 10b  Effective noise floor range annotation
        ax = fig.add_subplot(gs[0, 1])
        x_grid   = np.linspace(self.df["x"].min(), self.df["x"].max(), 300)
        floor    = sigmoid(p["b0"] + p["b_i"] * x_grid)
        gamma    = np.exp(p["g0"] + p["g_i"] * x_grid)
        ax.fill_between(x_grid, floor * self.kmax,
                         self.kmax * np.ones_like(x_grid), alpha=0.12, color="#EE6677",
                         label="Saturated region")
        ax.fill_between(x_grid, np.zeros_like(x_grid), floor * self.kmax, alpha=0.12,
                        color="#4477AA", label="Floor region")
        ax.plot(x_grid, floor * self.kmax, lw=2.5, color="#EE6677", label="Floor (count)")
        ax.set(title="Noise floor in count units vs interference",
               xlabel="Interference (centred)", ylabel="Floor (count units)")
        ax.legend(fontsize=8)

        # 10c  Saturation midpoint (where tanh=0.5 ↔ k* = arctanh(0.5)/gamma * kmax)
        ax = fig.add_subplot(gs[0, 2])
        with np.errstate(divide="ignore"):
            k_half = np.arctanh(0.5) / (gamma + 1e-12) * self.kmax
        ax.plot(x_grid, np.clip(k_half, 0, self.kmax), lw=2.5, color="#228833")
        ax.set(title="Half-saturation count k* vs interference\n(tanh reaches 0.5 at k*)",
               xlabel="Interference (centred)", ylabel="k* (count units)")

        # 10d  Precision exponent a_f: how precision scales with k
        ax = fig.add_subplot(gs[1, 0])
        k_grid = np.arange(int(self.kmax) + 1, dtype=float)
        xi_mid = self.x_pcts[2]
        phi_base = np.exp(p["a0"] + p["a_f"] * k_grid + p["a_i"] * xi_mid)
        ax.semilogy(k_grid, phi_base, lw=2.5, color="#4477AA",
                    label=f"Median interference (x={xi_mid:.2f})")
        ax.set(title=f"Base precision vs k (a_f={p['a_f']:.3f})",
               xlabel="k", ylabel="φ_base (log scale)")
        ax.legend(fontsize=8)

        # 10e  Implied beta distribution shapes at selected (k, x)
        ax = fig.add_subplot(gs[1, 1])
        y_plot = np.linspace(0.01, 0.99, 300)
        for k_sel, xi_sel, lbl, col in [
            (2, xi_mid,  "k=2, med x", "#4477AA"),
            (5, xi_mid,  "k=5, med x", "#EE6677"),
            (2, self.x_pcts[4], "k=2, noisy", "#228833"),
        ]:
            _, m = model_mean_curve(p, xi_sel, self.kmax)
            mu_n = np.clip(m[k_sel] / self.kmax, 1e-4, 1 - 1e-4)
            _, phi_b = model_precision_curve(p, xi_sel, self.kmax)
            phi_b_clip = np.clip(phi_b[k_sel], 1, 1e4)
            nn_phi_ex  = 10.0   # example NN precision
            alpha_v    = p.get("alpha_v", 1.0)
            phi_eff    = np.clip(phi_b_clip * (nn_phi_ex ** alpha_v), 1, 1e4)
            a_b = mu_n * phi_eff; b_b = (1 - mu_n) * phi_eff
            ax.plot(y_plot * self.kmax, beta_dist.pdf(y_plot, a_b, b_b), lw=2,
                    color=col, label=lbl)
        ax.set(title=f"Beta likelihood shapes (φ_nn={nn_phi_ex:.0f})",
               xlabel="NN mean (count units)", ylabel="Density")
        ax.legend(fontsize=8)

        # 10f  Textual parameter table
        ax = fig.add_subplot(gs[1, 2])
        ax.axis("off")
        param_info = {
            "a0":     "Log-precision intercept",
            "a_f":    "Precision–count slope",
            "a_i":    "Precision–interference slope",
            "alpha_v":"NN precision trust weight",
            "b0":     "Noise-floor logit intercept",
            "b_i":    "Noise-floor interference slope",
            "g0":     "Log-gamma intercept",
            "g_i":    "Log-gamma interference slope",
            "K_MAX":  "Maximum count support",
            "x_interf_center_mean": "Interference centering offset",
        }
        rows = [[k, f"{p.get(k, 'N/A'):.4f}" if isinstance(p.get(k), float) else str(p.get(k, "N/A")), v]
                for k, v in param_info.items()]
        tbl = ax.table(
            cellText=rows,
            colLabels=["Param", "Value", "Description"],
            cellLoc="left", loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.0, 1.4)
        ax.set_title("Parameter reference", fontsize=11)

        savefig(fig, pdf, f"{base}_fig10_parameters.png")

    # ------------------------------------------------------------------
    # Master entry point
    # ------------------------------------------------------------------

    def build(self, outfile: str):
        """
        Generate all ten figures.  Saves:
          - A multi-page PDF at `outfile`
          - Individual PNGs next to the PDF with suffixes _fig1_…_fig10_….
        """
        base = outfile.replace(".pdf", "")
        print(f"[INFO] Writing calibration report to {outfile} …")

        with PdfPages(outfile) as pdf:
            self.fig_nn_outputs(base, pdf)
            self.fig_ground_truth(base, pdf)
            self.fig_mean_channel(base, pdf)
            self.fig_precision_channel(base, pdf)
            self.fig_mean_fit(base, pdf)
            self.fig_precision_fit(base, pdf)
            self.fig_posterior_surfaces(base, pdf)
            self.fig_residual_analysis(base, pdf)
            self.fig_interference(base, pdf)
            self.fig_parameters(base, pdf)

        print(f"[INFO] Done. {outfile}")


# ============================================================================
# CLI entry point
# ============================================================================

def main():
    params_path = os.path.join(
        config.CHECKPOINT_DIR,
        "best.keras_multiband_calibration_calibrated_v2.json"
    )
    csv_path = os.path.join(
        config.CHECKPOINT_DIR,
        "best.keras_multiband_calibration.csv"
    )
    out_pdf = params_path.replace(".json", "_calibration_report.pdf")

    CalibrationReport(params_path, csv_path).build(out_pdf)


if __name__ == "__main__":
    main()