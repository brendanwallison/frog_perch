#!/usr/bin/env python3
"""
visualize_calibration.py

Comprehensive publication-style diagnostics for the frog sensor calibration.

Produces one PDF (or PNG series) containing the following figure groups:
  Figure 5  — Model Fit vs Empirical Saturation (mean domain)
  Figure 6  — Precision-Domain Fit Diagnostics
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
    
    # Return -1.0 for invalid values so the 'ok' mask can filter them
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
        if ok.sum() > 0:
            ax.scatter(np.log10(phi_nn_safe[ok]), np.log10(phi_eff[ok]),
                       c=gt[ok], s=SCATTER_S, alpha=SCATTER_A, cmap=CMAP_SEQ)
            mn = np.log10(phi_nn_safe[ok]).min()
            mx = np.log10(phi_nn_safe[ok]).max()
            ax.plot([mn, mx], [mn, mx], "k--", lw=1)
        else:
            ax.text(0.5, 0.5, "No valid precision", ha="center", va="center", transform=ax.transAxes, color="red")
        ax.set(title="φ_eff vs φ_nn (log₁₀)", xlabel="log₁₀(φ_nn)", ylabel="log₁₀(φ_eff)")

        # 6b  If gt_var available: compare implied gt precision vs phi_eff
        if self.truth_var is not None:
            gtv     = self.df[self.truth_var].values
            gt_mu_n = np.clip(gt / self.kmax, 1e-4, 1 - 1e-4)
            gt_v_n  = np.clip(gtv / self.kmax ** 2, 1e-6, 1)
            gt_phi  = mu_var_to_precision(gt_mu_n, gt_v_n)
            ok2     = ok & np.isfinite(gt_phi) & (gt_phi > 0)

            ax = fig.add_subplot(gs[0, 1])
            if ok2.sum() > 0:
                ax.scatter(np.log10(gt_phi[ok2]), np.log10(phi_eff[ok2]),
                           c=gt[ok2], s=SCATTER_S, alpha=SCATTER_A, cmap=CMAP_SEQ)
                mn = min(np.log10(gt_phi[ok2]).min(), np.log10(phi_eff[ok2]).min())
                mx = max(np.log10(gt_phi[ok2]).max(), np.log10(phi_eff[ok2]).max())
                ax.plot([mn, mx], [mn, mx], "k--", lw=1)
                r, _ = pearsonr(np.log10(gt_phi[ok2]), np.log10(phi_eff[ok2]))
                ax.set(title=f"φ_eff vs φ_gt (log₁₀)  r = {r:.3f}",
                       xlabel="log₁₀(φ_gt)", ylabel="log₁₀(φ_eff)")
            else:
                ax.text(0.5, 0.5, "No valid GT precision", ha="center", va="center", transform=ax.transAxes, color="red")
                ax.set(title="φ_eff vs φ_gt (log₁₀)", xlabel="log₁₀(φ_gt)", ylabel="log₁₀(φ_eff)")
        else:
            ax = fig.add_subplot(gs[0, 1])
            ax.text(0.5, 0.5, "gt_var not available", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="grey")
            ax.set(title="φ_eff vs φ_gt (unavailable)")

        # 6c  Precision residual vs interference
        ax = fig.add_subplot(gs[0, 2])
        if ok.sum() > 0:
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
        if ok.sum() > 0:
            sc = ax.scatter(var_from_phi[ok], nn_var_n[ok],
                            c=gt[ok], s=SCATTER_S, alpha=SCATTER_A, cmap=CMAP_SEQ)
            plt.colorbar(sc, ax=ax, label="gt_mu")
            mx = max(var_from_phi[ok].max(), nn_var_n[ok].max())
            ax.plot([0, mx], [0, mx], "k--", lw=1)
        else:
            ax.text(0.5, 0.5, "No valid variance", ha="center", va="center", transform=ax.transAxes, color="red")
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
        Generate figures 5, 6, and 10.  Saves:
          - A multi-page PDF at `outfile`
          - Individual PNGs next to the PDF with suffixes _fig5_… etc.
        """
        base = outfile.replace(".pdf", "")
        print(f"[INFO] Writing calibration report to {outfile} …")

        with PdfPages(outfile) as pdf:
            self.fig_mean_fit(base, pdf)
            self.fig_precision_fit(base, pdf)
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