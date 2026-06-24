# #!/usr/bin/env python3
# """
# visualize_calibration.py

# Publication-style diagnostics for the frog sensor calibration.
# Focuses on resolving power: Mean saturation curves, Beta density ridgeplots,
# and an exhaustive parameter-by-parameter visual summary.
# """

# from __future__ import annotations

# import json
# import os
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# import numpy as np
# import pandas as pd
# from matplotlib.backends.backend_pdf import PdfPages
# from scipy.special import expit as sigmoid
# from scipy.stats import beta as beta_dist

# import configs.nn_config as config

# # ---------------------------------------------------------------------------
# # Global Config & Aesthetics
# # ---------------------------------------------------------------------------
# TRUE_COUNT_COLUMN = "gt_mu"

# mpl.rcParams.update({
#     "figure.dpi": 150, "savefig.dpi": 250, "axes.spines.top": False,
#     "axes.spines.right": False, "axes.grid": True, "grid.linewidth": 0.5,
#     "font.size": 10
# })

# # ---------------------------------------------------------------------------
# # Helper utilities
# # ---------------------------------------------------------------------------

# def mu_var_to_precision(mu: np.ndarray, var: np.ndarray) -> np.ndarray:
#     """Derive Beta precision phi = mu(1-mu)/var - 1 in normalized domain."""
#     mu_c = np.clip(mu, 1e-4, 1.0 - 1e-4)
#     var_c = np.clip(var, 1e-6, (mu_c * (1.0 - mu_c)) - 1e-7)
#     phi = (mu_c * (1.0 - mu_c) / var_c) - 1.0
#     return np.where(phi < 0, -1.0, phi)

# def model_mean_curve(p: dict, x_interf: float, k_vec: np.ndarray, kmax: float) -> np.ndarray:
#     """Returns un-normalized mu given an array of k values."""
#     floor = sigmoid(p["b0"] + p["b_i"] * x_interf)
#     gamma = np.exp(p["g0"] + p["g_i"] * x_interf)
#     mu_n = floor + (1.0 - floor) * np.tanh(gamma * (k_vec / kmax))
#     return np.clip(mu_n, 1e-4, 1 - 1e-4) * kmax

# def savefig(fig: plt.Figure, pdf: PdfPages, path: str):
#     pdf.savefig(fig, bbox_inches="tight")
#     fig.savefig(path, dpi=250, bbox_inches="tight")
#     plt.close(fig)

# # ============================================================================
# # CalibrationReport
# # ============================================================================

# class CalibrationReport:
#     def __init__(self, params_path: str, csv_path: str):
#         with open(params_path) as f: self.p = json.load(f)
#         self.df = pd.read_csv(csv_path)
#         self.kmax = float(self.p["K_MAX"])
#         self.truth = TRUE_COUNT_COLUMN if TRUE_COUNT_COLUMN in self.df.columns else "gt_mu"

#         # Map to [0,1] space immediately for all Beta math
#         self.df["mu_n"] = self.df["nn_mu"] / self.kmax
#         self.df["var_n"] = self.df["nn_var"] / (self.kmax ** 2)
#         self.df["nn_phi"] = mu_var_to_precision(self.df["mu_n"].values, self.df["var_n"].values)
        
#         # Center interference
#         self.df["x"] = self.df["log_mean_rms_1000_1500"] - self.p["x_interf_center_mean"]

#     def fig_resolving_power(self, base: str, pdf: PdfPages):
#         """
#         A 3-panel figure replacing the standard outputs to focus purely on 
#         the continuous saturation physics and the density distributions.
#         """
#         fig = plt.figure(figsize=(18, 14))
#         # Top row: Saturation Curves & Variance. Bottom row: Density Ridgeplot
#         gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1.5], wspace=0.2, hspace=0.3)
#         fig.suptitle("Resolving Power: Saturation Physics and Beta Density Fits", fontsize=16, fontweight="bold")

#         gt = self.df[self.truth].values
#         mu_raw = self.df["nn_mu"].values
#         mu_n = self.df["mu_n"].values
#         x = self.df["x"].values

#         # Pre-calculate representative precisions per k-bin for Median interference
#         k_ints = np.round(gt).astype(int)
#         unique_k = sorted(np.unique(k_ints))
#         unique_k = [k for k in unique_k if k <= self.kmax and (k_ints == k).sum() > 5]
        
#         median_phi_eff = {}
#         for k_val in unique_k:
#             mask = (k_ints == k_val)
#             phi_base = np.exp(self.p["a0"] + self.p["a_f"] * k_val + self.p["a_i"] * x[mask])
#             phi_nn_safe = np.maximum(1e-3, self.df["nn_phi"].values[mask])
#             phi_eff = np.clip(phi_base * (phi_nn_safe ** self.p.get("alpha_v", 1.0)), 1, 1e4)
#             median_phi_eff[k_val] = np.median(phi_eff)

#         # --- PANEL A: Continuous Learned Curves ---
#         ax_curves = fig.add_subplot(gs[0, 0])
#         ax_curves.scatter(gt, mu_raw, s=5, alpha=0.1, color="gray", label="Empirical Observations")
        
#         # Generate a high-resolution, continuous k-vector to reveal the tanh curve
#         k_smooth = np.linspace(0, self.kmax, 200)
        
#         # Plot Quiet, Median, and Noisy curves
#         pcts = np.percentile(x, [10, 50, 90])
#         colors = ["#1a9641", "#fdae61", "#d7191c"]
#         labels = ["Quiet (10th pctl)", "Average (50th pctl)", "Noisy (90th pctl)"]
        
#         for pctl_val, col, lbl in zip(pcts, colors, labels):
#             mu_smooth = model_mean_curve(self.p, pctl_val, k_smooth, self.kmax)
#             ax_curves.plot(k_smooth, mu_smooth, lw=2.5, color=col, label=lbl)

#         ax_curves.plot([0, self.kmax], [0, self.kmax], "k--", lw=1, alpha=0.5, label="Identity")
#         ax_curves.set(title="A. Learned Continuous Saturation Curves", xlabel="True Count (k)", ylabel="NN Mean Output")
#         ax_curves.legend()

#         # --- PANEL B: Variance Parity ---
#         ax_var = fig.add_subplot(gs[0, 1])
#         emp_vars, mod_vars = [], []
#         x_med = np.median(x)
        
#         for k_val in unique_k:
#             mask = (k_ints == k_val)
#             emp_vars.append(np.var(mu_n[mask]))
#             mu_val_n = model_mean_curve(self.p, x_med, np.array([k_val]), self.kmax)[0] / self.kmax
#             phi_val = median_phi_eff[k_val]
#             mod_vars.append((mu_val_n * (1 - mu_val_n)) / (phi_val + 1))

#         emp_vars = np.array(emp_vars) * (self.kmax**2)
#         mod_vars = np.array(mod_vars) * (self.kmax**2)

#         ax_var.scatter(emp_vars, mod_vars, color="#4477AA", s=30, zorder=5)
#         lim_max = max(max(emp_vars, default=1), max(mod_vars, default=1)) * 1.1
#         ax_var.plot([0, lim_max], [0, lim_max], "k--", lw=1)
#         ax_var.set(title="B. Variance Calibration Parity", xlabel="Empirical Variance", ylabel="Expected Model Variance")

#         # --- PANEL C: The Density Ridgeplot (Joyplot) ---
#         ax_ridge = fig.add_subplot(gs[1, :])
        
#         y_grid_norm = np.linspace(0.001, 0.999, 300)
#         y_grid_raw = y_grid_norm * self.kmax
#         overlap_factor = 1.2 
        
#         for i, k_val in enumerate(unique_k):
#             offset = i * overlap_factor
#             mask = (k_ints == k_val)
#             empirical_mu_raw = mu_raw[mask]

#             counts, bins = np.histogram(empirical_mu_raw, bins=30, density=True)
#             scale_factor = overlap_factor * 0.85 / (np.max(counts) + 1e-9)
#             counts_scaled = counts * scale_factor
            
#             ax_ridge.bar(bins[:-1], counts_scaled, width=np.diff(bins), 
#                          bottom=offset, align="edge", color="#4477AA", alpha=0.4, edgecolor="white", lw=0.5)

#             mu_val_n = model_mean_curve(self.p, x_med, np.array([k_val]), self.kmax)[0] / self.kmax
#             phi_val = median_phi_eff[k_val]
#             pdf_vals = beta_dist.pdf(y_grid_norm, mu_val_n * phi_val, (1 - mu_val_n) * phi_val)
            
#             pdf_scaled = (pdf_vals / self.kmax) * scale_factor
#             ax_ridge.plot(y_grid_raw, pdf_scaled + offset, color="#EE6677", lw=2)
#             ax_ridge.text(-0.5, offset + 0.2, f"True Count = {k_val}", va="center", ha="right", fontweight="bold")

#         ax_ridge.axvline(self.kmax, color="red", linestyle="--", lw=1.5, alpha=0.6)
#         ax_ridge.text(self.kmax - 0.1, ax_ridge.get_ylim()[1] * 0.9, "Theoretical Saturation Wall (K_MAX)", 
#                       rotation=90, color="red", va="top", ha="right", alpha=0.6)

#         ax_ridge.set(title="C. Density Ridgeplot: Beta Precision vs Empirical Dispersion", 
#                      xlabel="NN Predicted Mean", ylabel="Count Regimes (Stacked vertically)")
#         ax_ridge.set_yticks([]) 
#         ax_ridge.set_xlim(-1, self.kmax + 1)
#         ax_ridge.grid(axis='y')

#         savefig(fig, pdf, f"{base}_fig1_resolving_power.png")

#     def fig_parameters(self, base: str, pdf: PdfPages):
#         """
#         Exhaustive 9-panel breakdown of every learned parameter.
#         """
#         fig = plt.figure(figsize=(20, 16))
#         gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.25)
#         fig.suptitle("Calibrated Parameter Summary: Deep Dive", fontsize=16, fontweight="bold")
#         p = self.p

#         x_grid = np.linspace(self.df["x"].min(), self.df["x"].max(), 300)
#         k_grid = np.linspace(0, self.kmax, 300)
#         x_med = np.median(self.df["x"])
#         k_med = self.kmax / 2.0

#         # --- 1. Parameter Bar Chart ---
#         ax = fig.add_subplot(gs[0, 0])
#         names  = ["a0", "a_f", "a_i", "alpha_v", "b0", "b_i", "g0", "g_i"]
#         values = [p.get(n, np.nan) for n in names]
#         colors = ["#4477AA" if v >= 0 else "#EE6677" for v in values]
#         ax.barh(names, values, color=colors, edgecolor="white", lw=0.4)
#         ax.axvline(0, color="k", lw=0.8)
#         ax.set(title="1. Raw Fitted Values", xlabel="Value")

#         # --- 2. Noise Floor (b0, b_i) ---
#         ax = fig.add_subplot(gs[0, 1])
#         floor = sigmoid(p["b0"] + p["b_i"] * x_grid) * self.kmax
#         ax.plot(x_grid, floor, lw=2.5, color="#EE6677")
#         ax.set(title="2. Noise Floor vs Interference (b0, b_i)", 
#                xlabel="Interference (x)", ylabel="Floor (count units)")

#         # --- 3. Saturation Scale (g0, g_i) ---
#         ax = fig.add_subplot(gs[0, 2])
#         gamma = np.exp(p["g0"] + p["g_i"] * x_grid)
#         ax.plot(x_grid, gamma, lw=2.5, color="#228833")
#         ax.set(title="3. Saturation Scale γ vs Interference (g0, g_i)", 
#                xlabel="Interference (x)", ylabel="Gamma multiplier (γ)")

#         # --- 4. Half-Saturation Point ---
#         ax = fig.add_subplot(gs[1, 0])
#         with np.errstate(divide="ignore"):
#             k_half = np.arctanh(0.5) / (gamma + 1e-12) * self.kmax
#         ax.plot(x_grid, np.clip(k_half, 0, self.kmax), lw=2.5, color="#228833")
#         ax.axhline(self.kmax, color="red", linestyle="--", lw=1, alpha=0.5, label="K_MAX")
#         ax.set(title="4. Half-Saturation Count (K*) vs Interference\n(Where curve hits 50% max)", 
#                xlabel="Interference (x)", ylabel="True Count (k*)")
#         ax.legend()

#         # --- 5. Base Precision vs Count (a0, a_f) ---
#         ax = fig.add_subplot(gs[1, 1])
#         phi_base_k = np.exp(p["a0"] + p["a_f"] * k_grid + p["a_i"] * x_med)
#         ax.semilogy(k_grid, phi_base_k, lw=2.5, color="#4477AA")
#         ax.set(title=f"5. Base Precision vs True Count (a0, a_f)\n[At median x={x_med:.2f}]", 
#                xlabel="True Count (k)", ylabel="φ_base (log scale)")

#         # --- 6. Base Precision vs Interference (a0, a_i) ---
#         ax = fig.add_subplot(gs[1, 2])
#         phi_base_x = np.exp(p["a0"] + p["a_f"] * k_med + p["a_i"] * x_grid)
#         ax.semilogy(x_grid, phi_base_x, lw=2.5, color="#4477AA")
#         ax.set(title=f"6. Base Precision vs Interference (a0, a_i)\n[At med count k={k_med:.1f}]", 
#                xlabel="Interference (x)", ylabel="φ_base (log scale)")

#         # --- 7. Trust Parameter (alpha_v) ---
#         ax = fig.add_subplot(gs[2, 0])
#         nn_phi_grid = np.logspace(-1, 3, 100)
#         alpha_v = p.get("alpha_v", 1.0)
#         for base_phi in [1, 10, 100]:
#             phi_eff = base_phi * (nn_phi_grid ** alpha_v)
#             ax.plot(np.log10(nn_phi_grid), np.log10(phi_eff), lw=2, label=f"φ_base = {base_phi}")
#         ax.plot([-1, 3], [-1, 3], "k--", lw=1, alpha=0.5, label="1:1 Trust")
#         ax.set(title=f"7. Effective vs NN Precision (alpha_v = {alpha_v:.2f})", 
#                xlabel="log10(φ_nn)", ylabel="log10(φ_eff)")
#         ax.legend(fontsize=8)

#         # --- 8. Beta Shapes (The Result) ---
#         ax = fig.add_subplot(gs[2, 1])
#         y_plot = np.linspace(0.01, 0.99, 300)
#         x_pcts = np.percentile(self.df["x"], [10, 90])
#         for k_sel, xi_sel, lbl, col in [
#             (2, x_med,  "k=2, med x", "#4477AA"),
#             (5, x_med,  "k=5, med x", "#EE6677"),
#             (2, x_pcts[1], "k=2, noisy x", "#228833"),
#         ]:
#             mu_n = model_mean_curve(p, xi_sel, np.array([k_sel]), self.kmax)[0] / self.kmax
#             phi_b = np.exp(p["a0"] + p["a_f"] * k_sel + p["a_i"] * xi_sel)
#             phi_eff = np.clip(phi_b * (10.0 ** alpha_v), 1, 1e4) # Assume phi_nn = 10
            
#             a_b, b_b = mu_n * phi_eff, (1 - mu_n) * phi_eff
#             ax.plot(y_plot * self.kmax, beta_dist.pdf(y_plot, a_b, b_b), lw=2, color=col, label=lbl)
            
#         ax.set(title="8. Resulting Beta Shapes (Assuming φ_nn=10)", 
#                xlabel="NN Mean Output", ylabel="Density")
#         ax.legend(fontsize=8)

#         # --- 9. Parameter Table ---
#         ax = fig.add_subplot(gs[2, 2])
#         ax.axis("off")
#         param_info = {
#             "a0":     "Log-precision intercept",
#             "a_f":    "Precision–count slope",
#             "a_i":    "Precision–interference slope",
#             "alpha_v":"NN precision trust exponent",
#             "b0":     "Noise-floor logit intercept",
#             "b_i":    "Noise-floor interference slope",
#             "g0":     "Log-gamma intercept",
#             "g_i":    "Log-gamma interference slope",
#             "K_MAX":  "Maximum count boundary",
#         }
#         rows = [[k, f"{p.get(k, 'N/A'):.4f}" if isinstance(p.get(k), float) else str(p.get(k, "N/A")), v]
#                 for k, v in param_info.items()]
#         tbl = ax.table(cellText=rows, colLabels=["Param", "Value", "Description"], cellLoc="left", loc="center")
#         tbl.auto_set_font_size(False)
#         tbl.set_fontsize(9)
#         tbl.scale(1.0, 1.4)
#         ax.set_title("9. Parameter Dictionary", fontsize=12)

#         savefig(fig, pdf, f"{base}_fig2_parameters.png")

#     def build(self, outfile: str):
#         base = outfile.replace(".pdf", "")
#         print(f"[INFO] Writing calibration report to {outfile} …")
#         with PdfPages(outfile) as pdf:
#             self.fig_resolving_power(base, pdf)
#             self.fig_parameters(base, pdf)
#         print(f"[INFO] Done. {outfile}")

# # ============================================================================
# # CLI entry point
# # ============================================================================

# def main():
#     params_path = os.path.join(config.CHECKPOINT_DIR, "best.keras_multiband_calibration_calibrated_v2.json")
#     csv_path = os.path.join(config.CHECKPOINT_DIR, "best.keras_multiband_calibration.csv")
#     out_pdf = params_path.replace(".json", "_resolving_power_report.pdf")
#     CalibrationReport(params_path, csv_path).build(out_pdf)

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
visualize_calibration.py

Publication-style diagnostics for the frog sensor calibration.
Focuses on resolving power: Mean saturation curves, Beta density ridgeplots,
and an exhaustive parameter-by-parameter visual summary.

UPDATED:
- Replaces tanh saturation with Weibull CDF saturation
- Adds h0 (Weibull shape parameter)
"""

from __future__ import annotations

import json
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy.special import expit as sigmoid
from scipy.stats import beta as beta_dist

import configs.nn_config as config

# ---------------------------------------------------------------------------
# Global Config & Aesthetics
# ---------------------------------------------------------------------------
TRUE_COUNT_COLUMN = "gt_mu"

mpl.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 250,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linewidth": 0.5,
    "font.size": 10
})

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def mu_var_to_precision(mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    """Derive Beta precision phi = mu(1-mu)/var - 1 in normalized domain."""
    mu_c = np.clip(mu, 1e-4, 1.0 - 1e-4)
    var_c = np.clip(var, 1e-6, (mu_c * (1.0 - mu_c)) - 1e-7)
    phi = (mu_c * (1.0 - mu_c) / var_c) - 1.0
    return np.where(phi < 0, -1.0, phi)


def model_mean_curve(p: dict, x_interf: float, k_vec: np.ndarray, kmax: float) -> np.ndarray:
    """
    Weibull-CDF saturation mean curve (normalized -> raw units).
    """
    floor = sigmoid(p["b0"] + p["b_i"] * x_interf)

    scale = np.exp(p["g0"] + p["g_i"] * x_interf)
    shape = np.exp(p.get("h0", 0.0))

    z = k_vec / kmax
    exponent = np.clip((scale * z) ** shape, 0.0, 60.0)

    sat = 1.0 - np.exp(-exponent)

    mu_n = floor + (1.0 - floor) * sat
    return np.clip(mu_n, 1e-4, 1 - 1e-4) * kmax


def savefig(fig: plt.Figure, pdf: PdfPages, path: str):
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

        self.df = pd.read_csv(csv_path)
        self.kmax = float(self.p["K_MAX"])
        self.truth = TRUE_COUNT_COLUMN if TRUE_COUNT_COLUMN in self.df.columns else "gt_mu"

        self.df["mu_n"] = self.df["nn_mu"] / self.kmax
        self.df["var_n"] = self.df["nn_var"] / (self.kmax ** 2)
        self.df["nn_phi"] = mu_var_to_precision(self.df["mu_n"].values, self.df["var_n"].values)

        self.df["x"] = self.df["log_mean_rms_1000_1500"] - self.p["x_interf_center_mean"]

    # -----------------------------------------------------------------------
    # FIG 1: resolving power
    # -----------------------------------------------------------------------

    def fig_resolving_power(self, base: str, pdf: PdfPages):

        fig = plt.figure(figsize=(18, 14))
        gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1.5], wspace=0.2, hspace=0.3)
        fig.suptitle("Resolving Power: Weibull Saturation Physics and Beta Fits", fontsize=16, fontweight="bold")

        gt = self.df[self.truth].values
        mu_raw = self.df["nn_mu"].values
        mu_n = self.df["mu_n"].values
        x = self.df["x"].values

        k_ints = np.round(gt).astype(int)
        unique_k = sorted(np.unique(k_ints))
        unique_k = [k for k in unique_k if k <= self.kmax and (k_ints == k).sum() > 5]

        # effective precision
        median_phi_eff = {}
        for k_val in unique_k:
            mask = (k_ints == k_val)
            phi_base = np.exp(self.p["a0"] + self.p["a_f"] * k_val + self.p["a_i"] * x[mask])
            phi_nn_safe = np.maximum(1e-3, self.df["nn_phi"].values[mask])
            phi_eff = np.clip(phi_base * (phi_nn_safe ** self.p.get("alpha_v", 1.0)), 1, 1e4)
            median_phi_eff[k_val] = np.median(phi_eff)

        # --- A: saturation curves ---
        ax_curves = fig.add_subplot(gs[0, 0])
        ax_curves.scatter(gt, mu_raw, s=5, alpha=0.1, color="gray")

        k_smooth = np.linspace(0, self.kmax, 200)
        pcts = np.percentile(x, [10, 50, 90])

        colors = ["#1a9641", "#fdae61", "#d7191c"]
        labels = ["Quiet", "Median", "Noisy"]

        for xi, col, lbl in zip(pcts, colors, labels):
            mu_smooth = model_mean_curve(self.p, xi, k_smooth, self.kmax)
            ax_curves.plot(k_smooth, mu_smooth, lw=2.5, color=col, label=lbl)

        ax_curves.plot([0, self.kmax], [0, self.kmax], "k--", lw=1, alpha=0.5)
        ax_curves.set(title="A. Weibull Saturation Curves", xlabel="k", ylabel="NN mean")
        ax_curves.legend()

        # --- B: variance parity ---
        ax_var = fig.add_subplot(gs[0, 1])
        emp_vars, mod_vars = [], []
        x_med = np.median(x)

        for k_val in unique_k:
            mask = (k_ints == k_val)
            emp_vars.append(np.var(mu_n[mask]))

            mu_val_n = model_mean_curve(self.p, x_med, np.array([k_val]), self.kmax)[0] / self.kmax
            phi_val = median_phi_eff[k_val]
            mod_vars.append((mu_val_n * (1 - mu_val_n)) / (phi_val + 1))

        emp_vars = np.array(emp_vars) * (self.kmax ** 2)
        mod_vars = np.array(mod_vars) * (self.kmax ** 2)

        ax_var.scatter(emp_vars, mod_vars, color="#4477AA", s=30)
        lim = max(emp_vars.max(initial=1), mod_vars.max(initial=1)) * 1.1
        ax_var.plot([0, lim], [0, lim], "k--")
        ax_var.set(title="B. Variance Parity", xlabel="Empirical", ylabel="Model")

        # --- C: ridgeplot ---
        ax_ridge = fig.add_subplot(gs[1, :])

        overlap = 1.2
        for i, k_val in enumerate(unique_k):
            offset = i * overlap
            mask = (k_ints == k_val)

            counts, bins = np.histogram(mu_raw[mask], bins=30, density=True)
            scale_factor = overlap * 0.85 / (np.max(counts) + 1e-9)

            ax_ridge.bar(
                bins[:-1],
                counts * scale_factor,
                width=np.diff(bins),
                bottom=offset,
                alpha=0.4
            )

            mu_val_n = model_mean_curve(self.p, x_med, np.array([k_val]), self.kmax)[0] / self.kmax
            phi_val = median_phi_eff[k_val]

            y = np.linspace(0.001, 0.999, 200)
            pdf_vals = beta_dist.pdf(y, mu_val_n * phi_val, (1 - mu_val_n) * phi_val)

            ax_ridge.plot(y * self.kmax, (pdf_vals / self.kmax) * scale_factor + offset)

        ax_ridge.axvline(self.kmax, color="red", linestyle="--")
        ax_ridge.set(title="C. Beta Density Ridgeplot", xlabel="NN output", ylabel="k bins")

        savefig(fig, pdf, f"{base}_fig1_resolving_power.png")

    # -----------------------------------------------------------------------
    # FIG 2: parameters
    # -----------------------------------------------------------------------

    def fig_parameters(self, base: str, pdf: PdfPages):

        fig = plt.figure(figsize=(20, 16))
        gs = gridspec.GridSpec(3, 3, figure=fig)

        p = self.p

        x_grid = np.linspace(self.df["x"].min(), self.df["x"].max(), 300)
        k_grid = np.linspace(0, self.kmax, 300)

        x_med = np.median(self.df["x"])
        k_med = self.kmax / 2

        # 1 parameters
        ax = fig.add_subplot(gs[0, 0])
        names = ["a0", "a_f", "a_i", "alpha_v", "b0", "b_i", "g0", "g_i", "h0"]
        vals = [p.get(n, np.nan) for n in names]
        ax.barh(names, vals)
        ax.set(title="Parameters")

        # 2 floor
        ax = fig.add_subplot(gs[0, 1])
        floor = sigmoid(p["b0"] + p["b_i"] * x_grid)
        ax.plot(x_grid, floor)

        # 3 scale
        ax = fig.add_subplot(gs[0, 2])
        scale = np.exp(p["g0"] + p["g_i"] * x_grid)
        ax.plot(x_grid, scale)

        # 4 half-saturation (WEIBULL INVERTED)
        ax = fig.add_subplot(gs[1, 0])

        scale_med = np.exp(p["g0"] + p["g_i"] * x_grid)
        shape = np.exp(p.get("h0", 0.0))

        z_half = (np.log(2.0) ** (1.0 / shape)) / (scale_med + 1e-12)
        k_half = np.clip(z_half * self.kmax, 0, self.kmax)

        ax.plot(x_grid, k_half)
        ax.set(title="Half-Saturation (Weibull)")

        # 5 precision vs k
        ax = fig.add_subplot(gs[1, 1])
        phi = np.exp(p["a0"] + p["a_f"] * k_grid + p["a_i"] * x_med)
        ax.semilogy(k_grid, phi)

        # 6 precision vs x
        ax = fig.add_subplot(gs[1, 2])
        phi = np.exp(p["a0"] + p["a_f"] * k_med + p["a_i"] * x_grid)
        ax.semilogy(x_grid, phi)

        # 7 trust
        ax = fig.add_subplot(gs[2, 0])
        ax.set_title("Trust (unchanged)")

        # 8 beta shapes
        ax = fig.add_subplot(gs[2, 1])
        y = np.linspace(0.01, 0.99, 300)

        for k_sel, xi in [(2, x_med), (5, x_med), (2, np.percentile(x_grid, 90))]:
            mu = model_mean_curve(p, xi, np.array([k_sel]), self.kmax)[0] / self.kmax
            phi = np.exp(p["a0"] + p["a_f"] * k_sel + p["a_i"] * xi)

            ax.plot(y * self.kmax, beta_dist.pdf(y, mu * phi, (1 - mu) * phi))

        # 9 table
        ax = fig.add_subplot(gs[2, 2])
        ax.axis("off")

        savefig(fig, pdf, f"{base}_fig2_parameters.png")

    # -----------------------------------------------------------------------

    def build(self, outfile: str):
        base = outfile.replace(".pdf", "")
        with PdfPages(outfile) as pdf:
            self.fig_resolving_power(base, pdf)
            self.fig_parameters(base, pdf)


def main():
    params_path = os.path.join(config.CHECKPOINT_DIR, "best.keras_multiband_calibration_calibrated_v2.json")
    csv_path = os.path.join(config.CHECKPOINT_DIR, "best.keras_multiband_calibration.csv")

    out_pdf = params_path.replace(".json", "_weibull_report.pdf")

    CalibrationReport(params_path, csv_path).build(out_pdf)


if __name__ == "__main__":
    main()