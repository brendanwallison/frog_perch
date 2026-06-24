"""
evaluate_nn_outputs.py

Drop-in evaluation script (replacement) focused on the user's requested metrics
and correct handling of probabilistic ground truth.

Key behaviors:
- Auto-discovers the *_multiband_calibration_full.csv in configs.nn_config.CHECKPOINT_DIR
- Treats q_k as probabilistic ground-truth PMF (compare to nn_count_probs)
- Derives ground-truth presence probability as p_presence_gt = 1 - q_k[0]
  and converts to binary for AUC using a configurable threshold (default 0.5)
- Compares gt_slice (probabilistic per-slice values in [0,1]) against nn_slice_logits:
  converts logits -> probs via sigmoid for continuous metrics; converts gt_slice to
  binary (thresholded) when computing slice-level AUC
- Computes prioritized metrics:
    * Slice-level: per-sample slice Brier, slice logloss, slice Pearson/Spearman,
      and slice AUC (binary GT via threshold) when applicable
    * Count head: predicted mean/var vs true mean/var (MAE, RMSE, bias)
    * Distribution: NLL (expected), CRPS (discrete), EMD (Wasserstein-1), KLs, entropy
    * Presence: predicted presence (nn_binary or 1 - p_pred(k=0)) vs GT presence (p_presence_gt)
      - Brier, logloss (expected), and binary AUC (after converting GT presence to binary)
- Aggregates overall and optionally per-gt-bin (rounded gt_mu) summaries
- Computes bootstrap CIs for a small set of key metrics
- Writes metrics_per_sample.csv and metrics_summary.csv into the CSV folder

Drop-in: place this file in your repo and run it with the same environment used by other scripts.
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

# Try to import project config (same pattern used in other scripts)
try:
    import configs.nn_config as config
except Exception:
    try:
        import config as config  # fallback
    except Exception:
        config = None

EPS = 1e-12


# -------------------------
# Utilities
# -------------------------
def parse_array(x: Any) -> Optional[np.ndarray]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, (list, tuple)):
        return np.asarray(x)
    s = str(x).strip()
    if s == "":
        return None
    try:
        parsed = json.loads(s)
        return np.asarray(parsed)
    except Exception:
        pass
    try:
        parsed = eval(s, {"__builtins__": None}, {})
        return np.asarray(parsed)
    except Exception:
        pass
    try:
        return np.fromstring(s.replace("[", "").replace("]", ""), sep=" ")
    except Exception:
        return None


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    pos = x >= 0
    out = np.empty_like(x, dtype=float)
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    out[~pos] = np.exp(x[~pos]) / (1.0 + np.exp(x[~pos]))
    return out


def safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Return ROC AUC or nan when undefined (single-class or empty)."""
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    valid = ~np.isnan(y_true)
    y_true = y_true[valid]
    y_score = y_score[valid]
    if y_true.size == 0:
        return float("nan")
    unique = np.unique(y_true)
    if unique.size < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")


def safe_kl(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = np.clip(p, EPS, 1.0)
    q = np.clip(q, EPS, 1.0)
    return float(np.sum(p * np.log(p / q)))


def discrete_crps(pred_pmf: np.ndarray, true_k_or_pmf: Any) -> float:
    p = np.asarray(pred_pmf, dtype=float)
    p = p / (p.sum() + EPS)
    Fp = np.cumsum(p)
    if isinstance(true_k_or_pmf, (int, np.integer)):
        k = int(true_k_or_pmf)
        J = len(p)
        Ft = np.zeros(J, dtype=float)
        if k >= 0:
            k_clamped = min(k, J - 1)
            Ft[: k_clamped + 1] = 1.0
    else:
        q = np.asarray(true_k_or_pmf, dtype=float)
        q = q / (q.sum() + EPS)
        Ft = np.cumsum(q)
    return float(np.sum((Fp - Ft) ** 2))


def emd_1d(pred_pmf: np.ndarray, true_pmf: np.ndarray) -> float:
    p = np.asarray(pred_pmf, dtype=float)
    q = np.asarray(true_pmf, dtype=float)
    p = p / (p.sum() + EPS)
    q = q / (q.sum() + EPS)
    return float(np.sum(np.abs(np.cumsum(p) - np.cumsum(q))))


def nll_from_pmf(pred_pmf: np.ndarray, true_k_or_pmf: Any) -> float:
    p = np.asarray(pred_pmf, dtype=float)
    p = np.clip(p / (p.sum() + EPS), EPS, 1.0)
    if isinstance(true_k_or_pmf, (int, np.integer)):
        k = int(true_k_or_pmf)
        if k < 0 or k >= len(p):
            return float(-np.log(EPS))
        return float(-np.log(p[k]))
    else:
        q = np.asarray(true_k_or_pmf, dtype=float)
        q = q / (q.sum() + EPS)
        return float(-np.sum(q * np.log(p)))


def entropy(pmf: np.ndarray) -> float:
    p = np.asarray(pmf, dtype=float)
    p = np.clip(p / (p.sum() + EPS), EPS, 1.0)
    return float(-np.sum(p * np.log(p)))


def predicted_mean_var_from_pmf(pmf: np.ndarray) -> Tuple[float, float]:
    p = np.asarray(pmf, dtype=float)
    p = p / (p.sum() + EPS)
    bins = np.arange(len(p))
    mu = float(np.sum(bins * p))
    var = float(np.sum((bins ** 2) * p) - mu ** 2)
    return mu, var


def ensure_len(arr: Optional[np.ndarray], length: int, pad_value: float = 0.0) -> Optional[np.ndarray]:
    if arr is None:
        return None
    a = np.asarray(arr, dtype=float).flatten()
    if len(a) == length:
        return a
    if len(a) > length:
        return a[:length]
    out = np.full(length, pad_value, dtype=float)
    out[: len(a)] = a
    out[len(a) :] = a[-1] if len(a) > 0 else pad_value
    return out


# -------------------------
# Per-sample metrics
# -------------------------
def compute_per_sample_metrics(
    row: pd.Series,
    max_bin: Optional[int] = None,
    slice_threshold: float = 0.5,
    presence_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute per-sample metrics. Returns a dict of metrics and raw arrays for later aggregation.
    - slice_threshold: threshold to binarize gt_slice for slice AUC
    - presence_threshold: threshold to binarize p_presence_gt for binary AUC
    """
    out: Dict[str, Any] = {}

    # Ground-truth PMF q_k
    q_k = parse_array(row.get("q_k"))
    if q_k is None:
        raise ValueError("Missing q_k")
    q_k = np.asarray(q_k, dtype=float).flatten()
    q_k = q_k / (q_k.sum() + EPS)
    gt_mu = float(np.sum(np.arange(len(q_k)) * q_k))
    gt_var = float(np.sum((np.arange(len(q_k)) ** 2) * q_k) - gt_mu ** 2)
    out["gt_mu"] = gt_mu
    out["gt_var"] = gt_var
    out["gt_bin_argmax"] = int(np.argmax(q_k))

    # Predicted aggregated PMF
    pred_pmf = parse_array(row.get("nn_count_probs"))
    if pred_pmf is None:
        raise ValueError("Missing nn_count_probs")
    pred_pmf = np.asarray(pred_pmf, dtype=float).flatten()
    if max_bin is not None:
        pred_pmf = ensure_len(pred_pmf, max_bin + 1, pad_value=0.0)
    pred_pmf = pred_pmf / (pred_pmf.sum() + EPS)
    out["_pred_pmf"] = pred_pmf.tolist()

    # Predicted mean/var
    nn_mu, nn_var = predicted_mean_var_from_pmf(pred_pmf)
    out["nn_mu"] = nn_mu
    out["nn_var"] = nn_var

    # Distribution metrics
    out["nll_expected"] = nll_from_pmf(pred_pmf, q_k)
    out["crps"] = discrete_crps(pred_pmf, q_k)
    out["emd"] = emd_1d(pred_pmf, q_k)
    out["kl_true_pred"] = safe_kl(q_k, pred_pmf)
    out["kl_pred_true"] = safe_kl(pred_pmf, q_k)
    out["entropy"] = entropy(pred_pmf)

    # Presence: GT probabilistic and predicted
    p_zero_gt = float(q_k[0]) if q_k.size > 0 else 1.0
    p_presence_gt = 1.0 - p_zero_gt
    # predicted presence: prefer nn_binary if present, else 1 - pred_pmf[0]
    nn_binary = row.get("nn_binary", None)
    if nn_binary is None or (isinstance(nn_binary, float) and np.isnan(nn_binary)):
        p_presence_pred = 1.0 - float(pred_pmf[0]) if len(pred_pmf) > 0 else 0.0
    else:
        try:
            p_presence_pred = float(nn_binary)
        except Exception:
            p_presence_pred = 1.0 - float(pred_pmf[0]) if len(pred_pmf) > 0 else 0.0

    out["count_presence_prob"] = float(p_presence_pred)
    out["count_presence_prob_gt"] = float(p_presence_gt)

    # Brier and expected logloss for presence (continuous GT)
    out["brier_presence"] = float((p_presence_pred - p_presence_gt) ** 2)
    pp = np.clip(p_presence_pred, EPS, 1.0 - EPS)
    out["logloss_presence"] = float(-(p_presence_gt * np.log(pp) + (1.0 - p_presence_gt) * np.log(1.0 - pp)))

    # Binary conversions for AUC (user requested): convert probabilistic GT to binary using threshold
    out["count_presence_gt_binary"] = int(1 if p_presence_gt >= presence_threshold else 0)
    out["count_presence_pred_binary"] = float(1 if p_presence_pred >= presence_threshold else 0.0)

    # Slice-level: compare gt_slice (probabilistic) to nn_slice_logits (logits -> probs)
    gt_slice = parse_array(row.get("gt_slice"))
    slice_probs_pred = None
    # prefer nn_count_slice_probs if present (already probabilities)
    count_slice_probs = parse_array(row.get("nn_count_slice_probs"))
    if count_slice_probs is not None:
        slice_probs_pred = np.asarray(count_slice_probs, dtype=float).flatten()
        if np.any(slice_probs_pred < 0) or np.any(slice_probs_pred > 1):
            slice_probs_pred = _sigmoid(slice_probs_pred)
    else:
        # try nn_slice_logits or nn_slice
        slice_logits = parse_array(row.get("nn_slice_logits")) or parse_array(row.get("nn_slice"))
        if slice_logits is not None:
            slice_logits = np.asarray(slice_logits, dtype=float).flatten()
            # if these are logits, convert; if already probs, sigmoid will still map into (0,1)
            slice_probs_pred = _sigmoid(slice_logits)

    out["count_slice_probs"] = None if slice_probs_pred is None else slice_probs_pred.tolist()

    # Compute slice-level continuous metrics if both arrays present and same length
    if gt_slice is not None and slice_probs_pred is not None:
        gt_slice = np.asarray(gt_slice, dtype=float).flatten()
        minlen = min(len(gt_slice), len(slice_probs_pred))
        if minlen > 0:
            gt_slice = gt_slice[:minlen]
            pred_slice = slice_probs_pred[:minlen]
            out["_gt_slice"] = gt_slice.tolist()
            # continuous metrics
            out["slice_brier_mean"] = float(np.mean((pred_slice - gt_slice) ** 2))
            pred_clip = np.clip(pred_slice, EPS, 1.0 - EPS)
            out["slice_logloss_mean"] = float(-np.mean(gt_slice * np.log(pred_clip) + (1.0 - gt_slice) * np.log(1.0 - pred_clip)))
            # correlations
            try:
                out["slice_pearson"] = float(np.corrcoef(pred_slice, gt_slice)[0, 1])
            except Exception:
                out["slice_pearson"] = float("nan")
            try:
                out["slice_spearman"] = float(spearmanr(pred_slice, gt_slice).correlation)
            except Exception:
                out["slice_spearman"] = float("nan")
            # binary AUC for slices: convert GT slice to binary using slice_threshold
            gt_slice_bin = (gt_slice >= slice_threshold).astype(int)
            # Only compute slice AUC if both classes present in the binarized GT
            if np.unique(gt_slice_bin).size > 1:
                out["slice_auc_sample"] = safe_roc_auc(gt_slice_bin, pred_slice)
            else:
                out["slice_auc_sample"] = float("nan")
            # also save fraction of positive GT slices (probabilistic -> thresholded)
            out["slice_pos_fraction_gt"] = float(np.mean(gt_slice_bin))
        else:
            out["_gt_slice"] = None
            out["slice_brier_mean"] = None
            out["slice_logloss_mean"] = None
            out["slice_pearson"] = None
            out["slice_spearman"] = None
            out["slice_auc_sample"] = None
            out["slice_pos_fraction_gt"] = None
    else:
        out["_gt_slice"] = None
        out["slice_brier_mean"] = None
        out["slice_logloss_mean"] = None
        out["slice_pearson"] = None
        out["slice_spearman"] = None
        out["slice_auc_sample"] = None
        out["slice_pos_fraction_gt"] = None

    # metadata
    out["pred_mode"] = int(np.argmax(pred_pmf))
    out["n_bins"] = int(len(pred_pmf))
    out["n_slices"] = int(len(slice_probs_pred)) if slice_probs_pred is not None else None
    out["dataset_idx"] = int(row.get("dataset_idx")) if row.get("dataset_idx") is not None and not (isinstance(row.get("dataset_idx"), float) and math.isnan(row.get("dataset_idx"))) else None
    out["file"] = row.get("file", None)
    out["offset"] = int(row.get("offset")) if row.get("offset") is not None and not (isinstance(row.get("offset"), float) and math.isnan(row.get("offset"))) else None

    return out


# -------------------------
# Aggregation
# -------------------------
def aggregate_metrics(per_sample: pd.DataFrame, groups: Iterable[Tuple[str, np.ndarray]]) -> pd.DataFrame:
    rows = []
    for group_name, mask in groups:
        dfg = per_sample.loc[mask]
        n = len(dfg)
        if n == 0:
            continue
        row = {"group": group_name, "n_samples": n}
        # Means
        mae_mu = float(np.mean(np.abs(dfg["nn_mu"].values - dfg["gt_mu"].values)))
        rmse_mu = float(np.sqrt(np.mean((dfg["nn_mu"].values - dfg["gt_mu"].values) ** 2)))
        bias_mu = float(np.mean(dfg["nn_mu"].values - dfg["gt_mu"].values))
        row.update({"mae_mu": mae_mu, "rmse_mu": rmse_mu, "bias_mu": bias_mu})

        # Variance
        mae_var = float(np.mean(np.abs(dfg["nn_var"].values - dfg["gt_var"].values)))
        rmse_var = float(np.sqrt(np.mean((dfg["nn_var"].values - dfg["gt_var"].values) ** 2)))
        row.update({"mae_var": mae_var, "rmse_var": rmse_var})

        # Correlation
        try:
            r, _ = pearsonr(dfg["nn_mu"].values, dfg["gt_mu"].values)
            r2 = float(r ** 2)
        except Exception:
            r = float("nan")
            r2 = float("nan")
        row.update({"pearson_r": float(r), "r2": float(r2)})

        # Distribution metrics
        row.update(
            {
                "nll": float(np.nanmean(dfg["nll_expected"].values)),
                "crps": float(np.nanmean(dfg["crps"].values)),
                "emd": float(np.nanmean(dfg["emd"].values)),
                "kl_true_pred": float(np.nanmean(dfg["kl_true_pred"].values)),
                "kl_pred_true": float(np.nanmean(dfg["kl_pred_true"].values)),
                "entropy": float(np.nanmean(dfg["entropy"].values)),
            }
        )

        # Presence: continuous aggregates and binary AUC (after converting GT probabilistic -> binary)
        # Continuous: mean Brier and logloss
        row.update(
            {
                "brier_presence_mean": float(np.nanmean(dfg["brier_presence"].values)),
                "logloss_presence_mean": float(np.nanmean(dfg["logloss_presence"].values)),
            }
        )
        # Correlation between predicted presence prob and GT presence prob
        try:
            corr_pres, _ = pearsonr(dfg["count_presence_prob"].values, dfg["count_presence_prob_gt"].values)
        except Exception:
            corr_pres = float("nan")
        row.update({"presence_pearson": float(corr_pres)})

        # Binary AUC for presence: convert GT probabilistic to binary using threshold stored in column
        if "count_presence_gt_binary" in dfg.columns:
            y_true_bin = dfg["count_presence_gt_binary"].values.astype(int)
            y_pred_prob = dfg["count_presence_prob"].values.astype(float)
            auc_presence = safe_roc_auc(y_true_bin, y_pred_prob)
        else:
            auc_presence = float("nan")
        row.update({"count_binary_auc": auc_presence})

        # Slice aggregates: mean slice brier/logloss/pearson where available
        row.update(
            {
                "slice_brier_mean": float(np.nanmean(dfg["slice_brier_mean"].values)) if "slice_brier_mean" in dfg.columns else float("nan"),
                "slice_logloss_mean": float(np.nanmean(dfg["slice_logloss_mean"].values)) if "slice_logloss_mean" in dfg.columns else float("nan"),
                "slice_pearson_mean": float(np.nanmean(dfg["slice_pearson"].values)) if "slice_pearson" in dfg.columns else float("nan"),
            }
        )

        # Slice AUC aggregated (flattened) using only samples with valid slice arrays
        gt_slices_all = []
        pred_slices_all = []
        for _, r_ in dfg.iterrows():
            gt_slice = r_.get("_gt_slice")
            pred_slice = r_.get("count_slice_probs")
            if gt_slice is None or pred_slice is None:
                continue
            gt_a = np.asarray(gt_slice, dtype=float).flatten()
            pred_a = np.asarray(pred_slice, dtype=float).flatten()
            minlen = min(len(gt_a), len(pred_a))
            if minlen == 0:
                continue
            # binarize GT slice for AUC using default 0.5 threshold
            gt_bin = (gt_a[:minlen] >= 0.5).astype(int)
            pred_prob = pred_a[:minlen]
            gt_slices_all.append(gt_bin)
            pred_slices_all.append(pred_prob)
        if len(gt_slices_all) > 0:
            gt_concat = np.concatenate(gt_slices_all)
            pred_concat = np.concatenate(pred_slices_all)
            slice_auc = safe_roc_auc(gt_concat, pred_concat)
        else:
            slice_auc = float("nan")
        row.update({"slice_auc": slice_auc})

        rows.append(row)
    return pd.DataFrame(rows)


# -------------------------
# Bootstrap CI helper
# -------------------------
def bootstrap_ci(metric_fn, idxs: np.ndarray, n_boot: int = 500, alpha: float = 0.05, rng=None):
    if rng is None:
        rng = np.random.default_rng(12345)
    N = len(idxs)
    if N == 0:
        return float("nan"), float("nan"), float("nan")
    est = metric_fn(idxs)
    boots = []
    for _ in range(n_boot):
        sample_idx = rng.integers(0, N, size=N)
        boots.append(metric_fn(idxs[sample_idx]))
    lo = float(np.percentile(boots, 100 * (alpha / 2.0)))
    hi = float(np.percentile(boots, 100 * (1.0 - alpha / 2.0)))
    return float(est), lo, hi


# -------------------------
# Main (no CLI)
# -------------------------
def main():
    if config is None:
        print("[ERROR] Could not import configs.nn_config; ensure project config is importable.", file=sys.stderr)
        sys.exit(2)

    ckpt_dir = getattr(config, "CHECKPOINT_DIR", None)
    if ckpt_dir is None:
        print("[ERROR] CHECKPOINT_DIR not found in config.", file=sys.stderr)
        sys.exit(2)

    pattern = os.path.join(ckpt_dir, "*_multiband_calibration_full.csv")
    matches = sorted(glob.glob(pattern))
    if len(matches) == 0:
        pattern2 = os.path.join(ckpt_dir, "*_multiband_calibration.csv")
        matches = sorted(glob.glob(pattern2))
        if len(matches) == 0:
            print(f"[ERROR] No calibration CSV found in {ckpt_dir}.", file=sys.stderr)
            sys.exit(2)

    csv_path = matches[-1]
    outdir = os.path.dirname(csv_path)
    print(f"[INFO] Using calibration CSV: {csv_path}")

    # thresholds (allow override from config)
    slice_threshold = getattr(config, "EVAL_SLICE_THRESHOLD", 0.5)
    presence_threshold = getattr(config, "EVAL_PRESENCE_THRESHOLD", 0.5)
    boot_n = getattr(config, "EVAL_BOOTSTRAP", 500)
    try:
        boot_n = int(boot_n)
    except Exception:
        boot_n = 500
    boot_n = max(100, boot_n)

    df = pd.read_csv(csv_path)
    print(f"[INFO] Loaded {len(df)} rows")

    per_sample_records = []
    failed = 0
    first_row = df.iloc[0]
    max_bin = None
    if "max_bin" in df.columns:
        try:
            max_bin = int(first_row.get("max_bin"))
        except Exception:
            max_bin = None

    for idx, row in df.iterrows():
        try:
            rec = compute_per_sample_metrics(row, max_bin=max_bin, slice_threshold=slice_threshold, presence_threshold=presence_threshold)
            rec["row_index"] = int(idx)
            per_sample_records.append(rec)
        except Exception as exc:
            failed += 1
            print(f"[WARN] failed computing metrics for row {idx}: {exc}")

    if len(per_sample_records) == 0:
        print("[ERROR] No per-sample metrics computed. Exiting.", file=sys.stderr)
        sys.exit(3)

    per_sample_df = pd.DataFrame(per_sample_records)

    # Preventive diagnostics (probabilistic-aware)
    print("[INFO] Running probabilistic diagnostics...")
    # p_presence_gt distribution
    per_sample_df["p_presence_gt"] = per_sample_df["_pred_pmf"].apply(lambda x: np.nan)  # placeholder
    # compute p_presence_gt from stored _q_k in original CSV rows: reconstruct quickly
    # We have gt_mu and gt_var but not q_k in DataFrame columns; use _q_k if present
    if "_q_k" in per_sample_df.columns:
        per_sample_df["p_presence_gt"] = per_sample_df["_q_k"].apply(lambda q: 1.0 - float(np.asarray(q)[0]) if q is not None else np.nan)
    else:
        # fallback: approximate presence from gt_mu > 0
        per_sample_df["p_presence_gt"] = (per_sample_df["gt_mu"].values > 0.0).astype(float)

    print("[INFO] p_presence_gt stats: min,median,mean,max:", per_sample_df["p_presence_gt"].min(), per_sample_df["p_presence_gt"].median(), per_sample_df["p_presence_gt"].mean(), per_sample_df["p_presence_gt"].max())
    print("[INFO] count samples with p_presence_gt >= 0.5:", int((per_sample_df["p_presence_gt"] >= presence_threshold).sum()))

    # gt_slice types
    def slice_type(s):
        if s is None:
            return "missing"
        a = np.asarray(s, dtype=float).flatten()
        if a.size == 0:
            return "empty"
        uniq = np.unique(a)
        if uniq.size == 1 and (uniq[0] in (0.0, 1.0)):
            return "binary"
        return "probabilistic"

    per_sample_df["_gt_slice_type"] = per_sample_df["_gt_slice"].apply(slice_type)
    print("[INFO] gt_slice types:", per_sample_df["_gt_slice_type"].value_counts().to_dict())

    # Show mismatch examples: p_presence_gt > 0 but empty gt_slice
    mask_mismatch = (per_sample_df["p_presence_gt"] > 1e-6) & (per_sample_df["_gt_slice_type"] == "empty")
    print(f"[INFO] samples with p_presence_gt>0 but empty gt_slice: {mask_mismatch.sum()} (examples indices: {per_sample_df.loc[mask_mismatch].head().index.tolist()})")

    # Save per-sample CSV
    per_sample_csv = os.path.join(outdir, "metrics_per_sample.csv")
    per_sample_df.to_csv(per_sample_csv, index=False)
    print(f"[INFO] Wrote per-sample metrics to {per_sample_csv} (failed rows: {failed})")

    # Aggregations: overall and optional stratified by rounded gt_mu
    groups = []
    mask_all = np.ones(len(per_sample_df), dtype=bool)
    groups.append(("overall", mask_all))

    # optional stratification by rounded gt_mu (useful diagnostics)
    try:
        per_sample_df["gt_mu_round"] = per_sample_df["gt_mu"].round().astype(int)
        unique_rounds = sorted(per_sample_df["gt_mu_round"].unique().tolist())
        for r in unique_rounds:
            groups.append((f"gt_mu_round_{r}", per_sample_df["gt_mu_round"].values == r))
    except Exception:
        pass

    summary_df = aggregate_metrics(per_sample_df, groups)

    # Bootstrap CIs for prioritized metrics
    rng = np.random.default_rng(123456)
    ci_results = []
    for _, row in summary_df.iterrows():
        gname = row["group"]
        if gname == "overall":
            mask = np.ones(len(per_sample_df), dtype=bool)
        else:
            # parse group name for gt_mu_round
            if gname.startswith("gt_mu_round_"):
                r = int(gname.split("_")[-1])
                mask = per_sample_df["gt_mu_round"].values == r
            else:
                mask = np.ones(len(per_sample_df), dtype=bool)
        idxs = np.where(mask)[0]

        def mae_mu_fn(idxs_arr):
            vals = per_sample_df["nn_mu"].values[idxs_arr] - per_sample_df["gt_mu"].values[idxs_arr]
            return float(np.mean(np.abs(vals)))

        def emd_fn(idxs_arr):
            return float(np.nanmean(per_sample_df["emd"].values[idxs_arr]))

        def nll_fn(idxs_arr):
            return float(np.nanmean(per_sample_df["nll_expected"].values[idxs_arr]))

        def slice_auc_fn(idxs_arr):
            gt_concat = []
            pred_concat = []
            for ii in idxs_arr:
                r = per_sample_df.iloc[ii]
                gt_slice = r.get("_gt_slice")
                pred_slice = r.get("count_slice_probs")
                if gt_slice is None or pred_slice is None:
                    continue
                gt_a = np.asarray(gt_slice, dtype=float).flatten()
                pred_a = np.asarray(pred_slice, dtype=float).flatten()
                minlen = min(len(gt_a), len(pred_a))
                if minlen == 0:
                    continue
                gt_bin = (gt_a[:minlen] >= slice_threshold).astype(int)
                pred_prob = pred_a[:minlen]
                gt_concat.append(gt_bin)
                pred_concat.append(pred_prob)
            if len(gt_concat) == 0:
                return float("nan")
            gt_all = np.concatenate(gt_concat)
            pred_all = np.concatenate(pred_concat)
            return safe_roc_auc(gt_all, pred_all)

        def count_auc_fn(idxs_arr):
            y_true = per_sample_df["count_presence_gt_binary"].values[idxs_arr].astype(int)
            y_pred = per_sample_df["count_presence_prob"].values[idxs_arr]
            return safe_roc_auc(y_true, y_pred)

        est_mae, lo_mae, hi_mae = bootstrap_ci(mae_mu_fn, idxs, n_boot=boot_n, rng=rng)
        est_emd, lo_emd, hi_emd = bootstrap_ci(emd_fn, idxs, n_boot=boot_n, rng=rng)
        est_nll, lo_nll, hi_nll = bootstrap_ci(nll_fn, idxs, n_boot=boot_n, rng=rng)
        est_slice_auc, lo_slice_auc, hi_slice_auc = bootstrap_ci(slice_auc_fn, idxs, n_boot=boot_n, rng=rng)
        est_count_auc, lo_count_auc, hi_count_auc = bootstrap_ci(count_auc_fn, idxs, n_boot=boot_n, rng=rng)

        ci_results.append(
            {
                "group": gname,
                "mae_mu": est_mae,
                "mae_mu_lo": lo_mae,
                "mae_mu_hi": hi_mae,
                "emd": est_emd,
                "emd_lo": lo_emd,
                "emd_hi": hi_emd,
                "nll": est_nll,
                "nll_lo": lo_nll,
                "nll_hi": hi_nll,
                "slice_auc": est_slice_auc,
                "slice_auc_lo": lo_slice_auc,
                "slice_auc_hi": hi_slice_auc,
                "count_binary_auc": est_count_auc,
                "count_binary_auc_lo": lo_count_auc,
                "count_binary_auc_hi": hi_count_auc,
            }
        )

    ci_df = pd.DataFrame(ci_results)
    metrics_summary = summary_df.merge(ci_df, on="group", how="left")
    summary_csv = os.path.join(outdir, "metrics_summary.csv")
    metrics_summary.to_csv(summary_csv, index=False)
    print(f"[INFO] Wrote aggregated metrics and bootstrap CIs to {summary_csv}")

    print("[INFO] Evaluation complete.")


if __name__ == "__main__":
    main()
