# frog_perch/experiments/logistic_sweep_helpers.py
"""
Helper utilities for logistic confidence sweeps.
Contains evaluate_without_smoothing(), which performs a clean
final validation pass without label smoothing and computes
one-vs-rest PR curves + AP scores per class.
"""

import os
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from frog_perch.datasets.frog_dataset import FrogPerchDataset
from frog_perch.training.dataset_builders import build_tf_val_dataset
import frog_perch.config as config


def evaluate_without_smoothing(model, batch_size, tag, sweep_dir):
    """
    Build deterministic validation windows without label smoothing,
    collect predictions, compute one-vs-rest PR curves, save plots,
    and return a dict of AP scores.
    """

    # Build val dataset with hard labels
    val_ds_obj = FrogPerchDataset(
        audio_dir=config.AUDIO_DIR,
        annotation_dir=config.ANNOTATION_DIR,
        train=False,
        pos_ratio=None,
        random_seed=config.RANDOM_SEED,
        label_mode=config.LABEL_MODE,
        val_stride_sec=1.0,
        q2_confidence=config.Q2_CONFIDENCE,
        equalize_q2_val=True,
        use_continuous_confidence=False,
        confidence_params={}
    )

    val_ds = build_tf_val_dataset(val_ds_obj, batch_size=batch_size)

    # gather predictions
    y_true = np.concatenate([y for _, y in val_ds], axis=0)
    y_pred = model.predict(val_ds, verbose=0)

    n_classes = y_true.shape[1]
    ap_scores = {}

    for i in range(n_classes):
        yt = y_true[:, i].astype(int)
        yp = y_pred[:, i]

        prec, rec, _ = precision_recall_curve(yt, yp)
        ap = average_precision_score(yt, yp)
        ap_scores[f"class_{i}_AP"] = float(ap)

        # save PR plot
        plt.figure()
        plt.plot(rec, prec, label=f"Class {i} (AP={ap:.3f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision–Recall Curve — {tag}, Class {i}")
        plt.legend(loc="lower left")

        out_path = os.path.join(sweep_dir, f"{tag}_pr_class{i}.png")
        plt.savefig(out_path)
        plt.close()

    return ap_scores
