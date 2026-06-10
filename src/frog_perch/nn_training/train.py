import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt   # ONLY ADDITION

from frog_perch.datasets.frog_dataset import FrogPerchDataset
from frog_perch.nn_models.downstream import build_downstream
from frog_perch.nn_training.dataset_builders import build_tf_dataset, build_tf_val_dataset

from frog_perch.nn_training.metrics import (
    NormalizedEarthMoversDistance1D,
    ExpectedCountMAE,
    SoftBinaryAccuracy, 
    SoftAUC
)

from frog_perch.nn_training.callbacks import (
    GPUMemoryCallback
)

class ValidationVisualizer(tf.keras.callbacks.Callback):
    def __init__(
        self,
        val_ds_obj,
        inspection_model=None,
        out_dir="./val_debug",
        freq=1,
        samples_per_epoch=12,
        num_bins=5,
        seed=42,
    ):
        super().__init__()
        self.val_ds_obj = val_ds_obj
        self.inspection_model = inspection_model
        self.out_dir = out_dir
        self.freq = freq
        self.samples_per_epoch = samples_per_epoch
        self.rng = np.random.default_rng(seed)

        os.makedirs(out_dir, exist_ok=True)

        # -----------------------------------------------------
        # STRATIFICATION (fixed once, reused every epoch)
        # -----------------------------------------------------
        self.strat = self._build_stratification(num_bins)

    def _build_stratification(self, num_bins):
        scores = []
        for i in range(len(self.val_ds_obj)):
            _, labels, _, _, _ = self.val_ds_obj[i]
            dist = np.array(labels["count_probs"])
            k = np.arange(len(dist))
            scores.append(np.sum(k * dist))

        scores = np.array(scores)
        bins = np.quantile(scores, np.linspace(0, 1, num_bins + 1))

        strat = {i: [] for i in range(num_bins)}
        for i, s in enumerate(scores):
            b = np.digitize(s, bins) - 1
            b = np.clip(b, 0, num_bins - 1)
            strat[b].append(i)

        return strat

    def _sample_index(self):
        bins = [b for b in self.strat if len(self.strat[b]) > 0]
        b = self.rng.choice(bins)
        return int(self.rng.choice(self.strat[b]))

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _pb_monte_carlo(self, p, n_samples=200):
        """
        Empirical PB approximation for slice branch.
        """
        draws = np.random.binomial(1, p[None, :], size=(n_samples, len(p)))
        return np.bincount(draws.sum(axis=1), minlength=len(p) + 1)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.freq != 0:
            return

        epoch_dir = os.path.join(self.out_dir, f"epoch_{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)

        for i in range(self.samples_per_epoch):
            idx = self._sample_index()

            spatial, labels, _, _, events = self.val_ds_obj[idx]
            predict_model = self.inspection_model or self.model
            preds = predict_model.predict(spatial[None, ...], verbose=0)

            # -------------------------------------------------
            # TRUE — (max_bin+1,)
            # -------------------------------------------------
            y_count = np.asarray(labels["count_probs"])
            k_count = np.arange(len(y_count))           # max_bin+1

            # -------------------------------------------------
            # MODEL OUTPUTS
            # -------------------------------------------------
            slice_logits = preds["slice"][0]            # (n_slices,)
            count_logits = preds["count_logits"][0]     # (n_slices,)
            slice_probs  = preds["slice_probs"][0]      # (n_slices,)
            count_probs  = preds["count_probs"][0]      # (max_bin+1,)
            y_slice = np.asarray(labels["slice"])   # (n_slices,)

            k_slices = np.arange(len(slice_logits))     # n_slices

            slice_probs_from_logits = self._sigmoid(slice_logits)
            count_probs_from_logits = self._sigmoid(count_logits)

            # =================================================
            # FIGURE 1: RAW LOGIT STRUCTURE COMPARISON
            # both (n_slices,) — no explicit k needed
            # =================================================
            plt.figure()
            plt.plot(slice_logits, label="slice_logits")
            plt.plot(count_logits, label="count_logits")
            plt.title(f"Raw logits comparison idx={idx}")
            plt.legend()
            plt.savefig(os.path.join(epoch_dir, f"sample_{i}_logits.png"))
            plt.close()

            # =================================================
            # FIGURE 2: INTENSITY FIELD COMPARISON
            # =================================================
            fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)

            for s, e, c in zip(events["starts"], events["ends"], events["conf"]):
                for ax in axes:
                    ax.axvspan(s, e, alpha=c * 0.3, color="green", label=None)

            axes[0].step(k_slices, y_slice, where="post", color="black")
            axes[0].set_ylabel("ground truth")
            axes[0].set_ylim(-0.05, 1.05)

            axes[1].step(k_slices, slice_probs_from_logits, where="post", color="tab:blue")
            axes[1].set_ylabel("slice head")
            axes[1].set_ylim(-0.05, 1.05)

            axes[2].step(k_slices, count_probs_from_logits, where="post", color="tab:orange")
            axes[2].set_ylabel("count head")
            axes[2].set_ylim(-0.05, 1.05)

            axes[2].set_xlabel("slice index")
            fig.suptitle(f"Per-slice intensities idx={idx}")
            plt.tight_layout()
            plt.savefig(os.path.join(epoch_dir, f"sample_{i}_intensity.png"))
            plt.close()

            # =================================================
            # FIGURE 3: PB DISTRIBUTION COMPARISON
            # y_count + count_probs: (max_bin+1,) — use k_count
            # slice MC approx:       (n_slices+1,) — use pb_k
            # =================================================
            slice_pb = self._pb_monte_carlo(slice_probs_from_logits)
            pb_k = np.arange(len(slice_pb))             # n_slices+1

            plt.figure()
            plt.bar(k_count, y_count, alpha=0.4, label="true")
            plt.plot(k_count, count_probs, label="count branch PB")
            plt.plot(pb_k, slice_pb / np.sum(slice_pb), label="slice branch PB (MC approx)")
            plt.title(f"PB comparison idx={idx}")
            plt.legend()
            plt.savefig(os.path.join(epoch_dir, f"sample_{i}_pb.png"))
            plt.close()

def inspect_ds(ds_obj: FrogPerchDataset, name: str) -> None:
    expected_counts = []
    for i in range(len(ds_obj)):
        _, labels, _, _, _ = ds_obj[i]
        dist = np.array(labels["count_probs"])
        k = np.arange(len(dist))
        expected_counts.append(np.sum(k * dist))

    expected_counts = np.array(expected_counts)
    print(f"\n========== {name} STATS ==========")
    print(f"Samples: {len(expected_counts)}")
    print(f"Mean expected events: {expected_counts.mean():.4f}")
    print(f"Max expected events:  {expected_counts.max():.4f}")


def train(cfg: dict) -> tuple[tf.keras.Model, tf.data.Dataset]:

    epochs = cfg.get("EPOCHS", 100)
    batch_size = cfg.get("BATCH_SIZE", 32)
    val_stride = cfg.get("VAL_STRIDE_SEC", 1.0)
    steps_per_epoch = cfg.get("STEPS_PER_EPOCH", 100)
    confidence_params = cfg.get("CONFIDENCE_PARAMS", {})
    max_bin = cfg.get("MAX_BIN", 16)
    n_slices = cfg.get("N_SLICES", 32)

    dataset_kwargs = dict(
        audio_dir=cfg["AUDIO_DIR"],
        annotation_dir=cfg["ANNOTATION_DIR"],
        random_seed=cfg.get("RANDOM_SEED", 42),
        confidence_params=confidence_params,
        n_slices=n_slices, 
        max_bin=max_bin,    
    )

    train_ds_obj = FrogPerchDataset(
        split_type='train', 
        sampling_alpha=cfg.get("SAMPLING_ALPHA", 0.5), 
        **dataset_kwargs
    )
    val_ds_obj   = FrogPerchDataset(split_type='val', val_stride_sec=val_stride, **dataset_kwargs)
    test_ds_obj  = FrogPerchDataset(split_type='test', val_stride_sec=val_stride, **dataset_kwargs)

    train_ds = build_tf_dataset(train_ds_obj, batch_size=batch_size)
    val_ds   = build_tf_val_dataset(val_ds_obj, batch_size=batch_size)
    test_ds  = build_tf_val_dataset(test_ds_obj, batch_size=batch_size)

    inspect_ds(val_ds_obj, "VALIDATION")
    inspect_ds(test_ds_obj, "TEST")

    model = build_downstream(
        spatial_shape=cfg.get("SPATIAL_SHAPE", (16, 4, 1536)),
        slice_hidden_dims=cfg.get("SLICE_HIDDEN_DIMS", (512, 256)),
        temporal_dim=cfg.get("TEMPORAL_DIM", 256),
        num_temporal_layers=cfg.get("NUM_TEMPORAL_LAYERS", 2),
        kernel_size=cfg.get("KERNEL_SIZE", 3),
        activation=cfg.get("ACTIVATION", "gelu"),
        dropout=cfg.get("DROPOUT", 0.1),
        l2_reg=cfg.get("L2_REG", 1e-4),
        use_gating=cfg.get("USE_GATING", True),
        max_bin=max_bin,
        n_slices=n_slices,
    )

    # Secondary model for visualizer — shares all weights, adds count_logits output
    count_logits_tensor = model.get_layer("count_dense").output
    count_logits_flat = tf.keras.layers.Flatten()(count_logits_tensor)
    slice_output = model.get_layer("slice").output
    slice_probs = tf.keras.layers.Activation("sigmoid")(slice_output)

    inspection_model = tf.keras.Model(
        inputs=model.input,
        outputs={
            **{k: model.output[k] for k in model.output},
            "slice_probs":  slice_probs,
            "count_logits": count_logits_flat,
        }
    )

    losses = {
        "binary":       tf.keras.losses.BinaryCrossentropy(),
        "slice":        tf.keras.losses.BinaryCrossentropy(from_logits=True),
        "count_probs":  NormalizedEarthMoversDistance1D(),
    }

    metrics = {
        "binary": [
            SoftBinaryAccuracy(name="bin_acc"),
            SoftAUC(name="bin_auc"),
        ],
        "slice": [
            SoftBinaryAccuracy(name="slice_acc", threshold=0.0),
            SoftAUC(name="slice_auc", from_logits=True), 
        ],
    }

    model.compile(
        optimizer=tf.keras.optimizers.Muon(
            learning_rate=cfg.get("LEARNING_RATE", 1e-3),
            exclude_layers=["slice_dense", "count_dense"],
        ),
        loss=losses,
        loss_weights={
            "binary": 0.1,
            "slice": 0.1,
            "count_probs": 1.0
        },
        metrics=metrics,
    )

    checkpoint_dir = cfg.get("CHECKPOINT_DIR", "./checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, "best.keras"),
            monitor="val_count_probs_loss", 
            save_best_only=True,
            mode="min", 
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_count_probs_loss", 
            patience=10,
            restore_best_weights=True,
            mode="min", 
        ),
        GPUMemoryCallback(),

        # =====================================================
        # FIXED: visualization (now stratified + safe)
        # =====================================================
        ValidationVisualizer(val_ds_obj, inspection_model=inspection_model, freq=1),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        verbose=2,
        callbacks=callbacks,
    )

    print("\n[INFO] Evaluating on test set...")
    test_results = model.evaluate(test_ds)
    print(f"Test Results: {test_results}")

    return model, val_ds