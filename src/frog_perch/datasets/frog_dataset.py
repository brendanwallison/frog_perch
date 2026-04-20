import os
import json
import random
import hashlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf

from frog_perch.nn_models.perch_wrapper import PerchWrapper

from frog_perch.utils.audio import load_audio, resample_array
from frog_perch.utils.annotations import (
    get_event_cache,
    compute_event_confidence,
    compute_window_overlap,
    compute_slice_overlap_matrix,
    binary_clip_probability,
    soft_count_distribution,
)

class FrogPerchDataset:
    """
    Dataset for frog call detection using Perch embeddings.

    This dataset supports:
    - Stochastic training sampling over valid frame ranges
    - Deterministic validation sliding-window evaluation
    - Multi-head label generation from shared event representation:
        * binary: probability of at least one event in window
        * count_probs: Poisson-binomial distribution over event counts
        * slice: temporal event distribution over fixed time slices

    All labels are derived from a unified event computation pipeline.
    """
    def __init__(
        self,
        audio_dir: str,       
        annotation_dir: str,      
        split_type: str = "train",
        test_split: float = 0.15,
        val_split: float = 0.1,
        sampling_alpha: float = 0.3,
        random_seed: int = 41,
        val_stride_sec: float = 1.0,
        use_continuous_confidence: bool = True,
        confidence_params: dict = None,
        # Exposed Perch/Audio settings with original defaults
        dataset_sample_rate: int = 16000,
        clip_duration_seconds: float = 5.0,
        perch_sample_rate: int = 32000,
        perch_clip_seconds: float = 5.0,
    ):
        self.audio_dir = audio_dir
        self.annotation_dir = annotation_dir
        self.split_type = split_type
        self.train = (split_type == "train")

        self.sampling_alpha = sampling_alpha
        self.test_split = test_split
        self.val_split = val_split
        self.val_stride_sec = val_stride_sec

        self.random_seed = random_seed
        self.use_continuous_confidence = use_continuous_confidence

        confidence_params = confidence_params or {}
        self.duration_stats = confidence_params.get("duration_stats")
        self.bandwidth_stats = confidence_params.get("bandwidth_stats")
        self.logistic_params = confidence_params.get("logistic_params", {})

        # Audio + model configuration (No longer relies on config module)
        self.sample_rate = dataset_sample_rate
        self.clip_samples = int(clip_duration_seconds * self.sample_rate)

        self.perch_sr = perch_sample_rate
        self.perch_samples = int(perch_sample_rate * perch_clip_seconds)

        self.window_stride = 0.1

        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        # Split + caching setup
        self.audio_files = self._load_or_create_split()

        if self.train:
            self.metadata_path = self._get_metadata_path()

            if os.path.exists(self.metadata_path):
                self._load_metadata()
            else:
                self._compute_and_save_metadata()

        else:
            self._build_validation_index()

        # Perch embedding model
        self.perch = PerchWrapper()

        # Caches
        self._annotation_cache = {}

    def _get_split_path(self):
        key = f"{self.random_seed}_{self.test_split}_{self.val_split}"
        h = hashlib.md5(key.encode()).hexdigest()

        return os.path.join(
            self.audio_dir,
            f"dataset_split_{h}.json"
        )


    def _group_files_by_date_hour(self):
        """
        Group recordings by embedded timestamp key.

        This prevents data leakage between splits by ensuring
        temporally adjacent clips remain in the same partition.
        """
        groups = {}

        for fname in os.listdir(self.audio_dir):
            if not fname.endswith(".wav"):
                continue

            parts = fname.split("__")

            if len(parts) < 2:
                # Fallback group for malformed filenames
                key = f"ungrouped_{fname}"
            else:
                # Extract YYYYMMDD_HHMMSS style prefix
                sub = parts[1].split("_")
                key = f"{sub[0]}_{sub[1]}"

            groups.setdefault(key, []).append(fname)

        return groups

    def _build_validation_index(self):
        """
        Build deterministic sliding-window evaluation index.

        Each audio file is converted into a fixed stride grid of
        clip start times for evaluation consistency.
        """
        self.val_index = []

        for audio_file in self.audio_files:
            path = os.path.join(self.audio_dir, audio_file)

            try:
                info = sf.info(path)
                total_samples = int(info.frames)
            except:
                continue

            total_seconds = total_samples / self.sample_rate
            clip_seconds = self.clip_samples / self.sample_rate

            # Generate evenly spaced evaluation windows
            starts = np.arange(
                0,
                max(0, total_seconds - clip_seconds),
                self.val_stride_sec,
            )

            for s in starts:
                self.val_index.append((audio_file, float(s)))

    def _get_metadata_path(self):
        """
        Return hash-stable path for cached training metadata.

        The hash ensures that:
        - dataset subset
        - sampling regime (alpha)
        - temporal discretization (stride)

        all define a unique cache.
        """

        key_parts = [
            "files:" + "|".join(sorted(self.audio_files)),
            f"alpha:{self.sampling_alpha}",
            f"stride:{self.window_stride}",
            f"split:{self.split_type}",
        ]

        key = "||".join(key_parts)
        h = hashlib.md5(key.encode()).hexdigest()

        return os.path.join(
            self.audio_dir,
            f"clip_metadata_{self.split_type}_{h}.json"
        )

    def _load_metadata(self):
        with open(self.metadata_path, "r") as f:
            meta = json.load(f)

        # --------------------------------------------------
        # core sampling structures
        # --------------------------------------------------
        self.window_intensity = {
            k: np.array(v, dtype=np.float32)
            for k, v in meta["window_intensity"].items()
        }

        self.window_background = {
            k: np.array(v, dtype=np.float32)
            for k, v in meta["window_background"].items()
        }

        self.file_weights = {
            k: float(v)
            for k, v in meta["file_weights"].items()
        }

        # --------------------------------------------------
        # config / reproducibility
        # --------------------------------------------------
        self.window_stride = meta["stride_sec"]
        self.sample_rate = meta["sample_rate"]
        self.clip_samples = meta["clip_samples"]
        self.sampling_alpha = meta["sampling_alpha"]


    def _compute_and_save_metadata(self):
        """
        Build and persist metadata using existing deterministic builders.
        This ensures cache consistency with runtime sampling logic.
        """

        print("Computing metadata (via existing builders).")

        # --------------------------------------------------
        # Step 1: build window-level statistics (canonical source)
        # --------------------------------------------------
        self._build_window_intensity()

        # --------------------------------------------------
        # Step 2: build file-level marginals
        # --------------------------------------------------
        self._build_global_file_marginals()

        # --------------------------------------------------
        # Step 3: serialize ONLY derived artifacts
        # --------------------------------------------------

        window_intensity = {
            k: v.tolist() for k, v in self.window_intensity.items()
        }

        window_background = {
            k: v.tolist() for k, v in self.window_background.items()
        }

        file_weights = {
            k: float(v) for k, v in self.file_weights.items()
        }

        metadata = {
            "window_intensity": window_intensity,
            "window_background": window_background,
            "file_weights": file_weights,

            # config for reproducibility
            "stride_sec": float(self.window_stride),
            "sample_rate": int(self.sample_rate),
            "clip_samples": int(self.clip_samples),
            "sampling_alpha": float(self.sampling_alpha),
        }

        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f)

        print(
            f"[METADATA] files={len(self.window_intensity)}, "
            f"alpha={self.sampling_alpha}"
        )

    def _build_labels(self, audio_file, clip_start, clip_end):
        """
        Compute all label heads. 
        The 'Ghost Bernoulli' (fn_floor) ensures all distributions are 
        mathematically valid and prevents EMD collapse.
        """
        # 1. Load cached events (should return empty arrays if no events exist)
        starts, ends, bandwidths = get_event_cache(self.annotation_dir, audio_file)
        
        fn_floor = 1e-5 
        n_slices = 16

        # 2. Compute Overlaps and Confidence
        # If starts/ends are empty, window_overlap will be empty [].
        window_overlap = compute_window_overlap(starts, ends, clip_start, clip_end)
        
        if len(window_overlap) > 0:
            durations = ends - starts
            conf = compute_event_confidence(
                durations, bandwidths, self.duration_stats,
                self.bandwidth_stats, self.logistic_params,
            )
            window_p = window_overlap * conf
        else:
            # No frogs in this window (or no frogs in the whole file)
            window_p = np.array([], dtype=np.float32)

        # 3. Inject the "Ghost Bernoulli"
        # This is our single source of truth for the False Negative Floor.
        window_p_with_ghost = np.append(window_p, fn_floor)

        # 4. Generate Head Outputs
        # count_probs now uses the ghost in the probability list
        count_probs = soft_count_distribution(window_p_with_ghost, epsilon=0)
        binary = binary_clip_probability(window_p_with_ghost)

        # 5. Slice Head
        # If starts is empty, slice_mat is [0, 16]. prod(axis=0) becomes 1.0. Result is 0.0.
        slice_mat = compute_slice_overlap_matrix(
            starts, ends, clip_start, clip_end, n_slices=n_slices
        )
        
        # Handle confidence weighting safely for empty arrays
        if len(window_overlap) > 0:
            slice_p = slice_mat * conf[:, None]
            slice_out = (1.0 - np.prod(1.0 - slice_p, axis=0)).astype(np.float32)
        else:
            slice_out = np.zeros(n_slices, dtype=np.float32)

        return {
            "binary": np.float32(binary),
            "count_probs": count_probs,
            "slice": slice_out, 
        }

    def __len__(self):
        return 10000 if self.train else len(self.val_index)

    def _build_global_file_marginals(self):
        """
        Computes P(file) ∝ sum_k w_{i,k}
        where w_{i,k} = (1 - α) * P_fg + α * P_bg
        """
        self.file_weights = {}

        alpha = self.sampling_alpha

        for audio_file in self.audio_files:
            p_fg = self.window_intensity[audio_file]
            p_bg = self.window_background[audio_file]

            w = (1.0 - alpha) * p_fg + alpha * p_bg

            self.file_weights[audio_file] = w.sum()

    def _build_window_intensity(self):
            stride_sec = self.window_stride
            self.window_intensity = {}
            self.window_background = {}

            for audio_file in self.audio_files:
                path = os.path.join(self.audio_dir, audio_file)

                try:
                    info = sf.info(path)
                    total_samples = int(info.frames)
                except:
                    continue

                clip_len = self.clip_samples
                max_start = max(0, total_samples - clip_len)

                K = max(1, int(max_start / (stride_sec * self.sample_rate)))

                starts, ends, bandwidths = get_event_cache(
                    self.annotation_dir, audio_file
                )

                # --------------------------------------------------
                # empty file case
                # --------------------------------------------------
                if len(starts) == 0:
                    p_fg = np.zeros(K, dtype=np.float32)
                    p_bg = np.ones(K, dtype=np.float32) / K  # uniform background

                    self.window_intensity[audio_file] = p_fg
                    self.window_background[audio_file] = p_bg
                    continue

                starts_s = (starts * self.sample_rate).astype(int)
                ends_s = (ends * self.sample_rate).astype(int)

                conf = np.ones(len(starts), dtype=np.float32)
                if self.use_continuous_confidence:
                    durations = ends - starts
                    conf = compute_event_confidence(
                        durations,
                        bandwidths,
                        self.duration_stats,
                        self.bandwidth_stats,
                        self.logistic_params,
                    )

                # --------------------------------------------------
                # 1. λ_raw(t): windowed event mass (unnormalized)
                # --------------------------------------------------
                lam_raw = np.zeros(K, dtype=np.float32)
                for k in range(K):
                    t0 = k * stride_sec * self.sample_rate
                    t1 = t0 + clip_len
                    overlap = np.logical_and(starts_s < t1, ends_s > t0)
                    lam_raw[k] = np.sum(conf[overlap])

                # --------------------------------------------------
                # 2. Foreground Distribution (P_fg)
                # Tempered (sqrt) to prevent chorus domination, then normalized.
                # --------------------------------------------------
                p_fg = np.sqrt(lam_raw)
                fg_sum = p_fg.sum()
                if fg_sum > 0:
                    p_fg = p_fg / fg_sum
                else:
                    p_fg = np.zeros(K, dtype=np.float32)

                # --------------------------------------------------
                # 3. Safe Background Distribution (P_bg)
                # Strict mask (S=2.0) to exclude confident and ambiguous calls.
                # --------------------------------------------------
                suppression_factor = 2.0
                m_bg = 1.0 - (lam_raw * suppression_factor)
                m_bg = np.clip(m_bg, 0.0, 1.0)
                
                bg_sum = m_bg.sum()
                if bg_sum < 1e-6:
                    # Fallback to uniform if file is 100% full of frogs
                    p_bg = np.ones(K, dtype=np.float32) / K
                else:
                    p_bg = m_bg / bg_sum

                self.window_intensity[audio_file] = p_fg.astype(np.float32)
                self.window_background[audio_file] = p_bg.astype(np.float32)

    def _sample_entry_train(self):
        alpha = self.sampling_alpha
        stride = self.window_stride
        sample_rate = self.sample_rate

        # --------------------------------------------------
        # Step 1: sample file from true marginal
        # P(i) ∝ Σ_k w_{i,k}
        # --------------------------------------------------
        audio_file = random.choices(
            self.audio_files,
            weights=[self.file_weights[f] for f in self.audio_files]
        )[0]

        p_fg = self.window_intensity[audio_file]
        p_bg = self.window_background[audio_file]

        p_fg = np.asarray(p_fg, dtype=np.float32)
        p_bg = np.asarray(p_bg, dtype=np.float32)

        # --------------------------------------------------
        # Step 2: conditional distribution (Convex Combination)
        # P(k|i) ∝ (1 - α) P_fg + α P_bg
        # --------------------------------------------------
        weights = (1.0 - alpha) * p_fg + alpha * p_bg
        
        weights = np.clip(weights, 0.0, None)
        weights = weights + 1e-6
        weights = weights / weights.sum()

        k = np.random.choice(len(weights), p=weights)

        # --------------------------------------------------
        # Step 3: continuous offset within stride
        # --------------------------------------------------
        start_sec = k * stride + random.uniform(0, stride)
        start = int(start_sec * sample_rate)

        return {
            "audio_file": audio_file,
            "start": start
        }

    def __getitem__(self, idx):
            """
            Fetch a single dataset item.

            Returns:
                - Perch embedding (computed inline)
                - Multi-head label dict
                - audio filename
                - start sample index
            """

            if self.train:
                entry = self._sample_entry_train()
                audio_file = entry["audio_file"]
                start_sample = int(entry["start"])
            else:
                audio_file, start_sec = self.val_index[idx]
                start_sample = int(start_sec * self.sample_rate)

            # 1. Load ONLY the targeted slice from disk
            path = os.path.join(self.audio_dir, audio_file)
            audio, _ = load_audio(
                path,
                target_sr=self.sample_rate,
                start_sample=start_sample,
                num_samples=self.clip_samples
            )

            # 2. `audio` is already the exact slice requested.
            # We just need to zero-pad it if we hit the end of the file (EOF).
            clip = np.zeros(self.clip_samples, dtype=np.float32)
            valid_frames = len(audio)

            if valid_frames > 0:
                clip[:valid_frames] = audio[:]

            # 3. Convert waveform to Perch embedding input format
            clip_perch = resample_array(
                clip,
                orig_sr=self.sample_rate,
                target_sr=self.perch_sr,
            )

            clip_perch = clip_perch[:self.perch_samples]

            if len(clip_perch) < self.perch_samples:
                clip_perch = np.pad(
                    clip_perch,
                    (0, self.perch_samples - len(clip_perch)),
                )

            # 4. Build unified label representation
            clip_start = start_sample / self.sample_rate
            clip_end = clip_start + self.clip_samples / self.sample_rate

            labels = self._build_labels(audio_file, clip_start, clip_end)

            # 5. Compute Perch embedding inline
            spatial_emb = self.perch.get_spatial_embedding(
                clip_perch
            ).astype(np.float32)

            return spatial_emb, labels, audio_file, start_sample

    def _load_or_create_split(self):
        """
        Load existing split if present; otherwise create deterministic split.

        Splitting strategy:
        - group files by recording block (date-hour key)
        - shuffle groups deterministically using seed
        - allocate into train/val/test partitions
        """
        split_path = self._get_split_path()

        if os.path.exists(split_path):
            with open(split_path, "r") as f:
                split = json.load(f)
            return split[self.split_type]

        # Grouping prevents leakage across temporally adjacent recordings
        groups = self._group_files_by_date_hour()

        keys = list(groups.keys())

        # Deterministic shuffle for reproducibility
        rng = random.Random(self.random_seed)
        rng.shuffle(keys)

        n = len(keys)
        n_test = int(n * self.test_split)
        n_val = int(n * self.val_split)

        split_map = {
            "train": keys[: n - n_test - n_val],
            "val": keys[n - n_test - n_val : n - n_test],
            "test": keys[n - n_test :],
        }

        # Expand grouped keys back into filenames
        full_split = {
            k: sorted([f for key in v for f in groups[key]])
            for k, v in split_map.items()
        }

        # Persist split for reproducibility across runs
        with open(split_path, "w") as f:
            json.dump(full_split, f, indent=2)

        return full_split[self.split_type]

