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

class RangeSampler:
    """
    Uniform sampler over contiguous integer frame ranges.

    Each range represents a contiguous region of valid clip start indices.
    Sampling is performed by selecting a global index uniformly over the
    concatenated ranges and mapping it back to a specific file + offset.
    """
    def __init__(self, ranges):
        self.ranges = ranges

        # Cumulative sum of range lengths for O(n) interval lookup
        self.cumulative = []
        total = 0

        for r in ranges:
            total += (r["end"] - r["start"])
            self.cumulative.append(total)

        self.total = total

    def sample(self):
        """
        Draw a single uniformly distributed (file, start_index) pair.
        """
        if self.total == 0:
            raise RuntimeError("RangeSampler has no valid ranges.")

        # Sample a global index across all ranges
        idx = random.randint(0, self.total - 1)

        # Resolve which range this index falls into
        for i, cum in enumerate(self.cumulative):
            if idx < cum:
                r = self.ranges[i]

                # Convert global index into local offset within range
                prev = self.cumulative[i - 1] if i > 0 else 0
                offset = idx - prev

                return {
                    "audio_file": r["audio_file"],
                    "start": r["start"] + offset,
                }


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
        audio_dir: str,           # Now required
        annotation_dir: str,      # Now required
        split_type: str = "train",
        test_split: float = 0.15,
        val_split: float = 0.1,
        pos_ratio: float = 0.5,
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

        self.pos_ratio = pos_ratio
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

            self.pos_sampler = RangeSampler(self.positive_ranges)
            self.neg_sampler = RangeSampler(self.negative_ranges)
        else:
            self._build_validation_index()

        # Perch embedding model
        self.perch = PerchWrapper()

        # Caches
        self._annotation_cache = {}

    def _get_split_path(self):
        """Return deterministic path for cached dataset split JSON."""
        return os.path.join(
            self.audio_dir,
            f"dataset_split_seed{self.random_seed}.json"
        )

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

        The hash ensures that different file subsets generate
        independent metadata files.
        """
        key = "_".join(sorted(self.audio_files))
        h = hashlib.md5(key.encode()).hexdigest()

        return os.path.join(
            self.audio_dir,
            f"clip_metadata_{self.split_type}_{h}.json"
        )

    def _load_metadata(self):
        """
        Load precomputed positive/negative sampling ranges.
        """
        with open(self.metadata_path, "r") as f:
            meta = json.load(f)

        self.positive_ranges = meta["positive_ranges"]
        self.negative_ranges = meta["negative_ranges"]

    def _compute_and_save_metadata(self):
        """
        Placeholder for existing metadata computation logic.

        This step builds:
        - positive frame ranges
        - negative frame ranges
        based on annotation overlap with audio clips.
        """
        raise NotImplementedError("Keep existing implementation unchanged.")

    def _build_labels(self, audio_file, clip_start, clip_end):
        """
        Compute all label heads from a shared event representation.

        Pipeline:
        1. Load cached annotation events for file
        2. Compute event confidence scores
        3. Compute window overlaps
        4. Project into:
            - binary probability
            - count distribution
            - slice-wise temporal distribution
        """

        # Load cached event representation (avoids repeated I/O/parsing)
        starts, ends, bandwidths = get_event_cache(
            self.annotation_dir,
            audio_file
        )

        if len(starts) == 0:
            n_slices = 16
            return {
                "binary": np.float32(0.0),
                "count_probs": np.zeros(17, dtype=np.float32),
                "slice": np.zeros(n_slices, dtype=np.float32),
            }

        durations = ends - starts

        # Event-level confidence weighting (shared across all heads)
        conf = compute_event_confidence(
            durations,
            bandwidths,
            self.duration_stats,
            self.bandwidth_stats,
            self.logistic_params,
        )

        # Compute overlap between events and current clip window
        window_overlap = compute_window_overlap(
            starts, ends, clip_start, clip_end
        )

        if len(window_overlap) == 0:
            n_slices = 16
            return {
                "binary": np.float32(0.0),
                "count_probs": np.zeros(17, dtype=np.float32),
                "slice": np.zeros(n_slices, dtype=np.float32),
            }

        # Convert overlaps into per-event probabilities
        window_p = window_overlap * conf

        # Clip-level probability (at least one event)
        binary = binary_clip_probability(window_p)

        # Count distribution (Poisson-binomial aggregation)
        count_probs = soft_count_distribution(window_p)

        # Slice-level temporal structure
        slice_mat = compute_slice_overlap_matrix(
            starts,
            ends,
            clip_start,
            clip_end,
            n_slices=16,
        )

        # Apply confidence weighting per event per slice
        slice_p = slice_mat * conf[:, None]

        # Aggregate slice probabilities (independent event assumption)
        slice_out = (1.0 - np.prod(1.0 - slice_p, axis=0)).astype(np.float32)

        return {
            "binary": np.float32(binary),
            "count_probs": count_probs,
            "slice": slice_out,
        }

    def __len__(self):
        return 10000 if self.train else len(self.val_index)

    def _sample_entry_train(self):
        """
        Sample a training entry using balanced positive/negative sampling.
        """
        use_pos = random.random() < self.pos_ratio
        sampler = self.pos_sampler if use_pos else self.neg_sampler
        return sampler.sample()

    def __getitem__(self, idx):
        """
        Fetch a single dataset item.

        Returns:
            - Perch embedding
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

        # Load full audio file
        path = os.path.join(self.audio_dir, audio_file)
        audio, _ = load_audio(path, target_sr=self.sample_rate)

        # Extract fixed-length clip
        clip = np.zeros(self.clip_samples, dtype=np.float32)
        valid = max(0, min(len(audio) - start_sample, self.clip_samples))

        if valid > 0:
            clip[:valid] = audio[start_sample:start_sample + valid]

        # Convert waveform to Perch embedding input format
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

        # Convert sample index into time window
        clip_start = start_sample / self.sample_rate
        clip_end = clip_start + self.clip_samples / self.sample_rate

        # Build unified label representation
        labels = self._build_labels(audio_file, clip_start, clip_end)

        # Compute Perch embedding
        spatial_emb = self.perch.get_spatial_embedding(
            clip_perch
        ).astype(np.float32)

        return spatial_emb, labels, audio_file, start_sample