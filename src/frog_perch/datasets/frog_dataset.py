# datasets/frog_dataset.py
import os
import json
import random
import hashlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf

from frog_perch.utils.audio import load_audio, resample_array
from frog_perch.utils.annotations import (
    load_annotations,
    has_frog_call,
    get_frog_call_weights,
    soft_count_distribution,
    reconciled_slice_and_count_targets,
    slice_binary_confidences
)
from frog_perch.models.perch_wrapper import PerchWrapper
import frog_perch.config as config


# =========================================================
#  RangeSampler (TRAIN-TIME ONLY)
# =========================================================
class RangeSampler:
    def __init__(self, ranges):
        self.ranges = ranges
        self.cumulative_sizes = []
        total = 0
        for r in ranges:
            size = r['end'] - r['start']
            total += size
            self.cumulative_sizes.append(total)
        self.total = total

    def sample(self):
        if self.total == 0:
            raise RuntimeError("RangeSampler empty")
        idx = random.randint(0, self.total - 1)
        for i, cum_size in enumerate(self.cumulative_sizes):
            if idx < cum_size:
                r = self.ranges[i]
                prev = self.cumulative_sizes[i - 1] if i > 0 else 0
                offset = idx - prev
                return {'audio_file': r['audio_file'], 'start': r['start'] + offset}


# =========================================================
#  FrogPerchDataset
# =========================================================
class FrogPerchDataset:
    def __init__(self,
                 audio_dir=None,
                 annotation_dir=None,
                 split_type='train',  # Changed from 'train=True'
                 test_split=None,
                 val_split=0.15,      # Added explicit val split
                 pos_ratio=None,
                 random_seed=None,
                 label_mode='count',
                 val_stride_sec=None,
                 q2_confidence=0.75,
                 equalize_q2_val=False,
                 use_continuous_confidence=False,
                 confidence_params=None):

        self.audio_dir = audio_dir or config.AUDIO_DIR
        self.annotation_dir = annotation_dir or config.ANNOTATION_DIR
        self.split_type = split_type # 'train', 'val', or 'test'
        self.train = (split_type == 'train') # Compatibility for internal logic
        self.pos_ratio = pos_ratio if pos_ratio is not None else getattr(config, "POS_RATIO", 0.5)
        
        self.test_split = test_split if test_split is not None else config.TEST_SPLIT
        self.val_split = val_split 
        self.random_seed = random_seed if random_seed is not None else config.RANDOM_SEED
        self.val_stride_sec = val_stride_sec or config.VAL_STRIDE_SEC
        self.q2_confidence = q2_confidence
        self.equalize_q2_val = equalize_q2_val
        self.use_continuous_confidence = use_continuous_confidence
        self.label_mode=label_mode

        # Guarantee it's a dictionary
        if confidence_params is None:
            confidence_params = {}

        self.duration_stats  = confidence_params.get("duration_stats", None)
        self.bandwidth_stats = confidence_params.get("bandwidth_stats", None)
        self.logistic_params = confidence_params.get("logistic_params", None)

        # sampling/clip definitions
        self.sample_rate = config.DATASET_SAMPLE_RATE
        self.clip_samples = int(config.CLIP_DURATION_SECONDS * self.sample_rate)
        self.perch_sr = config.PERCH_SAMPLE_RATE
        self.perch_samples = config.PERCH_CLIP_SAMPLES

        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        # load train/test split
        self.audio_files = self._load_or_create_split()

        # TRAIN MODE: load or compute metadata
        if self.train:
            self.metadata_path = self._get_metadata_path()
            if os.path.exists(self.metadata_path):
                self._load_metadata()
            else:
                self._compute_and_save_metadata()

            self.pos_sampler = RangeSampler(self.positive_ranges)
            self.neg_sampler = RangeSampler(self.negative_ranges)

        # VAL or TEST MODE: build deterministic index
        else:
            self._build_validation_index()
            if self.equalize_q2_val and self.split_type in ['val', 'test']:
                self.q2_confidence = 1.0

        # load Perch model
        self.perch = PerchWrapper()

    # =========================================================
    #  Split handling
    # =========================================================
    def _get_split_path(self):
            # We include the seed in the filename to ensure reproducibility
            return os.path.join(
                self.audio_dir,
                f"dataset_split_seed{self.random_seed}.json"
            )

    def _load_or_create_split(self):
        """High-level orchestration for dataset splitting."""
        split_path = self._get_split_path()
        
        if os.path.exists(split_path):
            print(f"[INFO] Loading EXISTING split from {split_path}")
            with open(split_path, 'r') as f:
                split = json.load(f)
            return split[self.split_type]

        print(f"[INFO] Split file not found. GENERATING NEW split at {split_path}")
        # 1. Group files to prevent leakage
        groups = self._group_files_by_date_hour()
        
        # 2. Split keys into train/val/test
        split_indices = self._compute_split_indices(list(groups.keys()))
        
        # 3. Map grouped files back to the final split structure
        full_split = {
            name: sorted([f for k in keys for f in groups[k]])
            for name, keys in split_indices.items()
        }

        # 4. Persistence
        with open(split_path, 'w') as f:
            json.dump(full_split, f, indent=2)

        return full_split[self.split_type]

    def _group_files_by_date_hour(self):
        """Groups filenames by YYYYMMDD_HHMMSS to prevent cross-mic leakage."""
        all_files = [f for f in os.listdir(self.audio_dir) if f.endswith('.wav')]
        groups = {}
        for fname in all_files:
            parts = fname.split("__")
            if len(parts) < 2:
                key = f"ungrouped_{fname}"
            else:
                # Extracts "20241113_170000" from "P4__20241113_170000_SYNC_clip12.wav"
                sub_parts = parts[1].split("_")
                key = f"{sub_parts[0]}_{sub_parts[1]}"
            groups.setdefault(key, []).append(fname)
        return groups

    def _compute_split_indices(self, keys):
        """Shuffles keys and returns a dict of {split_name: [keys]}."""
        keys = sorted(keys)
        rng = random.Random(self.random_seed)
        rng.shuffle(keys)

        n = len(keys)
        test_count = int(n * self.test_split)
        val_count = int(n * self.val_split)
        train_count = n - test_count - val_count

        return {
            'train': keys[:train_count],
            'val':   keys[train_count : train_count + val_count],
            'test':  keys[train_count + val_count:]
        }

    def _get_metadata_path(self):
        # Crucial: unique hash per split_type to avoid index errors
        key = "_".join(sorted(self.audio_files))
        hash_key = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.audio_dir, f"clip_metadata_{self.split_type}_{hash_key}.json")

    def _load_metadata(self):
        with open(self.metadata_path, 'r') as f:
            metadata = json.load(f)

        self.positive_ranges = metadata['positive_ranges']
        self.negative_ranges = metadata['negative_ranges']
        self.audio_files = metadata['audio_files']

        pos_total = sum(r['end'] - r['start'] for r in self.positive_ranges)
        neg_total = sum(r['end'] - r['start'] for r in self.negative_ranges)
        print(f"[TRAIN] Loaded metadata: {len(self.audio_files)} files; pos {pos_total} neg {neg_total}")

    def _compute_and_save_metadata(self):
        print("Computing metadata (positive and negative ranges).")
        self.positive_ranges = []
        self.negative_ranges = []

        def process_file(audio_file):
            annotations = load_annotations(self.annotation_dir, audio_file)
            audio_path = os.path.join(self.audio_dir, audio_file)
            try:
                info = sf.info(audio_path)
                total_samples = int(info.frames)
            except:
                return [], []

            all_start = 0
            all_end = max(0, total_samples - self.clip_samples + 1)
            all_indices = set(range(all_start, all_end))

            pos_indices = set()
            # derive positive indices
            for _, row in annotations.iterrows():
                if 'white dot' not in row['Annotation']:
                    continue
                ann_start = float(row['Begin Time (s)'])
                ann_end = float(row['End Time (s)'])
                ann_duration = ann_end - ann_start
                if ann_duration <= 0:
                    continue

                min_overlap = 0.5 * ann_duration
                clip_start_min = ann_start + min_overlap - (self.clip_samples / float(self.sample_rate))
                clip_start_max = ann_end - min_overlap

                start_idx_min = max(0, int(np.floor(clip_start_min * self.sample_rate)))
                start_idx_max_raw = clip_start_max * self.sample_rate
                start_idx_max = min(int(np.floor(start_idx_max_raw)), total_samples - self.clip_samples)

                if start_idx_max >= start_idx_min:
                    pos_indices.update(range(start_idx_min, start_idx_max + 1))

            neg_indices = sorted(all_indices - pos_indices)
            pos_indices = sorted(pos_indices)

            def to_ranges(start_list):
                if not start_list:
                    return []
                ranges = []
                current_start = start_list[0]
                prev = start_list[0]
                for s in start_list[1:]:
                    if s == prev + 1:
                        prev = s
                    else:
                        ranges.append({
                            'audio_file': audio_file,
                            'start': current_start,
                            'end': prev + 1
                        })
                        current_start = s
                        prev = s
                ranges.append({'audio_file': audio_file, 'start': current_start, 'end': prev + 1})
                return ranges

            return to_ranges(pos_indices), to_ranges(neg_indices)

        with ThreadPoolExecutor(max_workers=config.METADATA_WORKERS) as ex:
            results = list(ex.map(process_file, self.audio_files))

        for pos, neg in results:
            self.positive_ranges.extend(pos)
            self.negative_ranges.extend(neg)

        metadata = {
            'positive_ranges': self.positive_ranges,
            'negative_ranges': self.negative_ranges,
            'audio_files': self.audio_files
        }
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f)

        pos_total = sum(r['end'] - r['start'] for r in self.positive_ranges)
        neg_total = sum(r['end'] - r['start'] for r in self.negative_ranges)
        print(f"[TRAIN] Computed metadata: pos {pos_total}, neg {neg_total}")

    # =========================================================
    #  VALIDATION: deterministic fixed-stride index
    # =========================================================
    def _build_validation_index(self):
        print("[VAL] Building deterministic validation windows...")
        self.val_index = []

        for audio_file in self.audio_files:
            audio_path = os.path.join(self.audio_dir, audio_file)
            try:
                info = sf.info(audio_path)
                total_samples = int(info.frames)
            except:
                continue

            total_seconds = total_samples / float(self.sample_rate)
            clip_seconds = self.clip_samples / float(self.sample_rate)

            # deterministic evenly spaced windows
            starts_sec = np.arange(
                0,
                max(0, total_seconds - clip_seconds),
                self.val_stride_sec
            )

            for s in starts_sec:
                self.val_index.append((audio_file, float(s)))

        print(f"[VAL] Total validation windows: {len(self.val_index)}")

    # =========================================================
    #  Dataset length
    # =========================================================
    def __len__(self):
        if self.train:
            return 1000  # infinite-like
        else:
            return len(self.val_index)

    # =========================================================
    #  TRAINING sampling (random balanced)
    # =========================================================
    def _sample_entry_train(self):
        if self.pos_sampler.total == 0 and self.neg_sampler.total == 0:
            raise RuntimeError("No clips available for sampling.")

        use_pos = (random.random() < self.pos_ratio) and (self.pos_sampler.total > 0)
        entry = self.pos_sampler.sample() if use_pos else self.neg_sampler.sample()
        return entry

    # =========================================================
    #  __getitem__
    # =========================================================
    def __getitem__(self, idx):
        # -----------------------------
        # TRAINING: random sampling
        # -----------------------------
        if self.train:
            entry = self._sample_entry_train()
            audio_file = entry['audio_file']
            start_sample = int(entry['start'])

        # -----------------------------
        # VALIDATION: fixed stride windows
        # -----------------------------
        else:
            audio_file, start_sec = self.val_index[idx]
            start_sample = int(round(start_sec * self.sample_rate))

        # load audio
        audio_path = os.path.join(self.audio_dir, audio_file)
        data, sr = load_audio(audio_path, target_sr=self.sample_rate)

        # deterministic slicing (no randomness in val)
        end = start_sample + self.clip_samples
        if end <= len(data):
            clip = data[start_sample:end]
        else:
            clip = np.zeros(self.clip_samples, dtype=np.float32)
            valid = max(0, len(data) - start_sample)
            if valid > 0:
                clip[:valid] = data[start_sample:start_sample + valid]

        # convert to Perch sample rate
        clip_perch = resample_array(clip, orig_sr=self.sample_rate, target_sr=self.perch_sr)

        # Training: no random cropping needed; just pad if slightly short from resampling
        if len(clip_perch) < self.perch_samples:
            clip_perch = np.pad(clip_perch, (0, self.perch_samples - len(clip_perch)))

        # Sanity crop if resampler produces +1 sample due to rounding
        clip_perch = clip_perch[:self.perch_samples]

        # compute label
        clip_start_time = start_sample / float(self.sample_rate)
        clip_end_time = clip_start_time + (self.clip_samples / float(self.sample_rate))
        annotations = load_annotations(self.annotation_dir, audio_file)

        if self.label_mode == 'binary':
            label = has_frog_call(
                annotations,
                clip_start_time,
                clip_end_time,
                q2_confidence=self.q2_confidence,
                use_continuous_confidence=self.use_continuous_confidence,
                duration_stats=self.duration_stats,
                bandwidth_stats=self.bandwidth_stats,
                logistic_params=self.logistic_params
            )
            # label = float(label_value)

        elif self.label_mode == 'count':
            weights = get_frog_call_weights(
                annotations,
                clip_start_time,
                clip_end_time,
                q2_confidence=self.q2_confidence,
                use_continuous_confidence=self.use_continuous_confidence,
                duration_stats=self.duration_stats,
                bandwidth_stats=self.bandwidth_stats,
                logistic_params=self.logistic_params
            )
            label = soft_count_distribution(weights)

        ### Fix
        elif self.label_mode == 'slice':
            # Single call to the reconciled utility
            # slice_probs, count_dist = reconciled_slice_and_count_targets(
            #     annotations,
            #     clip_start_time,
            #     clip_end_time,
            #     n_slices=16,
            #     max_count_bin=16,
            #     q2_confidence=self.q2_confidence,
            #     use_continuous_confidence=self.use_continuous_confidence,
            #     duration_stats=self.duration_stats,
            #     bandwidth_stats=self.bandwidth_stats,
            #     logistic_params=self.logistic_params
            # )

            slice_probs= slice_binary_confidences(
                annotations,
                clip_start_time,
                clip_end_time,
                n_slices=16,
                q2_confidence=self.q2_confidence,
                use_continuous_confidence=self.use_continuous_confidence,
                duration_stats=self.duration_stats,
                bandwidth_stats=self.bandwidth_stats,
                logistic_params=self.logistic_params
            )
                        
            # Concatenate to the final 33-length vector
            # label = np.concatenate([slice_probs, count_dist], axis=0).astype(np.float32)

            label = slice_probs

        else:
            raise ValueError(f"Unknown label_mode: {self.label_mode}")

        # get Perch embedding
        spatial_emb = self.perch.get_spatial_embedding(clip_perch).astype(np.float32)

        return spatial_emb, label, audio_file, start_sample
