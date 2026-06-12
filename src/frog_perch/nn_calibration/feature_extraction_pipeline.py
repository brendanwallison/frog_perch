import os
import numpy as np
import pandas as pd
from pathlib import Path

from frog_perch.utils.audio import load_audio, resample_array
from frog_perch.nn_models.perch_wrapper import PerchWrapper
from frog_perch.nn_models.model_utils import load_custom_model
from frog_perch.nn_calibration.feature_extraction import build_feature_record

def process_file(path_audio: Path, model, perch: PerchWrapper, out_csv: Path, batch_size: int, step_seconds: float, config_dict: dict) -> None:
    """Processes field recordings by sliding a temporal extraction window over raw audio."""
    sample_rate = config_dict.get("DATASET_SAMPLE_RATE", 32000)
    clip_seconds = config_dict.get("CLIP_DURATION_SECONDS", 5.0)
    clip_samples = int(sample_rate * clip_seconds)
    perch_sr = config_dict.get("PERCH_SAMPLE_RATE", 32000)
    perch_samples = config_dict.get("PERCH_CLIP_SAMPLES", 160000)

    audio, _ = load_audio(str(path_audio), target_sr=sample_rate)
    total_seconds = len(audio) / sample_rate
    max_start = max(0.0, total_seconds - clip_seconds)

    starts = np.arange(0.0, max_start + 1e-8, step_seconds)
    rows = []
    emb_batch, clip_batch, time_batch = [], [], []

    def _flush_batch():
        X = np.stack(emb_batch)
        preds_dict = model.predict(X, verbose=0)
        
        for idx, st in enumerate(time_batch):
            # Extract single slice dictionary predictions out of unified batch tracking
            single_pred = {k: v[idx:idx+1] for k, v in preds_dict.items()}
            res = build_feature_record(clip_batch[idx], sample_rate, single_pred)
            res["time_sec"] = st
            rows.append(res)

    for st in starts:
        st_samp = int(round(st * sample_rate))
        clip = np.zeros(clip_samples, dtype=np.float32)
        available = max(0, len(audio) - st_samp)
        if available > 0:
            take = min(available, clip_samples)
            clip[:take] = audio[st_samp : st_samp + take]

        # Generate Perch Embeddings
        clip_p = resample_array(clip, sample_rate, perch_sr)
        if len(clip_p) < perch_samples:
            clip_p = np.pad(clip_p, (0, perch_samples - len(clip_p)))
        embedding = perch.get_spatial_embedding(clip_p[:perch_samples]).astype(np.float32)

        emb_batch.append(embedding)
        clip_batch.append(clip)
        time_batch.append(float(st))

        if len(emb_batch) >= batch_size:
            _flush_batch()
            emb_batch, clip_batch, time_batch = [], [], []

    if emb_batch:
        _flush_batch()

    pd.DataFrame(rows).to_csv(out_csv, index=False)

def process_directory(config_dict: dict, input_dir: str | Path, output_dir: str | Path, ckpt_filename: str = "best.keras", batch_size: int = 32, step_seconds: float = 5.0, exts: list[str] = None, overwrite: bool = False) -> None:
    """Main external entry point for bulk processing field scripts."""
    if exts is None: exts = ["wav", "flac", "mp3"]
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = os.path.join(config_dict.get("CHECKPOINT_DIR", ""), ckpt_filename)
    model = load_custom_model(ckpt_path)
    perch = PerchWrapper()
    _ = perch.get_spatial_embedding(np.zeros(160000, dtype=np.float32)) # Warmup

    for af in sorted(input_dir.iterdir()):
        if af.is_file() and af.suffix.lower().lstrip(".") in exts:
            out_csv = output_dir / f"{af.stem}.csv"
            if out_csv.exists() and not overwrite: continue
            try:
                process_file(af, model, perch, out_csv, batch_size, step_seconds, config_dict)
            except Exception as e:
                print(f"[ERROR] Failed processing {af.name}: {e}")