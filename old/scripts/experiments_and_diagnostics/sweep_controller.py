#!/usr/bin/env python3
"""
Hardcoded stable controller: launches worker by absolute path, waits for it,
ensures result JSON exists, waits for GPU idle, writes summary.csv.
Run from repo root: python experiments/sweep_controller.py
"""
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import shutil

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "experiments" / "logistic_sweep_results_2"
WORKER_PATH = REPO_ROOT / "experiments" / "sweep_worker.py"
PY = sys.executable

x0_values = [-5, -3]
k_values  = [0.5, 1.0]
grid = list(itertools.product(x0_values, k_values))

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def wait_for_gpu_idle(timeout=30.0, poll=0.25):
    if shutil.which("nvidia-smi") is None:
        time.sleep(0.5)
        return
    start = time.time()
    while True:
        try:
            out = subprocess.check_output(
                "nvidia-smi --query-compute-apps=pid --format=csv,noheader",
                shell=True
            ).decode().strip()
        except Exception:
            time.sleep(0.5)
            return
        if out == "":
            time.sleep(0.25)
            return
        if (time.time() - start) > timeout:
            return
        time.sleep(poll)


def main():
    print(f"Launching sweep {len(grid)} configs → {RESULTS_DIR}")
    for x0, k in grid:
        tag = f"x0={x0}_k={k}"
        out_json = RESULTS_DIR / f"{tag}.json"

        if out_json.exists():
            print(f"[SKIP] {tag}")
            continue

        cmd = [PY, str(WORKER_PATH), str(x0), str(k), str(out_json)]
        print(f"[RUN] {tag}")
        try:
            proc = subprocess.run(cmd, check=False, timeout=3 * 3600)
        except subprocess.TimeoutExpired:
            print(f"[TIMEOUT] {tag}")
            proc = None

        if not out_json.exists():
            err_obj = {"x0": x0, "k": k, "error": f"worker_failed_rc_{None if proc is None else proc.returncode}"}
            tmp = str(out_json) + ".tmp"
            with open(tmp, "w") as f:
                json.dump(err_obj, f, indent=2)
                f.flush(); os.fsync(f.fileno())
            os.replace(tmp, str(out_json))
            print(f"[WARN] Wrote stub {out_json}")

        wait_for_gpu_idle()
        time.sleep(5.0)

    # build summary
    rows = []
    for p in sorted(RESULTS_DIR.glob("x0=*_*k=*.json")):
        try:
            rows.append(json.load(open(p)))
        except Exception:
            pass

    if not rows:
        print("No results found.")
        return

    fieldnames = set().union(*(r.keys() for r in rows))
    core = ["x0", "k", "best_val_loss", "seconds"]
    extras = [f for f in sorted(fieldnames) if f not in core]
    header = core + extras

    csv_path = RESULTS_DIR / "summary.csv"
    import csv
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})

    print(f"[DONE] Summary: {csv_path}")


if __name__ == "__main__":
    main()
