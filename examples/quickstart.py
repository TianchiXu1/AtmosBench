#!/usr/bin/env python3
"""Run CPU inference with a released PhysicalSDC checkpoint on eight sample pairs."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "adair_physicalsdc_run1"
CHECKPOINT = f"{MODEL_ID}.pt"
CHECKPOINT_URL = f"https://github.com/TianchiXu1/AtmosBench/releases/download/v1.1.0/{CHECKPOINT}"
CHECKPOINT_SHA256 = "caf2bfea68efceec37580c2c9bb8af110226b48446f2d048ead6923f8b7a3e07"
EXPECTED_LOG_MAE = 0.01835656


def verify_checkpoint(path: Path) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint:
        for block in iter(lambda: checkpoint.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != CHECKPOINT_SHA256:
        raise ValueError(f"SHA-256 mismatch for {path}. Use the published {CHECKPOINT}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="reuse a local copy of the released checkpoint")
    args = parser.parse_args()
    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    output = output_dir / "quickstart.json"
    try:
        with tempfile.TemporaryDirectory(prefix="quickstart-", dir=output_dir) as temporary_dir:
            scratch = Path(temporary_dir)
            checkpoint = args.checkpoint or output_dir / CHECKPOINT
            if args.checkpoint is None and not checkpoint.exists():
                print(f"Downloading {CHECKPOINT} (116 MB)", flush=True)
                download = scratch / CHECKPOINT
                with urllib.request.urlopen(CHECKPOINT_URL, timeout=30) as response, download.open("wb") as handle:
                    shutil.copyfileobj(response, handle)
                verify_checkpoint(download)
                download.replace(checkpoint)
            else:
                verify_checkpoint(checkpoint)
            if checkpoint.name != CHECKPOINT:
                shutil.copyfile(checkpoint, scratch / CHECKPOINT)
                checkpoint = scratch / CHECKPOINT

            arrays = scratch / "arrays"
            result_path = scratch / "quickstart.json"
            with zipfile.ZipFile(ROOT / "examples" / "benchmark_sample.zip") as sample:
                sample.extractall(arrays)
            env = dict(os.environ)
            env.setdefault("OMP_NUM_THREADS", "2")
            subprocess.run([sys.executable, str(ROOT / "code/scripts/evaluate.py"), "--split", "test",
                            "--data-root", str(arrays), "--checkpoints", str(checkpoint.parent),
                            "--models", MODEL_ID, "--device", "cpu", "--smoke", "--output", str(result_path)],
                           check=True, env=env)
            result = json.loads(result_path.read_text())["results"]["clean"][MODEL_ID]
            log_mae = result["cell_macro"]["log10_mae"]
            if result["items"] != 8 or not math.isclose(log_mae, EXPECTED_LOG_MAE, rel_tol=0, abs_tol=1e-5):
                raise ValueError(f"Unexpected sample result: {result['items']} pairs, log-MAE {log_mae:.8f}.")
            result_path.replace(output)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Quickstart failed: {error}\n")
    print(f"8 pairs, log-MAE {log_mae:.5f} dex. Results: outputs/quickstart.json")


if __name__ == "__main__":
    main()
