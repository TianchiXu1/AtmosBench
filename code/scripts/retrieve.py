#!/usr/bin/env python3
"""Retrieve intrinsic BC from one native AtmosBench ATB curtain."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from atmosbench_evaluation import encode, valid_atb
from atmosbench_models import DEFAULT_MODEL_ID, load_fresh_model, load_model
from atb_encoding import log_bounds
from dataset_protocol import load_top_down_array


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atb", type=Path, required=True, help="native bottom-up 200×200 NPY curtain")
    parser.add_argument("--band", choices=("355", "532"), required=True)
    checkpoints = parser.add_mutually_exclusive_group(required=True)
    checkpoints.add_argument("--checkpoint", type=Path, help="selected paper PhysicalSDC checkpoint")
    checkpoints.add_argument("--fresh-checkpoint", type=Path, help="best.pt produced by scripts/train.py")
    parser.add_argument("--model", help=f"registered model ID for --checkpoint (default: {DEFAULT_MODEL_ID})")
    parser.add_argument("--output", type=Path, required=True, help="bottom-up BC NPY in m^-1 sr^-1")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.fresh_checkpoint and args.model:
        parser.error("--model is only used with --checkpoint.")
    stats = json.loads((ROOT / "configs/train_normalization.json").read_text())
    raw = load_top_down_array(args.atb)
    device = torch.device(args.device)
    if args.fresh_checkpoint:
        model, _, row = load_fresh_model(args.fresh_checkpoint, ROOT / "third_party", device)
    else:
        models = json.loads((ROOT / "configs/evaluation.json").read_text())["models"]
        model_id = args.model or DEFAULT_MODEL_ID
        if model_id not in models:
            parser.error(f"Unknown registered model ID: {model_id}")
        row = models[model_id]
        model, _ = load_model(row, args.checkpoint, ROOT / "third_party", device)
    conditioning = encode(raw, stats, args.band, row["representation"], row)
    lower, upper = log_bounds(stats, args.band, "bc")
    with torch.inference_mode():
        mean_norm = model(torch.from_numpy(conditioning[None]).to(device))["mean_norm"][0, 0].float()
        bc = torch.pow(10.0, mean_norm * (upper - lower) + lower).cpu().numpy()
    bc[~valid_atb(raw)] = np.nan
    with args.output.open("xb") as handle:
        np.save(handle, bc[::-1], allow_pickle=False)


if __name__ == "__main__":
    main()
