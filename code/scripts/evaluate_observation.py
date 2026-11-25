#!/usr/bin/env python3
"""Evaluate a selected checkpoint on one registered observation curtain."""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from atmosbench_evaluation import write_new_json
from atmosbench_models import DEFAULT_MODEL_ID, load_model
from atmosbench_observations import infer_beta, score_case, PAD_MULTIPLE, TARGETS


def json_values(value):
    if isinstance(value, dict):
        return {key: json_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_values(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return None
    return value.item() if isinstance(value, np.generic) else value


def main():
    models = json.loads((ROOT / "configs/evaluation.json").read_text())["models"]
    observation_models = (DEFAULT_MODEL_ID, "adair_physicalsdc_run2", "adair_physicalsdc_run3",
                          "adair_rawlog_run1", "adair_clip_run1", "adair_null_run1", "adair_asinh_run1")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=TARGETS, required=True)
    parser.add_argument("--registered", type=Path, required=True, help="top-down registered NPZ")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID,
                        choices=observation_models)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tile-core-width", type=int, default=512)
    parser.add_argument("--tile-halo", type=int, default=64)
    args = parser.parse_args()
    if args.tile_core_width < PAD_MULTIPLE or args.tile_core_width % PAD_MULTIPLE:
        parser.error(f"Tile core width must be a positive multiple of {PAD_MULTIPLE}.")
    if args.tile_halo < 0 or args.tile_halo % PAD_MULTIPLE:
        parser.error(f"Tile halo must be a nonnegative multiple of {PAD_MULTIPLE}.")
    if not args.registered.is_file():
        parser.error(f"Registered NPZ does not exist: {args.registered}")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Specify --device cpu for a CPU run.")
    row = models[args.model]
    stats = json.loads((ROOT / "configs/train_normalization.json").read_text())
    with np.load(args.registered, allow_pickle=False) as stored:
        case = {key: stored[key] for key in stored.files}
    atb_key = "atl" if args.source == "cpr" else "atb_m"
    if atb_key not in case:
        parser.error(f"Registered NPZ must contain {atb_key!r} for --source {args.source}.")
    atb = np.asarray(case[atb_key], dtype=np.float32)
    if atb.ndim != 2 or not atb.size:
        parser.error(f"Registered ATB curtain must have a nonempty two-dimensional shape, got {atb.shape}.")
    if args.source == "hsrl":
        required = {"learned_common_signed_mask", "reference_beta_m", "altitude_asl_m", "ground_altitude_asl_m"}
        missing = required.difference(case)
        if missing:
            parser.error(f"Registered HSRL NPZ is missing arrays: {sorted(missing)}.")
    model, epoch = load_model(row, args.checkpoint, ROOT / "third_party", device)
    predicted = infer_beta(model, atb, stats, args.source, row["representation"], device,
                           args.tile_core_width, args.tile_halo)
    write_new_json(args.output, json_values({"source": args.source, "model": args.model,
                                            "selected_epoch": epoch, "targets": score_case(case, predicted, args.source)}))


if __name__ == "__main__":
    main()
