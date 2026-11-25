#!/usr/bin/env python3
"""Evaluate selected AtmosBench checkpoints on validation or held-out events."""
import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_MANIFEST = ROOT.parent / "atmosbench/split_manifest.csv"
PAIR_INVENTORY = ROOT.parent / "atmosbench/pair_inventory.csv"
sys.path.insert(0, str(ROOT / "src"))
from atmosbench_evaluation import FrozenDataset, evaluate, paired_event_stats, write_new_json
from atmosbench_models import DEFAULT_MODEL_ID, load_fresh_model, load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--data-root", type=Path, required=True, help="AtmosBench arrays directory")
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    checkpoints = parser.add_mutually_exclusive_group(required=True)
    checkpoints.add_argument("--checkpoints", type=Path, help="directory of paper checkpoints")
    checkpoints.add_argument("--fresh-checkpoint", type=Path, help="best.pt produced by scripts/train.py")
    parser.add_argument("--models", nargs="+", metavar="ID", help=f"registered model IDs, default: {DEFAULT_MODEL_ID}")
    parser.add_argument("--conditions", nargs="+", default=["clean"], metavar="NAME")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="one pair per event and wavelength")
    args = parser.parse_args()
    if args.fresh_checkpoint and args.models:
        parser.error("Use --models with --checkpoints.")
    model_names = args.models or [DEFAULT_MODEL_ID]
    config = json.loads((ROOT / "configs/evaluation.json").read_text())
    conditions = config["conditions"]
    known = set(config["models"])
    unknown_models = set(model_names) - known
    unknown_conditions = set(args.conditions) - set(conditions)
    if unknown_models or unknown_conditions:
        parser.error(f"Unknown model IDs {sorted(unknown_models)} or conditions {sorted(unknown_conditions)}.")
    if args.batch_size < 1 or args.workers < 0:
        parser.error("Batch size must be positive and worker count must be nonnegative.")
    if len(set(model_names)) != len(model_names) or len(set(args.conditions)) != len(args.conditions):
        parser.error("Model IDs and conditions must each be unique.")
    for path in (args.data_root,) if args.fresh_checkpoint else (args.data_root, args.checkpoints):
        if not path.is_dir():
            parser.error(f"Required input directory does not exist: {path}")
    if not args.split_manifest.is_file():
        parser.error(f"Split manifest does not exist: {args.split_manifest}")
    if not args.smoke and not PAIR_INVENTORY.is_file():
        parser.error(f"Frozen pair inventory is missing: {PAIR_INVENTORY}")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")

    config.update(split=args.split, data_root=str(args.data_root), split_manifest=str(args.split_manifest),
                  stats_json=str(ROOT / "configs/train_normalization.json"),
                  events=config["splits"][args.split], pair_inventory=str(PAIR_INVENTORY))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Specify --device cpu for a CPU run.")
    result = {"split": args.split, "kind": "smoke" if args.smoke else "inference",
              "device": str(device), "batch_size": args.batch_size, "results": {},
              "paired_event_comparisons": {}, "inapplicable_model_conditions": {}}
    third_party = ROOT / "third_party"
    if args.fresh_checkpoint:
        if not args.fresh_checkpoint.is_file():
            parser.error(f"Fresh checkpoint does not exist: {args.fresh_checkpoint}")
        model, epoch, row = load_fresh_model(args.fresh_checkpoint, third_party, device)
        if row["smoke"] and not args.smoke:
            parser.error("A smoke-training checkpoint requires --smoke evaluation.")
        roster = {"fresh_checkpoint": row}
    else:
        roster = {name: config["models"][name] for name in model_names}
    if args.split == "validation" and set(args.conditions) - {"clean"}:
        if set(args.conditions) - {"clean", "additive_1"} or any(
                row.get("training_condition") != "additive_1" for row in roster.values()):
            parser.error("Use --split test for stress conditions. On validation, additive_1 is available for noise-trained models.")
    for name, row in roster.items():
        active_conditions = [key for key in args.conditions
                             if name in conditions[key].get("models", roster)]
        if not active_conditions:
            parser.error(f"No requested condition is available for model {name}.")
        skipped = [key for key in args.conditions if key not in active_conditions]
        if skipped:
            result["inapplicable_model_conditions"][name] = skipped
        if not args.fresh_checkpoint:
            checkpoint = args.checkpoints / row["file"]
            model, epoch = load_model(row, checkpoint, third_party, device)
        for key in active_conditions:
            dataset = FrozenDataset(config, row, conditions[key], smoke=args.smoke)
            scores = evaluate(model, dataset, device, batch_size=args.batch_size,
                              workers=args.workers, amp=row.get("amp", False))
            result["results"].setdefault(key, {})[name] = {"selected_epoch": epoch, **scores}
        del model
    if not args.smoke:
        for condition, models in result["results"].items():
            if DEFAULT_MODEL_ID in models:
                comparisons = {
                    name: paired_event_stats(models[DEFAULT_MODEL_ID], summary)
                    for name, summary in models.items()
                    if name != DEFAULT_MODEL_ID and not roster[name].get("released", False)
                }
                if comparisons:
                    result["paired_event_comparisons"][condition] = comparisons
    write_new_json(args.output, result)


if __name__ == "__main__":
    main()
