#!/usr/bin/env python3
"""Score a directory of saved BC predictions on a fixed AtmosBench split."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from atmosbench_evaluation import FrozenDataset, score_predictions, write_new_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--data-root", type=Path, required=True, help="AtmosBench arrays directory")
    parser.add_argument("--predictions", type=Path, required=True,
                        help="directory containing bottom-up BC arrays named <item_id>.npy")
    parser.add_argument("--output", type=Path, required=True, help="path for a new JSON score file")
    parser.add_argument("--smoke", action="store_true", help="score only the first pair per event and wavelength")
    args = parser.parse_args()
    for path in (args.data_root, args.predictions):
        if not path.is_dir():
            parser.error(f"Required directory does not exist: {path}")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")

    config = json.loads((ROOT / "configs/evaluation.json").read_text())
    config.update(split=args.split, data_root=str(args.data_root),
                  stats_json=str(ROOT / "configs/train_normalization.json"),
                  split_manifest=str(ROOT.parent / "atmosbench/split_manifest.csv"),
                  events=config["splits"][args.split],
                  pair_inventory=str(ROOT.parent / "atmosbench/pair_inventory.csv"))
    try:
        dataset = FrozenDataset(config, None, config["conditions"]["clean"], smoke=args.smoke)
        scores = score_predictions(args.predictions, dataset)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    write_new_json(args.output, {"split": args.split, "kind": "smoke" if args.smoke else "predictions", **scores})


if __name__ == "__main__":
    main()
