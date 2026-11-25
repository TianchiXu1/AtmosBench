#!/usr/bin/env python3
"""Train an AtmosBench retrieval model and select its checkpoint on validation."""
import argparse
import json
import math
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from atmosbench_evaluation import FrozenDataset, evaluate, write_new_json
from atmosbench_models import TRAINING_SCHEMA, build_model
from dataset_protocol import load_split_manifest


def sample_losses(outputs, target, valid, head):
    reduction = (1, 2, 3)
    prediction = outputs["mean_norm"]
    valid = valid.float()
    count = valid.sum(dim=reduction).clamp_min(1)
    error = prediction - target
    if head == "gaussian":
        logvar = outputs["logvar_norm"]
        nll = 0.5 * (logvar + error.square() / torch.exp(logvar) + math.log(2 * math.pi))
        return (nll * valid).sum(dim=reduction) / count

    l1 = (error.abs() * valid).sum(dim=reduction) / count
    mse = (error.square() * valid).sum(dim=reduction) / count
    vertical_valid = valid[:, :, 1:, :] * valid[:, :, :-1, :]
    horizontal_valid = valid[:, :, :, 1:] * valid[:, :, :, :-1]
    vertical_error = (prediction[:, :, 1:, :] - prediction[:, :, :-1, :]) - (
        target[:, :, 1:, :] - target[:, :, :-1, :])
    horizontal_error = (prediction[:, :, :, 1:] - prediction[:, :, :, :-1]) - (
        target[:, :, :, 1:] - target[:, :, :, :-1])
    vertical_count = vertical_valid.sum(dim=reduction)
    horizontal_count = horizontal_valid.sum(dim=reduction)
    vertical = (vertical_error.abs() * vertical_valid).sum(dim=reduction) / vertical_count.clamp_min(1)
    horizontal = (horizontal_error.abs() * horizontal_valid).sum(dim=reduction) / horizontal_count.clamp_min(1)
    directions = (vertical_count > 0).float() + (horizontal_count > 0).float()
    return l1 + 0.25 * mse + 0.05 * ((vertical + horizontal) / directions.clamp_min(1))


def train_epoch(model, loader, optimizer, device, *, accumulation, head, smoke):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sum = sample_count = update_count = 0
    total_batches = 1 if smoke else len(loader)
    for batch_number, batch in enumerate(loader, 1):
        conditioning = batch["conditioning"].to(device, non_blocking=True)
        target = batch["target_norm"].to(device, non_blocking=True)
        valid = batch["target_valid"].to(device, non_blocking=True)
        loss = sample_losses(model(conditioning), target, valid, head).mean()
        if not torch.isfinite(loss):
            raise ValueError(f"Nonfinite training loss in batch {batch_number}.")

        window_start = ((batch_number - 1) // accumulation) * accumulation
        window_size = min(accumulation, total_batches - window_start)
        (loss / window_size).backward()
        if batch_number == window_start + window_size:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            update_count += 1
            optimizer.zero_grad(set_to_none=True)
        loss_sum += float(loss.detach().cpu()) * len(target)
        sample_count += len(target)
        if smoke:
            break

    return {"train_loss": loss_sum / sample_count, "samples": sample_count, "updates": update_count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, default=ROOT.parent / "atmosbench" / "split_manifest.csv")
    parser.add_argument("--output", type=Path, required=True, help="New training output directory")
    parser.add_argument("--architecture", choices=("adair", "logunet", "promptir", "datprl"), default="adair")
    parser.add_argument("--representation", choices=("physical_sdc", "fullrange", "physical_clip",
                                                       "physical_null", "signed_asinh"), default="physical_sdc")
    parser.add_argument("--head", choices=("point", "gaussian"), default="point")
    parser.add_argument("--encoding-floor", type=float, choices=(1e-8, 1e-7, 1e-6), default=1e-7)
    parser.add_argument("--numerical-guard", type=float, choices=(1e-45, 1e-30), default=1e-45)
    parser.add_argument("--condition", choices=("clean", "additive_1"), default="clean")
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--validation-batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--smoke", action="store_true", help="One optimizer update and eight validation pairs")
    args = parser.parse_args()
    if args.head == "gaussian" and args.architecture != "logunet":
        parser.error("The Gaussian head requires --architecture logunet.")
    if args.micro_batch_size <= 0 or 12 % args.micro_batch_size:
        parser.error("Training micro-batch size must divide the effective batch size of 12.")
    if args.validation_batch_size <= 0 or args.workers < 0:
        parser.error("Validation batch size must be positive and worker count must be nonnegative.")
    if not 0 <= args.seed < 2**32:
        parser.error("Seed must be between 0 and 2**32 - 1.")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Use --device cpu to run on CPU.")

    config = json.loads((ROOT / "configs/evaluation.json").read_text())
    config.update(data_root=str(args.data_root), split_manifest=str(args.split_manifest),
                  stats_json=str(ROOT / "configs/train_normalization.json"),
                  pair_inventory=str(ROOT.parent / "atmosbench/pair_inventory.csv"))
    train_events = set(json.loads(Path(config["stats_json"]).read_text())["event_ids"])
    expected_splits = {"train": train_events, **{name: set(events) for name, events in config["splits"].items()}}
    split_rows = load_split_manifest(args.split_manifest)
    if any({eid for eid, row in split_rows.items() if row["split"] == split} != events
           for split, events in expected_splits.items()):
        parser.error("The split must match the event rosters used in AtmosBench.")
    condition = {**config["conditions"][args.condition], "encoding_floor": args.encoding_floor,
                 "numerical_guard": args.numerical_guard}
    model_config = {"architecture": args.architecture, "representation": args.representation, "head": args.head}
    training = FrozenDataset({**config, "split": "train", "events": sorted(train_events), "expected_pairs": 30240},
                             model_config, condition, smoke=args.smoke)
    validation = FrozenDataset({**config, "split": "validation", "events": config["splits"]["validation"]},
                               model_config, condition, smoke=args.smoke)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(training, batch_size=args.micro_batch_size, shuffle=True, generator=generator,
                        num_workers=args.workers, persistent_workers=args.workers > 0,
                        pin_memory=device.type == "cuda")
    model = build_model(model_config, ROOT / "third_party").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-4)
    args.output.mkdir(parents=True, exist_ok=False)
    epochs = 1 if args.smoke else 10
    settings = {"training_schema": TRAINING_SCHEMA, "smoke": args.smoke, "seed": args.seed,
                "precision": "fp32", "epochs": epochs,
                "effective_batch_size": 12,
                "micro_batch_size": args.micro_batch_size, "validation_batch_size": args.validation_batch_size,
                "architecture": args.architecture, "head": args.head, "atb_representation": args.representation,
                "encoding_floor": args.encoding_floor, "numerical_guard": args.numerical_guard,
                "training_condition": args.condition}
    write_new_json(args.output / "training_config.json", settings)
    best = float("inf")
    for epoch in range(1, epochs + 1):
        train_metrics = train_epoch(model, loader, optimizer, device, accumulation=12 // args.micro_batch_size,
                                    head=args.head, smoke=args.smoke)
        evidence = evaluate(model, validation, device, batch_size=args.validation_batch_size,
                            workers=args.workers, amp=False)
        loss = evidence["cell_macro"]["log10_mae"]
        if loss is None or not math.isfinite(loss):
            raise ValueError("Validation produced no finite macro log MAE.")
        write_new_json(args.output / f"epoch_{epoch:02}.json", {"epoch": epoch, **train_metrics,
                       "validation_cells": evidence["cell_metrics"], "validation_macro": evidence["cell_macro"]})
        if loss < best:
            best = loss
            temporary = args.output / "best.pt.tmp"
            torch.save({**settings, "model": model.state_dict(), "epoch": epoch, "validation_log10_mae": loss}, temporary)
            temporary.replace(args.output / "best.pt")
        print(json.dumps({"epoch": epoch, **train_metrics, "validation_macro_log10_mae": loss}), flush=True)


if __name__ == "__main__":
    main()
