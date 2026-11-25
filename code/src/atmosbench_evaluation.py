"""Frozen AtmosBench evaluation, input stress and event-paired statistics."""
from __future__ import annotations

import itertools
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from atb_encoding import (  # noqa: E402
    FILL_ABS_THRESHOLD, PHYSICAL_ATB_MIN, PHYSICAL_FLOOR, log_bounds,
    optical_depth_from_atb_numpy, physical_sdc_channel,
    prepare_atb_conditioning, prepare_atb_fullrange, prepare_atb_physical_clip,
    prepare_atb_signed_asinh, prepare_sample_arrays,
)
from dataset_protocol import (  # noqa: E402
    iter_pairs, load_stats_json, load_top_down_array, read_csv_rows, visible_mask_from_observability,
)

METRICS = ("log10_mae", "linear_mae", "linear_rmse")


def valid_atb(raw: np.ndarray) -> np.ndarray:
    return np.isfinite(raw) & (np.abs(raw) < FILL_ABS_THRESHOLD)


def sigma_for_band(config: dict, band: str) -> float:
    source = config["sigma"]
    if source["status"] != "frozen_at_input_resolution":
        raise ValueError("Input-resolution noise sigma is unavailable.")
    sigma = float(source["per_band_m_inv_sr_inv"][band])
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("source sigma must be positive and finite")
    return sigma


def perturb_raw(raw: np.ndarray, condition: dict, *, band: str, item_id: str,
                config: dict) -> np.ndarray:
    """Same per-item Gaussian field in every model and every noise multiplier."""
    result = np.array(raw, dtype=np.float32, copy=True)
    use = valid_atb(result)
    kind = condition["kind"]
    if kind == "additive" and condition["multiplier"] != 0:
        event, wavelength, index = item_id.split("_")
        rng = np.random.default_rng(np.random.SeedSequence(
            [config["perturbation_seed"], int(event[1:]), int(wavelength), int(index)]))
        noise = rng.standard_normal(result.shape).astype(np.float32)
        result[use] += float(condition["multiplier"]) * sigma_for_band(config, band) * noise[use]
    elif kind == "background_bias":
        result[use] += float(condition["multiplier"]) * sigma_for_band(config, band)
    elif kind == "calibration":
        result[use] *= float(condition["gain"])
    elif kind == "linear_quantization":
        step = float(condition["step_m_inv_sr_inv"])
        result[use] = np.rint(result[use] / step) * step
    elif kind == "hard_censor":
        result[use] = np.maximum(result[use], float(condition["floor"]))
    elif kind not in ("clean", "additive", "coordinate_mismatch"):
        raise ValueError(f"unknown perturbation {kind}")
    if not np.isfinite(result[use]).all():
        raise ValueError("raw-input perturbation produced nonfinite observations")
    return result


def encode(raw: np.ndarray, stats: dict, band: str, representation: str,
           condition: dict) -> np.ndarray:
    """Rebuild all channels after perturbation using fixed definitions."""
    if representation == "default":
        return prepare_atb_conditioning(raw, stats, band).conditioning
    if representation == "signed_asinh":
        return prepare_atb_signed_asinh(raw, stats, band).conditioning
    floor = float(condition.get("encoding_floor", PHYSICAL_FLOOR))
    guard = float(condition.get("numerical_guard", PHYSICAL_ATB_MIN))
    if representation == "fullrange":
        return prepare_atb_fullrange(raw, stats, band, numerical_guard=guard).conditioning
    if representation not in ("physical_sdc", "physical_null", "physical_clip"):
        raise ValueError(f"Unknown representation {representation!r}")
    base = prepare_atb_physical_clip(raw, stats, band, physical_floor=floor).conditioning
    if representation == "physical_sdc":
        deficit = physical_sdc_channel(raw, physical_floor=floor, numerical_guard=guard)
        base = np.concatenate([base, deficit[None]], axis=0)
    elif representation == "physical_null":
        base = np.concatenate([base, base[0:1].copy()], axis=0)
    # Floor sensitivity changes the encoding only; observability uses the fixed F.
    return np.ascontiguousarray(base)


def input_strata(raw: np.ndarray, stats: dict, band: str) -> dict[str, np.ndarray]:
    """Fixed clean-input support; neither targets nor predictions define strata."""
    use = valid_atb(raw)
    pc = prepare_atb_physical_clip(raw, stats, band)
    clean = np.where(use, np.maximum(raw, PHYSICAL_FLOOR), PHYSICAL_FLOOR).astype(np.float32)
    depth = optical_depth_from_atb_numpy(clean, stats, band)
    visible = visible_mask_from_observability(pc.conditioning[1], stats, band)
    return {"all": use, "positive_below_F": use & (raw > 0) & (raw < PHYSICAL_FLOOR),
            "at_or_above_F": use & (raw >= PHYSICAL_FLOOR), "nonpositive": use & (raw <= 0),
            "low_observability": use & ~visible, "visible": use & visible,
            "input_tau_le_train_p95": use & (depth <= 1), "input_tau_gt_train_p95": use & (depth > 1)}


class FrozenDataset(Dataset):
    def __init__(self, config: dict, row: dict | None, condition: dict, *, smoke: bool = False):
        self.config, self.row = config, row
        encoding = {key: value for key, value in (row or {}).items()
                    if key in ("encoding_floor", "numerical_guard")}
        self.condition = {**encoding, **condition}
        self.stats = load_stats_json(config["stats_json"])
        self.samples = [sample for band in config["bands"] for sample in iter_pairs(
            data_root=config["data_root"], split_manifest_csv=config["split_manifest"],
            band=band, split=config["split"], max_items_per_event=1 if smoke else None,
            strict=True)]
        expected_cells = set(itertools.product(config["events"], config["bands"]))
        per_cell, remainder = divmod(config["expected_pairs"], len(expected_cells))
        if remainder or per_cell == 0:
            raise ValueError("Expected pair count must divide evenly over the event–wavelength cells")
        if smoke:
            per_cell = 1
        actual = defaultdict(int)
        for sample in self.samples:
            actual[(sample.eid, sample.wavelength_nm)] += 1
        if actual.keys() - expected_cells:
            raise ValueError(f"Unexpected event–wavelength cells: {sorted(actual.keys() - expected_cells)}")
        for event, band in sorted(expected_cells):
            count = actual[(event, band)]
            if count != per_cell:
                raise ValueError(f"Expected {per_cell} pairs for {event} at {band} nm, found {count}")
        if not smoke and config.get("pair_inventory"):
            self.validate_inventory(Path(config["pair_inventory"]))

    def validate_inventory(self, inventory: Path):
        rows = [row for row in read_csv_rows(inventory) if row["split"] == self.config["split"]]
        expected = {row["item_id"]: (row["atb_relative_path"], row["bc_relative_path"]) for row in rows}
        if len(rows) != len(expected) or len(expected) != len(self.samples):
            raise ValueError(f"Unexpected pair count or duplicate item IDs in {inventory}")
        root = Path(self.config["data_root"])
        for sample in self.samples:
            actual = tuple(Path(path).relative_to(root).as_posix() for path in (sample.x_path, sample.y_path))
            if actual != expected.get(sample.item_id):
                raise ValueError(f"Pair paths for {sample.item_id} differ from the frozen inventory: {actual}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        spec = self.samples[index]
        band = str(spec.wavelength_nm)
        raw = load_top_down_array(spec.x_path).astype(np.float32)
        target = prepare_sample_arrays(raw, load_top_down_array(spec.y_path), self.stats, band)
        perturbed = perturb_raw(raw, self.condition, band=band, item_id=spec.item_id, config=self.config)
        sample = {"target_norm": torch.from_numpy(target.target_norm),
                  "target_valid": torch.from_numpy(target.target_valid)}
        if self.row is not None:
            conditioning = encode(perturbed, self.stats, band, self.row["representation"], self.condition)
            sample["conditioning"] = torch.from_numpy(conditioning)
        if self.config["split"] == "train":
            return sample
        changed = valid_atb(raw) & (raw != perturbed)
        use = valid_atb(raw)
        return {**sample,
                "target_beta": torch.from_numpy(target.target_beta),
                "strata": {name: torch.from_numpy(mask) for name, mask in input_strata(raw, self.stats, band).items()},
                "input_counts": np.array([use.sum(), changed.sum(), (use & (raw <= 0)).sum(),
                                           (use & (raw > 0) & (raw < 1e-30)).sum(),
                                           (use & (perturbed < 0)).sum()], dtype=np.int64),
                "eid": spec.eid, "band": band, "item_id": spec.item_id}


def finalize(row: dict) -> dict:
    count = int(row.get("count", 0))
    return {"count": count, "log10_mae": row["log10_abs_sum"] / count if count else None,
            "linear_mae": row["linear_abs_sum"] / count if count else None,
            "linear_rmse": math.sqrt(row["linear_sq_sum"] / count) if count else None}


def macro(rows: list[dict]) -> dict:
    values = {key: [row[key] for row in rows if row[key] is not None] for key in METRICS}
    return {key: float(np.mean(series)) if series else None for key, series in values.items()}


@torch.inference_mode()
def evaluate(model, dataset: FrozenDataset, device: torch.device, *, batch_size: int,
             workers: int, amp: bool) -> dict:
    def predict(batch):
        conditioning = batch["conditioning"].to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            return model(conditioning)["mean_norm"].float()

    return _score(dataset, device, predict, batch_size=batch_size, workers=workers)


@torch.inference_mode()
def score_predictions(prediction_dir: Path, dataset: FrozenDataset) -> dict:
    """Score saved native bottom-up BC predictions in physical units."""
    def predict(batch):
        predictions = []
        for item_id in batch["item_id"]:
            path = prediction_dir / f"{item_id}.npy"
            array = np.load(path, allow_pickle=False)
            shape = tuple(batch["target_beta"].shape[-2:])
            if array.shape != shape or array.dtype.kind != "f":
                raise ValueError(f"{path}: expected a floating-point array of shape {shape}")
            predictions.append(array[::-1].astype(np.float32, copy=True))
        return torch.from_numpy(np.stack(predictions)[:, None])

    return _score(dataset, torch.device("cpu"), predict, batch_size=1, workers=0, physical_units=True)


def _score(dataset: FrozenDataset, device: torch.device, predict, *, batch_size: int,
           workers: int, physical_units: bool = False) -> dict:
    sums = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    input_counts = np.zeros(5, dtype=np.int64)
    started = time.monotonic()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    for batch_index, batch in enumerate(loader):
        prediction = predict(batch)
        for i, (event, band) in enumerate(zip(batch["eid"], batch["band"])):
            target_norm = batch["target_norm"][i, 0].to(device).float()
            target_beta = batch["target_beta"][i, 0].to(device).float()
            valid = batch["target_valid"][i, 0].to(device).bool()
            lo, hi = log_bounds(dataset.stats, str(band), "bc")
            span = float(hi - lo)
            # Preserve the original evaluator's normalized/clipped log target
            # and unclipped positive linear target exactly.
            if physical_units:
                beta = prediction[i, 0]
                if not (torch.isfinite(beta[valid]) & (beta[valid] > 0)).all():
                    raise ValueError(f"Expected positive finite BC on valid pixels for {batch['item_id'][i]}")
                pred = (torch.log10(beta) - lo) / span
            else:
                pred = prediction[i, 0]
                beta = torch.pow(10.0, pred * span + lo)
                if not torch.isfinite(beta[valid]).all():
                    raise ValueError(f"nonfinite prediction for {batch['item_id'][i]}")
            for name, masks in batch["strata"].items():
                use = valid & masks[i].to(device).bool()
                count = int(use.sum().item())
                row = sums[name][f"{event}_{band}"]
                row["count"] += count
                if count:
                    error = beta[use] - target_beta[use]
                    row["log10_abs_sum"] += float((torch.abs(pred[use] - target_norm[use]) * span).sum().cpu())
                    row["linear_abs_sum"] += float(torch.abs(error).sum().cpu())
                    row["linear_sq_sum"] += float(error.square().sum().cpu())
        input_counts += batch["input_counts"].sum(dim=0).numpy()
        if batch_index == 0 or (batch_index + 1) % 100 == 0:
            print(json.dumps({"samples_completed": min((batch_index + 1) * batch_size, len(dataset)),
                              "total_samples": len(dataset), "elapsed_seconds": round(time.monotonic() - started, 2)}), flush=True)
    strata = {}
    for name, cells in sums.items():
        metrics = {cell: finalize(row) for cell, row in sorted(cells.items())}
        strata[name] = {"cell_metrics": metrics, "cell_macro": macro(list(metrics.values()))}
    all_cells = strata["all"]["cell_metrics"]
    if any(row["count"] == 0 for row in all_cells.values()):
        raise ValueError("Every event–wavelength cell must have valid target pixels")
    event_metrics = {event: macro([all_cells[f"{event}_{band}"] for band in dataset.config["bands"]])
                     for event in dataset.config["events"]}
    wavelength_macros = {band: macro([all_cells[f"{event}_{band}"] for event in dataset.config["events"]])
                         for band in dataset.config["bands"]}
    return {"items": len(dataset), "cell_metrics": strata["all"]["cell_metrics"],
            "cell_macro": strata["all"]["cell_macro"], "wavelength_macros": wavelength_macros,
            "event_metrics_mean_over_wavelengths": event_metrics, "input_strata": strata,
            "input_counts": dict(zip(("valid", "perturbed", "clean_nonpositive", "clean_positive_below_1e_minus_30_guard", "perturbed_negative"), input_counts.tolist())),
            "elapsed_seconds": round(time.monotonic() - started, 3)}


def paired_event_stats(ours: dict, comparator: dict) -> dict:
    """Exact four-event nonparametric bootstrap; wavelengths move together."""
    a, b = ours["cell_metrics"], comparator["cell_metrics"]
    if set(a) != set(b) or any(a[k]["count"] != b[k]["count"] for k in a):
        raise ValueError("paired comparison requires identical event/wavelength support")
    events = sorted({key.split("_")[0] for key in a})
    result = {}
    for metric in METRICS:
        diffs = np.array([np.mean([a[k][metric] - b[k][metric] for k in a if k.startswith(event + "_")]) for event in events])
        draws = np.array([diffs[list(indices)].mean() for indices in itertools.product(range(len(events)), repeat=len(events))])
        result[metric] = {"event_differences_ours_minus_comparator": dict(zip(events, diffs.tolist())),
                          "mean_difference": float(diffs.mean()), "percentile_95_interval": np.quantile(draws, [0.025, 0.975]).tolist(),
                          "event_count": len(events), "bootstrap_draws_exact": len(draws)}
    return {"method": "exact event bootstrap, wavelengths clustered; descriptive with only four events",
            "negative_difference_favors": "ours", "metrics": result}


def write_new_json(path: Path, result: dict):
    serialized = json.dumps(result, indent=2, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(serialized + "\n")


