#!/usr/bin/env python3
"""Deterministic identical-support metrics for full-flight HSRL-2 rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


LOG_FLOOR = 1.0e-12
PHYSICAL_FLOOR = 1.0e-7
LAYER_MIN_AGL_M = 300.0
LAYER_MAX_AGL_M = 12_000.0
LAYER_MIN_BINS = 3
PROFILE_MIN_COMMON_BINS = 12


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_values)) + 1]
    ends = np.r_[starts[1:], values.size]
    for start, end in zip(starts, ends):
        ranks[order[start:end]] = 0.5 * (start + end - 1)
    return ranks


def _corr(a: np.ndarray, b: np.ndarray, *, ranks: bool = False) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 2:
        return float("nan")
    if ranks:
        x, y = _rank(x), _rank(y)
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def amplitude_metrics(
    prediction: np.ndarray, reference: np.ndarray, common_mask: np.ndarray
) -> dict[str, Any]:
    pred = np.asarray(prediction, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    mask = np.asarray(common_mask, dtype=bool)
    if not (pred.shape == ref.shape == mask.shape):
        raise ValueError("amplitude shapes differ")
    if not mask.any():
        raise ValueError("empty common mask")
    if not (np.isfinite(ref[mask]) & (ref[mask] > 0.0)).all():
        raise ValueError("reference violates frozen mask")
    if not np.isfinite(pred[mask]).all():
        raise ValueError("prediction is nonfinite on frozen mask")
    p, r = pred[mask], ref[mask]
    lp = np.log10(np.clip(p, LOG_FLOOR, None))
    lr = np.log10(r)
    error = lp - lr
    return {
        "count": int(mask.sum()),
        "log10_mae": float(np.mean(np.abs(error))),
        "log10_bias": float(np.mean(error)),
        "linear_mae": float(np.mean(np.abs(p - r))),
        "spearman": _corr(lp, lr, ranks=True),
        "nonpositive_prediction_fraction": float(np.mean(p <= 0.0)),
    }


def vertical_gradient_metrics(
    prediction: np.ndarray, reference: np.ndarray, common_mask: np.ndarray
) -> dict[str, Any]:
    pred = np.asarray(prediction, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    mask = np.asarray(common_mask, dtype=bool)
    if not (pred.shape == ref.shape == mask.shape) or pred.ndim != 2:
        raise ValueError("gradient inputs must be height x profile")
    if not np.isfinite(pred[mask]).all():
        raise ValueError("prediction is nonfinite on frozen mask")
    adjacent = mask[:-1] & mask[1:]
    pred_gradient = np.diff(np.log10(np.clip(pred, LOG_FLOOR, None)), axis=0)
    ref_gradient = np.diff(np.log10(ref), axis=0)
    return {
        "count": int(adjacent.sum()),
        "pearson": _corr(pred_gradient[adjacent], ref_gradient[adjacent]),
        "spearman": _corr(
            pred_gradient[adjacent], ref_gradient[adjacent], ranks=True
        ),
        "gradient_mae_log10_per_bin": float(
            np.mean(np.abs(pred_gradient[adjacent] - ref_gradient[adjacent]))
        )
        if adjacent.any()
        else float("nan"),
    }


def integrated_profile_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
    common_mask: np.ndarray,
    altitude_asl_m: np.ndarray,
) -> dict[str, Any]:
    pred = np.asarray(prediction, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    mask = np.asarray(common_mask, dtype=bool)
    altitude = np.asarray(altitude_asl_m, dtype=np.float64).reshape(-1)
    if pred.shape != ref.shape or pred.shape != mask.shape or pred.shape[0] != altitude.size:
        raise ValueError("integrated-profile shapes differ")
    if not np.isfinite(pred[mask]).all():
        raise ValueError("prediction is nonfinite on frozen mask")
    dz = float(np.median(np.abs(np.diff(altitude))))
    eligible = mask.sum(axis=0) >= PROFILE_MIN_COMMON_BINS
    pred_path = np.sum(np.where(mask, np.maximum(pred, 0.0), 0.0), axis=0) * dz
    ref_path = np.sum(np.where(mask, ref, 0.0), axis=0) * dz
    eligible &= np.isfinite(pred_path) & np.isfinite(ref_path) & (ref_path > 0.0)
    if not eligible.any():
        return {"eligible_profiles": 0, "log10_mae": float("nan")}
    lp = np.log10(np.clip(pred_path[eligible], LOG_FLOOR, None))
    lr = np.log10(ref_path[eligible])
    error = lp - lr
    return {
        "eligible_profiles": int(eligible.sum()),
        "quantity": "masked vertically integrated backscatter proxy",
        "units": "sr^-1",
        "minimum_common_bins": PROFILE_MIN_COMMON_BINS,
        "log10_mae": float(np.mean(np.abs(error))),
        "log10_bias": float(np.mean(error)),
        "spearman": _corr(lp, lr, ranks=True),
    }


@dataclass(frozen=True)
class LayerSummary:
    top_agl_m: np.ndarray
    base_agl_m: np.ndarray
    layer_count: np.ndarray
    eligible: np.ndarray


def layer_summary(
    beta: np.ndarray,
    common_mask: np.ndarray,
    altitude_asl_m: np.ndarray,
    ground_altitude_asl_m: np.ndarray,
) -> LayerSummary:
    field = np.asarray(beta, dtype=np.float64)
    mask = np.asarray(common_mask, dtype=bool)
    altitude = np.asarray(altitude_asl_m, dtype=np.float64).reshape(-1)
    ground = np.asarray(ground_altitude_asl_m, dtype=np.float64).reshape(-1)
    if field.shape != mask.shape or field.shape != (altitude.size, ground.size):
        raise ValueError("layer-summary shapes differ")
    top = np.full(ground.size, np.nan)
    base = np.full(ground.size, np.nan)
    count = np.zeros(ground.size, dtype=np.int64)
    eligible = np.zeros(ground.size, dtype=bool)
    for profile in range(ground.size):
        agl = altitude - ground[profile]
        available = (
            mask[:, profile]
            & np.isfinite(field[:, profile])
            & (agl >= LAYER_MIN_AGL_M)
            & (agl <= LAYER_MAX_AGL_M)
        )
        eligible[profile] = int(available.sum()) >= PROFILE_MIN_COMMON_BINS
        active = available & (field[:, profile] >= PHYSICAL_FLOOR)
        indices = np.flatnonzero(active)
        if indices.size == 0:
            continue
        split = np.flatnonzero(np.diff(indices) > 1) + 1
        runs = [run for run in np.split(indices, split) if run.size >= LAYER_MIN_BINS]
        count[profile] = len(runs)
        if not runs:
            continue
        selected = max(
            runs,
            key=lambda run: float(np.sum(np.maximum(field[run, profile], 0.0))),
        )
        top[profile] = float(np.max(agl[selected]))
        base[profile] = float(np.min(agl[selected]))
    return LayerSummary(top, base, count, eligible)


def layer_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
    common_mask: np.ndarray,
    altitude_asl_m: np.ndarray,
    ground_altitude_asl_m: np.ndarray,
) -> dict[str, Any]:
    if not np.isfinite(np.asarray(prediction)[np.asarray(common_mask, dtype=bool)]).all():
        raise ValueError("prediction is nonfinite on frozen mask")
    pred = layer_summary(
        prediction, common_mask, altitude_asl_m, ground_altitude_asl_m
    )
    ref = layer_summary(reference, common_mask, altitude_asl_m, ground_altitude_asl_m)
    ref_detected = ref.eligible & np.isfinite(ref.top_agl_m) & np.isfinite(ref.base_agl_m)
    paired = ref_detected & np.isfinite(pred.top_agl_m) & np.isfinite(pred.base_agl_m)
    output = {
        "operator": {
            "threshold_m-1_sr-1": PHYSICAL_FLOOR,
            "agl_range_m": [LAYER_MIN_AGL_M, LAYER_MAX_AGL_M],
            "minimum_contiguous_bins": LAYER_MIN_BINS,
            "dominant_layer_selection": "largest integrated beta among qualifying runs",
        },
        "reference_detected_profiles": int(ref_detected.sum()),
        "paired_detected_profiles": int(paired.sum()),
        "prediction_coverage": float(paired.sum() / max(int(ref_detected.sum()), 1)),
    }
    if paired.any():
        top_error = pred.top_agl_m[paired] - ref.top_agl_m[paired]
        base_error = pred.base_agl_m[paired] - ref.base_agl_m[paired]
        output.update(
            {
                "top_mae_m": float(np.mean(np.abs(top_error))),
                "top_bias_m": float(np.mean(top_error)),
                "base_mae_m": float(np.mean(np.abs(base_error))),
                "base_bias_m": float(np.mean(base_error)),
                "layer_count_mae": float(
                    np.mean(np.abs(pred.layer_count[paired] - ref.layer_count[paired]))
                ),
                "multilayer_exact_accuracy": float(
                    np.mean(
                        (pred.layer_count[paired] >= 2)
                        == (ref.layer_count[paired] >= 2)
                    )
                ),
            }
        )
    return output
