"""Registered CALIOP, EarthCARE and PACE-PAX evaluation routines."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from scipy.special import erf

from atb_encoding import log_bounds, prepare_atb_fullrange, prepare_atb_physical_clip, prepare_atb_signed_asinh, physical_sdc_channel, physical_null_channel
from hsrl_metrics import amplitude_metrics, layer_metrics, layer_summary

FILL_ABS_THRESHOLD = 1e30
PAD_MULTIPLE = 8
CPR_DETECTION_LINEAR = 10.0 ** (-35.0 / 10.0)
TARGETS = {"caliop": ("cpro", "apro"), "atlid": ("ebd",), "cpr": ("cpr",), "hsrl": ("hsrl",)}
BANDS = {"caliop": "532", "atlid": "355", "cpr": "355", "hsrl": "532"}
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
_SQRT2 = math.sqrt(2.0)


def _phi(z: np.ndarray) -> np.ndarray:
    return _INV_SQRT_2PI * np.exp(-0.5 * np.square(z))


def _big_phi(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + erf(z / _SQRT2))


def gaussian_crps(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> np.ndarray:
    sigma_safe = np.maximum(np.asarray(sigma, dtype=np.float64), 1.0e-12)
    z = (np.asarray(y, dtype=np.float64) - np.asarray(mu, dtype=np.float64)) / sigma_safe
    return sigma_safe * (
        z * (2.0 * _big_phi(z) - 1.0)
        + 2.0 * _phi(z)
        - 1.0 / math.sqrt(math.pi)
    )


def selective_risk(
    absolute_log_error: np.ndarray,
    uncertainty_log10: np.ndarray,
) -> dict[str, float]:
    error = np.asarray(absolute_log_error, dtype=np.float64).reshape(-1)
    uncertainty = np.asarray(uncertainty_log10, dtype=np.float64).reshape(-1)
    use = np.isfinite(error) & np.isfinite(uncertainty) & (uncertainty >= 0.0)
    if not use.any():
        raise ValueError("selective-risk evaluation has no finite samples")
    error = error[use]
    uncertainty = uncertainty[use]
    order = np.argsort(uncertainty, kind="stable")
    sorted_error = error[order]
    cumulative_risk = np.cumsum(sorted_error) / np.arange(1, sorted_error.size + 1)
    n80 = max(1, int(math.ceil(0.8 * sorted_error.size)))
    return {
        "aurc": float(cumulative_risk.mean()),
        "risk_at_80_coverage": float(cumulative_risk[n80 - 1]),
        "selective_pixels": int(sorted_error.size),
    }


def _median_profile_correlation(
    prediction: np.ndarray,
    reference: np.ndarray,
    valid: np.ndarray,
) -> tuple[float, int]:
    correlations = []
    for column in range(prediction.shape[1]):
        use = valid[:, column]
        if int(use.sum()) < 3:
            continue
        x = np.log10(prediction[use, column])
        y = np.log10(reference[use, column])
        if np.std(x) <= 0.0 or np.std(y) <= 0.0:
            continue
        correlations.append(float(np.corrcoef(x, y)[0, 1]))
    if not correlations:
        return float("nan"), 0
    return float(np.median(correlations)), len(correlations)


def evaluate_target(
    prediction_beta: np.ndarray,
    reference_beta: np.ndarray,
    reference_valid: np.ndarray,
    *,
    uncertainty_log10: np.ndarray | None = None,
) -> dict[str, Any]:
    prediction = np.asarray(prediction_beta, dtype=np.float64)
    reference = np.asarray(reference_beta, dtype=np.float64)
    reference_mask = np.asarray(reference_valid, dtype=bool)
    if prediction.shape != reference.shape or reference_mask.shape != reference.shape:
        raise ValueError("prediction/reference/valid shapes differ")
    valid = reference_mask & np.isfinite(reference) & (reference > 0.0)
    if not valid.any():
        raise ValueError("retrieval target has no valid positive reference pixels")
    invalid_prediction = valid & (~np.isfinite(prediction) | (prediction <= 0.0))
    if invalid_prediction.any():
        raise ValueError(
            "prediction is non-finite or non-positive on the frozen common evaluation mask"
        )
    linear_error = prediction[valid] - reference[valid]
    log_error = np.log10(prediction[valid]) - np.log10(reference[valid])
    absolute_log_error = np.abs(log_error)
    height_count = valid.sum(axis=1).astype(np.int64)
    height_log_abs_sum = np.where(
        valid,
        np.abs(
            np.log10(np.clip(prediction, 1.0e-30, None))
            - np.log10(np.clip(reference, 1.0e-30, None))
        ),
        0.0,
    ).sum(axis=1)
    correlation, correlation_profiles = _median_profile_correlation(
        prediction,
        reference,
        valid,
    )
    result: dict[str, Any] = {
        "valid_pixels": int(valid.sum()),
        "median_abs_dlog10_beta": float(np.median(absolute_log_error)),
        "log10_mae": float(absolute_log_error.mean()),
        "mae_beta": float(np.abs(linear_error).mean()),
        "bias_beta": float(linear_error.mean()),
        "profile_correlation_median": correlation,
        "profile_correlation_profiles": correlation_profiles,
        "per_height_valid_count": height_count.tolist(),
        "per_height_log_abs_sum": height_log_abs_sum.tolist(),
    }
    if uncertainty_log10 is not None:
        sigma_full = np.asarray(uncertainty_log10, dtype=np.float64)
        if sigma_full.shape != reference.shape:
            raise ValueError("uncertainty/reference shapes differ")
        sigma = np.maximum(sigma_full[valid], 1.0e-12)
        mu = np.log10(prediction[valid])
        y = np.log10(reference[valid])
        z = (y - mu) / sigma
        q05 = mu - 1.6449 * sigma
        q95 = mu + 1.6449 * sigma
        result.update(
            {
                "gaussian_nll_log10": float(
                    (
                        0.5 * np.log(2.0 * math.pi * np.square(sigma))
                        + 0.5 * np.square(z)
                    ).mean()
                ),
                "gaussian_crps_log10": float(gaussian_crps(mu, sigma, y).mean()),
                "picp90": float(((y >= q05) & (y <= q95)).mean()),
                "interval_width90_log10": float((q95 - q05).mean()),
                **selective_risk(absolute_log_error, sigma),
            }
        )
    return result


def common_eval_mask(atb: np.ndarray, reference_valid: np.ndarray) -> np.ndarray:
    """Freeze one method-independent pixel set for a CALIOP target.

    The log-domain raw-ATB comparator is defined only on positive, finite L1
    input. Learned and conditional rows must use exactly that same support;
    otherwise their finite inpainted values would be scored on extra pixels.
    """
    input_atb = np.asarray(atb, dtype=np.float64)
    reference_mask = np.asarray(reference_valid, dtype=bool)
    if input_atb.shape != reference_mask.shape:
        raise ValueError("ATB/reference-valid shapes differ")
    input_support = (
        np.isfinite(input_atb)
        & (np.abs(input_atb) < FILL_ABS_THRESHOLD)
        & (input_atb > 0.0)
    )
    return reference_mask & input_support


def horizontal_tiles(
    width: int,
    core_width: int,
    halo: int,
) -> list[tuple[int, int, int, int]]:
    if width < 1 or core_width < 1 or halo < 0:
        raise ValueError("tile dimensions must be positive and halo non-negative")
    if core_width % PAD_MULTIPLE or halo % PAD_MULTIPLE or width % PAD_MULTIPLE:
        raise ValueError("tile core, halo, and padded width must be multiples of 8")
    tiles = []
    for core_start in range(0, width, core_width):
        core_end = min(width, core_start + core_width)
        read_start = max(0, core_start - halo)
        read_end = min(width, core_end + halo)
        tiles.append((core_start, core_end, read_start, read_end))
    if tiles[0][0] != 0 or tiles[-1][1] != width:
        raise AssertionError("tile coverage is incomplete")
    if any(left[1] != right[0] for left, right in zip(tiles, tiles[1:])):
        raise AssertionError("tile cores overlap or leave gaps")
    return tiles


def _finite_inputs(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    label_array = np.asarray(labels, dtype=bool).reshape(-1)
    score_array = np.asarray(scores, dtype=np.float64).reshape(-1)
    if label_array.shape != score_array.shape:
        raise ValueError(f"label/score shape mismatch: {label_array.shape} vs {score_array.shape}")
    finite = np.isfinite(score_array)
    return label_array[finite], score_array[finite]


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Tie-safe Mann-Whitney ROC-AUC."""
    y, score = _finite_inputs(labels, scores)
    positives = int(y.sum())
    negatives = int(y.size - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("ROC-AUC requires both reference classes")
    order = np.argsort(score, kind="stable")
    sorted_score = score[order]
    ranks = np.empty(score.size, dtype=np.float64)
    start = 0
    while start < score.size:
        stop = start + 1
        while stop < score.size and sorted_score[stop] == sorted_score[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * ((start + 1) + stop)
        start = stop
    rank_sum_positive = float(ranks[y].sum())
    return (rank_sum_positive - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """Threshold-grouped AP; invariant to ordering within score ties."""
    y, score = _finite_inputs(labels, scores)
    positives = int(y.sum())
    if positives == 0:
        raise ValueError("average precision requires at least one positive reference")
    order = np.argsort(-score, kind="stable")
    sorted_score = score[order]
    sorted_y = y[order]
    cumulative_positive = 0
    cumulative_total = 0
    result = 0.0
    start = 0
    while start < score.size:
        stop = start + 1
        while stop < score.size and sorted_score[stop] == sorted_score[start]:
            stop += 1
        group_positive = int(sorted_y[start:stop].sum())
        cumulative_positive += group_positive
        cumulative_total += stop - start
        result += (group_positive / positives) * (
            cumulative_positive / cumulative_total
        )
        start = stop
    return float(result)


def evaluate_score(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    y, score = _finite_inputs(labels, scores)
    if y.size == 0:
        raise ValueError("score evaluation has no finite pixels")
    positives = int(y.sum())
    negatives = int(y.size - positives)
    if positives == 0 or negatives == 0:
        raise ValueError(
            "threshold-free evaluation requires both CPR proxy classes; "
            f"positive={positives}, negative={negatives}"
        )
    unique = np.unique(score)
    return {
        "roc_auc": float(roc_auc(y, score)),
        "average_precision": float(average_precision(y, score)),
        "reference_positive_fraction": float(y.mean()),
        "reference_positive_pixels": positives,
        "reference_negative_pixels": negatives,
        "reference_both_classes": True,
        "valid_pixels": int(y.size),
        "unique_score_count": int(unique.size),
        "constant_score": bool(unique.size < 2),
        "score_min": float(score.min()),
        "score_max": float(score.max()),
    }


def radar_proxy_target(
    cpr_linear: np.ndarray,
    cpr_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the QA-valid binary target from CPR arrays only.

    Linear radar reflectivity cannot be negative. Excluding non-finite and
    negative values here makes the QA contract explicit without consulting
    ATLID measurements or model output. The returned target is a CPR structure
    proxy, never BC truth.
    """
    cpr = np.asarray(cpr_linear, dtype=np.float64)
    valid = np.asarray(cpr_valid, dtype=bool)
    if cpr.shape != valid.shape:
        raise ValueError(f"CPR/QA shape mismatch: {cpr.shape} vs {valid.shape}")
    radar_qa = valid & np.isfinite(cpr) & (cpr >= 0.0)
    target = radar_qa & (cpr > CPR_DETECTION_LINEAR)
    return target, radar_qa


def evaluate_registered_score(
    registered: dict[str, np.ndarray],
    scores: np.ndarray,
    *,
    require_full_radar_support: bool = True,
) -> dict[str, Any]:
    """Evaluate one continuous score curtain with explicit support guards."""
    required = {"atl", "cpr", "atl_valid", "cpr_valid"}
    missing = required.difference(registered)
    if missing:
        raise ValueError(f"registered curtain missing arrays {sorted(missing)}")

    atl = np.asarray(registered["atl"], dtype=np.float64)
    atl_valid = np.asarray(registered["atl_valid"], dtype=bool)
    score = np.asarray(scores, dtype=np.float64)
    if atl.shape != atl_valid.shape:
        raise ValueError(f"ATLID/QA shape mismatch: {atl.shape} vs {atl_valid.shape}")
    if score.shape != atl.shape:
        raise ValueError(f"score shape mismatch: {score.shape} vs {atl.shape}")

    target, radar_qa = radar_proxy_target(registered["cpr"], registered["cpr_valid"])
    if target.shape != atl.shape:
        raise ValueError(f"CPR/ATLID registered shape mismatch: {target.shape} vs {atl.shape}")

    score_support = atl_valid & np.isfinite(atl) & np.isfinite(score)
    eligible = radar_qa & score_support
    radar_qa_pixels = int(radar_qa.sum())
    score_support_pixels = int(score_support.sum())
    eligible_pixels = int(eligible.sum())
    if radar_qa_pixels == 0:
        raise ValueError("CPR radar-only QA target has no valid pixels")
    if eligible_pixels == 0:
        raise ValueError("score and CPR radar-only QA target have no jointly eligible pixels")
    if require_full_radar_support and eligible_pixels != radar_qa_pixels:
        raise ValueError(
            "incomplete score support over CPR radar-only QA target; "
            f"eligible={eligible_pixels}, radar_qa={radar_qa_pixels}"
        )

    metrics = evaluate_score(target[eligible], score[eligible])
    metrics.update(
        {
            "radar_only_qa_target": True,
            "radar_qa_pixels": radar_qa_pixels,
            "radar_qa_positive_pixels": int(target[radar_qa].sum()),
            "score_support_pixels": score_support_pixels,
            "eligible_pixels": eligible_pixels,
            "score_support_fraction_of_radar_qa": eligible_pixels / radar_qa_pixels,
        }
    )
    return metrics


def prepare_conditioning(
    atb: np.ndarray,
    stats: dict,
    band: str,
    representation: str,
) -> np.ndarray:
    if representation == "fullrange":
        return prepare_atb_fullrange(atb, stats, band).conditioning
    if representation == "signed_asinh":
        return prepare_atb_signed_asinh(atb, stats, band).conditioning
    clipped = prepare_atb_physical_clip(atb, stats, band).conditioning
    if representation == "physical_clip":
        return clipped
    if representation == "physical_sdc":
        seventh = physical_sdc_channel(atb)
    elif representation == "physical_null":
        seventh = physical_null_channel(clipped[0])
    else:
        raise ValueError(representation)
    return np.concatenate([clipped, seventh[None]], axis=0).astype(np.float32)


def forward_prepared(
    model: torch.nn.Module,
    conditioning_np: np.ndarray,
    device: torch.device,
    tile_core_width: int | None,
    tile_halo: int,
) -> np.ndarray:
    """FP32 AdaIR inference with the frozen full-frame/uniform-halo policy."""
    conditioning_cpu = torch.from_numpy(conditioning_np).unsqueeze(0)
    height, width = conditioning_np.shape[-2:]
    if tile_core_width is None:
        output = model(conditioning_cpu.to(device))["mean_norm"]
        return output[0, 0].float().cpu().numpy()
    prediction = np.empty((height, width), dtype=np.float32)
    for core_start, core_end, read_start, read_end in horizontal_tiles(
        width, tile_core_width, tile_halo
    ):
        tile = conditioning_cpu[..., read_start:read_end].to(device)
        output = model(tile)["mean_norm"]
        crop_start = core_start - read_start
        crop_end = crop_start + core_end - core_start
        prediction[:, core_start:core_end] = (
            output[0, 0, :, crop_start:crop_end].float().cpu().numpy()
        )
        del tile, output
    return prediction


def score_case(case, prediction, source, targets=None):
    """Per-product scoring; missing APro never discards valid CPro, or vice versa."""
    results = {}
    for target in TARGETS[source] if targets is None else targets:
        if source == "hsrl":
            mask = np.asarray(case["learned_common_signed_mask"], bool)
            reference = case["reference_beta_m"]
            z, ground = case["altitude_asl_m"], case["ground_altitude_asl_m"]
            if not mask.any():
                results[target] = {"status": "reference_unavailable", "reason": "empty frozen common reference mask", "common_reference_pixels": 0}
                continue
            try:
                metrics = amplitude_metrics(prediction, reference, mask)
                structure = layer_metrics(prediction, reference, mask, z, ground)
                pred, ref = layer_summary(prediction, mask, z, ground), layer_summary(reference, mask, z, ground)
                results[target] = {"status": "ok", "metrics": metrics, "layer_metrics": structure,
                    "reference_scope": "cloud-screened aerosol-backscatter threshold layers, not independent cloud boundaries",
                    "layer_profiles": {"reference_top_m": ref.top_agl_m.tolist(), "reference_base_m": ref.base_agl_m.tolist(),
                        "reference_eligible": ref.eligible.tolist(), "prediction_top_m": pred.top_agl_m.tolist(),
                        "prediction_base_m": pred.base_agl_m.tolist()}, "common_reference_pixels": int(mask.sum())}
            except ValueError as exc:
                results[target] = {"status": "scoring_failure", "reason": str(exc), "common_reference_pixels": int(mask.sum())}
            continue
        if source == "cpr":
            required = {"atl", "atl_valid", "cpr", "cpr_valid"}
        else:
            required = {"atb_m", f"{target}_beta_m", f"{target}_valid"}
        missing = sorted(required.difference(case))
        if missing:
            results[target] = {"status": "reference_unavailable", "reason": f"missing arrays: {missing}"}
            continue
        if source == "cpr":
            labels, qa = radar_proxy_target(case["cpr"], case["cpr_valid"])
            details = {"reference_qa_pixels": int(qa.sum()), "reference_positive_pixels": int(labels.sum()),
                       "reference_negative_pixels": int(qa.sum() - labels.sum()), "cpr_is_bc_truth": False}
        else:
            reference = case[f"{target}_beta_m"]
            mask = common_eval_mask(case["atb_m"], case[f"{target}_valid"])
            mask = mask & np.isfinite(reference) & (reference > 0)
            details = {"reference_qa_pixels": int(np.asarray(case[f"{target}_valid"], bool).sum()),
                       "common_reference_pixels": int(mask.sum()),
                       "common_pixel_policy": "source QA AND finite positive nonfill ATB AND finite positive reference"}
            if not mask.any():
                results[target] = {"status": "reference_unavailable", "reason": "no valid positive reference on common input support", **details}
                continue
        try:
            metrics = (evaluate_registered_score(case, prediction) if source == "cpr" else
                       evaluate_target(prediction, case[f"{target}_beta_m"],
                                       common_eval_mask(case["atb_m"], case[f"{target}_valid"])))
            results[target] = {"status": "ok", "metrics": metrics, **details}
        except ValueError as exc:
            # Do not drop unsupported/nonfinite predictions or relax the shared mask.
            results[target] = {"status": "scoring_failure", "reason": str(exc), **details}
    return results


def infer_beta(model, atb, stats, source, representation, device, core, halo):
    height, width = atb.shape
    padded = np.pad(atb, ((0, (-height) % 8), (0, (-width) % 8)), constant_values=np.nan)
    conditioning = prepare_conditioning(padded, stats, BANDS[source], representation)
    with torch.inference_mode():
        norm = forward_prepared(model, conditioning, device, core, halo)
    lo, hi = log_bounds(stats, BANDS[source], "bc")
    return np.power(10.0, norm[:height, :width].astype(np.float64) * (hi - lo) + lo).astype(np.float32)
