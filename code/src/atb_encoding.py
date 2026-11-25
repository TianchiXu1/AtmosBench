"""ATB-only encoding and separate BC target preparation for AtmosBench."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from dataset_protocol import (
    HEIGHT_KM,
    observability_from_atb_numpy,
    stats_for_band,
)


FILL_ABS_THRESHOLD = 1.0e30
EPS = 1.0e-12

# Nominal encoding floor shared across instruments (m^-1 sr^-1).
PHYSICAL_FLOOR = 1.0e-7
PHYSICAL_ATB_MIN = 1.0e-45  # below any physically meaningful ATB

# ATB representation modes
ATB_REPRESENTATIONS = (
    "fullrange",
    "signed_asinh",
    "physical_clip",
    "physical_sdc",
    "physical_null",
)

SIGNED_ASINH_UNIT = "m^-1 sr^-1"
SIGNED_ASINH_FORMULA = "asinh(x / s_b) / asinh(a_b / s_b)"
SIGNED_ASINH_NEGATIVE_POLICY = (
    "Preserve every finite signed ATB value with abs(x) < FILL_ABS_THRESHOLD; "
    "map non-finite/fill values to zero and mark them invalid; do not clip valid values."
)


def optical_depth_from_atb_numpy(atb: np.ndarray, stats: dict[str, Any], band: str) -> np.ndarray:
    """Cumulative optical-depth proxy derived from ATB alone.

    Used for input stratification, integrating a 5×5-smoothed log-ATB signal
    and normalizing by the train-only `tau_proxy_p95`.
    """

    rule = stats_for_band(stats, band)["mask_rule_stats"]
    eps = float(rule["positive_eps"])
    log_clip_hi = float(rule["log_atb_p99_9"])
    log_atb = np.log10(np.clip(atb.astype(np.float32, copy=False), eps, 10.0 ** log_clip_hi))
    padded = np.pad(log_atb, 2, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (5, 5))
    smooth = windows.mean(axis=(-2, -1)).astype(np.float32)
    denom = max((float(rule["log_atb_p99_9"]) - float(rule["log_atb_p50"])) / 8.0, 1e-6)
    attenuator_prob = 1.0 / (1.0 + np.exp(-np.clip((smooth - float(rule["log_atb_p90"])) / denom, -60.0, 60.0)))
    dz_km = HEIGHT_KM / max(atb.shape[0] - 1, 1)
    tau_proxy = np.cumsum(attenuator_prob, axis=0) * dz_km
    tau_norm = tau_proxy / max(float(rule["tau_proxy_p95"]), 1e-6)
    return np.clip(tau_norm, 0.0, 4.0).astype(np.float32)


def log_bounds(stats: dict[str, Any], band: str, kind: str) -> tuple[float, float]:
    values = stats["wavelengths"][str(band)]
    lo_value = float(values[f"{kind}_clip_min"])
    hi_value = float(values[f"{kind}_clip_max"])
    lo = math.log10(max(lo_value, EPS))
    hi = math.log10(max(hi_value, lo_value * 10.0, EPS))
    return lo, hi


def log_normalize(values: np.ndarray, stats: dict[str, Any], band: str, kind: str) -> np.ndarray:
    lo, hi = log_bounds(stats, band, kind)
    log_values = np.log10(np.clip(np.asarray(values, dtype=np.float32), 10.0**lo, 10.0**hi))
    return np.clip((log_values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


@dataclass(frozen=True)
class PreparedSample:
    conditioning: np.ndarray
    target_norm: np.ndarray
    target_beta: np.ndarray
    target_valid: np.ndarray
    atb_valid: np.ndarray
    observability: np.ndarray


@dataclass(frozen=True)
class PreparedConditioning:
    conditioning: np.ndarray
    atb_valid: np.ndarray
    observability: np.ndarray


def prepare_atb_conditioning(
    atb_raw: np.ndarray,
    stats: dict[str, Any],
    band: str,
) -> PreparedConditioning:
    """Build the exact deployed six-channel input without requiring a target."""
    atb = np.asarray(atb_raw, dtype=np.float32)
    if atb.ndim != 2:
        raise ValueError(f"ATB must be a 2D array, got {atb.shape}")
    atb_valid = np.isfinite(atb) & (np.abs(atb) < FILL_ABS_THRESHOLD)
    atb_floor = max(float(stats["wavelengths"][str(band)]["atb_clip_min"]), EPS)
    atb_clean = np.where(atb_valid, np.maximum(atb, atb_floor), atb_floor).astype(np.float32)
    atb_norm = log_normalize(atb_clean, stats, band, "atb")
    observability = observability_from_atb_numpy(atb_clean, stats, band)
    observability = (observability * atb_valid.astype(np.float32)).astype(np.float32)
    height = np.linspace(0.0, 1.0, atb.shape[0], dtype=np.float32)[:, None]
    height = np.broadcast_to(height, atb.shape)
    track = np.linspace(0.0, 1.0, atb.shape[1], dtype=np.float32)[None, :]
    track = np.broadcast_to(track, atb.shape)
    wavelength = np.full(atb.shape, -1.0 if str(band) == "355" else 1.0, dtype=np.float32)
    conditioning = np.stack(
        [atb_norm, observability, atb_valid.astype(np.float32), height, track, wavelength],
        axis=0,
    ).astype(np.float32)
    if not np.isfinite(conditioning).all():
        raise ValueError("conditioning contains non-finite values after ATB-only preparation")
    return PreparedConditioning(
        conditioning=conditioning,
        atb_valid=atb_valid.astype(np.float32)[None, ...],
        observability=observability[None, ...],
    )


# Physical-floor ATB representations


def _build_six_channel_conditioning(
    atb_norm: np.ndarray,
    atb_valid: np.ndarray,
    atb_clean: np.ndarray,
    stats: dict[str, Any],
    band: str,
) -> np.ndarray:
    """Stack the six standard channels from a pre-computed atb_norm."""
    observability = observability_from_atb_numpy(atb_clean, stats, band)
    observability = (observability * atb_valid.astype(np.float32)).astype(np.float32)
    height = np.linspace(0.0, 1.0, atb_norm.shape[0], dtype=np.float32)[:, None]
    height = np.broadcast_to(height, atb_norm.shape)
    track = np.linspace(0.0, 1.0, atb_norm.shape[1], dtype=np.float32)[None, :]
    track = np.broadcast_to(track, atb_norm.shape)
    wavelength = np.full(atb_norm.shape, -1.0 if str(band) == "355" else 1.0, dtype=np.float32)
    conditioning = np.stack(
        [atb_norm, observability, atb_valid.astype(np.float32), height, track, wavelength],
        axis=0,
    ).astype(np.float32)
    if not np.isfinite(conditioning).all():
        raise ValueError("conditioning contains non-finite values after physical ATB preparation")
    return conditioning


def prepare_atb_fullrange(
    atb_raw: np.ndarray,
    stats: dict[str, Any],
    band: str,
    *,
    numerical_guard: float = PHYSICAL_ATB_MIN,
) -> PreparedConditioning:
    """Six-channel full-range logarithmic ATB representation (RawLog).

    Preserves every finite positive pre-clipping ATB value over the full
    simulator-supported range.  Uses fixed train-only log bounds from the
    stats file: lo = log10(numerical_guard), hi = log10(atb_clip_max).
    Positive ATB is never clipped upward to PHYSICAL_FLOOR or atb_clip_min.
    Nonpositive/fill values map to numerical_guard (default1e-45) before log.
    """
    atb = np.asarray(atb_raw, dtype=np.float32)
    if atb.ndim != 2:
        raise ValueError(f"ATB must be a 2D array, got {atb.shape}")
    atb_valid = np.isfinite(atb) & (np.abs(atb) < FILL_ABS_THRESHOLD)
    atb_clip_max = float(stats["wavelengths"][str(band)]["atb_clip_max"])
    lo = float(np.log10(numerical_guard))
    hi = float(np.log10(max(atb_clip_max, PHYSICAL_FLOOR * 10.0)))
    # Positive ATB: preserve as-is (no upward clip). Nonpositive/fill: map to PHYSICAL_ATB_MIN.
    atb_for_log = np.where(atb_valid & (atb > 0), atb, numerical_guard).astype(np.float32)
    atb_norm = np.clip(
        (np.log10(np.clip(atb_for_log, numerical_guard, 10.0 ** hi)) - lo) / (hi - lo),
        0.0, 1.0,
    ).astype(np.float32)
    # Observability uses max(ATB, PHYSICAL_FLOOR) for stable logistic inputs, but
    # does NOT alter the atb_norm channel.
    atb_clean = np.where(atb_valid, np.maximum(atb, PHYSICAL_FLOOR), PHYSICAL_FLOOR).astype(np.float32)
    conditioning = _build_six_channel_conditioning(
        atb_norm, atb_valid, atb_clean, stats, band,
    )
    return PreparedConditioning(
        conditioning=conditioning,
        atb_valid=atb_valid.astype(np.float32)[None, ...],
        observability=None,  # computed from atb_clean (physical floor), separate from atb_norm
    )


def signed_asinh_parameters(stats: dict[str, Any], band: str) -> tuple[float, float]:
    """Return the frozen train-only signed-asinh scale and normalization anchor.

    The scale ``s_b`` is the frozen train ATB ``atb_positive_eps`` and the
    positive anchor ``a_b`` is the frozen train ATB ``atb_clip_max``.  Both are
    in ``m^-1 sr^-1``.  This function deliberately fails closed unless the
    stats artifact carries train-split and manifest provenance; callers must
    never fit either value from the current sample, validation/real data, or BC
    labels.
    """
    if stats.get("source_split") != "train":
        raise ValueError("signed_asinh requires stats with source_split='train'")
    if not stats.get("data_manifest_hash"):
        raise ValueError("signed_asinh requires a frozen train data_manifest_hash")
    if not stats.get("split_manifest"):
        raise ValueError("signed_asinh requires split_manifest provenance")
    event_ids = stats.get("event_ids")
    if not isinstance(event_ids, list) or not event_ids:
        raise ValueError("signed_asinh requires non-empty train event_ids provenance")

    band_stats = stats["wavelengths"][str(band)]
    scale = float(band_stats["atb_positive_eps"])
    anchor = float(band_stats["atb_clip_max"])
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"signed_asinh scale must be positive finite, got {scale}")
    if not np.isfinite(anchor) or anchor <= 0.0:
        raise ValueError(f"signed_asinh anchor must be positive finite, got {anchor}")
    return scale, anchor


def signed_asinh_encode(values: np.ndarray, *, scale: float, anchor: float) -> np.ndarray:
    """Strictly monotone, reversible signed-asinh encoding in physical units.

    ``R_b(x) = asinh(x / s_b) / asinh(a_b / s_b)``.  There is intentionally
    no clipping: finite values outside ``[-a_b, a_b]`` map outside ``[-1, 1]``
    and remain invertible.  Computation is float64 so every accepted float32
    ATB value has a finite representation.
    """
    scale = float(scale)
    anchor = float(anchor)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("signed_asinh scale must be positive and finite")
    if not np.isfinite(anchor) or anchor <= 0.0:
        raise ValueError("signed_asinh anchor must be positive and finite")
    values64 = np.asarray(values, dtype=np.float64)
    denominator = np.arcsinh(anchor / scale)
    encoded = np.arcsinh(values64 / scale) / denominator
    if not np.isfinite(encoded).all():
        raise ValueError("signed_asinh encoding produced non-finite values")
    return encoded


def signed_asinh_decode(encoded: np.ndarray, *, scale: float, anchor: float) -> np.ndarray:
    """Exact inverse of :func:`signed_asinh_encode` before output rounding."""
    scale = float(scale)
    anchor = float(anchor)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("signed_asinh scale must be positive and finite")
    if not np.isfinite(anchor) or anchor <= 0.0:
        raise ValueError("signed_asinh anchor must be positive and finite")
    encoded64 = np.asarray(encoded, dtype=np.float64)
    decoded = scale * np.sinh(encoded64 * np.arcsinh(anchor / scale))
    if not np.isfinite(decoded).all():
        raise ValueError("signed_asinh decoding produced non-finite values")
    return decoded


def signed_asinh_metadata(stats: dict[str, Any], bands: Sequence[str]) -> dict[str, Any]:
    """Machine-readable transform and frozen train-stat provenance."""
    per_band: dict[str, Any] = {}
    for band in bands:
        scale, anchor = signed_asinh_parameters(stats, str(band))
        per_band[str(band)] = {
            "scale_s_b": scale,
            "anchor_a_b": anchor,
            "scale_stats_key": "atb_positive_eps",
            "anchor_stats_key": "atb_clip_max",
        }
    return {
        "formula": SIGNED_ASINH_FORMULA,
        "inverse_formula": "s_b * sinh(r * asinh(a_b / s_b))",
        "unit": SIGNED_ASINH_UNIT,
        "negative_policy": SIGNED_ASINH_NEGATIVE_POLICY,
        "source_split": stats["source_split"],
        "split_manifest": stats["split_manifest"],
        "data_manifest_hash": stats["data_manifest_hash"],
        "train_event_count": len(stats["event_ids"]),
        "per_band": per_band,
    }


def prepare_atb_signed_asinh(
    atb_raw: np.ndarray,
    stats: dict[str, Any],
    band: str,
) -> PreparedConditioning:
    """Six-channel signed full-range control (Asinh).

    Channel 0 uses the reversible signed-asinh representation with scale and
    anchor read only from the frozen train stats.  Finite negative sensor
    excursions retain their sign; zero maps exactly to zero.  Invalid/fill
    values map to zero and remain identified by channel 2.  Channels 1--5 use
    the same floor-stabilized ATB-only construction as AdaIR-FullRange, so the
    model graph and every non-primary input channel are unchanged.
    """
    atb = np.asarray(atb_raw, dtype=np.float32)
    if atb.ndim != 2:
        raise ValueError(f"ATB must be a 2D array, got {atb.shape}")
    atb_valid = np.isfinite(atb) & (np.abs(atb) < FILL_ABS_THRESHOLD)
    scale, anchor = signed_asinh_parameters(stats, band)
    atb_signed = np.where(atb_valid, atb, 0.0).astype(np.float64)
    atb_norm = signed_asinh_encode(atb_signed, scale=scale, anchor=anchor).astype(np.float32)
    atb_clean = np.where(atb_valid, np.maximum(atb, PHYSICAL_FLOOR), PHYSICAL_FLOOR).astype(np.float32)
    conditioning = _build_six_channel_conditioning(
        atb_norm, atb_valid, atb_clean, stats, band,
    )
    return PreparedConditioning(
        conditioning=conditioning,
        atb_valid=atb_valid.astype(np.float32)[None, ...],
        observability=None,
    )


def prepare_atb_physical_clip(
    atb_raw: np.ndarray,
    stats: dict[str, Any],
    band: str,
    *,
    physical_floor: float = PHYSICAL_FLOOR,
) -> PreparedConditioning:
    """Six-channel sensor-floor-clipped ATB (Clip).

    Clips ATB at max(ATB, physical_floor) before log-normalization.
    Observability uses the fixed PHYSICAL_FLOOR=1e-7.
    """
    atb = np.asarray(atb_raw, dtype=np.float32)
    if atb.ndim != 2:
        raise ValueError(f"ATB must be a 2D array, got {atb.shape}")
    atb_valid = np.isfinite(atb) & (np.abs(atb) < FILL_ABS_THRESHOLD)
    atb_clip_max = float(stats["wavelengths"][str(band)]["atb_clip_max"])
    atb_clean = np.where(atb_valid, np.maximum(atb, physical_floor), physical_floor).astype(np.float32)
    lo = float(np.log10(physical_floor))
    hi = float(np.log10(max(atb_clip_max, physical_floor * 10.0)))
    atb_norm = np.clip(
        (np.log10(np.clip(atb_clean, physical_floor, 10.0 ** hi)) - lo) / (hi - lo),
        0.0, 1.0,
    ).astype(np.float32)
    # Encoding-floor sensitivity keeps the observability input fixed.
    observability_atb = np.where(atb_valid, np.maximum(atb, PHYSICAL_FLOOR), PHYSICAL_FLOOR).astype(np.float32)
    conditioning = _build_six_channel_conditioning(
        atb_norm, atb_valid, observability_atb, stats, band,
    )
    return PreparedConditioning(
        conditioning=conditioning,
        atb_valid=atb_valid.astype(np.float32)[None, ...],
        observability=None,
    )


# --- Physical-floor SDC channel (uses F=1e-7, NOT the simulator atb_clip_min) ---


def physical_sdc_channel(
    atb_raw: np.ndarray,
    *,
    physical_floor: float = PHYSICAL_FLOOR,
    numerical_guard: float = PHYSICAL_ATB_MIN,
) -> np.ndarray:
    """Subfloor-deficit channel relative to PHYSICAL_FLOOR (F=1e-7 m^-1 sr^-1).

    D = 0                                              if ATB_raw >= F
    D = [log10(F) - log10(max(ATB_raw, guard))] / log10(F/guard)
                                                       if 0 < ATB_raw < F

    Clipped to [0, 1]. Nonpositive and fill values map to zero.
    """
    atb = np.asarray(atb_raw, dtype=np.float32)
    valid_finite = np.isfinite(atb) & (np.abs(atb) < FILL_ABS_THRESHOLD)
    # Only positive ATB below the physical floor carries deficit information
    subfloor = valid_finite & (atb > 0) & (atb < physical_floor)
    clipped = np.clip(atb, numerical_guard, physical_floor)
    deficit_span = float(np.log10(physical_floor) - np.log10(numerical_guard))
    deficit = np.where(
        subfloor,
        (np.log10(physical_floor) - np.log10(clipped)) / deficit_span,
        0.0,
    )
    return np.clip(deficit, 0.0, 1.0).astype(np.float32)


def physical_null_channel(atb_norm_ch0: np.ndarray) -> np.ndarray:
    """Null-copy seventh channel: duplicates the clipped ATB norm channel.

    Equal-capacity control for PhysicalSDC (Null).
    """
    return np.asarray(atb_norm_ch0, dtype=np.float32).copy()


def prepare_sample_arrays(
    atb_raw: np.ndarray,
    target_raw: np.ndarray,
    stats: dict[str, Any],
    band: str,
) -> PreparedSample:
    atb = np.asarray(atb_raw, dtype=np.float32)
    target = np.asarray(target_raw, dtype=np.float32)
    if atb.shape != target.shape or atb.ndim != 2:
        raise ValueError(f"ATB/target must be matched 2D arrays, got {atb.shape} and {target.shape}")
    prepared_conditioning = prepare_atb_conditioning(
        atb,
        stats,
        band,
    )
    atb_valid = prepared_conditioning.atb_valid[0].astype(bool)
    target_finite = np.isfinite(target) & (np.abs(target) < FILL_ABS_THRESHOLD)
    target_valid = atb_valid & target_finite & (target > 0)
    target_floor = max(float(stats["wavelengths"][str(band)]["bc_clip_min"]), EPS)
    target_clean = np.where(target_finite & (target > 0), target, target_floor).astype(np.float32)
    target_norm = log_normalize(target_clean, stats, band, "bc")
    return PreparedSample(
        conditioning=prepared_conditioning.conditioning,
        target_norm=target_norm[None, ...],
        target_beta=target_clean[None, ...],
        target_valid=target_valid.astype(np.float32)[None, ...],
        atb_valid=atb_valid.astype(np.float32)[None, ...],
        observability=prepared_conditioning.observability,
    )
