"""AtmosBench split enumeration and train-only input normalization."""
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "atmosbench.normalization.v1"
HEIGHT_KM = 20.0
DEFAULT_VISIBLE_THRESHOLD = 0.5
EVENT_DIR_RE = re.compile(r"^(E\d+)_")
SPLIT_ALIASES = {
    "train": "train",
    "val": "validation",
    "validation": "validation",
    "test": "test",
}


def canonical_split(split: str) -> str:
    try:
        return SPLIT_ALIASES[split]
    except KeyError as exc:
        raise ValueError(f"unknown split {split!r}; expected one of {sorted(SPLIT_ALIASES)}") from exc


@dataclass(frozen=True)
class SampleSpec:
    item_id: str
    eid: str
    split: str
    wavelength_nm: str
    x_path: str
    y_path: str
    frame_idx: int | None = None
    curtain_idx: int | None = None
    case_id: str | None = None
    cut_dir: str | None = None
    source_manifest: str | None = None


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def event_id_from_dir(path: str | Path) -> str:
    name = Path(path).name
    match = EVENT_DIR_RE.match(name)
    if not match:
        raise ValueError(f"cannot parse event id from directory name: {name}")
    return match.group(1)


def load_split_manifest(path: str | Path) -> dict[str, dict[str, str]]:
    rows = read_csv_rows(path)
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        eid = row.get("EID") or row.get("event") or row.get("eid")
        if not eid:
            raise ValueError(f"manifest row lacks EID/event: {row}")
        if eid in out:
            raise ValueError(f"duplicate event id in split manifest: {eid}")
        row["split"] = canonical_split(row["split"])
        out[eid] = row
    return out


def discover_event_dirs(data_root: str | Path, *, strict: bool = False) -> dict[str, Path]:
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"data_root does not exist: {root}")
    out: dict[str, Path] = {}
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        try:
            eid = event_id_from_dir(child)
        except ValueError:
            continue
        if strict and eid in out:
            raise ValueError(f"Multiple directories match event {eid}: {out[eid].name}, {child.name}")
        out[eid] = child
    return out


def _band_dir(event_dir: Path, band: str, xy: str) -> Path:
    direct = event_dir / "npy" / f"{band}_{xy}"
    if direct.exists():
        return direct
    generated = event_dir / "proc" / "npy" / f"{band}_{xy}"
    if generated.exists():
        return generated
    return direct


def _paired_files(event_dir: Path, band: str, *, strict: bool = False) -> list[tuple[Path, Path]]:
    x_dir = _band_dir(event_dir, band, "x")
    y_dir = _band_dir(event_dir, band, "y")
    x_files = {p.name: p for p in x_dir.glob("*.npy") if p.is_file()}
    y_files = {p.name: p for p in y_dir.glob("*.npy") if p.is_file()}
    if strict and x_files.keys() != y_files.keys():
        raise ValueError(f"Unpaired files in {event_dir.name} at {band} nm: "
                         f"{len(x_files.keys() - y_files.keys())} ATB-only, "
                         f"{len(y_files.keys() - x_files.keys())} BC-only")
    names = sorted(set(x_files) & set(y_files))
    return [(x_files[name], y_files[name]) for name in names]


def iter_pairs(
    *,
    data_root: str | Path,
    split_manifest_csv: str | Path,
    band: str,
    split: str,
    max_items: int | None = None,
    max_items_per_event: int | None = None,
    allow_missing_events: bool = False,
    strict: bool = False,
) -> list[SampleSpec]:
    split_rows = load_split_manifest(split_manifest_csv)
    event_dirs = discover_event_dirs(data_root, strict=strict)
    wanted_split = canonical_split(split)
    specs: list[SampleSpec] = []
    for eid, row in sorted(split_rows.items()):
        if row["split"] != wanted_split:
            continue
        if eid not in event_dirs:
            if allow_missing_events:
                continue
            raise FileNotFoundError(f"event {eid} missing under {data_root}")
        event_pairs = _paired_files(event_dirs[eid], str(band), strict=strict)
        if max_items_per_event is not None:
            event_pairs = event_pairs[: max(0, int(max_items_per_event))]
        for idx, (x_path, y_path) in enumerate(event_pairs):
            specs.append(
                SampleSpec(
                    item_id=f"{eid}_{band}_{idx:04d}",
                    eid=eid,
                    split=wanted_split,
                    wavelength_nm=str(band),
                    x_path=str(x_path),
                    y_path=str(y_path),
                    source_manifest=str(split_manifest_csv),
                )
            )
            if max_items is not None and len(specs) >= max_items:
                return specs
    if not specs:
        raise ValueError(f"no AtmosBench samples for split={wanted_split} band={band} under {data_root}")
    return specs


def load_top_down_array(path: str | Path, *, height: int = 200, width: int = 200) -> np.ndarray:
    arr = np.load(path).astype(np.float32, copy=False)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D array at {path}, got shape={arr.shape}")
    if arr.shape[0] < height or arr.shape[1] < width:
        raise ValueError(f"array {path} shape={arr.shape} smaller than requested {height}x{width}")
    top = 0 if arr.shape[0] == height else (arr.shape[0] - height) // 2
    left = 0 if arr.shape[1] == width else (arr.shape[1] - width) // 2
    cropped = arr[top : top + height, left : left + width]
    return cropped[::-1, :].copy()


def load_stats_json(path: str | Path) -> dict[str, Any]:
    stats = json.loads(Path(path).read_text())
    if stats.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unexpected stats schema_version={stats.get('schema_version')}")
    if stats.get("source_split") != "train":
        raise ValueError(f"normalization stats source_split must be train, got {stats.get('source_split')}")
    return stats


def stats_for_band(stats: dict[str, Any], band: str) -> dict[str, Any]:
    try:
        return stats["wavelengths"][str(band)]
    except KeyError as exc:
        raise KeyError(f"stats JSON lacks wavelength {band}") from exc


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def observability_from_atb_numpy(atb: np.ndarray, stats: dict[str, Any], band: str) -> np.ndarray:
    b = stats_for_band(stats, band)
    rule = b["mask_rule_stats"]
    eps = float(rule["positive_eps"])
    log_clip_hi = float(rule["log_atb_p99_9"])
    log_atb = np.log10(np.clip(atb.astype(np.float32, copy=False), eps, 10.0**log_clip_hi))
    # Fixed 3x3 Gaussian-like smoothing without scipy.
    padded = np.pad(log_atb, 1, mode="edge")
    smooth = (
        padded[:-2, :-2]
        + 2 * padded[:-2, 1:-1]
        + padded[:-2, 2:]
        + 2 * padded[1:-1, :-2]
        + 4 * padded[1:-1, 1:-1]
        + 2 * padded[1:-1, 2:]
        + padded[2:, :-2]
        + 2 * padded[2:, 1:-1]
        + padded[2:, 2:]
    ) / 16.0
    noise_floor = float(rule["high_altitude_noise_median"])
    noise_mad = max(float(rule["high_altitude_noise_mad"]), 1e-6)
    signal_score = (smooth - noise_floor) / noise_mad
    signal_prob = _sigmoid_np((signal_score - float(rule["signal_threshold"])) / float(rule["signal_temperature"]))
    denom = max((float(rule["log_atb_p99_9"]) - float(rule["log_atb_p50"])) / 8.0, 1e-6)
    attenuator_prob = _sigmoid_np((smooth - float(rule["log_atb_p90"])) / denom)
    dz_km = HEIGHT_KM / max(atb.shape[0] - 1, 1)
    tau_proxy = np.cumsum(attenuator_prob, axis=0) * dz_km
    observability = signal_prob * np.exp(-tau_proxy / max(float(rule["tau_proxy_p95"]), 1e-6))
    return np.clip(observability, 0.0, 1.0).astype(np.float32)


def visible_mask_from_observability(observability: Any, stats: dict[str, Any], band: str) -> Any:
    threshold = float(stats_for_band(stats, band)["mask_rule_stats"].get("visible_threshold", DEFAULT_VISIBLE_THRESHOLD))
    return observability >= threshold
