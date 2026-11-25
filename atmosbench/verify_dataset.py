"""Verify the AtmosBench event rosters, array layout and optional hashes."""

import argparse
from ast import literal_eval
from collections import Counter, defaultdict
import csv
from datetime import datetime, timedelta
import hashlib
import json
from os import fstat
from pathlib import Path
import re


METADATA = Path(__file__).resolve().parent
WAVELENGTHS = (355, 532)
EVENT_COUNTS = {"train": 40, "validation": 4, "test": 4}
PAIRS_PER_WAVELENGTH = 378
CURTAINS_PER_HOUR = 21
RETAINED_HOURS = range(13, 31)
ARRAY_BYTES = 160_128


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def load_events(metadata=METADATA):
    events = read_rows(metadata / "events.csv")
    partitions = read_rows(metadata / "split_manifest.csv")
    event_ids = {f"E{index:02d}" for index in range(1, 49)}
    for name, rows in (("events.csv", events), ("split_manifest.csv", partitions)):
        if len(rows) != 48 or {row["EID"] for row in rows} != event_ids:
            raise ValueError(f"{name} must contain exactly the 48 unique event IDs")

    splits = {row["EID"]: row["split"] for row in partitions}
    if Counter(splits.values()) != EVENT_COUNTS:
        raise ValueError("Expected a 40/4/4 event partition")
    expected_held_out = {
        "validation": {"E04", "E16", "E28", "E40"},
        "test": {"E02", "E14", "E26", "E48"},
    }
    for split, expected in expected_held_out.items():
        if {eid for eid, value in splits.items() if value == split} != expected:
            raise ValueError(f"Unexpected {split} event IDs")

    centers = Counter(row["center_selection"] for row in events)
    if centers != {"ERA5_moisture_refined": 34, "sector_center": 14}:
        raise ValueError("Unexpected production center selection counts")
    for event in events:
        expected_directory = f"{event['EID']}_{event['region_tag']}_{event['start_date'].removesuffix('Z')}"
        if event["array_directory"] != expected_directory or not re.fullmatch(
            r"E\d{2}_B[0-3]L\d{2}_\d{4}-\d{2}-\d{2}_\d{2}", expected_directory
        ):
            raise ValueError(f"Unexpected event directory for {event['EID']}")
        event["split"] = splits[event["EID"]]
    return sorted(events, key=lambda row: row["EID"])


def event_files(root, event, inventory=None):
    event_directory = root / event["array_directory"]
    directory = event_directory / "npy"
    if not directory.is_dir():
        directory = event_directory / "proc" / "npy"
    start = datetime.strptime(event["start_date"], "%Y-%m-%d_%HZ")
    expected_hours = {
        (start + timedelta(hours=hour)).strftime("%Y%m%d%H"): CURTAINS_PER_HOUR
        for hour in RETAINED_HOURS
    }
    files = []
    first_band_curtains = None
    for wavelength in WAVELENGTHS:
        atb_dir, bc_dir = (directory / f"{wavelength}_{role}" for role in ("x", "y"))
        atb = {path.name for path in atb_dir.glob("*.npy") if path.is_file()}
        bc = {path.name for path in bc_dir.glob("*.npy") if path.is_file()}
        if atb != bc or len(atb) != PAIRS_PER_WAVELENGTH:
            raise ValueError(
                f"{event['EID']} at {wavelength} nm has {len(atb & bc)} paired, "
                f"{len(atb - bc)} ATB-only and {len(bc - atb)} BC-only files. "
                f"Expected {PAIRS_PER_WAVELENGTH} paired files."
            )
        observed_hours = Counter()
        band_curtains = set()
        for basename in sorted(atb):
            parsed = re.fullmatch(rf"atb{wavelength}_(T(\d{{10}})_[A-Za-z0-9.\-_]+\.npy)", basename)
            if parsed is None:
                raise ValueError(f"Unexpected array basename {basename}")
            curtain, hour = parsed.groups()
            observed_hours[hour] += 1
            band_curtains.add(curtain)
            files.extend((atb_dir / basename, bc_dir / basename))
        if observed_hours != expected_hours:
            raise ValueError(f"Expected {CURTAINS_PER_HOUR} curtains at each retained hour for {event['EID']} at {wavelength} nm")
        if first_band_curtains is None:
            first_band_curtains = band_curtains
        elif band_curtains != first_band_curtains:
            raise ValueError(f"The 355 and 532 nm curtain locations or times differ for {event['EID']}")
    observed = {path for path in event_directory.rglob("*.npy") if path.is_file()}
    if observed != set(files):
        raise ValueError(f"Unexpected arrays under {event_directory.name}")
    if inventory is not None and {canonical_array_path(root, path) for path in files} != inventory[event["EID"]]:
        raise ValueError(f"Array filenames differ from the canonical pair inventory for {event['EID']}")
    return files


def canonical_array_path(root, path):
    relative = path.relative_to(root)
    parts = relative.parts
    if parts[1] == "proc":
        relative = Path(parts[0], *parts[2:])
    return relative.as_posix()


def read_pair_inventory(path, events):
    if not path.is_file():
        return None
    roster = {event["EID"]: event for event in events}
    pairs = defaultdict(list)
    paths = defaultdict(set)
    for row in read_rows(path):
        eid, band, item = row["EID"], row["wavelength_nm"], row["item_id"]
        event = roster[eid]
        if row["split"] != event["split"] or band not in {"355", "532"}:
            raise ValueError(f"Invalid split or wavelength for {item} in {path.name}")
        x, y = row["atb_relative_path"], row["bc_relative_path"]
        match = re.fullmatch(rf"{re.escape(event['array_directory'])}/npy/{band}_x/(atb{band}_T\d{{10}}_[A-Za-z0-9.\-_]+\.npy)", x)
        if match is None or y != f"{event['array_directory']}/npy/{band}_y/{match[1]}" or x in paths[eid]:
            raise ValueError(f"Invalid or duplicated pair for {item} in {path.name}")
        paths[eid].update((x, y))
        pairs[eid, band].append((x, item))
    if len(pairs) != len(events) * len(WAVELENGTHS):
        raise ValueError(f"{path.name} must contain both wavelengths for every event")
    for (eid, band), rows in pairs.items():
        if len(rows) != PAIRS_PER_WAVELENGTH or any(
            item != f"{eid}_{band}_{index:04d}" for index, (_, item) in enumerate(sorted(rows))
        ):
            raise ValueError(f"Invalid canonical item IDs for {eid} at {band} nm in {path.name}")
    return paths


def read_array(path, read_body=False):
    with path.open("rb") as stream:
        preamble = stream.read(10)
        if preamble[:8] != b"\x93NUMPY\x01\x00":
            raise ValueError(f"{path} uses an unexpected NPY header version")
        if int.from_bytes(preamble[8:], "little") != 118 or fstat(stream.fileno()).st_size != ARRAY_BYTES:
            raise ValueError(f"{path} does not have the expected byte layout")
        raw_header = stream.read(118)
        try:
            header = literal_eval(raw_header.decode("ascii").strip())
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"{path} contains an invalid NPY header") from exc
        if header != {"shape": (200, 200), "descr": "<f4", "fortran_order": False}:
            raise ValueError(f"{path} must be a C-order 200x200 little-endian float32 array")
        if read_body:
            contents = preamble + raw_header + stream.read()
            if len(contents) != ARRAY_BYTES:
                raise ValueError(f"{path} changed while it was being read")
            return contents
    return None


def read_checksums(path, events):
    rows = read_rows(path)
    checksums = {row["path"]: row["sha256"] for row in rows}
    count = 48 * len(WAVELENGTHS) * PAIRS_PER_WAVELENGTH * 2
    if len(checksums) != len(rows) or len(checksums) != count:
        raise ValueError(f"Checksum manifest must contain {count} unique array paths and SHA-256 digests")
    directories = {row["array_directory"] for row in events}
    histogram = Counter()
    for path, digest in checksums.items():
        relative = re.fullmatch(r"([^/]+)/npy/(355|532)_([xy])/atb(355|532)_T\d{10}_[A-Za-z0-9.\-_]+\.npy", path)
        if (relative is None or relative[1] not in directories or relative[2] != relative[4]
                or re.fullmatch(r"[a-f0-9]{64}", digest) is None):
            raise ValueError(f"Invalid checksum entry for {path}")
        histogram[relative[1], relative[2], relative[3]] += 1
    if len(histogram) != len(events) * len(WAVELENGTHS) * 2 or set(histogram.values()) != {PAIRS_PER_WAVELENGTH}:
        raise ValueError("Checksum manifest does not contain the full roster for every event and channel")
    return checksums


def archive_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_archives(manifest, events, split):
    rows = read_rows(manifest)
    by_id = {row["EID"]: row for row in rows}
    if len(rows) != len(events) or set(by_id) != {event["EID"] for event in events}:
        raise ValueError("Archive manifest must have one shard per event")
    verified = 0
    for event in events:
        eid, event_split = event["EID"], event["split"]
        row = by_id[eid]
        expected_path = f"data/{event_split}/{eid}.tar"
        if row["path"] != expected_path or row["split"] != event_split or re.fullmatch(r"[a-f0-9]{64}", row["sha256"]) is None:
            raise ValueError(f"Invalid archive entry for {eid}")
        size = int(row["bytes"])
        if size <= 0:
            raise ValueError(f"Invalid archive size for {eid}")
        if split != "all" and split != event_split:
            continue
        shard = manifest.parent / expected_path
        if shard.stat().st_size != size or archive_digest(shard) != row["sha256"]:
            raise ValueError(f"Size or SHA-256 mismatch for {expected_path}")
        verified += 1
    return verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrays", type=Path, default=METADATA / "arrays")
    parser.add_argument("--split", choices=("all", *EVENT_COUNTS), default="all")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--metadata-only", action="store_true")
    mode.add_argument("--archives", type=Path, help="Verify TAR shards listed in archives.csv before extraction")
    parser.add_argument("--checksums", type=Path, help="Verify against the published checksums.csv")
    args = parser.parse_args()
    if (args.metadata_only or args.archives) and args.checksums:
        parser.error("--checksums requires array verification")

    try:
        all_events = load_events()
        inventory = read_pair_inventory(METADATA / "pair_inventory.csv", all_events)
        events = [row for row in all_events if args.split == "all" or row["split"] == args.split]
        counts = Counter(row["split"] for row in events)
        result = {
            "event_counts": dict(counts),
            "pair_counts": {split: count * len(WAVELENGTHS) * PAIRS_PER_WAVELENGTH for split, count in counts.items()},
            "pair_inventory_available": inventory is not None,
            "array_files_verified": 0,
            "sha256_verified": False,
        }
        if args.archives:
            result["archives_verified"] = verify_archives(args.archives, all_events, args.split)
            result["sha256_verified"] = True
        elif not args.metadata_only:
            checksums = read_checksums(args.checksums, all_events) if args.checksums else {}
            if args.split == "all":
                expected = {row["array_directory"] for row in events}
                observed = {
                    path.name for path in args.arrays.glob("E[0-9][0-9]_*") if path.is_dir()
                }
                if expected != observed:
                    raise ValueError(f"Array root event directories differ from the manifest at {args.arrays}")
            for event in events:
                for path in event_files(args.arrays, event, inventory):
                    contents = read_array(path, read_body=bool(checksums))
                    if checksums:
                        relative_path = canonical_array_path(args.arrays, path)
                        if checksums.get(relative_path) != hashlib.sha256(contents).hexdigest():
                            raise ValueError(f"SHA-256 mismatch for {relative_path}")
                    result["array_files_verified"] += 1
            result["sha256_verified"] = bool(checksums)
        print(json.dumps(result, indent=2))
    except (KeyError, OSError, TypeError, ValueError) as exc:
        parser.exit(1, f"AtmosBench verification failed. {exc}\n")


if __name__ == "__main__":
    main()
