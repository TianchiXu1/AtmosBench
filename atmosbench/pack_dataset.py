"""Create one uncompressed AtmosBench TAR shard per event and SHA-256 manifests."""

import argparse
import csv
import hashlib
from io import BytesIO
from pathlib import Path
from shutil import copyfile
import tarfile
from tempfile import TemporaryDirectory

from verify_dataset import METADATA, archive_digest, canonical_array_path, event_files, load_events, read_array, read_pair_inventory


RELEASE_METADATA = (
    "README.md",
    "SOURCES.md",
    "events.csv",
    "split_manifest.csv",
    "verify_dataset.py",
    "pack_dataset.py",
    "generation/README.md",
    "generation/input.nml",
    "generation/output.nml",
    "generation/render_pair.py",
    "references/caliop_v4_51.csv",
    "references/caliop_v4_51_expanded.csv",
    "references/earthcare_cpr_frames.csv",
    "references/earthcare_cpr_frames_expanded.csv",
    "references/earthcare_ebd_orbits.csv",
    "references/pace_pax_hsrl2_r1.csv",
    "references/pace_pax_hsrl2_r1_expanded.csv",
)


def pack_event(arrays, files, shard, checksums):
    with tarfile.open(shard, "x", format=tarfile.USTAR_FORMAT) as archive:
        for path in files:
            relative_path = canonical_array_path(arrays, path)
            contents = read_array(path, read_body=True)
            member = tarfile.TarInfo(f"arrays/{relative_path}")
            member.size = len(contents)
            archive.addfile(member, BytesIO(contents))
            checksums.writerow((relative_path, hashlib.sha256(contents).hexdigest()))


def pack_dataset(arrays, output, events, metadata=METADATA):
    expected_events = {event["array_directory"] for event in events}
    found_events = {path.name for path in arrays.glob("E[0-9][0-9]_*") if path.is_dir()}
    if found_events != expected_events:
        raise ValueError("Source event directories must match the event manifest")
    inventory = read_pair_inventory(metadata / "pair_inventory.csv", events)
    input_files = {event["EID"]: event_files(arrays, event, inventory) for event in events}
    if output.exists():
        raise FileExistsError(f"Output already exists at {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temp:
        staging = Path(temp) / "payload"
        staging.mkdir()
        with (staging / "checksums.csv").open("x", newline="", encoding="utf-8") as file_stream:
            checksums = csv.writer(file_stream, lineterminator="\n")
            checksums.writerow(("path", "sha256"))
            archives = []
            for event in events:
                eid = event["EID"]
                split = event["split"]
                relative_shard = Path("data") / split / f"{eid}.tar"
                shard = staging / relative_shard
                shard.parent.mkdir(parents=True, exist_ok=True)
                pack_event(arrays, input_files[eid], shard, checksums)
                archives.append((eid, split, relative_shard.as_posix(), shard.stat().st_size, archive_digest(shard)))
                print(f"Packed {relative_shard}", flush=True)

        with (staging / "archives.csv").open("x", newline="", encoding="utf-8") as archive_stream:
            writer = csv.writer(archive_stream, lineterminator="\n")
            writer.writerow(("EID", "split", "path", "bytes", "sha256"))
            writer.writerows(archives)
        metadata_files = list(RELEASE_METADATA)
        if inventory is not None:
            metadata_files.append("pair_inventory.csv")
        for relative_path in metadata_files:
            target = staging / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            copyfile(metadata / relative_path, target)
        if output.exists():
            raise FileExistsError(f"Output already exists at {output}")
        staging.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory for release shards and manifests")
    args = parser.parse_args()
    arrays = args.arrays.resolve()
    output = args.output.resolve()
    if output.is_relative_to(arrays):
        parser.error("Choose an output directory outside the source arrays")

    try:
        pack_dataset(arrays, output, load_events())
    except (KeyError, OSError, ValueError, tarfile.TarError) as exc:
        parser.exit(1, f"AtmosBench archive creation failed. {exc}\n")


if __name__ == "__main__":
    main()
