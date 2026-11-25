#!/usr/bin/env python3
"""Prepare a WRF event or render an AtmosBench same-state ATB/BC pair."""
import argparse
import csv
from datetime import datetime, timedelta
import os
from pathlib import Path
import re
from string import Template
import subprocess
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent.parent / "atmosbench"
WRF_PREFIX = "wrfout_d01_"


def load_event(path, eid):
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["EID"] == eid:
                return row
    raise ValueError(f"Unknown event {eid} in {path}")


def wrf_config(args):
    event = load_event(args.events, args.event)
    start = datetime.strptime(event["start_date"], "%Y-%m-%d_%HZ")
    end = start + timedelta(hours=30)
    parameters = dict(zip(("SY", "SM", "SD", "SHH"), start.strftime("%Y %m %d %H").split()))
    parameters.update(zip(("EY", "EM", "ED", "EH"), end.strftime("%Y %m %d %H").split()))
    parameters.update(REF_LAT=event["ref_lat"], REF_LON=event["ref_lon"],
                      GEOG_PATH=str(args.geography.resolve()), DT=str(args.time_step), NMET="38")
    args.output.mkdir(parents=True)
    for file, template in [("namelist.wps", "wps.nml.in"), ("namelist.input", "wrf.nml.in")]:
        text = Template((ROOT / template).read_text()).substitute(parameters)
        (args.output / file).write_text(text)


def render_pair(tile, work, atb, bc):
    import xarray as xr

    template = (ROOT / "cosp_input.nml").read_text()
    with xr.open_dataset(tile, decode_times=False) as ds:
        sizes = dict(ds.sizes)
    (work / "input.nc").symlink_to(tile.resolve())
    values = {"NPOINTS": sizes["lat"] * sizes["lon"], "NLEVELS": sizes["level"]}
    for key, value in values.items():
        template = re.sub(rf"(?m)^[ \t]*{key}[ \t]*=.*$", f"  {key}={value},", template)
    (work / "output.nml").write_text((ROOT / "cosp_output.nml").read_text())
    for branch, binary in (("atb", atb), ("bc", bc)):
        namelist = re.sub(r"(?m)^[ \t]*FOUTPUT[ \t]*=.*$", f"  FOUTPUT='{branch}.nc',", template)
        (work / f"{branch}.nml").write_text(namelist)
        subprocess.run([str(binary.resolve()), f"{branch}.nml", "output.nml"], cwd=work, check=True)
        if not (work / f"{branch}.nc").is_file():
            raise ValueError(f"COSP {branch} did not produce a NetCDF for {tile.name}")
    with xr.open_dataset(work / "atb.nc", decode_times=False) as x, xr.open_dataset(
        work / "bc.nc", decode_times=False
    ) as y:
        for coordinate in ("longitude", "latitude", "lev"):
            if not x[coordinate].equals(y[coordinate]):
                raise ValueError(f"ATB and BC grids disagree on {coordinate} for {tile.name}")
        for band in (355, 532):
            field = f"atb{band}"
            if x[field].sizes != y[field].sizes or x[field].dims != y[field].dims:
                raise ValueError(f"ATB and BC signal layouts disagree at {band} nm for {tile.name}")


def cosp_example(args):
    subprocess.run([sys.executable, str(DATA / "generation" / "render_pair.py"),
                    "--ukmo-input", str(args.input), "--atb-executable", str(args.cosp_atb),
                    "--bc-executable", str(args.cosp_bc), "--output", str(args.output)], check=True)


def render_event(args):
    directory = args.event_dir.resolve()
    event = load_event(args.events, args.event or directory.name.split("_", 1)[0])
    cut = event["cut_dir"]
    if args.cut is not None and args.cut != cut:
        raise ValueError(f"Manifest specifies cut direction {cut} for {event['EID']}")
    output = directory / "proc"
    if output.exists():
        raise FileExistsError(f"Output already exists at {output}")
    start = datetime.strptime(event["start_date"], "%Y-%m-%d_%HZ")
    frames = [directory / "wrf_run" / f"{WRF_PREFIX}{start + timedelta(hours=hour):%Y-%m-%d_%H:%M:%S}"
              for hour in range(13, 31)]
    for frame in frames:
        if not frame.is_file():
            raise ValueError(f"Missing retained WRF frame {frame.name} in {directory / 'wrf_run'}")
    with TemporaryDirectory(prefix=".atmosbench-render-", dir=directory) as temp:
        staging = Path(temp)
        inputs = staging / "wrf_run"
        inputs.mkdir()
        for frame in frames:
            (inputs / frame.name).symlink_to(frame)
        proc = staging / "proc"
        env = dict(os.environ, ATMOSBENCH_EVENT_DIR=str(staging), CUT_DIR=cut,
                   NLAT_FORCE="200", NLON_FORCE="200", N_CURTAIN="21", TARGET_ALONG="200", TRIM_CELLS="10")
        subprocess.run([sys.executable, str(ROOT / "prepare_cosp_input.py")], env=env, check=True)
        subprocess.run([sys.executable, str(ROOT / "interpolate_cosp_grid.py"),
                        str(proc / "cosp_input"), str(proc / "cosp_grid"), "200", "200"], env=env, check=True)
        tiles = sorted((proc / "cosp_grid").glob("*.nc"))
        if len(tiles) != len(frames):
            raise ValueError(f"Expected {len(frames)} COSP tiles, got {len(tiles)}")
        for branch in ("atb", "bc"):
            (proc / f"out_{branch}").mkdir()
        for tile in tiles:
            with TemporaryDirectory(dir=proc) as tile_temp:
                work = Path(tile_temp)
                render_pair(tile, work, args.cosp_atb, args.cosp_bc)
                for branch in ("atb", "bc"):
                    (work / f"{branch}.nc").rename(proc / f"out_{branch}" / f"cosp2_{tile.name}")
        for branch in ("atb", "bc"):
            subprocess.run([sys.executable, str(ROOT / "extract_curtains.py"), "--branch", branch], env=env, check=True)
        for band in (355, 532):
            x, y = [{file.name for file in (proc / f"npy/{band}_{side}").glob("*.npy")} for side in ("x", "y")]
            if len(x) != 378 or x != y:
                raise ValueError(f"Expected 378 paired curtains at {band} nm")
        if output.exists():
            raise FileExistsError(f"Output already exists at {output}")
        proc.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    wrf = commands.add_parser("wrf-config")
    wrf.add_argument("--events", type=Path, default=DATA / "events.csv")
    wrf.add_argument("--event", required=True)
    wrf.add_argument("--geography", type=Path, required=True)
    wrf.add_argument("--time-step", type=int, choices=(3, 4, 6), default=6)
    wrf.add_argument("--output", type=Path, required=True)
    wrf.set_defaults(run=wrf_config)
    case = commands.add_parser("cosp-example")
    case.add_argument("--input", type=Path, required=True)
    case.add_argument("--output", type=Path, required=True)
    case.set_defaults(run=cosp_example)
    event = commands.add_parser("render-event")
    event.add_argument("--event-dir", type=Path, required=True)
    event.add_argument("--events", type=Path, default=DATA / "events.csv")
    event.add_argument("--event", help="Event ID, inferred from an event directory named EID or EID_...")
    event.add_argument("--cut", choices=("H", "V"), help="Optional check against the manifest cut direction")
    event.set_defaults(run=render_event)
    for command in (case, event):
        command.add_argument("--cosp-atb", type=Path, required=True)
        command.add_argument("--cosp-bc", type=Path, required=True)
    args = parser.parse_args()
    try:
        args.run(args)
    except (ImportError, KeyError, OSError, ValueError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"AtmosBench generation failed. {exc}\n")


if __name__ == "__main__":
    main()
