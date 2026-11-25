"""Render the same six UKMO profiles with the ATB and BC COSP executables."""

import argparse
import json
import os
from pathlib import Path
import re
from shutil import copyfile
import subprocess


def render_binary(executable, branch, output, template):
    config = output / f"{branch}_input.nml"
    text, count = re.subn(r"(?m)^\s*FOUTPUT\s*=.*$", f"  FOUTPUT='{branch}.nc',", template)
    if count != 1:
        raise ValueError("Input namelist must declare one FOUTPUT")
    config.write_text(text, encoding="utf-8")
    with (output / f"{branch}.log").open("x", encoding="utf-8") as log:
        subprocess.run(
            [str(executable), config.name, "output.nml"],
            cwd=output,
            env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=120,
        )
    log_text = (output / f"{branch}.log").read_text(encoding="utf-8", errors="replace")
    if re.search(r"ERROR|FAILURE", log_text, flags=re.IGNORECASE):
        raise ValueError(f"COSP reported a failure in {branch}.log")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ukmo-input", type=Path, required=True)
    parser.add_argument("--atb-executable", type=Path, required=True)
    parser.add_argument("--bc-executable", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        import numpy as np
        import xarray as xr

        metadata = Path(__file__).resolve().parent
        template = (metadata / "input.nml").read_text(encoding="utf-8")
        with xr.open_dataset(args.ukmo_input, decode_times=False, decode_timedelta=False) as source:
            if source.sizes.get("point", 0) < 6 or source.sizes.get("level") != 50 or "height" not in source:
                raise ValueError("Expected a complete point-mode UKMO input with 50 levels")
            subset = source.isel(point=slice(0, 6)).load()

        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=False)
        subset.to_netcdf(output / "input.nc")
        copyfile(metadata / "output.nml", output / "output.nml")
        render_binary(args.atb_executable.resolve(strict=True), "atb", output, template)
        render_binary(args.bc_executable.resolve(strict=True), "bc", output, template)

        with xr.open_dataset(output / "atb.nc", decode_times=False) as atb, xr.open_dataset(
            output / "bc.nc", decode_times=False
        ) as bc:
            for coordinate in ("longitude", "latitude", "lev"):
                if not np.array_equal(atb[coordinate].values, bc[coordinate].values):
                    raise ValueError(f"ATB and BC grids disagree for {coordinate}")
            heights = subset.height.transpose("level", "point").values[::-1]
            if not np.array_equal(atb.lev.values, heights[:, 0]):
                raise ValueError("COSP output levels disagree with the input heights")

            paired = xr.Dataset({"height_m": (("level", "point"), heights, {"units": "m"})})
            report = {}
            for wavelength in (355, 532):
                signal = f"atb{wavelength}"
                x, y = atb[signal].values.squeeze(), bc[signal].values.squeeze()
                if x.shape != (50, 6) or y.shape != x.shape:
                    raise ValueError(f"Expected 50 levels by six columns at {wavelength} nm")
                valid = np.isfinite(x) & np.isfinite(y) & (np.abs(x) < 1e20) & (np.abs(y) < 1e20)
                exceedances = int((x[valid] > y[valid] * (1 + 1e-5) + 1e-12).sum())
                if not valid.all() or exceedances:
                    raise ValueError(f"Matched finite support or ATB ≤ BC check failed at {wavelength} nm")
                report[str(wavelength)] = {
                    "finite_cells": int(valid.sum()),
                    "atb_greater_than_bc_cells": exceedances,
                    "atb_min": float(x.min()),
                    "atb_max": float(x.max()),
                    "bc_min": float(y.min()),
                    "bc_max": float(y.max()),
                }
                paired[signal] = (("level", "point"), x, {"units": "m-1 sr-1"})
                paired[f"bc{wavelength}"] = (("level", "point"), y, {"units": "m-1 sr-1"})
            paired.to_netcdf(output / "paired_example.nc")

        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
    except (ImportError, KeyError, OSError, ValueError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"COSP example failed. {exc}\n")


if __name__ == "__main__":
    main()
