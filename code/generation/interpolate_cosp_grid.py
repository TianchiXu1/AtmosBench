#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Trim WRF boundaries and interpolate COSP input to a regular grid.

The AtmosBench workflow uses 200 latitudes and 200 longitudes. The two
COSP branches must be compatible with this gridded input and point output.

Usage:
  python interpolate_cosp_grid.py IN_DIR OUT_DIR [NLAT NLON]
"""
import argparse
import os
from pathlib import Path
import numpy as np

def tile_bounds_name(lat0, lat1, lon0, lon1):
    """Build the lat_AtoB_lon_CtoD suffix used by the existing pipeline."""
    return f"lat_{int(round(lat0))}to{int(round(lat1))}_lon_{int(round(lon0))}to{int(round(lon1))}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_dir', type=Path, nargs='?', default=Path('.'))
    parser.add_argument('output_dir', type=Path, nargs='?', default=Path('cosp_grid'))
    parser.add_argument('nlat', type=int, nargs='?', default=200)
    parser.add_argument('nlon', type=int, nargs='?', default=200)
    args = parser.parse_args()
    import xarray as xr

    trim = int(os.environ.get('TRIM_CELLS', '10'))
    files = sorted(args.input_dir.glob('cosp_extract*.nc'))
    if not files:
        parser.error(f'No cosp_extract*.nc in {args.input_dir}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in files:
        with xr.open_dataset(path) as ds:
            # Remove the WRF specified/relaxation boundary region.
            if trim > 0 and ds.sizes.get('lat', 0) > 2 * trim and ds.sizes.get('lon', 0) > 2 * trim:
                ds = ds.isel(lat=slice(trim, -trim), lon=slice(trim, -trim))
            lat = ds['lat'].values
            lon = ds['lon'].values
            lat0, lat1 = float(np.nanmin(lat)), float(np.nanmax(lat))
            lon0, lon1 = float(np.nanmin(lon)), float(np.nanmax(lon))
            new_lat = np.linspace(lat0, lat1, args.nlat)
            new_lon = np.linspace(lon0, lon1, args.nlon)
            # Interpolate onto the regular rectangular grid.
            ds_i = ds.interp(lat=new_lat, lon=new_lon)
            suffix = tile_bounds_name(lat0, lat1, lon0, lon1)
            out_name = f'{path.stem}_{suffix}.nc'
            enc = {v: {"zlib": True} for v in ds_i.data_vars}
            ds_i.to_netcdf(args.output_dir / out_name, encoding=enc)
        print(f'{path.name} -> {out_name} ({args.nlat}x{args.nlon})')


if __name__ == "__main__":
    main()
