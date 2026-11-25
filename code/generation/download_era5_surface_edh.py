#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download one event's ERA5 surface forcing from Earth Data Hub.

Usage: python download_era5_surface_edh.py REF_LAT REF_LON START_DATE OUT_NC

The output uses CDS variable names and matches the pressure-level domain and
six-hourly schedule from event initialization to forecast hour 30.
"""
import argparse
import datetime as dt
import os
from pathlib import Path

SURL = "https://data.earthdatahub.destine.eu/era5/reanalysis-era5-single-levels-v0.zarr"
VARS = ["sp", "msl", "t2m", "d2m", "u10", "v10", "skt", "sst", "lsm", "z", "sd",
        "stl1", "stl2", "stl3", "stl4", "swvl1", "swvl2", "swvl3", "swvl4"]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('ref_lat', type=float)
    parser.add_argument('ref_lon', type=float)
    parser.add_argument('start_date', help='UTC initialization in YYYY-MM-DD_HH format')
    parser.add_argument('output', type=Path)
    args = parser.parse_args()

    from aiohttp import BasicAuth
    import numpy as np
    import xarray as xr

    token = os.environ.get('EARTHDATAHUB_TOKEN')
    if not token:
        parser.error('Set EARTHDATAHUB_TOKEN to your Earth Data Hub access token')
    # Match the CDS request domain.
    north, south = round(args.ref_lat + 5), round(args.ref_lat - 5)
    west, east = round(args.ref_lon - 5), round(args.ref_lon + 5)
    west360, east360 = west % 360, east % 360
    t0 = dt.datetime.strptime(args.start_date, '%Y-%m-%d_%H')
    times = [np.datetime64(t0 + dt.timedelta(hours=6 * i)) for i in range(6)]
    print(f'ERA5 surface event: center=({args.ref_lat},{args.ref_lon}) start={args.start_date} '
          f'area={[north, west, south, east]}', flush=True)

    with xr.open_dataset(SURL, engine='zarr', chunks={}, decode_times=True,
                         backend_kwargs={'storage_options': {'client_kwargs': {'auth': BasicAuth('edh', token)}}}) as ds:
        # Handle a domain that crosses the 0/360-degree longitude boundary.
        if west360 <= east360:
            sub = ds[VARS].sel(latitude=slice(north, south), longitude=slice(west360, east360))
        else:
            a = ds[VARS].sel(latitude=slice(north, south), longitude=slice(west360, 359.75))
            b = ds[VARS].sel(latitude=slice(north, south), longitude=slice(0, east360))
            sub = xr.concat([a, b], dim='longitude')
        sub = sub.sel(valid_time=times, method='nearest').load()

    got = [str(v)[:13] for v in sub.valid_time.values]
    expected = [str(v)[:13] for v in times]
    if got != expected:
        raise ValueError(f'Expected timestamps {expected}; received {got}')
    if sub.latitude.size != 41 or sub.longitude.size != 41:
        raise ValueError(f'Expected a 41x41 grid; received {sub.latitude.size}x{sub.longitude.size}')
    # Convert to the CDS longitude convention and preserve ascending ordering.
    lon = sub.longitude.values.copy()
    lon = np.where(lon > 180, lon - 360, lon)
    sub = sub.assign_coords(longitude=lon)
    if not np.all(np.diff(sub.longitude.values) > 0):
        sub = sub.sortby('longitude')

    args.output.parent.mkdir(parents=True, exist_ok=True)
    enc = {v: {'zlib': True} for v in sub.data_vars}
    sub.to_netcdf(args.output, encoding=enc)
    print(f'Wrote {args.output}: {sub.latitude.size}x{sub.longitude.size} grid, '
          f'{sub.valid_time.size} timestamps, {len(sub.data_vars)} variables', flush=True)


if __name__ == '__main__':
    main()
