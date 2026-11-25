#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download one event's ERA5 forcing through the Climate Data Store.

Usage: python download_era5_event.py REF_LAT REF_LON START_DATE OUT_DIR [NLEV]

Request IDs are persisted under OUT_DIR/.jobs so interrupted requests can be
resumed. Existing complete NetCDF files are reused. Set PL_ONLY=1 when surface
fields are downloaded separately from Earth Data Hub.
"""
import argparse
import os
import time
from datetime import datetime, timedelta


def _read_cdsapirc():
    """Load CDS credentials from CDSRC or the standard ~/.cdsapirc."""
    cfg = {}
    p = os.environ.get("CDSRC", os.path.expanduser("~/.cdsapirc"))
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line.startswith("url:"):
                cfg["url"] = line.split(":", 1)[1].strip()
            elif line.startswith("key:"):
                cfg["key"] = line.split(":", 1)[1].strip()
    return cfg


STD_TIMES = ["00:00", "06:00", "12:00", "18:00"]

PLEVS_37 = ["1","2","3","5","7","10","20","30","50","70","100","125","150","175","200",
            "225","250","300","350","400","450","500","550","600","650","700","750",
            "775","800","825","850","875","900","925","950","975","1000"]
PLEVS_19 = ["10","20","50","70","100","150","200","250","300","400","500","600",
            "700","800","850","900","925","950","1000"]
PL_VARS = ["geopotential","temperature","u_component_of_wind","v_component_of_wind",
           "relative_humidity","specific_humidity"]
SL_VARS = ["surface_pressure","mean_sea_level_pressure","2m_temperature","2m_dewpoint_temperature",
           "10m_u_component_of_wind","10m_v_component_of_wind","skin_temperature","sea_surface_temperature",
           "land_sea_mask","geopotential","snow_depth","soil_temperature_level_1","soil_temperature_level_2",
           "soil_temperature_level_3","soil_temperature_level_4","volumetric_soil_water_layer_1",
           "volumetric_soil_water_layer_2","volumetric_soil_water_layer_3","volumetric_soil_water_layer_4"]

def _status(remote):
    try:
        return remote.status
    except Exception:
        return None


def get_or_submit(client, job_dir, dataset, req, tag):
    """Resume the saved CDS request or submit a new one."""
    jf = os.path.join(job_dir, tag + ".jobid")
    remote = None
    if os.path.exists(jf):
        rid = open(jf).read().strip()
        if rid:
            try:
                remote = client.get_remote(rid)
                st = _status(remote)
                if st in ("accepted", "running", "successful"):
                    print(f"[dl] {tag} reattach {rid} status={st}", flush=True)
                else:
                    print(f"[dl] {tag} job {rid} status={st} -> resubmit", flush=True)
                    remote = None
            except Exception as e:
                print(f"[dl] {tag} stale {rid} ({str(e)[:50]}) -> resubmit", flush=True)
                remote = None
        if remote is None:
            try: os.remove(jf)
            except Exception: pass
    if remote is None:
        wait = 60
        for i in range(80):
            try:
                remote = client.submit(dataset, req)
                open(jf, "w").write(remote.request_id)
                print(f"[dl] {tag} submitted {remote.request_id}", flush=True)
                break
            except Exception as e:
                msg = str(e)
                transient = any(s in msg for s in
                                ("temporarily limited", "rejected", "429", "Too many",
                                 "queued requests", "Bad Request"))
                if transient and i < 79:
                    print(f"[dl] {tag} submit busy (try {i+1}) wait {wait}s: {msg[:70]}", flush=True)
                    time.sleep(wait); wait = min(int(wait * 1.6), 600); continue
                raise
    return remote, jf


def fetch(client, job_dir, dataset, req, tag, target):
    big = 1e6 if "pressure" in dataset else 1e5
    if os.path.exists(target) and os.path.getsize(target) > big:
        print(f"[dl] skip existing {target}", flush=True); return
    remote, jf = get_or_submit(client, job_dir, dataset, req, tag)
    # Reattach to the same request after transient connection failures.
    while True:
        st = _status(remote)
        if st == "successful":
            break
        if st == "failed":
            try: os.remove(jf)
            except Exception: pass
            raise RuntimeError(f"{tag} job failed")
        if st is None:
            print(f"[dl] {tag} poll err, reconnect in 60s", flush=True)
            time.sleep(60)
            try: remote = client.get_remote(remote.request_id)
            except Exception: pass
            continue
        time.sleep(30)
    remote.download(target)
    print(f"[dl] got {target} {round(os.path.getsize(target)/1e6,1)} MB", flush=True)
    try: os.remove(jf)
    except Exception: pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('ref_lat', type=float)
    parser.add_argument('ref_lon', type=float)
    parser.add_argument('start_date', help='UTC initialization in YYYY-MM-DD_HH format')
    parser.add_argument('output_dir')
    parser.add_argument('nlev', type=int, nargs='?', default=37, choices=(19, 37))
    args = parser.parse_args()

    import ecmwf.datastores as eds

    t0 = datetime.strptime(args.start_date, '%Y-%m-%d_%H')
    times6 = [t0 + timedelta(hours=h) for h in (0, 6, 12, 18, 24, 30)]
    by_month = {}
    for timestamp in times6:
        by_month.setdefault((timestamp.year, timestamp.month), set()).add(timestamp.day)
    area = [round(args.ref_lat + 5), round(args.ref_lon - 5),
            round(args.ref_lat - 5), round(args.ref_lon + 5)]
    levels = PLEVS_37 if args.nlev == 37 else PLEVS_19
    pressure_only = os.environ.get('PL_ONLY', '0') == '1'
    print(f'ERA5 event: center=({args.ref_lat},{args.ref_lon}) start={args.start_date} levels={args.nlev} area={area}',
          flush=True)
    job_dir = os.path.join(args.output_dir, '.jobs')
    os.makedirs(job_dir, exist_ok=True)
    client = eds.Client(**_read_cdsapirc())

    for (year, month), days in sorted(by_month.items()):
        y, m = f'{year:04d}', f'{month:02d}'
        dd = [f'{day:02d}' for day in sorted(days)]
        tag = f'{y}{m}'
        fetch(client, job_dir, 'reanalysis-era5-pressure-levels', {
            'product_type': ['reanalysis'], 'variable': PL_VARS, 'year': [y], 'month': [m], 'day': dd,
            'time': STD_TIMES, 'pressure_level': levels, 'data_format': 'netcdf',
            'download_format': 'unarchived', 'area': area}, f'pl_{tag}', f'{args.output_dir}/era5pl_{tag}.nc')
        if not pressure_only:
            fetch(client, job_dir, 'reanalysis-era5-single-levels', {
                'product_type': ['reanalysis'], 'variable': SL_VARS, 'year': [y], 'month': [m], 'day': dd,
                'time': STD_TIMES, 'data_format': 'netcdf', 'download_format': 'unarchived', 'area': area},
                f'sl_{tag}', f'{args.output_dir}/era5sl_{tag}.nc')
    print('ERA5 event download complete', flush=True)


if __name__ == '__main__':
    main()
