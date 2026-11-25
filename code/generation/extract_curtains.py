#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract ATB or BC curtains from COSP point output at 355 and 532 nm.

Branch output is read from <event>/proc/out_<branch> with matching height
profiles from <event>/proc/cosp_grid. COSP points are flattened with longitude
varying fastest. Interpolate along the height and along-track axes, and save
the canonical paired basenames in proc/npy/{355,532}_{x,y}. The BC branch
uses the same COSP variable names and basenames as ATB.
"""

import argparse
import os
import re
from pathlib import Path
import numpy as np
from scipy.interpolate import interp1d
from datetime import datetime

# variables to read (adjust if your files use different names)
VAR_NAMES = {
    'atb532': ['atb532', 'ATB532', 'atb_532', 'ATB_532'],
    'atb355': ['atb355', 'ATB355', 'atb_355', 'ATB_355'],
    'lat': ['latitude', 'lat', 'Latitude'],
    'lon': ['lon', 'longitude', 'Longitude'],
    'height': ['height_half', 'height', 'height_half(:)', 'height_half_3d']
}

# interpolation targets
TARGET_ALONG = int(os.environ.get('TARGET_ALONG', '200'))  # along-track target points
N_CURTAIN = int(os.environ.get('N_CURTAIN', '21'))          # curtains per frame
CUT_DIR = os.environ.get('CUT_DIR', 'H').upper()           # H=horizontal(sample lat, curtain along lon); V=vertical(sample lon, curtain along lat)
NLAT_FORCE = int(os.environ.get('NLAT_FORCE', '0'))        # >0 forces square-grid nlat (avoid ambiguity)
NLON_FORCE = int(os.environ.get('NLON_FORCE', '0'))

# helper functions
def find_var(ds, candidates):
    """Return first variable name in ds that matches candidates, else None"""
    for n in candidates:
        if n in ds.variables:
            return n
    return None

def extract_time_tag_from_name(fname):
    """Return the simulation timestamp as TYYYYMMDDHH from the filename."""
    base = os.path.basename(fname)
    m = re.search(r'(\d{4})[-]?(\d{2})[-]?(\d{2})[_-]?(\d{2})', base)
    if m is None:
        raise ValueError(f"Missing simulation timestamp in {base}")
    timestamp = datetime.strptime(''.join(m.groups()), '%Y%m%d%H')
    return timestamp.strftime('T%Y%m%d%H')

def infer_nlon_nlat_from_lat_flat(lat_flat):
    """
    Given lat_flat (len N), find smallest k>0 such that lat_flat[k] != lat_flat[0]
    returns (nlon, nlat)
    """
    N = lat_flat.size
    if N == 0:
        return None, None
    first = lat_flat[0]
    nlon = None
    for k in range(1, N):
        if not np.isclose(lat_flat[k], first):
            nlon = k
            break
    if nlon is None:
        nlon = N
    if N % nlon != 0:
        return None, None
    nlat = N // nlon
    return int(nlon), int(nlat)

def read_height_from_grid(grid_path, expected_nlev, expected_nlat, expected_nlon):
    """
    Read height_half (or equivalent) from the shared gridded COSP input.
    Ensure the returned array has shape (nlev, nlat, nlon). Try permutations if needed.
    """
    import xarray as xr

    if not os.path.exists(grid_path):
        raise FileNotFoundError(f"COSP grid file not found: {grid_path}")
    ds = xr.open_dataset(grid_path, mask_and_scale=False)
    hvn = find_var(ds, VAR_NAMES['height'])
    if hvn is None:
        # try common direct name
        if 'height_half' in ds.variables:
            hvn = 'height_half'
        else:
            ds.close()
            raise KeyError("Height variable not found in COSP grid file.")
    h = np.array(ds[hvn].values)
    ds.close()
    # canonicalize to (nlev, nlat, nlon)
    if h.ndim == 1:
        # unexpected 1D
        if h.size == expected_nlev:
            # broadcast to spatial dims
            return np.broadcast_to(h[:, None, None], (expected_nlev, expected_nlat, expected_nlon)).astype(np.float64)
        else:
            raise ValueError("height is 1D but length mismatch")
    if h.shape == (expected_nlev, expected_nlat, expected_nlon):
        return h.astype(np.float64)
    # try permutations to match expected sizes
    for perm in [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]:
        hp = np.transpose(h, perm)
        if hp.shape == (expected_nlev, expected_nlat, expected_nlon):
            return hp.astype(np.float64)
    # last resort: try to reduce extra dims by mean across extras
    if h.shape[0] == expected_nlev:
        # try to collapse remaining axes to match lat/lon counts by reshaping if possible
        rest = int(np.prod(h.shape[1:]))
        if rest == expected_nlat * expected_nlon:
            hp = h.reshape((expected_nlev, expected_nlat, expected_nlon))
            return hp.astype(np.float64)
    raise ValueError(f"height array shape {h.shape} cannot be matched to expected {(expected_nlev, expected_nlat, expected_nlon)}")

def ensure_strict_monotonic(x, y=None):
    """
    Ensure x is strictly monotonic (ascending). If x is descending, flip x and y along axis -1.
    If duplicates exist, remove duplicate x entries and corresponding y slices.
    Returns (x_new, y_new) where y_new may be None if input y was None.
    y is expected to have shape (..., len(x))
    """
    x = np.asarray(x)
    if x.size < 2:
        return x, y
    # remove NaNs in x
    if np.isnan(x).any():
        good = ~np.isnan(x)
        x = x[good]
        if y is not None:
            y = np.take(y, np.where(good)[0], axis=-1)
    # check monotonic direction
    diffs = np.diff(x)
    if np.all(diffs > 0):
        asc = True
    elif np.all(diffs < 0):
        asc = False
    else:
        # not strictly monotonic -> try to uniquify by taking first occurrence
        ux, idx = np.unique(x, return_index=True)
        order = np.argsort(idx)
        x = ux[order]
        if y is not None:
            y = np.take(y, idx[order], axis=-1)
        diffs = np.diff(x)
        if np.all(diffs > 0):
            asc = True
        elif np.all(diffs < 0):
            asc = False
        else:
            # give up, return as-is (interp will likely fail)
            asc = True
    if not asc:
        x = x[::-1]
        if y is not None:
            y = np.flip(y, axis=-1)
    return x, y

def interpolate_2d_height_lon_varying(slice_2d, heights_per_lon, lon_src, heights_tgt, lon_tgt, method='linear'):
    """
    Robust interpolation when vertical coordinate varies with lon.

    slice_2d: (nlev, nlon_src)
    heights_per_lon: either 1D (nlev,) or 2D (nlev, nlon_src)
    lon_src: (nlon_src,)
    heights_tgt: (nhtgt,)
    lon_tgt: (nlon_tgt,)

    Algorithm:
      - If heights_per_lon is 1D: use regular 2D interp (interp1d then interp1d)
      - If heights_per_lon is 2D: for each original lon_i, do 1D vertical interp from heights_per_lon[:,i] -> heights_tgt (result shape (nhtgt, nlon_src))
        then for each target height j, do horizontal interp across lon_src -> lon_tgt on the row j.
    Returns: (nhtgt, nlon_tgt) float32 array
    """
    nlev, nlon_src = slice_2d.shape
    nhtgt = len(heights_tgt)
    # normalize lon monotonicity
    lon_src, slice_2d = ensure_strict_monotonic(lon_src, slice_2d)
    # prepare tmp vertical-interpolated array shape (nhtgt, nlon_src)
    tmp = np.full((nhtgt, nlon_src), np.nan, dtype=np.float64)

    heights_per_lon = np.asarray(heights_per_lon)
    if heights_per_lon.ndim == 1:
        # same vertical grid for all longitudes: do vertical interp once then horizontal
        hsrc = heights_per_lon
        hsrc, _ = ensure_strict_monotonic(hsrc, None)
        # vertical interpolation for each lon: use interp1d across hsrc
        for i in range(nlon_src):
            col = slice_2d[:, i]
            try:
                hs = hsrc
                vals = col
                # remove NaN/duplicates in hs
                hs, vals = ensure_strict_monotonic(hs, vals)
                if hs.size < 2:
                    tmp[:, i] = np.nan
                else:
                    f = interp1d(hs, vals, kind=method, bounds_error=False, fill_value=np.nan, assume_sorted=True)
                    tmp[:, i] = f(heights_tgt)
            except Exception:
                tmp[:, i] = np.nan
    elif heights_per_lon.ndim == 2:
        # heights_per_lon shape (nlev, nlon_src)
        if heights_per_lon.shape[1] != nlon_src or heights_per_lon.shape[0] != nlev:
            raise ValueError("heights_per_lon shape mismatch")
        for i in range(nlon_src):
            hs = heights_per_lon[:, i]
            col = slice_2d[:, i]
            # ensure monotonic hs
            hs, col = ensure_strict_monotonic(hs, col)
            if hs.size < 2:
                tmp[:, i] = np.nan
                continue
            try:
                f = interp1d(hs, col, kind=method, bounds_error=False, fill_value=np.nan, assume_sorted=True)
                tmp[:, i] = f(heights_tgt)
            except Exception:
                tmp[:, i] = np.nan
    else:
        raise ValueError("heights_per_lon must be 1D or 2D")

    # now tmp is (nhtgt, nlon_src); horizontal interp for each height level
    # ensure lon_src monotonic (done above)
    # prepare lon_tgt monotonic
    lon_tgt_sorted = np.asarray(lon_tgt)
    if lon_tgt_sorted[0] > lon_tgt_sorted[-1]:
        lon_tgt_sorted = lon_tgt_sorted[::-1]
        # we'll flip final result afterwards if needed
        flip_final_lon = True
    else:
        flip_final_lon = False

    result = np.full((nhtgt, len(lon_tgt_sorted)), np.nan, dtype=np.float32)
    for j in range(nhtgt):
        row = tmp[j, :]
        # remove NaN in row's lon interpolation
        try:
            # ensure monotonic lon_src and row; assume lon_src already monotonic
            f_lon = interp1d(lon_src, row, kind='linear', bounds_error=False, fill_value=np.nan, assume_sorted=True)
            result[j, :] = f_lon(lon_tgt_sorted)
        except Exception:
            result[j, :] = np.nan

    if flip_final_lon:
        result = result[:, ::-1]
    return result.astype(np.float32)

def process_file(path, grid_dir, output_dirs):
    import xarray as xr

    print(f"[INFO] Processing: {path}")
    try:
        ds = xr.open_dataset(path, mask_and_scale=False)
    except Exception as e:
        print(f"[ERROR] open {path}: {e}")
        return

    # find variables
    vn_atb532 = find_var(ds, VAR_NAMES['atb532'])
    vn_atb355 = find_var(ds, VAR_NAMES['atb355'])
    vn_lat = find_var(ds, VAR_NAMES['lat'])
    vn_lon = find_var(ds, VAR_NAMES['lon'])

    if vn_lat is None or vn_lon is None:
        print(f"[WARN] lat/lon variables not found in {path}. Skipping.")
        ds.close()
        return
    if vn_atb532 is None and vn_atb355 is None:
        print(f"[WARN] neither atb532 nor atb355 found in {path}. Skipping.")
        ds.close()
        return

    lat_flat = ds[vn_lat].values
    lon_flat = ds[vn_lon].values
    N = lat_flat.size
    if N == 0:
        print(f"[WARN] empty lat array in {path}")
        ds.close()
        return

    nlon, nlat = infer_nlon_nlat_from_lat_flat(lat_flat)
    if NLAT_FORCE > 0 and NLON_FORCE > 0 and NLAT_FORCE * NLON_FORCE == N:
        nlat, nlon = NLAT_FORCE, NLON_FORCE  # explicit square-grid order (lat-major, lon fastest)
    if nlon is None or nlat is None or nlon * nlat != N:
        print(f"[ERROR] cannot infer grid dims from flattened lat (N={N}, nlon={nlon}, nlat={nlat}). Skipping.")
        ds.close()
        return

    # reconstruct per-row lon and per-row lat
    lats = lat_flat.reshape(nlat, nlon)[:, 0]   # shape (nlat,)
    lons = lon_flat.reshape(nlat, nlon)[0, :]   # shape (nlon,) take first row's lon

    # determine expected vertical level count by sampling atb
    sample_atb = None
    if vn_atb532 is not None:
        sample_atb = ds[vn_atb532].values
    elif vn_atb355 is not None:
        sample_atb = ds[vn_atb355].values

    if sample_atb is None:
        print(f"[ERROR] no atb sample found in {path}")
        ds.close()
        return

    # normalize sample_atb -> ensure shape (nlev, N)
    arr = np.array(sample_atb)
    arr = np.squeeze(arr)
    if arr.ndim == 1 and N == arr.size:
        print(f"[ERROR] atb has only 1D shape {arr.shape}. Skipping.")
        ds.close()
        return
    if arr.ndim == 2:
        nlev, NN = arr.shape
    elif arr.ndim == 3:
        # collapse singleton/time dims where appropriate to get (nlev, N)
        if arr.shape[-1] == N:
            # try to make first dim nlev
            if arr.shape[0] == N and arr.shape[-1] != N:
                arr = np.transpose(arr, (2,1,0))
            arr = arr.reshape(arr.shape[0], -1)
        else:
            arr = arr.reshape(arr.shape[0], -1)
        nlev, NN = arr.shape
    else:
        print(f"[ERROR] unexpected atb array ndim={arr.ndim} shape={arr.shape}")
        ds.close()
        return

    if NN != N:
        if arr.shape[0] == N and arr.shape[1] == nlev:
            arr = arr.T
            nlev, NN = arr.shape

    if NN != N:
        print(f"[ERROR] after normalization atb shape mismatch: expected second dim {N}, got {NN}. Skipping.")
        ds.close()
        return

    # Read height_half from the corresponding shared COSP grid file.
    basename = os.path.basename(path)
    if basename.startswith('cosp2_'):
        grid_name = basename[len('cosp2_'):]
    else:
        grid_name = basename
    grid_path = grid_dir / grid_name

    # Preserve the longitude-range suffix in canonical curtain basenames.
    lon_range_tag = ''
    m_lon = re.search(r'_lon_([+-]?\d+to[+-]?\d+)', basename, flags=re.IGNORECASE)
    if m_lon:
        lon_range_tag = m_lon.group(1)
    try:
        height_half = read_height_from_grid(grid_path, nlev, nlat, nlon)
    except Exception as e:
        print(f"[ERROR] reading height from COSP grid file {grid_path}: {e}")
        ds.close()
        return

    # Target axes: 200 vertical bins centered at 50–19,950 m.
    heights_tgt = np.arange(50, 20050, 100)  # 200 levels (0-20km / 100m)
    lon_tgt = np.linspace(np.nanmin(lons), np.nanmax(lons), TARGET_ALONG)  # H: along-lon
    lat_tgt = np.linspace(np.nanmin(lats), np.nanmax(lats), TARGET_ALONG)  # V: along-lat

    def process_band(vname, out_dir):
        vn = find_var(ds, VAR_NAMES[vname])
        if vn is None:
            print(f"[INFO] {vname} not present in {path}, skipping {vname}.")
            return
        try:
            raw = ds[vn].values
            raw = np.squeeze(raw)
            # ensure shape (nlev, N)
            if raw.ndim == 2:
                if raw.shape[1] == N and raw.shape[0] == nlev:
                    arr2 = raw
                elif raw.shape[0] == N and raw.shape[1] == nlev:
                    arr2 = raw.T
                else:
                    arr2 = raw.reshape(nlev, -1)
            elif raw.ndim == 1:
                print(f"[WARN] {vn} is 1D - skipping")
                return
            else:
                arr2 = raw.reshape(nlev, -1)
            if arr2.shape[1] != N:
                print(f"[ERROR] {vn} shape normalization failed -> {arr2.shape}. skip band.")
                return
        except Exception as e:
            print(f"[ERROR] read {vn} from {path}: {e}")
            return

        # reshape to (nlev, nlat, nlon)
        try:
            arr3 = arr2.reshape((nlev, nlat, nlon))
        except Exception as e:
            print(f"[ERROR] cannot reshape {vn} to (nlev,nlat,nlon): {e}")
            return

        time_tag = extract_time_tag_from_name(path)

        # exactly N_CURTAIN evenly-spaced sample lines on the 200 sampling axis
        n_axis = nlon if CUT_DIR == 'V' else nlat  # V: sample lon (curtain along lat); H: sample lat (curtain along lon)
        sample_idx = np.unique(np.linspace(0, n_axis - 1, N_CURTAIN).round().astype(int))
        if sample_idx.size < N_CURTAIN:  # rounding collapsed dupes (n_axis>=200 makes this rare)
            sample_idx = np.linspace(0, n_axis - 1, N_CURTAIN).astype(int)

        for s_idx in sample_idx:
            s_idx = int(s_idx)
            if CUT_DIR == 'V':
                slice_2d = arr3[:, :, s_idx]             # (nlev, nlat)
                heights_line = height_half[:, :, s_idx]  # (nlev, nlat)
                horiz_src, horiz_tgt = lats, lat_tgt
                tag = f"Lon{float(lons[s_idx]):.4f}"
            else:
                slice_2d = arr3[:, s_idx, :]             # (nlev, nlon)
                heights_line = height_half[:, s_idx, :]  # (nlev, nlon)
                horiz_src, horiz_tgt = lons, lon_tgt
                tag = f"Lat{float(lats[s_idx]):.4f}"
            if np.all(np.isnan(slice_2d)):
                print(f"[INFO] {vn} {tag} all NaN -> skip")
                continue
            try:
                interp = interpolate_2d_height_lon_varying(slice_2d, heights_line, horiz_src, heights_tgt, horiz_tgt)
            except Exception as e:
                print(f"[ERROR] interp {vn} {tag}: {e}")
                continue
            if lon_range_tag:
                fname_out = f"{vname}_{time_tag}_{tag}_Lon{lon_range_tag}.npy"
            else:
                fname_out = f"{vname}_{time_tag}_{tag}.npy"
            out_path = os.path.join(out_dir, fname_out)
            try:
                np.save(out_path, interp.astype(np.float32))
            except Exception as e:
                print(f"[ERROR] save {out_path}: {e}")
                continue
            print(f"[SAVE] {out_path} (shape {interp.shape})")

    # process both bands
    for band in (532, 355):
        process_band(f'atb{band}', output_dirs[band])

    ds.close()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--branch', choices=('atb', 'bc'), required=True)
    parser.add_argument('--event-dir', type=Path, default=os.environ.get('ATMOSBENCH_EVENT_DIR'))
    args = parser.parse_args()
    if args.event_dir is None:
        parser.error('Specify --event-dir or set ATMOSBENCH_EVENT_DIR')
    proc = args.event_dir / 'proc'
    input_dir = proc / f'out_{args.branch}'
    grid_dir = proc / 'cosp_grid'
    side = 'x' if args.branch == 'atb' else 'y'
    output_dirs = {band: proc / 'npy' / f'{band}_{side}' for band in (355, 532)}
    files = sorted(input_dir.glob('*.nc'))
    if not files:
        parser.error(f'No NetCDF files in {input_dir}')
    # Check every timestamp before writing any curtains.
    for path in files:
        extract_time_tag_from_name(path)
    for directory in output_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    for path in files:
        process_file(path, grid_dir, output_dirs)
    print("All done.")

if __name__ == '__main__':
    main()
