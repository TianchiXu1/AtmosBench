#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare gridded COSP input fields from WRF model states.

Read the WRF output, compute the cloud microphysical parameters, and write
NetCDF input fields for the paired COSP branches.
"""
import argparse
import numpy as np
import os
from pathlib import Path
from effective_radius import calculate_reff

# WRF dynamics: Skamarock et al. (2019), NCAR/TN-556+STR.
# Microphysics: Morrison et al. (2005); Thompson et al. (2008).
# Cloud overlap: Hogan and Illingworth (2000).
# Surface radiation: Iacono et al. (2008), RRTMG.
# Effective-radii references are listed in effective_radius.py.
GRAVITY = 9.81  # m/s²
HYDRO_NAMES = ['Large-scale (stratiform) liquid', 'Large-scale (stratiform) ice', 'Large-scale (stratiform) rain', 'Large-scale (stratiform) snow', 'Convective liquid', 
              'Convective ice', 'Convective rain', 'Convective snow', 'Large-scale (stratiform) graupel']


def calc_phalf(pfull):
    """Calculate half-level pressures by averaging adjacent full levels."""
    phalf = np.zeros_like(pfull)
    phalf[0,:,:] = pfull[0,:,:]
    phalf[-1,:,:] = pfull[-1,:,:]
    for k in range(1, pfull.shape[0]-1):
        phalf[k,:,:] = 0.5 * (pfull[k-1,:,:] + pfull[k,:,:])
    return phalf

def calc_tca(cldfra):
    """Calculate cumulative cloud cover from each level to the top."""
    nlev = cldfra.shape[0]
    tca = np.zeros_like(cldfra)
    for k in range(nlev):
        tca[k] = 1.0 - np.prod(1.0 - cldfra[k:], axis=0)
    return tca

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--event-dir', type=Path, default=os.environ.get('ATMOSBENCH_EVENT_DIR'))
    args = parser.parse_args()
    if args.event_dir is None:
        parser.error('Specify --event-dir or set ATMOSBENCH_EVENT_DIR')

    import xarray as xr
    from tqdm import tqdm

    input_dir = args.event_dir / 'wrf_run'
    output_dir = args.event_dir / 'proc' / 'cosp_input'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    wrf_files = sorted(f for f in os.listdir(input_dir) if f.startswith("wrfout"))
    
    for wrf_file in tqdm(wrf_files):
        wrf_path = os.path.join(input_dir, wrf_file)
        output_file = os.path.join(output_dir, wrf_file.replace("wrfout", "cosp_extract") + ".nc") 
        
        if os.path.exists(output_file):
            print(f"Using existing COSP input {output_file}")
            continue

        ds = xr.open_dataset(wrf_path)
        time_idx = 0

        # Rectilinear latitude and longitude axes.
        lat = ds['XLAT'][time_idx, :, 0].values
        lon = ds['XLONG'][time_idx, 0, :].values
        nlev = len(ds['bottom_top'])

        # Retained potential-temperature field and layer order for replay.
        T_abs = (ds['T'][time_idx] + ds['T00']).isel(Time=time_idx).values[::-1]

        pfull = (ds['PB'][time_idx] + ds['P'][time_idx]).values
        
        phalf = calc_phalf(pfull)

        # Half-level heights from geopotential.
        height = (ds['PHB'][time_idx] + ds['PH'][time_idx]) / GRAVITY
        height_half = height.isel(bottom_top_stag=slice(None, -1)).values

        # Hydrometeor mixing ratios.
        qc = ds['QCLOUD'][time_idx].values
        qr = ds['QRAIN'][time_idx].values
        mr_lsliq = qc + qr
        
        qi = ds['QICE'][time_idx].values
        qs = ds['QSNOW'][time_idx].values
        qg = ds['QGRAUP'][time_idx].values
        mr_lsice = qi + qs + qg
        
        mr_ccliq = qc.copy()
        mr_ccice = qi.copy()

        cca = ds['CLDFRA'][time_idx].values
        tca = calc_tca(cca)

        emsfc_lw = ds['EMISS'][time_idx].mean().item()

        reff = calculate_reff(wrf_path, time_idx)

        times = ds['Times'][time_idx].values.astype(str)

        ds_cosp = xr.Dataset(
            data_vars={
                'mr_lsliq': (['level', 'lat', 'lon'], mr_lsliq),
                'mr_lsice': (['level', 'lat', 'lon'], mr_lsice),
                'mr_ccliq': (['level', 'lat', 'lon'], mr_ccliq),
                'mr_ccice': (['level', 'lat', 'lon'], mr_ccice),
                'T_abs': (['level', 'lat', 'lon'], T_abs),
                'pfull': (['level', 'lat', 'lon'], pfull),
                'phalf': (['level', 'lat', 'lon'], phalf),
                'height_half': (['level', 'lat', 'lon'], height_half),
                'cca': (['level', 'lat', 'lon'], cca),
                'tca': (['level', 'lat', 'lon'], tca),
                'Reff': (['hydro', 'level', 'lat', 'lon'], reff),
                'emsfc_lw': emsfc_lw,
                'Times': times
            },
            coords={
                'lat': (['lat'], lat, {'units': 'degrees_north'}),
                'lon': (['lon'], lon, {'units': 'degrees_east'}),
                'level': (['level'], np.arange(nlev), {'positive': 'down'}),
                'hydro': (['hydro'], np.arange(len(HYDRO_NAMES)))
            }
        )

        meta_config = {
            'mr_lsliq': {'units': 'kg kg-1', 'source': 'Morrison05+Milbrandt05'},
            'mr_lsice': {'units': 'kg kg-1', 'source': 'Thompson08+Mitchell96'},
            'mr_ccliq': {'units': 'kg kg-1', 'source': 'Heymsfield20 convective proxy'},
            'mr_ccice': {'units': 'kg kg-1', 'source': 'Heymsfield20 convective proxy'},
            'T_abs': {'units': 'K', 'source': 'WRF thermodynamic Eq.'},
            'pfull': {'units': 'Pa', 'source': 'PB+P perturbation'},
            'phalf': {'units': 'Pa', 'source': 'interpolated from pfull'},
            'height_half': {'units': 'm', 'source': 'PHB+PH scaled by 1/g'},
            'cca': {'units': 'fraction', 'source': 'WRF cloud scheme output'},
            'tca': {'units': 'fraction', 'source': 'Hogan00 overlap scheme'},
            'emsfc_lw': {'units': 'ratio', 'source': 'RRTMG surface emissivity'},
            # Preserve the historical custom COSP input attribute. The radius
            # array contains meter-valued numbers from calculate_reff.
            'Reff': {'units': 'μm', 'source': 'species-dependent parameterization'},
            'Times': {'units': 'UTC', 'source': 'WRF simulation time'}
        }

        for var, attrs in meta_config.items():
            ds_cosp[var].attrs.update(attrs)

        encoding = {var: {'zlib': True} for var in ds_cosp.data_vars}
        ds_cosp.to_netcdf(output_file, encoding=encoding)
        ds_cosp.close()
        ds.close()
        print(f"Wrote COSP input {output_file}")

if __name__ == '__main__':
    main()
