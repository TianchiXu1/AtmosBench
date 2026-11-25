#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""37-level ERA5 pressure (CDS nc) + surface (zarr nc) -> WPS intermediate.
Usage: python convert_era5_to_wps.py PL_NC[,PL_NC2,...] SL_NC OUTDIR
PL_NC may be a comma-separated list of pressure-level files; they are
concatenated along the time dimension (e.g. day1 + day2 boundary times).
"""
import argparse
import os
import numpy as np
G = 9.80665
SOIL_LEVS = ['000007','007028','028100','100289']
def relative_humidity_magnus(t2,d2):
    a,b=17.625,243.04; tc,dc=t2-273.15,d2-273.15
    return np.clip(100*np.exp((a*dc/(b+dc))-(a*tc/(b+tc))),0,100)
def time_dimension(ds):
    for d in ("valid_time","time"):
        if d in ds.dims: return d
    raise KeyError('ERA5 input must contain valid_time or time')
def open_pressure_levels(spec):
    import xarray as xr

    files=[f for f in spec.split(',') if f]
    if len(files)==1:
        return xr.open_dataset(files[0])
    parts=[xr.open_dataset(f) for f in files]
    tn=time_dimension(parts[0])
    ds=xr.concat(parts, dim=tn)
    ds=ds.sortby(tn)
    # drop duplicate times if any overlap
    _,uniq=np.unique(ds[tn].values, return_index=True)
    return ds.isel({tn: np.sort(uniq)})
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pressure_levels', help='NetCDF path or comma-separated list of pressure-level paths')
    parser.add_argument('surface', help='Surface-level NetCDF path')
    parser.add_argument('output_dir')
    args = parser.parse_args()

    import pywinter.winter as pyw
    import xarray as xr
    import pandas as pd

    plnc,slnc,outdir=args.pressure_levels,args.surface,args.output_dir
    os.makedirs(outdir,exist_ok=True)
    pds=open_pressure_levels(plnc); sds=xr.open_dataset(slnc)
    pt,st=time_dimension(pds),time_dimension(sds)
    plname="pressure_level" if "pressure_level" in pds.coords else "isobaricInhPa"
    plevs=pds[plname].values.astype(float)*100.0  # hPa->Pa (WPS intermediate requires Pa)
    lat=pds.latitude.values; lon=pds.longitude.values
    asc=lat[::-1]; dlat=abs(lat[1]-lat[0]); dlon=abs(lon[1]-lon[0])
    geo=pyw.Geo0(float(asc[0]),float(lon[0]),float(dlat),float(dlon))
    def flip(a): return a[...,::-1,:]
    # surface times -> dict
    stimes={pd.Timestamp(v).strftime("%Y-%m-%d_%H"):i for i,v in enumerate(sds[st].values)}
    for i in range(pds[pt].size):
        t=pd.Timestamp(pds[pt].values[i]); tag=t.strftime("%Y-%m-%d_%H")
        if tag not in stimes:
            print("skip (no surface):",tag); continue
        pp=pds.isel({pt:i}); ss=sds.isel({st:stimes[tag]})
        TT=flip(pp['t'].values); RH=flip(pp['r'].values)
        UU=flip(pp['u'].values); VV=flip(pp['v'].values); GHT=flip(pp['z'].values)/G
        f3=[pyw.V3dp('GHT',GHT,plevs),pyw.V3dp('TT',TT,plevs),pyw.V3dp('RH',RH,plevs),
            pyw.V3dp('UU',UU,plevs),pyw.V3dp('VV',VV,plevs)]
        sp=flip(ss['sp'].values); msl=flip(ss['msl'].values); t2=flip(ss['t2m'].values); d2=flip(ss['d2m'].values)
        u10=flip(ss['u10'].values); v10=flip(ss['v10'].values); skt=flip(ss['skt'].values); sst=flip(ss['sst'].values)
        lsm=flip(ss['lsm'].values); zsfc=flip(ss['z'].values)/G; snow=flip(ss['sd'].values)*1000.0
        sst=np.where(np.isfinite(sst),sst,skt)
        f2=[pyw.V2d('PSFC',sp),pyw.V2d('PMSL',msl),pyw.V2d('SKINTEMP',skt),pyw.V2d('SST',sst),
            pyw.V2d('SOILHGT',zsfc),pyw.V2d('LANDSEA',lsm),pyw.V2d('SNOW',snow),
            pyw.V2d('TT',t2),pyw.V2d('RH',relative_humidity_magnus(t2,d2)),pyw.V2d('UU',u10),pyw.V2d('VV',v10)]
        stt=np.stack([flip(ss[f'stl{j}'].values) for j in (1,2,3,4)],axis=0)
        smm=np.stack([flip(ss[f'swvl{j}'].values) for j in (1,2,3,4)],axis=0)
        fsoil=[pyw.Vsl('ST',stt,SOIL_LEVS),pyw.Vsl('SM',smm,SOIL_LEVS)]
        pyw.cinter('ERA5',tag,geo,f3+f2+fsoil,outdir)
        print(f"wrote ERA5:{tag} ({len(plevs)} plevs)",flush=True)
    pds.close()
    sds.close()
if __name__=="__main__": main()
