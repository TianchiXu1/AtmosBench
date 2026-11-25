"""Effective radii for the nine COSP hydrometeor categories."""

import numpy as np

def calculate_reff(wrf_file, time_idx=0):
    """Return effective radii in meters with shape (9, level, lat, lon)."""
    import xarray as xr

    with xr.open_dataset(wrf_file) as ds:
        t00 = ds["T00"][time_idx].values  # Base temperature (K)
        t = ds["T"][time_idx].values + t00
        p = ds["PB"][time_idx].values + ds["P"][time_idx].values
        rho = p / (287.0 * t)  # Air density (kg/m³)

        # Hydrometeor mixing ratios (kg/kg).
        qcloud = np.maximum(ds["QCLOUD"][time_idx].values, 0.0)
        qice = np.maximum(ds["QICE"][time_idx].values, 0.0)
        qrain = np.maximum(ds["QRAIN"][time_idx].values, 0.0)
        qsnow = np.maximum(ds["QSNOW"][time_idx].values, 0.0)
        qgraup = np.maximum(ds["QGRAUP"][time_idx].values, 0.0)

        # Number concentrations (kg⁻¹).
        qnice = ds["QNICE"][time_idx].values
        qnrain = ds["QNRAIN"][time_idx].values

    reff = np.zeros((9, *qcloud.shape), dtype=np.float32)

    # Stratiform cloud liquid (I_LSCLIQ=1).
    # Morrison et al. (2005); number concentration in m⁻³.
    n_liq = np.where(qcloud > 1e-6, 1e8, 1e6)
    reff_liq = (3 * rho * qcloud / np.maximum(4 * np.pi * n_liq * 1000.0, 1e-15)) ** (1/3)
    reff[0] = np.clip(reff_liq, 5e-6, 50e-6)

    # Stratiform cloud ice (I_LSCICE=2); Thompson et al. (2008).
    n_ice = np.where(qnice > 1e-6, qnice * rho, 1e4)
    reff_ice = (3 * rho * qice / np.maximum(4 * np.pi * n_ice * 917.0, 1e-15)) ** (1/3)
    reff[1] = np.clip(reff_ice, 10e-6, 200e-6)

    # Stratiform rain (I_LSRAIN=3); Milbrandt and Yau (2005).
    n_rain = np.where(qnrain > 1e-6, qnrain * rho, 8e6)
    reff_rain = (3 * rho * qrain / np.maximum(4 * np.pi * n_rain * 1000.0, 1e-15)) ** (1/3)
    reff[2] = np.clip(reff_rain, 100e-6, 3000e-6)

    # Stratiform snow (I_LSSNOW=4); mass-dimension relation (Mitchell 1996).
    base_snow = np.maximum(qsnow * 1e3, 1e-15)
    reff_snow = 300e-6 * base_snow ** 0.2
    reff[3] = np.clip(reff_snow, 50e-6, 2000e-6)

    # Convective cloud liquid (I_CVCLIQ=5); Heymsfield et al. (2020).
    reff[4] = 10e-6
    # Convective ice, rain and snow (I_CVCICE, I_CVRAIN, I_CVSNOW).
    # Scale the corresponding stratiform radii (Liu et al. 2012).
    reff[5] = reff[1] * 1.2
    reff[6] = reff[2] * 1.5
    reff[7] = reff[3] * 1.5

    # Stratiform graupel (I_LSGRPL=9); assumed density 400 kg/m³.
    rho_graup = 400.0
    n_graup = 1e4 * (qgraup * 1e3)**0.5  # Empirical power-law number concentration (m⁻³)
    reff_graup = (3 * rho * qgraup / np.maximum(4 * np.pi * n_graup * rho_graup, 1e-15)) ** (1/3)
    reff[8] = np.clip(reff_graup, 500e-6, 5000e-6)

    return reff
