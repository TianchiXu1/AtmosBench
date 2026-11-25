# AtmosBench generation

The workflow follows the [48-event roster](../../atmosbench/events.csv) and the camera-ready paper. Use the canonical [AtmosBench arrays](../../atmosbench/README.md) for numerical reproduction of the benchmark. The workflow below runs the recoverable WRF setup and the paired rendering with compatible COSP executables.

## ERA5 and WRF

Set up WRF v4.3.3, WPS, and the WPS geography data. Accept the terms for both ERA5 products in the Climate Data Store. See the [source guide](../../atmosbench/SOURCES.md) for citations, variables, and attribution. Install the Python dependencies in a WRF-compatible environment.

```bash
python -m pip install numpy scipy xarray pandas netCDF4 tqdm pywinter ecmwf-datastores
```

Run the example below from the repository root. E02 initializes on 20 February 2023 at 12 UTC, centered at 45°S and 135°W. The ERA5 downloader uses `$CDSRC` or `~/.cdsapirc` for the CDS configuration. The conversion produces WPS intermediates for the 37 ERA5 pressure levels and single-level fields, including the six timestamps at forecast hours 0, 6, 12, 18, 24 and 30.

```bash
EVENT="$PWD/work/E02_B0L01_2023-02-20_12"
python code/generation/download_era5_event.py -45 -135 2023-02-20_12 "$EVENT/era5" 37
python code/generation/convert_era5_to_wps.py \
  "$EVENT/era5/era5pl_202302.nc" "$EVENT/era5/era5sl_202302.nc" "$EVENT/intermediate"
python code/generation/generate.py wrf-config --event E02 \
  --geography /path/to/WPS_GEOG --output "$EVENT/config"
```

For an event spanning two calendar months, run the conversion once for each matching pressure/surface file pair and use the same intermediate output directory. If using the Earth Data Hub route for surface fields, set `PL_ONLY=1` when running the CDS downloader, install `dask`, `fsspec`, `aiohttp` and `zarr<3`, and supply your `EARTHDATAHUB_TOKEN` to `download_era5_surface_edh.py` via the environment. Save one single-level NetCDF with the six boundary timestamps and pass it to the converter for each pressure-level file.

In a WPS working directory, use the generated `namelist.wps`, make `ERA5:*` intermediate files and the matching `GEOGRID.TBL` and `METGRID.TBL` available, and run `geogrid.exe` then `metgrid.exe`. In `<event>/wrf_run`, make the WRF run tables and `met_em.d01.*.nc` files available, copy in the generated `namelist.input`, then run `real.exe` and `wrf.exe`. Set up parallel execution according to the local WRF build. The default WRF time step is 6 s. The configuration command accepts `--time-step 4` or `--time-step 3` for numerical recovery.

## Same-state COSP and curtains

Use compatible standard ATB and zero-attenuation BC COSP executables for both wavelengths. The retained `COSPv2.0-2.1.7` configuration uses a BC branch that returns `beta` directly from `cmp_backsignal`. The [six-column UKMO check](../../atmosbench/generation/README.md) exercises both branches on the same supplied input and checks their geometry and attenuation behavior. Use the released AtmosBench arrays for benchmark scoring. The exact historical executable and WRF input mapping remain incomplete.

The retained WRF conversion writes potential temperature to `T_abs` and reverses that field's vertical order relative to the other fields. The [COSP2 reference driver](https://github.com/CFMIP/COSPv2.0/blob/v2.1.7/driver/src/cosp2_test.f90#L467-L480) expects air temperature aligned with pressure by layer. The UKMO check starts with a prepared COSP input. Validate the temperature conversion and layer alignment for any fresh WRF-derived dataset. The influence of the retained convention on the historical arrays remains unquantified.

```bash
python code/generation/generate.py render-event --event-dir "$EVENT" \
  --cosp-atb /path/to/cosp2_test_ATB --cosp-bc /path/to/cosp2_test_BC
```

The command selects WRF outputs at forecast hours 13–30 from `<event>/wrf_run`, so a complete hourly WRF output directory also works. It takes the curtain direction from the event manifest, trims ten grid cells per edge and interpolates the retained state to a 200 × 200 horizontal tile in `proc/cosp_grid/`. Both COSP branches read the same tile. Only the output filename differs between the branch namelists. `extract_curtains.py` applies the same interpolation to the ATB and BC branches, extracting 21 curtains per hour and wavelength into `<event>/proc/npy/{355,532}_{x,y}`, with 200 vertical bins at 100 m spacing and 200 along-track samples. The BC arrays use the same basenames as the corresponding ATB arrays. The command checks paired output coordinates and basenames before writing the new `proc` directory. Use a fresh event directory for a second render.

The retained COSP input convention stores effective radius values in meters while preserving the historical `Reff` NetCDF attribute `units=μm`. Confirm the numerical input convention of any replacement COSP reader rather than converting these values based on the attribute alone.

## Verify and arrange generated arrays

For evaluation with the fixed inventory, use the event's exact `array_directory` from [events.csv](../../atmosbench/events.csv), and place its `npy/` directory directly under that event directory. For the example above:

```bash
mkdir -p "$PWD/generated-arrays/$(basename "$EVENT")"
cp -R "$EVENT/proc/npy" "$PWD/generated-arrays/$(basename "$EVENT")/"
```

Generate or obtain all events in the selected split before evaluation. The test split comprises E02, E14, E26 and E48. After arranging their arrays under `generated-arrays/`, run:

```bash
python atmosbench/verify_dataset.py --arrays generated-arrays --split test
```

Use `generated-arrays` as the evaluation `--data-root` only when the verification passes. This check validates the event and pair layout against the published inventory. For reproducing the reported numeric results, use the released canonical arrays and their published verification manifests. Exact numerical generation of the original arrays is not established by the available historical COSP build records.

To package a complete 48-event array tree, use `atmosbench/pack_dataset.py --arrays generated-arrays --output /path/to/new-shards`, which requires the complete event roster and writes the canonical archive layout.

## Six-column same-state check

Select the 50-level COSP2 UKMO point-mode example `driver/data/inputs/UKMO/cosp_input_um.nc` dated 26 February 2007. The command compares the first six columns on the identical output grid at each wavelength. It writes measurements and `paired_example.nc` with explicit heights.

```bash
python code/generation/generate.py cosp-example --input /path/to/cosp_input_um.nc \
  --cosp-atb /path/to/cosp2_test_ATB --cosp-bc /path/to/cosp2_test_BC \
  --output results/cosp-example
```

The [retained check](../../results/simulation.json) records 300 finite matched cells and zero ATB > BC tolerance exceedances at each wavelength. The check uses the UKMO fixture. The benchmark uses the separate 2022–2025 WRF events.
