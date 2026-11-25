# COSP paired rendering example

The example runs both COSP branches on the first six columns of a UKMO point-mode atmospheric state dated 26 February 2007, with 50 model levels. The modified BC branch returns the lidar scattering term directly at each range bin. The standard ATB branch applies attenuation to those same atmospheric inputs. This fixture checks branch behavior. The canonical AtmosBench pairs use the 2022–2025 WRF event states.

The driver requires Python with `numpy`, `xarray` and `netCDF4`, the complete UKMO point-mode fixture at `driver/data/inputs/UKMO/cosp_input_um.nc` in a compatible COSP checkout, and runnable ATB and zero-optical-depth BC builds. Run the command from the directory containing this README. The exact historical build-to-source mapping for the customized production executables is incomplete.

```bash
python render_pair.py \
  --ukmo-input /path/to/cosp_input_um.nc \
  --atb-executable /path/to/cosp2_test_ATB \
  --bc-executable /path/to/cosp2_test_BC \
  --output /path/to/cosp-pair
```

The driver writes the shared input subset, both branch outputs, execution logs and `paired_example.nc` with ATB, BC and explicit `height_m`. It checks actual coordinates and finite support and compares ATB with BC within floating-point tolerance. The two input namelists differ in output filename only. The retained execution returned these results.

| Wavelength | Finite paired cells | ATB maximum (m⁻¹ sr⁻¹) | BC maximum (m⁻¹ sr⁻¹) | ATB > BC cells |
| ---: | ---: | ---: | ---: | ---: |
| 355 nm | 300 | 4.0451199 × 10⁻⁵ | 4.5839753 × 10⁻³ | 0 |
| 532 nm | 300 | 7.3100760 × 10⁻⁵ | 4.5784637 × 10⁻³ | 0 |

The standard routine evaluates a division by optical depth at the top level, which yielded an undefined value at exactly zero optical depth in the limiting-case check. The modified BC routine is finite there and returns the supplied scattering term directly. The two COSP branch files can both contain variables named `atb355` and `atb532`. In the paired output, the BC variables use a `bc` prefix and preserve the original branch values.
