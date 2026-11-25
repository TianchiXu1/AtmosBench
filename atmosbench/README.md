---
pretty_name: AtmosBench
tags:
  - earth-observation
  - lidar
  - atmospheric-science
  - inverse-problems
  - synthetic-data
---

# AtmosBench

AtmosBench is an event-disjoint benchmark for retrieving atmospheric backscatter coefficient (BC) from attenuated total backscatter (ATB). It contains 36,288 aligned input–target pairs from 48 ERA5-driven WRF events at 355 and 532 nm. Standard COSP produces the ATB input. A zero-optical-depth COSP branch applied to the same atmospheric state produces the BC target.

## Data and splits

| Split | Events | Pairs at 355 nm | Pairs at 532 nm | Total pairs |
| --- | ---: | ---: | ---: | ---: |
| Train | 40 | 15,120 | 15,120 | 30,240 |
| Validation | 4 | 1,512 | 1,512 | 3,024 |
| Test | 4 | 1,512 | 1,512 | 3,024 |
| Total | 48 | 18,144 | 18,144 | 36,288 |

[The split manifest](split_manifest.csv) assigns complete events to each partition. Validation uses E04, E16, E28 and E40. The reserved test events are E02, E14, E26 and E48, with training and checkpoint selection restricted to their respective splits in the reported PhysicalSDC experiments. An earlier model evaluated these four events before the PhysicalSDC study. [The event manifest](events.csv) records the domain centers, initialization dates, sector assignments, curtain orientations and array directories. [The pair inventory](pair_inventory.csv) lists the exact ATB and BC paths and the corresponding item ID for each pair.

Each event contributes 18 hourly WRF states at forecast hours 13 through 30 and 21 curtains per state at each wavelength. Each input and target is a `200 × 200` NumPy `float32` array in m⁻¹ sr⁻¹. The row coordinate is height at 100 m spacing, from 50 to 19,950 m. Columns cover approximately 200 km at 1 km spacing. The `H` and `V` entries in `cut_dir` indicate tracks along longitude and latitude, respectively. Each pair has the same basename in `npy/<wavelength>_x` for ATB and `npy/<wavelength>_y` for BC, including the `atb` prefix in target filenames. The evaluator enumerates paired basenames in lexical order per event and wavelength to assign item IDs such as `E02_355_0000`.

Input validity is defined from finite ATB with absolute value below `1e30`. Finite subfloor and negative ATB values remain available to the input encoding. Supervised validity also requires finite positive BC below `1e30` and is used for training and scoring. The model consumes one ATB wavelength and ATB-derived conditioning fields. The simulator's BC target and target-validity mask are for supervision and evaluation.

## Array archives

The archive format uses one TAR file per event under `data/train/`, `data/validation/` or `data/test/`. Each shard expands into `arrays/<event directory>/npy/`. A complete dataset has 72,576 array files totaling 11,621,449,728 bytes before archiving. `archives.csv` records each shard's size and SHA-256 hash. `checksums.csv` records SHA-256 hashes for the original array bytes and paths relative to `arrays/`.

Follow the [asset guide](../ASSETS.md) to download and place the archives. Run these commands from the directory containing `verify_dataset.py` and the archive manifests, using Python 3.10 or later. Verify the shard SHA-256 hashes before extraction. The test-only example extracts four event shards and verifies their 6,048 array files. For all splits, verify the archive and array manifests without `--split test` and extract every shard listed in `archives.csv`.

```bash
python verify_dataset.py --metadata-only
python verify_dataset.py --archives archives.csv --split test
for event in E02 E14 E26 E48; do tar -xf "data/test/$event.tar"; done
python verify_dataset.py --arrays arrays --split test --checksums checksums.csv
```

To generate archives from an existing canonical array tree, choose a fresh output directory. The packaging command preserves array bytes and native pair basenames. It accepts `npy/` and `proc/npy/` event layouts and writes the documented `arrays/<event directory>/npy/` layout to each shard.

```bash
python pack_dataset.py --arrays /path/to/arrays --output /path/to/atmosbench-shards
```

## Provenance and use

AtmosBench supports controlled comparisons of atmospheric retrieval methods at the two modeled wavelengths. It covers 48 events from 2022–2025 across four latitude bands and twelve longitude sectors. The synthetic pairs inherit WRF and COSP assumptions. Scientific comparisons on satellite and airborne instruments use independent observational products with distinct reference roles. The [source and acquisition guide](SOURCES.md) identifies these products and describes historical reconstruction limits.

The [benchmark data license](LICENSE-DATA.md) is CC BY 4.0. When citing the benchmark, cite Tianchi Xu and colleagues, *The Floor Is Not the Limit: Representing Physics-Aligned Weak LiDAR Returns*, and include the dataset version used.
