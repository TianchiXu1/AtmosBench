# Data and model assets

Use the [latest camera-ready source release](https://github.com/TianchiXu1/AtmosBench/releases/latest). Download the paired arrays from the [AtmosBench dataset release v1.0.0](https://github.com/TianchiXu1/AtmosBench/releases/tag/v1.0.0) and the three PhysicalSDC checkpoints from the [weight release v1.1.0](https://github.com/TianchiXu1/AtmosBench/releases/tag/v1.1.0). The [reproducibility guide](code/REPRODUCIBILITY.md) maps experiments to the released inputs and paper results.

| Asset | Release | Contents | Local destination |
| --- | --- | --- | --- |
| AtmosBench native arrays | v1.0.0 | `E01.tar` through `E48.tar`, containing 36,288 ATB–BC pairs in 72,576 NumPy files | Download to `atmosbench/data/<split>/` and extract in `atmosbench/` |
| AtmosBench archive metadata | v1.0.0 | `archives.csv` and `checksums.csv`, with archive paths, byte sizes and SHA-256 hashes | `atmosbench/` |
| Model weights | v1.1.0 | Three AdaIR PhysicalSDC `.pt` files listed in the [checkpoint roster](code/assets/checkpoints.csv) | `code/checkpoints/` |
| Checkpoint integrity | v1.1.0 | `checkpoint_checksums.csv`, with byte sizes and SHA-256 hashes for the released inference checkpoints | `code/checkpoints/` |
| Independent observational data | Provider archives | Provider products and granule rosters | See the [source and acquisition guide](atmosbench/SOURCES.md) |

## Download and verify test data

Run from the repository root. This selects the four test events. Set `atmosbench_split=train` for training, `atmosbench_split=validation` for validation or `atmosbench_split=all` for all 48 events. Test-only downloads are about 1 GB. A complete download plus extraction needs about 24 GB of storage.

```bash
(
  set -eu
  cd atmosbench
  atmosbench_url=https://github.com/TianchiXu1/AtmosBench/releases/download/v1.0.0
  atmosbench_split=test
  curl -fL --retry 3 -O "$atmosbench_url/archives.csv"
  curl -fL --retry 3 -O "$atmosbench_url/checksums.csv"

  while IFS=, read -r event split path bytes sha256; do
    [ "$event" = EID ] && continue
    if [ "$atmosbench_split" = all ] || [ "$split" = "$atmosbench_split" ]; then
      mkdir -p "$(dirname "$path")"
      curl -fL --retry 3 -o "$path.part" "$atmosbench_url/$event.tar"
      mv "$path.part" "$path"
    fi
  done < archives.csv

  python verify_dataset.py --archives archives.csv --split "$atmosbench_split"
  while IFS=, read -r event split path bytes sha256; do
    [ "$event" = EID ] && continue
    if [ "$atmosbench_split" = all ] || [ "$split" = "$atmosbench_split" ]; then
      tar -xf "$path"
    fi
  done < archives.csv
  python verify_dataset.py --arrays arrays --split "$atmosbench_split" --checksums checksums.csv
)
```

The [data card](atmosbench/README.md) describes the array format. The [paired-file inventory](atmosbench/pair_inventory.csv) lists the canonical ATB and BC paths.

## Download PhysicalSDC weights

The three released AdaIR PhysicalSDC checkpoints correspond to the matched runs in the paper's reserved-test comparison. Each checkpoint uses the epoch selected on validation data for that run.

| Model ID | Seed | Selected epoch | File |
| --- | ---: | ---: | --- |
| `adair_physicalsdc_run1` | 20260710 | 9 | `adair_physicalsdc_run1.pt` |
| `adair_physicalsdc_run2` | 20260714 | 8 | `adair_physicalsdc_run2.pt` |
| `adair_physicalsdc_run3` | 20260718 | 7 | `adair_physicalsdc_run3.pt` |

Run from the repository root to download and verify the three checkpoints.

```bash
(
  set -eu
  mkdir -p code/checkpoints
  cd code/checkpoints
  atmosbench_url=https://github.com/TianchiXu1/AtmosBench/releases/download/v1.1.0
  curl -fL --retry 3 -O "$atmosbench_url/checkpoint_checksums.csv"
  while IFS=, read -r model checkpoint bytes digest original_digest seed epoch; do
    [ "$model" = model_id ] && continue
    curl -fL --retry 3 -o "$checkpoint.part" "$atmosbench_url/$checkpoint"
    printf '%s  %s\n' "$digest" "$checkpoint.part" | shasum -a 256 -c -
    mv "$checkpoint.part" "$checkpoint"
  done < checkpoint_checksums.csv
)
```

The release files contain the selected trained state dictionaries and inference metadata. `checkpoint_checksums.csv` gives the SHA-256 of each download and, where recorded, the SHA-256 of the original training file. The [checkpoint roster](code/assets/checkpoints.csv) identifies the three available models. The additional model configurations in `code/configs/evaluation.json` document paper experiments and can be evaluated with a compatible checkpoint obtained separately. See the [paper results](results/README.md) for reported baseline and ablation summaries, the [code guide](code/README.md) for inference and evaluation, and the [data and weights license](atmosbench/LICENSE-DATA.md) for reuse terms.
