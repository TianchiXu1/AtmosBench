# AtmosBench and PhysicalSDC

Implementation for *The Floor Is Not the Limit: Representing Physics-Aligned Weak LiDAR Returns*. We construct AtmosBench to pair attenuated total backscatter (ATB) and backscatter coefficient (BC) at 355 and 532 nm. PhysicalSDC is our single-wavelength BC retrieval method. Its parameter-free Physics-Aligned Encoding module adds the logarithmic distance below the nominal signal floor as an input channel for finite positive weak returns.

## Install

Run the commands below from `code/`. Use Python 3.10 or 3.11 and PyTorch 2.5.1. The original GPU replay used CUDA 12.1. Install the CUDA build or the matching CPU build for your system using the [PyTorch wheel instructions](https://pytorch.org/get-started/previous-versions/). Then install the pinned Python dependencies:

```bash
python -m pip install -r requirements.txt
```

The main PhysicalSDC configuration uses an AdaIR-derived retrieval backbone, included under its MIT license. The matched runs pair the same encoding with other backbones using the representation option in the training command. Optional restoration baseline setup is in [THIRD_PARTY.md](THIRD_PARTY.md).

## Train PhysicalSDC

Download both the training and validation arrays using the [asset guide](../ASSETS.md#download-and-verify-test-data). Use the fixed event manifest from the [data card](../atmosbench/README.md).

```bash
python scripts/train.py --data-root ../atmosbench/arrays --output outputs/train-sdc
```

The command selects the validation checkpoint over eight event–wavelength cells. Defaults are AdaIR, PhysicalSDC, FP32, seed 20260710, 10 epochs, AdamW, and effective batch size 12. Use `--representation fullrange` for RawLog. `python scripts/train.py --help` lists the matched controls and alternate backbones. For a bounded CPU execution check with the training and validation arrays, add `--device cpu --workers 0 --smoke` to perform one optimizer update and score eight validation pairs.

The training entry point implements the matched FP32 representation protocol. The paper's broader restoration comparison uses FP32 for PhysicalSDC and standard AdaIR and automatic mixed precision (AMP) for PromptIR, DATPRL, Log-domain U-Net, and Gaussian U-Net. The [evaluation config](configs/evaluation.json) records those settings and selected epochs. Reproducing those mixed-precision training runs requires their training implementation and runtime settings. Fresh training can yield different weights across hardware and runtime versions.

For the reported FP32 runs below, set both `--micro-batch-size` and `--validation-batch-size` to the listed value. All use an effective batch size of 12.

| Backbone and seed | Representation | Micro-batch size |
| --- | --- | ---: |
| AdaIR, 20260710 | All five arms | 2 |
| AdaIR, 20260714 | All five arms | 6 |
| AdaIR, 20260718 | PhysicalSDC (Ours) | 1 |
| AdaIR, 20260718 | Null | 2 |
| AdaIR, 20260718 | RawLog, Clip, Asinh | 6 |
| PromptIR and DATPRL, 20260710 | PhysicalSDC (Ours) and RawLog | 6 |
| Gaussian U-Net, 20260710 | PhysicalSDC (Ours) and RawLog | 12 |

## Run inference

The native array layout and split file are documented in the [AtmosBench dataset card](../atmosbench/README.md). Put the [three released PhysicalSDC checkpoints](../ASSETS.md#download-physicalsdc-weights) in `checkpoints/` using the filenames in [assets/checkpoints.csv](assets/checkpoints.csv). Download the matching validation or test data before running either evaluation command. The additional paper experiment configurations in [configs/evaluation.json](configs/evaluation.json) require compatible checkpoints obtained separately. See the [paper results](../results/README.md) for the reported comparisons with ablations and baselines.

```bash
python scripts/retrieve.py --atb native_atb.npy --band 532 \
  --model adair_physicalsdc_run1 --checkpoint checkpoints/adair_physicalsdc_run1.pt \
  --output retrieved_bc.npy --device cpu

python scripts/evaluate.py --split validation --data-root ../atmosbench/arrays \
  --split-manifest ../atmosbench/split_manifest.csv --checkpoints checkpoints \
  --models adair_physicalsdc_run1 --output results/validation.json

python scripts/evaluate.py --split test --data-root ../atmosbench/arrays \
  --split-manifest ../atmosbench/split_manifest.csv --checkpoints checkpoints \
  --models adair_physicalsdc_run1 adair_physicalsdc_run2 adair_physicalsdc_run3 \
  --output results/test.json
```

`retrieve.py` expects a native bottom-up curtain and writes BC in the same orientation and physical units. An invalid ATB cell receives `NaN` in the standalone retrieval file. Evaluation uses validation-selected weights and uses the BC targets only for scoring. A full evaluation measures 3,024 pairs across eight cells per model in either partition. The three-model command returns scores for each released run. The `--smoke` mode measures one pair per cell to check the execution path. Use `--device cpu` on systems without CUDA. CPU and accelerator arithmetic can differ slightly. GPU inference uses the precision recorded for each checkpoint.

For another registered checkpoint, add `--model MODEL_ID` to `retrieve.py`. The ID refers to its configuration entry and records the architecture, ATB encoding and selected epoch.

To retrieve with a checkpoint from `scripts/train.py`, use `--fresh-checkpoint outputs/train-sdc/best.pt` in place of the archived `--checkpoint` argument.

Use the registered observation path for top-down arrays retained after the original provider acquisition and registration process:

```bash
python scripts/evaluate_observation.py --source caliop --registered registered_case.npz \
  --model adair_physicalsdc_run1 --checkpoint checkpoints/adair_physicalsdc_run1.pt \
  --output results/observation.json
```

| Observation family | Registered NPZ fields |
|---|---|
| CALIOP | `atb_m`, `cpro_beta_m`, `cpro_valid`, and, where available, `apro_beta_m`, `apro_valid` |
| ATLID EBD | `atb_m`, `ebd_beta_m`, `ebd_valid` |
| ATLID–CPR | `atl`, `atl_valid`, `cpr`, `cpr_valid` |
| PACE-PAX R1 HSRL-2 | `atb_m`, `reference_beta_m`, `learned_common_signed_mask`, `altitude_asl_m`, `ground_altitude_asl_m` |

The observation evaluator scores one registered curtain. CALIOP comparisons share evaluation support based on valid positive ATB and valid reference retrievals. Radar AP and AUROC measure structural agreement with a common quality-screened CPR detection reference above −35 dBZ. The PACE-PAX diagnostics compare layer boundaries on the documented reference support. The [source guide](../atmosbench/SOURCES.md#observational-references) describes the provider files and limits of reproducing the registered cohorts.

## Score predictions from another model

See [Prediction format and scoring](PREDICTIONS.md) for the standalone scorer. It takes BC arrays in physical units and applies the benchmark protocol independently of the model architecture.

## Generate paired data

The [generation guide](generation/README.md) describes the event configuration, WRF-to-COSP input preparation and geometry, and the same-input UKMO example. Use the canonical paired arrays to evaluate the paper's models. The guidance for generating new arrays documents the limits of reconstructing the distributed dataset from raw inputs.

## Release scope

The source distribution includes encoders, training and evaluation entry points, generation scripts and settings, paper result records, and a table aggregation script. The [asset guide](../ASSETS.md) gives availability and expected locations for the arrays and the three author checkpoints. The [reproducibility guide](REPRODUCIBILITY.md) specifies the inputs and verification scope for each experiment.
