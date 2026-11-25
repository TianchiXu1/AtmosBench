# Reproducibility

| Task | Input and available reproduction path |
| --- | --- |
| Run the CPU quickstart | Eight paired AtmosBench simulation curtains and released checkpoint `adair_physicalsdc_run1` (epoch 9). See the [README](../README.md). |
| Evaluate the three released PhysicalSDC runs | The [fixed test split](../atmosbench/split_manifest.csv), 3,024 paired ATB–BC curtains per run, training-only normalization and the three validation-selected [checkpoints](../ASSETS.md#download-physicalsdc-weights). `scripts/evaluate.py` emits scores by event and wavelength for each run. Use `--split validation` with the validation archives for validation scores. |
| Score another retrieval method | The same canonical pairs and the method's predictions in physical units. [Prediction format and scoring](PREDICTIONS.md) defines the required inputs. |
| Recompute supported paper aggregates | Run `python code/scripts/reproduce_tables.py --output reproduced_results` from the repository root. The [results guide](../results/README.md) identifies the underlying reported scores and supported calculations. Other published summaries remain available as archived values. |
| Train a retrieval model | Training and validation archives and the [training settings](README.md#train-physicalsdc). `scripts/train.py` implements matched FP32 training, chooses the epoch on validation data and writes `best.pt`. |
| Check a registered observation | A registered NPZ conforming to [the field table](README.md#run-inference) and a compatible checkpoint. `scripts/evaluate_observation.py` scores a single registered curtain. |
| Configure a simulation event or render curtains | The [WRF–COSP guide](generation/README.md) provides the event configuration, required executables and inputs. |

## Checkpoints and analysis scope

The three downloadable AdaIR PhysicalSDC checkpoints use seeds 20260710, 20260714 and 20260718 and validation-selected epochs 9, 8 and 7, respectively. Each checkpoint was checked on the eight reserved-test pairs included in the repository. The quickstart runs this check for the first checkpoint. A full evaluation uses 3,024 pairs per checkpoint from the downloaded event archives.

The paper reports three matched seeds each for AdaIR and Log-domain U-Net, and one matched run for Gaussian U-Net. The [result guide](../results/README.md) links to the event scores for each reported cohort. Mean and sample standard deviation summarize the training runs. The paired-event intervals average runs and wavelengths within each event and resample whole events.

The [evaluation config](configs/evaluation.json) records the paper's baseline and ablation settings. Fresh inference for a baseline or ablation requires its compatible checkpoint. The training command implements the matched FP32 experiments. Reproducing the broader comparison's mixed-precision baseline training requires its original implementation and runtime settings.

The robustness table distinguishes input perturbations applied to fixed clean-trained models from the separate 1σ noise-trained comparison for PhysicalSDC, RawLog and Clip. The floor and guard table uses matched training with the specified encoding constants. The evaluation config also exposes separate encoding-only sensitivities for fixed checkpoints.

## Data generation and independent observations

Use the canonical paired arrays for numerical comparisons. The [generation guide](generation/README.md) documents the retained WRF–COSP configuration, the same-input rendering check, and the limits of recreating the dataset from source inputs.

The [observation source guide](../atmosbench/SOURCES.md#observational-references) identifies the provider products. The evaluator takes registered curtains. CALIOP CPro scores use common valid support with finite positive ATB and reference backscatter. EarthCARE CPR uses QA-valid radar detections above −35 dBZ as binary labels and continuous predicted backscatter to rank pixels for AP and AUROC. The matched-representation results average scores over orbits. Full raw-product registration and cohort reconstruction require additional records. The [results directory](../results/README.md) provides the reported cohort summaries and supporting scores.
