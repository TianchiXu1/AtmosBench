# Paper results

These files collect the released numerical summaries associated with [the paper](https://openreview.net/forum?id=5b9wzLyUdE). Each JSON key names a table or series. Values are stored as decimal text to retain their precision. See the [reproducibility guide](../code/REPRODUCIBILITY.md) for the corresponding data and commands.

| File | Contents |
| --- | --- |
| [Simulation](simulation.json) | Fixed validation and test comparisons, paired-seed statistics and their supporting reported seed/event scores, representation ablations and event-level validation reductions and the same-input COSP check |
| [Observations](observations.json) | CALIOP, EarthCARE and PACE-PAX cohort summaries and available paired intervals |
| [Robustness](robustness.json) | Frozen ATB perturbations, matched noise training, floor and numerical guard comparisons |
| [Training and cost](training_and_cost.json) | Validation curves and RTX 3090 cost measurements |

## Reproduce the reported tables

From the repository root, run the following command. It uses only the Python standard library and the released scores, so model weights and the full dataset are not needed.

```bash
python code/scripts/reproduce_tables.py --output reproduced_results
```

The command creates a new output directory with four JSON files following the table structure above. It independently recomputes and verifies the main validation, reserved-test, representation ablation, and paired-seed tables from the event scores. For the paired-seed tables, it first averages the eight event–wavelength cells for each run, then calculates the mean and sample standard deviation over the reported runs. Paired-event intervals resample the four events, jointly retaining wavelengths and analysis runs within each event.

The reserved-test AdaIR and Log-domain U-Net comparisons each summarize three matched seeds. The Gaussian U-Net comparison uses one matched run. Each checkpoint uses the epoch selected on validation data. The released event scores, means, sample standard deviations and paired-event intervals refer to the reported cohorts. See the [reproducibility guide](../code/REPRODUCIBILITY.md) for the evaluation scope.

The paper label **PhysicalSDC (Ours)** refers to retrieval with the parameter-free Physics-Aligned Encoding input module. In each matched comparison the backbone is fixed. The result records use `physical_sdc` for PhysicalSDC (Ours), `fullrange` for RawLog, `physical_clip` for Clip, `physical_null` for Null, and `signed_asinh` for Asinh. The [checkpoint manifest](../code/assets/checkpoints.csv) maps individual released weights to their run and seed.

## Supporting scores and scope

The [evidence directory](evidence/) contains the scored units needed to recompute the observational and simulation summaries.

| Evidence | Scored units |
| --- | --- |
| `simulation_event_scores.csv` | Four validation and four test events at each wavelength for the fixed-checkpoint comparison |
| `dataset_validity.json` | Corpus size, supervised-valid cell count, and finite negative input count |
| `attenuation_strata.csv` | Event–wavelength cell counts and errors for the paper's single-run attenuation analysis |
| `adair_validation_curves.csv` | Validation log MAE for the five AdaIR encodings in Figure 5(a), by epoch |
| `original_cpr_orbit_scores.csv` | Multi-architecture comparison, six co-registered ATLID–CPR orbits per method |
| `original_cpro_unit_scores.csv` | Multi-architecture comparison, 29 CALIOP CPro evaluation units per method |
| `cpr_orbit_scores.csv` | 1,298 EarthCARE orbits per matched representation |
| `cpro_paired_orbit_differences.csv` | 1,270 CALIOP orbits per control, paired with PhysicalSDC |
| `hsrl_flight_scores.csv` | Five flights for the Fernald comparison and 12 flight-days for the matched representations |
| `hsrl_fernald_shared_flight_differences.csv` | Five flights, shared successful profiles, and the specified lidar-ratio settings |

The 184 reported seed/event scores are included directly in `simulation.json`. The reproduction command also recomputes the supported original and expanded observation aggregates and paired intervals. For the expanded CPro cohort, the paired differences and intervals are independently recomputed and checked against the released absolute means. Absolute CPro means, robustness summaries, recorded runtime and training curves, and the same-input COSP check are retained as reported. The frozen input-stress series preserves both `log10_mae` and the paper's display value `reported_log10_mae_millidex`; for three rows, the display value differs by 0.01 millidex from rounding the retained metric. Complete raw-to-table reruns of these results require the data and model artifacts identified in the [reproducibility guide](../code/REPRODUCIBILITY.md).

Figure 5(a) compares five encodings with the same AdaIR backbone. PhysicalSDC (Ours) and RawLog have at least 71.9% lower validation log MAE than Clip and Null at epoch 1. Figure 5(b), (c), and (d) pair PhysicalSDC (Ours) with RawLog within PromptIR, DATPRL-IR, and Gaussian U-Net, respectively. The per-epoch scores and checkpoint selections for these three pairs are in [training_and_cost.json](training_and_cost.json).
