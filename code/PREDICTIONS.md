# Score your own retrieval model

Generate BC predictions from the canonical ATB inputs using any model or runtime. The scorer applies the checkpoint evaluator's valid-cell mask. Log MAE uses reference BC clipped to wavelength-specific bounds estimated on the training events. Linear MAE and RMSE use untruncated valid BC in physical units.

| Field | Format |
| --- | --- |
| Filename | `<item_id>.npy`, such as `E02_355_0000.npy`. Use the exact IDs from the [pair inventory](../atmosbench/pair_inventory.csv). |
| Array | Real floating-point NumPy array of shape `200 × 200`, bottom-up as stored in the native benchmark arrays. Scoring uses float32, matching checkpoint evaluation. |
| Value | Positive finite BC in `m⁻¹ sr⁻¹` on the benchmark's scoring support. |
| Coverage | One prediction for each pair in the selected validation or test split. |

After [downloading the canonical pairs](../ASSETS.md), run from the repository root:

```bash
python code/scripts/score_predictions.py --split test \
  --data-root atmosbench/arrays --predictions predictions/ \
  --output scores.json
```

The output includes log₁₀ MAE, linear MAE and linear RMSE per event and wavelength, and the equal-weight macro over event–wavelength cells. The scorer requires no model ID or checkpoint. To exercise the interface on the eight included test pairs, extract `examples/benchmark_sample.zip`, pass the extracted directory as `--data-root`, and add `--smoke`. Use the first pair per test event and wavelength (`0000`). Sample scores describe those eight pairs.
