"""Verify scoring support and native orientation on canonical sample pairs."""
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from zipfile import ZipFile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class ScorePredictionsTest(unittest.TestCase):
    def test_canonical_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arrays, predictions = root / "arrays", root / "predictions"
            predictions.mkdir()
            with ZipFile(ROOT / "examples/benchmark_sample.zip") as sample:
                sample.extractall(arrays)
            with (ROOT / "atmosbench/pair_inventory.csv").open(newline="") as inventory:
                for row in csv.DictReader(inventory):
                    target = arrays / row["bc_relative_path"]
                    if target.is_file():
                        shutil.copyfile(target, predictions / f"{row['item_id']}.npy")
            self.assertEqual(len(list(predictions.glob("*.npy"))), 8)
            output = root / "scores.json"
            command = [sys.executable, str(ROOT / "code/scripts/score_predictions.py"),
                       "--split", "test", "--data-root", str(arrays), "--predictions", str(predictions),
                       "--output", str(output), "--smoke"]
            subprocess.run(command, check=True, capture_output=True, text=True)
            scores = json.loads(output.read_text())
            self.assertEqual(scores["items"], 8)
            self.assertEqual(scores["cell_macro"]["linear_mae"], 0)
            self.assertEqual(scores["cell_macro"]["linear_rmse"], 0)

            invalid = predictions / "E02_355_0000.npy"
            np.save(invalid, np.zeros((200, 200), dtype=np.float32))
            output.unlink()
            failed = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("Expected positive finite BC", failed.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
