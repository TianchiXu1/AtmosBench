"""Verify the reported reserved-test table and its analysis population."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ReproduceTablesTest(unittest.TestCase):
    def test_paper_table(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results"
            subprocess.run([sys.executable, str(ROOT / "code/scripts/reproduce_tables.py"),
                            "--output", str(output)], check=True, capture_output=True, text=True)
            simulation = json.loads((output / "simulation.json").read_text())
            scores = simulation["reported_seed_event_scores"]
            self.assertEqual(len(scores), 184)
            self.assertEqual({int(row["seed"]) for row in scores if row["backbone"] == "AdaIR"},
                             {20260710, 20260714, 20260718})

            expected = {
                "physical_sdc": ((40.18, 0.32), (34.12, 2.24), (190.07, 2.76)),
                "fullrange": ((45.90, 0.62), (35.14, 2.61), (199.66, 6.97)),
                "physical_clip": ((242.87, 11.01), (74.79, 1.28), (325.04, 3.76)),
                "physical_null": ((233.66, 14.29), (73.87, 0.78), (325.02, 0.51)),
                "signed_asinh": ((105.39, 8.16), (57.63, 3.75), (271.22, 6.75)),
            }
            rows = {(row["representation"], row["metric"]): row
                    for row in simulation["paired_seed_display"] if row["backbone"] == "AdaIR"}
            for representation, metrics in expected.items():
                for (metric, scale), (mean, sd) in zip((('log10_mae', 1e3), ('linear_mae', 1e6),
                                                        ('linear_rmse', 1e6)), metrics):
                    row = rows[representation, metric]
                    self.assertEqual(round(float(row["mean"]) * scale, 2), mean)
                    self.assertEqual(round(float(row["sample_sd"]) * scale, 2), sd)


if __name__ == "__main__":
    unittest.main()
