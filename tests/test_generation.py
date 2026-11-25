"""Regression checks for deterministic curtain timestamps and interpolation."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / 'code/generation/extract_curtains.py'
SPEC = importlib.util.spec_from_file_location('extract_curtains', SCRIPT)
EXTRACTION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXTRACTION)


class GenerationTest(unittest.TestCase):
    def test_timestamp_requires_simulation_time(self):
        self.assertEqual(
            EXTRACTION.extract_time_tag_from_name('cosp2_cosp_extract_d01_2023-02-20_13:00:00.nc'),
            'T2023022013',
        )
        self.assertEqual(EXTRACTION.extract_time_tag_from_name('cosp2_2023022013.nc'), 'T2023022013')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            branch = root / 'proc/out_atb'
            branch.mkdir(parents=True)
            (branch / 'untimed.nc').touch()
            result = subprocess.run([sys.executable, str(SCRIPT), '--branch', 'atb', '--event-dir', str(root)],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Missing simulation timestamp', result.stderr)
            self.assertFalse((root / 'proc/npy').exists())

    def test_linear_interpolation(self):
        along = np.array([0., 1., 2.])
        heights = np.array([[0., 0., 0.], [100., 120., 140.], [200., 220., 240.]])
        values = heights + 2. * along
        along_target = np.array([.5, 1.5])
        height_target = np.array([50., 150.])
        output = EXTRACTION.interpolate_2d_height_lon_varying(
            values, heights, along, height_target, along_target,
        )
        self.assertEqual(output.dtype, np.float32)
        np.testing.assert_array_equal(output, height_target[:, None] + 2. * along_target)


if __name__ == '__main__':
    unittest.main()
