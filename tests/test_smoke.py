import os
import sys
import unittest
import numpy as np


ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from config import PORTFOLIOS
from crn import generate_common_inputs
from dispatch import simulate_replication
from experiment import run_voll_sensitivity


class TestSmoke(unittest.TestCase):
    def test_single_replication_outputs(self):
        inputs = generate_common_inputs(base_seed=42, replication=0)
        F, C, U, Y = simulate_replication(PORTFOLIOS[0], inputs)
        for val in [F, C, U, Y]:
            self.assertTrue(np.isfinite(val))
            self.assertGreaterEqual(val, 0.0)

    def test_voll_sensitivity_api(self):
        comp = {
            0: {"F": [1_000.0], "C": [2_000.0], "U": [10.0]},
            1: {"F": [1_100.0], "C": [1_900.0], "U": [8.0]},
        }
        out = run_voll_sensitivity(comp, [0, 1], [5_000, 10_000])
        self.assertIn("stable_best", out)
        self.assertEqual(len(out["voll_values"]), 2)
        self.assertIn("by_voll", out)


if __name__ == "__main__":
    unittest.main()
