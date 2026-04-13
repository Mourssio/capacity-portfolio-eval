import os
import sys
import unittest


ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from config import PORTFOLIOS, N_PORTFOLIOS


class TestPortfolios(unittest.TestCase):
    def test_portfolio_count(self):
        self.assertEqual(N_PORTFOLIOS, 16)
        self.assertEqual(len(PORTFOLIOS), 16)

    def test_fraction_sum(self):
        for p in PORTFOLIOS:
            total = p.firm_frac + p.renew_frac + p.storage_frac
            self.assertAlmostEqual(total, 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
