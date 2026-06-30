import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from generate_ablation_barcodes import gap_pixels  # noqa: E402


class GapPixelEncodingTest(unittest.TestCase):
    def test_non_overlapping_gap_boundaries(self):
        cases = {
            0: 0,
            1: 0,
            10: 0,
            11: 1,
            100: 1,
            101: 2,
            1000: 2,
            1001: 3,
            10000: 3,
            10001: 4,
            100000: 4,
            100001: 5,
        }

        for distance, expected_pixels in cases.items():
            with self.subTest(distance=distance):
                self.assertEqual(gap_pixels(distance), expected_pixels)


if __name__ == "__main__":
    unittest.main()
