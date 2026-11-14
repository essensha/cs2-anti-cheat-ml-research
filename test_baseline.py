import tempfile
import unittest
from pathlib import Path
from generate_synthetic_data import generate
from run_baseline import run

class BaselineTests(unittest.TestCase):
    def test_deterministic_generation(self):
        a = generate(1200, 42)
        b = generate(1200, 42)
        self.assertTrue(a.equals(b))
        self.assertEqual(a.shape, (1200, 11))
        self.assertFalse(a.isna().any().any())

    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.csv"
            generate(3000, 42).to_csv(path, index=False)
            result = run(path, Path(tmp) / "out")
            self.assertGreater(result["logistic_regression"]["roc_auc"], 0.5)
            self.assertEqual(len(result["skill_segments"]), 5)
            self.assertTrue((Path(tmp) / "out" / "metrics.json").exists())

if __name__ == "__main__":
    unittest.main()
