import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.check_environment import verify_artifact
from scripts.synthetic_training_smoke import FEATURES, run_smoke
from scripts.validate_dataset import EXPECTED_COLUMNS, validate_dataset


class ReproducibilityTests(unittest.TestCase):
    def test_synthetic_training_smoke_trains_eight_feature_pipeline_and_reports_metrics(self):
        result = run_smoke()
        self.assertEqual(len(FEATURES), 8)
        self.assertEqual(result["feature_count"], 8)
        self.assertIn("synthetic", result["data"])
        self.assertIn("fraud_class_precision", result)
        self.assertIn("fraud_class_recall", result)
        self.assertIn("fraud_class_f1", result)
        self.assertEqual(len(result["confusion_matrix"]), 2)

    def test_artifact_checksum_matches_metadata(self):
        verify_artifact()

    def test_dataset_validator_accepts_expected_schema_and_binary_target(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.csv"
            frame = pd.DataFrame({column: [0, 1] for column in sorted(EXPECTED_COLUMNS)})
            frame["type"] = ["PAYMENT", "TRANSFER"]
            frame.to_csv(path, index=False)
            self.assertEqual(validate_dataset(path), (True, []))

    def test_dataset_validator_reports_missing_required_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.csv"
            pd.DataFrame({"isFraud": [0, 1]}).to_csv(path, index=False)
            valid, problems = validate_dataset(path)
            self.assertFalse(valid)
            self.assertTrue(any("Missing required columns" in problem for problem in problems))

    def test_dataset_validator_rejects_non_binary_target(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.csv"
            frame = pd.DataFrame({column: [0, 1] for column in sorted(EXPECTED_COLUMNS)})
            frame["type"] = ["PAYMENT", "TRANSFER"]
            frame["isFraud"] = [0, 2]
            frame.to_csv(path, index=False)
            valid, problems = validate_dataset(path)
            self.assertFalse(valid)
            self.assertTrue(any("only 0 and 1" in problem for problem in problems))


if __name__ == "__main__":
    unittest.main()
