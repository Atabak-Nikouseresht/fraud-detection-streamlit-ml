import subprocess
import sys
import unittest
from pathlib import Path

from sklearn.pipeline import Pipeline
from streamlit.testing.v1 import AppTest

from app.fraud_app import (
    FEATURE_COLUMNS,
    MODEL_PATH,
    TRANSACTION_TYPES,
    build_transaction_features,
    load_model,
    predict_transaction,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class FraudAppModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_model()

    def test_model_artifact_loads_as_predictive_pipeline(self):
        self.assertTrue(MODEL_PATH.is_file())
        self.assertIsInstance(self.model, Pipeline)
        self.assertTrue(callable(self.model.predict))

    def test_streamlit_app_runs_and_predicts_without_dataset(self):
        application = AppTest.from_file(str(REPOSITORY_ROOT / "app" / "fraud_app.py"))
        application.run(timeout=20)
        self.assertEqual(len(application.exception), 0)
        application.button[0].click().run(timeout=20)
        self.assertEqual(len(application.exception), 0)
        self.assertTrue(
            any(item.value.startswith("Prediction:") for item in application.subheader)
        )

    def test_ui_schema_matches_saved_model_schema(self):
        self.assertEqual(list(self.model.feature_names_in_), list(FEATURE_COLUMNS))

    def test_ui_transaction_options_cover_model_categories(self):
        encoder = next(
            transformer
            for name, transformer, _columns in self.model.named_steps["prep"].transformers_
            if name == "cat"
        )
        categories = set(encoder.categories_[0])
        self.assertEqual(set(TRANSACTION_TYPES), categories)

    def test_feature_builder_reproduces_training_balance_differences(self):
        row = {
            "type": "TRANSFER",
            "amount": 100.0,
            "oldbalanceOrg": 500.0,
            "newbalanceOrig": 400.0,
            "oldbalanceDest": 30.0,
            "newbalanceDest": 130.0,
        }
        frame = build_transaction_features(row)
        self.assertEqual(list(frame.columns), list(FEATURE_COLUMNS))
        self.assertEqual(frame.loc[0, "balanceDiffOrig"], 100.0)
        self.assertEqual(frame.loc[0, "balanceDiffDest"], 100.0)

    def test_app_module_imports_without_running_the_interactive_ui(self):
        result = subprocess.run(
            [sys.executable, "-c", "import app.fraud_app"],
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_prediction_returns_binary_class_for_every_model_category(self):
        for transaction_type in TRANSACTION_TYPES:
            row = {
                "type": transaction_type,
                "amount": 1000.0,
                "oldbalanceOrg": 10000.0,
                "newbalanceOrig": 9000.0,
                "oldbalanceDest": 0.0,
                "newbalanceDest": 0.0,
            }
            with self.subTest(transaction_type=transaction_type):
                self.assertIn(predict_transaction(self.model, row), (0, 1))


if __name__ == "__main__":
    unittest.main()
