"""Lightweight clean-namespace notebook export and evidence regression checks."""
import ast
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "Fraud_Detection.ipynb"


class ExportEvidenceTests(unittest.TestCase):
    def test_notebook_export_from_clean_namespace_writes_candidate_with_metadata(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        namespace = {}
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                tree = ast.parse("".join(cell["source"]))
                for node in tree.body:
                    if isinstance(node, ast.Import) and any(
                        alias.name == "joblib" for alias in node.names
                    ):
                        exec(compile(ast.Module(body=[node], type_ignores=[]), "imports", "exec"), namespace)
        self.assertIn("joblib", namespace, "Notebook must import its export dependency")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "models").mkdir()
            (root / "notebooks").mkdir()
            (root / "notebooks" / NOTEBOOK.name).write_bytes(NOTEBOOK.read_bytes())
            shipped = root / "models" / "fraud_detection_model.pkl"
            shipped.write_bytes(b"preserve shipped artifact")
            namespace.update(
                REPOSITORY_ROOT=root,
                pipeline=joblib.load(ROOT / "models" / "fraud_detection_model.pkl"),
            )
            exec("".join(notebook["cells"][-1]["source"]), namespace)
            candidate = root / "models" / "fraud_detection_model.candidate.pkl"
            self.assertEqual(namespace["MODEL_PATH"], candidate)
            self.assertEqual(shipped.read_bytes(), b"preserve shipped artifact")
            self.assertTrue(callable(joblib.load(candidate).predict))
            metadata = json.loads(candidate.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["sha256"], hashlib.sha256(candidate.read_bytes()).hexdigest())
            self.assertEqual(len(metadata["feature_schema"]), 8)
            self.assertEqual(len(metadata["active_predictors"]), 6)
            self.assertEqual(metadata["status"], "candidate-unvalidated")

    def test_artifact_validation_rejects_incorrect_active_predictors(self):
        from unittest.mock import patch
        from scripts.check_environment import verify_artifact

        metadata = json.loads((ROOT / "models" / "artifact-metadata.json").read_text(encoding="utf-8"))
        metadata["active_predictors"] = metadata["feature_schema"]
        with tempfile.TemporaryDirectory() as directory:
            sidecar = Path(directory) / "metadata.json"
            sidecar.write_text(json.dumps(metadata), encoding="utf-8")
            with patch("scripts.check_environment.METADATA", sidecar):
                with self.assertRaisesRegex(SystemExit, "predictor"):
                    verify_artifact()

    def test_notebook_saved_outputs_are_not_presented_as_current_results(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        self.assertIn("not been rerun", "".join(notebook["cells"][0]["source"]))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])


if __name__ == "__main__":
    unittest.main()
