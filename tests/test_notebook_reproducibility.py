import ast
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from app.fraud_app import FEATURE_COLUMNS
from repository_paths import find_repository_root, resolve_dataset_path, resolve_model_path

REPOSITORY = Path(__file__).resolve().parents[1]
NOTEBOOK = REPOSITORY / "notebooks" / "Fraud_Detection.ipynb"


def load_path_helpers():
    return {
        "find_repository_root": find_repository_root,
        "resolve_dataset_path": resolve_dataset_path,
        "resolve_model_path": resolve_model_path,
    }


class NotebookReproducibilityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "project"
        self.notebooks = self.root / "notebooks"
        self.data = self.root / "data"
        self.notebooks.mkdir(parents=True)
        self.data.mkdir()
        (self.root / "repository_paths.py").touch()
        (self.root / "app").mkdir()
        (self.root / "app" / "fraud_app.py").touch()
        (self.notebooks / "Fraud_Detection.ipynb").touch()
        self.dataset = self.data / "AIML Dataset.csv"
        self.dataset.write_text("value\n7\n", encoding="utf-8")
        self.helpers = load_path_helpers()

    def tearDown(self):
        self.temporary.cleanup()

    def test_resolves_dataset_from_repository_root(self):
        result = self.helpers["resolve_dataset_path"](self.root)
        self.assertEqual(result, self.dataset)

    def test_resolves_dataset_from_notebooks_directory(self):
        result = self.helpers["resolve_dataset_path"](self.notebooks)
        self.assertEqual(result, self.dataset)

    def test_fixture_csv_is_readable_from_resolved_path(self):
        dataset_path = self.helpers["resolve_dataset_path"](self.notebooks)
        frame = pd.read_csv(dataset_path)
        self.assertEqual(frame.loc[0, "value"], 7)

    def test_missing_dataset_error_is_actionable(self):
        self.dataset.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "AIML Dataset.csv"):
            self.helpers["resolve_dataset_path"](self.notebooks)

    def test_notebook_json_and_code_cells_parse(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        self.assertEqual(notebook.get("nbformat"), 4)
        for number, cell in enumerate(notebook.get("cells", [])):
            if cell.get("cell_type") == "code":
                ast.parse("".join(cell.get("source", [])), filename=f"cell-{number}")

    def test_shared_repository_paths_work_from_root_and_notebooks_directory(self):
        for working_directory in (self.root, self.notebooks):
            with self.subTest(working_directory=working_directory):
                self.assertEqual(find_repository_root(working_directory), self.root)
                self.assertEqual(resolve_dataset_path(working_directory), self.dataset)
                self.assertEqual(
                    resolve_model_path(working_directory),
                    self.root / "models" / "fraud_detection_model.pkl",
                )

    def test_notebook_trains_all_application_features_reproducibly(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook.get("cells", [])
            if cell.get("cell_type") == "code"
        )
        trees = [
            ast.parse("".join(cell.get("source", [])))
            for cell in notebook.get("cells", [])
            if cell.get("cell_type") == "code"
        ]
        assignments = {
            target.id: ast.literal_eval(node.value)
            for tree in trees
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name) and target.id in {"categorical", "numerical"}
        }
        self.assertEqual(set(assignments), {"categorical", "numerical"})
        self.assertEqual(
            set(assignments["categorical"] + assignments["numerical"]),
            set(FEATURE_COLUMNS),
        )
        self.assertEqual(
            len(assignments["categorical"] + assignments["numerical"]),
            len(FEATURE_COLUMNS),
        )
        self.assertIn("RANDOM_STATE = 42", source)
        self.assertIn("random_state=RANDOM_STATE", source)
        self.assertIn("average_precision_score", source)
        self.assertIn("roc_auc_score", source)
        self.assertIn("ROC-AUC:", source)


if __name__ == "__main__":
    unittest.main()
