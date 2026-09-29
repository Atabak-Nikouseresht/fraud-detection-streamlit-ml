import ast
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

REPOSITORY = Path(__file__).resolve().parents[1]
NOTEBOOK = REPOSITORY / "notebooks" / "Fraud_Detection.ipynb"


def load_path_helpers():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in notebook.get("cells", []):
        source = "".join(cell.get("source", []))
        if "def find_repository_root(" not in source:
            continue
        tree = ast.parse(source)
        names = {"find_repository_root", "resolve_dataset_path"}
        definitions = [
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in names
        ]
        namespace = {"Path": Path}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
        return namespace
    raise AssertionError("Notebook does not define its dataset path helpers")


class NotebookReproducibilityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "project"
        self.notebooks = self.root / "notebooks"
        self.data = self.root / "data"
        self.notebooks.mkdir(parents=True)
        self.data.mkdir()
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


if __name__ == "__main__":
    unittest.main()
