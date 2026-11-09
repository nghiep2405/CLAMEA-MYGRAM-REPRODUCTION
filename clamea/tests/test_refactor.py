from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from clamea_refactor import run_script


class RefactorStructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(
            (PROJECT_ROOT / "pipeline_manifest.json").read_text(encoding="utf-8")
        )
        cls.notebook = json.loads(
            (PROJECT_ROOT / "clamea.ipynb").read_text(encoding="utf-8")
        )

    def test_manifest_has_all_original_cells_in_order(self) -> None:
        stages = self.manifest["stages"]
        self.assertEqual(len(stages), 23)
        self.assertEqual([stage["order"] for stage in stages], list(range(23)))
        self.assertEqual(
            [stage["source_cell"] for stage in stages],
            list(range(23)),
        )

    def test_every_stage_compiles_and_matches_manifest_hash(self) -> None:
        for stage in self.manifest["stages"]:
            path = PROJECT_ROOT / stage["path"]
            source = path.read_text(encoding="utf-8")
            compile(source, str(path), "exec")
            digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
            self.assertEqual(digest, stage["sha256"], stage["path"])

    def test_notebook_contains_no_saved_results(self) -> None:
        for cell in self.notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])

    def test_every_notebook_code_cell_compiles(self) -> None:
        for index, cell in enumerate(self.notebook["cells"]):
            if cell["cell_type"] == "code":
                source = "".join(cell.get("source", []))
                compile(source, f"clamea.ipynb:cell-{index}", "exec")

    def test_notebook_runs_every_manifest_stage_once(self) -> None:
        notebook_source = "\n".join(
            "".join(cell.get("source", [])) for cell in self.notebook["cells"]
        )
        for stage in self.manifest["stages"]:
            self.assertEqual(notebook_source.count(stage["path"]), 1)


class RunnerTests(unittest.TestCase):
    def test_runner_shares_state_and_restores_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.py"
            second = Path(directory) / "second.py"
            first.write_text("value = 40\n", encoding="utf-8")
            second.write_text("value += 2\n", encoding="utf-8")
            namespace = {"__file__": "notebook.py"}

            run_script(first, namespace)
            run_script(second, namespace)

            self.assertEqual(namespace["value"], 42)
            self.assertEqual(namespace["__file__"], "notebook.py")


if __name__ == "__main__":
    unittest.main()

