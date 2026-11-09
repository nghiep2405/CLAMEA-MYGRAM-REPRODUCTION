from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.notebook_runner import run_script


def normalized_source(cell: dict) -> str:
    source = "".join(cell.get("source", []))
    return source if not source or source.endswith("\n") else source + "\n"


class RefactorStructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(
            (ROOT / "notebook_workflows.json").read_text(encoding="utf-8")
        )

    def test_three_workflows_and_nineteen_unique_scripts_are_registered(self) -> None:
        workflows = self.manifest["workflows"]
        self.assertEqual(set(workflows), {"base", "table1", "table2"})
        paths = {
            step["path"]
            for workflow in workflows.values()
            for step in workflow["steps"]
        }
        self.assertEqual(len(paths), 19)

    def test_each_workflow_script_compiles_and_matches_its_hash(self) -> None:
        checked = set()
        for workflow in self.manifest["workflows"].values():
            for step in workflow["steps"]:
                if step["path"] in checked:
                    continue
                checked.add(step["path"])
                path = ROOT / step["path"]
                source = path.read_text(encoding="utf-8")
                compile(source, str(path), "exec")
                actual = hashlib.sha256(source.encode("utf-8")).hexdigest()
                self.assertEqual(actual, step["sha256"], step["path"])

    def test_notebooks_are_clean_and_call_every_step_once(self) -> None:
        for workflow in self.manifest["workflows"].values():
            notebook = json.loads((ROOT / workflow["notebook"]).read_text(encoding="utf-8"))
            combined = "\n".join(
                "".join(cell.get("source", [])) for cell in notebook["cells"]
            )
            for index, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] != "code":
                    continue
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
                compile(normalized_source(cell), f"{workflow['notebook']}:cell-{index}", "exec")
            for step in workflow["steps"]:
                self.assertEqual(combined.count(step["path"]), 1, step["path"])

    def test_inline_bootstrap_cells_match_manifest(self) -> None:
        for workflow in self.manifest["workflows"].values():
            notebook = json.loads((ROOT / workflow["notebook"]).read_text(encoding="utf-8"))
            for bootstrap in workflow["bootstrap"]:
                source = normalized_source(notebook["cells"][bootstrap["cell"]])
                actual = hashlib.sha256(source.encode("utf-8")).hexdigest()
                self.assertEqual(actual, bootstrap["sha256"])

    def test_generated_results_are_not_part_of_source_tree(self) -> None:
        forbidden = [
            ROOT / "environment.json",
            ROOT / "audit",
            ROOT / "data",
            ROOT / "dist",
            ROOT / "outputs",
        ]
        forbidden.extend((ROOT / "docs").glob("stage_*_report.md"))
        self.assertEqual([str(path) for path in forbidden if path.exists()], [])


class NotebookRunnerTests(unittest.TestCase):
    def test_steps_share_notebook_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.py"
            second = Path(directory) / "second.py"
            first.write_text("value = 40\n", encoding="utf-8")
            second.write_text("value += 2\n", encoding="utf-8")
            namespace: dict[str, object] = {}

            run_script(first, namespace)
            run_script(second, namespace)

            self.assertEqual(namespace["value"], 42)


if __name__ == "__main__":
    unittest.main()

