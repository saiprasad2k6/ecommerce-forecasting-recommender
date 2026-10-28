"""Exercise reset behavior only inside temporary directories."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from src.common.reset_repository import reset_repository


class ResetRepositoryTests(unittest.TestCase):
    def run_reset(self, root: Path, **kwargs):
        with redirect_stdout(io.StringIO()):
            reset_repository(root, **kwargs)

    def test_default_is_read_only_and_apply_preserves_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ["data/raw/input.csv", "data/warehouse/olist.duckdb", "mlruns/run/artifact.csv",
                         "mlartifacts/run/result.csv", "artifacts/report.json", "models/model.pkl",
                         "src/data/clean.py", "scripts/data/download.py", "configs/settings.yaml",
                         ".git/config", ".venv/keep", ".env", "documents/notes.md",
                         "src/data/__pycache__/clean.pyc", ".pytest_cache/state"]:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("keep or remove")
            notebook = {"cells": [
                {"cell_type": "markdown", "source": ["Keep explanation"], "metadata": {}},
                {"cell_type": "code", "source": ["print(1)"], "execution_count": 1,
                 "outputs": [{"output_type": "stream", "text": ["1"]}],
                 "metadata": {"execution": {"timestamp": "old"}, "tags": ["keep"]}}],
                "metadata": {"kernelspec": {"name": "python3"}, "widgets": {"old": "state"}},
                "nbformat": 4, "nbformat_minor": 5}
            path = root / "notebooks/example.ipynb"
            path.parent.mkdir()
            path.write_text(json.dumps(notebook))
            before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
            self.run_reset(root)
            self.assertEqual(before, {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()})
            self.run_reset(root, dry_run=False)
            for name in ["data", "mlruns", "mlartifacts", "artifacts", "models", ".pytest_cache", "src/data/__pycache__"]:
                self.assertFalse((root / name).exists())
            for name in ["src/data/clean.py", "scripts/data/download.py", "configs/settings.yaml",
                         ".git/config", ".venv/keep", ".env", "documents/notes.md"]:
                self.assertEqual((root / name).read_bytes(), before[Path(name)])
            cleaned = json.loads(path.read_text())
            self.assertEqual(cleaned["cells"][0], notebook["cells"][0])
            self.assertEqual(cleaned["cells"][1]["source"], ["print(1)"])
            self.assertEqual(cleaned["cells"][1]["metadata"], {"tags": ["keep"]})
            self.assertEqual(cleaned["cells"][1]["outputs"], [])
            self.assertIsNone(cleaned["cells"][1]["execution_count"])
            self.assertEqual(cleaned["metadata"], {"kernelspec": {"name": "python3"}})
            # Re-running an applied reset succeeds without changing source.
            self.run_reset(root, dry_run=False)

    def test_symlinks_do_not_delete_external_files(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root, external = Path(directory), Path(outside)
            (external / "keep.txt").write_text("keep")
            (external / "__pycache__").mkdir()
            (external / "__pycache__/keep.pyc").write_text("keep")
            (root / "data").symlink_to(external, target_is_directory=True)
            (root / "src").mkdir()
            (root / "src/linked").symlink_to(external, target_is_directory=True)
            (root / "src/__pycache__").symlink_to(external, target_is_directory=True)
            (root / "notebooks").mkdir()
            (external / "example.ipynb").write_text("not a notebook")
            (root / "notebooks/linked.ipynb").symlink_to(external / "example.ipynb")
            self.run_reset(root, dry_run=False)
            self.assertFalse((root / "data").is_symlink())
            self.assertFalse((root / "src/__pycache__").is_symlink())
            self.assertTrue((root / "src/linked").is_symlink())
            self.assertEqual((external / "keep.txt").read_text(), "keep")
            self.assertTrue((external / "__pycache__/keep.pyc").is_file())
            self.assertEqual((external / "example.ipynb").read_text(), "not a notebook")

    def test_invalid_notebook_stops_before_deletion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            (root / "data/keep.csv").write_text("still here")
            (root / "notebooks").mkdir()
            (root / "notebooks/broken.ipynb").write_text("invalid JSON")
            with self.assertRaises(json.JSONDecodeError):
                self.run_reset(root, dry_run=False)
            self.assertEqual((root / "data/keep.csv").read_text(), "still here")

    def test_caches_across_project_preserve_environments_and_git(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            remove = ["__pycache__/root.pyc", "configs/__pycache__/settings.pyc",
                      "custom_tools/nested/__pycache__/tool.pyc", "src/data/__pycache__/clean.pyc",
                      "custom_tools/legacy.pyc", "custom_tools/legacy.pyo"]
            preserve = [".venv/lib/__pycache__/package.pyc", ".git/hooks/__pycache__/hook.pyc",
                        "custom_env/lib/__pycache__/package.pyc", "custom_tools/tool.py"]
            for name in remove + preserve:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("original")
            (root / "custom_env/pyvenv.cfg").write_text("home = /python")
            self.run_reset(root)
            for name in remove + preserve:
                self.assertTrue((root / name).is_file())
            self.run_reset(root, dry_run=False)
            for name in remove:
                self.assertFalse((root / name).exists())
            for name in preserve:
                self.assertEqual((root / name).read_text(), "original")
            for name in ["__pycache__", "configs/__pycache__", "custom_tools/nested/__pycache__", "src/data/__pycache__"]:
                self.assertFalse((root / name).exists())


if __name__ == "__main__":
    unittest.main()
