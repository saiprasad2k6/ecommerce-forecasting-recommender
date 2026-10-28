"""Verify local data deletion using disposable directories only."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest

from src.common.cleanup_data import cleanup_data


class CleanupDataTests(unittest.TestCase):
    def run_cleanup(self, root, **kwargs):
        with redirect_stdout(io.StringIO()):
            cleanup_data(root, **kwargs)

    def test_preview_apply_and_repeat(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_files = ["raw/dataset.zip", "raw/orders.csv", "processed/orders.csv",
                          "curated/orders.csv", "features/train.parquet", "warehouse/olist.duckdb",
                          "warehouse/olist.sqlite", "warehouse/olist.sqlite-wal", "baselines/mlflow.db"]
            preserved = ["src/data/source.py", "mlruns/artifact.csv", "models/model.pkl",
                         "notebooks/example.ipynb", ".venv/keep", ".git/config"]
            for filename in [*[f"data/{name}" for name in data_files], *preserved]:
                path = root / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("original")
            self.run_cleanup(root)
            for name in data_files:
                self.assertEqual((root / "data" / name).read_text(), "original")
            self.run_cleanup(root, dry_run=False)
            self.assertFalse((root / "data").exists())
            for name in preserved:
                self.assertEqual((root / name).read_text(), "original")
            self.run_cleanup(root, dry_run=False)

    def test_symlink_target_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root, external = Path(directory), Path(outside)
            (external / "database.sqlite").write_text("keep")
            (root / "data").symlink_to(external, target_is_directory=True)
            self.run_cleanup(root, dry_run=False)
            self.assertFalse((root / "data").is_symlink())
            self.assertEqual((external / "database.sqlite").read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
