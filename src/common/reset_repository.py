"""Reset local pipeline outputs while retaining source code and environments."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIRS = ("data", "mlruns", "mlartifacts", "artifacts", "models")
SOURCE_DIRS = ("src", "scripts", "tests", "pipelines", "api", "notebooks")
CACHE_DIRS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".ipynb_checkpoints"}
PROTECTED_DIRS = {".git", ".venv", "venv", "env", "ENV", "node_modules",
                  ".agents", ".codex", ".aws", ".kaggle"}


def reset_plan(root: Path) -> tuple[list[Path], list[tuple[Path, dict]]]:
    """Find fixed output paths, local caches and saved notebook results.

    Symlinked output paths are unlinked, never traversed. Cache discovery spans
    the project except environments and protected folders. Notebook output
    clearing remains restricted to source directories. Symlinks are not followed.
    """
    targets = [root / name for name in OUTPUT_DIRS
               if (root / name).exists() or (root / name).is_symlink()]
    notebooks = []
    for current, dirs, files in os.walk(root, followlinks=False):
        current = Path(current)
        for child in dirs[:]:
            path = current / child
            if child in CACHE_DIRS:
                targets.append(path)
                dirs.remove(child)
            elif (path.is_symlink() or child in PROTECTED_DIRS
                  or (current == root and child in OUTPUT_DIRS)
                  or (path / "pyvenv.cfg").is_file()):
                dirs.remove(child)
        for filename in files:
            path = current / filename
            # Includes legacy bytecode stored beside .py files and dangling cache links.
            if filename in CACHE_DIRS or path.suffix in {".pyc", ".pyo"}:
                targets.append(path)
                continue
            if (path.suffix != ".ipynb" or path.is_symlink()
                    or path.relative_to(root).parts[0] not in SOURCE_DIRS):
                continue
            notebook = json.loads(path.read_text(encoding="utf-8"))
            changed = "widgets" in notebook.get("metadata", {})
            notebook.get("metadata", {}).pop("widgets", None)
            for cell in notebook.get("cells", []):
                if cell.get("cell_type") == "code":
                    changed |= bool(cell.get("outputs")) or cell.get("execution_count") is not None
                    changed |= "execution" in cell.get("metadata", {})
                    cell["outputs"] = []
                    cell["execution_count"] = None
                    cell.get("metadata", {}).pop("execution", None)
            if changed:
                notebooks.append((path, notebook))
    return sorted(targets), notebooks


def reset_repository(root: Path, *, dry_run: bool = True) -> None:
    root = root.resolve()
    # Parse every notebook before starting any deletion.
    targets, notebooks = reset_plan(root)
    print(f"Repository: {root}")
    for path in targets:
        print(f"Delete: {path.relative_to(root)}")
    for path, _ in notebooks:
        print(f"Clear saved notebook outputs: {path.relative_to(root)}")
    if not targets and not notebooks:
        print("Nothing to reset.")
        return
    if dry_run:
        print("Dry run: nothing changed. Add --yes to apply this reset.")
        return

    for path in targets:
        if path.is_symlink() or not path.is_dir():
            path.unlink()
        else:
            shutil.rmtree(path)
    for path, notebook in notebooks:
        # Atomic replacement keeps the notebook intact if writing is interrupted.
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(notebook, stream, indent=1, ensure_ascii=False)
            stream.write("\n")
        try:
            temporary.chmod(path.stat().st_mode & 0o777)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    print("Reset complete. Start again with scripts/data/download_olist_data.py.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--yes", action="store_true", help="Delete local outputs and Python caches, including raw data, and clear notebook results")
    mode.add_argument("--dry-run", action="store_true", help="Preview only (the default)")
    args = parser.parse_args()
    reset_repository(ROOT, dry_run=not args.yes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
