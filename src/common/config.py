"""Shared defaults for historical replay pipelines and command-line programs."""
from pathlib import Path

import yaml

SETTINGS = yaml.safe_load((Path(__file__).resolve().parents[2] / "configs/settings.yaml").read_text())
REPLAY = SETTINGS["replay"]
