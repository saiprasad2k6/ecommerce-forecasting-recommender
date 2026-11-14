#!/usr/bin/env python3
"""Regenerate the Swagger/OpenAPI contract without loading models or datasets."""
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from api.app import app

if __name__ == "__main__":
    schema = app.openapi()
    target = ROOT / "documentation"
    target.mkdir(exist_ok=True)
    (target / "inference-openapi.json").write_text(json.dumps(schema, indent=2) + "\n")
    (target / "inference-openapi.yaml").write_text(yaml.safe_dump(schema, sort_keys=False))
    print("Exported documentation/inference-openapi.{json,yaml}")
