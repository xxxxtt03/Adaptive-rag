"""Export OpenAPI 3.1.0 JSON specification schema for client SDK generation.

Usage:
    python scripts/export_openapi.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure UTF-8 console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from app.main import create_app

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "docs/openapi.json"


def export_openapi() -> None:
    app = create_app()
    openapi_schema = app.openapi()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(openapi_schema, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    paths_count = len(openapi_schema.get("paths", {}))
    print(f"✅ Successfully exported OpenAPI schema to {OUTPUT_PATH}")
    print(f"   API Paths: {paths_count} routes registered")


if __name__ == "__main__":
    export_openapi()
