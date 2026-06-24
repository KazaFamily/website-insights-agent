"""
Simple JSON-file memory for the agent.
Reads and writes to data/memory/*.json.
Designed to be upgraded to S3 later — just swap the read/write calls.
"""

import json
import os
from datetime import datetime, timezone


def _load_json(path: str, default):
    """Load a JSON file, returning default if missing or malformed."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def get_memory(memory_dir: str = "data/memory") -> dict:
    """Load and merge last_run.json and page_analyses.json from memory_dir."""
    last_run = _load_json(os.path.join(memory_dir, "last_run.json"), {})
    page_analyses = _load_json(os.path.join(memory_dir, "page_analyses.json"), {})
    return {"last_run": last_run, "page_analyses": page_analyses}


def save_memory(
    recommendations: str,
    pages_analyzed: list[str],
    memory_dir: str = "data/memory",
) -> dict:
    """Persist this run's recommendations and update per-page analysis timestamps."""
    os.makedirs(memory_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()

    last_run = {
        "timestamp": timestamp,
        "recommendations": recommendations,
        "pages_analyzed": pages_analyzed,
    }
    with open(os.path.join(memory_dir, "last_run.json"), "w") as f:
        json.dump(last_run, f, indent=2)

    page_analyses = _load_json(os.path.join(memory_dir, "page_analyses.json"), {})
    for url in pages_analyzed:
        page_analyses[url] = {"last_analyzed": timestamp}
    with open(os.path.join(memory_dir, "page_analyses.json"), "w") as f:
        json.dump(page_analyses, f, indent=2)

    return {"status": "ok", "timestamp": timestamp}
