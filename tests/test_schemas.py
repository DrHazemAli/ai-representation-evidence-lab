import json
from pathlib import Path


def test_schema_files_are_valid_json():
    schema_dir = Path("schemas")

    for path in schema_dir.glob("*.schema.json"):
        decoded = json.loads(path.read_text(encoding="utf-8"))
        assert decoded["$schema"] == "https://json-schema.org/draft/2020-12/schema"
