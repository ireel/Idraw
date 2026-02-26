import json
from pathlib import Path


def create_session_dir(base_output_dir, timestamp):
    base_output_dir = Path(base_output_dir)
    session_name = timestamp.strftime("%Y%m%d_%H%M%S")
    return base_output_dir / session_name


def write_json(path, payload):
    path = Path(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
