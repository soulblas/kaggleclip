from __future__ import annotations
from pathlib import Path
import json
from datetime import datetime, timezone

class StageLogger:
    def __init__(self, log_file: Path):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def stage(self, name: str, payload: dict | None = None):
        payload = payload or {}
        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "stage": name,
            "payload": payload,
        }
        with self.log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
