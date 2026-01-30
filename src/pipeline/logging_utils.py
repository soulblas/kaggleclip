from __future__ import annotations
from pathlib import Path
import json
import logging
import sys
import time
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


def log_flush(logger: logging.Logger | None = None) -> None:
    logger = logger or logging.getLogger("viralshort")
    for h in logger.handlers:
        try:
            h.flush()
        except Exception:
            pass
    try:
        sys.stdout.flush()
    except Exception:
        pass


class StageTimer:
    def __init__(self, stage_id: int, name: str, logger: logging.Logger | None = None):
        self.stage_id = stage_id
        self.name = name
        self.logger = logger or logging.getLogger("viralshort")
        self.t0 = None
        self.dt = None

    def __enter__(self):
        self.t0 = time.monotonic()
        self.logger.info(f"[STAGE {self.stage_id:02d}] START - {self.name}")
        log_flush(self.logger)
        return self

    def __exit__(self, exc_type, exc, tb):
        self.dt = time.monotonic() - (self.t0 or time.monotonic())
        if exc:
            self.logger.error(f"[STAGE {self.stage_id:02d}] FAIL  - {self.name} ({self.dt:.2f}s): {exc}")
        else:
            self.logger.info(f"[STAGE {self.stage_id:02d}] END   - {self.name} ({self.dt:.2f}s)")
        log_flush(self.logger)
        return False
