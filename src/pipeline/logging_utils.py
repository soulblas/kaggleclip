from __future__ import annotations

from pathlib import Path
import json
import logging
import os
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


def _use_unicode() -> bool:
    return os.getenv("LOG_UNICODE", "0") == "1"


def _stage_header(stage_id: int, name: str) -> str:
    if _use_unicode():
        return f"\u250f\u2501 STAGE {stage_id:02d}: {name}"
    return f"=== STAGE {stage_id:02d}: {name} ==="


def _stage_footer(stage_id: int, name: str, elapsed_sec: float, ok: bool = True, err: str | None = None) -> str:
    if _use_unicode():
        prefix = "\u2517\u2501 END" if ok else "\u2517\u2501 FAIL"
        tail = f"(elapsed={elapsed_sec:.2f}s)"
        if err:
            tail += f" | {err}"
        return f"{prefix} STAGE {stage_id:02d}: {name} {tail}"
    status = "END" if ok else "FAIL"
    tail = f"(elapsed={elapsed_sec:.2f}s)"
    if err:
        tail += f" | {err}"
    return f"=== {status} STAGE {stage_id:02d}: {name} {tail} ==="


_WARNINGS: list[str] = []


def _ok_tag() -> str:
    return "[\u2713]" if _use_unicode() else "[OK]"


def log_i(logger: logging.Logger, msg: str) -> None:
    logger.info(f"[i] {msg}")


def log_ok(logger: logging.Logger, msg: str) -> None:
    logger.info(f"{_ok_tag()} {msg}")


def log_warn(logger: logging.Logger, msg: str) -> None:
    logger.warning(f"[!] {msg}")
    if msg not in _WARNINGS:
        _WARNINGS.append(msg)


def log_err(logger: logging.Logger, msg: str) -> None:
    logger.error(f"[x] {msg}")


def _bar(count: int, max_count: int, width: int = 10) -> str:
    if max_count <= 0:
        return ""
    n = int(round((count / max_count) * width))
    return "#" * max(0, n)


def log_bar_chart(logger: logging.Logger, title: str, items: list[tuple[str, int]], width: int = 10) -> None:
    if not items:
        return
    max_count = max(c for _, c in items) if items else 0
    log_i(logger, title)
    for label, count in items[:10]:
        bar = _bar(count, max_count, width=width)
        logger.info(f"    {label}: {bar} ({count})")


def log_duration_hist(logger: logging.Logger, durations: list[float]) -> None:
    if not durations:
        return
    short = sum(1 for d in durations if 15.0 <= d <= 35.0)
    mid = sum(1 for d in durations if 45.0 <= d <= 75.0)
    long = sum(1 for d in durations if 75.0 < d <= 120.0)
    other = max(0, len(durations) - (short + mid + long))
    items = [
        ("dur_15_35", short),
        ("dur_45_75", mid),
        ("dur_75_120", long),
    ]
    if other:
        items.append(("dur_other", other))
    log_bar_chart(logger, "duration_hist", items)


def log_warning_panel(logger: logging.Logger, limit: int = 20) -> None:
    if not _WARNINGS:
        return
    logger.warning(f"[!] WARNING SUMMARY ({len(_WARNINGS)})")
    for i, msg in enumerate(_WARNINGS[:limit], 1):
        logger.warning(f"    {i:02d}. {msg}")


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
        log_i(self.logger, _stage_header(self.stage_id, self.name))
        log_flush(self.logger)
        return self

    def __exit__(self, exc_type, exc, tb):
        self.dt = time.monotonic() - (self.t0 or time.monotonic())
        if exc:
            log_err(self.logger, _stage_footer(self.stage_id, self.name, self.dt, ok=False, err=str(exc)))
        else:
            log_ok(self.logger, _stage_footer(self.stage_id, self.name, self.dt, ok=True))
        log_flush(self.logger)
        return False


def setup_pipeline_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("viralshort")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    log_file.parent.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger
