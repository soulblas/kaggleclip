from __future__ import annotations
from pathlib import Path
import hashlib
import re

def load_agent_contract(agent_path: str = "AGENTS.md") -> dict:
    p = Path(agent_path)
    if not p.exists():
        return {
            "agent_file_present": False,
            "agent_sha256": "",
            "agent_version": "missing",
        }

    text = p.read_text(encoding="utf-8", errors="replace")
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()

    m = re.search(r"(?im)^\s*Agent Version\s*:\s*(.+?)\s*$", text)
    version = m.group(1).strip() if m else "unknown"

    return {
        "agent_file_present": True,
        "agent_sha256": sha,
        "agent_version": version,
    }
