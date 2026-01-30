from __future__ import annotations
from pathlib import Path
import json

def ensure_outdir(out_dir: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

def write_agent_meta(out_dir: str, agent: dict):
    out = Path(out_dir) / "agent_meta.json"
    out.write_text(json.dumps(agent, ensure_ascii=False, indent=2), encoding="utf-8")
