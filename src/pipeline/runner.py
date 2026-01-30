from pathlib import Path
from .agent_contract import load_agent_contract
from .logging_utils import StageLogger
from .export import ensure_outdir, write_agent_meta

def run_pipeline(input_video: str, out_dir: str = "outputs", **kwargs):
    out_dir = str(out_dir)
    ensure_outdir(out_dir)

    agent = load_agent_contract("AGENTS.md")
    write_agent_meta(out_dir, agent)

    log = StageLogger(Path(out_dir) / "pipeline_log.txt")
    log.stage("stage_0_init", {"input_video": input_video, "out_dir": out_dir, "agent_version": agent.get("agent_version")})

    # Placeholder: nanti diisi stage beneran (segmentation -> candidates -> asr -> scoring -> selection -> export)
    log.stage("stage_99_done", {"status": "ok_placeholder"})

    return {"status": "ok_placeholder", "out_dir": out_dir, "agent": agent}
