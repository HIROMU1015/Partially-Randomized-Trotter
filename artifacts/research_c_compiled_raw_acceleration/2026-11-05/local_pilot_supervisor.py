"""Operational reuse of the existing supervisor;no numerical SOURCE change."""
from pathlib import Path
import runpy
import sys

if __name__ == '__main__':
    root=Path(__file__).resolve().parents[3]
    functions=runpy.run_path(str(root/'scripts/run_research_c_stage_p_completion.py'))
    command=[sys.executable,'-u','-X','faulthandler',
             str(root/'scripts/prepare_research_c_compiled_raw_acceleration.py'),
             '--execute-local-pilot']
    code=functions['supervise'](command,Path(__file__).resolve().parent/'local-pilot-launch01')
    sys.exit(code if code>=0 else 128-code)
