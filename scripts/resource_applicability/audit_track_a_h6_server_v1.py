#!/usr/bin/env python3
"""Static prepared binding audit. Optional audit of saved Phase A JSON only."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks.resource_applicability.h6_server_contract_v1 import verify_bindings,grant_draft,flags
from trottertracks.resource_applicability.h6_server_execution_v1 import admit_new_group
from trottertracks.resource_applicability.ax2a_preparation import digest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path);a=p.parse_args();m=json.loads(a.manifest.read_text())
    verify_bindings(ROOT,m)
    result=dict(binding_verified=True,manifest_digest=digest(m),grant_draft_approved=False,**flags())
    if a.output:
        terminal=json.loads((a.output/'execution_terminal.json').read_text())
        if terminal.get('mandatory_stop') is not True or terminal.get('next_stage_authorized') is not False:
            raise ValueError('PHASE_A_TERMINAL_FLAGS')
        if terminal['status']=='H6_SERVER_PHASE_A_COMPLETE_MANDATORY_STOP':
            result['wrappers']=len(admit_new_group(a.output/'cost_exploration_0447',m['phase_scope']['task'],m['source_commit'])['rows'])
        result['terminal']=terminal
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':raise SystemExit(main())
