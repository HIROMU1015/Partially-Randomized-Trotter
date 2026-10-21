"""Consume completed shared input without rerunning its one-shot stage."""
import argparse
import json
import recomputed_s2_runtime_v2 as r

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--system',choices=['H6','H8'],required=True)
    system=parser.parse_args().system
    r.context(system)  # Require complete, hash-verified shared input.
    rows=[]
    for c in sorted(r.read(r.BASE/'candidates.json'),key=lambda c:(c['role']!='PR',c['id'])):
        if c['system']!=system:continue
        try:
            row=r.candidate_stages(system,c['id'])
        except BaseException as exc:
            row=dict(system=system,candidate=c['id'],status='FAILED_NO_RETRY',error=str(exc))
        rows.append(row);print(json.dumps(row),flush=True)
    terminal=dict(system=system,status='SYSTEM_FINISHED_MANDATORY_STOP',candidates=rows,
                  completed_input_reused=True,input_stage_rerun=False)
    r.save(r.BASE/(system+'_terminal.json'),terminal)
