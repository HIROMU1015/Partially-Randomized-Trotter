"""Execute only the approved one-preparation/two-law physical diagnostic."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true');p.add_argument('--source',required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if not a.execute:p.error('Scientific leaf requires --execute; import/technical tests never run it')
    a.out.mkdir(parents=True,exist_ok=False)
    def write(name,value):
        with (a.out/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
    try:
        from trottertracks.h4_pr_physical_closure import execute
        result=execute(ROOT,a.source);write('RESULT.json',result)
        write('STOP.json',dict(status='H4_PHYSICAL_CLOSURE_DIAGNOSTIC_COMPLETE_MANDATORY_STOP',source=a.source,
            small_closed_state=result['state_equivalence']['status'],automatic_follow_on=False))
        print(json.dumps({'status':result['status'],'closure':result['state_equivalence']['status'],
            'components':result['preparation']['component_count'],'bases':result['preparation']['basis_count'],
            'basis_reset_exceptions':len(result['basis_reset_sufficient_criterion']['z_only_exceptions']),
            'phase_barrier':result['phase_control_barrier']['reverse_head_resets_before_control_Z']}))
    except Exception as e:
        write('STOP.json',dict(status='H4_PHYSICAL_CLOSURE_DIAGNOSTIC_FAILED_STOP',source=a.source,
            error=type(e).__name__+': '+str(e),automatic_follow_on=False))
        raise


if __name__=='__main__':main()
