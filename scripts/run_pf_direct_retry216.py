"""Owned coordinator for the single explicitly approved retry scope."""
import argparse
from pathlib import Path
import pf_direct_compile_contract as c
import pf_direct_retry_contract as r
from run_pf_direct_execution import run

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sealed-directory',type=Path,required=True)
    args=p.parse_args();spec=c.read_json(args.sealed_directory/(r.GROUPS[0]+'_spec.json'))
    grant=c.read_json(args.sealed_directory/(r.GROUPS[0]+'_grant.json'));r.authorize(spec,grant)
    def estimate(row):return spec['memory_admission_bytes'][row['task_id']]
    # Admit slow H8 cases early; scope and physics are unchanged.
    def priority(task):return (0 if task['system']=='H8_DF28' else 1,-int(task['pair'][0]['q']),task['cid'])
    run(args.sealed_directory,execution_api=r,worker_script='scripts/pf_direct_retry_worker.py',estimator=estimate,priority=priority)
