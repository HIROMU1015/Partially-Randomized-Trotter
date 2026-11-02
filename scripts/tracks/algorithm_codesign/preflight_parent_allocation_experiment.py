#!/usr/bin/env python3
"""Hash/metadata inspection only, with no registered constructor or worker."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))

def run(root):
    from trottertracks.algorithm_codesign.parent_experiment.launch import read_contract,check_identities,CONTRACT
    from trottertracks.algorithm_codesign.rte_development.evidence import digest
    c=read_contract(root);m=check_identities(root,c)
    a=json.loads((root/c['execution']['authorization_path']).read_text())
    if a.get('science_execution_authorized') is not False:raise PermissionError('preparation preflight requires pending authority')
    if (root/c['execution']['result_directory']).exists():raise FileExistsError('unexpected result namespace')
    from trottertracks.algorithm_codesign.synthesis_placement.wrapper_launch import verify_runtime
    runtime=verify_runtime(root,dict(tool_identity=c['acquisition']['tool_identity']))
    return dict(status='READY_FOR_SEPARATE_ONE_SHOT_AUTHORIZATION',contract_sha256=digest(root/CONTRACT),
        critical_files=len(m['critical_sha256']),runtime=runtime,science_execution_authorized=False,
        registered_evaluations=0,synthesis=0,matrix=0,ranking=0,approved_authorization_created=False,mandatory_STOP=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output');a=p.parse_args();value=run(ROOT)
    text=json.dumps(value,indent=2)+'\n'
    if a.output:
        with Path(a.output).open('x') as f:f.write(text)
    print(text)
