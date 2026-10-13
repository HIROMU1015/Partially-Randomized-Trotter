#!/usr/bin/env python3
"""Static source/evidence/preservation check. No numerical package imports."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
BASE='839410511da2dfde46b20b1775964c4f15313a57'
FOLDER='artifacts/qc_mixed_access_source_v1/2026-10-13'


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def check(root):
    root=Path(root)
    def original(name): return subprocess.run(['git','show',BASE+':'+name],cwd=root,capture_output=True,check=True).stdout
    folder=root/FOLDER; manifest=json.loads((folder/'source_manifest.json').read_text())
    for name,value in manifest['files'].items():
        if sha(root/name)!=value: raise ValueError('SOURCE_HASH:'+name)
    contract='artifacts/qc_mixed_access_feasibility/2026-10-13/feasibility_contract.json'
    if (root/contract).read_bytes()!=original(contract): raise ValueError('FROZEN_SCIENCE_CONTRACT_CHANGED')
    protected=json.loads((root/'artifacts/qc_mixed_access_feasibility/2026-10-13/base_source_and_evidence_index.json').read_text())
    for row in protected['files']:
        if sha(root/row['path'])!=row['sha256']: raise ValueError('PROTECTED_OLD_FILE:'+row['path'])
    old=json.loads(original('artifacts/validation_manifest.json'))
    new=json.loads((root/'artifacts/validation_manifest.json').read_text())
    if new['result_sets'][:len(old['result_sets'])]!=old['result_sets']: raise ValueError('OLD_RESULT_ENTRIES')
    if {k:v for k,v in new.items() if k!='result_sets'}!={k:v for k,v in old.items() if k!='result_sets'}:
        raise ValueError('GLOBAL_MANIFEST_FIELDS_CHANGED')
    if 'description' in new['relevant_commits'][7]: raise ValueError('UNAUTHORIZED_GLOBAL_DEFECT_REPAIR')
    report=json.loads((folder/'synthetic_report.json').read_text())
    if report['status']!='PASS' or report['tests']!=22 or report['molecular_jobs']!=0: raise ValueError('SYNTHETIC_SCOPE')
    for name,value in report['source_SHA256'].items():
        if sha(root/name)!=value: raise ValueError('TEST_SOURCE_BINDING:'+name)
    readiness=json.loads((folder/'launch_readiness.json').read_text())
    if readiness['launch_allowed'] or not readiness['science_runner_technically_prepared']:
        raise ValueError('READINESS_GATE')
    if (root/'artifacts/qc_mixed_access_feasibility_run_v1/2026-10-13/attempt_1').exists():
        raise ValueError('UNEXPECTED_REGISTERED_ATTEMPT')
    return {'status':'PASS','locked_files':len(manifest['files']),'synthetic_source_bindings':len(report['source_SHA256']),
            'old_result_sets_preserved':len(old['result_sets']),'old_priority_files_preserved':len(protected['files']),
            'old_global_description_defect_retained':True,'registered_science_attempts':0,'new_GPT_reviews':0,
            'qualification':'static local source/evidence check;not immutable CI or molecular reproduction','mandatory_stop':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--report',type=Path)
    args=parser.parse_args(); result=check(ROOT)
    if args.report: args.report.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
