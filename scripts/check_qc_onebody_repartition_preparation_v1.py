#!/usr/bin/env python3
"""Identity/preservation audit only; no numerical kernel import or execution."""
from pathlib import Path
import argparse,ast,json,hashlib,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from trottertracks.ground_pf_repartition.qc_onebody_repartition_execution_v1 import ARTIFACT,OUTPUT,preflight,digest,write_json


def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,stderr=subprocess.PIPE)


def check():
    report=preflight(ROOT);base='9e288089ca22e2729976ae75d096c4df00e49a7d'
    protected=json.loads((ROOT/ARTIFACT/'preservation_index.json').read_text())
    assert protected['count']==len(protected['files'])==417
    for path,identity in protected['files'].items():
        data=(ROOT/path).read_bytes()
        assert data==git('show',base+':'+path) and hashlib.sha256(data).hexdigest()==identity['SHA256'] and len(data)==identity['bytes'],path
    old=json.loads(git('show',base+':artifacts/validation_manifest.json'))
    new=json.loads((ROOT/'artifacts/validation_manifest.json').read_text())
    assert len(old['result_sets'])==150 and new['result_sets'][:150]==old['result_sets']
    assert {k:v for k,v in old.items() if k!='result_sets'}=={k:v for k,v in new.items() if k!='result_sets'}
    assert 'description' not in new['relevant_commits'][7]
    received=json.loads((ROOT/ARTIFACT/'received_review_identity.json').read_text())
    assert digest(ROOT/received['published_path'])==received['SHA256'] and (ROOT/received['published_path']).stat().st_size==received['bytes']
    freeze=json.loads((ROOT/ARTIFACT/'source_manifest.json').read_text())
    for path,identity in freeze['files'].items():
        body=git('show',freeze['source_commit']+':'+path)
        assert body==(ROOT/path).read_bytes() and hashlib.sha256(body).hexdigest()==identity['SHA256'],path
        if path.endswith('.py'):ast.parse(body)
    synthetic=json.loads((ROOT/ARTIFACT/'synthetic_tests.json').read_text())
    assert synthetic['status']=='PASS' and synthetic['test_count']==37
    assert synthetic['new_registered_H_eigen_RDM_expectation_PF_compile_calls']==0
    target=json.loads((ROOT/ARTIFACT/'authorization_target.json').read_text())
    assert target['approved'] is False and target['status']=='UNISSUED_APPROVAL_TARGET'
    assert target['source_manifest_SHA256']==digest(ROOT/ARTIFACT/'source_manifest.json') and target['contract_SHA256']==digest(ROOT/ARTIFACT/'contract.json')
    assert not (ROOT/OUTPUT).exists()
    report.update({'status':'PREPARATION_IDENTITY_PRESERVATION_AUDIT_PASS','base_commit':base,
                   'source_commit':freeze['source_commit'],'frozen_files':len(freeze['files']),
                   'protected_files_matched':417,'old_manifest_records_preserved':150,
                   'old_global_description_FAIL_preserved':True,'whole_global_validator':'NOT_RUN_NOT_GLOBAL_PASS',
                   'received_review_verbatim_bytes':received['bytes'],'synthetic_tests':37,
                   'old27_checks_reexecuted':False,'new_registered_scientific_calls':0,
                   'authorization_issued':False,'attempt_consumed':False,'current_CI':False,
                   'new_GPT_review_started':False,'mandatory_stop':True})
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path)
    args=parser.parse_args();result=check()
    if args.output:write_json(args.output,result)
    print(json.dumps(result,ensure_ascii=False,indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
