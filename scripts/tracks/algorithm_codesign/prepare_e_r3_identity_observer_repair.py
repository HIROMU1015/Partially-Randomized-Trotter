"""Static-only v2 metadata preparation; no registered runner/evaluation call."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[3]
OLD=Path('artifacts/track_b_e_r3_identity_diagnostic_preparation/2026-10-12')
NEW=OLD/'v2'
S='8377c1d12439cabab3b97bcc1f57ac030a479785'
BASE='101e225a5cc8fe2f385f8fd802682dc87b708c16'
CODE='src/trottertracks/algorithm_codesign/e_r3_identity_diagnostic/'
CHANGED={CODE+'evaluation.py',CODE+'launch.py','tests/tracks/algorithm_codesign/test_e_r3_identity_diagnostic_preparation.py'}

def sha(b):return hashlib.sha256(b).hexdigest()
def load(p):return json.loads((ROOT/p).read_bytes())
def identity(p):
    b=(ROOT/p).read_bytes();return {'path':str(p),'bytes':len(b),'sha256':sha(b)}
def write(name,data):
    with (ROOT/NEW/name).open('x') as f:json.dump(data,f,ensure_ascii=False,indent=2);f.write('\n')
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)

def pending(contract_hash,manifest_hash):
    return {'status':'PENDING_SEPARATE_E_R3_IDENTITY_ONLY_AUTHORIZATION',
            'diagnostic_execution_authorized':False,'science_execution_authorized':False,
            'source_commit':None,'contract_sha256':contract_hash,'manifest_sha256':manifest_hash,
            'runs':0,'retries':0,'mandatory_STOP':True,'explicit_user_instruction':None,
            'authorization_only_allowed_paths':[str(NEW/'authorization.json'),
                'docs/tracks/algorithm_codesign/e_r3_identity_diagnostic_execution_authorization_v2.md']}

def construction_prefix(data):
    tree=ast.parse(data);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='observe_path')
    prefix=[]
    for n in fn.body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='calls' for t in n.targets):break
        prefix.append(n)
    return ast.dump(ast.Module(body=prefix,type_ignores=[]),include_attributes=False)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--finalize',action='store_true');args=parser.parse_args()
    original=(ROOT/OLD/'contract_v1.json').read_bytes();current=(ROOT/NEW/'contract_v1.json').read_bytes()
    assert current==original,'Scientific diagnostic contract must remain byte-identical'
    assert not (ROOT/'artifacts/track_b_e_r3_identity_diagnostic_result/2026-10-12/v1').exists()
    assert not (ROOT/'artifacts/track_b_e_r3_identity_diagnostic_result/2026-10-12/v2').exists()
    if not args.finalize:
        write('authorization.json',pending(sha(current),None))
        write('focused_test_plan_v2.json',{
            'scope':'34 preparation regressions + 26 previously audited controls + 17 observer repair controls',
            'historical_bug_reproductions_deselected':2,'expected_tests':77,
            'fixtures':'same analytic 2-mode/rank1 control, separate 8x8 diagonal QSD control, metadata/mocks ONLY',
            'registered_diagnostic_calls':0,'native_transpile_body_calls':0,'new_synthesis_science_calls':0,
            'test_limits':{'CPU_seconds':60,'aggregate_RSS_MiB':512,'worker_RSS_MiB':448,
                           'output_bytes':8388608,'log_bytes':1048576,'wall_cap':None,'workers':1,'threads':1},
            'mandatory_STOP':True})
        return
    checks=[]
    def check(label,ok):
        checks.append({'check':label,'passed':bool(ok)})
        assert ok,label
    old=load(OLD/'source_manifest_v1.json')['files'];diff=[]
    for row in old:
        historical=git('show',S+':'+row['path'])
        check('original S object '+row['path'],len(historical)==row['bytes'] and sha(historical)==row['sha256'])
        now=identity(row['path'])
        if now!=row:
            check('intentional v2 source change '+row['path'],row['path'] in CHANGED)
            diff.append({'path':row['path'],'old':row,'new':now})
    check('only three protected source paths changed',{r['path'] for r in diff}==CHANGED)
    check('Q/T/U/input construction AST unchanged',construction_prefix(git('show',S+':'+CODE+'evaluation.py'))==construction_prefix((ROOT/(CODE+'evaluation.py')).read_bytes()))
    old_results=load('artifacts/track_b_df_correction_comparison_execution_audit/2026-10-12/saved_output_audit_v1.json')['result_file_inventory']
    for row in old_results:check('old result retained '+row['path'],identity(row['path'])==row)
    historical_artifacts=git('ls-tree','-r','--name-only',BASE,'artifacts/track_b_e_r3_identity_diagnostic_preparation',
        'artifacts/track_b_e_r3_identity_final_technical_audit').decode().splitlines()
    for path in historical_artifacts:check('old preparation/audit bytes '+path,(ROOT/path).read_bytes()==git('show',BASE+':'+path))
    check('contract remains exact bytes',current==original)
    t=load(NEW/'focused_tests_v2.json')
    check('focused77 pass',t['passed'] and t['test_counts']=={'tests':77,'failures':0,'errors':0,'skipped':0})
    for row in t['source_files']:check('tested source unchanged '+row['path'],identity(row['path'])==row)
    auth=load(NEW/'authorization.json')
    check('new pending only',auth==pending(sha(current),None))
    source_paths={r['path'] for r in old}|set(historical_artifacts)|{r['path'] for r in t['source_files']}
    source_paths|={str(NEW/p) for p in ['contract_v1.json','repair_scope_v2.json','focused_test_plan_v2.json','focused_tests_v2.json']}
    source_paths|={'PROJECT_MAP.md','docs/README.md','scripts/README.md','src/trotterlib/README.md',
                   'tests/README.md','docs/tracks/algorithm_codesign/README.md','docs/research/研究概要・現状.md',
                   'docs/research/research_project_instructions_gpt_codex_review_approval_2026-10-10.md',
                   'docs/tracks/algorithm_codesign/e_r3_identity_observer_repair_preparation_20261012.md',
                   'docs/research/研究ノート/2026-10-12_e_r3_identity_observer_repair.md'}
    check('no NPZ input in manifest',not any(p.lower().endswith('.npz') for p in source_paths))
    manifest={'kind':'E_R3_IDENTITY_DIAGNOSTIC_OBSERVER_REPAIR_SOURCE_MANIFEST_V2',
              'previous_source':S,'base_technical_audit':BASE,'intentional_v2_changes':diff,
              'files':[identity(p) for p in sorted(source_paths)],'self_and_new_authorization_excluded':True,
              'new_source_commit_in_publication_receipt':True,'diagnostic_execution_authorized':False,'mandatory_STOP':True}
    write('source_manifest_v1.json',manifest)
    # Only pending draft metadata written by THIS preparation is completed here.
    # The original authorization is immutable; a real approved child is never made.
    auth['manifest_sha256']=identity(NEW/'source_manifest_v1.json')['sha256']
    (ROOT/NEW/'authorization.json').write_text(json.dumps(auth,indent=2)+'\n')
    check('pending manifest binding',auth['manifest_sha256']==identity(NEW/'source_manifest_v1.json')['sha256'])
    write('source_scope_verification_v2.json',{
        'kind':'OBSERVER_REPAIR_V2_STATIC_AND_TEST_BOUND_AUDIT','checks':checks,'all_checks_passed':True,
        'previous_protected_count':len(old),'deliberate_changed_protected_paths':diff,
        'old_result_files_retained':len(old_results),'new_source_manifest_files':len(source_paths),
        'classification':'READY_FOR_SEPARATE_ONE_SHOT_AUTHORIZATION','registered_diagnostic_calls':0,
        'source_policy_expected_contract_caps_changed':False,'mandatory_STOP':True})
    write('STOP.json',{'status':'READY_FOR_SEPARATE_ONE_SHOT_AUTHORIZATION',
        'previous_source':S,'new_source_fixed_in_publication_receipt':True,
        'diagnostic_execution_authorized':False,'science_execution_authorized':False,
        'runs':0,'retries':0,'new_authorization_only_commit_created':False,'mandatory_STOP':True})
    print(json.dumps({'status':'READY_FOR_SEPARATE_ONE_SHOT_AUTHORIZATION','checks':len(checks),
                      'manifest_files':len(source_paths),'contract_bytes_identical':True,'mandatory_STOP':True}))

if __name__=='__main__':main()
