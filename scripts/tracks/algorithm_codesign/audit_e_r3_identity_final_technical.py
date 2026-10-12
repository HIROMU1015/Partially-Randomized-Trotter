"""Read-only frozen evidence verification and optional GitHub byte retrieval.

Writes only NEW technical-audit artifacts. Does not import numeric modules.
"""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[3]
S='8377c1d12439cabab3b97bcc1f57ac030a479785'
P='0cf991d4371250dc6a2d8b346ed4ed7414f4a198'
PREP=Path('artifacts/track_b_e_r3_identity_diagnostic_preparation/2026-10-12')
ART=ROOT/'artifacts/track_b_e_r3_identity_final_technical_audit/2026-10-12'
RESULT=Path('artifacts/track_b_e_r3_identity_diagnostic_result/2026-10-12/v1')

def sha(b): return hashlib.sha256(b).hexdigest()
def git(*args): return subprocess.check_output(['git',*args],cwd=ROOT)
def load(path): return json.loads((ROOT/path).read_bytes())
def write(name,data):
    ART.mkdir(parents=True,exist_ok=True)
    with (ART/name).open('x') as f: json.dump(data,f,ensure_ascii=False,indent=2);f.write('\n')
def identity(path):
    b=(ROOT/path).read_bytes();return {'path':str(path),'bytes':len(b),'sha256':sha(b)}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--remote',action='store_true');args=parser.parse_args()
    m=load(PREP/'source_manifest_v1.json');c=load(PREP/'contract_v1.json');a=load(PREP/'authorization.json')
    rows=[]
    for r in m['files']:
        frozen=git('show',S+':'+r['path']);actual=identity(r['path'])
        rows.append(dict(r,local_matches=actual==r,git_S_matches=len(frozen)==r['bytes'] and sha(frozen)==r['sha256']))
    extras=[PREP/'authorization.json',PREP/'source_manifest_v1.json',PREP/'focused_tests_v2.json',
            PREP/'source_scope_audit_v1.json',PREP/'publication_receipt_v1.json',
            Path('docs/tracks/algorithm_codesign/e_r3_identity_diagnostic_source_review_and_gpt_handoff_20261012.md')]
    old=load('artifacts/track_b_df_correction_comparison_execution_audit/2026-10-12/saved_output_audit_v1.json')
    old_rows=[dict(r,local_matches=identity(r['path'])==r) for r in old['result_file_inventory']]
    job=load(c['inputs']['saved_job']['path']);inputs=load(c['inputs']['saved_inputs']['path'])
    expected=load(c['inputs']['historical_expected_record']['path'])
    oldmarker=ROOT/'artifacts/track_b_df_correction_comparison_result/2026-10-11/v2/one_shot_marker.json'
    checks={
        'protected_260_local_and_S':len(rows)==260 and all(r['local_matches'] and r['git_S_matches'] for r in rows),
        'old_result_inventory_unchanged':all(r['local_matches'] for r in old_rows),
        'pending_authorization':a['runs']==0 and a['source_commit'] is None and not a['diagnostic_execution_authorized'] and not a['science_execution_authorized'],
        'contract_binding':a['contract_sha256']==sha((ROOT/PREP/'contract_v1.json').read_bytes()),
        'manifest_binding':a['manifest_sha256']==sha((ROOT/PREP/'source_manifest_v1.json').read_bytes()),
        'saved_job_frames_match_r3_metadata':job['r']==3 and job['stage']=='native' and job['frames']==inputs['3'],
        'saved_rank_shapes_only':len(job['frames'])==3 and all(len(v)==3 and all(len(row)==2 for row in v) for v in job['frames']),
        'old_marker_identity':sha(oldmarker.read_bytes())==c['old_marker_sha256']==job['marker_sha256'],
        'historical_expected_unchanged':c['historical_expected_fingerprint']==next(r['circuit_fingerprint'] for r in expected['central_compilations'] if r['active_modes']==3 and r['controlled'] is False),
        'path_scope':c['path_order']==['current','legacy'] and c['scope']['signs']==[1,-1,1] and c['scope']['controlled'] is False and c['scope']['nested_commutator']=='comm(Q0,comm(Q1,Q2))' and c['scope']['raw_rotation']=='expm(1j*.03125*T/4)',
        'new_result_namespace_absent':not (ROOT/RESULT).exists(),
        'publication_is_receipt_only_child_of_S':git('rev-list','--parents','-n','1',P).decode().split()==[P,S] and git('diff','--name-only',S,P).decode().splitlines()==[str(PREP/'publication_receipt_v1.json')],
        'saved34_PASS_bound':load(PREP/'focused_tests_v2.json')['test_counts']=={'tests':34,'failures':0,'errors':0,'skipped':0},
        'scope_audit_saved_PASS':load(PREP/'source_scope_audit_v1.json')['all_checks_passed'],
    }
    ev=ast.parse((ROOT/'src/trottertracks/algorithm_codesign/e_r3_identity_diagnostic/evaluation.py').read_bytes())
    names=[n.func.id if isinstance(n.func,ast.Name) else n.func.attr if isinstance(n.func,ast.Attribute) else '' for n in ast.walk(ev) if isinstance(n,ast.Call)]
    checks['static_calls_one_payload_one_expm_no_prohibited_direct_calls']=names.count('_circuit_payload')==1 and names.count('expm')==1 and not set(names)&{'transpile','synthesize','cd_task','e_recipe','pauli_table','Operator','default_rng'}
    result={'kind':'FROZEN_S_FINAL_TECHNICAL_IDENTITY_AUDIT','source':S,'publication':P,'checks':checks,
            'all_identity_checks_passed':all(checks.values()),'protected_files':rows,'old_result_files':old_rows,
            'additional_metadata':[identity(p) for p in extras],
            'frame_hex_metadata_only_no_array_construction':True,'registered_Q_T_U_payload_calls':0,'mandatory_STOP':True}
    if args.remote:
        selections={r['path']:dict(r) for r in m['files']}
        for path in extras:selections.setdefault(str(path),identity(path))
        def fetch(row):
            ref=P if row['path']==str(PREP/'publication_receipt_v1.json') else S
            url='https://raw.githubusercontent.com/HIROMU1015/Partially-Randomized-Trotter/'+ref+'/'+quote(row['path'],safe='/')
            try:
                with urlopen(url,timeout=30) as response: b=response.read()
                return dict(row,ref=ref,url=url,fetched_bytes=len(b),fetched_sha256=sha(b),
                            matches=len(b)==row['bytes'] and sha(b)==row['sha256'],failure=None)
            except HTTPError as e:
                return dict(row,ref=ref,url=url,matches=False,failure='HTTP_'+str(e.code),
                            failure_class='HTTP_NOT_FOUND_REF_OR_PATH_NEEDS_DIAGNOSIS' if e.code==404 else 'HTTP_ACCESS_OR_SERVICE_ERROR')
            except (URLError,TimeoutError,OSError) as e:
                return dict(row,ref=ref,url=url,matches=False,failure=type(e).__name__+': '+str(e),failure_class='CONNECTION_ERROR_NOT_UNPUSHED_EVIDENCE')
        with ThreadPoolExecutor(max_workers=4) as pool:fetched=list(pool.map(fetch,selections.values()))
        remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/track-b-e-r3-identity-diagnostic-preparation-20261012'],cwd=ROOT,text=True).split()
        result['GitHub_retrieval']={'files':fetched,'all_bytes_match':all(r['matches'] for r in fetched),
                                    'preparation_branch_remote_SHA':remote[0] if remote else None,
                                    'expected_publication':P,'remote_matches':bool(remote) and remote[0]==P,'fetch_workers':4}
    write('identity_and_remote_audit_v1.json' if args.remote else 'identity_audit_v1.json',result)
    print(json.dumps({'identity_checks':checks,'files':len(rows),
                      'remote_files':len(result.get('GitHub_retrieval',{}).get('files',[])),
                      'remote_all_match':result.get('GitHub_retrieval',{}).get('all_bytes_match')}))
    if not all(checks.values()) or args.remote and not result['GitHub_retrieval']['all_bytes_match']:raise SystemExit(1)

if __name__=='__main__':main()
