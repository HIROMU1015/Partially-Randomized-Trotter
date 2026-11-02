"""Read-only technical packet verifier; never imports/executes science or authority."""
import ast
from hashlib import sha1, sha256
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PREFIX = str(HERE.relative_to(ROOT))+'/'
BASE = '0619178152a5365275c96a374d85f10ee346f912'
SOURCES = ['src/trottertracks/algorithm_codesign/g6_w2c_b3_admission.py',
           'tests/tracks/algorithm_codesign/test_g6_w2c_b3_admission.py']

def git(*args):
    return subprocess.check_output(['git','-C',str(ROOT),*args],env=dict(os.environ,GIT_OPTIONAL_LOCKS='0'))

def load(name):
    return json.loads((HERE/name).read_text())

def digest(b):
    return sha256(b).hexdigest()

def verify():
    inventory=load('source_inventory.json')
    adoption=load('r1_adoption_record.json')
    prep=load('execution_contract_preparation.json')
    tests=load('artificial_test_results.json')
    checks={}
    checks['base_is_ancestor']=git('merge-base',BASE,'HEAD').decode().strip()==BASE
    checks['origin_allowed']=git('remote','get-url','origin').decode().strip()=='git@github.com:HIROMU1015/Partially-Randomized-Trotter.git'
    entries=git('ls-tree','-r','-z',BASE).split(b'\0')
    changed_old=[]
    oldpaths=[]
    # Git blob identity checked from bytes, using NUL-delimited names (Unicode safe).
    for entry in entries:
        if not entry:continue
        meta,name=entry.split(b'\t',1)
        mode,kind,oid=meta.split()
        p=os.fsdecode(name);oldpaths.append(p)
        f=ROOT/p
        if mode==b'120000':
            data=os.fsencode(os.readlink(f)) if f.is_symlink() else None
        else:
            data=f.read_bytes() if f.is_file() else None
        if data is None or sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()!=oid.decode():
            changed_old.append(p)
    checks['all_base_tracked_bytes_unchanged']=not changed_old
    checks['legacy_source_pins_unchanged']=all(digest((ROOT/p).read_bytes())==h for p,h in inventory['legacy_source_pins_unchanged'].items())
    checks['readiness_source_pins_unchanged']=all(digest((ROOT/p).read_bytes())==h for p,h in inventory['prior_readiness_source_pins_unchanged'].items())
    checks['new_source_bytes_sha']=all((ROOT/r['path']).stat().st_size==r['bytes'] and digest((ROOT/r['path']).read_bytes())==r['sha256'] for r in inventory['new_sources'])
    checks['fixed_evidence_blob_sha']=all(digest(git('show',r['commit']+':'+r['path']))==r['sha256'] and r['public_bytes_equal_git_blob'] for r in inventory['frozen_public_evidence'])
    checks['r1_adopted_without_science']=adoption['status']=='R1_CONTENT_FORMALLY_ADOPTED' and adoption['technical_implementation_authorized'] is True and adoption['science_execution_authorized'] is False and adoption['real_authorization_claim_issuance_consumption_authorized'] is False
    checks['a6_exact']=adoption['angle']=={'id':'A6','tau':'1/1000000','K_phi':32,'J':32,'mode':'MATHEMATICAL_ENCLOSURE_ONLY_NO_PHYSICAL_CHANGE'}
    checks['adoption_pin']=prep['r1_adoption_record_sha256']==digest((HERE/'r1_adoption_record.json').read_bytes())
    checks['r1_matches_frozen_spec']=adoption['fixed_science']==prep['fixed_science']==json.loads((ROOT/'docs/research/track_b_g6_w2c_b3_adoption_readiness_20261019/execution_contract_candidate_refined.json').read_text())['fixed_science']
    checks['future_values_null']=all(v is None for v in prep['future_finalization'].values())
    checks['unapproved_f01_pending']=prep['run_ready'] is False and prep['contract_grants_execution'] is False and prep['separate_execution_approval'] is False and prep['status']=='F01_INTEGRATION_PENDING'
    checks['budgets_unchanged']=prep['fixed_science']['call_budget']['total_sampling_calls']==31710848 and prep['fixed_science']['call_budget']['correctness_sampling_calls']==256 and prep['fixed_science']['call_budget']['baseline_full_N_sampling_calls']==15855296 and prep['fixed_science']['call_budget']['instrumented_full_N_sampling_calls']==15855296
    checks['tests_pass_and_logs_match']=tests['status']=='PASS' and tests['total_tests']==183 and sum(r['tests'] for r in tests['runs'])==183 and all(r['exit_code']==0 and digest((HERE/r['log']).read_bytes())==r['log_sha256'] and (HERE/r['log']).read_bytes().rstrip().endswith(b'OK') for r in tests['runs'])
    checks['no_real_science_authority']=tests['real_science_calls']==tests['real_authorization_claim_issuance_consumption']==0 and tests['scientific_modules_imported'] is False
    tree=ast.parse((ROOT/SOURCES[0]).read_text())
    classes={n.name:n for n in tree.body if isinstance(n,ast.ClassDef)}
    checks['artificial_port_no_admit_once']=not any(isinstance(n,ast.FunctionDef) and n.name=='admit_once' for n in classes['ArtificialAdmission'].body)
    # Pending management receipt may be added later; changed files must be new.
    added=git('status','--porcelain=v1','--untracked-files=all').decode().splitlines()
    changed=git('diff','--name-only',BASE).decode().splitlines()+[line[3:] for line in added if line.startswith('?? ')]
    checks['dedicated_scope_only']=all(p.startswith(PREFIX) or p in SOURCES for p in changed)
    if 'management_verifier' in inventory:
        checks['verifier_pin']=inventory['management_verifier']['sha256']==digest(Path(__file__).read_bytes())
    return {'schema':'b3_r1_admission_preparation_verification_v1','passed':all(checks.values()),'checks':checks,'all_base_tracked_files':len(oldpaths),'changed_base_files':changed_old,'frozen_public_files':len(inventory['frozen_public_evidence']),'new_artificial_tests':69,'total_artificial_tests':183,'real_science_calls':0,'real_authorization_claim_issuance_consumption':0,'F01_INTEGRATION_PENDING':True,'final_source_bundle_frozen':False,'mandatory_STOP':True,'local_observation_not_CI':True}

if __name__=='__main__':
    result=verify()
    print(json.dumps(result,ensure_ascii=False,indent=2))
    sys.exit(0 if result['passed'] else 1)
