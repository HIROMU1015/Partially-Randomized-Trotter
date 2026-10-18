"""Read-only integration verification. Never runs science or initializes authority."""
import ast
from hashlib import sha1, sha256
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
BASE='1c15bacab349b8686c09d2981e1c0bc70e9c3ec9'
F01='a777a2ab54d85a95dce91a06e65d4ec0511403ae'
PREFIX=str(HERE.relative_to(ROOT))+'/'
DRIVER='src/trottertracks/algorithm_codesign/g6_w2c_b3_driver.py'
ADM='src/trottertracks/algorithm_codesign/g6_w2c_b3_admission.py'
HOST='src/trottertracks/algorithm_codesign/g6_w2c_b3_host.py'
TEST69='tests/tracks/algorithm_codesign/test_g6_w2c_b3_admission.py'
TESTF='tests/tracks/algorithm_codesign/test_g6_w2c_b3_f01_provenance.py'
FIX='tests/tracks/algorithm_codesign/fixtures/g6_w2c_b3_f01_normal_reference.json'
TESTH='tests/tracks/algorithm_codesign/test_g6_w2c_b3_host_integration.py'
SOURCES={DRIVER,ADM,HOST,TEST69,TESTF,FIX,TESTH}
MODIFIED_BASE={DRIVER,ADM,TEST69}

def git(*a):return subprocess.check_output(['git','-C',str(ROOT),*a],env=dict(os.environ,GIT_OPTIONAL_LOCKS='0'))
def sha(b):return sha256(b).hexdigest()
def load(n):return json.loads((HERE/n).read_text())
def decl(b):return {getattr(n,'name',ast.unparse(n)):ast.dump(n,include_attributes=False)for n in ast.parse(b).body}

def verify():
    inv=load('source_inventory.json');manifest=load('integration_manifest.json')
    tests=load('artificial_test_results.json');tech=load('technical_contract_candidate.json')
    template=tech['formal_v2_template_not_executable'];c=template['candidate']
    base=json.loads((ROOT/'docs/research/track_b_g6_w2c_b3_driver_preparation_20261019/execution_contract_candidate.json').read_text())
    checks={}
    checks['base_ancestor']=git('merge-base',BASE,'HEAD').decode().strip()==BASE
    checks['origin_allowed']=git('remote','get-url','origin').decode().strip()=='git@github.com:HIROMU1015/Partially-Randomized-Trotter.git'
    changed=[];preserved=0
    for e in git('ls-tree','-r','-z',BASE).split(b'\0'):
        if not e:continue
        meta,name=e.split(b'\t',1);mode,kind,oid=meta.split();p=os.fsdecode(name)
        if p in MODIFIED_BASE:continue
        f=ROOT/p
        b=(os.fsencode(os.readlink(f))if f.is_symlink()else f.read_bytes()if f.is_file()else None)
        if b is None or sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()!=oid.decode():changed.append(p)
        preserved+=1
    checks['all_other_base_tracked_bytes_unchanged']=not changed
    checks['source_complete_SHA256']=all((ROOT/r['path']).stat().st_size==r['bytes']and sha((ROOT/r['path']).read_bytes())==r['sha256']for r in inv['integrated_sources_and_tests'])
    checks['fixed_public_evidence_blobs']=all(sha(git('show',r['commit']+':'+r['path']))==r['sha256']and r['public_bytes_equal_git_blob']for r in inv['frozen_public_evidence'])
    checks['science_pins_unchanged']=all(sha((ROOT/p).read_bytes())==h for p,h in inv['unchanged_science_pins'].items())
    guard=inv['unchanged_readiness_guard'];checks['guard_unchanged']=sha((ROOT/guard['path']).read_bytes())==guard['sha256']
    checks['driver_exact_fixed_F01']=(ROOT/DRIVER).read_bytes()==git('show',F01+':'+DRIVER)
    checks['old_driver_still_rejected']=manifest['old_driver_sha256']=='d7947af16baccbd3e0707fa9b4ffcd26650a6569eb6606ee3c08ec443d1dd7ee' and "!= OLD_DRIVER_SHA" in (ROOT/ADM).read_text()
    oldD,newD=decl(git('show',BASE+':'+DRIVER)),decl((ROOT/DRIVER).read_bytes())
    checks['driver_only_F01_declarations_changed']=[k for k in oldD if oldD[k]!=newD[k]]==['Stream','run_pair','execute']
    oldA,newA=decl(git('show',BASE+':'+ADM)),decl((ROOT/ADM).read_bytes())
    checks['durable_ledger_and_independent_approval_unchanged']=all(oldA[k]==newA[k]for k in ('ClaimLedger','PinnedApproval','TrustAnchor'))
    checks['F01_test_fixture_fixed']=(ROOT/TESTF).read_bytes()==git('show',F01+':'+TESTF)and(ROOT/FIX).read_bytes()==git('show',F01+':'+FIX)
    checks['R1_fields_frozen']=all(c[k]==base[k]for k in ('input','policy','cells','saved_evidence','saved_rows','passes','correctness_trials_per_cell','call_budget','acceptance','one_shot','seed_proposal','schedule_proposal'))
    checks['A6_exact']=c['decisions']['angle']=={'tau':'1/1000000','K_phi':32,'J':32,'mode':'MATHEMATICAL_ENCLOSURE_ONLY_NO_PHYSICAL_CHANGE'}
    checks['host_v2_required']=template['schema']=='b3_formal_admission_v2'and template['source_roles']['host']==[HOST]and template['entrypoint']=={'path':HOST,'sha256':sha((ROOT/HOST).read_bytes()),'api':'execute_approved_v1'}
    checks['technical_only_no_final_release']=tech['run_ready']is False and tech['contract_grants_execution']is False and all(tech[k]is None for k in ('science_release_commit','formal_execution_contract_digest','real_authorization','fresh_scope_claim')) and template['release']['commit']is None and template['scope']is None and template['claim_id']is None
    checks['technical_candidate_byte_hash']=manifest['technical_candidate_file_sha256']==sha((HERE/'technical_contract_candidate.json').read_bytes())
    ids=[i for row in tests['suites']for i in row['collected']]
    checks['unique_275_tests_PASS']=tests['status']=='PASS'and len(ids)==len(set(ids))==275 and tests['existing_tests']==209 and tests['new_integration_tests']==66 and all(row['successful']and row['exit_code']==0 and row['fail']==row['error']==row['skip']==0 and sha((HERE/row['log']).read_bytes())==row['log_sha256']and not row['forbidden_science_modules_imported']for row in tests['suites'])
    checks['no_real_science_authority']=manifest['real_science_calls']==manifest['real_authorization_claim_issuance_consumption']==0 and tests['artificial_scope']['real_G7_import_sampling_correctness_profiling']==0
    checks['no_performance_result_or_other_worktree_access']=inv['performance_results_accessed']is False and inv['other_worktree_interiors_index_lock_process_accessed']is False
    paths=git('diff','--name-only',BASE).decode().splitlines()
    for line in git('status','--porcelain=v1','--untracked-files=all').decode().splitlines():
        if line.startswith('?? '):paths.append(line[3:])
    checks['dedicated_change_scope']=all(p.startswith(PREFIX)or p in SOURCES for p in paths)
    if 'management_verifier'in inv:checks['verifier_pin']=sha(Path(__file__).read_bytes())==inv['management_verifier']['sha256']
    for p in HERE.glob('*.json'):
        if p.name != 'verification.json':json.loads(p.read_text())  # Output may be redirected here.
    return dict(schema='b3_r1_f01_integration_verification_v1',passed=all(checks.values()),checks=checks,
        unchanged_base_tracked_files_checked=preserved,unexpected_changed_base_files=changed,
        explicitly_modified_base_files=sorted(MODIFIED_BASE),fixed_public_files=len(inv['frozen_public_evidence']),
        existing_tests=209,new_integration_tests=66,total_PASS=275,real_science_calls=0,
        real_authorization_claim_issuance_consumption=0,technical_integration_complete=True,
        final_science_source_release_fixed=False,formal_science_contract_digest_fixed=False,
        local_observation_not_CI=True,mandatory_STOP=True)

if __name__=='__main__':
    r=verify();print(json.dumps(r,ensure_ascii=False,indent=2));sys.exit(0 if r['passed']else 1)
