"""Read-only SOURCE/packet verification. No science, benchmark, or claim creation."""
import ast
import gzip
from hashlib import sha1,sha256
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
BASE='564409487a35e15d73f1d0f0c2611824b400ab02'
PREFIX=str(HERE.relative_to(ROOT))+'/'
D='src/trottertracks/algorithm_codesign/g6_w2c_b3_driver.py'
MODIFIED={D,*['tests/tracks/algorithm_codesign/test_'+n+'.py'for n in ('g6_w2c_b3_driver','g6_w2c_b3_admission','g6_w2c_b3_f01_provenance')]}

def git(*a):return subprocess.check_output(['git','-C',str(ROOT),*a],env=dict(os.environ,GIT_OPTIONAL_LOCKS='0'))
def sha(b):return sha256(b).hexdigest()
def load(n):return json.loads((HERE/n).read_text())
def decl(b):return {getattr(n,'name',ast.unparse(n)):ast.dump(n,include_attributes=False)for n in ast.parse(b).body}

def verify():
    inv=load('source_inventory.json');tests=load('artificial_test_results.json');perf=load('performance_results.json');compat=load('r1_compatibility.json')
    checks={};bad=[];preserved=0
    checks['base_ancestor']=git('merge-base',BASE,'HEAD').decode().strip()==BASE
    checks['origin_allowed']=git('remote','get-url','origin').decode().strip()=='git@github.com:HIROMU1015/Partially-Randomized-Trotter.git'
    for e in git('ls-tree','-r','-z',BASE).split(b'\0'):
        if not e:continue
        meta,name=e.split(b'\t',1);mode,kind,oid=meta.split();p=os.fsdecode(name)
        if p in MODIFIED:continue
        f=ROOT/p;b=os.fsencode(os.readlink(f))if f.is_symlink()else f.read_bytes()if f.is_file()else None
        if b is None or sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()!=oid.decode():bad.append(p)
        preserved+=1
    checks['all_other_base_bytes_preserved']=not bad
    checks['protected_science_control_bytes_unchanged']=all(sha((ROOT/p).read_bytes())==h and (ROOT/p).read_bytes()==git('show',BASE+':'+p)for p,h in inv['protected_source_pins_unchanged'].items())
    checks['new_source_full_SHA']=all(sha((ROOT/r['path']).read_bytes())==r['sha256']and(ROOT/r['path']).stat().st_size==r['bytes']for r in inv['new_or_modified_sources_and_tests'])
    checks['fixed_public_blob_SHA']=all(sha(git('show',r['commit']+':'+r['path']))==r['sha256']and r['public_bytes_equal_git_blob']for r in inv['frozen_public_evidence'])
    old=decl(git('show',BASE+':'+D));new=decl((ROOT/D).read_bytes())
    allowed={'identity','Stream','candidate_errors','run_pair','execute'}
    changed=[k for k in old if old[k]!=new.get(k)]
    checks['driver_change_scope']=set(changed)<=allowed
    checks['Meter_Observer_Boundary_endpoint_seed_unchanged']=all(old[n]==new[n]for n in ('Meter','FunctionObserver','Boundary','LegacyEndpoint','seed_for','load_modules','check_pins','load_saved','schedule'))
    text=(ROOT/D).read_text()
    restored=text.replace('return sha256(serialization.canonical(value).encode()).hexdigest()','return audit.fingerprint(exact(value))').replace('serialization.canonical(payload)','audit.canonical(exact(payload))').replace("serialization.canonical({'kind': result.kind, 'event': result.event})","audit.canonical(exact({'kind': result.kind, 'event': result.event}))").replace('serialization.canonical(provenance)','audit.canonical(exact(provenance))')
    norm=decl(restored)
    checks['F01_and_execution_only_codec_call_replacements']=all(old[n]==norm[n]for n in ('identity','Stream','run_pair','execute'))
    checks['codec_pin_required']='EXACT_SERIALIZATION_SOURCE_PIN_REQUIRED'in text and 'FASTPATH_SOURCE'in text
    ids=[i for row in tests['suites']for i in row['collected']]
    checks['339_unique_PASS']=tests['pass']==tests['collected_unique']==len(ids)==len(set(ids))==339 and tests['existing_tests']==275 and tests['new_tests']==64 and all(row['successful']and row['exit_code']==0and row['fail']==row['error']==row['skip']==0and not row['forbidden_science_modules_imported']and sha((HERE/row['log']).read_bytes())==row['log_sha256']for row in tests['suites'])
    raw=gzip.decompress((HERE/'benchmark_results.json.gz').read_bytes());b=json.loads(raw)
    checks['benchmark_artifact_SHA']=sha(raw)==perf['uncompressed_measurements_sha256']and sha((HERE/'benchmark_results.json.gz').read_bytes())==perf['compressed_measurements_sha256']
    checks['four_variants_five_repeats']=len(b['measurements'])==40 and len(b['summary'])==8 and {r['variant']for r in b['measurements']}=={'baseline','fused','recurrence','combined'}and all(r['repeats']==5for r in b['summary'])
    exact=True
    for width in (96,256):
        for repeat in range(5):
            group=[r for r in b['measurements']if r['width']==width and r['repeat']==repeat];ref=next(r for r in group if r['variant']=='baseline')
            for r in group:
                for x,y in zip(ref['pairs'],r['pairs']):
                    exact &=x['summary']==y['summary']and all(x['readback'][k]==y['readback'][k]for k in ('normalized_file_sha256','normalized_chain_sha256','normalized_bytes','rows'))
    checks['exact_artificial_performance_results']=exact and perf['summary']==b['summary']
    checks['measured_source_pins_current']=all(sha((ROOT/p).read_bytes())==h for p,h in b['source_pins'].items())
    checks['A_B_classification_no_adoption']=[r['classification']for r in compat['candidates']]==['A','B','B']and compat['adopted']is False and all(compat[k]is None for k in ('science_release_commit','formal_execution_contract_digest','real_authorization','real_scope_claim'))
    checks['no_real_science_authority']=tests['scope']['real_G7_sampling_correctness']==tests['scope']['real_F3_sessions']==tests['scope']['real_authorization_claim_issuance_consumption']==0 and b['real_G7_science_calls']==b['real_authorization_claim_issuance_consumption']==0
    checks['recovered_bundle_verified']=inv['prototype_recovery']['bundle_verify_passed']and inv['prototype_recovery']['bundle_sha256']=='d8b6a7da4be2947d4b2d1e2f648e96148eb793917c219ed6f5fd16e632735270'
    checks['bundle_verify_raw_log_SHA']=sha(gzip.decompress((HERE/'bundle_verify.raw.log.gz').read_bytes()))==inv['prototype_recovery']['bundle_verify_raw_log_sha256']
    allowed_paths={r['path']for r in inv['new_or_modified_sources_and_tests']}
    paths=git('diff','--name-only',BASE).decode().splitlines()
    paths +=[s[3:]for s in git('status','--porcelain=v1','--untracked-files=all').decode().splitlines()if s.startswith('?? ')]
    checks['dedicated_change_scope']=all(p.startswith(PREFIX)or p in allowed_paths for p in paths)
    if 'management_verifier'in inv:checks['verifier_pin']=sha(Path(__file__).read_bytes())==inv['management_verifier']['sha256']
    return dict(schema='b3_exact_fastpath_verification_v1',passed=all(checks.values()),checks=checks,
        unchanged_base_tracked_files_checked=preserved,unexpected_base_changes=bad,explicitly_modified_base_files=sorted(MODIFIED),
        fixed_public_files=len(inv['frozen_public_evidence']),existing_tests=275,new_tests=64,total_PASS=339,
        real_G7_science_calls=0,real_F3_sessions=0,real_authorization_claim_issuance_consumption=0,
        artificial_precision_reference_execution_authorized=True,scientific_SOURCE_files_unchanged=True,
        final_science_release_created=False,formal_science_contract_digest_created=False,mandatory_STOP=True,local_observation_not_CI=True)

if __name__=='__main__':
    result=verify();print(json.dumps(result,ensure_ascii=False,indent=2));sys.exit(0 if result['passed']else 1)
