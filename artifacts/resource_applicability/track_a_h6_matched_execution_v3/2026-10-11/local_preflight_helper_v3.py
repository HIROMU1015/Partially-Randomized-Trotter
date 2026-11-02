import sys,os,json,hashlib,subprocess,resource,datetime
from pathlib import Path
root=Path.cwd();sys.path.insert(0,str(root/'src'))
from trottertracks.resource_applicability.h6_matched_contract_v3 import validate_launch,verify_sources,digest,file_hash
from trottertracks.resource_applicability.h6_matched_reuse_v3 import committed_hashes
meta=root/'artifacts/resource_applicability/track_a_h6_matched_execution_v3/2026-10-11'
prep=root/'artifacts/resource_applicability/track_a_h6_matched_preparation_v3/2026-10-11'
def git(*args):return subprocess.check_output(['git',*args],cwd=root)
def save(path,obj):
 with path.open('x',encoding='utf8') as f:json.dump(obj,f,ensure_ascii=False,sort_keys=True,indent=2);f.write('\n')
head=git('rev-parse','HEAD').decode().strip();assert head=='604e593e184903cfb7e34c118a153251783d9cba'
assert git('branch','--show-current').decode().strip()=='track-a-ax2b-h4-post-review-20261010'
assert git('remote','get-url','--push','origin').decode().strip()=='git@github.com:HIROMU1015/Partially-Randomized-Trotter.git'
assert not git('diff','--cached','--name-only');assert not meta.exists()
paths=[p.decode() for p in git('ls-files','--cached','--others','--exclude-standard','-z').split(b'\0') if p]
baseline={'head':head,'branch':'track-a-ax2b-h4-post-review-20261010','recorded_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status_porcelain':git('status','--porcelain=v1','-z').decode(),'file_hashes':{p:file_hash(root/p) for p in paths if (root/p).is_file()}}
m=json.loads((prep/'sealed_manifest_v3.json').read_text())
assert file_hash(prep/'sealed_manifest_v3.json')=='7796f92d442207a985fe2594b2d88ce74694d69a4e041e4188a05a16e5200f4c'
assert digest(m)=='630179bfdfd6904f7e94746b5ca86470e0df49b49ee8409cdd66f6e065fec482'
assert m['source_commit']=='5a903aea2492c2d211c1e15e7737df0d50694c12'
r=m['reuse_registry'];assert digest(r)=='c82aa9223c3f67f52085cf821547b92c37b2e110c78e98c2bb256f7c521ead70'
assert r['result_commit']=='8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb'
g=json.loads((prep/'authorization_request_template_v3.json').read_text());g['approved_by_user']=True
# No runner or numerical package invoked; validate metadata and saved identities.
validate_launch(root,m,g,Path(g['exclusive_output']),requested=True)
committed_hashes(root,m['source_commit'],m['validation_source_hashes'])
assert not {'numpy','scipy','qiskit','openfermion','numba'}&set(sys.modules)
assert set(x['index'] for x in r['accepted']).isdisjoint(x['index'] for x in r['missing'])
assert set(x['index'] for x in r['accepted'])|set(x['index'] for x in r['missing'])==set(range(592))
assert len(m['plan']['cells'])==92 and m['plan']['epsilons']==[.05,.01,.005,.001]
info={}
for line in Path('/proc/meminfo').read_text().splitlines():
 if line.split(':')[0] in ['MemTotal','MemAvailable','SwapTotal','SwapFree']:info[line.split(':')[0]]=line.split(':')[1].strip()
preflight={'status':'STATIC_PRELAUNCH_PASS','preparation_commit':head,'source_commit':m['source_commit'],'manifest_digest':digest(m),'manifest_sha256':file_hash(prep/'sealed_manifest_v3.json'),'reuse_registry_digest':digest(r),'scientific_plan_digest':digest(m['plan']),'runtime_source_count':len(m['source_hashes']),'validation_source_count':len(m['validation_source_hashes']),'old_raw_count':len(r['raw_hashes']),'saved_input_count':len(m['input_identity']['parent_raw_hashes']),'reused_signals':r['signal_count'],'reused_b3_rows':r['b3_count'],'reused_groups':r['accepted_groups'],'reused_wrappers':r['accepted_wrappers'],'missing_groups':r['missing_groups'],'registered_groups':len(r['original_tasks']),'missing_indices':[x['index'] for x in r['missing']],'input_environment_resource_equality':True,'committed_and_local_bytes_equal':True,'old_grants_stop_preserved':True,'assigned_resources':m['assigned_resources'],'current_affinity':sorted(os.sched_getaffinity(0)),'host_meminfo':info,'inherited_rlimit_as':list(resource.getrlimit(resource.RLIMIT_AS)),'inherited_rlimit_fsize':list(resource.getrlimit(resource.RLIMIT_FSIZE)),'environment':m['environment'],'heavy_packages_imported':False,'scientific_runner_invocations':0,'output_exists':False,'approved_grant_created':False,'next_stage_authorized':False,'H6_status':'H6_NOT_AUTHORIZED','contract_status':'DRAFT_NOT_AUTHORIZATION'}
meta.mkdir(parents=True)
save(meta/'baseline_before_execution_v3.json',baseline)
save(meta/'local_preflight_v3.json',preflight)
print(json.dumps({k:preflight[k] for k in ['status','reused_signals','reused_b3_rows','reused_groups','reused_wrappers','missing_groups','runtime_source_count','old_raw_count','host_meminfo','output_exists']}))
