import json,hashlib,subprocess,datetime
from pathlib import Path
r=Path.cwd();bare=Path('/tmp/track_a_h4_supplement_remote_gq0j0iwr');ref='refs/heads/h6-matched-preparation-verify';meta=r/'artifacts/resource_applicability/track_a_h6_matched_execution_v3/2026-10-11';prep=r/'artifacts/resource_applicability/track_a_h6_matched_preparation_v3/2026-10-11'
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=r,text=True).strip();assert head=='c83e55095aaeb2a758826427ffec873e93d5c2bf'
assert subprocess.check_output(['git','rev-parse',ref],cwd=bare,text=True).strip()==head
m=json.loads((prep/'sealed_manifest_v3.json').read_text());expected={**m['source_hashes'],**m['validation_source_hashes'],**m['reuse_registry']['raw_hashes'],**m['input_identity']['parent_raw_hashes']}
for p in ['artifacts/resource_applicability/track_a_h6_matched_execution_v3/2026-10-11/authorization_v3.json','artifacts/resource_applicability/track_a_h6_matched_preparation_v3/2026-10-11/sealed_manifest_v3.json','docs/research/track_a_h6_matched_supplement_execution_contract_v3.md','artifacts/resource_applicability/track_a_h6_matched_preparation_v3/2026-10-11/README.md']:
 expected[p]=hashlib.sha256((r/p).read_bytes()).hexdigest()
names=list(expected)
for start in range(0,len(names),64):
 batch=names[start:start+64];raw=subprocess.check_output(['git','cat-file','--batch'],cwd=bare,input=''.join(head+':'+p+'\n' for p in batch).encode());pos=0
 for p in batch:
  end=raw.index(b'\n',pos);header=raw[pos:end].split();assert len(header)==3 and header[1]==b'blob',p
  size=int(header[2]);data=raw[end+1:end+1+size];pos=end+size+2;assert hashlib.sha256(data).hexdigest()==expected[p],p
for c in [m['source_commit'],m['reuse_registry']['source_commit'],m['reuse_registry']['result_commit'],'604e593e184903cfb7e34c118a153251783d9cba']:
 subprocess.run(['git','cat-file','-e',c+'^{commit}'],cwd=bare,check=True)
assert not Path(m['exclusive_output']['absolute_path']).exists()
obj={'status':'INDEPENDENT_REMOTE_PRELAUNCH_BYTES_PASS','publication_commit':head,'grant_commit':head,'grant_sha256':expected['artifacts/resource_applicability/track_a_h6_matched_execution_v3/2026-10-11/authorization_v3.json'],'source_commit':m['source_commit'],'recorded_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'verified_path_count':len(expected),'expected_hashes':expected,'bare_database':str(bare),'remote_ref':ref,'scientific_runner_invocations':0,'output_exists':False,'one_shot_grant_published':True,'next_stage_authorized':False}
with (meta/'remote_prelaunch_verification_v3.json').open('x') as f:json.dump(obj,f,sort_keys=True,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({k:obj[k] for k in ['status','publication_commit','grant_sha256','verified_path_count','output_exists']}))
