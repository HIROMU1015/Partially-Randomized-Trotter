"""Metadata-only remote blob verification after a separately completed GitHub fetch.
No scientific imports, numeric aggregation, source execution, or network mutations.
"""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import subprocess,json,hashlib,importlib.util,sys
W=Path('/home/abe/Project/Partially Randomized Trotter/.worktrees/qc-onebody-repartition-h-boundary-preparation-v1-20261015');ROOT=W.parents[1];A='artifacts/qc_m_fixed_circuit_k0/2026-10-15';BASE='0e923f73754d1ebd168bf21a1b9e73971946e603';BRANCH='qc-m-fixed-circuit-k0-audit-20261015';CACHE=Path('/tmp/prt-qc-candidates-published-20261013.git');FINAL='--final' in sys.argv
local=lambda *a:subprocess.check_output(['git',*a],cwd=W,stderr=subprocess.PIPE)
remote=lambda *a:subprocess.check_output(['git','--git-dir='+str(CACHE),*a],stderr=subprocess.PIPE)
h=local('rev-parse','HEAD').decode().strip();assert h==remote('rev-parse','FETCH_HEAD').decode().strip();assert not local('status','--porcelain=v1','--untracked-files=all').strip();assert not(CACHE/'objects/info/alternates').exists()
assert local('branch','--show-current').decode().strip()==BRANCH
assert local('remote','get-url','origin').decode().strip()=='git@github.com:HIROMU1015/Partially-Randomized-Trotter.git'
batch=subprocess.Popen(['git','--git-dir='+str(CACHE),'cat-file','--batch'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
def blob(ref,path,compare=False,collect=False):
 batch.stdin.write((ref+':'+path+'\n').encode());batch.stdin.flush();hdr=batch.stdout.readline().decode().split();assert len(hdr)==3 and hdr[1]=='blob',(ref,path,hdr)
 left=int(hdr[2]);size=left;sha=hashlib.sha256();pieces=[];stream=(W/path).open('rb') if compare else None
 while left:
  b=batch.stdout.read(min(left,1048576));assert b;left-=len(b);sha.update(b)
  if collect:pieces.append(b)
  if stream:assert stream.read(len(b))==b,path
 if stream:assert not stream.read(1);stream.close()
 assert batch.stdout.read(1)==b'\n';m={'SHA256':sha.hexdigest(),'bytes':size};return(m,b''.join(pieces)) if collect else m
read=lambda p:json.loads(blob(h,p,collect=True)[1])
changed=[p.decode() for p in local('diff','--name-only','-z',BASE,h).split(b'\0') if p];verified={p:blob(h,p,True) for p in changed}
protected=read(A+'/preservation_index.json')['files'];assert len(protected)==5177
for p,x in protected.items():assert blob(h,p,True)==blob(BASE,p)==x,p
review=read(A+'/review_identity.json');assert blob(h,review['saved_path'])==review['saved_identity'];assert(W/review['saved_path']).read_bytes()==Path(review['source_path']).read_bytes()
old=json.loads(blob(BASE,'artifacts/validation_manifest.json',collect=True)[1]);new=read('artifacts/validation_manifest.json');assert len(old['result_sets'])==164 and new['result_sets'][:164]==old['result_sets'] and len(new['result_sets'])==(166 if FINAL else 165)
assert{k:v for k,v in old.items() if k!='result_sets'}=={k:v for k,v in new.items() if k!='result_sets'} and 'description'not in new['relevant_commits'][7]
refs=read(A+'/reference_identity.json')['sources']
for x in refs:assert blob(x['commit'],x['path'])=={k:x[k] for k in ['SHA256','bytes']},x['id']
spec=importlib.util.spec_from_file_location('m_doc_identity',W/A/'audit_tools/verify_document_identity.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);proof=m.verify(W,ROOT)
if FINAL:
 receipt=read(A+'/publication/receipt.json')
 for p,x in receipt['changed_blobs'].items():assert blob(receipt['verified_commit'],p)==x,p
 stop=read(A+'/publication_stop.json');assert stop['mandatory_stop'] and stop['payload_commit_GitHub_verified']==receipt['verified_commit']
batch.stdin.close();assert batch.wait()==0
out={'status':'GITHUB_REVIEW_M_K0_BYTE_SHA_RETRIEVAL_PASS','phase':'FINAL_METADATA'if FINAL else 'PAYLOAD','repository':'HIROMU1015/Partially-Randomized-Trotter','branch':BRANCH,'base_commit':BASE,'verified_commit':h,'checked_at_JST':datetime.now(timezone(timedelta(hours=9))).isoformat(),'remote_fetch':'Actual GitHub SSH fetch into independent bare cache without local alternates;streamed blob byte comparison','changed_blobs':verified,'changed_blob_count':len(verified),'changed_bytes':sum(x['bytes'] for x in verified.values()),'review_SHA256':review['SHA256'],'review_bytes':review['bytes'],'review_original_byte_match':True,'protected_old_files_GitHub_matched':len(protected),'fixed_reference_bodies_matched':len(refs),'old_manifest_records_preserved':164,'old_manifest_non_result_and_known_FAIL_preserved':True,'metadata_only_integrity':proof,'K0':'UNDETERMINED_NOT_READY','counterterm_projection_fit_core':'REDUCES_TO_KNOWN','independent_algorithmic_contribution':'NOT_IDENTIFIED','K1':'NOT_EXECUTED_K0_NOT_PASSED','first_candidate_formal_status':'AWAITING_USER_ADOPTION_DECISION','M_adopted':False,'new_scientific_calculations':0,'new_saved_numeric_aggregation':0,'GPT_scientific_review_started':False,'current_CI_or_external_scientific_reproduction':False,'mandatory_stop':True}
args=[s for s in sys.argv[1:] if not s.startswith('--')]
if args:Path(args[0]).write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ('changed_blobs','metadata_only_integrity')},ensure_ascii=False,indent=2))
