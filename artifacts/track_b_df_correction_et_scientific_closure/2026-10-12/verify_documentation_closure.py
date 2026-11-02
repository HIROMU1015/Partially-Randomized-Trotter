"""Read-only hashes/links/metadata verification; never imports science code."""
from pathlib import Path
import hashlib,json,re,subprocess

BASE='901cafcd594b0399b8638ee1d4ea612bb56fc69e'
INPUT_COMMIT='1d39fc687bf328c8f8b5a9c5f7251950f0fee814'
REVIEW='docs/research/track_b_df_correction_et_scientific_review_20261012.md'
NOTE='docs/tracks/algorithm_codesign/df_correction_et_series_closure_20261012.md'
DATED='docs/research/研究ノート/2026-10-12_track_b_df_correction_et_scientific_closure.md'
ART='artifacts/track_b_df_correction_et_scientific_closure/2026-10-12'
INDEXES=['PROJECT_MAP.md','docs/research/研究概要・現状.md','scripts/README.md','src/trotterlib/README.md','docs/README.md','docs/tracks/algorithm_codesign/README.md']
ROOT=Path(__file__).resolve().parents[3]
def sha(b):return hashlib.sha256(b).hexdigest()
def git(*args,cwd=ROOT):return subprocess.check_output(['git',*args],cwd=cwd)
def load(p):return json.loads((ROOT/p).read_text())
def row(p):
 b=(ROOT/p).read_bytes();return {'path':p,'bytes':len(b),'sha256':sha(b)}
def main():
 checks=[]; inventories=[]; paths_verified=set()
 def check(name,predicate):
  assert predicate,name
  checks.append({'name':name,'PASS':True})
 b=(ROOT/REVIEW).read_bytes()
 check('review original bytes/SHA256',len(b)==39039 and sha(b)=='00f19797037936c2c4e7ac2bf41610fec636113cca3df9ab3c91a87d40276076')
 check('published review original unchanged',b==git('show',INPUT_COMMIT+':'+REVIEW))
 manifests=[('original science source S7','artifacts/track_b_df_correction_et_phase_preparation/2026-10-12/v7/source_manifest_v7.json'),('saved recovery source S9','artifacts/track_b_df_correction_et_saved_recovery_preparation/2026-10-12/v2/source_manifest.json'),('original v7 raw result','artifacts/track_b_df_correction_et_execution_audit/2026-10-12/v4/original_run_inventory.json'),('saved recovery raw result','artifacts/track_b_df_correction_et_saved_recovery_audit/2026-10-12/v2/original_recovery_inventory.json')]
 for label,path in manifests:
  m=load(path);prefixes=[]
  for r in m['files']:
   raw=(ROOT/r['path']).read_bytes(); paths_verified.add(r['path'])
   if r['path'] in INDEXES:
    check_bytes=raw[:r['bytes']];prefixes.append(r['path'])
   else:check_bytes=raw
   assert len(check_bytes)==r['bytes'] and sha(check_bytes)==r['sha256'],(label,r['path'])
  inventories.append({'label':label,'manifest_path':path,'manifest_sha256':row(path)['sha256'],'file_count':len(m['files']),'all_bytes_sha256_match':True,'documentation_prefix_only':prefixes})
  check(label+' all protected identities',True)
 for p in INDEXES:
  old=git('show',BASE+':'+p);raw=(ROOT/p).read_bytes()
  check('index baseline prefix unchanged '+p,raw.startswith(old) and len(raw)>len(old))
 changed=git('diff','--name-only','-z',BASE).decode().rstrip('\0').split('\0')
 allowed=set(INDEXES+[REVIEW]); allowed.update(str(p.relative_to(ROOT)) for p in (ROOT/ART).rglob('*') if p.is_file())
 allowed.update([NOTE,DATED])
 check('tracked diff documentation/metadata only',all(p in allowed for p in changed))
 untracked=git('ls-files','--others','--exclude-standard','-z').decode().rstrip('\0').split('\0')
 check('untracked files selected documentation/metadata only',all(not p or p in allowed for p in untracked))
 # Check links in new documents and only appended portions of existing indexes.
 links=[]
 texts={NOTE:(ROOT/NOTE).read_text(),DATED:(ROOT/DATED).read_text()}
 for p in INDEXES:texts[p]=(ROOT/p).read_bytes()[len(git('show',BASE+':'+p)):].decode()
 for p,content in texts.items():
  for link in re.findall(r'\]\(([^)]+)\)',content):
   if '://' in link or link.startswith('#'):continue
   path=link.split('#',1)[0]
   target=(ROOT/p).parent/path
   check('local link '+p+' -> '+path,target.is_file())
   links.append({'document':p,'target':path,'exists':True})
 # Preserve original authorizations and all old source/result/marker/status data.
 for p in ['artifacts/track_b_df_correction_et_phase_preparation/2026-10-12/v7/authorization.json','artifacts/track_b_df_correction_et_phase_preparation/2026-10-12/v7/contract_v7.json','artifacts/track_b_df_correction_et_saved_recovery_preparation/2026-10-12/v2/authorization.json','artifacts/track_b_df_correction_et_saved_recovery_preparation/2026-10-12/v2/recovery_contract.json']:
  check('authorization/contract unchanged '+p,(ROOT/p).read_bytes()==git('show',BASE+':'+p))
 for name,parent in [('84562f8c5af9249a902662324200456498a83d15','74682c40d0cdcca92d244a9edf4bb2852361eef5'),('eea9c497421c5dcfc541dcaa508f607112dbd0d5','3afef9e418a8d2173a2780f5534404390bfbe6fb')]:
  check('old direct-child authorization preserved '+name,git('show','-s','--format=%P',name).decode().strip()==parent)
 old=ROOT.parent/'track-b-df-correction-et-saved-recovery-v2-20261012'
 check('old execution worktree HEAD preserved',git('rev-parse','HEAD',cwd=old).decode().strip()==BASE)
 check('old execution worktree clean',not git('status','--porcelain',cwd=old))
 rows=load('artifacts/track_b_df_correction_et_saved_recovery_audit/2026-10-12/v2/comparison_rows.json')
 expected=[(3,False,576,112,57),(3,True,692,236,179),(6,False,576,116,5519),(6,True,None,232,11622)]
 for r,e in zip(rows,expected):
  check('saved transcription row '+str(e[:2]),(r['r'],r['controlled'],r['A_fixed_plan_T_lower'],r['B_certified_fixed_plan_T_upper'],r['A_unknown_occurrences'])==e and r['A_actual_T'] is None)
 check('4 saved rows preserved',len(rows)==4)
 # These are document transcriptions, not recomputed classifications or inequalities.
 total_identity_rows=sum(r['file_count'] for r in inventories)
 out={'kind':'DOCUMENTATION_ONLY_CLOSURE_TECHNICAL_AUDIT','base_result_publication_commit':BASE,'review_input_commit':INPUT_COMMIT,'metadata_checks_PASS':len(checks),'metadata_checks_FAIL':0,'protected_identity_rows_verified':total_identity_rows,'protected_unique_paths_verified':len(paths_verified),'identity_row_count_includes_overlap_between_manifests':True,'inventory_checks':inventories,'checks':checks,'new_document_links':links,'existing_science_classification_unchanged':'TECHNICAL_INCONCLUSIVE','existing_recovery_classification_unchanged':'SAVED_E_T_ACCOUNTING_COMPLETE','added_science_or_focused_tests':0,'new_science_calls':0,'old_science_runners_invoked':0,'new_authorization_or_marker_created':False,'mandatory_STOP':True}
 p=ROOT/ART/'protected_evidence_audit.json';p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'metadata_PASS':len(checks),'protected_identity_rows':total_identity_rows,'links':len(links),'science_calls':0,'added_tests':0}))
if __name__=='__main__':main()
