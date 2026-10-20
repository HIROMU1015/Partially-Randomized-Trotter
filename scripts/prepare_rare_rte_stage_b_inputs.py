"""Read-only extraction of existing P metadata; never imports a scientific provider."""
import csv,gzip,hashlib,io,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PRIVATE=ROOT/'.stage_b_private'
OUT=ROOT/'artifacts/resource_applicability/pr_rare_rte_stage_b/2026-10-21'
BASE='42ed0876c0505502428629c1fb4863233247ebaf'
READS={}
def sha(b):return hashlib.sha256(b).hexdigest()
def canon(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()
def git(ref,p):return subprocess.check_output(['git','show',ref+':'+p],cwd=ROOT)
def fixed(ref,p):
 b=git(ref,p);READS[ref+':'+p]=dict(kind='GIT',commit=ref,path=p,sha256=sha(b),bytes=len(b));return json.loads(b)
def read(p,expected=None):
 p=Path(p);b=p.read_bytes();h=sha(b)
 if expected:assert h==expected,('ORIGINAL_HASH_MISMATCH',p.name)
 READS[str(p)]=dict(kind='PRIVATE_ORIGINAL',path=str(p),sha256=h,bytes=len(b));return json.loads(b)
def local(name):return json.loads((PRIVATE/name).read_text())
def condition_records():return fixed(BASE,'artifacts/resource_applicability/pr_rare_rte_existing_evidence/2026-10-21/conditions.json')
def task(cell,measurement,trace,condition,expected_trace_sha=None):
 t=read(trace,expected_trace_sha);assert t['cell']['id']==cell
 assert measurement['cell_id']==cell
 assert measurement['event_digest']==t['event_digest']==sha(canon(t['events']))
 assert t['conditional'] is False and t['quantum_weights_changed'] is False
 for axis in ['cosine','sine']:
  m=measurement['axes'][axis]
  assert m['event_digest']==t['event_digest']
  assert m['compiler_hash']=='1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b'
  assert type(m['RZ']) is int and m['RZ']>0
 return dict(replica=measurement['replica'],cohort=measurement['cohort'],
   event_digest=t['event_digest'],trace_sha256=sha(Path(trace).read_bytes()),law=t['original_RTE_distribution'],
   events=t['events'],axes={axis:{k:measurement['axes'][axis][k] for k in ['RZ','CX','compiler_hash','fingerprint','event_digest']} for axis in ['cosine','sine']},saved_stratum=measurement.get('stratum'))
def prepare():
 OUT.mkdir(parents=True,exist_ok=True);cs=condition_records();known=local('known_worktrees.json');conditions={c['candidate']:dict(c,tasks=[]) for c in cs}
 b=fixed('f2d1a86c','artifacts/resource_applicability/track_a_h2_pr_resource/2026-10-19/P_completed_run01/confirmation_result.json')
 loc=local('H2_spec_locator.json');sp=read(loc['path']);h2=Path(sp['attempt'])/'output';raw=read(h2/'result.json',b['scientific_result_sha256']);assert raw['accounting']==b['accounting']
 tr=fixed('f2d1a86c','artifacts/resource_applicability/track_a_h2_pr_resource/2026-10-19/P_completed_run01/trajectory_receipts_projection.json');tr={r['task_id']:r for r in tr['rows']}
 ab=fixed('f2d1a86c','artifacts/resource_applicability/track_a_h2_pr_resource/2026-10-19/P_completed_run01/execution_binding.json');ar=next(r for r in ab['inputs'] if r['role']=='A_result');signals=read(ar['path'],ar['sha256'])['signals']
 for row in raw['measurements']:
  cid=row['cell_id'];d=h2/('first_control_workers' if row['replica']==0 else 'remaining_workers')/(cid+'_'+str(row['replica']));r=read(d/'task_result.json');assert r==row
  conditions[cid]['tasks'].append(task(cid,row,d/'trajectory.json',conditions[cid],tr[d.name]['trajectory_sha256']))
 for c in cs[:2]:conditions[c['candidate']]['saved_log_B']=next(r['log_B'] for r in signals if r['cell']['id']==c['candidate'])
 h4=next(Path(p) for p in known if p.endswith('/h4-tol-p-seal-20261019'))/'.h4_tol_p_private';oldmanifest=read(h4/'completed_P_preservation_manifest_PRIVATE.json');expected={r['path']:r['sha256'] for r in oldmanifest}
 raw4=read(h4/'run01/output/result.json');pub4=fixed('821cc0f6','artifacts/resource_applicability/track_a_h4_tol_p_completed/2026-10-19/result.json');assert raw4==pub4
 rows=read(h4/'run01/output/paired_measurements.json')['rows'];signals=read(h4/'saved_A_signals_PRIVATE.json')['signals']
 for row in rows:
  cid=row['cell_id'];d=h4/'run01/output'/('first_control_workers' if row['replica']==0 else 'remaining_workers')/(cid+'_'+str(row['replica']));tp=d/'trajectory.json'
  conditions[cid]['tasks'].append(task(cid,row,tp,conditions[cid],expected.get(str(tp))))
 for c in cs[2:4]:conditions[c['candidate']]['saved_log_B']=next(r['log_B'] for r in signals if r['cell']['id']==c['candidate'])
 pub6=fixed('3cccfcb2','artifacts/resource_applicability/track_a_h6_pr_confirmation_science/2026-10-19/result.json');tr6=list(csv.DictReader(io.StringIO(git('3cccfcb2','artifacts/resource_applicability/track_a_h6_pr_confirmation_science/2026-10-19/trajectory_receipts.csv').decode())));tr6={(r['cell_id'],int(r['replica'])):r for r in tr6}
 for s in local('H6_specs.json'):
  spec=read(s['path']);p=Path(s['output']);binding=next(x for x in pub6['scopes'] if x['scope']==spec['scope']);r=read(p/'result.json',binding['result_sha256']);assert r['primary']==binding['primary']
  maps=read(p/'native_receipt.json')['maps']
  for m in maps:
   assert m['all_reaped']
   for w in m['workers']:
    assert w['exit_code']==0 and w['reaped'];d=Path(w['directory']);row=read(d/'task_result.json');cid=row['cell_id'];t=tr6[(cid,row['replica'])]
    assert sha((d/'task_result.json').read_bytes())==t['result_sha256']
    conditions[cid]['tasks'].append(task(cid,row,d/'trajectory.json',conditions[cid],t['trajectory_sha256']))
  payload=read(spec['payload']);signals=read(payload['A_signal'])['signals'];cid=spec['target']['cell']['id'];conditions[cid]['saved_log_B']=next(x['log_B'] for x in signals if x['cell']['id']==cid)
 # Redacted H8 task IDs map to original files through the pre-existing private index.
 ix=read(local('H8_index_locator.json')['path']);mp={r['evidence_id']:r for r in ix['packet_to_original']};pk=fixed('a6246f03','artifacts/resource_applicability/track_a_h8_gpt_share_packet/2026-10-20/candidate/06_task_axis_links.json')
 for item in pk['tasks']:
  ref=mp[item['task_result_origin']['evidence_id']];row=read(ref['path'],ref['sha256']);ref=mp[item['trace_origin']['evidence_id']];cid=row['cell_id'];conditions[cid]['tasks'].append(task(cid,row,ref['path'],conditions[cid],ref['sha256']))
 for eid,c in [('E00007',cs[6]),('E01024',cs[7])]:
  r=mp[eid];raw=read(r['path'],r['sha256']);assert raw['primary']['mean_exact']==c['mean_exact'];assert raw['primary']['paired_SE']==float(c['paired_SE'])
  # The continuation run root is identified by an original fixed batch receipt.
  run=Path(mp['E00001']['path']).parent.parent
  spec=read(run/'sealed01'/raw['scope']/'technical_seal.json');payload=read(spec['payload']);signals=read(payload['A_signal'])['signals'];conditions[c['candidate']]['saved_log_B']=next(x['log_B'] for x in signals if x['cell']['id']==c['candidate'])
 for cid,c in conditions.items():
  assert len(c['tasks'])==c['n'];assert {t['replica'] for t in c['tasks']}==set(range(c['n']))
  c['tasks'].sort(key=lambda t:t['replica']);law=c['tasks'][0]['law'];assert all(t['law']==law for t in c['tasks']);c['law']=law
  for t in c['tasks']:
   assert len(t['events'])==c['q'];assert all(len(es)==c['r'] for es in t['events'])
  c['cohort']=c['tasks'][0]['cohort'];assert all(t['cohort']==c['cohort'] for t in c['tasks'])
  c['shots']={k:int(v) for k,v in c['shots'].items()}
 dataset=dict(schema='RARE_RTE_STAGE_B_PRIVATE_INPUT_V1',base=BASE,conditions=list(conditions.values()),old_cohorts_pooled=False)
 (PRIVATE/'inputs.json').write_bytes(canon(dataset)+b'\n')
 (PRIVATE/'original_reads.json').write_text(json.dumps(list(READS.values()),indent=2)+'\n')
 public=[dict(input_id='I'+str(i).zfill(5),kind=r['kind'],sha256=r['sha256'],bytes=r['bytes'],**({k:r[k] for k in ['commit','path']} if r['kind']=='GIT' else {})) for i,r in enumerate(READS.values())]
 (OUT/'input_acquisition.json').write_text(json.dumps(dict(status='PASS_METADATA_ONLY_NO_SCIENCE_EVALUATION',base=BASE,dataset_sha256=sha((PRIVATE/'inputs.json').read_bytes()),conditions=8,trajectories=sum(c['n'] for c in cs),read_files=public,private_raw_or_locators_published=False),ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(dict(status='ACQUIRED_AND_JOINED_METADATA',conditions=8,trajectory_records=sum(c['n'] for c in cs),input_files=len(READS))))
if __name__=='__main__':prepare()
