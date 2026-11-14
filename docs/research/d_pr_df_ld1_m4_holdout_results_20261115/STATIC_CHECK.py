from pathlib import Path
from fractions import Fraction
import hashlib,json,subprocess
ROOT=Path.cwd();B=Path('docs/research/d_pr_df_ld1_m4_holdout_results_20261115')
SOURCE='7bdfc6f629aa74a5c1347abf84ad6486c669c9d8';RAW='3042e6e66ad693ed514274334adecd3d1b42dc02'
git=lambda *a:subprocess.check_output(['git',*a],text=True)
blob=lambda commit,path:subprocess.check_output(['git','show',commit+':'+str(path)])
load=lambda p:json.loads(p.read_text())
plan=load(B/'INPUTS.json');admit=load(B/'ADMITTED_RECEIPTS.json');c=load(B/'COMPARISON.json');handoff=load(B/'C_HANDOFF.json')
assert git('diff','2592d442ae8f36f71e42820a26e847b1fa38b59d','--','docs/research/d_pr_df_ld1_m3_results_20261115')==''
source_paths=['scripts/run_pr_df_ld1_m4.py','scripts/run_pr_df_ld1_m3.py',str(B/'INPUTS.json'),'tests/test_pr_df_ld1_m4.py']
for p in source_paths:assert Path(p).read_bytes()==blob(SOURCE,p)
pipeline_paths=['src/trottertracks/p1_efficient_zz.py','src/trottertracks/p2_direct_pilot.py','src/trottertracks/compiled_price_fast.py','src/trottertracks/efficient_controlled_zz.py','scripts/h46_b3_fast_fingerprint.py','scripts/run_p2_direct_pilot.py']
for p in pipeline_paths:assert Path(p).read_bytes()==blob('39ddd59edf54a3a901e5302207b2765ddbd03fbb',p)
for p in git('ls-tree','-r','--name-only',RAW,'--',str(B)).splitlines():assert Path(p).read_bytes()==blob(RAW,p)
assert load(B/'RECEIPT_BINDING_CHECK.json')['checked_rows']==8
assert len(admit['rows'])==8 and admit['rows']==handoff['rows']
assert c['units']==handoff['units'] and c['campaign']==handoff['status']=='DESCRIPTIVE_COMPLETE'
FROZEN='3cac53da1a4a8b2228245593c74c1ff9ec1f5212';frozen=json.loads(blob(FROZEN,'docs/research/pr_df_ld1_residual_freeze_20261115/run01/FROZEN_MODELS.json'))
locator_count=0
for r in admit['rows']:
 assert r['valid'] and r['physical_id']==plan['cells'][0]['candidate_id'] and r['cohort']==plan['cells'][0]['cohort']
 assert r['seeds']==plan['cells'][0]['seeds'][r['index']] and r['execution_source']==SOURCE and r['result_publication_commit']==RAW
 for name in ('request','seed_source','path_result','original_receipt','axis_receipt','structural_receipt','gate'):
  v=r[name];raw=blob(v['commit'],v['path']);value=json.loads(raw)
  for k in v['selector']:value=value[k]
  if 'sha256' in v:assert hashlib.sha256(raw).hexdigest()==v['sha256']
  if name=='axis_receipt':assert value['fingerprint']==r['wrapper_fingerprint'] and value['metrics']['rz_count']==r['C']
  elif name=='seed_source':assert value==r['seeds']
  elif name=='path_result':
   ai=('X','Y').index(r['axis']);assert value['fingerprints'][ai]==r['wrapper_fingerprint'] and value['S'][ai]==r['S'] and value['C'][ai]==r['C']
   assert value['event_digest']==r['event_digest']
  locator_count+=1
for unit in c['units']:
 axis=unit['axis'];rs=[r for r in admit['rows'] if r['axis']==axis]
 assert sorted(r['index'] for r in rs)==[0,1,2,3]
 P=Fraction(**frozen['models'][axis]['m4_prediction']['predicted_RZ']);O=Fraction(sum(r['C'] for r in rs),4);d=P-O
 for name,v in dict(P=P,O=O,signed_error=d,absolute_error=abs(d),signed_relative_error=d/O,absolute_relative_error=abs(d)/O).items():assert Fraction(**unit[name])==v
 assert unit['model_verdict']==c['model_verdict']=='NOT_ASSESSED_NO_ADOPTED_TOLERANCE'
assert c['tolerance'] is None and handoff['full_wrapper_statevector_validation'] is False
assert all(handoff[k] is False for k in ('fit','prediction_regeneration','posthoc_threshold','additional_paths','qualified_E_C'))
assert all(handoff[k] is None for k in ('CI','law_I_Q_distance','rare_contribution','long_round_transfer','N','G','winner'))
manifest=load(Path('artifacts/validation_manifest.json'));old=json.loads(blob('2592d442ae8f36f71e42820a26e847b1fa38b59d','artifacts/validation_manifest.json'))
assert {k:v for k,v in manifest.items() if k!='d_pr_df_ld1_m4_holdout_results_20261115'}==old
r=load(B/'run01/RESULT.json');assert r['new_compiles']==r['compile_attempts']==8
for i in range(4):
 logs=[json.loads(s) for s in (B/f'run01/PROGRESS_p{i}.jsonl').read_text().splitlines()]
 seed=[v for v in logs if (v['stage'],v['state'])==('fixed_seed_lookup','CHECKED')]
 assert len(seed)==1 and seed[0]['seeds']==plan['cells'][0]['seeds'][i] and len(seed[0]['seeds'])==16
 assert logs[-1]['stage']=='path' and logs[-1]['state']=='COMPLETE'
 assert (B/f'run01/STACK_p{i}.txt').stat().st_size==0
summary=dict(status='STATIC_DELIVERY_CHECKS_PASSED',checks=['m3 raw/input/handoff unchanged','sealed SOURCE/INPUTS unchanged','six original P2 pipeline files unchanged',
 'published m4 raw bytes unchanged','8 valid receipt rows and 56 fixed locators resolved','fixed seeds on actual sampler route','two Fraction comparisons against fixed axis JSON','old manifest entries preserved'],
 execution_source=SOURCE,raw_publication_commit=RAW,planned_compiles=8,reused=0,new_compiles=8,valid=8,withheld_or_missing=0,
 compiled_receipt_failures=0,locator_count=locator_count,comparison_units=2,campaign=c['campaign'],model_verdict=c['model_verdict'],
 technical_tests_passed=13,real_scientific_retries=0,new_scientific_calls_in_this_check=0,full_wrapper_statevector_validation=False,
 model_qualification=False,C_consumer_admission='PENDING',mandatory_stop=True)
with (B/'FINAL_CHECK.json').open('x') as f:json.dump(summary,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(summary,ensure_ascii=False,indent=2))
