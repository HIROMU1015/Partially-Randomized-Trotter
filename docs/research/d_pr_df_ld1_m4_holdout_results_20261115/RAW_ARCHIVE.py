from pathlib import Path
import json,hashlib,subprocess,shutil
B=Path('docs/research/d_pr_df_ld1_m4_holdout_results_20261115');plan=json.loads((B/'INPUTS.json').read_text());out=B/'run01'
result=json.loads((out/'RESULT.json').read_text());source=result['execution_source'];cell=plan['cells'][0]
assert result['status']=='M4_SC_PAIRS_ACQUIRED' and result['new_compiles']==8 and result['compile_attempts']==8
assert len(result['records'])==4 and not result['fit'] and not result['prediction_regeneration']
assert source==subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
preparation=json.loads((out/'PREPARATION_CHECK.json').read_text())
assert preparation['preparation_hash']==plan['expected_preparation_hash'] and preparation['scalar_hex']==plan['scalar_saved']['hex']
assert preparation['fragment_order']==[0]
checks=[];files=[]
for record in result['records']:
 assert record['ok'] and record['exitcode']==0
 p=record['value'];i=p['index'];directory=Path(p['private_directory'])
 assert p['source']['execution_source']==source
 assert p['request']==plan['menus']['H4']['candidates'][0]['candidate']
 assert p['m']==4 and p['cohort']==cell['cohort'] and p['seeds']==cell['seeds'][i] and p['role']=='HOLDOUT'
 assert p['conditional'] is False and p['lowering']==plan['lowering']['id'] and p['scalar_hex']==plan['scalar_saved']['hex']
 assert [r['axis'] for r in p['compile_log']]==['cosine','sine'] and all(r['status']=='COMPLETE' for r in p['compile_log'])
 events=json.loads((directory/'events_PRIVATE.json').read_text())
 digest=hashlib.sha256(json.dumps(events['events'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
 assert digest==p['event_digest'] and events['seeds']==cell['seeds'][i] and events['laws']==p['laws'] and events['conditional'] is False
 assert len(events['events'])==16 and all(len(row)==32 for row in events['events'])
 progress=[json.loads(line) for line in (Path(plan['private'])/f'PROGRESS_p{i}.jsonl').read_text().splitlines()]
 seed_checks=[r for r in progress if r['stage']=='fixed_seed_lookup' and r['state']=='CHECKED']
 assert len(seed_checks)==1 and seed_checks[0]['seeds']==cell['seeds'][i]
 structural=json.loads((directory/'STRUCTURAL.json').read_text());original=json.loads((directory/'result.json').read_text())
 assert structural['S']==p['S'] and structural['event_digest']==digest and structural['scientific_source']==source
 assert original['C']==p['C'] and original['S']==p['S'] and original['event_digest']==digest and original['source']['execution_source']==source
 dest=out/'receipts'/f'p{i}';dest.mkdir(parents=True)
 copies={}
 for filename in ('result.json','STRUCTURAL.json','cosine.json','sine.json','TECHNICAL_TIMING.json','M4_PATH.json'):
  raw=(directory/filename).read_bytes();(dest/filename).write_bytes(raw)
  copies[filename]=dict(path=str(dest/filename),original_private_path=str(directory/filename),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
 for ai,a in enumerate(('cosine','sine')):
  r=json.loads((directory/(a+'.json')).read_text())
  assert r['metrics']['rz_count']==p['C'][ai] and r['fingerprint']==p['fingerprints'][ai] and r['compiler']==plan['compiler']['settings_hash']
  assert r['cache_hit'] is False
  assert r['fingerprint']!=p['fingerprints'][1-ai]
  assert type(p['S'][ai]) is int and type(p['C'][ai]) is int
 eventraw=(directory/'events_PRIVATE.json').read_bytes()
 files.append(dict(index=i,copies=copies,event_original=dict(path=str(directory/'events_PRIVATE.json'),sha256=hashlib.sha256(eventraw).hexdigest(),bytes=len(eventraw),event_digest=digest)))
 checks.append(dict(index=i,cohort=cell['cohort'],seeds=p['seeds'],status='SAME_PATH_SC_IDENTITY_CHECKED',
  physical_request=True,preparation_input_scalar_fragment_order=True,normal_seed=True,signed_event_allocation=True,
  same_saved_event_digest=True,axis_compiler_fingerprint=True,wrapper_measurement_and_mutation_checks=True,
  numerical_action_gate='NOT_RUN_AS_IN_ORIGINAL_P2_FRESH',full_wrapper_statevector_validation=False,
  source=source,lowering=plan['lowering']['id'],m4_acquisition=True))
for name in ('PARENT_PROGRESS.jsonl','PARENT_STACK.txt',*[f'PROGRESS_p{i}.jsonl' for i in range(4)],*[f'STACK_p{i}.txt' for i in range(4)]):
 f=Path(plan['private'])/name
 if f.is_file():shutil.copyfile(f,out/name)
for name,v in [('ARCHIVED_RECEIPTS.json',dict(execution_source=source,pipeline_source=plan['pipeline_source'],paths=files)),('RECEIPT_CHECK.json',dict(status='EIGHT_SAME_PATH_SC_PAIRS_CHECKED',execution_source=source,valid_axis_rows=8,checks=checks,new_compiles=8,reused=0,missing=0,withheld=0,fit=False,m4=True,mandatory_stop=True))]:
 with (out/name).open('x') as f:json.dump(v,f,indent=2);f.write('\n')
print(json.dumps({'source':source,'new_compiles':8,'valid_axis_pairs':8,'S_C':[{'index':r['value']['index'],'S':r['value']['S'],'C':r['value']['C'],'events':r['value']['events']} for r in result['records']],'wall_seconds':result['total_wall_seconds']}))
