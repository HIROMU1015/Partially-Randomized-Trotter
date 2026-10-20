"""One new saved-data Stage B grant, one native worker, no retry or resume."""
import argparse,ast,csv,hashlib,json,math,os,sys,time,traceback
from pathlib import Path
import rare_rte_stage_b_core as core
import rare_rte_stage_b_verify as independent
ROOT=Path(__file__).resolve().parents[1]
PR=ROOT/'.stage_b_private'
OUT=ROOT/'artifacts/resource_applicability/pr_rare_rte_stage_b/2026-10-21'
FILES=('scripts/rare_rte_stage_b_core.py','scripts/rare_rte_stage_b_verify.py','scripts/run_rare_rte_stage_b.py','scripts/prepare_rare_rte_stage_b_inputs.py','tests/test_rare_rte_stage_b.py')
TARGETS=('B2_p1_q1_2nd_r2_K4','B2_p1_q4_2nd_r2_K4','B2_p3_q1_2nd_r4_K2','B2_p2_q8_2nd_r2_K4','B2_p4_q2_2nd_r4_K2','B2_p4_q8_2nd_r4_K2','B2_p8_q2_2nd_r4_K4','B2_p7_q16_2nd_r4_K4')
BUDGET=dict(saved_analysis_pass=1,primary_conditions=8,independent_numeric_checks=8,render_probability_figure=1,sampling=0,simulation=0,signal=0,SCF_DF_state_generation=0,circuit_generation=0,compile=0,candidate_selection=0,old_cohort_pooling=0)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def read(path):return json.loads(Path(path).read_text())
def original_gate():
 rows=read(PR/'original_reads.json')
 for r in rows:
  if r['kind']=='PRIVATE_ORIGINAL':assert sha(r['path'])==r['sha256'],'ORIGINAL_BYTES_CHANGED'
 return dict(status='PASS_ALL_USED_ORIGINAL_BYTES',files=len(rows),unrelated_worktree_HEAD_or_status_frozen=False)
def resources():
 mem={line.split(':')[0]:line.split(':')[1].strip() for line in Path('/proc/meminfo').read_text().splitlines()}
 rel=next(x.split(':',2)[2] for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::'))
 cg=Path('/sys/fs/cgroup')/rel.lstrip('/');obs=[]
 while cg.is_relative_to(Path('/sys/fs/cgroup')):
  obs.append({k:(cg/k).read_text().strip() for k in ['cpu.max','memory.max','memory.current','memory.events','memory.pressure'] if (cg/k).exists()})
  if cg==Path('/sys/fs/cgroup'):break
  cg=cg.parent
 return dict(allowed_affinity=sorted(os.sched_getaffinity(0)),MemAvailable=mem['MemAvailable'],loadavg=Path('/proc/loadavg').read_text().strip(),cgroup_chain=obs,scheduler_affinity_cgroup_quota_changed=False,arbitrary_CPU_RAM_worker_cap=False,execution_strategy='one ordinary OS-scheduled worker: small scalar/Fraction dataset; parallel startup and duplicate input decoding offer no demonstrated benefit; no resource cap or affinity assignment')
def runtime():
 files={sys.executable:sha(sys.executable)}
 for m in list(sys.modules.values()):
  f=getattr(m,'__file__',None)
  if f and Path(f).is_file():files[str(Path(f).resolve())]=sha(f)
 return files

def preflight():
 contract=read(OUT/'contract.json');assert contract['budget']==BUDGET
 assert sha(PR/'inputs.json')==contract['input_sha256'];assert sha(PR/'original_reads.json')==contract['original_inventory_sha256']
 for name,h in contract['source_hashes'].items():assert sha(ROOT/name)==h,'NEW_SOURCE_BYTES_CHANGED'
 for name,h in read(PR/'runtime_manifest.json').items():assert sha(name)==h,'NEW_RUNTIME_BYTES_CHANGED'
 original=original_gate();data=read(PR/'inputs.json');assert len(data['conditions'])==8
 assert [c['candidate'] for c in data['conditions']]==contract['candidates']==list(TARGETS)
 for c in data['conditions']:
  assert len(c['tasks'])==c['n'];assert c['K'] in (2,4);assert c['q']>0 and c['r']>0
  assert all(t['law']==c['law'] for t in c['tasks'])
 # Static gates do not import any saved scientific SOURCE or evaluate its functions.
 for path in FILES:
  tree=ast.parse((ROOT/path).read_text())
  for node in ast.walk(tree):
   if isinstance(node,ast.Import):names=[x.name for x in node.names]
   elif isinstance(node,ast.ImportFrom):names=[node.module or '']
   else:continue
   assert not any(x.split('.')[0] in {'numpy','scipy','qiskit','pyscf','trotterlib','trottertracks'} for x in names),'FORBIDDEN_SCIENTIFIC_IMPORT'
 return dict(status='PASS_PREFLIGHT_METADATA_NO_REAL_NUMERICAL_ANALYSIS',SOURCE=contract['SOURCE'],contract_sha256=sha(OUT/'contract.json'),input_sha256=sha(PR/'inputs.json'),source_hashes=contract['source_hashes'],original_gate=original,resources=resources(),runtime_files=len(read(PR/'runtime_manifest.json')),synthetic_tests=read(OUT/'synthetic_validation.json'),grant_not_consumed=not (PR/'ONE_SHOT.json').exists())

def csvwrite(name,rows):
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def plot(results):
 lines=['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="430" viewBox="0 0 1100 430"><rect width="1100" height="430" fill="white"/><g font-family="sans-serif" font-size="13" fill="#172b4d"><text x="20" y="24">Finite RTE stratum probabilities — canonical law at saved tau; no cost upper bound</text>']
 colors=['#617891','#007c91','#b66c00','#a02d70']
 for i,r in enumerate(results):
  y=67+i*43;lines.append(f'<text x="15" y="{y}">{r["system"]} eps={r["epsilon"]}</text>')
  for j,g in enumerate(r['groups']):
   x=210+j*220;p=float(g['probability']);width=max(0.,160*(1+math.log10(p)/16)) if p>0 else 0
   label=format(p,'.6g') if p else 'support-out'
   lines.append(f'<text x="{x}" y="{y-11}">{g["stratum"]}: {label}</text><rect x="{x}" y="{y-7}" width="{width:.3f}" height="10" fill="{colors[j]}"/>')
 lines.append('<text x="20" y="420">Bars use log10(p), visual floor 1e-16; positive values retained in CSV/JSON. K2 S4 is outside support.</text></g></svg>')
 (OUT/'stratum_probabilities.svg').write_text('\n'.join(lines)+'\n')
def scientific_worker():
 budget=dict(BUDGET);used={k:0 for k in BUDGET}
 def take(key):
  used[key]+=1;assert used[key]<=budget[key];save(PR/'analysis_ledger.json',dict(budget=budget,used=used,nonrefundable=True))
 take('saved_analysis_pass');data=read(PR/'inputs.json');results=[];audits=[]
 for c in data['conditions']:
  take('primary_conditions');r=core.analyze(c);save(PR/'partial'/('primary_'+c['condition']+'.json'),r)
  take('independent_numeric_checks');audit=independent.verify(c,r);audits.append(dict(condition=c['condition'],**audit));results.append(r)
 take('render_probability_figure');plot(results)
 probability=[];groups=[];sensitivity=[];tasks=[]
 for r in results:
  for g in r['groups']:
   probability.append(dict(condition=r['condition'],candidate=r['candidate'],stratum=g['stratum'],probability=g['probability'],support_status=g['support_status']))
   groups.append(dict(condition=r['condition'],candidate=r['candidate'],stratum=g['stratum'],n=g['G']['n'],mean_G_exact=g['G']['mean_exact'],SE_G=g['G']['SE'],sample_min_G=g['G']['sample_min'],sample_max_G=g['G']['sample_max'],unknown_population_mean_G=None))
  s=r['conditional_expectation'];sensitivity.append(dict(condition=r['condition'],candidate=r['candidate'],finite_sample_mean_G=r['saved_finite_sample']['mean_exact'],PF_G=r['fixed_PF_G_exact'],observed_plugin_G=s['observed_plugin_G'],unknown_mass=s['unknown_mass'],PF_minus_observed_plugin=s['PF_minus_observed_plugin'],unknown_mixture_mean_threshold_G=s['unknown_mixture_mean_threshold_G'],threshold_over_observed_S0_mean=s['threshold_over_observed_S0_mean'],conditional_direction=s['conditional_direction'],population_guarantee=False))
  tasks.extend(r['tasks'])
 csvwrite('probabilities.csv',probability);csvwrite('observed_strata.csv',groups);csvwrite('sensitivity.csv',sensitivity);csvwrite('trajectory_costs.csv',tasks)
 save(OUT/'result.json',dict(status='COMPLETE_STAGE_B_CONDITIONAL_ANALYSIS_NOT_POPULATION_CERTIFICATE',SOURCE=read(OUT/'contract.json')['SOURCE'],contract_sha256=sha(OUT/'contract.json'),input_sha256=sha(PR/'inputs.json'),conditions=results,old_cohorts_pooled=False,new_sampling_simulation_signal_circuits_compile=False,new_mathematical_guarantee_adopted=False,next_stage_authorized=False))
 save(OUT/'independent_numeric_audit.json',dict(status='PASS_NUMERICAL_VALIDATION_NOT_SCIENTIFIC_GUARANTEE',rows=audits,independent_AI_reviewer=False))
 assert used==budget;save(OUT/'calls.json',dict(budget=budget,used=used,nonrefundable=True))

def execute(grant_path):
 grant=read(grant_path);contract=read(OUT/'contract.json')
 assert grant['approved_by_user'] and grant['one_shot'] and not grant['retry'] and not grant['next_stage_authorized']
 assert grant['SOURCE']==contract['SOURCE'] and grant['contract_sha256']==sha(OUT/'contract.json')
 assert grant['approval_receipt_sha256']==sha(OUT/'execution_approval_receipt.json')
 assert read(OUT/'execution_approval_receipt.json')['explicit_stage_B_one_shot_authorized']
 pf=preflight();assert pf['grant_not_consumed'];save(OUT/'preflight.json',pf)
 fd=os.open(PR/'ONE_SHOT.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'w') as f:json.dump(dict(SOURCE=contract['SOURCE'],contract_sha256=sha(OUT/'contract.json'),grant_sha256=sha(grant_path),consumed=True,nonrefundable=True),f)
 start=time.monotonic();pid=os.fork()
 if pid==0:
  try:scientific_worker()
  except BaseException:
   (PR/'scientific_failure.txt').write_text(traceback.format_exc());os._exit(1)
  os._exit(0)
 child,status,usage=os.wait4(pid,0);code=os.waitstatus_to_exitcode(status)
 receipt=dict(worker_pid=child,native_exit_code=code,same_parent_wait4=True,reaped=True,remaining_scientific_workers=0,wall_seconds=time.monotonic()-start,CPU_seconds=usage.ru_utime+usage.ru_stime,single_worker_peak_RSS_bytes=usage.ru_maxrss*1024,new_worker_count=1,subagents=0,arbitrary_resource_caps=False)
 save(OUT/'native_exit.json',receipt);save(OUT/'grant_consumption_receipt.json',read(PR/'ONE_SHOT.json'))
 try:
  assert code==0,'SCIENTIFIC_WORKER_FAILED_NO_RETRY'
  assert sha(PR/'inputs.json')==contract['input_sha256']
  final=original_gate()
  for name,h in contract['source_hashes'].items():assert sha(ROOT/name)==h
  for name,h in read(PR/'runtime_manifest.json').items():assert sha(name)==h
  save(OUT/'completion_audit.json',dict(status='PASS_STAGE_B_COMPLETE_MANDATORY_STOP',native_exit=receipt,old_evidence=final,source_input_runtime_final_gate=True,scientific_passes=1,independent_numeric_audit=True,certificate=False,new_sampling_simulation_signal_circuit_compile=0,old_grant_reused=False,old_SOURCE_raw_results_grant_ledger_marker_STOP_modified=False,other_worktree_or_job_intervention=False,push=False,next_stage_authorized=False))
 except BaseException:
  save(OUT/'STOP.json',dict(status='MANDATORY_STOP_SCIENCE_OR_FINAL_GATE_FAILED',native_exit=receipt,automatic_retry=False,resume=False,next_stage_authorized=False,reason=traceback.format_exc().splitlines()[-1]));raise
 save(OUT/'STOP.json',dict(status='MANDATORY_STOP_STAGE_B_COMPLETE',automatic_retry=False,resume=False,next_stage_authorized=False,C_or_D_authorized=False,detailed_scientific_review=False,final_evidence_freeze=False,push=False))
 print(json.dumps(dict(status='COMPLETE_STAGE_B_MANDATORY_STOP',native_exit_code=code,conditions=8)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['preflight','execute']);p.add_argument('--grant',type=Path);a=p.parse_args()
 if a.action=='preflight':print(json.dumps(preflight(),ensure_ascii=False))
 else:
  if a.grant is None:p.error('--grant required')
  execute(a.grant)
