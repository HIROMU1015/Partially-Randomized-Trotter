"""Prepare or explicitly launch a single saved-H8 B1 fit; default never computes."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'src/trotterlib'))
import h8_phase_b1_fit as f
import h8_b0_saved_anchor_audit as a
import h8_b0_anchor_contract as old
import h8_b0_anchor_runtime as gate
import launch_h8_strengthened_phase_a as monitor
import probe_h8_b0_anchor_cpus as cpu
from h8_phase_b1_contract import REQUIRED,eligible
SOURCES=(*old.SOURCES,'src/trotterlib/h8_phase_b1_fit.py','src/trotterlib/h8_phase_b1_contract.py','scripts/h8_phase_b1_execution.py')
AUDIT_COMMIT='b19c337f0b16513d68b59eb3366dc6eff175c313'
B1_ART='artifacts/resource_applicability/track_a_h8_phase_b1_readiness/2026-10-16'
PYTHON='/home/AbeHiromu/projects/h6-environments/20261012/native-dependency-v2/venv/bin/python'


def write(path,value):
 raw=(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
 with Path(path).open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())


def check_rows(rows):
 for row in rows:
  p=Path(row['path']);f.require(p.stat().st_size==row['bytes'] and a.sha(p)==row['sha256'],'SEALED_BYTES_CHANGED')


def preserve(before):
 for path,row in before['protected'].items():
  f.require(Path(path).stat().st_size==row['bytes'] and a.sha(path)==row['sha256'],'OLD_PROTECTED_BYTES_CHANGED')
 for row in before['worktrees']:
  f.require(a.git_bytes(row['path'],'rev-parse','HEAD').decode().strip()==row['HEAD'] and
   hashlib.sha256(a.git_bytes(row['path'],'status','--porcelain=v1','-z')).hexdigest()==row['status_sha256'],'OLD_WORKTREE_CHANGED')
 f.require(a.sha(before['old_V2_untracked']['path'])==before['old_V2_untracked']['sha256'],'OLD_V2_UNTRACKED_CHANGED')


def verify_all(config):
 check_rows(config['files'])
 for name,expected in config['source_hashes'].items():
  f.require(a.sha(ROOT/name)==expected and hashlib.sha256(a.git_bytes(ROOT,'show',config['SOURCE']+':'+name)).hexdigest()==expected,'NEW_SOURCE_CHANGED')
 f.require(len(config['runtime_hashes'])==34273 and a.digest(config['runtime_hashes'])==a.RUNTIME_MAP,'FULL_RUNTIME_INVENTORY')
 for path,expected in config['runtime_hashes'].items():f.require(a.sha(path)==expected,'RUNTIME_CHANGED')
 preserve(a.read(config['before']))


def prepare(base,source):
 base=base.resolve();f.require((base/'before_PRIVATE.json').is_file(),'BASELINE_REQUIRED')
 f.require(a.git_bytes(ROOT,'rev-parse','HEAD').decode().strip()==source,'SOURCE_HEAD')
 f.require(not a.git_bytes(ROOT,'status','--porcelain=v1','-z'),'CLEAN_SOURCE_REQUIRED')
 project=ROOT.parents[1];wt=project/'partially-randomized-trotter-worktrees'
 prior=project/'h6-handoff-evidence/20261016/track-a-h8-b0-anchor-execution/run02'
 audited=a.audit_saved(prior,wt/'h8-b0-anchor-cpu-gate-fix-20261016',wt/'h8-b0-anchor-run02-stop-20261016')
 write(base/'B0_reuse_audit.json',audited)
 files=[]
 def bind(path,logical,expected=None):
  path=Path(path);raw=path.read_bytes();row=dict(path=str(path),logical_name=logical,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
  if expected:f.require(row['bytes']==expected['bytes'] and row['sha256']==expected['sha256'],'PRIOR_RECEIPT_CHANGED')
  if not any(r['path']==row['path'] for r in files):files.append(row)
  return raw
 for row in a.read(wt/'h8-b0-anchor-run02-stop-20261016'/a.ARTIFACT/'original_execution_file_hashes_v1.json')['files']:
  bind(prior/a.relative_path(row['logical_path']),'B0_original:'+row['logical_path'],row)
 registry_path=ROOT/B1_ART/'anchor_registry_v1.json'
 f.require(a.git_bytes(ROOT,'show',AUDIT_COMMIT+':'+B1_ART+'/anchor_registry_v1.json')==registry_path.read_bytes(),'FIXED_REGISTRY')
 registry=json.loads(bind(registry_path,'prior_anchor_registry'))
 evidence_path=ROOT/B1_ART/'source_evidence_v1.json'
 f.require(a.git_bytes(ROOT,'show',AUDIT_COMMIT+':'+B1_ART+'/source_evidence_v1.json')==evidence_path.read_bytes(),'FIXED_SOURCE_EVIDENCE')
 evidence=json.loads(bind(evidence_path,'prior_source_evidence'))['evidence_files']
 for row in evidence:
  if row['logical_name'].startswith('B0I') and 'path' in row:
   root=wt/'h8-native-b0i-full-family-results-20261016';path=root/row['path'];raw=bind(path,row['logical_name'],row)
   f.require(a.git_bytes(root,'show',row['commit']+':'+row['path'])==raw,'B0I_GIT_DISK_BINDING')
 private=project/'h6-handoff-evidence/20261015/h8-native-b0i-full-family-run01'
 original_contract=json.loads(bind(private/'final_contract_v1.json','B0I_original_contract',next(x for x in evidence if x['logical_name']=='B0I_original_contract')))
 f.require(original_contract['expected_profile_digest']==a.PROFILE and original_contract['binding']['compiler']==a.COMPILER,'B0I_PROFILE_COMPILER')
 oldconfig=a.read(prior/'prepared/config_UNAPPROVED.json')
 f.require(original_contract['profile']['runtime_file_hashes']==oldconfig['runtime_file_hashes'],'COMMON_FULL_RUNTIME')
 rows=[]
 for name in ('train_measurements.json','holdout_measurements.json'):
  saved=json.loads(bind(prior/'prepared/output01'/name,'B0_'+name))
  for r in saved['axes']:
   p,q=int(r['case_id'].split('_')[1][1:]),int(r['case_id'].split('_')[2][1:])
   rows.append(dict(id=r['case_id']+'_'+r['axis'],method='B0',prefix=p,q=q,axis=r['axis'],formula='2nd',role=r['role'],
    saved_RZ=r['measured_RZ'],origin_SOURCE=r['SOURCE'],runtime_profile_sha256=r['profile_sha256'],compiler_hash=r['compiler_hash'],
    control=r['control'],scalar_policy=r['scalar_policy'],cost_preparation_hash=r['preparation_binding']['saved_preparation_hash'],
    fingerprint=r['fingerprint_before'],evidence={k:True for k in REQUIRED},original_acquisition_status='STOP',reuse_audit_sha256=a.sha(base/'B0_reuse_audit.json')))
 for r in registry['anchors']:
  if r['method']!='B0I':continue
  p,q,axis=r['prefix'],r['q'],r['axis'];logical=f'B0I_p{p}_q{q}_{axis}_raw_cost'
  raw=json.loads(bind(private/f'run01/case_{p}_{q}_{axis}_cost.json',logical,next(x for x in evidence if x['logical_name']==logical)))
  f.require(eligible(r) and raw['metrics']['RZ']==r['saved_RZ'] and raw['fingerprint']==r['fingerprint']==raw['input_fingerprint_check']['before']==raw['input_fingerprint_check']['after'] and raw['compiler_hash']==a.COMPILER and raw['worker']['native_exit_code']==0,'B0I_RAW_LINEAGE')
  rows.append(dict(r,id=f'B0I_p{p}_q{q}_2nd_{axis}',role='TRAIN'))
 f.validate_anchors(rows)
 anchors=base/'anchors.json';write(anchors,dict(rows=rows))
 bind(anchors,'normalized_24_axes');bind(base/'B0_reuse_audit.json','B0_readonly_audit')
 for logical,path in [('B2_selection',ROOT/B1_ART/'B2_selection_v1.json'),('prior_B1_contract',ROOT/B1_ART/'execution_contract_DRAFT_v1.json'),('acquisition_plan',ROOT/old.PLAN_PATH),('before',base/'before_PRIVATE.json')]:bind(path,logical)
 source_hashes={name:a.sha(ROOT/name) for name in SOURCES}
 for name,h in source_hashes.items():f.require(hashlib.sha256(a.git_bytes(ROOT,'show',source+':'+name)).hexdigest()==h,'SOURCE_BYTES_NOT_COMMITTED')
 config=dict(schema='H8_B1_SEALED_CONFIG_V1',SOURCE=source,plan_sha256=f.digest(f.plan()),source_hashes=source_hashes,
  runtime_hashes=oldconfig['runtime_file_hashes'],profile_sha256=a.PROFILE,compiler_hash=a.COMPILER,
  files=files,anchors=str(anchors),B2_selection=str(ROOT/B1_ART/'B2_selection_v1.json'),before=str(base/'before_PRIVATE.json'),
  output=str(base/'output01'),control=str(base/'control01'),lock=str(base/'one_shot.lock'),python=PYTHON,
  accepted_B0_saved_with_original_STOP=True,historical_thread_compliance_certified=False)
 write(base/'config.json',config)
 print(json.dumps(dict(status='SEALED_NO_FIT_STARTED',SOURCE=source,axes=24,files=len(files),new_compile=0,new_fit=0)))


def authorized(config,grant):
 p=f.plan();f.require(grant.get('schema')=='H8_B1_NEW_SCOPE_GRANT_V1' and grant.get('approved_by_user') is True and
  grant.get('runtime_authorization') is True and grant.get('approval_text') and grant.get('scope')==f.SCOPE and
  grant.get('SOURCE')==config['SOURCE'] and grant.get('config_sha256')==f.digest(config) and
  grant.get('plan_sha256')==config['plan_sha256']==f.digest(p) and grant.get('resources')==f.RESOURCES and
  grant.get('call_caps')==f.CAPS and grant.get('accept_saved_B0_with_original_STOP') is True and
  all(grant.get(k) is v for k,v in [('one_shot',True),('retry',False),('resume',False),('B2_authorized',False),('C_authorized',False),('next_stage_authorized',False)]),'NEW_EXPLICIT_B1_GRANT_REQUIRED')
 cpus=grant.get('allowed_cpus',[])
 f.require(len(cpus)==2 and len(set(cpus))==2 and all(type(x)is int for x in cpus) and set(cpus)<=os.sched_getaffinity(0),'TWO_EXPLICIT_CPUS')
 return cpus


def launch(base):
 launch_affinity=os.sched_getaffinity(0)
 cfg=a.read(base/'config.json');g=a.read(base/'grant_APPROVED_ONCE.json');cpus=authorized(cfg,g)
 for name in ('launcher_attempt','fresh_gate.json','phase_terminal.json'):
  f.require(not (base/name).exists(),'B1_ATTEMPT_ALREADY_SPENT')
 for name in ('output','control','lock'):f.require(not Path(cfg[name]).exists(),'EXCLUSIVE_NEW_B1_PATHS')
 write(base/'launcher_attempt',dict(SOURCE=cfg['SOURCE'],scope=f.SCOPE,one_shot=True))
 try:
  verify_all(cfg)
  inventory=cpu.inventory(os.sched_getaffinity(0));cores={(inventory[c]['package_id'],inventory[c]['core_id']) for c in cpus}
  f.require(len(cores)==2,'TWO_PHYSICAL_CORES')
  group=Path('/sys/fs/cgroup')/next(s.split(':',2)[2].lstrip('/') for s in Path('/proc/self/cgroup').read_text().splitlines() if s.startswith('0::'));root=Path('/sys/fs/cgroup')
  while group.is_relative_to(root):
   if (group/'cpu.max').exists():
    q,t=(group/'cpu.max').read_text().split();f.require(q=='max' or int(q)>=2*int(t),'CPU_QUOTA_BELOW_TWO')
   if group==root:break
   group=group.parent
  siblings={s for c in cpus for s in inventory[c]['siblings']};idle=gate.measure_idle(siblings)
  write(base/'CPU_fresh.json',idle);f.require(idle['status']=='PASS','PHYSICAL_CORE_BUSY')
  memory=monitor.snapshot();f.require(monitor.resource_reason(memory,memory,2**30) is None,'FRESH_MEMORY_PSI_OOM')
  fs=gate.filesystem(base);f.require(fs['bytes']>=2*2**30 and fs['inodes']>=40000,'FRESH_DISK_INODES')
  fresh=dict(SOURCE=cfg['SOURCE'],config_sha256=f.digest(cfg),grant_sha256=f.digest(g),monotonic_seconds=time.monotonic(),
   observer_identity=list(monitor.owned.identity(os.getpid())),allowed_cpus=cpus,resource_snapshot=memory,filesystem=fs,CPU_gate_sha256=a.sha(base/'CPU_fresh.json'))
  write(base/'fresh_gate.json',fresh)
  env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
  for key in gate.THREAD_KEYS:env[key]='1'
  os.sched_setaffinity(0,{cpus[1]})
  command=[cfg['python'],'-B',str(ROOT/'scripts/h8_phase_b1_execution.py'),'driver-approved',str(base)]
  monitor.supervise(command,env=env,control=cfg['control'],output=cfg['output'],limits=f.RESOURCES,allowed_cpus=cpus)
  verify_all(cfg)
  terminal=a.read(Path(cfg['output'])/'terminal.json');f.require(terminal['status']=='COMPLETE_MANDATORY_STOP','FIT_NOT_COMPLETE')
  write(base/'phase_terminal.json',dict(status='COMPLETE_MANDATORY_STOP',SOURCE=cfg['SOURCE'],model_sha256=terminal['model_sha256'],post_source_input_runtime_preservation_PASS=True,original_acquisition_status='STOP',B2_authorized=False,C_authorized=False,next_stage_authorized=False))
 except BaseException as exc:
  write(base/'phase_terminal.json',dict(status='STOP',SOURCE=cfg['SOURCE'],reason=str(exc),automatic_retry=False,next_stage_authorized=False));raise
 finally:os.sched_setaffinity(0,launch_affinity)


def driver(base):
 cfg=a.read(base/'config.json');g=a.read(base/'grant_APPROVED_ONCE.json');cpus=authorized(cfg,g);fresh=a.read(base/'fresh_gate.json')
 f.require(fresh['SOURCE']==cfg['SOURCE'] and fresh['config_sha256']==f.digest(cfg) and fresh['grant_sha256']==f.digest(g) and 0<=time.monotonic()-fresh['monotonic_seconds']<=30,'FRESH_GATE_REQUIRED')
 f.require(list(monitor.owned.identity(os.getppid()))==fresh['observer_identity'],'OWNED_LIVE_OBSERVER_PARENT')
 f.require(Path(sys.executable).resolve()==Path(cfg['python']).resolve() and all(os.environ.get(k)=='1' for k in gate.THREAD_KEYS),'PINNED_PYTHON_THREAD_ONE')
 check_rows(cfg['files'])
 for name,h in cfg['source_hashes'].items():f.require(a.sha(ROOT/name)==h,'DRIVER_SOURCE_BYTES')
 out=Path(cfg['output']);f.require(not out.exists() and not Path(cfg['lock']).exists(),'ONE_SHOT_ALREADY_SPENT')
 write(cfg['lock'],dict(scope=f.SCOPE,SOURCE=cfg['SOURCE']));out.mkdir(mode=0o700)
 os.sched_setaffinity(0,{cpus[0]});r=f.RESOURCES
 resource.setrlimit(resource.RLIMIT_AS,(r['driver_AS_bytes'],r['driver_AS_bytes']));resource.setrlimit(resource.RLIMIT_CPU,(r['CPU_seconds'],r['CPU_seconds']))
 used={k:0 for k in f.CAPS};start=time.monotonic();baseline=monitor.snapshot();samples=[]
 def guard():
  current=monitor.process_record(os.getpid());rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
  f.require(current['threads']==1 and rss<=r['driver_RSS_bytes'] and current['AS_bytes']<=r['driver_AS_bytes'],'FIT_PROCESS_RESOURCE_LIMIT')
  f.require(time.monotonic()-start<=r['wall_seconds'] and current['CPU_seconds']<=r['CPU_seconds'],'FIT_CPU_WALL_LIMIT')
  f.require(monitor.resource_reason(monitor.snapshot(),baseline,512*2**20) is None,'FIT_PSI_OOM_MEMORY')
  samples.append(dict(wall=time.monotonic()-start,RSS_peak_bytes=rss,AS_bytes=current['AS_bytes'],CPU_seconds=current['CPU_seconds'],threads=current['threads']))
 def take(key):
  guard();f.require(used[key]<f.CAPS[key],'CALL_CAP_'+key);used[key]+=1
  with (out/'calls.jsonl').open('a') as stream:stream.write(json.dumps(dict(call=key,used=used[key],refund=False))+'\n')
 def saved(name,value):
  raw=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode();f.require(len(raw)<=r['output_bytes'],'OUTPUT_LIMIT')
  current=sum(p.stat().st_size for p in out.rglob('*') if p.is_file() and p!=out/name)
  f.require(current+len(raw)<=r['output_bytes'],'AGGREGATE_OUTPUT_LIMIT')
  # Mutable partial filenames belong only to this new execution. Raw task and grant files are never rewritten.
  (out/name).write_bytes(raw)
 def stop(*args):raise RuntimeError('FIT_STOP_SIGNAL_OR_DEADLINE')
 for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM):signal.signal(sig,stop)
 signal.alarm(r['wall_seconds'])
 try:
  guard();result=f.execute(a.read(cfg['anchors'])['rows'],a.read(cfg['B2_selection'])['axes'],take=take,write=saved,guard=guard)
  guard();f.require(used==f.CAPS,'EXACT_CALL_COUNTS')
  check_rows(cfg['files']);saved('self_resource_observations.json',samples)
  saved('terminal.json',dict(status='COMPLETE_MANDATORY_STOP',SOURCE=cfg['SOURCE'],scope=f.SCOPE,actual_calls=used,wall_seconds=time.monotonic()-start,**result))
 except BaseException as exc:
  saved('self_resource_observations.json',samples);saved('terminal.json',dict(status='STOP',SOURCE=cfg['SOURCE'],error=str(exc),actual_calls=used,automatic_retry=False,next_stage_authorized=False));raise
 finally:signal.alarm(0)


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','launch-approved','driver-approved'],nargs='?');p.add_argument('base',type=Path,nargs='?');p.add_argument('--source');args=p.parse_args()
 if args.mode is None:print(json.dumps(f.plan(),indent=2));return
 if args.base is None:p.error('dedicated base required')
 if args.mode=='prepare':
  if not args.source:p.error('immutable source required')
  prepare(args.base,args.source)
 elif args.mode=='launch-approved':launch(args.base)
 else:driver(args.base)


if __name__=='__main__':main()
