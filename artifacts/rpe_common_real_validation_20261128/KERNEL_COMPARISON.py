from pathlib import Path
import subprocess,sys,json,hashlib,time,resource,importlib.util
root=Path('/home/abe/Project/Partially Randomized Trotter')
base=root/'.worktrees/rpe-common-review-fixes-20261128/artifacts/rpe_common_real_validation_20261128'
fixed=Path('/tmp/rpe-review-real-bd42dd1e9')
sys.path.insert(0,str(fixed/'src'))
from trottertracks import rpe_total_dp_parallel as P
R=P.R
helper=root/'.worktrees/rpe-common-review-fixes-20261128/src/trottertracks/rpe_dp_prefix_reuse.py'
spec=importlib.util.spec_from_file_location('_rpe_real_review_prefix',helper)
C=importlib.util.module_from_spec(spec);sys.modules[spec.name]=C;spec.loader.exec_module(C)
job=root/'.rpe-central/state/runs/search-538c35a68be6425b1b4c/jobs/PR_DF/a651c4cac3335804e4fa'
inputs=json.loads((job/'CASE_INPUTS.json').read_text());case=inputs['cases'][0]
def doc(loc):
    raw=subprocess.check_output(['git','-C',str(root),'show',loc['commit']+':'+loc['path']])
    return json.loads(raw),hashlib.sha256(raw).hexdigest()
config,ch=doc(case['configuration']);packet,sh=doc(case['signal_packet'])
assert config['id']==packet['configuration_id']==case['original_configuration_id']
assert [r['m'] for r in config['rounds']]==[r['m'] for r in packet['means']]==[r['m'] for r in case['allocation']['shots']]
opts=inputs['options'];scope=dict(opts['total_state_refinement'],source={'validation':'R43_FIXED_KERNEL_COMPARISON','case_inputs_sha256':hashlib.sha256((job/'CASE_INPUTS.json').read_bytes()).hexdigest()})
entry={'prepared':{'h':config['spec']['h'],'rows':[dict(m=s['m'],mean=m,shots=s) for m,s in zip(packet['means'],case['allocation']['shots'])]},'evaluation_target':case['evaluation_target']}
h=float(R.a.exact_scalar(entry['prepared']['h']));target=entry['evaluation_target']['energy_ha']
tables=R.tables_for(entry,opts)
identity=dict(config_sha256=ch,signal_sha256=sh,configuration_id=config['id'],binding_id=case['binding_id'],h=h,target=target,L_D=config['spec']['L_D'],delta_pf=config['spec']['delta_pf'],formula=config['spec']['formula'],shots=case['allocation']['shots'],scope=scope,kernel_commit='bd42dd1e95eb1d4152216d62792ad3421b803f7c',kernel_sha256=hashlib.sha256(Path(P.__file__).read_bytes()).hexdigest(),helper_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),python=sys.version)
mode=sys.argv[1];out=base/('kernel-'+mode+('-'+sys.argv[2] if len(sys.argv)>2 else ''));out.mkdir(exist_ok=False)
(out/'INPUT_IDENTITY.json').write_text(json.dumps(identity,indent=2)+'\n')
events=[]
t=time.perf_counter();start=resource.getrusage(resource.RUSAGE_SELF);child_start=resource.getrusage(resource.RUSAGE_CHILDREN)
with C.installed(P,directory=out/'cache',cache_namespace=identity,checkpoint_rounds=[5,11],parallel_residual=True,profile_residual=True,persistent_residual_pool=(mode=='persistent'),event_sink=events.append):
    engine=P.Accelerator(workers=2,outcome_batch=8)
    result=engine(tables,h=h,target=target,scope=scope)
end=resource.getrusage(resource.RUSAGE_SELF);children=resource.getrusage(resource.RUSAGE_CHILDREN)
metrics=dict(wall_seconds=time.perf_counter()-t,parent_cpu=end.ru_utime+end.ru_stime-start.ru_utime-start.ru_stime,child_cpu=children.ru_utime+children.ru_stime-child_start.ru_utime-child_start.ru_stime,parent_peak_rss_kib=end.ru_maxrss,child_max_rss_kib=children.ru_maxrss)
(out/'RESULT.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
(out/'EVENTS.json').write_text(json.dumps(events,indent=2)+'\n')
(out/'METRICS.json').write_text(json.dumps(metrics,indent=2)+'\n')
print(mode,result['status'],metrics, 'residual_events',sum(e.get('event')=='RESIDUAL_TIMING' for e in events), 'reused',sum(e.get('pool_reused') is True for e in events),flush=True)
