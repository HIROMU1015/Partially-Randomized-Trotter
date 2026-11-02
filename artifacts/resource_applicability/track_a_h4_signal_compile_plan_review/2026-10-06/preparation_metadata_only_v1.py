"""Metadata-only H4 signal/compile preparation; never launches science."""
import ast,copy,hashlib,json,math,os,runpy,sys,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path.cwd()
SCIENCE=Path('/home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/worktrees/track-a-h4-worker-bootstrap-run02-20261006')
OUTPUT=Path('/home/AbeHiromu/projects/partially-randomized-trotter/artifacts/resource_applicability/track_a_h4_geometry_execution/track-a-h4-geometry-v2-20261006-run02')
OLD=SCIENCE/'artifacts/resource_applicability/track_a_h4_worker_bootstrap_run02/2026-10-06'
BUNDLE=ROOT/'artifacts/resource_applicability/track_a_h4_signal_compile_plan_review/2026-10-06'
SOURCE='049e69919af16ad29a67a217dc7a407d6b1754a6'
sys.path.insert(0,str(SCIENCE/'src'))
from trottertracks.resource_applicability.h4_geometry.identity import fingerprint,Stop
from trottertracks.resource_applicability.h4_geometry.gates import structural_gate,authorize,checkout_gate,Permit
from trottertracks.resource_applicability.h4_geometry.resources import observe_memory,admission,Monitor,WALL_CAP,OUTPUT_CAP

def read_json(p):return json.loads(p.read_bytes())
def save(name,value):
    with (BUNDLE/name).open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False);stream.write('\n')
def file_sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def proc_cpu():
    rows={}
    for line in Path('/proc/stat').read_text().splitlines():
        x=line.split()
        if x[0].startswith('cpu') and x[0][3:].isdigit():
            a=list(map(int,x[1:]));rows[int(x[0][3:])]=(sum(a[:8]),a[3]+a[4])
    return rows
def cpu_set(text):
    found=set()
    for part in text.strip().split(','):
        span=part.split('-');found.update(range(int(span[0]),int(span[-1])+1))
    return found

assert ROOT.name.startswith('track-a-h4-signal-compile-plan-review-20261006')
assert not BUNDLE.exists()
# Allow only identity/freeze and read-only budget journal in the frozen runtime.
def audit(event,args):
    if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
        p=os.fsdecode(args[0])
        if p.endswith(('.npz','.npy','.pkl','.pickle','.sqlite','.db')):raise RuntimeError('scientific arrays/cache access forbidden')
        if p.startswith(str(OUTPUT)+'/'):
            if p not in (str(OUTPUT/'generation-freeze.json'),str(OUTPUT/'byte-budget.journal')):raise RuntimeError('runtime outside metadata scope')
            if isinstance(args[2],int) and args[2] & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):raise RuntimeError('runtime mutation forbidden')
    if event in ('os.system','os.posix_spawn','os.fork','os.exec','os.sched_setaffinity'):raise RuntimeError('launch/affinity mutation forbidden')
    if event=='subprocess.Popen':
        cmd=args[1]
        if not isinstance(cmd,(list,tuple)) or cmd[0]!='git' or 'show' not in cmd:raise RuntimeError('only read-only git show permitted')
sys.addaudithook(audit)
baseline_files={str(p):file_sha(p) for p in OLD.iterdir() if p.is_file()}
freeze=read_json(OUTPUT/'generation-freeze.json')
completion=read_json(OLD/'input_generation_completion_audit_v1.json')
assert file_sha(OUTPUT/'generation-freeze.json')==completion['freeze_file_sha256']
assert fingerprint('h4-generation-freeze-v1',freeze)==completion['freeze_fingerprint']
assert freeze['stage']=='INPUTS_FROZEN_STOP' and freeze['mandatory_stop'] is True and freeze['next_stage_authorized'] is False
oldplan=read_json(OLD/'input_generation_plan_run02_v1.json')
plan=copy.deepcopy(oldplan)
plan.update(stage='signal_compile',binding='INPUT_BOUND',inputs=copy.deepcopy(freeze['inputs']),
    generation_freeze_digest=completion['freeze_fingerprint'])
assert plan['source_commit']==SOURCE and freeze['source_commit']==SOURCE and len(plan['source_hashes'])==19
for path,expected in plan['source_hashes'].items():assert file_sha(SCIENCE/path)==expected
assert set(plan['inputs'])==set(plan['distances'])
for entry in completion['input_files']:
    assert plan['inputs'][entry['distance']]['bytes_sha256']==entry['sha256']
fp=fingerprint('h4-execution-plan-v1',plan)
authorization={'schema_version':'h4-native-authorization-v1','stage':'signal_compile','run_id':plan['run_id'],
    'permission':'signal_compile','one_shot':True,'allowed_cpus':[3,5,6,7,8,9],'result_prior':True,'plan_fingerprint':fp}
auth_digest=fingerprint('h4-authorization-v1',authorization)
review={'schema_version':'h4-native-stage-review-v1','stage':'signal_compile','run_id':plan['run_id'],
    'approved':False,'plan_fingerprint':fp,'authorization_digest':auth_digest}
structural_gate(plan,authorization,review)
metadata_permit=Permit('signal_compile',plan,plan['source_root'],fingerprint('h4-review-v1',review))
contract,options=checkout_gate(metadata_permit)  # metadata only; no launch/worker/input array load
negative=[]
for explicit,auth,rev,label in [(False,authorization,review,'no_explicit_launch'),
    (True,authorization,review,'unapproved_review'),(True,read_json(OLD/'authorization_run02_v1.json'),review,'old_generation_authorization')]:
    try:authorize('signal_compile',plan,auth,rev,explicit_launch=explicit)
    except Stop as exc:negative.append({'case':label,'rejected':True,'reason':str(exc)})
    else:raise RuntimeError('negative authorization gate passed')
random=sum(t['method'] in ('B2','B3') for t in plan['templates'])
baseline=len(plan['templates'])-random
logical=6*(random*32*2+baseline*2)
assert (len(plan['templates']),random,baseline,logical)==(218,194,24,74784)
prior=freeze['consumed_seconds']
remaining=WALL_CAP-prior
journal=(OUTPUT/'byte-budget.journal').read_bytes()
assert len(journal)%128==0
prior_charge=sum(int(journal[i:i+128].strip()) for i in range(0,len(journal),128))
assert prior_charge<OUTPUT_CAP
# Upper-size JSON specimens contain placeholders only, no science result or seed.
# Even extreme finite float64-derived ceil() shot integers need no more than309 digits.
metric=10**20-1
floatwide=-1.7976931348623157e308
point={'epsilon':floatwide,'eligible':True,'shots':[10**309-1,10**309-1],'work':floatwide,'SE':floatwide,
    'engineering_interval':[floatwide,floatwide]}
metrics={k:metric for k in ('rz_count','rz_depth','cx_count','cx_depth','total_depth','circuit_size')}
identity={k:'f'*64 for k in ('H','DF','state','input','template','compiler','environment')}
identity.update(geometry='0.70',source=SOURCE,wrapper_semantics='h4-full-gaussian-paired-wrapper-v1')
specimen={'geometry':'0.70','template_id':max((t['template_id'] for t in plan['templates']),key=len),
    'identity':identity,'candidate_fingerprint':'f'*64,
    'signal':{'corrected':[floatwide,floatwide],'raw':[floatwide,floatwide],'normalization':floatwide},
    'saved':{'normalization':floatwide,'bias':{'cosine':floatwide,'sine':floatwide},'paired_costs':[[metric,metric] for _ in range(32)]},
    'paired_metrics':[[dict(metrics),dict(metrics)] for _ in range(32)],
    'display':[dict(point) for _ in range(302)],'research_decision':None,'next_stage_authorized':False}
json_size=lambda x:len((json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode())
specimen_size=json_size(specimen)
assert specimen_size<=512*2**10
# source constants and counts; no Ledger instantiated and no sample/build/compile
watch=math.ceil(remaining)+2
deltas=1+2*logical
signals=6*218
row_specs=[
    ('wrapper_checkpoint_JSON',logical,4096,'ledger.complete publishes one record per logical wrapper; 4 KiB allowance'),
    ('ledger_delta_JSON',deltas,4096,'initial delta plus reserve/complete deltas; no reuse assumed; 4 KiB allowance'),
    ('worker_log_text',logical,8192,'at most one log per actual compile; hard UTF-8 bytes cap8192; assume all nonempty'),
    ('signal_JSON_with_302_display_points',signals,512*2**10,'extreme-width metadata specimen plus allowance; no science results'),
    ('monitor_wall_JSON',watch,128,'1-second watcher across remaining cumulative72h; two boundary allowance'),
    ('ledger_lock',1,0,'fresh signal-stage ledger lock; separate from input-stage byte journal'),
    ('map_complete_JSON',1,4096,'fixed completion metadata allowance')]
block=4096
round_block=lambda n:((n+block-1)//block)*block
pubs=sum(count for _,count,_,_ in row_specs)
inode_count=pubs+8+2  # up to eight concurrent temporary publishers and two future control dirs
journal_growth=pubs*128
physical={name:count*round_block(size) for name,count,size,_ in row_specs}
physical.update(byte_budget_journal_growth=round_block(journal_growth),
    concurrent_temporary_files_extra_allowance=8*512*2**10,
    directory_entries_indexes_allowance=round_block((pubs+8+4)*256),
    directory_initial_blocks=2*block,driver_control_log_margin=8*2**20,
    filesystem_extent_ACL_journal_misc_margin=64*2**20)
bound=sum(physical.values())
required=math.ceil(bound/(2**28))*2**28  # round upward to0.25GiB
required_inodes=math.ceil(inode_count/10000)*10000
payload=sum(count*size for _,count,size,_ in row_specs)
charge=2*payload+pubs*128
assert prior_charge+charge<OUTPUT_CAP
quota_method=Path('/tmp/h4_storage_stage_review.py')
assert file_sha(quota_method)=='c2e2f412f8a6da08ecb5eac4791dcdf2948eaa3c47b1c9fd93588d468cdae547'
getter=runpy.run_path(str(quota_method),run_name='readonly_getter_only')['quota_and_filesystem']
fs=getter()
assert fs['filesystem_type']=='ext4' and fs['fragment_size']==block
mem1=observe_memory()
cpus=[3,5,6,7,8,9]
online=cpu_set(Path('/sys/devices/system/cpu/online').read_text())
top=[]
for c in cpus:
    p=Path('/sys/devices/system/cpu/cpu%d/topology'%c)
    top.append({'CPU':c,'package':int((p/'physical_package_id').read_text()),'core':int((p/'core_id').read_text()),
        'siblings':sorted(cpu_set((p/'thread_siblings_list').read_text())),
        'NUMA':[int(x.name[4:]) for x in p.parent.glob('node[0-9]*')]})
assert len({(x['package'],x['core']) for x in top})==6
ticks1=proc_cpu();started=time.monotonic()
time.sleep(3)
ticks2=proc_cpu();duration=time.monotonic()-started
loads=[]
for item in top:
    busy={}
    for c in item['siblings']:
        a,b=ticks1[c],ticks2[c];total=b[0]-a[0];idle=b[1]-a[1]
        busy[str(c)]=None if total<=0 else 100*(total-idle)/total
    loads.append({'CPU':item['CPU'],'siblings_busy_percent':busy,'max_busy_percent':max(x for x in busy.values() if x is not None)})
mem2=observe_memory()
memory_ok=(mem2['available']>=72*2**30 and mem1['psi_full_avg10']==0 and mem2['psi_full_avg10']==0 and mem1['oom_events']==mem2['oom_events'])
cpu_ok=(set(cpus)<=online and set(cpus)<=mem2['process_cpus'] and all(x['max_busy_percent']<50 for x in loads))
fs_verified=fs['quota_user_group_project_status_verified'] and fs['readonly_existing_parent_write_search_access']
capacity_ok=fs_verified and fs['available_bytes_nonroot']>=required and fs['available_inodes']>=required_inodes
verdict='足りる' if capacity_ok else ('不足' if fs_verified and (fs['available_bytes_nonroot']<required or fs['available_inodes']<required_inodes) else '確認不能')
status='H4_SIGNAL_COMPILE_PLAN_PREPARED_AWAITING_SEPARATE_APPROVAL' if capacity_ok else 'H4_SIGNAL_COMPILE_PLAN_PREPARED_STORAGE_BLOCKED_STOP'
BUNDLE.mkdir(parents=True)
save('signal_compile_plan_v1.json',plan);save('authorization_draft_v1.json',authorization);save('stage_review_draft_v1.json',review)
save('authority_scope_v1.json',{'schema_version':'h4-signal-compile-preparation-authority-v1','user_instruction':'その方針で進めて',
    'interpreted_scope':'INPUT_BOUND plan and separate authorization/review drafts; stage capacity/resource preparation only',
    'prior_user_CPU_permission_scope':'six-input generation and freeze STOP only; proposed CPUs for next stage require stage-specific approval',
    'proposed_CPUs':cpus,'next_stage_CPU_permission_granted':False,'next_stage_review_approved':False,'explicit_launch_granted':False,
    'source_changed':False,'scientific_stage_started':False,'mandatory_stop':True})
save('binding_audit_v1.json',{'schema_version':'h4-signal-compile-metadata-binding-audit-v1','status':status,
    'observed_utc':datetime.now(timezone.utc).isoformat(),'plan_sha256':file_sha(BUNDLE/'signal_compile_plan_v1.json'),
    'plan_fingerprint':fp,'authorization_digest':auth_digest,'review_digest':fingerprint('h4-review-v1',review),
    'generation_freeze_file_sha256':completion['freeze_file_sha256'],'generation_freeze_fingerprint':completion['freeze_fingerprint'],
    'source_commit':SOURCE,'source_root':str(SCIENCE),'source19_unchanged':True,'templates218_unchanged':True,
    'input_byte_hashes_transcribed_from_verified_generation_freeze':True,'NPZ_read_or_load_this_preparation':False,
    'input_generation_plan_unchanged':True,'plan_changed_fields':[k for k in plan if plan[k]!=oldplan[k]],
    'structural_gate_passed':True,'source_contract_dependency_compiler_metadata_gate_passed':True,
    'negative_authorization_gates':negative,'approved':False,'explicit_launch_executed':False,
    'scientific_operations':{'SCF_DF_state_input_generation':0,'signal':0,'trajectory_seed_or_sampling':0,'circuit_build_compile_transpile':0,'GPU':0,'production_taskset_worker_runner_launch':0},
    'old_run01_source_and_failed_run_untouched':True})
save('stage_storage_estimate_v1.json',{'schema_version':'h4-signal-compile-stage-storage-estimate-v1','source_commit':SOURCE,
    'source_hashes_used':{p:plan['source_hashes'][p] for p in plan['source_hashes'] if Path(p).name in ('execution.py','ledger.py','workers.py','resources.py','signal.py','circuits.py')},
    'scope':'SIX_FROZEN_INPUTS_SIGNAL_COMPILE_THEN_MAP_COMPLETE_STOP','templates_per_geometry':218,
    'random_templates':random,'baseline_templates':baseline,'random_paired_trajectories':32,'logical_wrappers':logical,
    'actual_transpile_cap':logical,'signal_records':signals,'display_points_per_signal':302,
    'prior_wall_seconds':prior,'remaining_cumulative_wall_seconds':remaining,'maximum_wall_JSON_files':watch,
    'components':[{'output':n,'count_bound':c,'payload_bytes_allowance_each':s,'allocated_final_bytes_bound':c*round_block(s),'basis':b}
        for n,c,s,b in row_specs],'other_physical_components_bytes':{k:v for k,v in physical.items() if k not in [x[0] for x in row_specs]},
    'physical_planning_bound_bytes':bound,'required_available_bytes':required,'required_available_inodes':required_inodes,
    'new_inode_count_bound':inode_count,'directory_entry_allowance_bytes':256,'filesystem_block_bytes':block,
    'max_concurrent_temporary_publishers':8,'temp_final_relation':'hard-link publication aliases same inode/data; additionally8 full512KiB temp copies allowed conservatively',
    'inode_table_policy':'ext4 fixed-size inode tables already occupy allocated filesystem blocks; inode count checked separately. No extra4KiB per inode double charge; directory/extent/ACL/journal margins included.',
    'inode_table_primary_reference':'https://www.kernel.org/doc/html/latest/filesystems/ext4/inodes.html',
    'JSON_assumptions':{'metric_integer_decimal_digits_allowance':20,'metric_width_basis':'counts/depth/size of in-memory transpiled circuit under8GiB per role; 20digits generous planning assumption, not a new production gate',
        'float_JSON_numeric_characters_allowance':25,'shot_integer_digits_ceiling_for_finite_binary64_math_ceil':309,
        'metadata_only_extreme_width_signal_specimen_bytes':specimen_size,'signal_payload_allowance_each':512*2**10,
        'record_and_delta_JSON_bytes_each':4096,'source_does_not_enforce_these_per_file_JSON_limits':True},
    'known_not_written':{'NPZ_NPY_arrays':True,'circuits_QPY_QASM':True,'SQLite_cache':True,'old_runtime_cache_checkpoint':True},
    'cache_storage_policy':'in-memory identity/reuse/registry; persisted wrapper records and ledger deltas counted; no circuit cache file',
    'prior_byte_budget_journal_bytes':len(journal),'prior_cumulative_charge_bytes':prior_charge,'stage_cumulative_charge_bound':charge,
    'combined_cumulative_charge_bound':prior_charge+charge,'source_total_campaign_charge_cap_bytes':OUTPUT_CAP,
    'charge_cap_requires_10GiB_free_before_stage':False,'publication_count_bound':pubs,
    'scientific_arrays_generated':0,'scientific_metadata_specimens_saved_as_results':False,'production_stage_executed':False})
save('filesystem_quota_observation_v1.json',fs)
save('resource_observation_v1.json',{'schema_version':'h4-signal-compile-preparation-resource-observation-v1',
    'observed_utc':datetime.now(timezone.utc).isoformat(),'context':'unlaunched preparation process; broad affinity is observation only, not permission',
    'proposed_CPUs':cpus,'proposed_workers':6,'own_run_taskset_mask':'0x3e8','physical_topology':top,
    'SMT_sibling_load_sample_seconds':duration,'SMT_sibling_loads':loads,'CPU_candidate_observation_pass':cpu_ok,
    'busy_threshold_percent':50,'busy_threshold_is_launch_proposal_not_science_condition':True,
    'memory_available_bytes':mem2['available'],'required_available_memory_bytes':72*2**30,
    'headroom_bytes':16*2**30,'per_role_AS_RSS_cap_bytes':8*2**30,
    'memory_pressure_OOM_observation_pass':memory_ok,'observation_before':{k:sorted(v) if isinstance(v,set) else v for k,v in mem1.items()},
    'observation_after':{k:sorted(v) if isinstance(v,set) else v for k,v in mem2.items()},'affinity_changed':False,'worker_launched':False,'GPU_query_or_use':False})
save('storage_verdict_v1.json',{'schema_version':'h4-signal-compile-stage-storage-verdict-v1','status':status,
    'observed_utc':fs['observed_utc'],'verdict':verdict,'required_bytes':required,'available_bytes':fs['available_bytes_nonroot'],
    'byte_margin':fs['available_bytes_nonroot']-required,'shortfall_bytes':max(0,required-fs['available_bytes_nonroot']),
    'required_inodes':required_inodes,'available_inodes':fs['available_inodes'],'quota_verified':fs_verified,
    'quota_status':fs['quota_interpretation'],'CPU_candidate_observation_pass':cpu_ok,'memory_observation_pass':memory_ok,
    'capacity_reserved':False,'launch_authorized':False,'review_approved':False,'mandatory_stop':True,
    'minimum_remedy_proposal_only':None if capacity_ok else 'Provide at least the recorded stage free-byte/inode requirement at fixed output filesystem; no deletion, reservation, output-root relocation or shared-setting change executed.'})
assert baseline_files=={str(p):file_sha(p) for p in OLD.iterdir() if p.is_file()}
assert file_sha(OUTPUT/'byte-budget.journal')==hashlib.sha256(journal).hexdigest()
assert file_sha(OUTPUT/'generation-freeze.json')==completion['freeze_file_sha256']
for path,expected in plan['source_hashes'].items():assert file_sha(SCIENCE/path)==expected
print(json.dumps({'status':status,'storage_verdict':verdict,'physical_bound_GiB':bound/2**30,
    'required_GiB':required/2**30,'available_GiB':fs['available_bytes_nonroot']/2**30,
    'shortfall_MiB':max(0,required-fs['available_bytes_nonroot'])/2**20,'required_inodes':required_inodes,
    'available_inodes':fs['available_inodes'],'quota':fs['quota_interpretation'],'memory_GiB':mem2['available']/2**30,
    'CPU_observation_pass':cpu_ok,'memory_observation_pass':memory_ok,
    'signal_JSON_metadata_specimen_bytes':specimen_size,'plan_fingerprint':fp,
    'combined_charge_GiB':(prior_charge+charge)/2**30,'review_approved':False,
    'source19_unchanged':True,'scientific_operations':0,'launches':0},sort_keys=True))
