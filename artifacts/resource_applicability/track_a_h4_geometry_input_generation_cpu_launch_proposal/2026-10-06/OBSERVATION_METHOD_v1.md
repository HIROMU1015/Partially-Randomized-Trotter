# 読み取り専用手順・metadata検査・全attempt履歴

この文書は今回実行した一時手順の記録で、production runnerでも追加実行の指示でもない。
repositoryへのPython module追加・source変更は行っていない。科学/tmp手順を再実行して既存資料を上書きしない。

全Python commandで既存absolute interpreterとthread/process1のprocess限定環境を使用した。
下記は実行記録。観測01/02の失敗ログを保持し、03では容量未達を解決済みと扱わず保存した。

| attempt | 結果 | log |
|---|---|---|
| CPU観測01 | fixed artifact親directory未作成でstatvfs失敗。科学/output作成0。 | [01](cpu-observation-attempt-01.log) |
| CPU観測02 | 既存親filesystemを測定、空きが10GiB cap未満という保守的条件で停止。 | [02](cpu-observation-attempt-02.log) |
| CPU観測03 | topology/load/resource/容量を保存。容量は未解決launch条件。 | [03](cpu-observation-attempt-03.log) |
| gate検査01 | 12 PASS、fail/error/skip0。保存review falseの拒否、合格経路はメモリ内模擬承認だけ。 | [gate](gate-tests-attempt-01.log) |

```bash
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
QISKIT_PARALLEL=false QISKIT_NUM_PROCS=1 \
/home/AbeHiromu/venvs/trotter-common/bin/python -B \
  /tmp/h4_cpu_launch_proposal.py observe \
  > artifacts/resource_applicability/track_a_h4_geometry_input_generation_cpu_launch_proposal/2026-10-06/cpu-observation-attempt-01.log 2>&1

PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
QISKIT_PARALLEL=false QISKIT_NUM_PROCS=1 \
/home/AbeHiromu/venvs/trotter-common/bin/python -B \
  /tmp/h4_cpu_launch_proposal.py observe \
  > artifacts/resource_applicability/track_a_h4_geometry_input_generation_cpu_launch_proposal/2026-10-06/cpu-observation-attempt-02.log 2>&1

PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
QISKIT_PARALLEL=false QISKIT_NUM_PROCS=1 \
/home/AbeHiromu/venvs/trotter-common/bin/python -B \
  /tmp/h4_cpu_launch_proposal.py observe \
  > artifacts/resource_applicability/track_a_h4_geometry_input_generation_cpu_launch_proposal/2026-10-06/cpu-observation-attempt-03.log 2>&1

PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
QISKIT_PARALLEL=false QISKIT_NUM_PROCS=1 \
/home/AbeHiromu/venvs/trotter-common/bin/python -B \
  /tmp/h4_cpu_launch_proposal.py draft-tests \
  > artifacts/resource_applicability/track_a_h4_geometry_input_generation_cpu_launch_proposal/2026-10-06/gate-tests-attempt-01.log 2>&1

```

一時手順snapshotのSHA-256：

- observation_attempt01: `e746e36fd056c3ee50730e037ab62f4c069e2ddd8f02e0913a81c38b264b51fe`
- observation_attempt02: `4751ee789d1b78957ad357431984c15d64c5a5f781f31ccba0ba8253f5ca7d04`
- final_observation_and_gate_tests: `45bd64ed4f3f46b7b7ab35fa66235f653629634a9bdcf9815d05894a7a7f4641`

01→02では固定outputには触れず、statvfsの対象を最も近い既存上位directoryへ変更した。
02→03では全10GiB cap分の空きチェックをsource gateとして扱わず、観測値/未解決運用条件として保存した。
science source、production guard、output capを変更していない。

観測の入力はonline/topology/NUMA sysfs、own /proc/status/cgroup/mountinfo、
/proc/stat・loadavg・meminfo・pressure、可視cgroup祖先のmemory/cpuset、既存artifact親filesystemだけ。
分子snapshot・output/registry・旧runtime/cache/checkpoint・他job個別process・GPUにはアクセスしない。
約3秒の受動CPU counter差分だけで、benchmark・taskset・worker起動はない。

最終一時手順の全文（review用、repository sourceへ追加しない）：

```python
"""One-off guarded, read-only CPU/resource observation and metadata gate checks."""
import argparse
import builtins
from datetime import datetime,timezone,timedelta
import functools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import types
import unittest
from unittest.mock import Mock,patch

SCIENCE=Path('/home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/worktrees/track-a-h4-geometry-resource-observer-fix-20261006')
BASE='a1b0ba5e1ae14c2fd7c345a4d979b85b1eff2538'
SOURCE='9ab38665920dfb5ac0a9d038233e1f3bf5d8fe5a'
OLD_AUTH='artifacts/resource_applicability/track_a_h4_geometry_input_generation_authorization/2026-10-06-v2'
BUNDLE='artifacts/resource_applicability/track_a_h4_geometry_input_generation_cpu_launch_proposal/2026-10-06'
STATUS='H4_INPUT_GENERATION_CPU_LAUNCH_PROPOSAL_FROZEN_AWAITING_APPROVAL'
ROOT=Path.cwd()
OUTPUT='/home/AbeHiromu/projects/partially-randomized-trotter/artifacts/resource_applicability/track_a_h4_geometry_execution/track-a-h4-geometry-v2-20261006-run01'
COUNTS=dict(molecular_access=0,molecular_import=0,science_processing=0,seed_generation=0,transpile=0,
    GPU=0,production_launch_worker_taskset=0,shared_environment_changes=0,other_job_changes=0)
ATTEMPTS=[]


def install_guard():
    expected={k:'1' for k in ('PYTHONNOUSERSITE','PYTHONDONTWRITEBYTECODE','OPENBLAS_NUM_THREADS',
        'OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','RAYON_NUM_THREADS','QISKIT_NUM_PROCS')}
    expected['QISKIT_PARALLEL']='false'
    if not all(os.environ.get(k)==v for k,v in expected.items()):raise RuntimeError('single-thread process environment required')
    def refuse(scope,value):
        COUNTS[scope]+=1;ATTEMPTS.append({'scope':scope,'attempt':str(value)})
        raise RuntimeError('zero-science/own-metadata boundary '+str(value))
    original_import=builtins.__import__
    def guarded_import(name,*a,**kw):
        if name.split('.')[0] in ('pyscf','openfermion','openfermionpyscf','trotterlib'):refuse('molecular_import',name)
        if name.split('.')[0] in ('cupy','torch','pynvml'):refuse('GPU',name)
        return original_import(name,*a,**kw)
    builtins.__import__=guarded_import
    def path_guard(path):
        if not isinstance(path,(str,bytes,os.PathLike)):return
        value=os.fsdecode(path)
        if value==OUTPUT or value.startswith(OUTPUT+'/') or value.endswith(('.npz','.npy','.pkl','.pickle','.db','.sqlite','.sqlite3')) or any(
                part in value.split('/') for part in ('.runtime','runtime','cache','caches','checkpoint','checkpoints','registry','registries')):
            refuse('molecular_access',value)
    stat,lstat,resolve,statvfs=os.stat,os.lstat,Path.resolve,os.statvfs
    def guarded_stat(path,*a,**kw):path_guard(path);return stat(path,*a,**kw)
    def guarded_lstat(path,*a,**kw):path_guard(path);return lstat(path,*a,**kw)
    def guarded_resolve(self,*a,**kw):path_guard(self);return resolve(self,*a,**kw)
    def guarded_statvfs(path,*a,**kw):path_guard(path);return statvfs(path,*a,**kw)
    os.stat,os.lstat,Path.resolve,os.statvfs=guarded_stat,guarded_lstat,guarded_resolve,guarded_statvfs
    def hook(event,arguments):
        if event=='open':path_guard(arguments[0])
        if event=='subprocess.Popen':
            command=arguments[1]
            good=isinstance(command,(list,tuple)) and len(command)>3 and command[0]=='git' and command[1]=='-C'
            action=command[3] if good else None
            good=good and action in ('show','rev-parse','diff','ls-tree','merge-base')
            if action=='merge-base':good=good and command[4]=='--is-ancestor'
            if not good:refuse('production_launch_worker_taskset',command)
        if event in ('os.system','os.posix_spawn','os.fork','os.forkpty','os.exec'):refuse('production_launch_worker_taskset',event)
        if event in ('os.kill','os.killpg','os.setpriority','os.sched_setaffinity'):refuse('other_job_changes',event)
    sys.addaudithook(hook)
    sys.path.insert(0,str(SCIENCE/'src'))
    prefix='trottertracks.resource_applicability.h4_geometry.'
    for name,symbols in {
        'execution':('launch','generation_stage','signal_stage','OwnedRun','_generate_worker','_compile_worker','input_boundary','load_new_input'),
        'workers':('OwnedPool','owned_worker_main','private_dispatch'),
        'inputs':('generate_input','freeze_input'),'signal':('sample_events','trajectory_seeds','corrected_signal'),
        'circuits':('build_evolution','wrapper'),'parallel':('compile_wrappers',)}.items():
        module=types.ModuleType(prefix+name)
        for symbol in symbols:setattr(module,symbol,Mock(side_effect=lambda *a,_name=name+'.'+symbol,**kw:refuse('science_processing',_name)))
        sys.modules[prefix+name]=module
    from trottertracks.resource_applicability.h4_geometry import gates,identity,resources
    identity.require(Path(gates.__file__).absolute()==SCIENCE/'src/trottertracks/resource_applicability/h4_geometry/gates.py','existing science checkout only')
    identity.trajectory_seed=Mock(side_effect=lambda *a,**kw:refuse('seed_generation','trajectory_seed'))
    identity.step_seed=Mock(side_effect=lambda *a,**kw:refuse('seed_generation','step_seed'))
    resources.OutputBudget=Mock(side_effect=lambda *a,**kw:refuse('science_processing','production OutputBudget'))
    resources.limit_owned_address_space=Mock(side_effect=lambda *a,**kw:refuse('production_launch_worker_taskset','set production AS'))
    import qiskit
    @functools.wraps(qiskit.transpile)
    def no_transpile(*a,**kw):refuse('transpile','qiskit.transpile')
    qiskit.transpile=no_transpile
    return gates,identity,resources,expected


def write_new(name,value,raw=None):
    path=ROOT/BUNDLE/name;path.parent.mkdir(parents=True,exist_ok=True)
    if raw is not None:
        with path.open('xb') as f:f.write(raw)
    else:
        with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False);f.write('\n')


def cpu_times():
    data={}
    raw=Path('/proc/stat').read_text()
    for line in raw.splitlines():
        row=line.split()
        if row and row[0].startswith('cpu') and row[0][3:].isdigit():
            # guest/guest_nice are already counted in user/nice; use first 8.
            values=list(map(int,row[1:9]));data[int(row[0][3:])]=values
    return data


def observe(gates,identity,resources):
    require=identity.require
    utc0=datetime.now(timezone.utc).isoformat();mono0=time.monotonic();first=cpu_times()
    online_text=Path('/sys/devices/system/cpu/online').read_text().strip();online=resources.cpus(online_text)
    status=Path('/proc/self/status').read_text()
    process_text=next(l.split(':',1)[1].strip() for l in status.splitlines() if l.startswith('Cpus_allowed_list:'))
    process_cpus=resources.cpus(process_text)
    membership=Path('/proc/self/cgroup').read_text();mountinfo=Path('/proc/self/mountinfo').read_text()
    namespace=os.readlink('/proc/self/ns/cgroup')
    hierarchy=resources.cgroup_hierarchy(membership,mountinfo,namespace)
    cpuset_records=[];effective_sets=[]
    for entry in hierarchy:
        p=entry['path']
        require(entry['v2'],'this proposal requires observed v2 launch context')
        controller_enabled=entry['is_root'] or 'cpuset' in (p.parent/'cgroup.subtree_control').read_text().split()
        try:
            text=(p/'cpuset.cpus.effective').read_text().strip();values=resources.cpus(text)
            effective_sets.append(values)
            cpuset_records.append({'path':str(p),'status':'READ_EFFECTIVE_SET','cpus':sorted(values),'raw':text,
                                   'parent_cpuset_enabled':controller_enabled})
        except FileNotFoundError:
            require(not controller_enabled,'enabled cpuset interface missing')
            cpuset_records.append({'path':str(p),'status':'NO_LOCAL_CPUSET_CONTROLLER_PARENT_DISABLED',
                                   'cpus':None,'parent_cpuset_enabled':False})
    require(effective_sets,'cpuset effective CPU set not observed')
    usable=set(online)&set(process_cpus)
    for cpus in effective_sets:usable &= cpus
    numa={}
    for p in Path('/sys/devices/system/node').glob('node[0-9]*'):
        if p.name[4:].isdigit():numa[int(p.name[4:])]=resources.cpus((p/'cpulist').read_text())
    require(numa,'NUMA mapping unavailable')
    topology={}
    for cpu in sorted(online):
        p=Path('/sys/devices/system/cpu')/('cpu'+str(cpu))/'topology'
        package=int((p/'physical_package_id').read_text());core=int((p/'core_id').read_text())
        siblings=resources.cpus((p/'thread_siblings_list').read_text())
        nodes=[node for node,cpus in numa.items() if cpu in cpus]
        require(package>=0 and core>=0 and cpu in siblings and len(nodes)==1,'incomplete CPU topology')
        topology[cpu]={'cpu':cpu,'physical_package_id':package,'core_id':core,'SMT_siblings':sorted(siblings),
                       'NUMA_node':nodes[0],'online':True,'process_cpuset_usable':cpu in usable}
    # Passive load sample only; no benchmark or worker/affinity manipulation.
    remaining=3.0-(time.monotonic()-mono0)
    if remaining>0:time.sleep(remaining)
    second=cpu_times();mono1=time.monotonic();utc1=datetime.now(timezone.utc).isoformat()
    require(Path('/sys/devices/system/cpu/online').read_text().strip()==online_text,'CPU online set changed during observation')
    require(Path('/proc/self/cgroup').read_text()==membership and Path('/proc/self/mountinfo').read_text()==mountinfo
            and os.readlink('/proc/self/ns/cgroup')==namespace,'context changed during CPU observation')
    usage={}
    for cpu in sorted(online):
        require(cpu in first and cpu in second,'CPU sample missing')
        delta=[b-a for a,b in zip(first[cpu],second[cpu])]
        require(all(x>=0 for x in delta) and sum(delta)>0,'CPU counter interval invalid')
        total=sum(delta);busy=total-delta[3]-delta[4]
        usage[cpu]={'busy_percent':100*busy/total,'iowait_percent':100*delta[4]/total,
                    'steal_percent':100*delta[7]/total,'counter_delta_first8':delta,
                    'counter_before_first8':first[cpu],'counter_after_first8':second[cpu]}
    physical={}
    for cpu in sorted(usable):
        row=topology[cpu];key=(row['physical_package_id'],row['core_id'])
        physical.setdefault(key,[]).append(cpu)
    node_options={}
    for key,candidates in physical.items():
        representative=min(candidates,key=lambda cpu:(usage[cpu]['busy_percent'],cpu))
        row=topology[representative];online_siblings=set(row['SMT_siblings'])&online
        require(all((topology[c]['physical_package_id'],topology[c]['core_id'])==key for c in online_siblings),'SMT/core mapping inconsistent')
        core_busy=max(usage[c]['busy_percent'] for c in online_siblings)
        record={**row,'representative_busy_percent':usage[representative]['busy_percent'],
                'max_online_SMT_busy_percent':core_busy,
                'sum_online_SMT_busy_percent':sum(usage[c]['busy_percent'] for c in online_siblings)}
        node_options.setdefault(row['NUMA_node'],[]).append(record)
    ranked={node:sorted(rows,key=lambda r:(r['max_online_SMT_busy_percent'],r['sum_online_SMT_busy_percent'],r['representative_busy_percent'],r['cpu']))
            for node,rows in node_options.items()}
    choices=[(node,rows[:6]) for node,rows in ranked.items() if len(rows)>=6]
    require(choices,'cannot select six distinct physical cores in an observed NUMA node; review smaller-worker option')
    node,selected=min(choices,key=lambda pair:(max(r['max_online_SMT_busy_percent'] for r in pair[1]),
        sum(r['max_online_SMT_busy_percent'] for r in pair[1]),sum(r['sum_online_SMT_busy_percent'] for r in pair[1]),pair[0]))
    require(max(r['max_online_SMT_busy_percent'] for r in selected)<50,
        'no sufficiently low-load six-core option in passive sample; review smaller-worker option')
    selected=sorted(selected,key=lambda r:r['cpu']);proposal=[r['cpu'] for r in selected]
    require(len(proposal)==6 and len({(r['physical_package_id'],r['core_id']) for r in selected})==6,'six distinct cores')
    memory=resources.observe_memory();memory['process_cpus']=sorted(memory['process_cpus'])
    # Query the existing artifact parent filesystem, not the fixed output path.
    fs_path=Path(gates.ARTIFACT_ANCHOR)/'artifacts/resource_applicability'
    missing_parent_candidates=[]
    while True:
        try:
            fs=os.statvfs(fs_path);break
        except FileNotFoundError:
            missing_parent_candidates.append(str(fs_path))
            require(fs_path!=Path(gates.ARTIFACT_ANCHOR),'project filesystem parent is not visible')
            fs_path=fs_path.parent
    fs_available=fs.f_bavail*fs.f_frsize
    mounts=[]
    for line in mountinfo.splitlines():
        before,after=line.split(' - ',1);fields=before.split();suffix=after.split()
        mount=Path(fields[4])
        if fs_path.is_relative_to(mount):mounts.append((len(mount.parts),fields,suffix))
        require(not Path(OUTPUT).is_relative_to(mount) or fs_path.is_relative_to(mount),
                'future output has a distinct mount below observed parent; filesystem proof needs review')
    require(mounts,'artifact filesystem mount not identified')
    _depth,fields,suffix=max(mounts,key=lambda row:row[0])
    taskset_path=shutil.which('taskset');env_path=shutil.which('env')
    require(taskset_path and env_path,'existing taskset/env not found; do not install')
    full_cap_space_available=fs_available>=10*2**30  # capacity evidence, never a guard relaxation
    report={'schema_version':'h4-input-cpu-launch-observation-v1','status':STATUS,
        'sample_start_utc':utc0,'sample_end_utc':utc1,'sample_seconds':mono1-mono0,
        'observed_JST':datetime.now(timezone(timedelta(hours=9))).isoformat(),
        'science_checkout_root':str(SCIENCE),'preparation_checkout_root':str(ROOT),
        'online_CPU_list':online_text,'online_CPUs':sorted(online),'process_CPU_list':process_text,'process_CPUs':sorted(process_cpus),
        'cpuset_hierarchy':cpuset_records,'usable_CPUs':sorted(usable),'NUMA_CPU_lists':{str(n):sorted(c) for n,c in numa.items()},
        'topology':{str(c):row for c,row in topology.items()},'CPU_usage':{str(c):row for c,row in usage.items()},
        'selection_rule':'One representative per package/core; lowest passive logical CPU load within core, conservatively rank core by busiest online SMT sibling and sum sibling load; choose six within one NUMA node with lowest worst/sum observed load. No preselected CPU0-5.',
        'operational_low_load_heuristic_max_percent':50,'heuristic_is_scientific_or_runtime_guard':False,
        'NUMA_options_six_lowest_per_core':{str(n):rows[:6] for n,rows in ranked.items() if len(rows)>=6},
        'proposed_CPUs':proposal,'selected_physical_cores':selected,'proposed_NUMA_node':node,'proposed_workers':6,
        'distinct_physical_core_count':6,'CPU_use_approved':False,'exclusive_reserved':False,'approval_status':'利用者承認未取得',
        'short_sample_predicts_future_load':False,'loadavg_raw':Path('/proc/loadavg').read_text().strip(),
        'memory_observation_utc':datetime.now(timezone.utc).isoformat(),'memory_observation':memory,
        'kernel_release':os.uname().release,'cgroup_namespace':namespace,'own_cgroup_membership':membership,
        'cgroup_mountinfo_lines':[line for line in mountinfo.splitlines() if ' - cgroup' in line],
        'filesystem':{'observed_existing_parent':str(fs_path),'absent_parent_candidates':missing_parent_candidates,'mountpoint':fields[4],'filesystem_root':fields[3],
            'filesystem_type':suffix[0],'filesystem_source':suffix[1],'available_bytes':fs_available,
            'total_bytes':fs.f_blocks*fs.f_frsize,'output_budget_bytes':10*2**30,'full_output_cap_space_available':full_cap_space_available,
            'fixed_output_resolve_stat_created':False,'space_reserved':False,'user_project_quota_verified':False},
        'taskset_binary_path':taskset_path,'env_binary_path':env_path,'taskset_executed':False,
        'actual_launch_admission_performed':False,'actual_workers':None,'execution_ready':False,
        'source_plan_environment_or_other_job_modified':False,'additional_transpile':0,'series_transpile_cumulative':28,
        'guard_counts':COUNTS,'protected_attempts':ATTEMPTS,'mandatory_stop':True}
    write_new('cpu_resource_observation_v1.json',report)
    print(json.dumps({'proposed_CPUs':proposal,'physical_core_count':6,'NUMA_node':node,
        'sample_seconds':mono1-mono0,'selected_core_busy_percent':[r['max_online_SMT_busy_percent'] for r in selected],
        'effective_available_GiB':memory['available']/2**30,'filesystem_available_GiB':fs_available/2**30,
        'PSI_full_avg10':memory['psi_full_avg10'],'OOM_events':memory['oom_events'],
        'full_output_cap_space_available':full_cap_space_available,'CPU_permission':'利用者承認未取得','taskset_executed':False,'guard_counts':COUNTS},sort_keys=True))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('observe','draft-tests'))
    args=parser.parse_args();gates,identity,resources,expected=install_guard()
    if args.mode=='observe':observe(gates,identity,resources);return
    observation=json.loads((ROOT/BUNDLE/'cpu_resource_observation_v1.json').read_bytes())
    cpus=observation['proposed_CPUs']
    oldp=(SCIENCE/OLD_AUTH/'input_generation_plan_v2.json').read_bytes()
    olda=json.loads((SCIENCE/OLD_AUTH/'authorization_draft_v2.json').read_bytes())
    oldr=json.loads((SCIENCE/OLD_AUTH/'stage_review_v2.json').read_bytes())
    plan=json.loads(oldp);auth={**olda,'allowed_cpus':cpus};review={**oldr,'authorization_digest':identity.fingerprint('h4-authorization-v1',auth),'approved':False}
    write_new('input_generation_plan_v2.json',None,oldp);write_new('authorization_proposal_v1.json',auth);write_new('stage_review_proposal_v1.json',review)
    preflight=json.loads(Path('/tmp/h4-cpu-launch-preflight.json').read_bytes());write_new('preflight_identity_v1.json',preflight)
    class GateTests(unittest.TestCase):
        def copy(self):return json.loads(json.dumps((plan,auth,review)))
        def simulated(self):
            p,a,r=self.copy();r['approved']=True;return p,a,r
        def permit(self,documents):return gates.authorize('input_generation',*documents,explicit_launch=True)
        def test_saved_review_false_rejected(self):
            gates.structural_gate(plan,auth,review)
            with self.assertRaisesRegex(identity.Stop,'review/authorization'):self.permit(self.copy())
        def test_simulated_approval_metadata_checkout_only(self):
            p,a,r=self.simulated();permit=self.permit((p,a,r));contract,options=gates.checkout_gate(permit)
            self.assertEqual(contract['templates'],p['templates']);self.assertEqual(options['num_processes'],1)
            self.assertEqual(permit.source_root,str(SCIENCE));self.assertIs(review['approved'],False)
        def test_plan_is_byte_identical_and_root_preserved(self):
            self.assertEqual((ROOT/BUNDLE/'input_generation_plan_v2.json').read_bytes(),oldp)
            self.assertEqual(identity.sha(oldp),'8d4ee43c3d7d74ba30cbd495a4c49dd0798ca27d5935df9df97125f3c069a256')
            self.assertEqual(plan['source_root'],str(SCIENCE));self.assertEqual(plan['requested_workers'],6)
            self.assertIsNone(plan['inputs']);self.assertIsNone(plan['generation_freeze_digest'])
        def test_only_allowed_CPU_and_review_auth_digest_changed(self):
            self.assertEqual({k:v for k,v in auth.items() if k!='allowed_cpus'},{k:v for k,v in olda.items() if k!='allowed_cpus'})
            self.assertEqual({k:v for k,v in review.items() if k!='authorization_digest'},{k:v for k,v in oldr.items() if k!='authorization_digest'})
            self.assertEqual(auth['allowed_cpus'],cpus);self.assertIs(review['approved'],False)
        def test_binding_matches_frozen_plan(self):
            self.assertEqual(auth['plan_fingerprint'],identity.fingerprint('h4-execution-plan-v1',plan))
            self.assertEqual(auth['plan_fingerprint'],'8514368280d33bd89aff09891bbb25204d5f6e41acc7adcd176b317c0e5d6a5d')
            self.assertEqual(review['authorization_digest'],identity.fingerprint('h4-authorization-v1',auth))
        def test_source_19_paths_actual_checkout_and_blobs(self):
            for path,expected in plan['source_hashes'].items():
                self.assertEqual(identity.sha((SCIENCE/path).read_bytes()),expected)
                self.assertEqual(gates.git_blob(SCIENCE,SOURCE,path),(SCIENCE/path).read_bytes())
                self.assertEqual((ROOT/path).read_bytes(),(SCIENCE/path).read_bytes())
        def test_old_AUTH_REVIEW_mixing_rejected(self):
            p,a,r=self.simulated();oldreview={**oldr,'approved':True}
            with self.assertRaises(identity.Stop):self.permit((p,a,oldreview))
            with self.assertRaises(identity.Stop):self.permit((p,olda,r))
        def test_CPU_list_topology_NUMA_available_intersection(self):
            self.assertEqual(len(cpus),6);self.assertEqual(len(set(cpus)),6)
            self.assertTrue(set(cpus)<=set(observation['usable_CPUs']))
            rows=[observation['topology'][str(c)] for c in cpus]
            self.assertEqual(len({(r['physical_package_id'],r['core_id']) for r in rows}),6)
            self.assertEqual(len({r['NUMA_node'] for r in rows}),1)
        def test_future_CPU_subset_predicate_and_synthetic_admission(self):
            identity.require(set(cpus)<=set(auth['allowed_cpus']),'process can use unpermitted CPU')
            with self.assertRaises(identity.Stop):identity.require(set(observation['process_CPUs'])<=set(cpus),'process can use unpermitted CPU')
            self.assertEqual(resources.admission(72*2**30,10,6,cpus,cpus,now=10),6)
            with self.assertRaises(identity.Stop):resources.admission(72*2**30,10,6,[],cpus,now=10)
        def test_no_launch_or_other_stage_or_permission(self):
            with self.assertRaises(identity.Stop):gates.authorize('input_generation',*self.simulated(),explicit_launch=False)
            with self.assertRaises(identity.Stop):gates.authorize('signal_compile',*self.simulated(),explicit_launch=True)
            p,a,r=self.simulated();a['permission']='signal_compile';r['authorization_digest']=identity.fingerprint('h4-authorization-v1',a)
            with self.assertRaises(identity.Stop):self.permit((p,a,r))
        def test_CPU_malformed_or_duplicate_proposals_rejected(self):
            for replacement in ([],[-1],[True],[str(cpus[0])],[cpus[0],cpus[0]]):
                p,a,r=self.simulated();a['allowed_cpus']=replacement;r['authorization_digest']=identity.fingerprint('h4-authorization-v1',a)
                with self.assertRaises(identity.Stop):self.permit((p,a,r))
        def test_new_preparation_root_not_science_root(self):
            p,a,r=self.simulated();p['source_root']=str(ROOT)
            a['plan_fingerprint']=identity.fingerprint('h4-execution-plan-v1',p);r['plan_fingerprint']=a['plan_fingerprint'];r['authorization_digest']=identity.fingerprint('h4-authorization-v1',a)
            with self.assertRaises(identity.Stop):gates.checkout_gate(self.permit((p,a,r)))
    started=time.monotonic();result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(GateTests))
    success=result.wasSuccessful() and not result.skipped and not ATTEMPTS
    report={'schema_version':'h4-cpu-launch-metadata-gate-tests-v1','status':'PASS' if success else 'FAIL',
        'observed_utc':datetime.now(timezone.utc).isoformat(),'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
        'skipped':len(result.skipped),'wall_seconds':time.monotonic()-started,'command_argv':sys.argv,
        'python_executable':sys.executable,'python_version':sys.version,'thread_environment':expected,
        'saved_review_approved':False,'CPU_permission':'利用者承認未取得','metadata_only_simulated_approval_in_memory':True,
        'private_science_output_runner_worker_boundaries_mocked':True,'taskset_executed':False,'production_launch_executed':False,
        'guard_counts':COUNTS,'protected_attempts':ATTEMPTS,'additional_transpile':0,'series_transpile_cumulative':28,'mandatory_stop':True}
    write_new('gate-tests-attempt-01.json',report)
    print('CPU_LAUNCH_METADATA_AUDIT '+json.dumps(report,sort_keys=True))
    if not success:raise SystemExit(1)


if __name__=='__main__':main()

```
