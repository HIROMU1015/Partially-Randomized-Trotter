"""H7-only input/dispatch adapter; reuse H5 own-process supervision unchanged."""
import os
import json
import resource
import time
from pathlib import Path
from .contract import Calls, call_limits, digest, tasks, verify_grant
from ..h5_open_shell.execution import write_json, file_hash, verify_assets
from .monitor import supervise


def runtime_observer(output,limits,deadline,*,host=None,self_usage=None):
    from ..h6_server_resources_v1 import host_memory,process_memory
    host=host or host_memory;self_usage=self_usage or (lambda:process_memory(os.getpid()))
    def observe():
        if time.monotonic()>=deadline:raise RuntimeError('H7_WALL_LIMIT')
        own=self_usage()
        if own['VmRSS_bytes']>limits['coordinator_RSS_bytes']:raise RuntimeError('H7_COORDINATOR_RSS')
        if own['VmSize_bytes']>limits['coordinator_AS_bytes']:raise RuntimeError('H7_COORDINATOR_AS')
        paths=list(Path(output).rglob('*'));files=[p for p in paths if p.is_file()]
        if len(paths)>limits['output_inodes']:raise RuntimeError('H7_OUTPUT_INODES')
        if sum(p.stat().st_size for p in files)>limits['output_bytes']-4*2**20:
            raise RuntimeError('H7_OUTPUT_TERMINAL_RESERVE')
        result=host();result['coordinator_memory']=own
        return result
    return observe


def input_worker(directory,task):
    from .input import rohf_integrals,weighted_df,array_identity
    from .science import solve_state,preparations,panel_signals,validate_primitives
    import numpy as np
    calls=Calls(call_limits(100))  # Admission cap, never a rank selection.
    def record(key,used):
        if used[key]==1 or used[key]%100==0:
            path=directory/'call_reservations.jsonl'
            if path.exists() and path.stat().st_size>4*2**20:raise RuntimeError('CALL_JOURNAL_CAP')
            with path.open('a') as f:f.write(json.dumps(dict(before_operation=key,used=used))+'\n');f.flush()
    calls.recorder=record
    scratch=directory/'scratch';scratch.mkdir()
    arrays,receipt=rohf_integrals(scratch,calls)
    with (directory/'integrals.npz').open('xb') as f:np.savez(f,**arrays)
    write_json(directory/'integral_receipt.json',receipt)
    ham,projection,raw=weighted_df(float(arrays['constant']),arrays['one_body'],arrays['two_body'],calls)
    with (directory/'raw_df.npz').open('xb') as f:np.savez(f,**raw)
    write_json(directory/'projection_receipt.json',projection)
    actual=call_limits(len(ham.lambdas))
    if any(calls.used[k]>actual[k] for k in calls.used):raise RuntimeError('ACTUAL_RANK_CALL_PLAN')
    if any(actual[k]>calls.caps[k] for k in actual):raise RuntimeError('ACTUAL_RANK_ADMISSION_EXCEEDED')
    calls.caps=actual
    sector,state,full,checks,matrix=solve_state(ham,calls)
    prepared=preparations(ham)
    primitives=validate_primitives(ham,sector,full,state,prepared,calls)
    checks['primitive_max_error']=primitives['max_error']
    records=panel_signals(ham,sector,state,matrix,checks,prepared,calls)
    # No hidden compressed cache or pickle; preserve the raw generation and state.
    snapshot=dict(constant=np.array(ham.constant),one_body=ham.one_body,lambdas=ham.lambdas,
        g_matrices=np.array(ham.g_matrices),sector_basis=sector.basis_indices,state=state,full=full)
    with (directory/'snapshot.npz').open('xb') as f:np.savez(f,**snapshot)
    checks=dict(checks,reference=dict(real=checks['reference'].real,imag=checks['reference'].imag))
    result=dict(task=task,input=receipt,DF=projection,state_checks=checks,primitive_checks=primitives,
        signal_records=records,calls_attempted=calls.used,call_caps=actual,
        snapshot=dict(path=str(directory/'snapshot.npz'),sha256=file_hash(directory/'snapshot.npz'),
            arrays={k:array_identity(v) for k,v in snapshot.items()}))
    return result


def execute(root,manifest,grant):
    """Only explicit fresh H7 authorization reaches this function."""
    from ..h6_server_resources_v1 import host_memory
    output=verify_grant(manifest,grant,manifest['SOURCE']);verify_assets(root,manifest)
    cpus=manifest['worker_cpus'];limits=manifest['limits']
    if not 1<=len(cpus)<=4 or len(set(cpus))!=len(cpus) or not set(cpus)<=os.sched_getaffinity(0):raise ValueError('FRESH_CPUS')
    if manifest['coordinator_cpu'] in cpus or manifest['coordinator_cpu'] not in os.sched_getaffinity(0):raise ValueError('COORDINATOR_CPU')
    if resource.getrlimit(resource.RLIMIT_AS)[1] not in (-1,resource.RLIM_INFINITY) and resource.getrlimit(resource.RLIMIT_AS)[1]<limits['AS_bytes']:
        raise ValueError('PARENT_HARD_AS_INHERITANCE')
    if output.exists() or output.is_symlink():raise ValueError('OUTPUT_OR_ONE_SHOT_USED')
    from ..h6_server_quota_v1 import quota_readonly
    fs=os.statvfs(output.parent)
    if fs.f_bavail*fs.f_frsize<limits['minimum_free_bytes'] or fs.f_favail<1000:raise ValueError('FRESH_FILESYSTEM')
    quota=quota_readonly(output.parent)
    if quota['status']!='KNOWN':raise ValueError('FRESH_QUOTA_UNKNOWN')
    for item in quota['items']:
        if item['status']=='ACTIVE':
            if item['available_bytes'] is not None and item['available_bytes']<limits['output_bytes']:raise ValueError('FRESH_QUOTA_BYTES')
            if item['available_inodes'] is not None and item['available_inodes']<1000:raise ValueError('FRESH_QUOTA_INODES')
    def cpu_ticks():
        return {int(line.split()[0][3:]):[int(v) for v in line.split()[1:9]]
            for line in Path('/proc/stat').read_text().splitlines() if line.split()[0].startswith('cpu') and line.split()[0][3:].isdigit()}
    def physical(cpu):
        p=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        return ((p/'physical_package_id').read_text().strip(),(p/'core_id').read_text().strip())
    assigned=cpus+[manifest['coordinator_cpu']]
    if len({physical(v) for v in assigned})!=len(assigned):raise ValueError('CPU_PHYSICAL_SIBLING')
    before=cpu_ticks();time.sleep(1);after=cpu_ticks()
    for cpu in assigned:
        delta=[b-a for a,b in zip(before[cpu],after[cpu])];total=sum(delta)
        if total<=0 or 1-(delta[3]+delta[4])/total>.1:raise ValueError('FRESH_CPU_BUSY')
    baseline=host_memory()
    if baseline['effective_available_bytes']<limits['aggregate_RSS_bytes']+limits['coordinator_RSS_bytes']+limits['headroom_bytes']:raise ValueError('FRESH_MEMORY')
    if any(v>=(1. if k=='host' else 1e-300) for k,v in baseline['full_psi_avg10'].items()):raise ValueError('FRESH_PRESSURE')
    output.mkdir(exist_ok=False)  # Exclusive one-shot claim before any science.
    os.sched_setaffinity(0,{manifest['coordinator_cpu']})
    # Keep the inherited hard ceiling available for child AS32GiB. Lowering the
    # parent's hard limit to its own 8GiB would prevent workers raising theirs.
    _,hard=resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS,(limits['coordinator_AS_bytes'],hard))
    write_json(output/'authorization.json',grant);write_json(output/'manifest.json',manifest)
    terminal=dict(status='STOP',first_stop=None,mandatory_stop=True,SOURCE=manifest['SOURCE'],
        coordinator_identity=None,all_workers_reaped=False)
    try:
        observe=runtime_observer(output,limits,time.monotonic()+limits['wall_seconds'])
        resume=None
        if manifest.get('resume'):
            from .resume import load_resume
            resume=load_resume(root,manifest)
            saved=resume['saved']
            write_json(output/'reuse_receipt.json',dict(seed_source=resume['seed_source'],input_digest=resume['input_digest'],
                accepted_references=resume['accepted_references'],remaining_tasks=resume['remaining_tasks'],
                cumulative_caps=resume['cumulative_caps'],old_partial_actual=None,reset=False,refund=False))
        else:
            initial=output/'input';initial.mkdir()
            demand=call_limits(100)
            write_json(output/'input_reservation.json',dict(reserved=demand,actual=None,refunded=False))
            summary=supervise(initial,[dict(stage='input',SOURCE=manifest['SOURCE'])],input_worker,cpus[:1],limits,baseline,observe=observe)
            saved=summary['completed'][0]
        snapshot=saved['snapshot'];rank=saved['DF']['actual_rank']
        if file_hash(snapshot['path'])!=snapshot['sha256']:raise ValueError('SNAPSHOT_BYTES')
        from .science import preparations,cost_group
        from .input import array_identity
        from ..h5_open_shell.sector import sector_for
        from trotterlib.df_hamiltonian import DFHamiltonian
        import numpy as np
        with np.load(snapshot['path'],allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
        if {k:array_identity(v) for k,v in data.items()}!=snapshot['arrays']:raise ValueError('SNAPSHOT_ARRAY_IDENTITY')
        ham=DFHamiltonian(float(data['constant']),data['one_body'],data['lambdas'],tuple(data['g_matrices']),{})
        sector=sector_for(7,4,3)
        if not np.array_equal(data['sector_basis'],sector.basis_indices):raise ValueError('SNAPSHOT_SECTOR')
        prepared=preparations(ham)  # Once in idle coordinator; forked read-only reuse.
        seed_source=resume['seed_source'] if resume else manifest['SOURCE']
        binding=resume['input_digest'] if resume else digest(dict(snapshot=snapshot['sha256'],projection=saved['DF'],source=manifest['SOURCE']))
        from .contract import registry
        write_json(output/'realized_input_binding.json',dict(SOURCE=manifest['SOURCE'],
            snapshot=snapshot,projection_digest=digest(saved['DF']),actual_rank=rank,
            registry=registry(rank),call_caps=call_limits(rank),input_digest=binding,
            compiler_hash=manifest['compiler_hash'],profile_digest=manifest['profile_digest'],
            carry_digest=manifest['carry_digest'],authorization_digest=digest(grant),output=str(output),
            seed_source=seed_source,initial_tasks=tasks(rank,seed_source,binding),technical_input_checks='PASS_ENGINEERING',
            scientific_certificate=False))
        def cost(directory,task):
            caps=dict.fromkeys(call_limits(rank),0);caps.update(compile=2,trajectory=1 if task['cell']['R'] else 0,
                sampling_call=1 if task['cell']['R'] else 0,occurrence=task['cell']['R'] or 0,control_probe=4 if task['validate'] else 0)
            expected=next(r['signals']['corrected'] for r in saved['signal_records'] if r['cell']['id']==task['cell']['id'])
            calls=Calls(caps)
            def record(key,used):
                with (directory/'call_reservations.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(before_operation=key,used=used))+'\n');stream.flush()
            calls.recorder=record
            return cost_group(ham,sector,data['full'],data['state'],prepared,task,calls,expected_signal=expected)
        accepted=list(resume['accepted']) if resume else []
        reused_count=len(accepted)
        eligible=[r['cell']['id'] for r in saved['signal_records'] if r['cell']['R'] and
            any(v['1.0']['cost_acquisition_eligible'] for v in r['precision_rows'].values())]
        if resume:
            group=resume['remaining_tasks'];directory=output/'resume_remaining';directory.mkdir()
            write_json(directory/'reservations.json',dict(tasks=group,compile=2*len(group),refund=False))
            accepted+=supervise(directory,group,cost,cpus,limits,baseline,observe=observe)['completed']
        else:
            for stage,registered in [('initial',tasks(rank,seed_source,binding))]:
                # The first six groups finish correctness before extension selection.
                heavy=[t for t in registered if t['heavy']];normal=[t for t in registered if not t['heavy']]
                for label,group,assigned in ((stage+'_heavy',heavy,cpus[:1]),(stage+'_normal',normal,cpus)):
                    directory=output/label;directory.mkdir();write_json(directory/'reservations.json',dict(tasks=group,compile=2*len(group),refund=False))
                    accepted+=supervise(directory,group,cost,assigned,limits,baseline,observe=observe)['completed']
            group=tasks(rank,seed_source,binding,extension=True,eligible=eligible)
            directory=output/'extension';directory.mkdir();write_json(directory/'reservations.json',dict(tasks=group,compile=2*len(group),refund=False))
            accepted+=supervise(directory,group,cost,cpus,limits,baseline,observe=observe)['completed']
        used=dict(saved['calls_attempted'])
        for result in accepted:
            if len(result['rows'])!=2 or {r['axis'] for r in result['rows']}!={'cosine','sine'}:raise ValueError('PAIRED_METRICS')
            for k,v in result['calls_attempted'].items():used[k]+=v
        if any(used[k]>call_limits(rank)[k] for k in used):raise ValueError('FINAL_CALL_ACCOUNTING')
        new_used=dict.fromkeys(used,0)
        for result in accepted[reused_count:]:
            for k,v in result['calls_attempted'].items():new_used[k]+=v
        if resume:
            if new_used['compile']!=2*len(resume['remaining_tasks']):raise ValueError('RESUME_NEW_CALL_ACCOUNTING')
            upper={k:used[k]+sum((2 if k in ('compile','occurrence') else 1) for _ in resume['failed_dispatched'])
                for k in ('compile','trajectory','sampling_call','occurrence')}
            if any(upper[k]>resume['cumulative_caps'][k] for k in upper):raise ValueError('CUMULATIVE_RETRY_CAP')
            write_json(output/'call_totals.json',dict(new_actual=new_used,logical_completed=used,
                cumulative_completed_plus_partial_upper=upper,prior_partial_actual=None,
                cumulative_caps=resume['cumulative_caps'],carry_digest=manifest['carry_digest'],reset=False,refund=False))
        else:
            write_json(output/'call_totals.json',dict(new_actual=used,old_carry_digest=manifest['carry_digest'],
                reset=False,refund=False,reservations_are_not_actual=True))
        verify_assets(root,manifest)
        terminal.update(status='COMPLETE_MANDATORY_STOP',groups=len(accepted),compiled=2*len(accepted),
            reused_groups=reused_count,new_groups=len(accepted)-reused_count,seed_source=seed_source,
            skipped_extension_PR=[r['cell']['id'] for r in saved['signal_records'] if r['cell']['R'] and r['cell']['id'] not in eligible])
    except BaseException as e:terminal['first_stop']=repr(e);raise
    finally:
        summaries=[json.loads(p.read_bytes()) for p in sorted(output.glob('*/summary.json'))]
        terminal['cleanup_errors']=[e for s in summaries for e in s['cleanup_errors']]
        terminal['retired']=[r for s in summaries for r in s['retired']]
        terminal['all_workers_reaped']=bool(summaries) and all(r['reaped'] for r in terminal['retired']) and not terminal['cleanup_errors']
        from ..h6_server_resources_v1 import process_identity
        terminal['coordinator_identity']=process_identity(os.getpid())
        write_json(output/'terminal.json',terminal)
