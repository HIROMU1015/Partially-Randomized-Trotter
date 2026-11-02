"""Independent coordinator, weighted slots, bounded sealed journals, one shot."""
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from .ax2a_preparation import digest
from .ax2b_h4_contract_v5 import file_hash
from .ax2b_supplement_records_v1 import AtomicWriter
from .h6_matched_execution_v2 import Progress, worker_environment, install_limits
from .h6_matched_execution_v3 import ResourceTraceV3
from .h6_server_resources_v1 import (process_identity, process_memory, host_memory,
    memory_reason, stop_owned_group, install_parent_guard, fresh_resources, THREAD_KEYS)
from .h6_parallel_execution_v1 import owned_exit_flag, retain_exiting_memory
from .h8_resource_map_contract_v1 import (RUNNER, CPUS, verify, plan, resources,
    observation, signal_groups, cost_tasks, selection, admission, checkpoint,
    task_seed, coverage, cells)


def read(path): return json.loads(Path(path).read_text())


class RotatingJournal:
    def __init__(self, directory, *, segment_bytes=8*2**20, total_bytes=2**30):
        self.directory=Path(directory);self.directory.mkdir(exist_ok=False)
        self.segment_cap=segment_bytes;self.total_cap=total_bytes
        self.count=0;self.total=0;self.size=0;self.stream=None;self.hash=None;self.sealed=[]
    def rotate(self):
        if self.stream is not None:
            self.stream.flush();os.fsync(self.stream.fileno());self.stream.close()
            self.sealed.append(dict(segment=self.count-1,bytes=self.size,sha256=self.hash.hexdigest()))
        self.stream=(self.directory/f'segment_{self.count:05d}.jsonl').open('xb')
        self.count+=1;self.size=0;self.hash=hashlib.sha256()
    def write(self,row):
        data=(json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
        if len(data)>self.segment_cap or self.total+len(data)>self.total_cap:
            raise RuntimeError('JOURNAL_TOTAL_CAP')
        if self.stream is None or self.size+len(data)>self.segment_cap:self.rotate()
        self.stream.write(data);self.stream.flush();self.hash.update(data)
        self.size+=len(data);self.total+=len(data)
    def close(self):
        if self.stream is not None:
            self.stream.flush();os.fsync(self.stream.fileno());self.stream.close();self.stream=None
            self.sealed.append(dict(segment=self.count-1,bytes=self.size,sha256=self.hash.hexdigest()))
        (self.directory/'sealed_segments.json').write_text(json.dumps(self.sealed,sort_keys=True)+'\n')


def rss_reservation(task):
    # No unmeasured p14/p21, r/K, B0, or rare circuit is promoted automatically.
    known={'B1_p28_q1_2nd','B1_p28_q1_4th','B1_p28_q8_2nd'}
    return (9 if task.get('kind')=='COST' and task['cell_id'] in known else 24)*2**30


def zero_order_measured_class(task, trajectory):
    known={'B2_p7_q1_2nd_r4_K2','B2_p7_q2_2nd_r4_K2',
        'B2_p7_q4_2nd_r4_K4','B2_p7_q8_2nd_r2_K2'}
    if task.get('kind')!='COST' or task['cell_id'] not in known:return False
    orders=trajectory.get('observed_orders')
    return (trajectory.get('task')==task and trajectory.get('event_digest')==digest(trajectory.get('events'))
        and isinstance(orders,list) and len(orders)==task['cell']['R']
        and all(type(x) is int and x==0 for x in orders))


def demand(task):
    if task['kind']=='SIGNAL':
        cs=[c for c in cells() if c['id'] in task['cell_ids']]
        primitive=coverage(cs)['primitive_actions']
        exponential=primitive+3+9+sum(c['R'] for c in cs if c['R'])
        return dict(compile=0,trajectory=0,occurrence=0,control_probe=0,
            signal=len(cs),normalization=sum(bool(c['R']) for c in cs),primitive=primitive,
            reference_matvec=exponential*100000+16+2*sum(c['R']*(c['K']+1) for c in cs if c['R']))
    return dict(compile=4 if task['ordinary'] else 2,trajectory=int(bool(task['cell']['R'])),
        occurrence=task['cell']['q'] if task['cell']['R'] else 0,
        control_probe=(8 if task['ordinary'] else 4) if task['validate'] else 0,
        primitive=0,reference_matvec=0,signal=0,normalization=0)


def inventory(output):
    count=0;size=0;logs=0
    for base,dirs,files in os.walk(output):
        # Runtime caches count towards bytes/inodes too; no shared scans.
        count+=len(dirs)+len(files)
        for name in files:
            p=Path(base)/name
            if p.is_symlink():raise ValueError('OUTPUT_SYMLINK')
            n=p.stat().st_size;size+=n
            if name.endswith('.log'):logs+=n
    return dict(inodes=count,bytes=size,log_bytes=logs)


def supervisor(root,run,phase,registrations,m,g,ledger):
    out=Path(run)/phase;out.mkdir(exist_ok=False);w=AtomicWriter(out,byte_cap=32*2**30)
    w.write('registered_tasks.json',dict(phase=phase,registrations=registrations,
        manifest_digest=digest(m),task_digest=digest(registrations)))
    accepted=m.get('resume_parent',{}).get('accepted',{}).get(phase,{})
    expected={str(row['index']):row['task'] for row in registrations}
    if any(i not in expected or record['task']!=expected[i] for i,record in accepted.items()):
        raise ValueError('H8_RESUME_PHASE_REGISTRATIONS')
    done={int(i):record['result'] for i,record in accepted.items()}
    pending=[row for row in registrations if row['index'] not in done]
    active={};retired=[];dispatched=[];first=None
    r=m['proposed_resources'];policy=m['observation_policy']
    baseline=read(Path(run)/'fresh_resources.json')['resources']['memory']
    journal=RotatingJournal(out/'observation_journal',segment_bytes=policy['journal_segment_bytes'],
                            total_bytes=policy['journal_bytes']//3)
    start=time.monotonic();last_sample=start;last_emit=0;last_inventory=start;peak=0;cleanup=[]
    single_heavy=phase=='exploration' and bool(pending) and pending[0]['task']['cell_id']=='B1_p28_q8_4th'
    def retire(slot,v):
        code=stop_owned_group(v['process'],v['identity'])
        receipt=dict(pid=v['process'].pid,start_ticks=v['identity']['start_ticks'],
            exit_code=code,reaped=True,index=v['row']['index'])
        retired.append(receipt)
        AtomicWriter(v['path'],byte_cap=32*2**20).write('native_reap_receipt.json',receipt,terminal=True)
        return code,receipt
    try:
        while pending or active:
            if (Path(run)/'STOP_REQUEST').exists():first=first or 'USER_STOP_REQUEST'
            if first:break
            while pending and len(active)<8:
                used=sum(v['rss_reservation'] for v in active.values())
                fit=next((i for i,row in enumerate(pending)
                    if used+rss_reservation(row['task'])<=72*2**30),None)
                if fit is None or (single_heavy and active):break
                if single_heavy:fit=0
                row=pending.pop(fit);slot=next(i for i in range(8) if i not in active)
                d=out/f'group_{row["index"]:04d}';d.mkdir(exist_ok=False)
                cache=d/'.runtime_cache';cache.mkdir();temp=cache/'tmp';temp.mkdir()
                env=worker_environment(1,cache);env.update(TMPDIR=str(temp),TEMP=str(temp),TMP=str(temp),
                    XDG_CACHE_HOME=str(cache/'xdg'))
                reservation=demand(row['task'])
                for k,v in reservation.items():
                    if ledger['known'].get(k,0)+ledger['reserved'].get(k,0)+v>m['call_budget'][k]:
                        raise ValueError('CORE_ADMISSION_CALL_CAP:'+k)
                for k,v in reservation.items():ledger['reserved'][k]+=v
                binding=dict(phase=phase,index=row['index'],task=row['task'],slot=slot,
                    cpus=r['cost_cpu_sets'][slot],rss_stop_bytes=rss_reservation(row['task']),
                    manifest_digest=digest(m),registration_digest=digest(registrations),reserved_calls=reservation)
                AtomicWriter(d,byte_cap=32*2**20).write('task_binding.json',binding)
                w.write(f'dispatch_{len(dispatched):04d}.json',dict(binding=binding,
                    ledger_before_spawn=deepcopy(ledger)))
                argv=[sys.executable,'-I','-P','-B',str(Path(root)/RUNNER),
                    '--worker',str(d),'--run-root',str(run)]
                with (d/'worker.log').open('xb') as log:
                    child=subprocess.Popen(argv,env=env,cwd=root,stdin=subprocess.DEVNULL,
                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                        preexec_fn=lambda:os.sched_setaffinity(0,set(r['assigned_cpus'])))
                v=dict(process=child,identity=None,path=d,row=row,rss_reservation=binding['rss_stop_bytes'],reservation=reservation)
                active[slot]=v;v['identity']=process_identity(child.pid)
                dispatched.append(dict(index=row['index'],pid=child.pid,identity=v['identity'],slot=slot))
                if single_heavy:break
            before=time.monotonic();samples=[]
            for slot,v in list(active.items()):
                child=v['process'];task=v['row']['task']
                if os.waitid(os.P_PID,child.pid,os.WEXITED|os.WNOHANG|os.WNOWAIT) is not None:
                    code,reap=retire(slot,v);del active[slot]
                    if code!=0:
                        first=first or 'WORKER_EXIT:'+str(code);continue
                    if task['kind']=='SIGNAL':
                        terminal=read(v['path']/'worker_terminal.json')
                        result=read(v['path']/'signal_result.json')
                        signals=read(v['path']/'signal_summary.json')['signals']
                        if (terminal['status']!='COMPLETE' or terminal['source_commit']!=m['source_commit']
                            or terminal['reason'] is not None or {x['cell']['id'] for x in signals}!=set(task['cell_ids'])):
                            raise ValueError('SIGNAL_COMPLETE_ADMISSION')
                        actual=dict(result['calls_attempted'],signal=result['signal_count'],
                            normalization=result['normalization_evaluations'])
                        result['signals']=signals
                    else:
                        result=admission(v['path'],task,m['source_commit'],reaped=True)
                        if task['cell']['method']=='B1' and 'feasibility_binding' in m:
                            old=read(Path(root)/m['feasibility_binding']['path'])['rows']
                            checks=[]
                            for row in result['rows']:
                                match=next((x for x in old if (x['cell_id'],x['axis'],x['control'])==
                                    (row['cell_id'],row['axis'],row['control'])),None)
                                if match is not None:
                                    checks.append(dict(axis=row['axis'],control=row['control'],
                                        canonical_equal=row['fingerprint']==match['fingerprint'],
                                        metrics_equal=row['metrics']==match['metrics']))
                                    if not checks[-1]['canonical_equal'] or not checks[-1]['metrics_equal']:
                                        raise ValueError('H8_SAVED_DETERMINISTIC_CANONICAL_METRIC_MISMATCH')
                            AtomicWriter(v['path'],byte_cap=32*2**20).write('saved_deterministic_continuity.json',
                                dict(checks=checks,quantum_cost_reused=False,compile_calls_refunded=False))
                        actual=dict(result['calls_attempted'],signal=0,normalization=0)
                        cp=checkpoint(v['path'],task,m['source_commit'],reap)
                        w.write(f'checkpoint_{v["row"]["index"]:04d}.json',cp)
                    if any(actual.get(k,0)>cap for k,cap in v['reservation'].items()):
                        raise ValueError('ACTUAL_EXCEEDS_TASK_RESERVATION')
                    for k,cap in v['reservation'].items():
                        ledger['reserved'][k]-=cap;ledger['known'][k]+=actual.get(k,0)
                    done[v['row']['index']]=result
                    w.write(f'complete_{v["row"]["index"]:04d}.json',dict(task_digest=digest(task),
                        result_digest=digest(result),native_reap=reap,calls=actual,ledger=deepcopy(ledger)))
                    if single_heavy:single_heavy=False
                    continue
                current=process_identity(child.pid)
                if current['start_ticks']!=v['identity']['start_ticks']:raise ValueError('WORKER_OWNERSHIP')
                tr=v['path']/'trajectory.json'
                if v['rss_reservation']==24*2**30 and tr.is_file() and zero_order_measured_class(task,read(tr)):
                    v['rss_reservation']=9*2**30
                    w.write(f'normal_promotion_{v["row"]["index"]:04d}.json',dict(
                        task_digest=digest(task),trajectory_sha256=file_hash(tr),
                        rss_stop_bytes=9*2**30,policy='observed p7 cell and actual zero-order events only',
                        new_source_peak_not_measured=True,retained_copy_margin='25 percent engineering headroom'))
                try:
                    memory=process_memory(child.pid);v['last_memory']=memory
                except (FileNotFoundError,ValueError):
                    if not owned_exit_flag(child,v['identity']):raise
                    limits=dict(r,worker_rss_stop_bytes=v['rss_reservation'])
                    memory,_=retain_exiting_memory(v,limits,time.monotonic())
                if memory['VmSize_bytes']>r['cost_address_space_bytes']:first=first or 'WORKER_AS_CAP'
                if memory['VmRSS_bytes']>v['rss_reservation']:first=first or 'WORKER_RSS_CAP'
                ds=sorted(v['path'].glob('diagnostic_*.json'));ps=sorted(v['path'].glob('progress_*.json'))
                context=read(ds[-1])['context'] if ds else read(ps[-1]) if ps else {'stage':'startup'}
                samples.append(dict(pid=child.pid,index=v['row']['index'],slot=slot,memory=memory,phase=context))
            host=host_memory();co=process_memory(os.getpid());after=time.monotonic()
            rss=co['VmRSS_bytes']+sum(x['memory']['VmRSS_bytes'] for x in samples);peak=max(peak,rss)
            first=first or memory_reason(host,baseline,policy)
            if after-before>5 or before-last_sample>5:first=first or 'OBSERVATION_STALE'
            if rss>r['aggregate_rss_bytes']:first=first or 'AGGREGATE_RSS_CAP'
            if co['VmRSS_bytes']>r['coordinator_rss_reserve_bytes']:first=first or 'COORDINATOR_RSS_CAP'
            stale=before-last_sample;last_sample=after
            if after-last_inventory>5:
                inv=inventory(run);last_inventory=after
                if inv['bytes']>32*2**30-8*32*2**20:first=first or 'OUTPUT_BYTES_CAP'
                if inv['inodes']>policy['output_inode_cap']-1000:first=first or 'OUTPUT_INODE_CAP'
                if inv['log_bytes']>128*2**20:first=first or 'OUTPUT_LOG_CAP'
                fs=os.statvfs(run)
                if fs.f_bavail*fs.f_frsize<10*2**30 or fs.f_favail<100000:first=first or 'FILESYSTEM_HEADROOM'
            if after-last_emit>=5 or first:
                journal.write(dict(wall_seconds=after-start,workers=samples,coordinator=co,
                    memory=host,aggregate_rss_bytes=rss,observation_seconds=after-before,
                    staleness_seconds=stale,first_stop_reason=first,pending=len(pending),active_slots=len(active)))
                last_emit=after
            if active:time.sleep(.2)
    except BaseException as exc:
        first=first or type(exc).__name__+':'+str(exc)[:512]
    finally:
        for slot,v in active.items():
            try:retire(slot,v)
            except Exception as exc:cleanup.append(type(exc).__name__+':'+str(exc)[:256])
        journal.close()
    terminal=dict(phase=phase,first_stop_reason=first,cleanup_errors=cleanup,
        dispatched=dispatched,retired=retired,all_dispatched_reaped=len(retired)==len(dispatched) and not cleanup,
        completed_indices=sorted(done),pending=len(pending),aggregate_peak_rss_bytes=peak,
        wall_seconds=time.monotonic()-start,ledger=deepcopy(ledger),automatic_retry=False,
        reused_complete_indices=sorted(int(i) for i in accepted))
    w.write('execution_terminal.json',terminal,terminal=True)
    if first or cleanup or len(done)!=len(registrations):raise RuntimeError(first or 'PHASE_INCOMPLETE_OR_CLEANUP')
    return done,terminal


def worker(root,directory,run):
    directory=Path(directory);run=Path(run);m=read(run/'manifest.json');g=read(run/'grant.json')
    verify(root,m,g,committed=False)
    b=read(directory/'task_binding.json');task=b['task'];r=m['proposed_resources']
    registrations=read(directory.parent/'registered_tasks.json')
    if (b['manifest_digest']!=digest(m) or b['registration_digest']!=digest(registrations['registrations'])
        or dict(index=b['index'],task=task) not in registrations['registrations']
        or b['cpus']!=r['cost_cpu_sets'][b['slot']] or b['rss_stop_bytes']!=rss_reservation(task)
        or directory.parent.parent!=run or directory.name!=f'group_{b["index"]:04d}'):
        raise ValueError('CORE_WORKER_BINDING')
    if task['kind']=='COST' and task['seed']!=task_seed(m['source_commit'],task['cell_id'],task['phase'],task['replica']):
        raise ValueError('CORE_WORKER_SEED')
    if any(os.environ.get(k)!='1' for k in THREAD_KEYS) or os.environ.get('QISKIT_PARALLEL')!='FALSE':
        raise ValueError('CORE_WORKER_THREADS')
    install_parent_guard(read(run/'coordinator_identity.json'))
    limits=install_limits(b['cpus'],32*2**30,32*2**20)
    w=AtomicWriter(directory,byte_cap=32*2**20,diagnostics_cap=1000)
    w.write('worker_claim.json',dict(source_commit=m['source_commit'],task_digest=digest(task),
        identity=process_identity(os.getpid()),resource_limits=limits,rss_stop_bytes=b['rss_stop_bytes']))
    progress=Progress(w,cap=1000);trace=ResourceTraceV3(w.write,lambda:time.monotonic()-progress.start)
    reason=None
    try:
        with trace.bind():
            if task['kind']=='SIGNAL':
                from .h8_resource_map_port_v1 import signal_task
                signal_task(root,m,task,w,progress)
            else:
                from .h8_resource_map_optimization_v1 import cost_port
                cost_port(root,m,task,w,progress,trace)
        verify(root,m,g,committed=False)
    except Exception as exc:
        reason=type(exc).__name__+':'+str(exc)[:512]
        try:trace.failure(exc)
        except Exception:pass
    w.write('worker_terminal.json',dict(status='COMPLETE' if reason is None else 'STOP',
        reason=reason,source_commit=m['source_commit'],task=task,memory=process_memory(os.getpid()),
        mandatory_stop=True,automatic_retry=False),terminal=True)
    return 0 if reason is None else 1


def fresh_gate(root,m):
    from .h48_preflight_contract_v1 import verify_candidate_runtime
    native=verify_candidate_runtime(root)
    fresh=fresh_resources(m['proposed_resources'],m['observation_policy'],Path(m['output']))
    policy=m['observation_policy']
    if fresh['available_bytes']<policy['minimum_storage_bytes'] or fresh['available_inodes']<policy['minimum_free_inodes']:
        raise ValueError('CORE_STORAGE_INODE_ADMISSION')
    for q in fresh['quota']['items']:
        if q['status']=='ACTIVE':
            if q['available_bytes'] is not None and q['available_bytes']<policy['minimum_storage_bytes']:raise ValueError('CORE_QUOTA_BYTES')
            if q['available_inodes'] is not None and q['available_inodes']<policy['minimum_free_inodes']:raise ValueError('CORE_QUOTA_INODES')
    return dict(native=native,resources=fresh)


def orchestrate(root,m,g):
    verify(root,m,g)
    run=Path(m['output']);lock=run.parent/(run.name+'.one_shot_lock')
    if run.exists() or lock.exists():raise ValueError('CORE_ONE_SHOT_USED')
    fresh=fresh_gate(root,m)
    with lock.open('x') as f:f.write(json.dumps(dict(SOURCE=m['source_commit'],manifest_digest=digest(m)))+'\n')
    run.mkdir(exist_ok=False);w=AtomicWriter(run,byte_cap=32*2**30)
    for n,v in [('manifest.json',m),('grant.json',g),('fresh_resources.json',fresh),
                ('coordinator_identity.json',process_identity(os.getpid()))]:w.write(n,v)
    install_limits([22,23],8*2**30,32*2**30,hard_address_bytes=32*2**30)
    keys=('compile','trajectory','occurrence','control_probe','primitive','reference_matvec','signal','normalization')
    ledger=dict(known={k:0 for k in keys},reserved={k:0 for k in keys},historical_unknown_actual=None,
        preserved_carry=m['cumulative_carry'],failed_calls_refunded=False,budget_reset=False)
    if 'resume_parent' in m:
        ledger=deepcopy(m['resume_parent']['parent_ledger'])
    reason=None
    try:
        groups,report=supervisor(root,run,'signal',signal_groups(),m,g,ledger)
        signals=[s for i in sorted(groups) for s in groups[i]['signals']]
        signals.sort(key=lambda s:next(i for i,c in enumerate(cells()) if c['id']==s['cell']['id']))
        if len(signals)!=92:raise ValueError('CORE_SIGNAL_COVERAGE')
        w.write('signal_summary.json',dict(signals=signals,calls=deepcopy(ledger),empirical_allowance=True))
        registrations=cost_tasks(signals,m['source_commit'],'exploration')
        # The sole measured heavy baseline is completed in isolation first.
        registrations.sort(key=lambda x:x['task']['cell_id']!='B1_p28_q8_4th')
        groups,report=supervisor(root,run,'exploration',registrations,m,g,ledger)
        exploration=[row for i in sorted(groups) for row in groups[i]['rows']]
        resource_maps=full_resource_maps
        w.write('exploration_resource_map.json',resource_maps(signals,exploration,m['plan'],'exploration'))
        selected=selection(signals,exploration,m['source_commit'])
        w.write('selection_before_confirmation.json',selected)
        groups,report=supervisor(root,run,'confirmation',selected['registrations'],m,g,ledger)
        confirmation=[row for i in sorted(groups) for row in groups[i]['rows']]
        deterministic=[r for r in exploration if r['cell_id'].startswith(('B0_','B1_'))]
        w.write('confirmation_resource_map.json',resource_maps(signals,deterministic+confirmation,m['plan'],'confirmation'))
        w.write('coverage.json',dict(signal_candidates=92,exploration_groups=len(registrations),
            confirmation_groups=len(groups),selected_ids=selected['selected_ids'],
            missing_cost_ids=[s['cell']['id'] for s in signals if not any(x['cell_id']==s['cell']['id'] for x in exploration)],
            status_policy='ineligible versus numeric unresolved versus missing versus resource STOP remain distinct',
            exploration_confirmation_pooled=False,formal_winner_certified=False,tail_bound=None))
        status_rows=[]
        for s in signals:
            for epsilon in m['plan']['epsilons']:
                accounting=s['precision_rows'][str(epsilon)]['1.0']
                eligibility=accounting['eligibility_under_declared_allowance']
                status=('ACQUIRED' if accounting['cost_acquisition_eligible'] and any(
                    x['cell_id']==s['cell']['id'] for x in exploration) else
                    'MISSING_COST' if accounting['cost_acquisition_eligible'] else
                    'NUMERICALLY_UNRESOLVED' if eligibility=='ELIGIBLE' else 'INELIGIBLE')
                status_rows.append(dict(cell_id=s['cell']['id'],epsilon=epsilon,status=status,
                    bias=s['error_decomposition']['total_signed'],u=s['allowance'],
                    headroom={a:accounting['axes'][a].get('headroom') for a in ('real','imag')},
                    missing_cost_is_zero=False))
        w.write('candidate_status_by_precision.json',dict(rows=status_rows,
            signal_precision='finite-time complex signal; not chemical accuracy/QPE energy'))
        verify(root,m,g)
    except Exception as exc:reason=type(exc).__name__+':'+str(exc)[:512]
    w.write('execution_terminal.json',dict(status='H8_CORE_COMPLETE_MANDATORY_STOP' if reason is None else 'H8_CORE_STOP',
        reason=reason,SOURCE=m['source_commit'],ledger=ledger,mandatory_stop=True,
        automatic_retry=False,next_stage_authorized=False),terminal=True)
    return 0 if reason is None else 1


from .h6_matched_accounting_v1 import resource_row
def full_resource_maps(signals, rows, plan, phase):
    result=[]
    for epsilon in plan['epsilons']:
        for s in signals:
            cr=[c for c in rows if c['cell_id']==s['cell']['id']]
            for metric in ('RZ','CX','RZ_depth','CX_depth','depth','size'):
                for factor in ('1.0','10.0'):
                    row=resource_row(s,cr,epsilon,metric=metric,factor=factor)
                    if row is not None:
                        row.update(phase=phase,allowance_factor=factor);result.append(row)
    primary=[r for r in result if r['metric']=='RZ' and r['allowance_factor']=='1.0']
    crossings=[]
    for a in primary:
        if a['method']!='B2':continue
        for b in primary:
            if b['method']=='B2' or a['epsilon']!=b['epsilon']:continue
            dn=a['N_total']-b['N_total']
            p=(b['G_point']-a['G_point'])/dn if dn else None
            crossings.append(dict(B2=a['cell_id'],baseline=b['cell_id'],epsilon=a['epsilon'],
                                  crossing_P=p if p is not None and p>=0 else None,
                                  linear_difference_intercept=a['G_point']-b['G_point'],linear_difference_slope=dn))
    comparisons=[]
    for epsilon in plan['epsilons']:
        best={}
        for method in ('B0','B1','B2'):
            candidates=[r for r in primary if r['epsilon']==epsilon and r['method']==method]
            best[method]=min(candidates,key=lambda r:(r['G_point'],r['cell_id'])) if candidates else None
        b2=best['B2']
        for method in ('B0','B1'):
            baseline=best[method]
            if b2 and baseline:
                se=b2['G_standard_error']
                comparisons.append(dict(epsilon=epsilon,B2=b2['cell_id'],baseline=baseline['cell_id'],
                    ratio_point=b2['G_point']/baseline['G_point'] if baseline['G_point'] else None,
                    descriptive_two_SE_separated=(abs(b2['G_point']-baseline['G_point'])>2*se if se is not None else None),
                    winner='UNDETERMINED_POPULATION_MEAN',formal_winner_certified=False))
    return dict(rows=result,common_preparation_crossings=crossings,comparisons=comparisons,formal_winner_certified=False,
        mean_scope='conditional point estimates; SD/SE descriptive, no tail or simultaneous guarantee')
