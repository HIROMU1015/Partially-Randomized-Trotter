"""Operational phase driver: immutable D0 API, global Phase A before B/C.

No estimator/shot/cost/selection/P implementation lives here. Those functions
are loaded unchanged from fa75c799…; this driver supplies authorization,
resource guards, the user-requested phase order, and evidence persistence.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time

HERE=Path(__file__).resolve().parent
PATHS=json.loads((HERE/'execution_paths.json').read_text())
ROOT=Path(PATHS['worktree'])
CONTRACT=Path(PATHS['contract'])
OUT=HERE/'output'
AUTH=HERE/'authorization_v1.json'
started=time.monotonic()
cpu_start=resource.getrusage(resource.RUSAGE_SELF)
spec=importlib.util.spec_from_file_location('fixed_d0',ROOT/'scripts/d0_saved_value_reanalysis_v1.py')
d0=importlib.util.module_from_spec(spec);spec.loader.exec_module(d0)
cbytes=CONTRACT.read_bytes();contract=d0.strict_json(cbytes);budget=contract['budget']
auth=d0.strict_json(AUTH.read_bytes())
resource.setrlimit(resource.RLIMIT_AS,(budget['address_space_bytes'],budget['address_space_bytes']))
resource.setrlimit(resource.RLIMIT_CPU,(budget['cpu_seconds'],budget['cpu_seconds']))
def wall_stop(*_):raise ValueError('D0_WALL_CAP')
signal.signal(signal.SIGALRM,wall_stop)
signal.setitimer(signal.ITIMER_REAL,max(.001,budget['wall_seconds']-(time.monotonic()-started)))
first_stop=None;phase='PRELAUNCH';context={};last_guard=time.monotonic();max_guard_interval=0.;observations=0
byte_count=0;diagnostics={};validation={};recordsets={};loaded={}
affinity=sorted(os.sched_getaffinity(0))

def write_output(name,obj):
    global byte_count
    with (OUT/name).open('x') as stream:
        for chunk in json.JSONEncoder(indent=2,ensure_ascii=False,allow_nan=False).iterencode(obj):
            byte_count+=len(chunk.encode())
            if byte_count>budget['output_bytes']:raise ValueError('D0_OUTPUT_CAP')
            stream.write(chunk)
            if time.monotonic()-last_guard>.5:guard()
        stream.write('\n');byte_count+=1

def guard():
    global last_guard,max_guard_interval,observations
    now=time.monotonic();gap=now-last_guard;max_guard_interval=max(max_guard_interval,gap);last_guard=now
    if gap>5.:raise ValueError('D0_OBSERVATION_STALE')
    if now-started>budget['wall_seconds']:raise ValueError('D0_WALL_CAP')
    usage=resource.getrusage(resource.RUSAGE_SELF)
    if usage.ru_maxrss*1024>budget['rss_bytes']:raise ValueError('D0_RSS_CAP')
    if usage.ru_utime+usage.ru_stime>budget['cpu_seconds']:raise ValueError('D0_CPU_CAP')
    if sorted(os.sched_getaffinity(0))!=affinity or len(list(Path('/proc/self/task').iterdir()))!=1:
        raise ValueError('AFFINITY_OR_THREAD_COUNT_CHANGED')
    d0.check_memory_snapshot(d0.memory_snapshot(),budget,admission);observations+=1

def differences(a,b,path=''):
    if type(a) is dict and type(b) is dict:
        return [d for k in sorted(set(a)|set(b)) for d in differences(a.get(k),b.get(k),path+'/'+k)]
    if a!=b:return [dict(field=path,saved=a,reproduced=b)]
    return []

try:
    d0.execution_gate(contract,auth,contract_sha=hashlib.sha256(cbytes).hexdigest(),output=OUT)
    if auth['review_commit']!='b941565835c539fc414d3a75dcb8ca439561a4d5':raise ValueError('REVIEW_BINDING')
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=auth['operational_driver_sha256']:
        raise ValueError('OPERATIONAL_DRIVER_BINDING')
    for p,h in contract['analysis_source_hashes'].items():
        if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h:raise ValueError('ANALYSIS_SOURCE_BINDING')
    profile_bytes=(ROOT/auth['profile_path']).read_bytes()
    if hashlib.sha256(profile_bytes).hexdigest()!=auth['runtime_profile_sha256']:raise ValueError('PROFILE_FILE_BINDING')
    profile=d0.strict_json(profile_bytes)
    if sys.version!=profile['python'] or sys.executable!=profile['executable']:raise ValueError('RUNTIME_PROFILE_BINDING')
    for p,h in profile['runtime_file_hashes'].items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h:raise ValueError('RUNTIME_FILE_BINDING')
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        if os.environ.get(name)!='1':raise ValueError('THREAD_ENV_REQUIRED')
    if contract['budget']!=auth['resource_limits'] or contract['inputs']!=auth['input_receipts']:
        raise ValueError('AUTH_CONTRACT_BINDING')
    admission=d0.fresh_memory_gate(budget)
    stat=os.statvfs(HERE)
    if stat.f_bavail*stat.f_frsize<budget['fresh_available_bytes'] or stat.f_favail<budget['fresh_available_inodes']:
        raise ValueError('FRESH_CAPACITY_GATE')
    paths=d0.strict_json(Path(PATHS['input_mapping']).read_bytes())
    data={r['alias']:d0.bound_input(paths[r['alias']],r) for r in contract['inputs']}
    lock=OUT.with_name(OUT.name+'.oneshot')
    with lock.open('x') as stream:
        json.dump(dict(contract_sha256=hashlib.sha256(cbytes).hexdigest(),pid=os.getpid(),
            analysis_source=contract['analysis_source'],authorization_sha256=hashlib.sha256(AUTH.read_bytes()).hexdigest()),stream)
        stream.flush();os.fsync(stream.fileno())
    OUT.mkdir()
    phase='A_ORIGINAL_EQUALITY'
    for scope,s in contract['scopes'].items():
        guard();signals=d0.strict_json(data[s['signal_alias']])['signals']
        rows=d0.parse_wrappers(data[s['wrapper_alias']])
        baselines=d0.parse_wrappers(data[s['deterministic_baseline_wrapper_alias']]) if 'deterministic_baseline_wrapper_alias' in s else []
        old=d0.strict_json(data[s['old_map_alias']]);loaded[scope]=(signals,rows,baselines,old)
        # Only published old metrics are calculated in A. No .0005 or P call.
        old_metrics={'RZ'} if scope==d0.SCOPES[0] else {r['metric'] for r in old['rows']}
        records=[];selected=set(s.get('confirmation_ids',[]));compiler,=s['compiler_hashes']
        for sig in signals:
            guard();cell=sig['cell'];cid=cell['id']
            if selected and cell['method']=='B2' and cid not in selected:continue
            rr,pp=(baselines,'exploration') if selected and cell['method']!='B2' else (rows,s['primary_phase'])
            pairs=d0.paired_rows(rr,cell_id=cid,phase=pp,compiler_hash=compiler)
            expected=1 if cell['method']!='B2' else 8 if pp=='exploration' else 32
            if pairs and len(pairs)!=expected:raise ValueError('PAIRED_SAMPLE_COUNT')
            for eps in contract['original_epsilons']:
                for factor in contract['allowance_factors']:
                    context=dict(scope=scope,cell_id=cid,epsilon=eps,factor=factor)
                    a=d0.accounting(sig,eps,factor)
                    saved=sig['precision_rows'][str(eps)][str(factor)]
                    if a!=saved:
                        diagnostics=dict(context=context,differences=differences(saved,a),type='ORIGINAL_PRECISION_ROWS_MISMATCH')
                        raise ValueError('ORIGINAL_PRECISION_ROWS_MISMATCH:'+cid)
                    status='DERIVED_SAVED_SCALAR' if a['cost_acquisition_eligible'] and pairs else (
                        'INELIGIBLE' if a['eligibility_under_declared_allowance']=='INELIGIBLE' else
                        'NUMERICALLY_UNRESOLVED' if not a['cost_acquisition_eligible'] else 'MISSING')
                    metrics={}
                    if status=='DERIVED_SAVED_SCALAR':
                        shots={axis:a['axes'][channel]['shots'] for axis,channel in d0.AXIS_CHANNEL.items()}
                        metrics={m:d0.weighted_metric(pairs,shots,m,point_rule=s['point_rule']) for m in old_metrics}
                    records.append(dict(cell_id=cid,epsilon=eps,factor=str(factor),accounting=a,status=status,metrics=metrics,N_total=a['N_total']))
        try:validation[scope]=d0.verify_original(signals,records,old,h4=scope==d0.SCOPES[0])
        except ValueError as e:
            by={(r['cell_id'],str(r['epsilon']),r['factor']):r for r in records}
            mismatch=[]
            for r in old['rows']:
                row=by.get((r['cell_id'],str(r['epsilon']),r.get('allowance_factor','1.0')))
                m='RZ' if scope==d0.SCOPES[0] else r['metric']
                if row and m in row['metrics']:
                    diff=differences(dict(G=r.get('G_RZ_point',r.get('G_point')),SE=r.get('G_RZ_SE',r.get('G_standard_error')),N=r['N_total']),
                        dict(G=row['metrics'][m]['G_point'],SE=row['metrics'][m]['G_standard_error'],N=row['N_total']))
                    if diff:mismatch.append(dict(cell_id=r['cell_id'],epsilon=r['epsilon'],factor=r.get('allowance_factor','1.0'),metric=m,differences=diff))
            diagnostics=dict(context=dict(scope=scope),mismatches=mismatch[:32],type=str(e));raise
    write_output('original_equality_v1.json',validation)
    phase='B_PRECISION_AND_SIX_METRICS'
    for scope,s in contract['scopes'].items():
        signals,rows,baselines,_=loaded[scope]
        recordsets[scope]=d0.analyze_scope(signals,rows,s,contract['epsilons'],contract['allowance_factors'],baselines,guard)
    write_output('candidate_accounting_v1.json',recordsets)
    selections={};sensitivity={};phase='C_PREPARATION_SENSITIVITY'
    for scope,records in recordsets.items():
        selections[scope]={};sensitivity[scope]={}
        for eps in contract['epsilons']:
            for factor in contract['allowance_factors']:
                key=str(eps)+'/'+str(factor);subset=[r for r in records if r['epsilon']==eps and r['factor']==str(factor)]
                selections[scope][key]={m:d0.family_selection(subset,m) for m in d0.METRICS}
                sensitivity[scope][key]={}
                for m in d0.METRICS:
                    guard();lines=[dict(cell_id=r['cell_id'],slope=r['N_total'],intercept=r['metrics'][m]['G_point'])
                        for r in subset if r['status']=='DERIVED_SAVED_SCALAR']
                    fixed_ids={e['metric_selected_cell'] for e in d0.family_selection(subset,m).values() if e['metric_selected_cell']}
                    fixed=[line for line in lines if line['cell_id'] in fixed_ids]
                    sensitivity[scope][key][m]=dict(units=m+' per shot',lower_envelope=d0.lower_envelope(lines),
                        fixed_pair_scope='metric-selected family representatives at P=0; no reselection for these crossings',
                        fixed_pairs=[dict(a=a['cell_id'],b=b['cell_id'],**d0.fixed_crossing(a,b)) for i,a in enumerate(fixed) for b in fixed[i+1:]])
    write_output('selection_metrics_v1.json',selections)
    write_output('preparation_sensitivity_v1.json',sensitivity)
    guard();usage=resource.getrusage(resource.RUSAGE_SELF)
    provenance=dict(schema='track_a_d0_reanalysis_result_v1',analysis_source=contract['analysis_source'],
        review_commit=auth['review_commit'],contract_sha256=hashlib.sha256(cbytes).hexdigest(),
        authorization_sha256=hashlib.sha256(AUTH.read_bytes()).hexdigest(),operational_driver_sha256=auth['operational_driver_sha256'],
        input_receipts=contract['inputs'],conditions=contract['scopes'],runtime_profile_sha256=auth['runtime_profile_sha256'],
        global_phase_A_gate='ALL_THREE_SCOPES_PASS_BEFORE_B_C',fresh_admission=admission,
        wall_seconds=time.monotonic()-started,cpu_seconds=usage.ru_utime+usage.ru_stime,peak_rss_bytes=usage.ru_maxrss*1024,
        observations=observations,maximum_guard_interval_seconds=max_guard_interval,
        processes=1,threads=1,affinity_changed=False,new_science_calls=0,H8_changes=0,H8_included=False,
        new_independent_evidence=False,formal_winner_certified=False)
    write_output('provenance_v1.json',provenance)
except Exception as error:
    first_stop=type(error).__name__+':'+str(error)
    if OUT.exists():
        if not (OUT/'original_equality_v1.json').exists():write_output('original_equality_v1.json',dict(status='FAIL_CLOSED_STOP',completed_scopes=validation,failure=diagnostics))
    else:
        (HERE/'prelaunch_failure.json').write_text(json.dumps(dict(first_stop_reason=first_stop,phase=phase),indent=2)+'\n')
finally:
    signal.setitimer(signal.ITIMER_REAL,0)
    usage=resource.getrusage(resource.RUSAGE_SELF)
    terminal=dict(status='FAIL_CLOSED_STOP' if first_stop else 'COMPLETE_MANDATORY_STOP',first_stop_reason=first_stop,
        phase=phase,wall_seconds=time.monotonic()-started,cpu_seconds=usage.ru_utime+usage.ru_stime,peak_rss_bytes=usage.ru_maxrss*1024,
        result_bytes=byte_count,observations=observations,maximum_guard_interval_seconds=max_guard_interval,
        automatic_retry=False,next_stage_authorized=False,new_science_calls=0,H8_changes=0)
    if OUT.exists():
        with (OUT/'terminal_v1.json').open('x') as stream:json.dump(terminal,stream,indent=2)
    (HERE/'native_process_receipt_pending_exit.json').write_text(json.dumps(dict(pid=os.getpid(),**terminal),indent=2)+'\n')
    print(json.dumps(terminal))
sys.exit(1 if first_stop else 0)
