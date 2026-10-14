"""Independent saved-scalar accounting. Real execution requires a separate grant.

Only stdlib; no molecular arrays, numerical providers, sampler or compiler.
Importing this module reads only its adjacent frozen rule source. Synthetic
tests call pure functions; real saved inputs are never loaded at import time.
"""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import resource
import signal
import statistics
import subprocess
import sys
import time

_rule_path = Path(__file__).with_name('d0_frozen_shot_rule_v1.py')
_spec = importlib.util.spec_from_file_location('d0_frozen_shot_rule_v1', _rule_path)
_rule = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rule)
METRICS = ('RZ', 'CX', 'RZ_depth', 'CX_depth', 'depth', 'size')
AXIS_CHANNEL = {'cosine': 'real', 'sine': 'imag'}
SCOPES = ('H4_NATIVE_FIXED16', 'H6_EXPLORATION92', 'H6_CONFIRMATION_UNION8')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def strict_json(data):
    def pairs(items):
        result = {}
        for k, v in items:
            if k in result:
                raise ValueError('DUPLICATE_JSON_KEY')
            result[k] = v
        return result
    def bad_constant(value):
        raise ValueError('NONFINITE_JSON:' + value)
    return json.loads(data, object_pairs_hook=pairs, parse_constant=bad_constant)


def finite(value, name, *, minimum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('FINITE_REAL_REQUIRED:' + name)
    try:
        value = float(value)
    except OverflowError as e:
        raise ValueError('FLOAT_OVERFLOW:' + name) from e
    if not math.isfinite(value) or (minimum is not None and value < minimum):
        raise ValueError('NONFINITE_OR_RANGE:' + name)
    return value


def accounting(record, epsilon, factor=1.0):
    """Exact legacy order, including log-domain-only and u/headroom guards."""
    factor = finite(factor, 'factor', minimum=0.0)
    if factor not in (1.0, 10.0):
        raise ValueError('UNREGISTERED_ALLOWANCE_FACTOR')
    b = record['error_decomposition']['total_signed']
    u = record['allowance']
    if u.get('kind') != 'EMPIRICAL' or u.get('certified') is not False:
        raise ValueError('SAVED_EMPIRICAL_ALLOWANCE_REQUIRED')
    bias = {a: abs(finite(b[a], 'bias')) for a in ('real', 'imag')}
    allowances = {a: factor*finite(u[a], 'u', minimum=0.0) for a in ('real', 'imag')}
    log_b = finite(record['log_B'], 'log_B', minimum=0.0)
    margin = finite(u['log_B_margin'], 'log_B_margin', minimum=0.0)
    result = _rule.u_aware_shots(bias, allowances, epsilon=epsilon,
        log_B_upper=log_b+factor*margin, evidence_kind='EMPIRICAL',
        evidence_ref=record['cell']['id']+'_signal.json', alpha_axis=.025)
    ratios = [r['u_over_headroom'] for r in result['axes'].values()]
    usable = (result['eligibility_under_declared_allowance'] == 'ELIGIBLE'
        and result['N_total'] is not None
        and all(x is not None and x <= .01 for x in ratios))
    result['cost_acquisition_eligible'] = usable
    result['boundary_guard'] = 'PASS' if usable else 'INELIGIBLE_OR_NUMERICALLY_UNRESOLVED'
    return result


def parse_wrappers(data, *, maximum_rows=4096):
    rows = []
    for r in csv.DictReader(io.StringIO(data.decode('utf-8'))):
        if len(rows) >= maximum_rows:
            raise ValueError('WRAPPER_ROW_CAP')
        required = {'cell_id','phase','replica','seed','control','axis','event_digest','compiler_hash',*METRICS}
        if not required <= r.keys() or None in r:
            raise ValueError('WRAPPER_SCHEMA')
        if r['axis'] not in AXIS_CHANNEL or r['control'] not in ('symmetric_directional','ordinary'):
            raise ValueError('WRAPPER_AXIS_CONTROL')
        for k in ('replica','seed',*METRICS):
            v = r[k]
            if not isinstance(v, str) or not v.isascii() or not v.isdigit() or len(v) > 32:
                raise ValueError('NONNEGATIVE_INTEGER_FIELD:' + k)
            r[k] = int(v)
        rows.append(r)
    return rows


def paired_rows(rows, *, cell_id, phase, compiler_hash):
    pairs = {}
    for r in rows:
        if r['cell_id'] != cell_id or r['control'] != 'symmetric_directional':
            continue
        if r['phase'] != phase or r['compiler_hash'] != compiler_hash:
            raise ValueError('WRAPPER_PHASE_OR_COMPILER_BINDING')
        p = pairs.setdefault(r['replica'], {})
        if r['axis'] in p:
            raise ValueError('DUPLICATE_PAIRED_AXIS')
        p[r['axis']] = r
    for p in pairs.values():
        if set(p) != set(AXIS_CHANNEL):
            raise ValueError('INCOMPLETE_PAIRED_AXES')
        if any(p['cosine'][k] != p['sine'][k] for k in ('seed','event_digest','compiler_hash')):
            raise ValueError('PAIRED_TRAJECTORY_IDENTITY')
    return [pairs[k] for k in sorted(pairs)]


def rational(value):
    x = Fraction(value)
    return {'numerator': str(x.numerator), 'denominator': str(x.denominator)}


def mean_only_metric(axis_means, shots):
    """Means suffice for a point, never for invented paired variance."""
    if set(axis_means)!=set(AXIS_CHANNEL) or any(v is None for v in axis_means.values()):
        return dict(status='NOT_SAVED',G_point=None,G_standard_error=None,missing_reason='AXIS_MEAN_NOT_SAVED')
    if any(type(n) is not int or n<=0 for n in shots.values()) or set(shots)!=set(AXIS_CHANNEL):
        raise ValueError('MEAN_ONLY_SHOTS')
    point=sum(shots[a]*finite(axis_means[a],'metric mean',minimum=0.0) for a in AXIS_CHANNEL)
    if not math.isfinite(point):raise ValueError('WEIGHTED_COST_OVERFLOW')
    return dict(status='DERIVED_SAVED_SCALAR',G_point=point,G_standard_error=None,
        missing_reason='PER_TRAJECTORY_PAIRED_COVARIANCE_NOT_SAVED',paired_samples=None)


def weighted_metric(pairs, shots, metric, *, point_rule='sum_axis_products'):
    if metric not in METRICS or set(shots) != set(AXIS_CHANNEL):
        raise ValueError('METRIC_OR_SHOT_SCHEMA')
    if any(type(n) is not int or n <= 0 or n >= 2**52 for n in shots.values()):
        raise ValueError('UNRELIABLE_INTEGER_SHOTS')
    if not pairs:
        return {'status':'MISSING','missing_reason':'COST_NOT_SAVED','G_point':None,'G_standard_error':None}
    values = {a: [p[a][metric] for p in pairs] for a in AXIS_CHANNEL}
    if any(type(x) is not int or x < 0 for v in values.values() for x in v):
        raise ValueError('INTEGER_METRIC_REQUIRED')
    totals = [sum(shots[a]*p[a][metric] for a in AXIS_CHANNEL) for p in pairs]
    means = {a: statistics.mean(v) for a, v in values.items()}
    if point_rule == 'sum_axis_products':
        point = sum(shots[a]*means[a] for a in AXIS_CHANNEL)
    elif point_rule == 'mean_paired_totals':
        point = statistics.mean(totals)
    else:
        raise ValueError('POINT_ARITHMETIC_RULE')
    se = statistics.stdev(totals)/math.sqrt(len(totals)) if len(totals) > 1 else None
    if not math.isfinite(point) or (se is not None and not math.isfinite(se)):
        raise ValueError('WEIGHTED_COST_OVERFLOW')
    return dict(status='DERIVED_SAVED_SCALAR',G_point=point,G_exact=rational(Fraction(sum(totals),len(totals))),
        G_standard_error=se,paired_samples=len(totals),wrapper_axes=2*len(totals),
        axis_means=means,one_shot_axis_max={a:max(v) for a,v in values.items()},
        uncertainty_scope='PAIRED_SAMPLE_SE_NOT_CI' if se is not None else 'DETERMINISTIC_N1_SE_NOT_DEFINED',
        point_rule=point_rule,population_winner_certified=False)


def family_selection(rows, metric):
    """RZ-selected reporting and metric-selected reporting remain distinct."""
    out = {}
    for family in ('B0','B1','B2'):
        eligible = [r for r in rows if r['method'] == family and r['status'] == 'DERIVED_SAVED_SCALAR']
        def choose(m):
            valid = [r for r in eligible if m in r['metrics']]
            return min(valid,key=lambda r:(r['metrics'][m]['G_point'],r['cell_id'])) if valid else None
        rz, own = choose('RZ'), choose(metric)
        out[family] = dict(RZ_selected_cell=rz['cell_id'] if rz else None,
            metric_at_RZ_selected=rz['metrics'][metric] if rz else None,
            metric_selected_cell=own['cell_id'] if own else None,
            metric_selected_value=own['metrics'][metric] if own else None,
            exact_RZ_ties=sorted(r['cell_id'] for r in eligible if rz and r['metrics']['RZ']['G_point']==rz['metrics']['RZ']['G_point']),
            point_minimum_only=True)
    return out


def fixed_crossing(a, b):
    """Exact crossing of two saved-cost lines; P >= 0 only."""
    da, db = Fraction(a['intercept']), Fraction(b['intercept'])
    na, nb = a['slope'], b['slope']
    if type(na) is not int or type(nb) is not int or min(na,nb) <= 0:
        raise ValueError('PREPARATION_SLOPE_INTEGER')
    if na == nb:
        return dict(status='IDENTICAL_LINES' if da==db else 'PARALLEL_NO_CROSSING',P=None)
    p = (db-da)/(na-nb)
    return dict(status='NONNEGATIVE_CROSSING' if p>=0 else 'NEGATIVE_DOMAIN_EXCLUDED',P=rational(p) if p>=0 else None)


def lower_envelope(lines):
    """All-candidate exact lower hull on P>=0; equal lines keep all ties."""
    by_slope = {}
    for line in lines:
        n = line['slope']
        if type(n) is not int or n <= 0:
            raise ValueError('PREPARATION_SLOPE_INTEGER')
        intercept = Fraction(line['intercept'])
        old = by_slope.get(n)
        if old is None or intercept < old['intercept']:
            by_slope[n] = dict(slope=n,intercept=intercept,cell_ids=[line['cell_id']])
        elif intercept == old['intercept']:
            old['cell_ids'].append(line['cell_id'])
    hull, starts = [], []
    for n in sorted(by_slope,reverse=True):
        line = by_slope[n]
        start = None
        while hull:
            start = (line['intercept']-hull[-1]['intercept'])/(hull[-1]['slope']-n)
            if len(hull)==1 or start > starts[-1]:
                break
            hull.pop(); starts.pop()
        hull.append(line); starts.append(start if len(hull)>1 else None)
    segments = []
    for i,line in enumerate(hull):
        left = max(Fraction(0),starts[i]) if starts[i] is not None else Fraction(0)
        right = starts[i+1] if i+1<len(hull) else None
        if right is not None and right<=left:
            continue
        segments.append(dict(cell_ids=sorted(line['cell_ids']),slope=line['slope'],
            intercept=rational(line['intercept']),P_start=rational(left),P_end=rational(right) if right is not None else None))
    # At vertices, lines with zero-width winning intervals can tie too.
    vertices = []
    for p in sorted({Fraction(0),*(s for s in starts if s is not None and s>=0)}):
        best = min((Fraction(l['intercept'])+l['slope']*p for l in lines),default=None)
        vertices.append(dict(P=rational(p),tied_cells=sorted(l['cell_id'] for l in lines
            if Fraction(l['intercept'])+l['slope']*p==best)))
    return dict(domain='P>=0',segments=segments,vertices=vertices,hypothetical_per_shot_only=True)


def analyze_scope(signals, rows, spec, epsilons, factors, baseline_rows=(), guard=None):
    """Future execution only on real inputs; fixtures exercise this path today."""
    records = []
    compiler, = spec['compiler_hashes']
    selected = set(spec.get('confirmation_ids', []))
    seen = set()
    for s in signals:
        if guard is not None: guard()
        cell = s['cell']; cid = cell['id']
        if cid in seen:
            raise ValueError('DUPLICATE_SIGNAL_CELL')
        seen.add(cid)
        if selected and cell['method']=='B2' and cid not in selected:
            continue
        source_rows, phase = rows, spec['primary_phase']
        if selected and cell['method']!='B2':
            source_rows, phase = baseline_rows, 'exploration'
        pairs = paired_rows(source_rows,cell_id=cid,phase=phase,compiler_hash=compiler)
        expected = 1 if cell['method']!='B2' else 8 if phase=='exploration' else 32
        if pairs and len(pairs)!=expected:
            raise ValueError('PAIRED_SAMPLE_COUNT')
        for epsilon in epsilons:
            for factor in factors:
                a = accounting(s,epsilon,factor)
                status = 'DERIVED_SAVED_SCALAR' if a['cost_acquisition_eligible'] and pairs else (
                    'INELIGIBLE' if a['eligibility_under_declared_allowance']=='INELIGIBLE' else
                    'NUMERICALLY_UNRESOLVED' if not a['cost_acquisition_eligible'] else 'MISSING')
                metrics = {}
                if status=='DERIVED_SAVED_SCALAR':
                    shots = {axis:a['axes'][channel]['shots'] for axis,channel in AXIS_CHANNEL.items()}
                    metrics = {m:weighted_metric(pairs,shots,m,point_rule=spec['point_rule']) for m in METRICS}
                records.append(dict(cell_id=cid,method=cell['method'],cell=cell,epsilon=epsilon,factor=str(factor),
                    accounting=a,status=status,metrics=metrics,N_total=a['N_total'],
                    missing_reason='COST_NOT_ACQUIRED' if status=='MISSING' else None,
                    sample_phase=phase,sample_count=len(pairs),confirmation_coverage=cid in selected,
                    new_independent_evidence=False))
    return records


def verify_original(signals, records, old_map, *, h4=False):
    """Future real-data equality gate. Not invoked on real data in preflight."""
    by={(r['cell_id'],str(r['epsilon']),r['factor']):r for r in records}
    shot_cases=cost_cases=0
    for s in signals:
        for eps,variants in s['precision_rows'].items():
            for factor,a in variants.items():
                new=by.get((s['cell']['id'],eps,factor))
                if new is None: continue
                if new['accounting']!=a:
                    raise ValueError('ORIGINAL_PRECISION_ROWS_MISMATCH:'+s['cell']['id'])
                shot_cases+=1
    for old in old_map['rows']:
        key=(old['cell_id'],str(old['epsilon']),old.get('allowance_factor','1.0'))
        new=by.get(key);metric='RZ' if h4 else old['metric']
        if new is None or new['status']!='DERIVED_SAVED_SCALAR':
            raise ValueError('ORIGINAL_MAP_COVERAGE')
        actual=new['metrics'][metric]
        if (actual['G_point']!=old['G_RZ_point' if h4 else 'G_point'] or
            actual['G_standard_error']!=old['G_RZ_SE' if h4 else 'G_standard_error'] or
            new['N_total']!=old['N_total']):
            raise ValueError('ORIGINAL_MAP_VALUE_MISMATCH:'+old['cell_id'])
        cost_cases+=1
    return dict(status='PASS',shot_cases=shot_cases,cost_cases=cost_cases)


def execution_gate(contract, authorization, *, contract_sha, output):
    if (authorization.get('approved_by_user') is not True or
            authorization.get('scope')!='D0_SAVED_SCALAR_ONE_SHOT' or
            authorization.get('contract_sha256')!=contract_sha or
            authorization.get('analysis_source')!=contract['analysis_source'] or
            authorization.get('exclusive_output')!=str(output) or
            authorization.get('automatic_retry') is not False):
        raise ValueError('SEPARATE_D0_EXECUTION_APPROVAL_REQUIRED')
    if contract.get('new_science_calls')!=0 or contract.get('H8_included') is not False:
        raise ValueError('D0_SCOPE_CHANGED')
    if output.is_symlink() or not output.is_absolute() or output.resolve()!=output or not output.is_relative_to('/home/AbeHiromu'):
        raise ValueError('EXCLUSIVE_HOME_OUTPUT_REQUIRED')
    if output.exists() or output.with_name(output.name+'.oneshot').exists():
        raise ValueError('OUTPUT_OR_ONE_SHOT_ALREADY_USED')


def memory_snapshot():
    """Host plus every visible ancestor of our own unified cgroup."""
    mem=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
    def full_psi(path):
        line=next(line for line in path.read_text().splitlines() if line.startswith('full '))
        return float(dict(part.split('=') for part in line.split()[1:])['avg10'])
    available=int(mem['MemAvailable'].split()[0])*1024
    host_psi=full_psi(Path('/proc/pressure/memory'))
    oom=int(dict(line.split() for line in Path('/proc/vmstat').read_text().splitlines())['oom_kill'])
    member=next(line[3:] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::'))
    mounts=[line.split(' - ',1) for line in Path('/proc/self/mountinfo').read_text().splitlines()]
    mounts=[left.split() for left,right in mounts if right.split()[0]=='cgroup2']
    if len(mounts)!=1:raise ValueError('UNIFIED_CGROUP_MOUNT_REQUIRED')
    mount_root,mountpoint=Path(mounts[0][3]),Path(mounts[0][4])
    try:relative=Path(member).relative_to(mount_root)
    except ValueError as e:raise ValueError('CGROUP_NAMESPACE_UNRESOLVED') from e
    current=mountpoint/relative;ancestors=[]
    if not current.is_dir():raise ValueError('OWN_CGROUP_NOT_VISIBLE')
    while True:
        if (current/'memory.max').is_file():
            maximum=(current/'memory.max').read_text().strip()
            usage=int((current/'memory.current').read_text())
            events=dict((k,int(v)) for k,v in (line.split() for line in (current/'memory.events').read_text().splitlines()))
            remaining=None if maximum=='max' else int(maximum)-usage
            if remaining is not None:available=min(available,remaining)
            ancestors.append(dict(path=str(current),full_psi_avg10=full_psi(current/'memory.pressure'),events=events))
        elif current!=mountpoint or mount_root!=Path('/'):
            raise ValueError('CGROUP_MEMORY_INTERFACE_MISSING')
        if current==mountpoint:break
        current=current.parent
    return dict(effective_available_bytes=available,host_full_psi_avg10=host_psi,host_oom_kill=oom,cgroups=ancestors)


def check_memory_snapshot(snapshot,budget,baseline=None):
    if (snapshot['effective_available_bytes']<budget['fresh_memory_available_bytes'] or
        snapshot['host_full_psi_avg10']>=1.0 or any(c['full_psi_avg10']>0. for c in snapshot['cgroups'])):
        raise ValueError('MEMORY_PRESSURE_OR_HEADROOM_GATE')
    if baseline is not None:
        if snapshot['host_oom_kill']!=baseline['host_oom_kill']:
            raise ValueError('HOST_OOM_INCREMENT')
        if [c['path'] for c in snapshot['cgroups']]!=[c['path'] for c in baseline['cgroups']]:
            raise ValueError('OWN_CGROUP_CHANGED')
        for new,old in zip(snapshot['cgroups'],baseline['cgroups'],strict=True):
            if any(new['events'].get(k,0)!=old['events'].get(k,0) for k in ('oom','oom_kill','oom_group_kill')):
                raise ValueError('CGROUP_OOM_INCREMENT')


def fresh_memory_gate(budget):
    """Own read-only admission; no affinity, quota or cgroup changes."""
    snapshot=memory_snapshot()
    check_memory_snapshot(snapshot,budget)
    # Quota-enabled filesystem needs a separately verified user/group receipt.
    mounts=Path('/proc/mounts').read_text().splitlines()
    relevant=[line.split() for line in mounts if len(line.split())>=4 and
        Path('/home/AbeHiromu').is_relative_to(line.split()[1])]
    if relevant:
        mount=max(relevant,key=lambda fields:len(fields[1]))
        options=set(mount[3].split(','))
        if any('quota' in opt and not opt.startswith('no') for opt in options):
            raise ValueError('QUOTA_HEADROOM_RECEIPT_REQUIRED')
    return dict(**snapshot,quota_mount_options=mount[3] if relevant else None)


def bound_input(path, receipt):
    path = Path(path)
    if path.is_symlink() or not path.is_absolute() or path.resolve()!=path or not path.is_file() or not path.is_relative_to('/home/AbeHiromu'):
        raise ValueError('INPUT_REGULAR_ABSOLUTE_FILE_REQUIRED')
    if path.suffix not in ('.json','.csv') or receipt['bytes']>16*2**20 or path.stat().st_size!=receipt['bytes']:
        raise ValueError('SCALAR_INPUT_SIZE_OR_FORMAT')
    with path.open('rb') as f:
        data=f.read(receipt['bytes']+1)
    if len(data)!=receipt['bytes'] or hashlib.sha256(data).hexdigest()!=receipt['sha256']:
        raise ValueError('INPUT_BYTES_SHA_MISMATCH')
    return data


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract',required=True)
    parser.add_argument('--authorization',required=True)
    parser.add_argument('--inputs',required=True,help='private alias-to-absolute-path JSON')
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    contract_data=Path(args.contract).read_bytes()
    contract=strict_json(contract_data)
    auth=strict_json(Path(args.authorization).read_bytes())
    execution_gate(contract,auth,contract_sha=hashlib.sha256(contract_data).hexdigest(),output=args.output)
    root=Path(__file__).resolve().parents[1]
    for p,h in contract['analysis_source_hashes'].items():
        disk=(root/p).read_bytes()
        fixed=subprocess.check_output(['git','-C',str(root),'show',contract['analysis_source']+':'+p])
        if disk!=fixed or hashlib.sha256(disk).hexdigest()!=h:
            raise ValueError('ANALYSIS_SOURCE_BINDING')
    if sys.version.split()[0]!=contract['analysis_python']:
        raise ValueError('ANALYSIS_PYTHON_BINDING')
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        if os.environ.get(k)!='1':raise ValueError('PROCESS_THREAD_ENV_REQUIRED')
    budget=contract['budget']
    resource.setrlimit(resource.RLIMIT_AS,(budget['address_space_bytes'],budget['address_space_bytes']))
    resource.setrlimit(resource.RLIMIT_CPU,(budget['cpu_seconds'],budget['cpu_seconds']+1))
    def timeout(*_):raise ValueError('D0_WALL_CAP')
    signal.signal(signal.SIGALRM,timeout)
    signal.signal(signal.SIGXCPU,timeout)
    signal.setitimer(signal.ITIMER_REAL,budget['wall_seconds'])
    stat=os.statvfs(args.output.parent)
    if stat.f_bavail*stat.f_frsize<budget['fresh_available_bytes'] or stat.f_favail<budget['fresh_available_inodes']:
        raise ValueError('FRESH_CAPACITY_GATE')
    admission=fresh_memory_gate(budget)
    paths=strict_json(Path(args.inputs).read_bytes())
    data={r['alias']:bound_input(paths[r['alias']],r) for r in contract['inputs']}
    # Take a durable exclusive marker before any real accounting. Never remove it.
    lock=args.output.with_name(args.output.name+'.oneshot')
    with lock.open('x') as f:json.dump(dict(contract_sha256=hashlib.sha256(contract_data).hexdigest(),pid=os.getpid()),f)
    args.output.mkdir()
    started=time.monotonic();first_stop=None
    def guard():
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>budget['rss_bytes']:
            raise ValueError('D0_RSS_CAP')
        check_memory_snapshot(memory_snapshot(),budget,admission)
    try:
        all_records={};validation={};selections={};sensitivity={}
        for scope,spec in contract['scopes'].items():
            signals=strict_json(data[spec['signal_alias']])['signals']
            rows=parse_wrappers(data[spec['wrapper_alias']])
            baselines=parse_wrappers(data[spec['deterministic_baseline_wrapper_alias']]) if 'deterministic_baseline_wrapper_alias' in spec else []
            # Reproduce registered values before extending epsilon or metrics/P.
            original=analyze_scope(signals,rows,spec,contract['original_epsilons'],contract['allowance_factors'],baselines,guard)
            validation[scope]=verify_original(signals,original,strict_json(data[spec['old_map_alias']]),h4=scope==SCOPES[0])
            records=analyze_scope(signals,rows,spec,contract['epsilons'],contract['allowance_factors'],baselines,guard)
            all_records[scope]=records;selections[scope]={};sensitivity[scope]={}
            for eps in contract['epsilons']:
                for factor in contract['allowance_factors']:
                    subset=[r for r in records if r['epsilon']==eps and r['factor']==str(factor)]
                    key=str(eps)+'/'+str(factor)
                    selections[scope][key]={m:family_selection(subset,m) for m in METRICS}
                    sensitivity[scope][key]={}
                    for m in METRICS:
                        guard()
                        lines=[dict(cell_id=r['cell_id'],slope=r['N_total'],intercept=r['metrics'][m]['G_point'])
                            for r in subset if r['status']=='DERIVED_SAVED_SCALAR']
                        fixed_ids={entry['metric_selected_cell'] for entry in family_selection(subset,m).values() if entry['metric_selected_cell']}
                        fixed=[line for line in lines if line['cell_id'] in fixed_ids]
                        sensitivity[scope][key][m]=dict(units=m+' per shot',lower_envelope=lower_envelope(lines),
                            fixed_pair_scope='metric-selected family representatives at P=0; no reselection for these crossings',
                            fixed_pairs=[dict(a=a['cell_id'],b=b['cell_id'],**fixed_crossing(a,b))
                                for i,a in enumerate(fixed) for b in fixed[i+1:]])
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>budget['rss_bytes']:
                raise ValueError('D0_RSS_CAP')
        provenance=dict(schema='track_a_d0_reanalysis_result_v1',analysis_source=contract['analysis_source'],
            contract_sha256=hashlib.sha256(contract_data).hexdigest(),input_receipts=contract['inputs'],conditions=contract['scopes'],
            fresh_admission=admission,
            new_science_calls=0,new_independent_evidence=False,H8_changes=0,formal_winner_certified=False,
            wall_seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        outputs={'candidate_accounting_v1.json':all_records,'original_equality_v1.json':validation,
            'selection_metrics_v1.json':selections,'preparation_sensitivity_v1.json':sensitivity,'provenance_v1.json':provenance}
        total=0
        for name,obj in outputs.items():
            with (args.output/name).open('xb') as f:
                last_check=time.monotonic()
                for chunk in json.JSONEncoder(ensure_ascii=False,indent=2,allow_nan=False).iterencode(obj):
                    payload=chunk.encode();total+=len(payload)
                    if total>budget['output_bytes']:raise ValueError('D0_OUTPUT_CAP')
                    f.write(payload)
                    if time.monotonic()-last_check>=.5:
                        guard();last_check=time.monotonic()
                f.write(b'\n');total+=1
        guard()
    except Exception as e:
        first_stop=type(e).__name__+':'+str(e)
        raise
    finally:
        with (args.output/'terminal_v1.json').open('x') as f:
            json.dump(dict(status='COMPLETE_MANDATORY_STOP' if first_stop is None else 'FAIL_CLOSED_STOP',
                first_stop_reason=first_stop,automatic_retry=False,next_stage_authorized=False),f)


if __name__=='__main__':
    main()
