"""One normal PR-LD2/m12 path, X/Y, at most two compiles. No retry or mean.

Passive plan/preflight; explicit run_worker is the scientific entry. The saved
loader, candidate Runtime, normal sampler and full fast acquisition are reused.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

from . import candidate_price_adapter as D
from . import ld12_price_requests as L
from . import p1_efficient_zz as P

BASE = '39678f5a063490641621e836e53737cb39eb6979'
COHORT = 'D_LONG_TIME_FEASIBILITY_V1'
PUBLIC = 'docs/research/d_ld2_m12_t2_20261113/run01'
PRIVATE = '/home/AbeHiromu/projects/d-long-time-price-t2/20261113/run01'
FILES = ('src/trottertracks/ld2_m12_t2.py', 'scripts/run_ld2_m12_t2.py',
    'src/trottertracks/candidate_price_adapter.py', 'src/trottertracks/compiled_price_fast.py',
    'src/trottertracks/ld12_price_requests.py', 'src/trottertracks/efficient_controlled_zz.py',
    'src/trottertracks/h4_df10_price_v1.py', 'src/trottertracks/track_d_native_resources.py',
    'scripts/h46_b3_fast_fingerprint.py')


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def resources(pid=None, disk_path=None):
    pid = pid or os.getpid()
    status = Path('/proc')/str(pid)/'status'
    fields = {}
    try:
        lines=status.read_text().splitlines()
    except FileNotFoundError:
        lines=[]
    if lines:
        for line in lines:
            k, _, v = line.partition(':')
            if k in ('VmRSS', 'VmHWM', 'VmSize'):
                fields[k+'_bytes'] = int(v.split()[0])*1024
    try:
        io=Path('/proc',str(pid),'io').read_text().splitlines()
        fields['write_bytes']=int(next(line.split(':')[1] for line in io if line.startswith('write_bytes:')))
    except FileNotFoundError:
        pass
    mem = {k: int(v.split()[0])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
           for k, v in [line.split(':', 1)] if k in ('MemAvailable', 'MemTotal', 'SwapFree')}
    result = dict(unix_time=time.time(), pid=pid, process=fields, memory=mem,
                  loadavg=list(os.getloadavg()), affinity_cpu_count=len(os.sched_getaffinity(0)))
    if disk_path:
        fs = os.statvfs(disk_path)
        result['disk_available_bytes'] = fs.f_bavail*fs.f_frsize
    return result


def resource_exhaustion(previous, current):
    """Actual free capacity versus observed self-growth; never PSI alone."""
    if previous is None:return False
    memory_growth=max(0,current['process'].get('VmRSS_bytes',0)-previous['process'].get('VmRSS_bytes',0))
    writes=max(0,current['process'].get('write_bytes',0)-previous['process'].get('write_bytes',0))
    return bool(memory_growth and current['memory']['MemAvailable'] < 2*memory_growth
                or writes and current['disk_available_bytes'] < 2*writes)


def plan(root):
    """No event generation. Pin only this finite selection and original seed rule."""
    r = L.request_for(root, 'PR-LD2', 12, axes=('X', 'Y'),
                      execution_source={'purpose': 'T2_PLAN_METADATA_ONLY'})
    c = r['physical_context']['physical_candidate']
    seed = P.trial_seed_function(root)
    seeds = [seed(c, COHORT, 0, outer, 0) for outer in range(4096)]
    runtime = json.loads(P._blob(root, *P.C1_PLAN))['runtime']
    return dict(schema='D_LD2_M12_T2_PLAN_V1', base_source=BASE, A_source=L.A_SOURCE,
        candidate='PR-LD2', m=12, axes=['X', 'Y'], cohort=COHORT, index=0, n_paths=1,
        compile_budget=2, q_PF=4096, R_outer=2559, T=409.6, logical_events=10481664,
        physical_id=c['id'], configuration_id=r['configuration_id'],
        seed_list_sha256=D._key(seeds), seed_rule_source=P.CARRIER,
        runtime=runtime, public=PUBLIC, private=PRIVATE,
        purpose='ONE_PATH_TECHNICAL_FEASIBILITY_NOT_MEAN_OR_CI',
        monitor=dict(interval_seconds=10, headroom_rule='AVAILABLE_BELOW_OWN_OBSERVED_GROWTH_OVER_NEXT_TWO_INTERVALS',
                     PSI_alone_never_stops=True, fixed_CPU_RAM_worker_cap=None),
        retry=False, mandatory_stop=True)


def check_scope(request):
    D.check_request(request)
    c = request['physical_context']['physical_candidate']
    expected = dict(system='H4', method='PR', prefix=2, formula='2nd', K=2,
                    q=4096, R_outer=2559, T=409.6, delta=.1)
    if (request['m'] != 12 or request['axes'] != ['X', 'Y']
            or any(c[k] != v for k, v in expected.items())
            or request['physical_context'].get('expected_scalar') is None):
        raise ValueError('T2_FIXED_LD2_M12_XY_SCOPE')


def preflight(root, source, fixed_plan):
    """SOURCE/input bytes checked before any physical restore/preparation."""
    root = Path(root)
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root).decode().strip() != source:
        raise ValueError('EXECUTION_SOURCE_HEAD_DIFFERS')
    for name in FILES:
        if (root/name).read_bytes() != P._blob(root, source, name):
            raise ValueError('EXECUTION_SOURCE_BYTES_DIFFER:'+name)
    if fixed_plan != plan(root):
        raise ValueError('FIXED_T2_PLAN_DIFFERS')
    r = L.request_for(root, 'PR-LD2', 12, axes=('X', 'Y'), execution_source={'commit': source})
    check_scope(r)
    science = P.load_c1_backend(root)['r_price_phase1_science']
    cfg = science.read(fixed_plan['runtime']['input_config'])['systems']['H4']
    science.m._binding_matches(cfg['binding'], r['physical_context']['binding'])
    old = json.loads(P._blob(root, science.m.H4V1,
                             'docs/research/h4_df10_price_v1_20261110/DEVELOPMENT.json'))
    inputs = {}
    for name, expected in old['input_sha256'].items():
        item = cfg['inputs'][name]
        if sha(item['path']) != expected or item['sha256'] != expected or Path(item['path']).stat().st_size != item['bytes']:
            raise ValueError('SAVED_INPUT_BYTES_DIFFER:'+name)
        inputs[name] = deepcopy(item)
    provider = Path(fixed_plan['runtime']['provider_root'])
    # These are the provider modules actually used by restore/prepare/build/compile.
    for name in ('df_hamiltonian','rte','df_partial_s2','df_rte_qiskit','rte_compiled_cost',
                 'df_rte_circuit','df_partial_randomized_pf','df_trotter/ops'):
        rel = 'src/trotterlib/'+name+'.py'
        if (provider/rel).read_bytes() != P._blob(root, D.zz.PROVIDER, rel):
            raise ValueError('PROVIDER_SOURCE_DIFFERS:'+rel)
    return r, inputs


def restore_h4(root, request, inputs, runtime_plan):
    """Original saved-array loader only; no SCF, DF generation or state solver."""
    provider = Path(runtime_plan['provider_root'])
    sys.path.insert(0, str(provider/'src'))
    import trottertracks
    if str(provider/'src/trottertracks') not in trottertracks.__path__:
        trottertracks.__path__.insert(0, str(provider/'src/trottertracks'))
    import numpy, qiskit
    if (numpy.__version__,qiskit.__version__)!=('1.26.4','1.3.0'):
        raise ValueError('FIXED_SCIENTIFIC_RUNTIME')
    science = P.load_c1_backend(root)['r_price_phase1_science']
    def require(ok, reason):
        if not ok: raise ValueError(reason)
    scope = dict(Path=Path, c=SimpleNamespace(read=science.read, sha=sha, require=require))
    science.functions(root, P.CARRIER, 'scripts/h46_m5_tol_saved_input.py', ['array_record', 'load'], scope)
    ham, sector, full, state, md = scope['load'](dict(mapping=dict(
        tol_snapshot=inputs['H4_tol_snapshot.npz']['path'], tol_receipt=inputs['H4_tol_receipt.json']['path'])))
    binding = request['physical_context']['binding']
    if (md['hamiltonian_hash'] != binding['hamiltonian'] or md['state_hash'] != binding['state']
            or md['rank'] != 10 or len(state) != 36):
        raise ValueError('RESTORED_H4_IDENTITY_OR_SECTOR')
    return ham, md


def run_worker(root, source, fixed_plan, request, inputs, public, private,
               *, restore=restore_h4, runtime_class=None, adapter_class=None):
    """Exactly one sampler call and one paired acquisition; exceptions are final."""
    runtime_class = runtime_class or D.NativeRuntime
    adapter_class = adapter_class or D.Adapter
    public, private = Path(public), Path(private)
    stages, compiles = [], []
    start = time.perf_counter()
    def observe(name, status, seconds=0., value=None):
        entry = dict(stage=name, status=status, seconds=seconds,
                     elapsed_seconds=time.perf_counter()-start, peak_rss_bytes=rss())
        if status == 'COMPLETE' and name == 'compose':
            entry['native_instructions'] = len(value.circuit.data)
            entry['native_fingerprint'] = value.circuit_fingerprint
        if status == 'COMPLETE' and name.endswith('_wrapper'):
            entry['instructions'] = len(value.data)
        if status == 'COMPLETE' and name == 'event_digest':
            entry['event_digest']=value
        stages.append(entry)
        with (public/'PIPELINE_STAGES.jsonl').open('a') as f:
            f.write(json.dumps(entry)+'\n')
        print(name, status, round(entry['elapsed_seconds'], 3), flush=True)
    def timed(name, fn):
        observe(name, 'STARTED'); then=time.perf_counter(); value=fn()
        observe(name, 'COMPLETE', time.perf_counter()-then, value); return value
    def interrupted(signum, frame):
        raise RuntimeError('SUPERVISOR_RESOURCE_STOP_SIGTERM')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        check_scope(request)
        write(public/'REQUEST.json', request)
        ham, md = timed('input_restore', lambda: restore(root, request, inputs, fixed_plan['runtime']))
        runtime = runtime_class(root, ham, binding=request['physical_context']['binding'],
            input_source=request['physical_context']['input_source'], execution_source={'commit': source})
        ctx = timed('preparation_context', lambda: runtime.context(request))
        resolved = dict(preparation_hash=ctx.prep.preparation_hash,
            tail_hash=ctx.prep.rte_preparation.symbolic_tail.tail_hash, scalar=ctx.scalar,
            lambda_R=ctx.prep.rte_preparation.symbolic_tail.lambda_r,
            deterministic_fragments=list(ctx.prep.deterministic_fragment_indices),
            randomized_fragments=list(ctx.prep.randomized_block_indices))
        pc = request['physical_context']
        if (resolved['deterministic_fragments'] != [0, 1] or resolved['randomized_fragments'] != list(range(2,10))
                or float(resolved['lambda_R']).hex() != float(D.p.accounting.exact_scalar(pc['split']['lambda_R'])).hex()
                or float(resolved['scalar']).hex() != float(pc['expected_scalar']).hex()):
            raise ValueError('T2_RESOLVED_CONTEXT_DIFFERS')
        write(public/'PREPARATION.json', dict(input_metadata=md, resolved_context=resolved,
            components=[dict(id=c.component_id, basis_hash=c.basis_hash)
                        for c in ctx.prep.rte_preparation.symbolic_tail.components]))
        original = ctx.compile
        def compile_once(qc, settings, **kwargs):
            if len(compiles) >= 2: raise ValueError('T2_COMPILE_BUDGET')
            axis='Y' if stages[-1]['stage']=='sine_compile' else 'X'
            entry=dict(axis=axis, status='STARTED', instructions=len(qc.data))
            compiles.append(entry); then=time.perf_counter()
            write(public/('COMPILE_'+entry['axis']+'_STARTED.json'), entry)
            scope=getattr(original,'__globals__',{})
            transpile=scope.get('transpile')
            transpile_seconds=[]
            def observed_transpile(*a,**kw):
                before=time.perf_counter()
                try:return timed('qiskit_transpile_'+axis,lambda:transpile(*a,**kw))
                finally:transpile_seconds.append(time.perf_counter()-before)
            if transpile is not None:scope['transpile']=observed_transpile
            try:
                cost=original(qc, settings, **kwargs)
                entry['status']='COMPLETE'
                entry['compiled_instructions']=len(cost.transpiled_circuit.data)
                return cost
            except BaseException:
                entry['status']='FAILED'
                raise
            finally:
                if transpile is not None:scope['transpile']=transpile
                elapsed=time.perf_counter()-then
                entry.update(seconds=elapsed, peak_rss_bytes=rss(),
                    qiskit_transpile_seconds=sum(transpile_seconds),
                    non_transpile_including_metrics_seconds=elapsed-sum(transpile_seconds))
                write(public/('COMPILE_'+entry['axis']+'_FINISHED.json'), entry)
        ctx.compile=compile_once
        requests=timed('normal_sampling', lambda: runtime.sample(request, COHORT, 0))
        seeds, laws=D._law_records(ctx, pc['physical_candidate'], requests)
        if D._key(seeds) != fixed_plan['seed_list_sha256']:
            raise ValueError('FROZEN_SEED_LIST_DIFFERS')
        if sum(len(r.rte_occurrence.events) for r in requests) != 10481664:
            raise ValueError('T2_LOGICAL_EVENT_COUNT')
        write(public/'SAMPLING.json',dict(cohort=COHORT,index=0,seeds=seeds,laws=laws,logical_events=10481664))
        adapter=adapter_class(root,runtime,cache_root=fixed_plan['runtime']['old_cache'],observer=observe)
        result=timed('paired_acquisition',lambda:adapter.acquire(request,requests,cohort=COHORT,index=0,
            directory=private/'path',conditional=False))
        if result['new_compiles'] != len(compiles) or len(result['records']) != 2:
            raise ValueError('T2_AXIS_OR_COMPILE_COUNT')
        timed('final_save',lambda:write(public/'T2_PATH_RESULT.json',dict(status='T2_COMPLETE',path=result,stages=stages,
            compile_log=compiles,wall_seconds=time.perf_counter()-start,peak_rss_bytes=rss(),
            mean_compiled_price=None,confidence_interval=None,qualified_expectation=False,mandatory_stop=True)))
    except BaseException:
        write(public/'WORKER_STOP.json',dict(status='STOP',error=traceback.format_exc(),stages=stages,
            compile_log=compiles,wall_seconds=time.perf_counter()-start,peak_rss_bytes=rss(),mandatory_stop=True))
        raise
