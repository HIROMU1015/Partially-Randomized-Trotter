"""Approved C4 limited execution, reusing the frozen Q13/PR/reference/compiler.

All scientific imports are lazy and follow verify_packet. The preparation
contract stays unchanged; the current user's technical-retry amendment is bound
separately to each fresh, immutable attempt and frozen execution commit.
"""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
import gzip
import hashlib
import json
import multiprocessing
import os
import platform
import resource
import subprocess
import time
import traceback

from .fixture import canonical_json, digest

CONTRACT = 'docs/research/research_c_rustiq_limited_contract_20261024.json'
CONTRACT_SHA = 'e948a2e7fdf47e20b8af8cd7debae0631dabbd5e6ad77f419a9a9e2c668319a1'
CASE_IDS = ('SS_motif_2', 'SS_motif_3', 'SS_motif_4', 'PP_motif_0')


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_packet(root, packet, inputs, output):
    """Read-only, standard-library approval and identity gate before science."""
    root, packet, inputs, output = map(lambda p: Path(p).resolve(), (root, packet, inputs, output))
    approval = json.loads((packet/'approval.json').read_bytes())
    if (approval.get('stage') != 'C4_Q13_LIMITED_V1'
            or approval.get('authority') != 'USER_EXPLICIT_C4_EXECUTION'
            or approval.get('contract_sha256') != CONTRACT_SHA
            or approval.get('approved') is not True):
        raise PermissionError('Fresh explicit C4 execution approval required')
    if str(output) != approval['output'] or output.exists():
        raise PermissionError('Fresh bound output required; no attempt reuse')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    if head != approval['source_commit']:
        raise ValueError('Execution commit drift')
    # Artifacts from previous attempts are untracked and preserved, never source.
    if subprocess.check_output(['git', 'diff', '--name-only', 'HEAD'], cwd=root):
        raise ValueError('Tracked worktree source drift')
    if file_sha(root/CONTRACT) != CONTRACT_SHA:
        raise ValueError('Preparation contract drift')
    contract = json.loads((root/CONTRACT).read_bytes())
    if tuple(c['id'] for c in contract['cases']) != CASE_IDS:
        raise ValueError('Case inventory drift')
    for name, expected in {**contract['source_files'], **approval['execution_files']}.items():
        if file_sha(root/name) != expected:
            raise ValueError('SOURCE drift: '+name)
    input_bytes = {}
    for name, expected in contract['input_files'].items():
        data = (inputs/name).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('Input drift: '+name)
        input_bytes[name] = data
    return dict(root=root, packet=packet, output=output, contract=contract, inputs=input_bytes,
                approval=approval, source_identity=head+':'+digest(contract['source_files']),
                input_identity=digest(contract['input_files']))


class Journal:
    def __init__(self, out, task):
        self.out = Path(out)
        self.path = self.out/'journals'/(task+'.jsonl')

    def call(self, group, identifier, function):
        if (self.out/'STOP.json').exists():
            raise RuntimeError('Cooperative STOP before '+identifier)
        tick, cpu = time.perf_counter(), time.process_time()
        with self.path.open('a') as stream:
            def log(record):
                stream.write(json.dumps(dict(group=group, id=identifier, **record),
                                        ensure_ascii=False, allow_nan=False)+'\n')
                stream.flush()
            log(dict(status='ATTEMPTED'))
            try:
                result = function()
            except BaseException as error:
                failure = dict(status='FAILED', error_type=type(error).__name__, error=str(error),
                               traceback=traceback.format_exc(), group=group, id=identifier)
                log({k: v for k, v in failure.items() if k not in ('group', 'id')})
                try:
                    write_json(self.out/'STOP.json', failure)
                except FileExistsError:
                    pass
                raise
            log(dict(status='COMPLETED', wall_seconds=time.perf_counter()-tick,
                cpu_seconds=time.process_time()-cpu,
                peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
            return result


def _save_circuit(out, name, circuit):
    from qiskit import qpy
    path = Path(out)/'circuits'/(name+'.qpy')
    with path.open('xb') as stream:
        qpy.dump(circuit, stream)


def _save_reference(out, name, ctx, values):
    payload = {}
    for key, value in values.items():
        if hasattr(value, 'rows'):
            payload[key] = [[ctx.nstr(value[i, j], ctx.dps) for j in range(value.cols)]
                            for i in range(value.rows)]
        elif key == 'signal':
            payload[key] = ctx.nstr(value, ctx.dps)
    with (Path(out)/'references'/(name+'.json.gz')).open('xb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as stream:
            stream.write(canonical_json(dict(digits=ctx.dps, values=payload)))


def _matrix(out, name, actual, expected, *, controlled=False):
    """One Operator materialization, same absolute/ unitarity gates as legacy.

    Unlike controlled_gate's report-only API, save that very matrix without a
    second Operator evaluation. Expected diag(I,U) is independently supplied.
    """
    import numpy as np
    from qiskit.quantum_info import Operator
    from . import gates
    matrix = Operator(actual).data
    report = gates.matrix_gate(matrix, expected, 1e-10, label=name)
    if controlled:
        report['unitarity'] = gates.matrix_gate(matrix.conj().T@matrix,
            np.eye(len(matrix)), 1e-10, label=name+'_UNITARITY')
    np.save(Path(out)/'operators'/(name+'.npy'), matrix, allow_pickle=False)
    return report


def _prepare(payload):
    from .f1_saved_execution import SavedF1
    from . import reference as ref, gates, rustiq_adapter as adapter
    from .prototype import atom_record
    case, shared, out = payload
    cid, arm = case['id'], case['arm']
    journal = Journal(out, cid+'_reference')
    engine = SavedF1(dict(contract=dict(fixture_bundle_path='handoff/f1_bundle.json'),
                          inputs=shared['inputs']))
    resolved = journal.call('motif_restore', cid, lambda: engine.resolve_case(case))
    if canonical_json(resolved['macros']) != canonical_json(case['macros']):
        raise ValueError('Literal motif drift: '+cid)
    macros = engine.macros(arm, resolved['macros'])
    atoms = engine.atoms[arm]
    provenance = adapter.route_identity('R1', input_identity=shared['input_identity'],
                                        source_identity=shared['source_identity'])
    plans = tuple(adapter.plan_sequence(e, atoms, 4, provenance=provenance) for e in macros)
    target = []
    for mi, plan in enumerate(plans):
        for run in plan.runs:
            for region in run.regions:
                if region.eligible:
                    target.append(dict(id=f'{cid}/macro{mi}/{region.region_id}', macro=mi,
                        region=region.region_id, occurrence_ordinals=[o.ordinal for o in region.occurrences]))
    if target != case['target_regions']:
        raise ValueError('Region inventory drift: '+cid)
    refs, reports = {}, {}
    for digits in (90, 128):
        rec = dict(resolved, reference_kind='motif', digits=digits)
        if digits == 128:
            rec['paired_reference'] = cid+'_90'
        value, report = journal.call('reference', f'{cid}_{digits}', lambda: engine.reference_case(rec))
        refs[digits] = value
        reports[str(digits)] = report
        engine.cache[cid+'_'+str(digits)] = value
        _save_reference(out, cid+'_'+str(digits), value['ctx'], value)
    full = refs[128]
    local = []
    for region_record in target:
        mi = region_record['macro']
        region = next(r for run in plans[mi].runs for r in run.regions
                      if r.region_id == region_record['region'])
        name = region_record['id'].replace('/', '_').replace(':', '_')
        pair = {}
        for digits in (90, 128):
            ctx = refs[digits]['ctx']
            tau = engine.independent_laws[(arm, digits)][1]
            def reference_region():
                operator = ctx.eye(16)
                for occurrence in region.occurrences:
                    event = macros[mi][occurrence.event_index]
                    alpha = occurrence.coefficient_sign*ctx.atan(tau/(event.taylor_order+1))
                    word = tuple('Z' if i in occurrence.support else 'I' for i in range(4))
                    pauli = ref.pauli_operator(ctx, word)
                    operator = (ctx.cos(alpha)*ctx.eye(16)-ctx.j*ctx.sin(alpha)*pauli)*operator
                return ref.controlled_operator(ctx, operator)
            pair[digits] = journal.call('reference', f'{name}_{digits}', reference_region)
            _save_reference(out, name+'_'+str(digits), ctx, dict(controlled=pair[digits]))
        agreement = gates.high_precision_agreement(pair[90], pair[128])
        local.append(dict(name=name, region=region, expected=ref.to_numpy(pair[128]), agreement=agreement))
    # Binary64 angles are archived explicitly, with no snapping or reconstruction.
    write_json(Path(out)/'reports'/(cid+'_input.json'), dict(case=case,
        events=[[e.to_dict() for e in macro] for macro in macros], atoms=[atom_record(a) for a in atoms],
        plans=[asdict(p) for p in plans], angle_hex=[[o.signed_angle.hex() if o.signed_angle is not None
            else None for o in p.occurrences] for p in plans], reference_reports=reports,
        local_agreement={x['name']: x['agreement'] for x in local}))
    return dict(case=case, spec=engine.spec, context=engine.context, atoms=atoms, macros=macros,
        frames=engine.provider.frames, local=local,
        expected=ref.to_numpy(ref.controlled_operator(full['ctx'], full['operator'])),
        wrappers={a: ref.to_numpy(full['wrapper_'+a]) for a in ('X', 'Y')},
        signal=complex(full['signal']))


def _local(payload):
    from qiskit import QuantumCircuit
    from . import circuits, rustiq_adapter as adapter
    record, route, out = payload
    name = record['name']+'_'+route
    journal = Journal(out, name)
    def build():
        if route == 'R1':
            network = adapter.region_network(record['region'], ancilla=4)
            return adapter._q13(5, network, adapter.Options())
        qc = QuantumCircuit(5)
        for occurrence in record['region'].occurrences:
            circuits.append_central(qc, occurrence.support, 4, occurrence.signed_angle)
        return qc
    qc = journal.call('local_build', name, build)
    _save_circuit(out, name, qc)
    report = journal.call('operator', name, lambda: _matrix(out, name, qc, record['expected'], controlled=True))
    write_json(Path(out)/'reports'/(name+'.json'), report)
    return report


def _full(payload):
    from . import circuits, rustiq_adapter as adapter
    prepared, route, shared, out = payload
    name = prepared['case']['id']+'_'+route
    journal = Journal(out, name+'_full')
    provider = circuits.BasisProvider(4, prepared['frames'])
    trace = []
    evolution = journal.call('full_build', name, lambda: adapter.build_evolution(
        prepared['spec'], prepared['context'], prepared['macros'], prepared['atoms'], provider,
        input_identity=shared['input_identity'], source_identity=shared['source_identity'],
        route=route, trace=trace))
    _save_circuit(out, name+'_evolution', evolution.circuit)
    report = journal.call('operator', name+'_controlled', lambda: _matrix(
        out, name+'_controlled', evolution.circuit, prepared['expected'], controlled=True))
    write_json(Path(out)/'reports'/(name+'_build.json'), dict(operator=report, trace=trace,
        fingerprint=evolution.fingerprint, basis_cache_entries=len(provider._cache)))
    return dict(name=name, route=route, prepared=prepared, evolution=evolution, trace=trace, operator=report)


def _adjoint(payload):
    from .circuits import actual_adjoint
    item, out = payload
    name = item['name']+'_adjoint'
    journal = Journal(out, name)
    qc = journal.call('adjoint', name, lambda: actual_adjoint(item['evolution']))
    _save_circuit(out, name, qc)
    report = journal.call('operator', name, lambda: _matrix(out, name, qc, item['prepared']['expected'].conj().T))
    write_json(Path(out)/'reports'/(name+'.json'), report)
    return report


def _wrapper(payload):
    from .circuits import hadamard_wrapper
    from . import gates, rustiq_adapter as adapter
    item, axis, out = payload
    name = item['name']+'_'+axis
    journal = Journal(out, name+'_raw')
    qc = journal.call('wrapper_build', name, lambda: hadamard_wrapper(
        item['evolution'], item['prepared']['spec'], axis, measure=True))
    _save_circuit(out, name+'_raw', qc)
    signal = journal.call('raw_signal', name, lambda: gates.wrapper_signal(qc, axis))
    return dict(name=name, parent=item['name'], axis=axis, route=item['route'], circuit=qc,
        identity=adapter.compile_identity(item['evolution'], route=item['route']),
        expected=item['prepared']['wrappers'][axis], expected_signal=item['prepared']['signal'],
        raw_signal=signal)


def _compile(payload):
    from .fixture import CompilerProfile
    from . import gates
    item, out = payload
    name, qc = item['name'], item['circuit']
    journal = Journal(out, name+'_compiled')
    cache = {}
    profile = CompilerProfile()
    cold = journal.call('compile_cold', name, lambda: gates.compiled_wrapper(
        qc, profile, cache=cache, identity_context=item['identity']))
    if len(cache) != 1:
        raise ValueError('Cold compilation must create exactly one cache entry')
    key = next(iter(cache))
    warm = journal.call('compile_warm', name, lambda: gates.compiled_wrapper(
        qc, profile, cache=cache, identity_context=item['identity']))
    if len(cache) != 1 or next(iter(cache)) != key or cold != warm:
        raise ValueError('Warm compilation must be an exact cache hit')
    _save_circuit(out, name+'_compiled', cold)
    operator = journal.call('operator', name+'_compiled', lambda: _matrix(
        out, name+'_compiled', cold.remove_final_measurements(inplace=False), item['expected']))
    signal = journal.call('compiled_signal', name, lambda: gates.wrapper_signal(cold, item['axis']))
    write_json(Path(out)/'reports'/(name+'_compiled.json'), dict(operator=operator,
        signal=signal, cache_key=key, cold_misses=1, warm_hits=1, warm_extra_transpile=0))
    return dict(name=name, parent=item['parent'], axis=item['axis'], route=item['route'],
        circuit=cold, operator=operator, compiled_signal=signal,
        expected_signal=item['expected_signal'], cache_key=key)


def _worker_init(threads):
    # Own processes only; source/lock/packages/shared caches remain untouched.
    os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
        NUMBA_NUM_THREADS='1', QISKIT_NUM_PROCS='1', RAYON_NUM_THREADS=str(threads))


def journal_counts(output):
    counts = {}
    lines = 0
    for path in sorted((Path(output)/'journals').glob('*.jsonl')):
        for line in path.read_text().splitlines():
            lines += 1
            row = json.loads(line)
            if row['status'] == 'COMPLETED':
                counts[row['group']] = counts.get(row['group'], 0)+1
            elif row['status'] == 'FAILED':
                raise RuntimeError('Failed registered request: '+row['id'])
    if lines > 4096:
        raise ValueError('Journal output limit exceeded')
    return counts, lines


def execute(verified):
    from importlib.metadata import version
    if platform.python_version() != verified['contract']['environment']['python']:
        raise ValueError('Python version drift')
    for name, expected in verified['contract']['environment'].items():
        if name in ('qiskit', 'numpy', 'scipy', 'mpmath') and version(name) != expected:
            raise ValueError('Environment drift: '+name)
    out = verified['output']
    out.mkdir(parents=True, exist_ok=False)
    for name in ('journals', 'references', 'operators', 'circuits', 'reports'):
        (out/name).mkdir()
    write_json(out/'approval.json', verified['approval'])
    write_json(out/'preparation_contract.json', verified['contract'])
    write_json(out/'execution_binding.json', dict(source_identity=verified['source_identity'],
        input_identity=verified['input_identity'], retry_policy='User authorizes technical fixes/fresh attempts; no scientific condition or tolerance changes. Preserve failed attempts.'))
    cpus = len(os.sched_getaffinity(0))
    workers = min(24, cpus)  # 24 independent registered wrapper tasks, not an artificial cap
    threads = max(1, cpus//workers)
    resources = dict(affinity_cpus=cpus, load_average=os.getloadavg(), workers=workers,
        thread_rule='BLAS/OpenMP/Numba/Qiskit=1; Rayon=available CPUs // active worker capacity',
        rayon_threads=threads, meminfo=Path('/proc/meminfo').read_text(),
        cpu_pressure=Path('/proc/pressure/cpu').read_text(),
        memory_pressure=Path('/proc/pressure/memory').read_text(),
        wall_cap=None, memory_cap=None, cpu_cap=None)
    write_json(out/'resources.json', resources)
    _worker_init(threads)
    shared = {k: verified[k] for k in ('inputs', 'input_identity', 'source_identity')}
    start, cpu_start = time.perf_counter(), time.process_time()
    journal = Journal(out, 'main')
    from . import gates
    from .f2_execution import circuit_metrics
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=_worker_init, initargs=(threads,)) as pool:
            def stage(name, function, jobs):
                for path, sha in {**verified['contract']['source_files'],
                                  **verified['approval']['execution_files']}.items():
                    if file_sha(verified['root']/path) != sha:
                        raise ValueError('Frozen SOURCE drift: '+path)
                print(name, 'START', len(jobs), flush=True)
                results = list(pool.map(function, jobs, chunksize=1))
                print(name, 'PASS', len(results), flush=True)
                return results
            prepared = stage('REFERENCES', _prepare,
                [(c, shared, str(out)) for c in verified['contract']['cases']])
            local = stage('LOCAL', _local, [(r, route, str(out)) for p in prepared
                for r in p['local'] for route in ('B0_B1_literal', 'R1')])
            full = stage('CONTROLLED', _full, [(p, route, shared, str(out))
                for p in prepared for route in ('B0', 'B1', 'R1')])
            adjoints = stage('ADJOINT', _adjoint, [(item, str(out)) for item in full])
            wrappers = stage('RAW_XY', _wrapper, [(item, axis, str(out)) for item in full for axis in ('X', 'Y')])
            raw_pairs = {}
            for item in full:
                pair = {w['axis']: w['raw_signal'] for w in wrappers if w['parent'] == item['name']}
                raw_pairs[item['name']] = journal.call('signal_pair', item['name']+'_raw',
                    lambda: gates.signal_gate(pair['X'], pair['Y'], item['prepared']['signal']))
            compiled = stage('COMPILED', _compile, [(item, str(out)) for item in wrappers])
            compiled_pairs = {}
            for item in full:
                pair = {w['axis']: w['compiled_signal'] for w in compiled if w['parent'] == item['name']}
                compiled_pairs[item['name']] = journal.call('signal_pair', item['name']+'_compiled',
                    lambda: gates.signal_gate(pair['X'], pair['Y'], item['prepared']['signal']))
            metrics = []
            for item in compiled:
                metric = journal.call('metrics', item['name'], lambda: circuit_metrics(item['circuit']))
                metrics.append(dict(case=item['parent'].rsplit('_', 1)[0], route=item['route'], axis=item['axis'], **metric))
        counts, lines = journal_counts(out)
        expected = dict(motif_restore=4, reference=28, local_build=20, full_build=12,
            adjoint=12, wrapper_build=24, operator=68, raw_signal=24, compile_cold=24,
            compile_warm=24, compiled_signal=24, signal_pair=24, metrics=24)
        if counts != expected:
            raise ValueError('Call accounting mismatch: '+str(counts))
        synthesis = [t for item in full for t in item['trace'] if t['kind'] == 'synthesis']
        if len(synthesis) != 10:
            raise ValueError('Full synthesis count drift')
        total_terms = sum(len(t['terms']) for t in synthesis)*2
        if total_terms != 60:
            raise ValueError('Q13 input triple count drift')
        write_json(out/'metrics.json', metrics)
        write_json(out/'signal_reports.json', dict(raw=raw_pairs, compiled=compiled_pairs))
        summary = dict(status='C4_LIMITED_CORRECTNESS_PASS_RESOURCES_RECORDED_MANDATORY_STOP',
            source_commit=verified['approval']['source_commit'], input_identity=verified['input_identity'],
            case_ids=list(CASE_IDS), counts=counts, journal_lines=lines,
            Rustiq_calls=20, Rustiq_triples=60, transpile_calls=24, cold_misses=24, warm_hits=24,
            local_max=max(r['max_entry'] for r in local), full_max=max(i['operator']['max_entry'] for i in full),
            adjoint_max=max(r['max_entry'] for r in adjoints),
            compiled_max=max(i['operator']['max_entry'] for i in compiled),
            raw_signal_max=max(r['error'] for r in raw_pairs.values()),
            compiled_signal_max=max(r['error'] for r in compiled_pairs.values()),
            wall_seconds=time.perf_counter()-start,
            parent_cpu_seconds=time.process_time()-cpu_start,
            children_cpu_seconds=resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime+
                resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime,
            certified_shots=None, total_round_resource_advantage=False, further_science_authorized=False)
        write_json(out/'summary.json', summary)
        write_json(out/'STOP.json', dict(status='COMPLETED_MANDATORY_STOP', further_science=False))
        return summary
    except BaseException as error:
        try:
            write_json(out/'STOP.json', dict(status='FAILED_ATTEMPT_PRESERVED', error_type=type(error).__name__,
                                            error=str(error), traceback=traceback.format_exc()))
        except FileExistsError:
            pass
        raise
