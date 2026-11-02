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
import csv
import multiprocessing
import os
import platform
import resource
import subprocess
import sys
import time
import traceback

from .fixture import canonical_json, digest

CONTRACT = 'docs/research/research_c_rustiq_limited_contract_20261024.json'
CONTRACT_SHA = 'e948a2e7fdf47e20b8af8cd7debae0631dabbd5e6ad77f419a9a9e2c668319a1'
CASE_IDS = ('SS_motif_2', 'SS_motif_3', 'SS_motif_4', 'PP_motif_0')
B2_SCOPE = 'C4_B2_SAVED_COMPARISON_V1'
B2_CONTRACT = 'docs/research/research_c_angle_fusion_b2_contract_20261024.json'


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_packet(root, packet, inputs, output, *, scope='C4_Q13_LIMITED_V1'):
    """Read-only, standard-library approval and identity gate before science."""
    if scope == B2_SCOPE:
        return _verify_b2_packet(root, packet, inputs, output)
    if scope != 'C4_Q13_LIMITED_V1':
        raise PermissionError('Unknown execution scope')
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


def _saved_payload(path, *, full):
    from .reference import validate_saved_reference
    return validate_saved_reference(json.loads(gzip.decompress(Path(path).read_bytes())), full=full)


def verify_b2_preflight(root, inputs, *, environment=True):
    """Technical bytes/schema checks only. This function grants no science."""
    root, inputs = Path(root).resolve(), Path(inputs).resolve()
    contract = json.loads((root/B2_CONTRACT).read_bytes())
    if (contract['schema'] != B2_SCOPE or contract['scientific_execution_authorized'] is not False
            or tuple(c['id'] for c in contract['cases']) != CASE_IDS
            or len([r for c in contract['cases'] for r in c['target_regions']]) != 10):
        raise ValueError('B2 contract scope drift')
    for name, expected in contract['source_files'].items():
        if file_sha(root/name) != expected:
            raise ValueError('B2 SOURCE drift: '+name)
    old = root/contract['saved_evidence']['root']
    manifest_path = old/'OUTPUT_HASH_MANIFEST.json'
    if file_sha(manifest_path) != contract['saved_evidence']['manifest_sha256']:
        raise ValueError('Saved packet manifest drift')
    manifest = json.loads(manifest_path.read_bytes())
    for name, record in manifest['files'].items():
        path = (old/name).resolve()
        if old not in path.parents or file_sha(path) != record['sha256']:
            raise ValueError('Saved packet byte drift: '+name)
    prior = json.loads((old/'preparation_contract.json').read_bytes())
    if json.loads((old/'summary.json').read_bytes())['status'] != contract['saved_evidence']['status']:
        raise ValueError('Saved correctness qualification drift')
    if (contract['cases'] != prior['cases'] or contract['compiler'] != prior['compiler']
            or contract['model'] != prior['model'] or contract['input_files'] != prior['input_files']
            or contract['tolerances'] != prior['tolerances']):
        raise ValueError('B2 scientific conditions drift')
    for name in contract['unchanged_source_files']:
        if file_sha(root/name) != contract['unchanged_source_files'][name]:
            raise ValueError('Shared builder/compiler/wrapper drift: '+name)
    input_bytes = {}
    for name, expected in contract['input_files'].items():
        data = (inputs/name).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('Input drift: '+name)
        input_bytes[name] = data
    for case in contract['cases']:
        cid = case['id']
        report = json.loads((old/'reports'/(cid+'_input.json')).read_bytes())
        if report['case'] != case or set(report['reference_reports']) != {'90', '128'}:
            raise ValueError('Saved case/reference metadata drift')
        _saved_payload(old/'references'/(cid+'_128.json.gz'), full=True)
        for region in case['target_regions']:
            name = region['id'].replace('/', '_').replace(':', '_')
            if name not in report['local_agreement']:
                raise ValueError('Missing saved precision qualification')
            _saved_payload(old/'references'/(name+'_128.json.gz'), full=False)
    if environment:
        _verify_b2_environment(contract)
    return dict(root=root, old_packet=old, contract=contract, inputs=input_bytes,
        input_identity=digest(contract['input_files']),
        source_identity=digest(contract['source_files']), scope=B2_SCOPE)


def _verify_b2_environment(contract):
    from importlib.metadata import version, distribution
    env = contract['environment']
    if platform.python_version() != env['python']:
        raise ValueError('Python version drift')
    for name, expected in env['package_versions'].items():
        if version(name) != expected:
            raise ValueError('Saved environment version drift: '+name)
    if file_sha(Path(sys.executable).resolve()) != env['python_executable_sha256']:
        raise ValueError('Python binary drift')
    dist = distribution('qiskit')
    for name, expected in env['qiskit_files'].items():
        if file_sha(dist.locate_file(name)) != expected:
            raise ValueError('Qiskit API/native binary drift: '+name)


def _verify_b2_packet(root, packet, inputs, output):
    root, packet, output = (Path(p).resolve() for p in (root, packet, output))
    approval = json.loads((packet/'approval.json').read_bytes())
    if (approval.get('stage') != B2_SCOPE or approval.get('approved') is not True
            or approval.get('authority') != 'USER_EXPLICIT_C4_B2_EXECUTION'
            or approval.get('contract_sha256') != file_sha(root/B2_CONTRACT)):
        raise PermissionError('Fresh explicit B2-only execution approval required')
    if output.exists() or approval['output'] != str(output):
        raise PermissionError('Fresh bound output required; no retry/resume')
    head = subprocess.check_output(['git','rev-parse','HEAD'], cwd=root, text=True).strip()
    if head != approval['source_commit'] or subprocess.check_output(['git','diff','--name-only','HEAD'], cwd=root):
        raise ValueError('Bound B2 publication commit/worktree drift')
    verified = verify_b2_preflight(root, inputs)
    if verified['old_packet'] == output or verified['old_packet'] in output.parents:
        raise PermissionError('Old evidence output is protected')
    verified.update(packet=packet, output=output, approval=approval,
                    source_identity=head+':'+verified['source_identity'])
    return verified


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


def _matrix(out, name, actual, expected, *, controlled=False, literal=None):
    """One Operator materialization, same absolute/ unitarity gates as legacy.

    Unlike controlled_gate's report-only API, save that very matrix without a
    second Operator evaluation. Expected diag(I,U) is independently supplied.
    """
    import numpy as np
    from qiskit.quantum_info import Operator
    from . import gates
    matrix = Operator(actual).data
    report = gates.matrix_gate(matrix, expected, 1e-10, label=name)
    if literal is not None:
        report['before_fusion_literal'] = gates.matrix_gate(matrix, literal, 1e-10,
                                                          label=name+'_BEFORE_FUSION')
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


def _prepare_saved(payload):
    """B2 science-only preparation: saved law/targets, zero new references."""
    import numpy as np
    from .f1_saved_execution import SavedF1
    from . import reference as ref, rustiq_adapter as adapter
    from .prototype import atom_record
    case, shared, out = payload
    cid, arm = case['id'], case['arm']
    old = Path(shared['old_packet'])
    journal = Journal(out, cid+'_saved_targets')
    engine = SavedF1(dict(contract=dict(fixture_bundle_path='handoff/f1_bundle.json'),
                          inputs=shared['inputs']))
    resolved = journal.call('motif_restore', cid, lambda: engine.resolve_case(case))
    if canonical_json(resolved['macros']) != canonical_json(case['macros']):
        raise ValueError('Literal motif drift: '+cid)
    macros, atoms = engine.macros(arm, resolved['macros']), engine.atoms[arm]
    provenance = adapter.route_identity('B2', input_identity=shared['input_identity'],
                                        source_identity=shared['source_identity'])
    plans = tuple(adapter.plan_sequence(e, atoms, 4, provenance=provenance) for e in macros)
    saved = json.loads((old/'reports'/(cid+'_input.json')).read_bytes())
    events = [[e.to_dict() for e in macro] for macro in macros]
    angle_hex = [[o.signed_angle.hex() if o.signed_angle is not None else None
                  for o in p.occurrences] for p in plans]
    if (canonical_json(events) != canonical_json(saved['events'])
            or canonical_json([atom_record(a) for a in atoms]) != canonical_json(saved['atoms'])
            or angle_hex != saved['angle_hex']):
        raise ValueError('Original saved event/angle/atom drift')
    for p, previous in zip(plans, saved['plans'], strict=True):
        current = asdict(p)
        for key in ('occurrences','runs','phases','phase','system_qubits','ancilla'):
            if canonical_json(current[key]) != canonical_json(previous[key]):
                raise ValueError('Original plan drift: '+key)
    target = [dict(id=f'{cid}/macro{mi}/{r.region_id}', macro=mi, region=r.region_id,
                   occurrence_ordinals=[o.ordinal for o in r.occurrences])
              for mi,p in enumerate(plans) for run in p.runs for r in run.regions if r.eligible]
    if target != case['target_regions']:
        raise ValueError('B2 target region inventory drift')
    def restore(name, full):
        value = journal.call('reference_decode', name, lambda: ref.restore_saved_reference(
            _saved_payload(old/'references'/(name+'_128.json.gz'), full=full), full=full))
        with (Path(out)/'references'/(name+'_128.json.gz')).open('xb') as stream:
            stream.write((old/'references'/(name+'_128.json.gz')).read_bytes())
        return value
    full = restore(cid, True)
    local = []
    for item in target:
        mi = item['macro']
        region = next(r for run in plans[mi].runs for r in run.regions if r.region_id == item['region'])
        name = item['id'].replace('/', '_').replace(':', '_')
        value = restore(name, False)
        expected = journal.call('target_matrix', name, lambda: ref.to_numpy(value['controlled']))
        literal = journal.call('literal_load', name, lambda: np.load(
            old/'operators'/(name+'_B0_B1_literal.npy'), allow_pickle=False))
        if literal.shape != (32,32) or not np.isfinite(literal).all():
            raise ValueError('Saved literal shape/nonfinite')
        local.append(dict(name=name, region=region, plan=plans[mi], case_id=cid,
            macro=mi, expected=expected, literal=literal, agreement=saved['local_agreement'][name]))
    write_json(Path(out)/'reports'/(cid+'_input.json'), dict(case=case, events=events,
        atoms=[atom_record(a) for a in atoms], plans=[asdict(p) for p in plans], angle_hex=angle_hex,
        reused_precision_reports=saved['reference_reports'], local_agreement=saved['local_agreement']))
    expected = journal.call('target_matrix', cid+'_controlled', lambda: ref.to_numpy(
        ref.controlled_operator(full['ctx'], full['operator'])))
    wrappers = {a: journal.call('target_matrix', cid+'_wrapper_'+a,
        lambda a=a: ref.to_numpy(full['wrapper_'+a])) for a in ('X','Y')}
    signal = journal.call('target_signal', cid, lambda: complex(full['signal']))
    return dict(case=case, spec=engine.spec, context=engine.context, atoms=atoms, macros=macros,
        frames=engine.provider.frames, local=local, expected=expected, wrappers=wrappers, signal=signal)


def _local(payload):
    from qiskit import QuantumCircuit
    from . import circuits, rustiq_adapter as adapter
    record, route, out = payload
    name = record['name']+'_'+route
    journal = Journal(out, name)
    ledger = []
    def build():
        if route == 'R1':
            network = adapter.region_network(record['region'], ancilla=4)
            return adapter._q13(5, network, adapter.Options())
        qc = QuantumCircuit(5)
        if route == 'B2':
            groups = adapter.fusion_groups(record['plan'], record['region'],
                case_id=record['case_id'], macro_index=record['macro'])
            ledger.extend(g.ledger for g in groups)
            for group in groups:
                circuits.append_central(qc, group.support, 4, group.signed_angle)
            return qc
        for occurrence in record['region'].occurrences:
            circuits.append_central(qc, occurrence.support, 4, occurrence.signed_angle)
        return qc
    qc = journal.call('local_build', name, build)
    _save_circuit(out, name, qc)
    report = journal.call('operator', name, lambda: _matrix(out, name, qc, record['expected'],
        controlled=True, **(dict(literal=record['literal']) if route == 'B2' else {})))
    if route == 'B2':
        report['fusion_ledger'] = ledger
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
        route=route, trace=trace,
        **(dict(case_id=prepared['case']['id']) if route == 'B2' else {})))
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


def _raw_diagnostic(payload):
    from qiskit import qpy
    from .f2_execution import circuit_metrics
    item, old_packet, out = payload
    cid = item['parent'].rsplit('_', 1)[0]
    journal = Journal(out, item['name']+'_raw_metrics')
    def load():
        with (Path(old_packet)/'circuits'/(cid+'_B1_'+item['axis']+'_raw.qpy')).open('rb') as stream:
            circuits = qpy.load(stream)
        if len(circuits) != 1 or circuits[0].num_qubits != 5:
            raise ValueError('Saved B1 raw wrapper inventory/register drift')
        return circuits[0]
    previous = journal.call('saved_raw_qpy_decode', item['name'], load)
    old_metric = journal.call('raw_metrics', item['name']+'_B1', lambda: circuit_metrics(previous))
    new_metric = journal.call('raw_metrics', item['name']+'_B2', lambda: circuit_metrics(item['circuit']))
    if old_metric['measurement_count'] != 1 or new_metric['measurement_count'] != 1:
        raise ValueError('Raw wrapper measurement drift')
    return dict(case=cid, axis=item['axis'], B1=old_metric, B2=new_metric)


def paired_metric_rows(saved, new, fields):
    """Science-only comparison on real evidence; preparation uses toy rows.

    Preserve case/axis correspondence; no sums, ranking, weights or inferred
    timings. Keep old/new native integer vectors side by side for the design's
    GO/STOP decision after the separately authorized correctness run.
    """
    by_key = {}
    for row in [*saved, *new]:
        key = (row['case'],row['axis'],row['route'])
        if key in by_key:
            raise ValueError('Duplicate compiled metric record')
        metric = {}
        for field in fields:
            raw = row[field]
            if (type(raw) is not int and not (isinstance(raw,str) and raw.isascii() and raw.isdecimal())):
                raise ValueError('Integer compiled metric required: '+field)
            metric[field] = int(raw)
            if metric[field] < 0:
                raise ValueError('Negative compiled metric')
        if metric['measurement_count'] != 1:
            raise ValueError('Full measured wrapper required')
        by_key[key] = metric
    if set(by_key) != {(cid,a,r) for cid in CASE_IDS for a in ('X','Y') for r in ('B0','B1','R1','B2')}:
        raise ValueError('Compiled metric correspondence drift')
    return [dict(case=cid,axis=a,metrics={r:by_key[(cid,a,r)] for r in ('B0','B1','R1','B2')})
            for cid in CASE_IDS for a in ('X','Y')]


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
    b2 = verified.get('scope') == B2_SCOPE
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
        input_identity=verified['input_identity'],
        retry_policy=('No retry/resume; fresh separate user approval required.' if b2 else
        'User authorizes technical fixes/fresh attempts; no scientific condition or tolerance changes. Preserve failed attempts.')))
    if b2:
        with (out/'source_diff.patch').open('xb') as stream:
            stream.write(subprocess.check_output(['git','diff',verified['contract']['design_commit'],
                'HEAD','--',*verified['contract']['changed_source_files']], cwd=verified['root']))
    cpus = len(os.sched_getaffinity(0))
    workers = min(10 if b2 else 24, cpus)  # maximum independent stage tasks, not an artificial cap
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
    if b2:
        shared['old_packet'] = str(verified['old_packet'])
    start, cpu_start = time.perf_counter(), time.process_time()
    journal = Journal(out, 'main')
    from . import gates
    from .f2_execution import circuit_metrics
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=_worker_init, initargs=(threads,)) as pool:
            def stage(name, function, jobs):
                for path, sha in {**verified['contract']['source_files'],
                                  **verified['approval'].get('execution_files', {})}.items():
                    if file_sha(verified['root']/path) != sha:
                        raise ValueError('Frozen SOURCE drift: '+path)
                print(name, 'START', len(jobs), flush=True)
                results = list(pool.map(function, jobs, chunksize=1))
                print(name, 'PASS', len(results), flush=True)
                return results
            prepared = stage('SAVED_TARGETS' if b2 else 'REFERENCES', _prepare_saved if b2 else _prepare,
                [(c, shared, str(out)) for c in verified['contract']['cases']])
            local = stage('LOCAL', _local, [(r, route, str(out)) for p in prepared
                for r in p['local'] for route in (('B2',) if b2 else ('B0_B1_literal', 'R1'))])
            full = stage('CONTROLLED', _full, [(p, route, shared, str(out))
                for p in prepared for route in (('B2',) if b2 else ('B0', 'B1', 'R1'))])
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
            raw_diagnosis = stage('RAW_IR_DIAGNOSIS', _raw_diagnostic,
                [(item, str(verified['old_packet']), str(out)) for item in wrappers]) if b2 else None
        counts, lines = journal_counts(out)
        expected = dict(motif_restore=4, reference=28, local_build=20, full_build=12,
            adjoint=12, wrapper_build=24, operator=68, raw_signal=24, compile_cold=24,
            compile_warm=24, compiled_signal=24, signal_pair=24, metrics=24)
        if b2:
            expected = verified['contract']['journal_counts']
        if counts != expected:
            raise ValueError('Call accounting mismatch: '+str(counts))
        synthesis = [t for item in full for t in item['trace'] if t['kind'] == 'synthesis']
        if b2:
            if synthesis or len(local) != 10 or len(full) != 4 or len(compiled) != 8:
                raise ValueError('B2 route/call inventory drift')
            if (sum('before_fusion_literal' in r for r in local) != 10
                    or sum('unitarity' in r for r in local) != 10
                    or sum('unitarity' in i['operator'] for i in full) != 4):
                raise ValueError('B2 absolute/ before-after /unitarity comparison count drift')
            fusion = [t for item in full for t in item['trace'] if t['kind'] == 'fusion']
            if len(fusion) != 10:
                raise ValueError('Full fusion region inventory drift')
            write_json(out/'raw_ir_diagnosis.json', raw_diagnosis)
            with (verified['old_packet']/'metrics.csv').open() as stream:
                saved_metrics = list(csv.DictReader(stream))
            keys = {(row['case'], row['route'], row['axis']) for row in saved_metrics}
            if keys != {(cid,r,a) for cid in CASE_IDS for r in ('B0','B1','R1') for a in ('X','Y')} or len(saved_metrics) != 24:
                raise ValueError('Saved compiled metric inventory drift')
            write_json(out/'saved_compiled_metrics.json', saved_metrics)
            write_json(out/'paired_compiled_metrics.json', paired_metric_rows(
                saved_metrics, metrics, verified['contract']['resources']['metrics']))
            write_json(out/'metrics.json', metrics)
            write_json(out/'signal_reports.json', dict(raw=raw_pairs, compiled=compiled_pairs))
            write_json(out/'provenance.json', dict(source_commit=verified['approval']['source_commit'],
                source_files=verified['contract']['source_files'], input_files=verified['contract']['input_files'],
                parent_evidence=verified['contract']['saved_evidence'], environment=verified['contract']['environment'],
                compiler=verified['contract']['compiler'], fusion_policy=verified['contract']['fusion_policy'],
                local_evidence='New local execution, not CI or independent external reproduction'))
            summary = dict(status='C4_B2_CORRECTNESS_PASS_RESOURCES_RECORDED_MANDATORY_STOP',
                source_commit=verified['approval']['source_commit'], counts=counts, journal_lines=lines,
                operator_comparisons=36, unitarity_checks=14, new_references=0,
                Rustiq_calls=0, transpile_calls=8, cold_misses=8, warm_hits=8,
                further_science_authorized=False, wall_seconds=time.perf_counter()-start,
                parent_cpu_seconds=time.process_time()-cpu_start,
                children_cpu_seconds=resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime+
                    resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime)
            write_json(out/'summary.json', summary)
            write_json(out/'STOP.json', dict(status='COMPLETED_MANDATORY_STOP', further_science=False))
            return summary
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
