"""P-1: Full-H kappa2 m3 (q8/R71/K2) wiring path with the CX.CRZ.CX controlled ZZ.

One fresh normal path (index 0), X/Y full measured wrappers, at most 2 compiles.
Reuses the ORIGINAL Full-H setup/context/prepare_full_h/full_oracle/trial_seed
(9b83, executed from its Git blob), the long-time native sample/compose/
cost_trial/compile cache and wrapper, and D's reversible builder swap (01e068af).

Two processes, as in the original Full-H studies: the parent does metadata,
SOURCE checks and C-guard sidecars in this tree; the child calls the original
`install()` FIRST so trotterlib/trottertracks come from the pinned provider
root (c31ef424 bytes, checked), then runs cost_trial under the swap.

Default mode is a metadata-only preflight. Only `--execute-approved-p1`
restores the saved H4 input, samples, builds and compiles (separate approval).
No retry, mandatory STOP. This module imports no project package at import time.

run02 (A, 2026-11-14): run01 STOPPED before any science because the execution
checkout lacked the tracked `references/` file that the child imports. Changes
from the fixed SOURCE 04bead115 are the output label (run01 is kept) and a
child import-only preflight (install() first, then the real-input import chain
and the files context() reads); scientific conditions are unchanged.

Scalar/event phase gate (A, 2026-11-14, after the run02 STOP): the old
`scalar_once` ("exactly one ancilla p") cannot tell the scalar phase from the
per-request event phase that the pinned provider builder appends (c31ef424
`_build`: one PhaseGate(sum of event/identity/product phases) at the end of a
request tail iff the sum != 0). `scalar_event_phase_gate` replaces it in the ok
decision: the ordered list of ancilla-touching instructions must equal the one
derived from the sampled events by the provider's own rule (products -> cz,
rotation -> crz (new ZZ: target s1), event phase -> p, scalar p(-scalar*T) last),
with exact float equality as in `instruction_parity`. `scalar_once` is kept only
as a diagnostic. The price records use B's kinds (DIRECT_COMPILED_FINITE_SAMPLE /
FINITE_COHORT_ONLY, n_paths=1, law_expectation=False).
"""
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'scripts') not in sys.path:
    sys.path.insert(0, str(ROOT / 'scripts'))

CALLER = 'scripts/run_full_h_kappa2_p1.py'
M = 3
COHORT = 'D_FULL_H_KAPPA2_NEW_ZZ_M3_WIRING_V1'
FULL_H = dict(commit='9b83b96b83f62142c9d87858dc24a94db7ac5d61', path='scripts/h46_full_h_pure_rte.py')
NEW_ZZ = dict(commit='01e068af7e06d9f1f5621a5b3800c30eb32ecb06', path='src/trottertracks/efficient_controlled_zz.py')
NATIVE = dict(commit='af284d853890155bacd19e61850c196bd1d5c660', path='scripts/h46_long_time_native.py')
WRAPPER = dict(commit=NATIVE['commit'], path='src/trottertracks/resource_applicability/ax2b_native_df_v5.py')
PROVIDER = dict(commit='c31ef424175c02379eafebe8fc6c7548c1d9aabe', path='src/trotterlib/df_rte_qiskit.py')
PROVIDER_MODULES = ('trotterlib.rte', 'trotterlib.df_partial_s2', 'trotterlib.df_rte_qiskit', 'trotterlib.rte_compiled_cost',
                    'trotterlib.df_rte_circuit', 'trotterlib.df_rte_tail',
                    'trottertracks.resource_applicability.ax2b_native_df_v5',
                    'trottertracks.resource_applicability.ax2b_stream_fingerprint_v5',
                    'trottertracks.resource_applicability.h6_hash_reuse_v1')
NATIVE_SCRIPTS = ('scripts/h46_long_time_native.py', 'scripts/h46_long_time_signal.py', 'scripts/h46_new2_native.py',
                  'scripts/h46_m5_contract.py', 'scripts/h46_pr_pilot_repair.py', 'scripts/h46_b3_fast_fingerprint.py')
LOWERING_ID = 'CONTROLLED_ZZ_CX_CRZ_CX_V1'
SCALAR_QUANTITY = 'HAMILTONIAN_CONSTANT_PLUS_EXTRACTED_FULL_H_IDENTITY'
RUN = 'run02'  # run01 (STOP_WIRING_CHECK before science) is preserved unchanged
OUTPUT = 'docs/research/d_full_h_kappa2_p1_results_20261114/' + RUN
PRIVATE_OUTPUT = Path('/home/AbeHiromu/projects/d-full-h-kappa-20261114-private/P1-KAPPA2-M3') / RUN
# Modules that setup()/context()/prepare_full_h/full_oracle/cost_trial import (many lazily) in the child.
CHILD_IMPORTS = ('numpy', 'scipy.linalg', 'qiskit', 'qiskit.qpy', 'qiskit.circuit.library',
                 'h46_m5_contract', 'h46_m5_tol_contract', 'h46_m5_tol_native', 'h46_m5_tol_saved_input',
                 'h46_new2_core', 'h46_new2_native', 'h46_pr_pilot_repair', 'h46_long_time_native',
                 'h46_b3_fast_fingerprint', 'trotterlib.rte', 'trotterlib.rte_compiled_cost', 'trotterlib.df_hamiltonian',
                 'trotterlib.df_rte_tail', 'trotterlib.df_trotter.decompose', 'trotterlib.df_trotter.ops',
                 'trotterlib.df_trotter.circuit', 'trotterlib.pr2_matched_accuracy_m1_execution',
                 'trottertracks.resource_applicability.h6_hash_reuse_v1')
SCOPE = dict(cells=1, normal_paths=1, index=0, axes=['X', 'Y'], compile_maximum=2, extra_uncompiled_legacy_compose=1,
             new_reference=0, retry=False, N=None, mean=None, G=None)


def blob(commit, path):
    return subprocess.check_output(['git', 'show', commit + ':' + path], cwd=ROOT)


def _meta():
    """This tree's C/I/S1 for metadata (parent process only)."""
    if str(ROOT / 'src') not in sys.path:
        sys.path.insert(0, str(ROOT / 'src'))
    from trottertracks import baseline_signal_pipeline as C
    from trottertracks import rpe_optimization_pipeline as I
    import run_full_h_kappa2_signals as S1
    return C, I, S1


def load_blob_module(d, name):
    module = types.ModuleType(name); module.__file__ = str(ROOT / d['path'])
    sys.modules[name] = module
    exec(compile(blob(d['commit'], d['path']), d['commit'] + ':' + d['path'], 'exec'), module.__dict__)
    return module


def full_h_module(public, private):
    """Original Full-H study code from its Git blob; only PUBLIC/PRIVATE are redirected."""
    full = sys.modules.get('h46_full_h_pure_rte')
    if full is None or getattr(full, '__d_source__', None) != FULL_H:
        full = load_blob_module(FULL_H, 'h46_full_h_pure_rte'); full.__d_source__ = FULL_H
    full.PUBLIC, full.PRIVATE = Path(public), Path(private)
    full.CONTEXT.clear()
    return full


def zz_module():
    return sys.modules.get('_d_efficient_controlled_zz_01e068af') or load_blob_module(NEW_ZZ, '_d_efficient_controlled_zz_01e068af')


def seed_function():
    import ast
    tree = ast.parse(blob(FULL_H['commit'], FULL_H['path']))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('digest', 'trial_seed')]
    env = dict(hashlib=hashlib, json=json)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), FULL_H['path'] + ':seed_only', 'exec'), env)
    return env['trial_seed']


def plan_row(root=ROOT, prepared=None):
    """m3 physical request from the fixed kappa2 plan, with the pinned wiring seeds (parent)."""
    C, I, S1 = _meta()
    prepared = prepared or S1.prepare(root, dict(commit='P1_METADATA_ONLY', caller=CALLER,
                                                 numerical_source=S1.NUMERICAL_SOURCE))
    req = prepared['plan']['pending']['requests'][M]
    cell = S1.read(S1.CANDIDATES)['proposed_wiring_cell']
    c = deepcopy(req['physical_candidate'])
    if (c['id'] != cell['physical_candidate_id'] or (c['q'], c['R_outer'], c['s'], c['K']) != (8, 71, 568, 2)
            or c['T'] != .8 or cell['cohort'] != COHORT or cell['lowering']['lowering_id'] != LOWERING_ID):
        raise ValueError('P1 wiring cell differs from the fixed kappa2 plan')
    seeds = [seed_function()(c, COHORT, 0, outer, 0) for outer in range(c['q'])]
    if seeds != cell['outer_piece_seeds']:
        raise ValueError('Original Full-H trial_seed no longer reproduces the fixed seeds')
    return prepared, req, c, seeds


def source_checks(root=ROOT):
    """Executed native/wrapper/provider files in this tree equal the pinned SOURCE blobs."""
    for d in (NATIVE, WRAPPER, PROVIDER):
        if blob(d['commit'], d['path']) != (Path(root) / d['path']).read_bytes():
            raise ValueError('Pinned native/provider SOURCE differs: ' + d['path'])
    for path in NATIVE_SCRIPTS:
        if blob(NATIVE['commit'], path) != (Path(root) / path).read_bytes():
            raise ValueError('Native script differs from the pinned carrier: ' + path)
    return True


def provider_checks():
    """Child: imported provider modules (from install()'s root) equal c31ef424 blobs."""
    files = {}
    for name in PROVIDER_MODULES:
        module = importlib.import_module(name)
        relative = 'src/' + name.replace('.', '/') + '.py'
        data = Path(module.__file__).read_bytes()
        if data != blob(PROVIDER['commit'], relative):
            raise ValueError('Provider SOURCE mismatch: ' + name)
        files[name] = dict(file=str(module.__file__), sha256=hashlib.sha256(data).hexdigest())
    return files


def op_counts(circuit):
    counts = {}
    for item in circuit.data:
        counts[item.operation.name] = counts.get(item.operation.name, 0) + 1
    return counts


def scalar_once(circuit, n, expected_angle):
    """Legacy diagnostic only (false negative with order>=2 event phases; see module docstring)."""
    phases = [i for i in circuit.data if i.operation.name == 'p'
              and [circuit.find_bit(q).index for q in i.qubits] == [n]]
    return len(phases) == 1 and float(phases[0].operation.params[0]) == expected_angle


PHASE_GATE_ID = 'SCALAR_EVENT_PHASE_ANCILLA_SIGNATURE_V1'
RECORD_SCHEMA = 'D_PRICE_RECORD_B_KINDS_V2'
FINITE_SAMPLE = dict(classification='DIRECT_COMPILED_FINITE_SAMPLE', expectation='FINITE_COHORT_ONLY')


def _event_view(event):
    """RTEEvent.to_dict() (live or JSON-saved) -> the fields the c31ef424 builder reads."""
    apps = [types.SimpleNamespace(role=a['role'], is_identity=bool(a['is_identity']),
                                  coefficient_sign=int(a['coefficient_sign']),
                                  support=None if a['diagonal_pauli_support'] is None
                                  else tuple(int(q) for q in a['diagonal_pauli_support']))
            for a in event['application_sequence']]
    angle = float(event['rotation_angle'])
    return types.SimpleNamespace(phase=complex(event['phase']['real'], event['phase']['imag']), apps=apps,
                                 taylor_order=int(event['taylor_order']),
                                 unsigned_rotation_angle=apps[-1].coefficient_sign * angle)  # RTEEvent property


def central_phase(app):
    """Phase returned by c31ef424 `_append_central_application` (01e068af returns the same 0.0 for ZZ)."""
    if app.role == 'product':
        return math.pi if app.coefficient_sign == -1 else 0.0
    if app.role == 'rotation':
        return 0.0
    raise ValueError('Unsupported RTE application role: ' + str(app.role))


def request_phase(events, provider):
    """Accumulated ancilla phase of one request tail, in the order of c31ef424 `_build`:
    all event phases first (provider `_event_phase_angle`), then per application the
    identity phase (provider `_identity_application_phase`) or the central return value."""
    views = [_event_view(e) for e in events]
    builder = provider.QiskitDFRTEEventCircuitBuilder
    total = 0.0
    for v in views:
        total += provider._event_phase_angle(v.phase)
    for v in views:
        for a in v.apps:
            total += builder._identity_application_phase(v, a) if a.is_identity else central_phase(a)
    return total


def expected_ancilla_ops(requests_events, n, scalar, T, provider, lowering=LOWERING_ID):
    """Ordered ancilla-touching instructions that compose() + the provider builder emit."""
    ops = []
    for r, events in enumerate(requests_events):
        for k, event in enumerate(events):
            v = _event_view(event)
            for a in v.apps:
                if a.is_identity:
                    continue
                if a.role == 'product':
                    ops += [dict(kind='product', request=r, event=k, op=('cz', (n, q), ())) for q in a.support]
                    continue
                angle = 2.0 * v.unsigned_rotation_angle
                if len(a.support) == 1:
                    op = ('crz', (n, a.support[0]), (angle,))
                elif lowering == LOWERING_ID:
                    op = ('crz', (n, a.support[1]), (angle,))
                else:
                    op = ('crzz', (n,) + a.support, (angle,))
                ops.append(dict(kind='rotation', request=r, event=k, op=op))
        phase = request_phase(events, provider)
        if phase != 0.0:
            ops.append(dict(kind='event_phase', request=r, op=('p', (n,), (phase,)),
                            minus_one_events=[k for k, e in enumerate(events) if _event_view(e).phase == -1]))
    if scalar * T != 0:
        ops.append(dict(kind='scalar', op=('p', (n,), (-scalar * T,))))
    return ops


def ancilla_ops(circuit, n, start=0, stop=None):
    stop = len(circuit.data) if stop is None else stop
    out = []
    for i in range(start, stop):
        item = circuit.data[i]
        qubits = tuple(circuit.find_bit(q).index for q in item.qubits)
        if n in qubits:
            out.append(dict(index=i, op=(item.operation.name, qubits, tuple(float(p) for p in item.operation.params))))
    return out


def scalar_event_phase_gate(circuit, n, requests_events, scalar, expected_scalar, T, *, provider,
                            lowering=LOWERING_ID, start=0, stop=None):
    """Scalar vs event phases on the ancilla of one controlled evolution [start, stop).

    PASS iff (1) the scalar equals the preparation value, (2) the ancilla-touching
    instruction list equals expected_ancilla_ops (so each event phase sits after the
    last event of its own request and before the first ancilla instruction of the next,
    and nothing is missing, duplicated, reordered or moved to another request), and
    (3) the scalar p is the last instruction of the evolution (compose appends it last).
    Placement relative to system-only gates inside one request gap is not resolved;
    those gates act on other qubits and commute with an ancilla phase."""
    stop = len(circuit.data) if stop is None else stop
    expected = expected_ancilla_ops(requests_events, n, scalar, T, provider, lowering)
    actual = ancilla_ops(circuit, n, start, stop)
    reasons = []
    if scalar != expected_scalar:
        reasons.append('SCALAR_VALUE_DIFFERS_FROM_PREPARATION')
    got, want = [a['op'] for a in actual], [e['op'] for e in expected]
    first = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), None if len(got) == len(want) else min(len(got), len(want)))
    if first is not None:
        reasons.append('ANCILLA_SIGNATURE_DIFFERS')
    scalar_angle = -scalar * T
    if not expected or expected[-1]['kind'] != 'scalar' or not actual or actual[-1]['index'] != stop - 1:
        reasons.append('SCALAR_NOT_LAST_INSTRUCTION_OF_EVOLUTION')
    phases = [dict(request=e['request'], angle=e['op'][2][0], minus_one_events=e['minus_one_events'],
                   index=a['index'] if a['op'] == e['op'] else None)
              for e, a in zip(expected, actual) if e['kind'] == 'event_phase']
    p_ops = [a for a in actual if a['op'][0] == 'p']
    return dict(gate=PHASE_GATE_ID, status='FAIL' if reasons else 'PASS', reasons=reasons,
                first_mismatch=None if first is None else dict(position=first, actual=got[first] if first < len(got) else None,
                                                               expected=want[first] if first < len(want) else None),
                scalar=scalar, expected_scalar=expected_scalar, scalar_angle=scalar_angle,
                scalar_index=actual[-1]['index'] if actual and actual[-1]['op'] == ('p', (n,), (scalar_angle,)) else None,
                evolution_bounds=[start, stop], ancilla_instructions=len(actual), expected_ancilla_instructions=len(expected),
                ancilla_p=len(p_ops), event_phases=phases, requests=len(requests_events),
                rule='c31ef424 _build: one PhaseGate(sum) per request tail iff sum != 0; compose: p(-scalar*T) last',
                legacy_scalar_once=len(p_ops) == 1 and p_ops[0]['op'][2] == (scalar_angle,))


def wrapper_evolution_bounds(wrapper, n, axis):
    """[start, stop) of U inside the af284d85 measured wrapper h(n) U [sdg(n)] h(n) measure(n)."""
    def op(i):
        item = wrapper.data[i]
        return item.operation.name, tuple(wrapper.find_bit(q).index for q in item.qubits)
    suffix = [('sdg', (n,))] if axis == 'sine' else []
    suffix += [('h', (n,)), ('measure', (n,))]
    if op(0) != ('h', (n,)) or [op(i) for i in range(len(wrapper.data) - len(suffix), len(wrapper.data))] != suffix:
        raise ValueError('Wrapper is not h(n) U [sdg(n)] h(n) measure(n) for axis ' + axis)
    return 1, len(wrapper.data) - len(suffix)


def xy_phase_gate(wrappers, n, requests_events, scalar, expected_scalar, T, *, provider, lowering=LOWERING_ID):
    """Apply the phase gate to each measured X/Y wrapper and require the same controlled U in both."""
    per_axis, segments = {}, {}
    for axis, wrapper in wrappers.items():
        start, stop = wrapper_evolution_bounds(wrapper, n, axis)
        per_axis[axis] = scalar_event_phase_gate(wrapper, n, requests_events, scalar, expected_scalar, T,
                                                 provider=provider, lowering=lowering, start=start, stop=stop)
        segments[axis] = [(i.operation.name, tuple(wrapper.find_bit(q).index for q in i.qubits),
                           tuple(float(p) for p in i.operation.params)) for i in wrapper.data[start:stop]]
    same = len({json.dumps(s) for s in segments.values()}) == 1
    ok = same and set(per_axis) == {'cosine', 'sine'} and all(g['status'] == 'PASS' for g in per_axis.values())
    return dict(gate=PHASE_GATE_ID, status='PASS' if ok else 'FAIL', same_controlled_evolution=same,
                evolution_instructions={a: len(s) for a, s in segments.items()}, axes=per_axis)


def pilot(full, row, *, compile_cached=None):
    """setup -> context -> cost_trial(index0) under the swap; legacy replay for parity."""
    native = full.setup()
    ctx = full.context(row['system'])
    native.CONTEXT[row['system']] = ctx
    if compile_cached is not None:
        native.compile_cached = compile_cached
    provider = importlib.import_module('trotterlib.df_rte_qiskit')
    zz = zz_module(); captured = {}
    original_compose = native.compose

    def spy(row_, prep, requests, scalar, fingerprint):
        evolution = original_compose(row_, prep, requests, scalar, fingerprint)
        captured.update(requests=requests, prep=prep, scalar=scalar, circuit=evolution.circuit,
                        builder=getattr(provider.QiskitDFRTEEventCircuitBuilder, 'lowering_id', None))
        return evolution

    native.compose = spy
    try:
        with zz.efficient_controlled_zz(provider):
            trial = native.cost_trial((row, COHORT, 0))
    finally:
        native.compose = original_compose
    if getattr(provider.QiskitDFRTEEventCircuitBuilder, 'lowering_id', None) is not None:
        raise ValueError('Builder swap was not restored')
    legacy = original_compose(row, captured['prep'], captured['requests'], captured['scalar'], lambda qc: 'LEGACY_REPLAY')
    n = captured['prep'].num_system_qubits
    counts = op_counts(captured['circuit'])
    parity = zz.instruction_parity(legacy.circuit, captured['circuit'])
    one_ids = {c.component_id for c in captured['prep'].rte_preparation.symbolic_tail.components
               if c.df_fragment_id == 'full-h-one-body'}
    one_body = sum(app.component_id in one_ids for r in captured['requests']
                   for e in r.rte_occurrence.events for app in e.application_sequence)
    expected_scalar = ctx.ham.constant + captured['prep'].extracted_identity_coefficient
    events = [[e.to_dict() for e in r.rte_occurrence.events] for r in captured['requests']]
    phase_gate = scalar_event_phase_gate(captured['circuit'], n, events, captured['scalar'], expected_scalar, row['T'],
                                         provider=provider)
    checks = dict(trial_gates=[c['check'] for c in trial['checks']], builder_lowering_id=captured['builder'],
        legacy_rzz_control_instructions=counts.get('crzz', 0), new_crz=counts.get('crz', 0), parity=parity,
        one_body_applications=one_body, scalar=captured['scalar'], scalar_expected=expected_scalar,
        scalar_event_phases=phase_gate,
        scalar_once_legacy_diagnostic=scalar_once(captured['circuit'], n, -captured['scalar'] * row['T']),
        provider_file=str(provider.__file__))
    ok = (checks['builder_lowering_id'] == LOWERING_ID and checks['legacy_rzz_control_instructions'] == 0
          and parity['status'] == 'PASS' and parity['replaced_controlled_zz'] > 0 and one_body > 0
          and checks['scalar'] == expected_scalar and phase_gate['status'] == 'PASS'
          and trial['status'] == 'PASS' and trial['index'] == 0 and set(trial['prices']) == {'cosine', 'sine'})
    return trial, checks, ok


def child(job_path):
    """Child process: original install() first, provider check, pilot, JSON result."""
    job = json.loads(Path(job_path).read_text())
    import h46_long_time_signal as signal
    signal.install()
    if any(name.split('.')[0] == 'trottertracks' and name != 'trottertracks' for name in list(sys.modules)):
        raise ValueError('Project trottertracks imported before the provider root')
    files = provider_checks()
    full = full_h_module(job['public'], job['private'])
    trial, checks, ok = pilot(full, job['row'])
    checks['provider_modules'] = files
    Path(job['result']).write_text(json.dumps(dict(trial=trial, checks=checks, ok=ok), indent=2, default=str))
    return 0 if ok else 2


def import_preflight(modules=CHILD_IMPORTS, system='H4'):
    """Child, no science: install() first and provider bytes as in child(), then import-only of the real-input
    chain and presence/SHA-256 of the files context() reads. No restore, sampling, circuit or compile."""
    import h46_long_time_signal as signal
    signal.install()
    if any(name.split('.')[0] == 'trottertracks' and name != 'trottertracks' for name in list(sys.modules)):
        raise ValueError('Project trottertracks imported before the provider root')
    provider_checks()
    failed = {}
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception as error:
            failed[name] = type(error).__name__ + ': ' + str(error)
    missing = []
    for name in ('h46_m5_contract', 'h46_m5_tol_contract'):
        if name in sys.modules:
            missing += [p for p in sys.modules[name].SOURCE_PATHS if not (ROOT / p).exists() and p not in missing]
    info = signal.origin_config()['systems'][system]
    files = {key: Path(item['path']).exists() and signal.sha(item['path']) == item['sha256']
             for key, item in info['inputs'].items()}
    base = load_blob_module(FULL_H, '_p1_import_preflight_full_h').BASE_PRIVATE / (system + '_context.json')
    files['saved_context_binding'] = base.exists() and signal.read(base)['binding'] == info['binding']
    reference = Path(info['reference_root']) / 'reference.npz'
    receipt = reference.parent / 'receipt.json'
    files['reference_bytes'] = (reference.exists() and receipt.exists()
                                and signal.sha(reference) == signal.read(receipt)['reference_sha256'])
    ok = not failed and not missing and all(files.values())
    return dict(status='PASS' if ok else 'FAIL', modules=len(modules), failed_imports=failed,
                missing_source_paths=missing, input_checks=files)


def child_import_preflight(root=ROOT):
    """Parent: run import_preflight() in a fresh child process (same interpreter and order as the real child)."""
    out = subprocess.run([sys.executable, '-B', str(Path(root) / CALLER), '--child-import-preflight'],
                         cwd=root, capture_output=True, text=True)
    try:
        report = json.loads(out.stdout)
    except ValueError:
        report = dict(status='FAIL', exit=out.returncode, stderr=out.stderr[-3000:])
    if out.returncode != 0 or report.get('status') != 'PASS':
        raise ValueError('Child import preflight failed: ' + json.dumps(report))
    return report


def price_bindings(root, prepared, trial, *, price_source, evidence_commit, evidence_path):
    """Direct single-path records + C guard sidecars/proofs (not a law mean)."""
    C, I, S1 = _meta()
    profile = prepared['plan']['configuration']['spec']['profile']
    base = [r for r in I.price_requests(root, prepared['plan']) if r['m'] == M]
    row = prepared['plan']['configuration']['rounds'][M]
    prices, sides, proofs = [], [], {}
    for req in base:
        axis = req['axis']; metrics = trial['prices']['cosine' if axis == 'X' else 'sine']
        price = deepcopy(req)
        price.update(**FINITE_SAMPLE, cohort=COHORT, n_paths=1, law_expectation=False,
            law_context=I.providers(str(root)).B.p.measurement(row, axis, saved_sampler=True)['law_context'],
            mean=dict(rz_count=metrics['rz_count']), metrics=deepcopy(metrics),
            price_status='DIRECT_FINITE_COHORT', record_schema=RECORD_SCHEMA,
            diagnostic=dict(pilot_role='DIRECT_SINGLE_PATH_WIRING', previous_labels=dict(
                classification='DIRECT_SINGLE_PATH_WIRING', expectation='NOT_A_LAW_MEAN', price_status='UNQUALIFIED_SINGLE_PATH'),
                qualified_expectation_promotion='FORBIDDEN', model_calibration_point=False, G_input=False),
            source=deepcopy(price_source), price_profile=dict(profile, lowering_id=LOWERING_ID),
            profile_extension=dict(lowering_id=LOWERING_ID, source=NEW_ZZ), normalization_is_price_multiplier=False)
        facts = dict(method='FULL_H_RTE', profile=deepcopy(profile), native_policy='H46_LONG_TIME_NATIVE_COMPOSE',
            rte_rotation_lowering=LOWERING_ID, rte_lowering_source=deepcopy(NEW_ZZ), native_source=deepcopy(NATIVE),
            wrapper_source=deepcopy(WRAPPER), scalar_source=deepcopy(FULL_H), scalar_quantity=SCALAR_QUANTITY,
            scalar_application='ONCE_AT_TOTAL_T', measurement=I.providers(str(root)).B.p.measurement(row, axis),
            resolved_scalar=trial['scalar'], expected_scalar=trial['scalar'])
        side = dict(identity=deepcopy(req['identity']), law_binding=deepcopy(req['law_binding']), axis=axis,
            cohort=COHORT, price_source=deepcopy(price_source), physical_context=deepcopy(req['physical_context']),
            profile_extension=dict(lowering_id=LOWERING_ID, source=NEW_ZZ), implementation=facts,
            implementation_evidence=dict(commit=evidence_commit, path=evidence_path, selector=[str(M) + axis]))
        proofs[str(M) + axis] = {k: deepcopy(side[k]) for k in ('price_source', 'identity', 'law_binding', 'axis',
                                 'cohort', 'physical_context', 'profile_extension', 'implementation')}
        prices.append(price); sides.append(side)
    return prices, sides, proofs


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False, default=str); f.write('\n')


def execute(root=ROOT):
    C, I, S1 = _meta()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    for path in (CALLER, S1.CALLER, I.MODULE, C.MODULE):
        if blob(commit, path) != (Path(root) / path).read_bytes():
            raise ValueError('Execution SOURCE bytes differ from HEAD: ' + path)
    source_checks(root)
    prepared, req, row, seeds = plan_row(root)
    imports = child_import_preflight(root)
    out = Path(root) / OUTPUT; out.mkdir(parents=True, exist_ok=False)
    PRIVATE_OUTPUT.mkdir(parents=True, exist_ok=False)
    source = dict(commit=commit, caller=CALLER, full_h=FULL_H, native=NATIVE, wrapper=WRAPPER, provider=PROVIDER,
                  lowering=NEW_ZZ, cohort=COHORT)
    write(out / 'SOURCE.json', dict(source=source, scope=SCOPE, row=row, seeds=seeds, child_import_preflight=imports,
          python=sys.version,
          platform=platform.platform(), started_utc=datetime.now(timezone.utc).isoformat(), automatic_retry=False))
    start = time.perf_counter(); result = dict(status='STOP_UNKNOWN', errors=[])
    job = dict(row=row, public=str(out / 'child_public'), private=str(PRIVATE_OUTPUT), result=str(PRIVATE_OUTPUT / 'child.json'))
    (PRIVATE_OUTPUT / 'job.json').write_text(json.dumps(job))
    try:
        code = subprocess.call([sys.executable, '-B', str(Path(root) / CALLER), '--child', str(PRIVATE_OUTPUT / 'job.json')],
                               cwd=root)
        data = json.loads(Path(job['result']).read_text()) if Path(job['result']).exists() else None
        if code != 0 or data is None or not data['ok']:
            result.update(status='STOP_WIRING_CHECK', child_exit=code, child=data)
        else:
            result.update(status='P1_WIRING_COMPLETE', trial=data['trial'], checks=data['checks'])
            prices, sides, proofs = price_bindings(root, prepared, data['trial'], price_source=dict(source, record='P1_RESULT.json'),
                evidence_commit='FILL_WITH_PUBLISHED_COMMIT', evidence_path=OUTPUT + '/IMPLEMENTATION_EVIDENCE.json')
            write(out / 'PRICES.json', prices); write(out / 'SIDECARS.json', sides)
            write(out / 'IMPLEMENTATION_EVIDENCE.json', proofs)
    except Exception as error:
        result['errors'].append(dict(type=type(error).__name__, message=str(error)))
    result.update(elapsed_seconds=time.perf_counter() - start, mandatory_stop=True, retry_performed=False,
                  N=None, mean=None, G=None)
    write(out / 'P1_RESULT.json', result)
    print(json.dumps(dict(status=result['status'], elapsed_seconds=result['elapsed_seconds'])), flush=True)
    return 0 if result['status'] == 'P1_WIRING_COMPLETE' else 2


def preflight(root=ROOT):
    source_checks(root)
    prepared, req, row, seeds = plan_row(root)
    return dict(status='PREFLIGHT_METADATA_ONLY', configuration_id=prepared['plan']['configuration']['id'],
                row=row, cohort=COHORT, seeds=seeds, scope=SCOPE, output=OUTPUT, private_output=str(PRIVATE_OUTPUT),
                child_import_preflight=child_import_preflight(root))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute-approved-p1', action='store_true',
                        help='restore saved H4, sample one path, build and compile X/Y (separate approval)')
    parser.add_argument('--child', help=argparse.SUPPRESS)
    parser.add_argument('--child-import-preflight', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.child:
        return child(args.child)
    if args.child_import_preflight:
        report = import_preflight()
        print(json.dumps(report))
        return 0 if report['status'] == 'PASS' else 2
    if args.execute_approved_p1:
        return execute(ROOT)
    print(json.dumps(preflight(ROOT), indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
