"""DF head + A's fixed Pauli tail -> native controlled full-wrapper circuit. Opt-in.

Thin connection between existing pieces only:
- A's P0 dictionary (ordered labels/coefficients, lambda_P, saved s_R) and the
  all-round packet row (identity/law/q/R/K) are read from pinned Git JSON and
  bound by A's canonical dictionary SHA-256; nothing is recollected or re-cut.
- Components/labels follow `pauli_signal_adapter.PauliSignalAdapter`
  (RTEComponent(str(j), |c|/lambda, |c|, sign), label j in dictionary order);
  events come from the existing `rte.sample_rte_events`.
- Each outer step's events act on the full 2^n system through
  `append_pauli_event`; `h4_d2_pauli_control.central_control_body` controls
  only the central rotations; the ordered DF head blocks and the controlled
  tail go to the existing `build_partial_native(..., controlled_tails=...)`.
- Hamiltonian constant c and the saved DF tail identity s_R are applied once,
  as one ancilla phase at total T; per-step native calls use scalar=0.

Bit order: A P0 saved operators are in the saved mode-0-MSB basis; the circuit and
labels are Qiskit little-endian (mode q -> qubit q). `saved_to_qiskit` is A's
reverse_bits permutation; `bit_order_checks` gates it against the saved arrays.
The independent oracle (`full_space_expected`) exponentiates the saved head blocks
on the full 2^n space and applies dense Pauli events; no sector projection.

No Hamiltonian/state input, no compile or price here. The default DF tail
factory is not touched; DF tail descriptors are rejected.
"""
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import math
import subprocess
from pathlib import Path

MODULE = 'src/trottertracks/pauli_tail_price_connection.py'
DICTIONARY_KIND = 'GLOBAL_COLLECTED_JW_PAULI_OF_SAVED_DF_TAIL'
SCALAR_QUANTITY = 'HAMILTONIAN_CONSTANT_PLUS_SAVED_DF_TAIL_IDENTITY_sR'
# C guard cfc0fae6 registered route: only the central rotation controlled (P3 native 9c4006bf D2).
LOWERING_ID = 'PAULI_D2_CENTRAL_ROTATION_CONTROL_9C4006BF'
ROUTE = 'PR_PAULI_TAIL'
D2_SOURCE = dict(commit='9c4006bff3e76ea2ba7fbe4a10e679a8693c3a5c', path='src/trotterlib/h4_d2_pauli_control.py')
NATIVE = dict(commit='af284d853890155bacd19e61850c196bd1d5c660', path='scripts/h46_long_time_native.py')
WRAPPER = dict(commit=NATIVE['commit'], path='src/trottertracks/resource_applicability/ax2b_native_df_v5.py')
NATIVE_POLICY = 'NATIVE_DF_HEAD_PLUS_PAULI_D2_TAIL'
QUANTITY = 'FULL_WRAPPER_MEAN_COMPILED_PRICE'
ROOT = Path(__file__).resolve().parents[2]


def _blob(root, commit, path):
    return subprocess.check_output(['git', 'show', commit + ':' + path], cwd=root)


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def load_dictionary(root, identity):
    """A's pinned dictionary, checked against the packet's dictionary identity."""
    if identity.get('kind') != DICTIONARY_KIND or identity.get('event_space') != 'FULL_2_TO_N_NOT_PROJECTED_PAULI':
        raise ValueError('Global JW Pauli dictionary on the full space required; DF tails use their own factory')
    data = json.loads(_blob(root, identity['source']['commit'], identity['source']['path']))
    if canonical_sha256(data) != identity['dictionary_sha256'] or data['prefix'] != identity['prefix']:
        raise ValueError('Pauli dictionary bytes/identity/prefix differ')
    return data


def components(dictionary):
    """Existing PauliSignalAdapter convention, in A's fixed dictionary order."""
    from trotterlib.rte import RTEComponent
    terms = dictionary['terms']; lam = dictionary['lambda_P']
    if math.fsum(abs(t['coefficient']) for t in terms) != lam:
        raise ValueError('lambda_P is not the binary64 fsum of |c|')
    out, labels = [], {}
    n = len(terms[0]['label_qiskit'])
    for j, t in enumerate(terms):
        label, c = t['label_qiskit'], t['coefficient']
        if (len(label) != n or set(label) - set('IXYZ') or set(label) == {'I'} or c == 0
                or float.fromhex(t['coefficient_hex']) != c or t['sign'] != (1 if c > 0 else -1)
                or abs(t['probability'] - abs(c) / lam) > 1e-15):
            raise ValueError('Invalid/identity/inconsistent Pauli term at ' + str(j))
        out.append(RTEComponent(str(j), abs(c) / lam, abs(c), 1 if c > 0 else -1)); labels[str(j)] = label
    return tuple(out), labels, n


def round_request(packet, m, dictionary):
    """One packet round -> physical schedule facts; identity/law echoed, not rebuilt."""
    row = next(r for r in packet['rounds'] if r['m'] == m)
    c = row['identity']['candidate']; st = row['identity']['structure']
    s = packet['configuration']['spec']
    lam = Fraction(s['split']['lambda_R']['numerator'], s['split']['lambda_R']['denominator'])
    if (lam != Fraction(dictionary['lambda_P']) or s['split']['L_D'] != dictionary['prefix']
            or not s['split'].get('tail_dictionary') or s['split']['tail_dictionary']['dictionary_sha256'] != packet['dictionary_identity']['dictionary_sha256']
            or c['method'] != 'PR' or c['formula'] != '2nd' or st['signed_weights'] != [1.0]
            or st['allocation'] != [c['R_outer']] or st['event_count'] != c['q'] * c['R_outer']
            or st['scalar_policy'] != 'scalar_identity_once_at_total_T'):
        raise ValueError('Packet round does not bind this Pauli dictionary/S2/allocation/scalar')
    T = Fraction(c['T']['numerator'], c['T']['denominator'])
    return dict(m=m, T=T, q=c['q'], R=c['R_outer'], K=c['K'], prefix=c['prefix'], candidate_id=row['candidate_id'],
                identity=deepcopy(row['identity']), law_binding=deepcopy(row['law_binding']),
                lambda_P=dictionary['lambda_P'], step_time=float(T) / c['q'])


_TRIAL_SEED = {}


def existing_trial_seed(root=ROOT):
    """The long-time native normal-path seed function (af284d85), AST-loaded only."""
    if 'f' not in _TRIAL_SEED:
        import ast
        tree = ast.parse(_blob(root, NATIVE['commit'], NATIVE['path']))
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'trial_seed')
        env = dict(hashlib=hashlib, json=json)
        exec(compile(ast.Module(body=[node], type_ignores=[]), NATIVE['path'] + ':trial_seed', 'exec'), env)
        _TRIAL_SEED['f'] = env['trial_seed']
    return _TRIAL_SEED['f']


def seed(candidate_id, cohort, index, outer, piece):
    """Existing family H46_LONG_TIME_20261025 keyed by the canonical (dictionary-bound) candidate ID."""
    return existing_trial_seed()(dict(id=candidate_id), cohort, index, outer, piece)


def sample_steps(comps, request, cohort, index):
    """Per outer step: R events from the existing sampler (q*R events in total)."""
    from trotterlib.rte import finite_rte_distribution, sample_rte_events
    dist = finite_rte_distribution(request['lambda_P'] * (request['step_time'] / request['R']), request['K'])
    steps = []
    for outer in range(request['q']):
        events = sample_rte_events(comps, dist, sample_count=request['R'],
                                   seed=seed(request['candidate_id'], cohort, index, outer, 0))
        steps.append(dict(outer=outer, signed_time=request['step_time'], events=events,
                          seed=seed(request['candidate_id'], cohort, index, outer, 0)))
    return steps, dist


def controlled_tail(events, labels, n):
    """Full-space Pauli events; D2 controls only the central rotations."""
    from qiskit import QuantumCircuit
    from trotterlib.pauli_signal_adapter import append_pauli_event
    from trotterlib.h4_d2_pauli_control import central_control_body
    source = QuantumCircuit(n)
    for event in events:
        append_pauli_event(source, event, labels)
    return central_control_body(source).body


def compose(blocks, n, steps, labels, *, T, scalar, control_policy='symmetric_directional', fingerprint=None):
    """Existing native head with controlled Pauli tails; scalar once at total T."""
    from qiskit import QuantumCircuit
    from trottertracks.resource_applicability import ax2b_native_df_v5 as native
    fingerprint = fingerprint or native.canonical_qiskit_circuit_fingerprint
    qc = QuantumCircuit(n + 1); bound = 1
    modes = ('ORDINARY', 'ORDINARY') if control_policy == 'ordinary' else ('UNCONTROLLED', 'DIRECTIONAL')
    base = sum(native.block_instruction_bound(b, mode) for b in blocks for mode in modes)
    for step in steps:
        tail = controlled_tail(step['events'], labels, n)
        part = native.build_partial_native(blocks, num_system_qubits=n, T=step['signed_time'], q=1, scalar=0.,
            controlled_tails=[tail], control_policy=control_policy, max_instructions=base + len(tail.data) + 1)
        qc.compose(part.circuit, inplace=True); bound += part.instruction_upper_bound
    if scalar * T != 0:
        qc.p(-scalar * T, n)
    return native.NativeEvolution(qc, n, T, len(steps), '2nd', control_policy, bound, fingerprint(qc))


def total_scalar(dictionary, hamiltonian_constant):
    """c + saved DF tail identity s_R, binary64 sum in that order (A's KEEP_SAVED_sR_AND_c_ONCE)."""
    return math.fsum((hamiltonian_constant, dictionary['saved_DF_scalar']))


def implementation_facts(profile, *, native_source, measurement, scalar, scalar_source):
    """Sidecar facts for C guard route PR_PAULI_TAIL (cfc0fae6)."""
    return dict(method='PR', route=ROUTE, profile=deepcopy(profile), native_policy=NATIVE_POLICY,
        rte_rotation_lowering=LOWERING_ID, rte_lowering_source=deepcopy(D2_SOURCE),
        native_source=deepcopy(native_source), wrapper_source=deepcopy(WRAPPER),
        scalar_source=deepcopy(scalar_source), scalar_quantity=SCALAR_QUANTITY, scalar_application='ONCE_AT_TOTAL_T',
        measurement=deepcopy(measurement), resolved_scalar=scalar, expected_scalar=scalar)


# --- Bit order and the independent full-2^n oracle ---------------------------------------------

def reverse_bits(n):
    """A P0 convention: saved mode-0-MSB index i <-> Qiskit little-endian index reverse_bits(n)[i]."""
    import numpy as np
    return np.array([int(format(j, '0' + str(n) + 'b')[::-1], 2) for j in range(2 ** n)])


def saved_to_qiskit(op):
    """A_q[r(i), r(j)] = A_s[i, j]; r is an involution, so A_q = A_s[r][:, r]."""
    import numpy as np
    op = np.asarray(op); n = int(round(math.log2(op.shape[0])))
    if op.shape != (1 << n, 1 << n):
        raise ValueError('Square 2^n operator required')
    r = reverse_bits(n)
    return op[np.ix_(r, r)]


qiskit_to_saved = saved_to_qiskit  # the bit reversal is its own inverse


def pauli_sum(dictionary, order):
    """sum_j c_j P_j from the dictionary labels in the requested basis order (no identity)."""
    import numpy as np
    from qiskit.quantum_info import Pauli
    key = dict(qiskit='label_qiskit', saved='label_in_saved_tensor_order')[order]
    n = len(dictionary['terms'][0]['label_qiskit'])
    out = np.zeros((1 << n, 1 << n), dtype=complex)
    for t in dictionary['terms']:
        out += t['coefficient'] * Pauli(t[key]).to_matrix()
    return out


def bit_order_checks(dictionary, operators, blocks, *, tolerance=1e-9):
    """Pre-gate on A P0 saved operators: labels, basis order, head/tail split, scalar."""
    import numpy as np
    terms = dictionary['terms']; prefix = dictionary['prefix']
    A = np.asarray(operators['Pauli_nonidentity']); eye = np.eye(len(A))
    head_sum = sum(np.asarray(blocks)[:prefix + 1]); tail_sum = sum(np.asarray(blocks)[prefix + 1:])
    out = dict(labels_reversed=all(t['label_in_saved_tensor_order'] == t['label_qiskit'][::-1] for t in terms),
        saved_vs_saved_labels=float(np.linalg.norm(A - pauli_sum(dictionary, 'saved'))),
        qiskit_vs_qiskit_labels=float(np.linalg.norm(saved_to_qiskit(A) - pauli_sum(dictionary, 'qiskit'))),
        unconverted_mismatch=float(np.linalg.norm(A - pauli_sum(dictionary, 'qiskit'))),
        head_vs_blocks=float(np.linalg.norm(np.asarray(operators['head']) - head_sum)),
        tail_vs_blocks=float(np.linalg.norm(np.asarray(operators['tail']) - tail_sum)),
        tail_vs_pauli_plus_pauli_identity=float(np.linalg.norm(np.asarray(operators['tail']) - A - dictionary['pauli_identity'] * eye)),
        tail_vs_pauli_plus_saved_sR=float(np.linalg.norm(np.asarray(operators['tail']) - A - dictionary['saved_DF_scalar'] * eye)))
    out['pass'] = (out['labels_reversed'] and max(out[k] for k in ('saved_vs_saved_labels', 'qiskit_vs_qiskit_labels',
                   'head_vs_blocks', 'tail_vs_blocks', 'tail_vs_pauli_plus_pauli_identity', 'tail_vs_pauli_plus_saved_sR')) < tolerance)
    return out


def event_matrix(event, labels, n):
    """Independent dense event operator in Qiskit order (application order, signs, order phase)."""
    import numpy as np
    from qiskit.quantum_info import Pauli
    result = np.eye(1 << n, dtype=complex)
    for app in event.application_sequence:
        p = Pauli(labels[app.component_id]).to_matrix()
        if app.role == 'product':
            op = app.coefficient_sign * p
        else:
            a = app.coefficient_sign * event.rotation_angle
            op = math.cos(a) * np.eye(1 << n) - 1j * math.sin(a) * p
        result = op @ result
    return event.phase * result


def full_space_expected(head_blocks_saved, steps, labels, *, scalar, T, state):
    """Independent full-2^n high-branch vector: per step head(dt/2) . events . reversed head(dt/2)."""
    import numpy as np
    H = [saved_to_qiskit(h) for h in head_blocks_saved]
    if any(np.linalg.norm(h - h.conj().T) > 1e-12 for h in H):
        raise ValueError('Head block operators must be Hermitian')
    spectra = [np.linalg.eigh(h) for h in H]; n = int(round(math.log2(len(state))))
    v = np.array(state, dtype=complex)
    for step in steps:
        half = [w @ (np.exp(-.5j * step['signed_time'] * e)[:, None] * w.conj().T) for e, w in spectra]
        for U in half:
            v = U @ v
        for event in step['events']:
            v = event_matrix(event, labels, n) @ v
        for U in reversed(half):
            v = U @ v
    return np.exp(-1j * scalar * T) * v


def index0_checks(evolution, state, expected, *, tolerance=1e-9):
    """Low-control identity, high-control leakage/action, measured X/Y wrapper signs."""
    import numpy as np
    from trotterlib.df_trotter.circuit import simulate_statevector
    from trottertracks.resource_applicability import ax2b_native_df_v5 as native
    d = len(state); zero = np.zeros_like(state)
    low = np.r_[state, zero]; high = np.r_[zero, state]
    checks = []
    def gate(name, value, tol=tolerance):
        value = float(value)
        if not math.isfinite(value) or value >= tol:
            raise ValueError(name + ':' + str(value))
        checks.append(dict(check=name, error=value, tolerance=tol))
    gate('low_control', np.linalg.norm(simulate_statevector(evolution.circuit, low) - low))
    out = simulate_statevector(evolution.circuit, high)
    gate('high_control_leakage', np.linalg.norm(out[:d]))
    gate('independent_full_space_vector', np.linalg.norm(out[d:] - expected))
    z = complex(np.vdot(state, expected))
    wrappers = {}
    for axis in ('cosine', 'sine'):
        w = native.build_native_hadamard_wrapper(evolution, axis=axis, include_measurement=True,
                                                 max_instructions=len(evolution.circuit.data) + 4)
        v = simulate_statevector(w.remove_final_measurements(inplace=False), low)
        observed = float(np.vdot(v[:d], v[:d]).real - np.vdot(v[d:], v[d:]).real)
        gate('Hadamard_' + axis, abs(observed - (z.real if axis == 'cosine' else z.imag)))
        wrappers[axis] = w
    return checks, z, wrappers


def price_records(config, m, *, trial_prices, scalar, physical_candidate, native_source, cohort, price_source,
                  evidence_commit, evidence_path, measurement):
    """Direct single-path records + guard sidecars/proofs for PR_PAULI_TAIL (no law mean)."""
    row = config['rounds'][m]; s = config['spec']; profile = s['profile']
    physical = dict(binding=deepcopy(s['binding']), profile=deepcopy(profile), physical_candidate=deepcopy(physical_candidate),
                    structure=deepcopy(row['structure']), split=deepcopy(s['split']), law=deepcopy(s['law']))
    prices, sides, proofs = [], [], {}
    for axis in ('X', 'Y'):
        meas = measurement(row, axis); meas['law_context']['ideal_sampler_distance_qualified'] = None
        metrics = trial_prices['cosine' if axis == 'X' else 'sine']
        price = dict(configuration_id=config['id'], m=m, axis=axis, identity=deepcopy(row['identity']),
            law_binding=deepcopy(row['law_binding']), **meas, physical_context=deepcopy(physical), quantity=QUANTITY,
            P=0, cohort=cohort, n_paths=1, mean=dict(rz_count=metrics['rz_count']), metrics=deepcopy(metrics),
            classification='DIRECT_SINGLE_PATH_WIRING', expectation='NOT_A_LAW_MEAN', price_status='UNQUALIFIED_SINGLE_PATH',
            source=deepcopy(price_source), price_profile=dict(profile, lowering_id=LOWERING_ID),
            profile_extension=dict(lowering_id=LOWERING_ID, source=D2_SOURCE), normalization_is_price_multiplier=False)
        facts = implementation_facts(profile, native_source=native_source, measurement=measurement(row, axis),
                                     scalar=scalar, scalar_source=dict(s['split']['tail_dictionary']['source'],
                                     selector=['saved_DF_scalar'], plus='HAMILTONIAN_CONSTANT_OF_BOUND_SNAPSHOT'))
        key = 'm%d%s' % (m, axis)
        side = dict(identity=deepcopy(row['identity']), law_binding=deepcopy(row['law_binding']), axis=axis, cohort=cohort,
            price_source=deepcopy(price_source), physical_context=deepcopy(physical),
            profile_extension=dict(lowering_id=LOWERING_ID, source=D2_SOURCE), implementation=facts,
            implementation_evidence=dict(commit=evidence_commit, path=evidence_path, selector=[key]))
        proofs[key] = {k: deepcopy(side[k]) for k in ('price_source', 'identity', 'law_binding', 'axis', 'cohort',
                       'physical_context', 'profile_extension', 'implementation')}
        prices.append(price); sides.append(side)
    return prices, sides, proofs


def legacy_physical_candidate(request, *, candidate_id):
    """Legacy PR physical row (no dictionary in its CID; C's bridge keeps the canonical identity)."""
    T = float(request['T'])
    c = dict(system='H4', method='PR', formula='2nd', prefix=request['prefix'], T=T, q=request['q'],
             delta=T / request['q'], R_outer=request['R'], K=request['K'])
    c['id'] = candidate_id(c)
    return c


PACKETS = dict(commit='714247619afb4714fd663eb15c0470ef32bdc7af',
               path='docs/research/df_head_pauli_tail_all_round_results_20261114/MEAN_HANDOFF.json')
PILOT_COHORT = 'D_PAULI_LD1_M0_WIRING_V1'


def pilot_plan(root, *, prefix=1, m=0, cohort=PILOT_COHORT, hamiltonian_constant=2.29310124732):
    """Metadata for the LD1/m0 normal-1-path X/Y pilot (not executed here)."""
    handoff = json.loads(_blob(root, PACKETS['commit'], PACKETS['path']))
    if handoff.get('C') is not None or handoff.get('G') is not None:
        raise ValueError('Pauli packet unexpectedly carries prices')
    packet = next(p for p in handoff['packets'] if p['prefix'] == prefix)
    dictionary = load_dictionary(root, packet['dictionary_identity'])
    comps, labels, n = components(dictionary)
    req = round_request(packet, m, dictionary)
    seeds = [seed(req['candidate_id'], cohort, 0, outer, 0) for outer in range(req['q'])]
    return dict(configuration_id=packet['configuration']['id'], prefix=prefix, m=m, request=req, seeds=seeds,
        cohort=cohort, n=n, terms=len(comps), dictionary_sha256=packet['dictionary_identity']['dictionary_sha256'],
        scalar=total_scalar(dictionary, hamiltonian_constant), head='prepare(ham, %d).deterministic_blocks' % prefix,
        scope=dict(normal_paths=1, index=0, axes=['X', 'Y'], compile_maximum=2, sampling='H4 events after approval only'),
        execution_caller=None, approved=False)
