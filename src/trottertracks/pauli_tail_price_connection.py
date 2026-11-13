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

No Hamiltonian/state input, no projected sector Pauli, no compile or price here.
The default DF tail factory is not touched; DF tail descriptors are rejected.
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
SEED_FAMILY = 'D_PAULI_TAIL_PRICE_20261114'
SCALAR_QUANTITY = 'HAMILTONIAN_CONSTANT_PLUS_SAVED_DF_TAIL_IDENTITY_sR'
LOWERING_ID = 'PAULI_D2_CENTRAL_ROTATION_CONTROL_WITH_NATIVE_DF_HEAD'


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


def seed(candidate_id, cohort, index, outer, piece):
    data = [SEED_FAMILY, cohort, candidate_id, index, outer, piece]
    return int(hashlib.sha256(json.dumps(data).encode()).hexdigest()[:16], 16)


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


def implementation_facts(profile, *, native_source, wrapper_source, scalar_source, measurement, scalar):
    """Sidecar facts in the C guard shape; this Pauli route is NOT one of its known RTE routes."""
    return dict(method='PR', profile=deepcopy(profile), native_policy='NATIVE_DF_HEAD_PLUS_PAULI_D2_TAIL',
        rte_rotation_lowering=LOWERING_ID, rte_lowering_source=dict(commit=None, path=MODULE),
        native_source=deepcopy(native_source), wrapper_source=deepcopy(wrapper_source),
        scalar_source=deepcopy(scalar_source), scalar_quantity=SCALAR_QUANTITY, scalar_application='ONCE_AT_TOTAL_T',
        measurement=deepcopy(measurement), resolved_scalar=scalar, expected_scalar=scalar)


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
