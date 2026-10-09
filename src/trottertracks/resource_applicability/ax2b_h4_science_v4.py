"""Future authorized H4 technical pilot; importing does not run science.

No molecule/state generation, H6/H8, GPU, quantum shots, fitting or selection.
This module is imported by the child only after contract/resource enforcement.
"""
from __future__ import annotations

import math
import gc
from itertools import groupby
from pathlib import Path
import resource
import time

import numpy as np
from scipy.linalg import expm

from trotterlib.df_hamiltonian import DFHamiltonian, PhysicalSector, df_linear_operator
from trotterlib.df_partial_s2 import DFDeterministicOneBodySpec
from trotterlib.df_partial_s2_repeated import (make_df_partial_s2_repeated_request,
                                             QiskitDFPartialS2RepeatedCircuitBuilder)
from trotterlib.df_trotter.circuit import simulate_statevector
from trotterlib.pr2_new_series_validation import _load_snapshot_once
from trotterlib.pr2_s0_s1_validation import _bit_reverse, _to_qiskit_state
from trotterlib.pr2_matched_accuracy_m1_execution import (
    _prepare, _prepare_discard, _eigendecomposition, _apply_exponential,
    _apply_spectral_values, _apply_outer_step, _explicit_cutoff_tolerance,
)
from trotterlib.rte import CompilerSettings, make_rte_config, finite_rte_distribution
from trotterlib.rte_compiled_cost import transpile_and_measure_cost
from .ax2b_stream_fingerprint_v4 import canonical_qiskit_circuit_fingerprint

from .ax2b_native_df_v4 import (make_native_block_action, build_deterministic_native,
                            partial_native_from_step_requests, build_native_hadamard_wrapper)
from .ax2a_state_action import (ActionBudget, partial_s2_signal, deterministic_pf_state,
                              project_primitive_checked, df_tail_operator, eigenphase_reference)
from .ax2a_preparation import digest
from .ax2b_h4_contract_v4 import SNAPSHOT, SNAPSHOT_SHA, file_hash, source_hashes
from .ax2b_limits import CallBudget, exclusive_json, output_size
from .ax2b_diagnostics_v4 import ResourceTrace


def require_agreement(error, tolerance, label):
    if not math.isfinite(float(error)) or error > tolerance:
        raise ValueError('CORRECTNESS_GATE:' + label)


def primitive_sector_certificate(ham, sector):
    """Exact coefficient-structure certificate, before squared A projection.

    Number conservation is intrinsic to a†Ga. For an Sz sector all cross-spin
    entries must be exactly zero. Near-zero coefficients are never discarded.
    Completeness of the stored sector is checked independently of its hash.
    """
    if sector.n_qubits != ham.n_qubits:
        raise ValueError('SECTOR_QUBITS')
    if sector.nelec_alpha is not None and sector.nelec_beta is not None:
        expected = PhysicalSector.spin_sector(n_qubits=ham.n_qubits,
                     nelec_alpha=sector.nelec_alpha, nelec_beta=sector.nelec_beta)
        cross_spin = np.arange(ham.n_qubits)[:, None] % 2 != np.arange(ham.n_qubits)[None, :] % 2
    elif sector.n_electrons is not None:
        expected = PhysicalSector.number_sector(n_qubits=ham.n_qubits, n_electrons=sector.n_electrons)
        cross_spin = np.zeros((ham.n_qubits, ham.n_qubits), dtype=bool)
    else:
        raise ValueError('SECTOR_QUANTUM_NUMBERS')
    if not np.array_equal(sector.basis_indices, expected.basis_indices):
        raise ValueError('INCOMPLETE_OR_REORDERED_SECTOR')
    for index, matrix in enumerate((ham.one_body, *ham.g_matrices)):
        if not np.isfinite(matrix).all() or np.any(matrix[cross_spin] != 0):
            raise ValueError('PRIMITIVE_SPIN_SECTOR:' + str(index))
        require_agreement(np.linalg.norm(matrix - matrix.conj().T), 1e-12, 'primitive_hermiticity')
    return {'scope': 'all a†Ga primitives before A squared; exact cross-spin zeros',
            'sector_dimension': sector.dimension, 'matrix_count': 1 + ham.n_blocks,
            'gate_intermediate_projection': False}


def checked_basis_bridge(ham, sector, full_state, sector_state, tolerance):
    if full_state.shape != (1 << ham.n_qubits,) or sector_state.shape != (sector.dimension,):
        raise ValueError('STATE_DIMENSION')
    if not np.isfinite(full_state).all() or not np.isfinite(sector_state).all():
        raise ValueError('STATE_NONFINITE')
    require_agreement(abs(np.linalg.norm(full_state) - 1), tolerance, 'full_state_norm')
    require_agreement(abs(np.linalg.norm(sector_state) - 1), tolerance, 'sector_state_norm')
    lifted = np.zeros_like(full_state)
    lifted[sector.basis_indices] = sector_state
    require_agreement(np.linalg.norm(lifted - full_state), tolerance, 'saved_state_bridge')
    qindices = tuple(_bit_reverse(int(i), ham.n_qubits) for i in sector.basis_indices)
    qstate = _to_qiskit_state(full_state, ham.n_qubits)
    require_agreement(np.linalg.norm(qstate[list(qindices)] - sector_state), tolerance, 'bit_order')
    return qindices, qstate


def dense_df_qiskit(ham, budget, per_action_cap):
    """H4-only full-space oracle, with before-call matvec accounting."""
    if ham.n_qubits > 8:
        raise ValueError('DENSE_REFERENCE_H4_ONLY')
    dimension = 1 << ham.n_qubits
    if dimension > per_action_cap:
        raise RuntimeError('REFERENCE_ACTION_CAP')
    full = PhysicalSector(ham.n_qubits, np.arange(dimension, dtype=np.int64))
    operator, _counter = df_linear_operator(ham, full, backend='python')
    matrix = np.empty((dimension, dimension), dtype=complex)
    for column in range(dimension):
        budget.take('reference_matvecs')
        vector = np.zeros(dimension, dtype=complex)
        vector[column] = 1
        matrix[:, column] = operator @ vector
    require_agreement(np.linalg.norm(matrix - matrix.conj().T), 1e-12, 'dense_hermiticity')
    permutation = [_bit_reverse(i, ham.n_qubits) for i in range(dimension)]
    return matrix[np.ix_(permutation, permutation)]


def dense_global_state(state, eigensystems, *, T, q, formula, scalar):
    """Independent unmerged S2/Yoshida composition for the action oracle."""
    if formula not in ('2nd', '4th') or type(q) is not int or q < 1:
        raise ValueError('GLOBAL_REFERENCE_PARAMETERS')
    weight = 1 / (2 - 2 ** (1 / 3))
    weights = (1,) if formula == '2nd' else (weight, 1 - 2 * weight, weight)
    current = state.copy()
    for _ in range(q):
        for w in weights:
            for eigensystem in tuple(eigensystems) + tuple(reversed(eigensystems)):
                current = _apply_exponential(eigensystem, current, w * T / (2 * q))
    return np.exp(-1j * scalar * T) * current


def dense_partial_signals(state, deterministic, tail, *, T, q, r, K, lambda_r, scalar):
    """M1 spectral numerator and raw normalization, independent of Horner."""
    delta = T / q
    tau = lambda_r * delta / r
    b = finite_rte_distribution(tau, K).exact_finite_distribution
    eigenvalues = tail[0]
    term = np.ones_like(eigenvalues, dtype=complex)
    polynomial = term.copy()
    for degree in range(1, K + 2):
        term = term * (-1j * tau * eigenvalues / degree)
        polynomial += term
    values = (polynomial ** r, (polynomial / b) ** r,
              np.exp(-1j * lambda_r * delta * eigenvalues))
    signals = []
    for value in values:
        current = state.copy()
        for _ in range(q):
            current = _apply_outer_step(current, deterministic, delta=delta,
                         phase=np.exp(-1j * scalar * delta),
                         tail_action=lambda vector: _apply_spectral_values(tail, vector, value))
        signals.append(complex(np.vdot(state, current)))
    return tuple(signals), q * r * math.log(b)


def complex_record(value):
    return {'real': float(value.real), 'imag': float(value.imag)}


class Pilot:
    def __init__(self, root, output, manifest):
        self.root, self.output = Path(root), Path(output)
        self.plan, self.manifest = manifest['plan'], manifest
        self.caps, self.gates = self.plan['caps'], self.plan['gates']
        self.calls = CallBudget(compile=self.caps['compile_calls'],
                               trajectory=self.caps['trajectory_samples'],
                               occurrence=self.caps['occurrence_samples'],
                               primitive=self.caps['primitive_validation_actions'],
                               control_probe=self.caps['control_probe_actions'],
                               reference_matvecs=self.caps['reference_matvecs_total'])
        self.completed = 0
        self.compiled = 0
        self.started = time.monotonic()
        self.trace = ResourceTrace(self.write, lambda: time.monotonic() - self.started)
        self._failure_reserve = bytearray(1048576)

    def write(self, name, payload):
        # Account before every result write, leaving space for the parent
        # terminal witness. The independent watchdog also polls logs/total.
        import json
        size = len((json.dumps(payload, ensure_ascii=False, sort_keys=True,
                               indent=2, allow_nan=False) + '\n').encode())
        if output_size(self.output) + size > self.caps['output_bytes'] - 65536:
            raise RuntimeError('OUTPUT_WRITE_CAP')
        exclusive_json(self.output / name, payload)

    def phase(self, name):
        self.write('phase_' + name + '.json', {'phase': name, 'elapsed': time.monotonic() - self.started})

    def action_budget(self):
        return ActionBudget(self.caps['tail_matvecs_per_signal'], self.caps['deterministic_actions_per_signal'])

    def setup(self):
        self.phase('input_reference')
        path = self.root / SNAPSHOT
        if file_hash(path) != SNAPSHOT_SHA:
            raise ValueError('H4_INPUT_CHANGED')
        ham, sector, full_state, state, metadata, layout = _load_snapshot_once(path)
        if ham.n_qubits != 8 or ham.n_blocks != 12 or metadata != self.manifest['snapshot']['metadata']:
            raise ValueError('H4_SNAPSHOT_MODEL')
        certificate = primitive_sector_certificate(ham, sector)
        qindices, qstate = checked_basis_bridge(ham, sector, full_state, state, self.gates['normalization_tolerance'])
        self.ham, self.sector, self.state, self.qindices, self.qstate = ham, sector, state, qindices, qstate
        limit = self.caps['reference_matvecs_per_action']
        full_matrix = dense_df_qiskit(ham, self.calls, limit)
        one = DFHamiltonian(0, ham.one_body, np.empty(0), (), {})
        self.one_matrix = dense_df_qiskit(one, self.calls, limit)
        self.fragments = tuple(dense_df_qiskit(DFHamiltonian(0, np.zeros_like(ham.one_body),
                      ham.lambdas[i:i+1], (ham.g_matrices[i],), {}), self.calls, limit) for i in range(12))
        reconstructed = ham.constant * np.eye(256) + self.one_matrix + sum(self.fragments)
        require_agreement(np.linalg.norm(full_matrix - reconstructed), 1e-10, 'DF_reconstruction')
        operator, counter = df_linear_operator(ham, sector, backend='python')
        sector_matrix = full_matrix[np.ix_(qindices, qindices)]
        sector_error = 0.0
        for index in range(sector.dimension):
            self.calls.take('reference_matvecs')
            vector = np.zeros(sector.dimension, dtype=complex)
            vector[index] = 1
            sector_error = max(sector_error, float(np.linalg.norm(operator @ vector - sector_matrix[:, index])))
        require_agreement(sector_error, self.gates['agreement_tolerance'], 'sector_all_columns')
        self.calls.take('reference_matvecs')
        phase = eigenphase_reference(state, lambda v: operator @ v, self.plan['T'])
        require_agreement(phase['signal_allowance'], self.gates['agreement_tolerance'], 'saved_state_phase_residual')
        evolved = expm(-1j * self.plan['T'] * full_matrix) @ qstate
        spectral = _apply_exponential(_eigendecomposition(full_matrix), qstate, self.plan['T'])
        discrepancy = float(np.linalg.norm(evolved - spectral))
        require_agreement(discrepancy, self.gates['reference_discrepancy_tolerance'], 'reference_expm_eigh')
        self.target = complex(np.vdot(qstate, evolved))
        # Residual is a bound for the exact Hermitian phase surrogate only.
        # Cross-method discrepancy is diagnostic, not a roundoff certificate.
        self.write('input_reference.json', {'metadata': metadata, 'layout': layout,
            'sector_certificate': certificate, 'sector_all_columns_error': sector_error,
            'basis_order': 'OpenFermion sector -> bit reverse -> full Qiskit; no gate-level projection',
            'reference_signal': complex_record(self.target), 'rayleigh_energy': phase['rayleigh_energy'],
            'residual': phase['residual'], 'exact_phase_surrogate_allowance': phase['signal_allowance'],
            'phase_surrogate_signal': complex_record(phase['signal']),
            'expm_eigh_state_discrepancy': discrepancy,
            'phase_surrogate_signal_difference': abs(phase['signal'] - self.target),
            'total_numerical_allowance_certified': False, 'eligibility_status': 'UNDETERMINED',
            'ground_state_certified': False, 'sector_reference_matvecs': counter['count']})
        self.preparations = {}
        self.eigensystems = {'one_body': _eigendecomposition(self.one_matrix)}
        self.eigensystems.update({f'fragment_{i}': _eigendecomposition(m) for i, m in enumerate(self.fragments)})

    def preparation_for(self, cell):
        key = (cell['method'] == 'B0', cell['prefix'])
        if key not in self.preparations:
            self.preparations[key] = (_prepare_discard if key[0] else _prepare)(self.ham, key[1])
        return self.preparations[key]

    def block_key(self, block):
        return 'one_body' if isinstance(block, DFDeterministicOneBodySpec) else f'fragment_{block.original_fragment_index}'

    def correctness(self):
        self.phase('correctness')
        # Molecular block lowering: both time signs, all sector columns and
        # the saved state, before projecting each complete block.
        full_preparation = self.preparation_for({'method': 'B1', 'prefix': self.ham.n_blocks})
        dimension = 1 << self.ham.n_qubits
        probes = [self.qstate]
        for index in self.qindices:
            vector = np.zeros(dimension, dtype=complex)
            vector[index] = 1
            probes.append(vector)
        primitive_error = 0.0
        for block in full_preparation.deterministic_blocks:
            action = make_native_block_action(block, max_instructions=self.caps['untranspiled_instructions'])
            eigen = self.eigensystems[self.block_key(block)]
            for time_value in (0.2, -0.2):
                for probe in probes:
                    self.calls.take('primitive')
                    actual = action(probe, time_value)
                    expected = _apply_exponential(eigen, probe, time_value)
                    primitive_error = max(primitive_error, float(np.linalg.norm(actual - expected)))
        require_agreement(primitive_error, self.gates['agreement_tolerance'], 'native_blocks_all_sector_columns')
        self.write('primitive_lowering.json', {'max_state_error': primitive_error,
                     'time_values': [0.2, -0.2], 'probe_count_per_block_time': len(probes)})
        for cell in self.plan['correctness_cells']:
            prep = self.preparation_for(cell)
            spectral = [(self.block_key(b), self.eigensystems[self.block_key(b)]) for b in prep.deterministic_blocks]
            actions = []
            for block in prep.deterministic_blocks:
                full_action = make_native_block_action(block, max_instructions=self.caps['untranspiled_instructions'])
                actions.append(lambda v, t, a=full_action: project_primitive_checked(
                    lambda lifted: a(lifted, t), v, self.qindices, dimension,
                    leakage_tolerance=self.gates['leakage_tolerance']))
            budget = self.action_budget()
            scalar = prep.constant_coefficient + prep.extracted_identity_coefficient
            extra = {}
            if cell['method'] in ('B2', 'B3'):
                tail, counter = df_tail_operator(self.ham, self.sector, range(cell['prefix'], self.ham.n_blocks),
                       extracted_identity=prep.extracted_identity_coefficient, primitive_sector_certified=True)
                lam = prep.exact_rte_lambda_r
                if lam <= 0:
                    raise ValueError('REGISTERED_RANDOM_CELL_EMPTY_TAIL')
                r = cell['R'] // cell['q']
                result = partial_s2_signal(self.state, actions, lambda v: (tail @ v) / lam,
                     lambda_r=lam, T=self.plan['T'], q=cell['q'], r=r, K=cell['K'],
                     phase_energy=scalar, budget=budget)
                tail_matrix = (sum(self.fragments[cell['prefix']:]) - prep.extracted_identity_coefficient * np.eye(dimension)) / lam
                oracle, log_B = dense_partial_signals(self.qstate, spectral, _eigendecomposition(tail_matrix),
                      T=self.plan['T'], q=cell['q'], r=r, K=cell['K'], lambda_r=lam, scalar=scalar)
                require_agreement(abs(result.corrected - oracle[0]), self.gates['agreement_tolerance'], 'finite_corrected')
                if result.raw is None or result.normalization is None:
                    raise ValueError('NORMALIZATION_REPRESENTATION_UNAVAILABLE')
                require_agreement(abs(result.raw - oracle[1]), self.gates['agreement_tolerance'], 'finite_raw')
                require_agreement(abs(result.log_normalization - log_B), 1e-12, 'log_B')
                require_agreement(abs(result.raw * result.normalization - result.corrected), 1e-9, 'B_raw_corrected')
                corrected = result.corrected
                extra = {'raw_signal': complex_record(result.raw), 'log_B': log_B,
                         'B': result.normalization, 'lambda_r': lam,
                         'pf_exact_tail_signal': complex_record(oracle[2]),
                         'signed_finite_error': complex_record(corrected - oracle[2]),
                         'signed_outer_pf_error': complex_record(oracle[2] - self.target),
                         'tail_operator_matvecs': counter['count']}
            else:
                actual = deterministic_pf_state(self.state, actions, T=self.plan['T'], q=cell['q'],
                          formula=cell['order'], scalar=scalar, budget=budget)
                expected = dense_global_state(self.qstate, [e for _, e in spectral], T=self.plan['T'],
                           q=cell['q'], formula=cell['order'], scalar=scalar)
                require_agreement(np.linalg.norm(actual - expected[list(self.qindices)]),
                                  self.gates['agreement_tolerance'], 'global_PF_state')
                corrected = complex(np.vdot(self.state, actual))
                if cell['method'] == 'B0':
                    truncated = self.ham.constant * np.eye(dimension) + self.one_matrix + sum(self.fragments[:cell['prefix']])
                    exact_truncated = complex(np.vdot(self.qstate, expm(-1j * self.plan['T'] * truncated) @ self.qstate))
                    extra = {'exact_truncated_signal': complex_record(exact_truncated),
                             'signed_discard_error': complex_record(exact_truncated - self.target),
                             'signed_PF_error': complex_record(corrected - exact_truncated)}
            self.write(cell['id'] + '_correctness.json', {'cell': cell, 'signal': complex_record(corrected),
                    'reference': complex_record(self.target), 'signed_total_error': complex_record(corrected - self.target),
                    'action_counts': {'tail': budget.tail_matvecs, 'deterministic': budget.deterministic_actions},
                    'accuracy_eligibility': 'UNDETERMINED', 'numerical_allowance_certified': False, **extra})
            self.completed += 1

    def trajectory(self, cell, seed):
        prep = self.preparation_for(cell)
        r = cell['R'] // cell['q']
        delta = self.plan['T'] / cell['q']
        tau = prep.exact_rte_lambda_r * delta / r
        config, distribution = make_rte_config(prep.rte_preparation.symbolic_tail,
              evolution_time=delta, rte_steps=r,
              truncation_tolerance=_explicit_cutoff_tolerance(tau, cell['K']),
              finite_taylor_order=cell['K'], seed=seed)
        self.calls.take('trajectory')
        self.calls.take('occurrence', cell['q'])
        request = make_df_partial_s2_repeated_request(prep, step_time=delta, repetition_count=cell['q'],
                   rte_config=config, rte_distribution=distribution, seed=seed,
                   controlled=True, ancilla_qubit=self.ham.n_qubits, construction_policy='raw_concatenation')
        steps = tuple(request.iter_step_requests())
        events = [[event.to_dict() for event in step.rte_occurrence.events] for step in steps]
        return request, steps, events

    def costs(self):
        if self.completed != 8:
            raise ValueError('H4_CORRECTNESS_REQUIRED_BEFORE_COST')
        self.phase('wrapper_cost')
        import qiskit
        cp = self.plan['compiler']
        compiler = CompilerSettings(tuple(cp['basis_gates']), None, None, cp['optimization_level'],
                                    None, None, cp['seed_transpiler'], qiskit.__version__)
        measured = {}
        dimension = 1 << self.ham.n_qubits
        first, last = np.zeros(dimension, dtype=complex), np.zeros(dimension, dtype=complex)
        first[self.qindices[0]], last[self.qindices[-1]] = 1, 1
        probes = (self.qstate, first, last)
        seen_groups = set()
        with self.trace.bind():
            for key, iterator in groupby(self.plan['wrapper_tasks'],
                                         key=lambda t: (t['cell']['id'], t['replica'])):
                tasks = tuple(iterator)
                if key in seen_groups or len(tasks) != 4 or {
                    (t['control_policy'], t['axis']) for t in tasks
                } != {(p, a) for p in ('ordinary', 'symmetric_directional') for a in ('cosine', 'sine')}:
                    raise ValueError('CONTIGUOUS_PAIRED_WRAPPER_GROUP_REQUIRED')
                seen_groups.add(key)
                with self.trace.stage('cell_replica_group', cell_id=key[0], replica=key[1]):
                    rows = self._cost_group(tasks, compiler, probes)
                for task, metrics in rows:
                    group = (task['cell']['id'], task['control_policy'], task['axis'])
                    measured.setdefault(group, []).append(metrics)
                # No circuit is returned or retained across groups, including
                # the last transpiled circuit and the legacy replay oracle.
                gc.collect()
                self.trace.event('group_released', cell_id=key[0], replica=key[1])
        summary = []
        for (cell, policy, axis), rows in measured.items():
            summary.append({'cell_id': cell, 'control_policy': policy, 'axis': axis, 'n': len(rows),
                'metrics': {name: {'mean': float(np.mean([row[name] for row in rows])),
                                   'min': min(row[name] for row in rows), 'max': max(row[name] for row in rows),
                                   'sample_sd': float(np.std([row[name] for row in rows], ddof=1)) if len(rows) > 1 else None}
                            for name in rows[0]}, 'cost_scope': 'technical sample, no population or ranking claim'})
        self.write('wrapper_cost_summary.json', {'groups': summary, 'compiled_wrappers': self.compiled})

    def _cost_group(self, tasks, compiler, probes):
        cell, first_task = tasks[0]['cell'], tasks[0]
        policies = {}
        evolution = legacy = wrapper = cost = request = steps = None
        rows = []
        try:
            prep = self.preparation_for(cell)
            events = None
            if cell['method'] in ('B2', 'B3'):
                with self.trace.stage('trajectory_requests'):
                    request, steps, events = self.trajectory(cell, first_task['trajectory_seed'])
                for policy in ('ordinary', 'symmetric_directional'):
                    with self.trace.stage('native_build', control_policy=policy):
                        policies[policy] = partial_native_from_step_requests(steps, control_policy=policy,
                                           max_instructions=self.caps['untranspiled_instructions'])
                if policies['ordinary'].instruction_upper_bound + 2 * cell['q'] > self.caps['untranspiled_instructions']:
                    raise RuntimeError('LEGACY_ORACLE_PRE_BUILD_CAP')
                with self.trace.stage('legacy_replay_build'):
                    legacy = QiskitDFPartialS2RepeatedCircuitBuilder().build(request).circuit
                    if len(legacy.data) > self.caps['untranspiled_instructions']:
                        raise RuntimeError('LEGACY_ORACLE_POST_BUILD_CAP')
            else:
                for policy in ('ordinary', 'symmetric_directional'):
                    with self.trace.stage('native_build', control_policy=policy):
                        policies[policy] = build_deterministic_native(prep.deterministic_blocks,
                          num_system_qubits=self.ham.n_qubits, T=self.plan['T'], q=cell['q'], formula=cell['order'],
                          scalar=prep.constant_coefficient, control_policy=policy,
                          max_instructions=self.caps['untranspiled_instructions'])
            errors = []
            with self.trace.stage('paired_control_probe'):
                for probe in probes:
                    lower = np.concatenate((probe, np.zeros_like(probe)))
                    upper = np.concatenate((np.zeros_like(probe), probe))
                    for value in (lower, upper):
                        self.calls.take('control_probe', 2 + int(legacy is not None))
                        ordinary = simulate_statevector(policies['ordinary'].circuit, value)
                        directional = simulate_statevector(policies['symmetric_directional'].circuit, value)
                        errors.append(float(np.linalg.norm(ordinary - directional)))
                        if legacy is not None:
                            errors.append(float(np.linalg.norm(ordinary - simulate_statevector(legacy, value))))
                        if value is lower:
                            errors.append(float(np.linalg.norm(ordinary - lower)))
                require_agreement(max(errors), self.gates['agreement_tolerance'], 'paired_control_and_legacy')
            # Release requests/oracle once the control checks finish. Explicit
            # events/digest remain available for all four paired wrappers.
            request = steps = legacy = None
            trajectory_record = {'cell': cell, 'replica': first_task['replica'], 'seed': first_task['trajectory_seed'],
                     'events_by_outer_step': events, 'event_digest': digest(events),
                     'comparison_max_state_error': max(errors),
                     'control_validation_scope': 'saved state and first/last sector columns, both ancilla branches',
                     'sample_count_scope': 'two trajectories/random cell, technical profile only'}
            self.write(f"{cell['id']}_rep{first_task['replica']}_trajectory.json", trajectory_record)
            for task in tasks:
                with self.trace.stage('wrapper_task', task_id=task['id'], control_policy=task['control_policy'], axis=task['axis']):
                    evolution = policies[task['control_policy']]
                    with self.trace.stage('hadamard_wrapper_build'):
                        wrapper = build_native_hadamard_wrapper(evolution, axis=task['axis'], include_measurement=True,
                                   max_instructions=self.caps['untranspiled_instructions'])
                    # Count the same attempted work as v3, including the
                    # pre-compile numeric fingerprint, then invoke the same compiler.
                    self.calls.take('compile')
                    start = time.monotonic()
                    fingerprint = canonical_qiskit_circuit_fingerprint(wrapper)
                    with self.trace.stage('transpile_and_metrics'):
                        cost = transpile_and_measure_cost(wrapper, compiler, actual_circuit_fingerprint=fingerprint)
                        if len(cost.transpiled_circuit.data) > self.caps['transpiled_instructions']:
                            raise RuntimeError('TRANSPILED_INSTRUCTION_CAP')
                        metrics = {name: getattr(cost, name) for name in ('rz_count', 'rz_depth', 'cx_count',
                                                            'cx_depth', 'total_depth', 'circuit_size')}
                    self.write(task['id'] + '_cost.json', {'task': task, 'event_digest': trajectory_record['event_digest'],
                               'metrics': metrics, 'wrapper_fingerprint': cost.actual_circuit_fingerprint,
                               'compiler_settings_hash': cost.compiler_settings_hash,
                               'pretranspile_instructions': len(wrapper.data),
                               'transpiled_instructions': len(cost.transpiled_circuit.data),
                               'classical_compile_wall_seconds': time.monotonic() - start,
                               'worker_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                               'quantum_scope': task['scope'], 'cost_statistic': 'individual trajectory wrapper',
                               'winner_claim': False, 'shot_estimate_performed': False})
                    self.compiled += 1
                    rows.append((task, metrics))
                    # Release before allocating the next wrapper/transpilation.
                    evolution = wrapper = cost = None
            return rows
        finally:
            policies.clear()
            evolution = legacy = wrapper = cost = request = steps = None

    def run(self):
        failure = None
        try:
            with self.trace.bind():
                with self.trace.stage('input_reference'):
                    self.setup()
                with self.trace.stage('correctness'):
                    self.correctness()
                self.costs()
            if source_hashes(self.root) != self.manifest['source_hashes'] or file_hash(self.root / SNAPSHOT) != SNAPSHOT_SHA:
                raise ValueError('INPUT_OR_SOURCE_CHANGED_DURING_RUN')
            status, reason = 'H4_TECHNICAL_PILOT_COMPLETE', None
        except Exception as error:
            status, reason = 'H4_TECHNICAL_PILOT_STOP', type(error).__name__ + ':' + str(error)[:1000]
            self._failure_reserve = None
            try:
                failure = self.trace.failure(error)
            except Exception as diagnostic_error:
                # The parent still classifies a missing/failed terminal as
                # STOP. Diagnostics must not mask the original failure reason.
                failure = {'diagnostic_error': type(diagnostic_error).__name__}
            gc.collect()
        self._failure_reserve = None
        self.write('worker_terminal.json', {'status': status, 'reason': reason,
                  'failure_diagnostics': failure, 'diagnostic_records_written': self.trace.sequence,
                  'completed_correctness_cells': self.completed, 'compiled_wrappers': self.compiled,
                  'calls': self.calls.used, 'mandatory_stop': True, 'next_stage_authorized': False,
                  'numerical_allowance_certified': False, 'accuracy_eligibility': 'UNDETERMINED'})
        return status == 'H4_TECHNICAL_PILOT_COMPLETE'
