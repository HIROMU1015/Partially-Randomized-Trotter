"""Explicit conditional probes. Import is inert; no molecular defaults here."""
from __future__ import annotations

import math
import os

PARENTS = ((0, 1), (8, 2), (10, 4), (11, 8), (13, 16), (17, 32))
TRAINING = '9dcdc4a28866dc223afb0ea212df645a3ed4a28e5c6dad6ddfc0ec1fcb28ba9e'
ATOL = '1e-10'
_PREP = _SUPPORT = _STOP = None


def signal_config(preparation, row):
    """Exact binary64 tau binding, followed by unchanged legacy validators."""
    from trotterlib.rte import RTEConfig, finite_rte_distribution
    r, k, delta = row['r'], row['K'], float(row['delta'])
    if type(r) is not int or r < 1 or k != 2:
        raise ValueError('UNKNOWN/STOP: fixed positive r and K2 required')
    tau = float(row['signal_tau64'])
    tail = preparation.rte_preparation.symbolic_tail
    expected = tail.lambda_r * (delta / r)
    if not math.isfinite(tau) or tau.hex() != expected.hex():
        raise ValueError('UNKNOWN/STOP: signal tau differs; no model-tau fallback')
    law = finite_rte_distribution(tau, k)
    config = RTEConfig(tail_id=tail.tail_id, tail_hash=tail.tail_hash,
        lambda_r=tail.lambda_r, evolution_time=delta, rte_steps=r, step_time=delta/r,
        dimensionless_step_time=tau, taylor_distribution='rte_even_taylor_paired',
        finite_taylor_order=k,
        truncation_tolerance=max(math.nextafter(law.step_truncation_residual_bound, math.inf), math.ulp(0.)),
        distribution_normalization=law.exact_finite_distribution,
        truncation_residual_bound=law.step_truncation_residual_bound, seed=0)
    return config, law


def select_fixed_components(preparation, export):
    components = preparation.rte_preparation.symbolic_tail.components
    indices = {c.component_id: i for i, c in enumerate(components)}
    saved = {c['id']: c for c in export['components']}
    names = ('df-fragment-10:Z0', 'df-fragment-11:Z0')
    for name, sign in zip(names, (-1, 1)):
        if name not in indices or name not in saved:
            raise ValueError('UNKNOWN/STOP: fixed component unavailable')
        c, old = components[indices[name]], saved[name]
        if (c.coefficient_sign != sign or old['sign'] != sign
                or c.basis_id != old['basis_id'] or list(c.diagonal_pauli_support) != old['support']
                or c.coefficient_abs.hex() != float(old['coefficient_abs']).hex()
                or c.probability.hex() != float(old['probability64']).hex()):
            raise ValueError('UNKNOWN/STOP: selected component identity differs')
    return tuple(indices[name] for name in names)


def profiles(components, law, r, indices):
    from trotterlib.rte import _make_event
    a, b = indices
    z0 = _make_event((a,), components, law, law.orders.index(0))
    z1 = _make_event((b,), components, law, law.orders.index(0))
    minus = _make_event((a, b, a), components, law, law.orders.index(2))
    plus = _make_event((b, b, b), components, law, law.orders.index(2))
    all0 = tuple((z0, z1)[i % 2] for i in range(r))
    stress = [(z0, z1)[i % 2] for i in range(2*r)]
    if r == 1:
        stress[:] = [minus, plus]
    else:
        stress[0], stress[r-1], stress[-1] = minus, plus, plus
    return ((1, 'all0', all0), (2, 'stress', tuple(stress)))


def request_for_probe(preparation, config, law, events, q, *, controlled):
    from trotterlib.df_partial_s2_cost import _request_for_events
    from trotterlib.df_partial_s2_repeated import DFPartialS2RepeatedRequest
    r = config.rte_steps
    if q not in (1, 2) or len(events) != q*r:
        raise ValueError('UNKNOWN/STOP: local probe event count')
    steps = [_request_for_events(preparation, step_time=config.evolution_time,
                config=config, distribution=law, events=events[i*r:(i+1)*r], seed=0,
                controlled=controlled,
                ancilla_qubit=preparation.num_system_qubits if controlled else None,
                cancel_adjacent_equal_bases=True) for i in range(q)]
    return DFPartialS2RepeatedRequest.from_step_requests(steps,
        sampling_policy='fixed_conditional_profile_no_rng', construction_policy='raw_concatenation')


def initialize_worker(preparation, support, stop):
    global _PREP, _SUPPORT, _STOP
    _PREP, _SUPPORT, _STOP = preparation, support, stop
    # This process owns these settings. No shared environment is modified.
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'


def evaluate_group(job, *, preparation=None, support=None, stop=None):
    """Same code for toy tests and future H4 jobs; caller supplies all inputs."""
    import numpy as np
    from flint import ctx
    from qiskit.quantum_info import Statevector
    from trotterlib.df_partial_s2_repeated import QiskitDFPartialS2RepeatedCircuitBuilder
    from trotterlib.research_direction_full_scope import apply_run_threshold_policy
    from trotterlib.rpe_hadamard_interrogation import RPEHadamardInterrogationRequest
    from trotterlib.rpe_hadamard_compiled_cost_benchmark import QiskitRPEHadamardBenchmarkCircuitBuilder
    from reference_adapter import scalar_residual, vector_residual
    preparation = _PREP if preparation is None else preparation
    support = _SUPPORT if support is None else support
    stop = _STOP if stop is None else stop
    counts, outputs, arrays = {}, [], {}

    def start(key):
        if stop is not None and stop.is_set():
            raise RuntimeError('STOP: sibling failure; no new action')
        counts[key] = counts.get(key, 0) + 1

    try:
        with np.errstate(over='raise', invalid='raise'):
            ctx.prec = 96
            # No FLINT work is large enough here to benefit from thread teams.
            ctx.threads = 1
            for spec in job['probes']:
                psi, target = job['psi'], spec['target']
                dim = len(psi)
                initial = np.r_[psi, np.zeros(dim)]
                initial1 = np.r_[np.zeros(dim), psi]
                checks, axis_results = {}, {}
                record = {'id': spec['id'], 'parent': job['parent'],
                    'q_probe': spec['q_probe'], 'profile': spec['profile'],
                    'conditional_not_distribution_sample': True,
                    'events': [e.to_dict() for e in spec['events']],
                    'reference_signal': spec['signal'], 'reference_signal_error': spec['signal_error'],
                    'reference_target_error': spec['target_error'], 'checks': checks, 'axes': axis_results, 'pass': False}
                outputs.append(record)
                arrays[spec['id']+'_reference_target'] = target
                unrequest = request_for_probe(preparation, job['config'], job['law'],
                    spec['events'], spec['q_probe'], controlled=False)
                request = request_for_probe(preparation, job['config'], job['law'],
                    spec['events'], spec['q_probe'], controlled=True)
                request = apply_run_threshold_policy(request, support, training_fingerprint=TRAINING)
                builder = QiskitDFPartialS2RepeatedCircuitBuilder()
                start('uncontrolled_builds')
                uncontrolled = builder.build(unrequest, construction_policy='raw_concatenation')
                start('uncontrolled_actions')
                # Qiskit 1.3 can update a shared buffer for circuit global_phase.
                vector = Statevector(psi.copy()).evolve(uncontrolled.circuit).data
                arrays[spec['id']+'_uncontrolled'] = np.asarray(vector)
                checks['uncontrolled'] = vector_residual(vector, target, spec['target_error'])
                if not checks['uncontrolled']['pass']:
                    raise ValueError('INCONSISTENT: uncontrolled reference')
                for policy in ('raw_concatenation', 'boundary_optimized'):
                    start('controlled_'+policy+'_builds')
                    result = builder.build(request, construction_policy=policy)
                    for name, inp, expected, error in (
                        ('zero', initial, initial, '0'),
                        ('one', initial1, np.r_[np.zeros(dim), target], spec['target_error']),
                    ):
                        start('controlled_branch_actions')
                        actual = Statevector(inp.copy()).evolve(result.circuit).data
                        arrays[spec['id']+'_'+policy+'_'+name] = np.asarray(actual)
                        check = vector_residual(actual, expected, error)
                        checks[policy+'_'+name] = check
                        if not check['pass']:
                            raise ValueError('INCONSISTENT: '+policy+'_'+name)
                    if policy == 'boundary_optimized':
                        optimized = result
                wrappers = QiskitRPEHadamardBenchmarkCircuitBuilder(maximum_repetition_count=2)
                for axis, coordinate in (('cosine', 'real'), ('sine', 'imag')):
                    start('wrapper_builds')
                    wrapper = wrappers.build(RPEHadamardInterrogationRequest(optimized, axis, include_measurement=True))
                    start('ideal_mean_actions')
                    vec = Statevector(initial.copy()).evolve(wrapper.circuit.remove_final_measurements(inplace=False)).data
                    arrays[spec['id']+'_'+axis+'_wrapper'] = np.asarray(vec)
                    mean = float(np.sum(abs(vec[:dim])**2)-np.sum(abs(vec[dim:])**2))
                    check = scalar_residual(mean, spec['signal'][coordinate], spec['signal_error'][coordinate])
                    axis_results[axis] = {'actual_mean': mean, **check}
                    if not check['pass']:
                        raise ValueError('INCONSISTENT: '+axis+' mean')
                record.update(support_applications=sum(s.rte_support_restricted_application_count for s in optimized.step_results),
                    constant_phase=optimized.constant_phase, extracted_identity_phase=optimized.extracted_identity_phase,
                    rte_relative_phase=optimized.rte_relative_phase, pass_=True)
                record['pass'] = record.pop('pass_')
        return {'r': job['parent']['r'], 'counts_started': counts, 'probes': outputs, 'pass': True, '_arrays': arrays}
    except Exception as exc:
        if stop is not None:
            stop.set()
        # Preserve partial scientific evidence; no retry or replacement profile.
        return {'r': job['parent']['r'], 'counts_started': counts, 'probes': outputs,
                'pass': False, 'error': type(exc).__name__+': '+str(exc), '_arrays': arrays}
