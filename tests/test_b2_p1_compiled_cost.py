"""Artificial inputs only; never load saved molecular numerical results."""
from dataclasses import asdict, replace
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from qiskit.quantum_info import Operator

from test_b2_rpe_connection import case, fake_compile
from trotterlib.b2_p1_compiled_cost import price_explicit_trajectory
from trotterlib.o1_m1_resource_bridge import MatchedNoPrepCostProvider
from trotterlib.rte import enumerate_rte_events
from trotterlib.rte_compiled_cost import transpile_and_measure_cost

spec = importlib.util.spec_from_file_location('b2_cost_runner',
    Path(__file__).resolve().parents[1]/'scripts/run_b2_p1_compiled_cost.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.fixture(autouse=True)
def no_molecular_io(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('No saved scientific input or molecular computation')
    monkeypatch.setattr(np, 'load', forbidden)
    monkeypatch.setattr(runner, 'git', forbidden)
    import trotterlib.df_hamiltonian as module
    monkeypatch.setattr(module, 'solve_df_ground_state', forbidden)
    monkeypatch.setattr(module, 'build_df_h_d_from_molecule', forbidden)


def invented():
    _, round_ = case(r=4)
    pool = enumerate_rte_events(round_.tail.components, round_.distribution, max_events=10000)
    negative = next(e for e in pool if e.taylor_order == 2 and e.phase == -1)
    return round_, (pool[0], negative, negative, pool[0])


def test_explicit_pair_is_complete_measured_and_not_an_expectation():
    round_, events = invented()
    observed = []
    def capture(circuit, compiler, **kwargs):
        observed.append(circuit)
        return fake_compile(circuit, compiler, **kwargs)
    result = price_explicit_trajectory(round_, MatchedNoPrepCostProvider(
        round_.context.compiler, compile_function=capture), events)
    assert len(observed) == 2
    assert observed[0].data[:-2] == observed[1].data[:-3]
    assert result['evaluation_method'] == 'specified_trajectory_full_no_prep_compiled_cost'
    assert result['expected_trajectory_cost'] is result['standard_error'] is None
    assert result['classical_sample_count'] is result['total_rpe_cost'] is None
    assert result['state_preparation_included'] is False
    assert result['trajectory_identity'] and result['evolution_fingerprint']
    assert result['request_instruction_upper_bound'] >= result['evolution_instructions']


def test_real_compiler_preserves_artificial_whole_wrapper_channel_and_measurement():
    round_, events = invented()
    captures = []
    def compile_check(circuit, compiler, **kwargs):
        cost = transpile_and_measure_cost(circuit, compiler, **kwargs)
        before = Operator(circuit.remove_final_measurements(inplace=False)).data
        after = Operator(cost.transpiled_circuit.remove_final_measurements(inplace=False)).data
        # The whole measured wrapper may acquire an unobservable global phase.
        # Compare the entire channel and the ancilla measurement effect, with
        # no fitted phase; relative controlled phases affect these comparisons.
        assert np.max(np.abs(np.kron(before.conj(), before)
                             - np.kron(after.conj(), after))) < 1e-10
        z = np.diag([1.] * (len(before)//2) + [-1.] * (len(before)//2))
        assert np.max(np.abs(before.conj().T @ z @ before
                             - after.conj().T @ z @ after)) < 1e-10
        assert cost.transpiled_circuit.count_ops()['measure'] == 1
        assert cost.cx_count == cost.transpiled_circuit.count_ops().get('cx', 0)
        assert cost.rz_count == cost.transpiled_circuit.count_ops().get('rz', 0)
        assert cost.backend_fingerprint == 'no_backend'
        captures.append(cost)
        return cost
    result = price_explicit_trajectory(round_, MatchedNoPrepCostProvider(
        round_.context.compiler, compile_function=compile_check), events)
    assert len(captures) == 2
    assert all(type(value) is int and value >= 0
               for row in result['axes'].values() for value in row.values())


@pytest.mark.parametrize('defect', ('event', 'compiler', 'missing_metric', 'compiler_failure'))
def test_ineligible_input_or_price_stops_without_substitution(defect):
    round_, events = invented()
    calls = []
    def compile_bad(circuit, compiler, **kwargs):
        calls.append(1)
        if defect == 'compiler_failure':
            raise RuntimeError('synthetic compiler failure')
        if defect == 'missing_metric':
            return type('Missing', (), {'compiler': compiler})()
        return fake_compile(circuit, compiler, **kwargs)
    compiler = round_.context.compiler
    if defect == 'compiler':
        compiler = replace(compiler, transpiler_seed=17)
    if defect == 'event':
        events = (replace(events[0], rotation_angle=events[0].rotation_angle+.1), *events[1:])
    with pytest.raises((ValueError, AttributeError, RuntimeError)):
        price_explicit_trajectory(round_, MatchedNoPrepCostProvider(
            compiler, compile_function=compile_bad), events)
    assert len(calls) == (0 if defect in ('event', 'compiler') else 1)


def test_saved_price_profile_requires_the_recorded_runtime_and_backend():
    round_, _ = invented()
    record = {'compiler': asdict(round_.context.compiler)}
    assert runner.compiler_from_saved(record, '1.3.0') == round_.context.compiler
    with pytest.raises(ValueError, match='Qiskit'):
        runner.compiler_from_saved(record, 'different_version')
    record['compiler']['backend_name'] = 'unapproved_backend'
    with pytest.raises(ValueError, match='backend'):
        runner.compiler_from_saved(record, '1.3.0')


def test_saved_representative_case_replays_canonical_signed_events():
    round_, events = invented()
    representatives = [dict(taylor_order=e.taylor_order, phase_real=complex(e.phase).real,
        phase_imag=complex(e.phase).imag, component_ids=list(e.selected_component_ids))
        for e in events[:2]]
    replay, selection = runner.fixed_events(round_,
        {'controlled_correctness': {'representative_events': representatives}})
    assert replay == events
    assert selection['representative_indices'] == (0, 1, 1, 0)
    assert selection['random_sampling'] is selection['price_based_selection'] is False
