"""Artificial two-mode inputs only; no molecular input or saved H4 replay."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import io
import json
import multiprocessing
from pathlib import Path
import sys

import numpy as np
import pytest
from qiskit import qpy
from qiskit.quantum_info import Operator, Statevector

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_h4_p3_price_confirmation as runner
from test_matched_partition_stage3 import tiny
K, A = runner.modules()


def dump(circuit):
    stream = io.BytesIO(); qpy.dump(circuit, stream)
    return stream.getvalue()


@pytest.mark.parametrize('arm', ['P', 'DF'])
def test_full_only_matches_old_F_literal_phase_oracle_and_measured_compiler(arm):
    inputs, saved, retained, events, angle = tiny(arm)
    body, sidecar = A.full_only(inputs, events, retained, angle)
    old, old_sidecar = K.trajectory_parts(inputs, events, retained, angle)
    assert K.canonical_qiskit_circuit_fingerprint(body) == K.canonical_qiskit_circuit_fingerprint(old['F'])
    assert sidecar == old_sidecar
    expected, error = K.DictionaryOracle(inputs, saved, 2).trajectory(events, retained, angle, part='F')
    assert K.program_check(body, expected, error)['status'] == 'PASS'
    for axis, circuit in K.wrappers(body).items():
        output, _ = K.price(circuit)
        z = np.diag([1.]*4 + [-1.]*4)
        before = Operator(circuit.remove_final_measurements(inplace=False)).data
        after = Operator(output.remove_final_measurements(inplace=False)).data
        np.testing.assert_allclose(after.conj().T@z@after, before.conj().T@z@before, atol=1e-12, rtol=0)
        mean = float(Statevector.from_instruction(circuit.remove_final_measurements(inplace=False)).expectation_value(z).real)
        assert mean == pytest.approx(expected[4,4].real if axis == 'cosine' else expected[4,4].imag, abs=1e-13)


def artificial_context(out):
    dictionaries, normalizations, laws, trajectories = {}, {}, {}, {}
    for arm in ('P', 'DF'):
        inputs, saved, retained, events, angle = tiny(arm)
        dictionaries[arm] = saved; trajectories[arm] = events
        normalizations[arm] = dict(lambda_R=K.fsum(abs(x['coefficient']) for x in saved['components']),
            dimensionless_step_time=K.fsum(abs(x['coefficient']) for x in saved['components'])*.1,
            weights=list(inputs.distribution.unnormalized_order_weights),
            order_probabilities=list(inputs.distribution.order_probabilities), beta=inputs.distribution.exact_finite_distribution)
        laws[arm] = dict(saved_Stage2_normalization=normalizations[arm],
            basis_metadata=json.loads(json.dumps([asdict(x) for x in inputs.registry.metadata()])),
            components=[dict(id=c.component_id, probability=c.probability, basis_id=c.basis_id, basis_hash=c.basis_hash) for c in inputs.components])
    decoded = {'half': dump(retained.half), 'half_actual_inverse': dump(retained.inverse),
               'controlled_half': dump(retained.controlled_half)}
    meta = dict(DF={}, common_half_fingerprint=K.canonical_qiskit_circuit_fingerprint(retained.controlled_half))
    for i,b in enumerate(dictionaries['DF']['basis']):
        basis = K.basis_from_saved(np.array(b['U_real']) + 1j*np.array(b['U_imag']))
        decoded[f'basis_DF_{i}'] = dump(basis.circuit)
        meta['DF'][str(i)] = dict(fingerprint=K.canonical_qiskit_circuit_fingerprint(basis.circuit))
    return dict(n=2, q=2, saved={'native_bases.json': meta, 'law_bindings.json': laws}, native_QPY=decoded,
        dictionaries=dictionaries, normalizations=normalizations, reference=retained.reference,
        reference_error=retained.reference_error, angle=angle, out=str(out)), trajectories


def test_saved_basis_law_and_native_workers_restore_without_eigen_or_givens(tmp_path):
    context, events = artificial_context(tmp_path)
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn'),
                             initializer=runner.initialize_native, initargs=(context,)) as pool:
        futures = [pool.submit(runner.native_worker, arm+'_00', arm, events[arm]) for arm in ('P', 'DF')]
        results = [f.result() for f in futures]
    for key, row in results:
        assert row['check']['status'] == 'PASS'
        assert row['operations'] == dict(native_operator_evaluations=1)
        assert len(row['input_QPY']) == 2
    context['saved']['law_bindings.json']['DF']['components'][0]['basis_hash'] = 'bad'
    with pytest.raises(ValueError, match='COMPONENT_BINDING'): runner.restore_context(context)


def test_restored_basis_never_drops_global_phase_or_bad_fingerprint():
    _, _, retained, _, _ = tiny('DF')
    c = retained.basis.circuit
    assert A.basis_ops(A.restored_circuit(dump(c)))
    with pytest.raises(ValueError, match='FINGERPRINT'): A.restored_circuit(dump(c), fingerprint='bad')
    mutated = c.copy(); mutated.global_phase = .2
    with pytest.raises(ValueError, match='PHASE'): A.basis_ops(mutated)


def test_statistics_unbiased_SE_paired_axes_and_independent_arms():
    prices = {}
    for arm, offset in [('P', 0), ('DF', 10)]:
        for i, value in enumerate((1, 3, 5)):
            for axis, factor in [('cosine', 1), ('sine', 2)]:
                prices[f'{arm}_{i:02d}/F/{axis}'] = {m: offset + factor*value for m in A.METRICS}
    result = A.price_statistics(prices, n=3)
    p = result['arms']['P']
    for m in A.METRICS:
        assert p['axes']['cosine'][m]['mean_exact'] == '3'
        assert p['axes']['cosine'][m]['unbiased_sample_variance'] == 4
        assert p['axes']['cosine'][m]['standard_error'] == pytest.approx(np.sqrt(4/3))
        assert p['cosine_sine_sample_covariance_exact'][m] == '8'
        assert result['independent_differences']['cosine'][m]['P_minus_DF'] == -10
        assert result['independent_differences']['cosine'][m]['independent_standard_error'] == pytest.approx(np.sqrt(8/3))
    assert p['population_mean_upper'] is None and result['old_pilot_pooled'] is False
    with pytest.raises(ValueError, match='COMPLETE'): A.price_statistics({**prices, 'old_pilot': {}}, n=3)
    assert set(A.SEEDS['P']).isdisjoint(A.SEEDS['DF']) and all(len(x)==32 for x in A.SEEDS.values())
