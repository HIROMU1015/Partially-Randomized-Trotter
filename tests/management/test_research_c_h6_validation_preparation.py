"""Invented matrices/circuits only; no saved molecular arrays or chemistry."""
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src'))
from trottertracks import c_h6_validation_preparation as m
from trottertracks.c1_h6_compiler_precision import compiled_wrapper
from trottertracks.pr_dictionary_validation.fixture import CompilerProfile


def artificial(modes=4):
    # A diagonal controlled circuit with a nontrivial scalar/identity phase.
    q = QuantumCircuit(modes+1)
    q.p(-.137, modes)
    for j in range(modes):
        q.crz(.13*(j+1), modes, j)
        q.cx(j, (j+1) % modes)
        q.crz(-.07, modes, (j+1) % modes)
        q.cx(j, (j+1) % modes)
    w = Operator(q).data[(1 << modes):, (1 << modes):]
    indices = np.array([1, 2, 5, 8])
    return q, indices, w[np.ix_(indices, indices)]


def test_passive_import_and_work_model():
    code = "import sys;sys.path.insert(0,'src');from trottertracks import c_h6_validation_preparation as m;m.specification();m.memory_model();m.workload();assert not any(x in sys.modules for x in ['numpy','qiskit','ffsim','pyscf','jax'])"
    subprocess.run([sys.executable, '-B', '-c', code], cwd=Path(__file__).resolve().parents[2], check=True)
    s = m.specification(); w = m.workload(); memory = m.memory_model()
    assert (s['rank'], s['LD'], s['sector_dimension']) == (14, 4, 400)
    assert w['stochastic_native_max'] == 1088
    assert w['native_kernels_max'] == 1092
    assert w['numerically_checked_kernels_max'] == 26
    assert w['state_action_columns_max'] == 124800
    assert m.workload(numerical_coverage='every_kernel')['state_action_columns_max'] == 5241600
    assert w['maximum_logical_units'] == sum(w['stage_units'].values())
    assert memory['full_controlled_matrix_bytes'] == 2**30
    assert memory['complete_restricted_columns_bytes'] == 104857600


def test_preparation_cli_is_passive_and_has_no_execution_option():
    root = Path(__file__).resolve().parents[2]
    code = "import runpy,sys;sys.argv=['prepare_research_c_h6_validation.py'];runpy.run_path('scripts/prepare_research_c_h6_validation.py',run_name='__main__');assert not any(x in sys.modules for x in ['numpy','qiskit','ffsim','pyscf','jax'])"
    subprocess.run([sys.executable, '-B', '-c', code], cwd=root, check=True)
    result = subprocess.run([sys.executable, '-B', 'scripts/prepare_research_c_h6_validation.py', '--execute'], cwd=root, capture_output=True, text=True)
    assert result.returncode == 2 and 'unrecognized arguments' in result.stderr


@pytest.mark.parametrize('batch', [1, 3, 16])
def test_complete_basis_actual_inverses_and_dense_parity(batch):
    q, indices, w = artificial()
    compiled = compiled_wrapper(q, CompilerProfile())
    with patch('numpy.load', side_effect=AssertionError('NO_MOLECULAR_INPUT')):
        report = m.check_kernel(q, compiled, w, indices, batch_columns=batch)
    assert report['input_dimension'] == 8 and report['output_dimension'] == 32
    assert report['state_action_columns'] == 48
    assert not report['full_Fock_input_coverage']
    assert report['rigorous_enclosure'] is None
    for row in report['checks'].values():
        assert row['covered_columns'] == 8
        assert row['max_entry'] < 1e-13
    target = np.block([[np.eye(16), np.zeros((16, 16))],
                       [np.zeros((16, 16)), Operator(q).data[16:, 16:]]])
    dense = Operator(compiled).data
    ids = np.r_[indices, 16+indices]
    expected_f = np.linalg.norm((dense-target)[:, ids])
    assert report['checks']['compiled_reference']['represented_difference_frobenius'] == pytest.approx(expected_f, abs=2e-14)


@pytest.mark.parametrize('fault', ['scalar_phase', 'leakage', 'inverse'])
def test_phase_leakage_and_independently_wrong_inverse_are_detected(fault):
    q, indices, w = artificial()
    compiled = compiled_wrapper(q, CompilerProfile())
    options = {}
    if fault == 'scalar_phase':
        compiled.global_phase += .001
        # Qiskit's phase-insensitive equivalence is deliberately insufficient.
        assert Operator(compiled).equiv(Operator(q))
    if fault == 'leakage':
        compiled.rx(.001, 0)
    if fault == 'inverse':
        options['inverse_factory'] = lambda circuit: circuit.copy()
    with pytest.raises(ValueError):
        m.check_kernel(q, compiled, w, indices, batch_columns=3, **options)


def test_restricted_scope_does_not_masquerade_as_full_Fock_coverage():
    q, indices, w = artificial()
    changed = q.copy()
    # Fully occupied state is outside this invented subspace.
    changed.mcp(.01, list(range(1, 5)), 0)
    compiled = compiled_wrapper(changed, CompilerProfile())
    report = m.check_kernel(changed, compiled, w, indices)
    assert not report['full_Fock_operator_equality_claim']
    assert np.max(abs(Operator(changed).data-Operator(q).data)) > .001


def rows():
    return [dict(geometry='H6_100', input_sha256='a'*64, arm='PR', start='C0',
        budget=64, strength=g, r=r, candidate_sha256=str(g),
        cohort_role='selection', correctness='PASS', samples=32,
        eligible=True, total_RZ=1000-r+g)
        for g in m.STRENGTHS for r in m.R_VALUES]


def test_H6_grid_keeps_existing_common_and_fair_selection_semantics():
    data = rows(); selected = m.select_training(data)
    assert selected['common_r32']['r'] == 32
    assert selected['fair_retuned']['r'] == 128
    assert selected['fair_retuned'] is data[2]
    for row in data: row['total_RZ'] = 1000.
    assert m.select_training(data)['fair_retuned']['r'] == 32
    for row in data:
        row['eligible'] = False; row['total_RZ'] = None
    assert m.select_training(data) == dict(common_r32=None, fair_retuned=None)


@pytest.mark.parametrize('fault', ['missing', 'final', 'candidate', 'correctness'])
def test_selection_rejects_unequal_opportunities_or_final_data(fault):
    data = rows()
    if fault == 'missing': data.pop()
    if fault == 'final': data[0]['cohort_role'] = 'final'
    if fault == 'candidate': data[0]['candidate_sha256'] = 'different'
    if fault == 'correctness': data[0]['correctness'] = 'FAIL'
    with pytest.raises(ValueError): m.select_training(data)


def test_nonunitary_and_invalid_embedding_rejected():
    q, indices, w = artificial()
    bad = compiled_wrapper(q, CompilerProfile()); bad.reset(0)
    with pytest.raises(ValueError): m.check_kernel(q, bad, w, indices)
    with pytest.raises(ValueError): m.check_kernel(q, q, w, np.array([1, 1, 2, 3]))
    with pytest.raises(ValueError): m.memory_model(batch_columns=0)


def test_sector_only_random_reference_reuses_existing_oracle():
    from fractions import Fraction
    from trottertracks.c1_molecule_extension import SectorCAR
    from trottertracks import c_pr_aware_df_stage_b as b
    ctx = SectorCAR(2)
    j = np.array([[[.1, -.02], [-.02, .2]], [[-.1, .03], [.03, .15]],
                  [[.2, .01], [.01, -.05]]])
    u = np.tile(np.eye(2), (3, 1, 1))
    packet = dict(h=np.diag([.1, .2]), g=np.zeros((2,)*4), constant=np.array(.3))
    arm = b.prepare_arm(**packet, J=j, U=u, role='proposed',
        source_identity='invented', input_identity='invented-input', ld=1)
    generators = ctx.generators(**packet, J=j, U=u)
    with patch.object(b.stage1, 'car_bilinears', side_effect=AssertionError('NO_FULL_FOCK_CAR')):
        result = m.sector_reference_context(arm, generators, ctx, Fraction(1, 10))
    assert result['left'].shape == (4, 4)
    assert all(a.shape == (4, 4) for a in result['atoms'].values())


def test_existing_fixed_N_Welch_accepts_final128_and_new_family_policy():
    from trottertracks import c_near_r32_statistics as statistics
    metrics = statistics.shared.ret.m4.METRICS
    left = [{key: 100+(i % 9) for key in metrics} for i in range(128)]
    right = [{key: 200+(i % 9) for key in metrics} for i in range(128)]
    policy = dict(statistics.POLICY, samples=128, confidence=.975)
    result = statistics.primary(left, right,
        dict(W=1000, shell={key: 5 for key in metrics}),
        dict(W=1000, shell={key: 5 for key in metrics}),
        expected_n=128, statistics=policy)
    assert result['samples_per_arm'] == 128 and result['confidence'] == .975
    assert result['difference_interval'][1] < 0
    assert result['ratio_interval'] is None and result['actual_CA_achieved'] is None
