"""Artificial H4-shaped packets only; saved molecular arrays are never decoded."""
import copy
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from trottertracks import c1_h4_xdf_prefix_budget as m


def fixture(tmp_path, monkeypatch):
    root = tmp_path / 'ARTIFICIAL'
    root.mkdir()
    h = np.diag([-.3, -.1, .2, .4])
    one = np.kron(h, np.eye(2)).astype(complex)
    lam = np.array([1, -1, 2, -2, 3, -3, 4, -4, 5, -5]) * 1e-5
    G = np.repeat(np.eye(8, dtype=complex)[None], 10, axis=0)
    input_path = root / 'artificial.npz'
    np.savez(input_path, one_mo=h, eri_chemist=np.zeros((4,) * 4), nuclear_repulsion=np.array(.5),
        mo_coeff=np.eye(4), df_one_body=one, df_lambdas=lam, df_g_matrices=G,
        unused_saved_energy=np.array('FORBIDDEN_TO_LOAD'))
    indices = m.stage1.sector_indices(4, 2, 2)
    # Independent occupation-number dense reference, in Qiskit little-endian order.
    diagonal = np.array([.5 + sum(one[p, p] for p in range(8) if state >> p & 1) for state in indices])
    np.savez(root / 'physical.npz', matrix=np.diag(diagonal).astype(complex))
    m.io.save(root / 'stage1.json', dict(status='STAGE1_REPRESENTATION_PASS', records=[
        dict(id='sector_DF_' + arm, error=0., allowance=1e-9) for arm in ('XDF', 'CDF')]))
    digest = m.io.sha(input_path)
    m.io.save(root / 'provenance.json', dict(input_sha256=digest))
    m.io.save(root / 'stage3.json', dict(df_error_bounds_ha=dict(XDF=1e-9, CDF=1e-9)))
    (root / 'source.txt').write_text('ARTIFICIAL SOURCE ONLY\n')
    m.io.save(root / 'SOURCE.json', dict(files={'source.txt': m.io.sha(root / 'source.txt')}))
    c = dict(scope=m.SCOPE, status='UNAPPROVED_SCIENCE_CONTRACT', conditions=copy.deepcopy(m.CONDITIONS),
        call_limits=m.LIMITS, maximum_logical_units=356,
        original_input=dict(sha256=digest), source_manifest='SOURCE.json',
        source_manifest_sha256=m.io.sha(root / 'SOURCE.json'),
        records=dict(physical='physical.npz', stage1='stage1.json', provenance='provenance.json', stage3='stage3.json'),
        saved_inputs={p: m.io.sha(root / p) for p in ['physical.npz', 'stage1.json', 'provenance.json', 'stage3.json']},
        evaluation_packages={}, fresh_output=m.FAMILY + '/<execution-date>/run01', **{k: 0 for k in m.FORBIDDEN})
    m.io.save(root / 'contract.json', c)
    monkeypatch.setattr(m.io, 'P_H4_SHA256', digest)
    return root, c, input_path


def execute(root, input_path, **kwargs):
    output = root / m.FAMILY / '2026-10-26/run01'
    # Git identity only is synthetic; evaluator and scientific call journal are real.
    original = m.subprocess.check_output
    def git_only(command, *args, **kw):
        if command[:2] == ['git', 'rev-parse']:
            return 'ARTIFICIAL_SOURCE\n'
        return original(command, *args, **kw)
    with patch.object(m.subprocess, 'check_output', side_effect=git_only):
        result = m.run(root, root / 'contract.json', input_path, output, **kwargs)
    return result, output


def test_passive_preflight_never_decodes_arrays_or_starts_work(tmp_path, monkeypatch):
    root, c, input_path = fixture(tmp_path, monkeypatch)
    with patch.object(np, 'load', side_effect=AssertionError('NO_DECODE')), \
         patch.object(m.stage1, 'legacy_df_module', side_effect=AssertionError('NO_OPERATORS')):
        assert m.preflight(root, root / 'contract.json', input_path) == c
    assert not (root / m.FAMILY).exists()


def test_real_artificial_e2e_356_units_nonmonotone_and_all_nine(tmp_path, monkeypatch):
    root, c, input_path = fixture(tmp_path, monkeypatch)
    forbidden = AssertionError('No new fit, decomposition, or physical Hamiltonian')
    with patch.object(m.stage1, 'fit_once', side_effect=forbidden), \
         patch.object(m.stage1, 'physical_sector', side_effect=forbidden):
        result, output = execute(root, input_path, workers=3)
    rows = result['prefixes']
    assert [r['rank'] for r in rows] == list(range(1, 10))
    assert [r['status'] for r in rows] == ['PASS', 'PASS', 'NOT_MET', 'PASS', 'NOT_MET', 'PASS', 'NOT_MET', 'PASS', 'NOT_MET']
    assert result['first_passing_saved_prefix'] == 1 and result['CA_achieved'] is None
    assert result['saved_anchors']['XDF']['recomputed'] is False
    stop = m.io.read(output / 'STOP.json')
    assert stop['call_counts'] == m.LIMITS and stop['scientific_units'] == 356
    assert not stop['failed'] and stop['retry'] == stop['resume'] == 0
    journal = [json.loads(line) for line in (output / 'science_calls.jsonl').read_text().splitlines()]
    assert sum(x['event'] == 'attempt' for x in journal) == sum(x['event'] == 'COMPLETE' for x in journal) == 356
    assert sum(x['event'] == 'COMPLETE' and x['kind'] == 'xdf_matvec' for x in journal) == 324
    for rank in range(1, 10):
        assert (output / f'prefix_{rank}.json').exists()


def test_permutation_norm_scale_and_guard_equal_old_helper(tmp_path, monkeypatch):
    root, c, input_path = fixture(tmp_path, monkeypatch)
    bundle = m.load_bundle(root, c, input_path)
    ctx = m.context()
    m.identity_gate(bundle, ctx)
    ref = m.attach_reference(bundle)
    p = bundle['packet']
    for rank in (1, 3, 9):
        marks = m.stage1.Calls()
        expected, scale = m.stage1.saved_xdf_sector(p['df_one_body'], p['df_lambdas'][:rank],
            p['df_g_matrices'][:rank], float(p['nuclear_repulsion']), 2, 2, ctx.indices, marks)
        actual = m.prefix_matrix(m.prefix_model(bundle, ctx, rank), ctx, rank, lambda k, label, f: f())
        np.testing.assert_array_equal(actual, expected)
        delta = bundle['physical'] - actual
        norm = m.error_norm(delta)
        assert abs(norm - abs(sum(p['df_lambdas'][:rank])) * 16) < 2e-15
        row = m.assess_prefix(bundle, ref, rank, actual, norm)
        lengths = m.stage1.operation_lengths(4, 36, rank)
        expected_guard = m.stage1.allowance(ref['scale'], scale, lengths['physical'], lengths['xdf'], 36, delta)
        assert row['allowance_Ha'] == expected_guard and row['xdf_scale'] == scale
        assert marks.counts['xdf_matvec'] == 36
    assert not np.array_equal(ctx.perm, np.arange(36))


def test_nonzero_original_g_kappa_is_checked_once_and_scalar_once(tmp_path, monkeypatch):
    root, c, input_path = fixture(tmp_path, monkeypatch)
    b = m.load_bundle(root, c, input_path)
    ctx = m.context()
    p = b['packet']
    p['eri_chemist'] = p['eri_chemist'].copy()
    p['eri_chemist'][0, 1, 1, 0] = .25
    p['df_one_body'] = np.kron(p['one_mo'] - np.diag([.125, 0, 0, 0]), np.eye(2)).astype(complex)
    with patch.object(m.stage1, 'corrected_kappa', wraps=m.stage1.corrected_kappa) as correction:
        assert m.identity_gate(b, ctx)['kappa_max_entry'] == 0
        assert correction.call_count == 1
        matrix = m.prefix_matrix(m.prefix_model(b, ctx, 3), ctx, 3, lambda k, label, f: f())
        assert correction.call_count == 1
    expected = [float(p['nuclear_repulsion']) + sum(p['df_one_body'][i, i] for i in range(8) if state >> i & 1)
                + 16 * sum(p['df_lambdas'][:3]) for state in ctx.indices]
    np.testing.assert_allclose(matrix, np.diag(expected), atol=2e-15, rtol=0)


def test_prefix_slices_keep_lambda_G_pair_order_and_saved_base(tmp_path, monkeypatch):
    root, c, input_path = fixture(tmp_path, monkeypatch)
    b = m.load_bundle(root, c, input_path)
    ctx = m.context()
    original = ctx.df.DFHamiltonian
    p = b['packet']
    tagged = p['df_g_matrices'].copy()
    for i in range(10):
        tagged[i] *= i + 1
    p['df_g_matrices'] = tagged
    with patch.object(ctx.df, 'DFHamiltonian', wraps=original) as constructor:
        m.prefix_model(b, ctx, 5)
    scalar, one, lambdas, matrices, meta = constructor.call_args.args
    assert scalar == .5 and one is p['df_one_body']
    np.testing.assert_array_equal(lambdas, p['df_lambdas'][:5])
    for i, G in enumerate(matrices):
        np.testing.assert_array_equal(G, tagged[i])


def test_all_not_met_is_normal_and_uses_saved_anchor(tmp_path, monkeypatch):
    root, c, input_path = fixture(tmp_path, monkeypatch)
    original = m.assess_prefix
    def fail_budget(b, ref, rank, matrix, error):
        return original(b, ref, rank, matrix, 1.)  # synthetic norm observation
    monkeypatch.setattr(m, 'assess_prefix', fail_budget)
    result, output = execute(root, input_path, workers=1)
    assert [x['status'] for x in result['prefixes']] == ['NOT_MET'] * 9
    assert result['first_passing_saved_prefix'] == 10
    assert not m.io.read(output / 'STOP.json')['failed']


@pytest.mark.parametrize('defect', ['kappa', 'basis', 'permutation', 'nonfinite', 'physical_hermiticity', 'prefix_hermiticity', 'matvec'])
def test_correctness_fail_stops_and_preserves_partial_evidence(tmp_path, monkeypatch, defect):
    root, c, input_path = fixture(tmp_path, monkeypatch)
    original_load, original_context = m.load_bundle, m.context
    def load(*args):
        b = original_load(*args)
        if defect == 'kappa':
            b['packet']['df_one_body'] = b['packet']['df_one_body'].copy() + .01 * np.eye(8)
        if defect == 'basis':
            b['basis_order'] = 'WRONG_ORDER'
        if defect in ('nonfinite', 'physical_hermiticity'):
            b['physical'] = b['physical'].copy()
            b['physical'][0, 1] = np.nan if defect == 'nonfinite' else .1j
        return b
    def ctx():
        x = original_context()
        if defect == 'permutation':
            x.perm = x.perm[::-1]
        return x
    monkeypatch.setattr(m, 'load_bundle', load)
    monkeypatch.setattr(m, 'context', ctx)
    if defect == 'matvec':
        def bad_model(bundle, ctx, rank):
            op, counter = m_model(bundle, ctx, rank)
            if rank == 3:
                from scipy.sparse.linalg import LinearOperator
                def broken(v): raise ValueError('ARTIFICIAL_MATVEC_FAIL')
                op = LinearOperator(op.shape, matvec=broken, dtype=complex)
            return op, counter
        m_model = m.prefix_model
        monkeypatch.setattr(m, 'prefix_model', bad_model)
    if defect == 'prefix_hermiticity':
        old_matrix = m.prefix_matrix
        def bad_matrix(model, ctx, rank, call):
            x = old_matrix(model, ctx, rank, call)
            if rank == 3: x[0, 1] += .01j
            return x
        monkeypatch.setattr(m, 'prefix_matrix', bad_matrix)
    with pytest.raises(ValueError):
        execute(root, input_path, workers=1)
    output = root / m.FAMILY / '2026-10-26/run01'
    stop = m.io.read(output / 'STOP.json')
    assert stop['failed'] and stop['scientific_units'] < 356 and stop['retry'] == stop['resume'] == 0
    assert (output / 'SOURCE.json').exists() and (output / 'failure.json').exists()
    if defect in ('matvec', 'prefix_hermiticity'):
        assert [x['rank'] for x in m.io.read(output / 'partial_prefixes.json')] == [1, 2]
    journal = [json.loads(line) for line in (output / 'science_calls.jsonl').read_text().splitlines()]
    first_fail = next(i for i, x in enumerate(journal) if x['event'] == 'FAIL')
    assert not any(x['event'] == 'attempt' for x in journal[first_fail + 1:])


@pytest.mark.parametrize('file', ['source.txt', 'physical.npz', 'artificial.npz'])
def test_source_or_input_drift_fails_passive_preflight(tmp_path, monkeypatch, file):
    root, c, input_path = fixture(tmp_path, monkeypatch)
    (root / file).write_bytes(b'ARTIFICIAL_DRIFT')
    with patch.object(np, 'load', side_effect=AssertionError('NO_DECODE')):
        with pytest.raises(ValueError, match='SOURCE|INPUT'):
            m.preflight(root, root / 'contract.json', input_path)
    assert not (root / m.FAMILY).exists()


@pytest.mark.parametrize('defect', ['ranks', 'calls', 'retry', 'environment'])
def test_contract_or_environment_mismatch_stops_before_decode(tmp_path, monkeypatch, defect):
    root, c, input_path = fixture(tmp_path, monkeypatch)
    c = copy.deepcopy(c)
    if defect == 'ranks': c['conditions']['ranks'] = [1, 2]
    if defect == 'calls': c['call_limits']['xdf_matvec'] = 323
    if defect == 'retry': c['retry'] = 1
    if defect == 'environment': c['evaluation_packages'] = {'numpy': 'WRONG'}
    (root / 'contract.json').unlink()
    m.io.save(root / 'contract.json', c)
    with patch.object(np, 'load', side_effect=AssertionError('NO_DECODE')):
        with pytest.raises(ValueError, match='SCOPE|ENVIRONMENT'):
            m.preflight(root, root / 'contract.json', input_path)


def test_parallel_failure_drains_started_calls_without_new_dependent_work(tmp_path, monkeypatch):
    root, c, input_path = fixture(tmp_path, monkeypatch)
    original = m.assess_prefix
    def wrong(b, ref, rank, matrix, error):
        if rank == 3: raise ValueError('ARTIFICIAL_PARALLEL_FAIL')
        return original(b, ref, rank, matrix, error)
    monkeypatch.setattr(m, 'assess_prefix', wrong)
    with pytest.raises(ValueError, match='CORRECTNESS_STOP'):
        execute(root, input_path, workers=3)
    output = root / m.FAMILY / '2026-10-26/run01'
    rows = [json.loads(line) for line in (output / 'science_calls.jsonl').read_text().splitlines()]
    first = next(i for i, row in enumerate(rows) if row['event'] == 'FAIL')
    assert not any(row['event'] == 'attempt' for row in rows[first + 1:])
    assert sum(row['event'] == 'attempt' for row in rows) == sum(row['event'] in ('COMPLETE', 'FAIL') for row in rows)
    assert m.io.read(output / 'failure.json')['worker_errors']


@pytest.mark.parametrize('rank', [0, 10, 11, -1, 1.5, True])
def test_unregistered_rank_never_builds_model(rank):
    with pytest.raises(ValueError, match='PREFIX_1_TO_9'):
        m.prefix_model(None, None, rank)


def test_call_limit_and_fresh_output_no_resume(tmp_path, monkeypatch):
    root, c, input_path = fixture(tmp_path, monkeypatch)
    result, output = execute(root, input_path, workers=1)
    with pytest.raises(ValueError, match='FRESH_OUTPUT'):
        execute(root, input_path, workers=1)
    call = m.integrated.Calls(tmp_path, dict(x=1), 1)
    call('x', 'first', lambda: None)
    with pytest.raises(ValueError, match='TOTAL_CALL_LIMIT'):
        call('x', 'second', lambda: pytest.fail('must not run'))


def test_import_is_passive_in_fresh_python():
    script = "import sys; sys.path.insert(0,sys.argv[1]); from trottertracks import c1_h4_xdf_prefix_budget; assert 'numpy' not in sys.modules"
    subprocess.run([sys.executable, '-c', script, str(ROOT / 'src')], check=True)
