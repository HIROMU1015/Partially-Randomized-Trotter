"""All numerical data invented; saved H4 is never decoded by these tests."""
from copy import deepcopy
from fractions import Fraction as F
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from trottertracks import c_pr_aware_df_kappa_diagnostic as d
from trottertracks import c1_stage1 as stage
from trottertracks.c1_stage3_integrated import Calls


def payload(n=4, complex_=False):
    dtype = np.complex128 if complex_ else np.float64
    return np.zeros((n, n), dtype=dtype), np.zeros((n,) * 4, dtype=dtype), np.array(1.)


def diagnose(p):
    h, g, c = p
    T, k = d.generate(*p)
    binary = d.binary64_diagnostics(*p, T, k)
    exact = d.dyadic_diagnostics(h, g, T, k)
    return binary, exact, d.classify(binary, exact)


@pytest.mark.parametrize('complex_', [False, True])
def test_exact_hermitian_no_mutation(complex_):
    p = payload(complex_=complex_)
    h, g, _ = p
    h[0, 1] = 1 + (2j if complex_ else 0)
    h[1, 0] = h[0, 1].conjugate()
    g[0, 0, 0, 0] = 2
    before = [x.copy() for x in p]
    b, e, r = diagnose(p)
    assert all(b[name]['exact_Hermitian'] for name in ('h', 'T', 'kappa'))
    assert r['literal_kappa_exact_Hermitian']
    assert not r['observations']['C_arithmetic_alone_breaks_exact_formula_symmetry']
    for a, old in zip(p, before):
        assert np.array_equal(a, old)
    assert not e['changed_input']


@pytest.mark.parametrize('size,within', [(2.**-50, True), (.1, False)])
@pytest.mark.parametrize('where', ['h', 'T'])
def test_input_causes_not_arithmetic_only(where, size, within):
    h, g, c = payload()
    if where == 'h':
        h[0, 1] = size
    else:
        g[0, 0, 0, 1] = 2 * size
    b, _, r = diagnose((h, g, c))
    assert not b['kappa']['exact_Hermitian']
    assert r['stage1_numerical_allowance_PASS'] == within
    obs = r['observations']
    assert obs['A_stored_h_asymmetry'] == (where == 'h')
    assert obs['B_exact_contraction_of_stored_g_asymmetry'] == (where == 'T')
    assert not obs['C_arithmetic_alone_breaks_exact_formula_symmetry']
    assert r['raw_CAR_bound'] is None and r['CAR_target'] is None


def test_binary64_contraction_loses_exact_symmetry():
    h, g, c = payload(3)
    g[0, 0, 0, 1], g[0, 1, 1, 1], g[0, 2, 2, 1] = 2.**53, 1., -2.**53
    g[1, 0, 0, 0], g[1, 1, 1, 0], g[1, 2, 2, 0] = 2.**53, -2.**53, 1.
    b, e, r = diagnose((h, g, c))
    assert not b['kappa']['exact_Hermitian']
    assert e['matrices']['delta_T_exact']['nonzero_indices'] == []
    assert r['observations']['C_arithmetic_alone_breaks_exact_formula_symmetry']
    assert r['observations']['contraction_antiHermitian_rounding']
    assert not r['observations']['B_exact_contraction_of_stored_g_asymmetry']


def test_identical_contracted_summands_preserve_symmetry():
    # Unlike the cancellation/reordered fixture above, enforce pairwise equality
    # of every contracted summand. No molecular tensors are used.
    h, g, c = payload()
    rng = np.random.default_rng(17)
    g[:] = rng.normal(size=g.shape)
    g[:] = (g + g.transpose(3, 2, 1, 0)) / 2
    h[:] = rng.normal(size=h.shape)
    h[:] = (h + h.T) / 2
    b, e, r = diagnose((h, g, c))
    assert b['T']['exact_Hermitian'] and b['kappa']['exact_Hermitian']
    assert not e['matrices']['delta_T_exact']['nonzero_indices']
    assert not r['observations']['C_arithmetic_alone_breaks_exact_formula_symmetry']


def test_inherited_asymmetry_can_cancel_in_kappa():
    p = payload()
    p[0][0, 1] = .25
    p[1][0, 0, 0, 1] = .5
    b, _, r = diagnose(p)
    assert r['observations']['A_stored_h_asymmetry']
    assert r['observations']['B_exact_contraction_of_stored_g_asymmetry']
    assert r['literal_kappa_exact_Hermitian']
    assert b['kappa']['residual']['Frobenius'] == 0


def test_complex_conjugation_and_exact_half_decomposition():
    h, g, c = payload(2, True)
    h[0, 0], h[0, 1], h[1, 0] = 1j, 2+3j, 2-1j
    b, e, _ = diagnose((h, g, c))
    m = e['matrices']
    k, H, A = (m[key]['entries'] for key in
               ('kappa_literal', 'kappa_H_exact_diagnostic_only', 'kappa_A_exact_diagnostic_only'))
    for i in range(2):
        for j in range(2):
            for part in (0, 1):
                assert F(k[i][j][part]) == F(H[i][j][part]) + F(A[i][j][part])
            assert F(H[i][j][0]) == F(H[j][i][0])
            assert F(H[i][j][1]) == -F(H[j][i][1])
            assert F(A[i][j][0]) == -F(A[j][i][0])
            assert F(A[i][j][1]) == F(A[j][i][1])
    assert [0, 0] in b['kappa']['unequal_indices']


def test_norms_and_registered_stage1_expression():
    p = payload()
    p[0][0, 1] = 3.
    b, _, _ = diagnose(p)
    r = b['kappa']['residual']
    assert r['Frobenius'] == pytest.approx(3 * 2.**.5)
    assert r['max_abs_entry'] == 3
    assert r['spectral2_numerical_pilot'] == 3
    delta = p[0] - p[0].T
    B = 1. + 2 * np.sum(abs(p[0]))
    m = stage.operation_lengths(4, 256, 8)['base']
    assert b['stage1_comparison']['allowance'] == 2 * stage.allowance(B, B, m, m, 4, delta)
    assert r['rigorous_spectral_bound'] is None


def test_stage1_numpy_norm_underflow_is_reported_separately():
    p = payload()
    p[0][0, 1] = np.nextafter(0., 1.)
    b, e, r = diagnose(p)
    assert b['stage1_comparison']['error'] == 0
    assert b['stage1_comparison']['stable_Frobenius_observation'] > 0
    assert e['matrices']['delta_kappa_returned']['nonzero_indices']
    assert not r['literal_kappa_exact_Hermitian']


def test_zero_scale_and_missing_stay_unresolved():
    b, _, _ = diagnose(payload())
    assert b['relative_antiHermitian_Frobenius'] is None
    assert b['relative_status'] == 'UNRESOLVED_ZERO_SCALE'
    for pair in ((None, None), (b, None), (None, {})):
        r = d.classify(*pair)
        assert r['status'].startswith('UNRESOLVED') and r['raw_CAR_bound'] is None


@pytest.mark.parametrize('bad', ['nonfinite', 'shape', 'dtype', 'complex_scalar'])
def test_invalid_input_stops(bad):
    h, g, c = payload()
    if bad == 'nonfinite': h[0, 1] = np.nan
    if bad == 'shape': g = g[:2]
    if bad == 'dtype': h, g = h.astype('float32'), g.astype('float32')
    if bad == 'complex_scalar': c = np.array(1j)
    with pytest.raises(ValueError): d.generate(h, g, c)


def contract(tmp_path):
    (tmp_path / 'source.py').write_text('synthetic source')
    (tmp_path / 'current.npz').write_bytes(b'not decoded in passive mode')
    (tmp_path / 'original.npz').write_bytes(b'not decoded either')
    d.io.save(tmp_path / 'SOURCE.json', {'files': {'source.py': d.io.sha(tmp_path / 'source.py')}})
    c = dict(scope=d.SCOPE, implementation=d.VERSION, base_commit=d.BASE,
        status='UNAPPROVED', policy=deepcopy(d.POLICY), call_limits=d.LIMITS,
        maximum_logical_units=6, fresh_output=d.FRESH, executable=True,
        source_manifest='SOURCE.json', source_manifest_sha256=d.io.sha(tmp_path / 'SOURCE.json'),
        inputs={name: d.io.sha(tmp_path / name) for name in ('current.npz', 'original.npz')},
        evidence={}, input='current.npz', original_input='original.npz',
        packages={}, python=list(sys.version_info[:3]))
    path = tmp_path / 'contract.json'
    d.io.save(path, c)
    return c, path


def test_passive_never_loads_arrays(tmp_path, monkeypatch):
    _, path = contract(tmp_path)
    monkeypatch.setattr(np, 'load', lambda *a, **k: pytest.fail('passive NPZ decode'))
    assert d.preflight(tmp_path, path)['scientific_calls'] == 0
    assert not (tmp_path / d.FRESH).exists()


@pytest.mark.parametrize('filename,tag', [('source.py', 'SOURCE'), ('current.npz', 'INPUT')])
def test_identity_mismatch(tmp_path, filename, tag):
    _, path = contract(tmp_path)
    (tmp_path / filename).write_text('changed')
    with pytest.raises(ValueError, match=tag): d.preflight(tmp_path, path)


@pytest.mark.parametrize('field,value', [('maximum_logical_units', 7), ('fresh_output', 'old_run'),
                                       ('base_commit', 'wrong'), ('status', 'APPROVED')])
def test_scope_mismatch(tmp_path, field, value):
    c, path = contract(tmp_path)
    c[field] = value
    path.write_text(json.dumps(c))
    with pytest.raises(ValueError, match='DIAGNOSTIC_SCOPE'): d.preflight(tmp_path, path)


def test_whitelist_loader_and_array_provenance(tmp_path):
    c, _ = contract(tmp_path)
    h, g, const = payload()
    np.savez(tmp_path / 'current.npz', h=h, g=g, constant=const)
    # Unrelated object array would fail allow_pickle=False if accessed.
    np.savez(tmp_path / 'original.npz', one_mo=h, eri_chemist=g,
             nuclear_repulsion=const, forbidden_state=np.array([object()], dtype=object))
    c['inputs'] = {name: d.io.sha(tmp_path / name) for name in c['inputs']}
    pair = d.load_inputs(tmp_path, c)
    assert d.input_identity(pair)['status'].endswith('PASS')
    pair[1][0][0, 1] = 1
    with pytest.raises(ValueError, match='ORIGINAL_INPUT_ARRAY'): d.input_identity(pair)


def test_success_e2e_synthetic_six_calls(tmp_path):
    c, _ = contract(tmp_path)
    out = tmp_path / 'test-output'
    out.mkdir()
    calls = Calls(out, d.LIMITS, 6)
    p = payload()
    r = d.execute(tmp_path, c, out, calls, loader=lambda *_: (p, deepcopy(p)))
    assert r['status'] == 'DIAGNOSTIC_COMPLETE'
    assert dict(calls.counts) == d.LIMITS
    assert all(r[k] is None for k in ('raw_CAR_bound', 'model_bridge', 'CA_achieved', 'shots'))


def test_partial_stop_no_dependent_work_or_retry(tmp_path, monkeypatch):
    _, path = contract(tmp_path)
    p = payload()
    def synthetic(root, c, out, call):
        call('input_load', 'synthetic', lambda: None)
        d.io.save(out / 'partial.json', {'scientific_input': 'invented'})
        call('input_identity', 'injected mismatch',
             lambda: (_ for _ in ()).throw(ValueError('ORIGINAL_INPUT_ARRAY_MISMATCH')))
        pytest.fail('dependent work must not start')
    monkeypatch.setattr(d, 'execute', synthetic)
    with pytest.raises(ValueError, match='ARRAY_MISMATCH'): d.run(tmp_path, path)
    out = tmp_path / d.FRESH
    stop = d.io.read(out / 'STOP.json')
    assert stop['scientific_units'] == 2 and stop['mandatory_stop']
    assert stop['retry'] == stop['resume'] == 0
    assert (out / 'partial.json').exists() and (out / 'failure.json').exists()
    with pytest.raises(ValueError, match='FRESH_OUTPUT'): d.run(tmp_path, path)


def test_exact_failure_retains_binary_partial(tmp_path, monkeypatch):
    c, path = contract(tmp_path)
    h, g, const = payload()
    np.savez(tmp_path / 'current.npz', h=h, g=g, constant=const)
    np.savez(tmp_path / 'original.npz', one_mo=h, eri_chemist=g, nuclear_repulsion=const)
    c['inputs'] = {name: d.io.sha(tmp_path / name) for name in c['inputs']}
    path.write_text(json.dumps(c))
    def fail(*_):
        raise ValueError('EXACT_DECOMPOSITION_IDENTITY_FAIL')
    monkeypatch.setattr(d, 'dyadic_diagnostics', fail)
    with pytest.raises(ValueError, match='DECOMPOSITION_IDENTITY'): d.run(tmp_path, path)
    out = tmp_path / d.FRESH
    assert (out / 'binary64.json').exists()
    assert not (out / 'result.json').exists()
    stop = d.io.read(out / 'STOP.json')
    assert stop['scientific_units'] == 5
    assert 'classification' not in stop['counts']


def test_existing_exact_gate_not_weakened():
    from trottertracks.c_pr_aware_df_fixed_car import _hermitian_input
    h, _, _ = payload()
    _hermitian_input(h)
    h[0, 1] = 2.**-50
    with pytest.raises(ValueError, match='EXACT_HERMITIAN_INPUT_REQUIRED'):
        _hermitian_input(h)
