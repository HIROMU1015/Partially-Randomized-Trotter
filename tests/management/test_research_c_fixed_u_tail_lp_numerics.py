"""Invented matrices and artificial CAR only; never decode stored H4 arrays."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import warnings

import numpy as np
import pytest
from scipy.optimize import linprog, OptimizeWarning
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(Path(__file__).parent))
from trottertracks import c_fixed_u_tail_lp_numerics as m
import test_research_c_fixed_u_tail_lp as fixtures
lp = m.lp


def fixed_u(seed):
    rng = np.random.default_rng(seed)
    U = np.array([np.linalg.qr(rng.normal(size=(4, 4)))[0] for _ in range(6)])
    a = rng.normal(size=(6, 4, 4))*.02
    return lp.problem(a+a.transpose(0, 2, 1), U)


def algebraic(A, T, j0, b=None):
    A, T, j0 = np.array(A, float), np.array(T, float), np.array(j0, float)
    singular = np.linalg.svd(A, compute_uv=False)
    cutoff = np.finfo(float).eps*max(A.shape)*singular[0]
    rank = int(np.sum(singular > cutoff))
    return lp.Problem(SimpleNamespace(q=len(j0), coefficients=lambda x: T@x),
        j0, A, csr_matrix(T), A@j0 if b is None else np.asarray(b),
        dict(retained_condition=float(singular[0]/singular[rank-1]),
             effective_rank=rank, diagnostic_cutoff=float(cutoff),
             equality_rows_dropped=False))


def evaluate(p, *, cutoff=None, presolve=True, shifted=False):
    """Study controls only. Shift retains b-Aj0 rather than replacing it by 0."""
    def solver(*a, **k):
        k['options']['presolve'] = presolve
        if cutoff is not None:
            k['options']['small_matrix_value'] = cutoff
        if shifted:
            q = p.mapping.q
            c0 = p.T@p.j0
            k['b_eq'] = p.b-p.A@p.j0
            k['b_ub'] = np.r_[-c0, c0]
            raw = linprog(*a, **k)
            if raw.x is not None:
                raw.x[:q] += p.j0
            return raw
        return linprog(*a, **k)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter('always')
        raw = lp.solve_once(p, solver=solver)
    report = lp.diagnose(p, raw)
    return dict(status=report['status'], solver_success=bool(raw.success),
        solver_status=int(raw.status), numerical_optimality=report['numerical_optimality'],
        original_lambda=report['original_lambda'], new_lambda=report.get('new_lambda'),
        original_equality_residual=report['original_equality_residual'],
        residuals=report.get('residuals'), gap=report.get('primal_dual_gap'),
        objective_guard=report.get('objective_arithmetic_guard'),
        dual_objective=report.get('dual_objective_pilot'),
        objective_limit=(lp.ALLOWANCES['objective_absolute']+
            lp.ALLOWANCES['objective_relative']*max(abs(report['epigraph_objective']),
                abs(report['dual_objective_pilot']), report['original_lambda']))
            if 'epigraph_objective' in report else None,
        options=dict(lp.OPTIONS, presolve=presolve, **({} if cutoff is None else
                         dict(small_matrix_value=cutoff))),
        coordinates='j=j0+d; exact floating RHS b-Aj0' if shifted else 'original j',
        warnings=[dict(category=w.category.__name__, message=str(w.message)) for w in seen])


def synthetic_study():
    rows = []
    profiles = [('default', None, True, False), ('presolve_off', None, False, False),
        ('shift_only', None, True, True), ('retention_v2', 1e-12, True, False)]
    with threadpool_limits(limits=1):
        for seed in (7, 19, 71):
            p = fixed_u(seed)
            for name, cutoff, presolve, shifted in profiles:
                rows.append(dict(case='invented_fixed_U_'+str(seed), profile=name,
                    **evaluate(p, cutoff=cutoff, presolve=presolve, shifted=shifted)))
        for perturb in (0., 2*np.finfo(float).eps):
            p = algebraic([[1, 1], [2, 2], [1, 1]], [[1, 0], [0, 2]], [.25, .75])
            p.b[-1] += perturb
            for name, cutoff, presolve, shifted in profiles:
                rows.append(dict(case='redundant_RHS_'+str(perturb), profile=name,
                    **evaluate(p, cutoff=cutoff, presolve=presolve, shifted=shifted)))
        for eps in (1e-8, 1e-9):
            p = algebraic([[1, 1, 5e-10], [1, 1+eps, 0]],
                          np.diag([4., 4., .25]), [1, 0, 1])
            for name, cutoff in [('default', None), ('retention_v2', 1e-12)]:
                rows.append(dict(case='near_dependent_'+str(eps), profile=name,
                                 **evaluate(p, cutoff=cutoff)))
        # 5e-12 lies between the selected minimum and the 1e-11 alternative.
        p = algebraic([[5e-12]], [[1]], [1000.])
        for name, cutoff in [('default', None), ('retention_v2', 1e-12),
                             ('cutoff_1em11_control', 1e-11)]:
            rows.append(dict(case='native_option_sentinel', profile=name,
                             **evaluate(p, cutoff=cutoff)))
    return dict(schema='C_FIXED_U_LP_INVENTED_NUMERICS_V1',
        inputs='Entirely invented seeds/matrices; no saved molecular arrays decoded',
        rows=rows, molecular_scientific_calls=0,
        statement='Numerical controls only; not a diagnosis or qualification of the stopped H4 LP')


def invented(root):
    c = fixtures.invented(root)
    for key, value in dict(parent_LP_result=dict(status='UNRESOLVED_LP', candidate=None),
        parent_LP_STOP=dict(failed=False, scientific_units=9,
            call_counts=dict(LP_solve=1), retry=0, resume=0, mandatory_stop=True),
        parent_LP_SOURCE=dict(execution_commit='56a6acb8f7f48cd4792c560b64bc4f4aa270e0ad')).items():
        name = key+'.json'
        fixtures.write(root/name, value)
        c['records'][key] = name
        c['saved_inputs'][name] = m.io.sha(root/name)
    c.update(scope=m.SCOPE, implementation=m.VERSION, base_commit=m.BASE,
        policy=m.POLICY, solver_options=m.OPTIONS, option_warning_policy=m.WARNING_POLICY,
        fresh_output=m.FRESH)
    fixtures.write(root/'contract.json', c)
    return c


def test_passive_import():
    code = "import sys;sys.path.insert(0,'src');from trottertracks import c_fixed_u_tail_lp_numerics as m;assert not any(k in sys.modules for k in ('numpy','scipy','ffsim','qiskit','jax'));print(m.MAX_CALLS)"
    result = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT,
                            capture_output=True, text=True, check=True)
    assert int(result.stdout) == 50 and m.LIMITS == m.previous.LIMITS


def test_original_full_rows_rhs_signed_bounds_objective_and_options():
    p = fixed_u(7)
    before = (p.A.copy(), p.b.copy(), p.T.copy(), p.j0.copy())
    options = lp.OPTIONS
    def spy(c, **kwargs):
        np.testing.assert_array_equal(kwargs['A_eq'].toarray()[:, :60], p.A)
        assert kwargs['b_eq'] is p.b
        assert kwargs['bounds'][:60] == [(None, None)]*60
        assert kwargs['options'] == m.OPTIONS and kwargs['method'] == 'highs-ds'
        return object()
    with m.option_scope():
        lp.solve_once(p, solver=spy)
    assert lp.OPTIONS is options
    for a, b in zip(before, (p.A, p.b, p.T, p.j0)):
        np.testing.assert_array_equal(a.toarray() if hasattr(a, 'toarray') else a,
                                      b.toarray() if hasattr(b, 'toarray') else b)
    with pytest.raises(RuntimeError), m.option_scope():
        raise RuntimeError('invented exception')
    assert lp.OPTIONS is options


def test_duplicate_warning_source_profile_and_unexpected_native_failure():
    good = [dict(category='OptimizeWarning', message=s) for s in m.expected_warnings()]
    m.warning_gate(good)
    for bad in ([], good[:1], good+good[-1:], good+[dict(category='RuntimeWarning', message='invented')]):
        with pytest.raises(ValueError, match='OPTION_HANDLING'):
            m.warning_gate(bad)


def test_native_option_effect_not_inferred_from_warning_text():
    p = algebraic([[5e-12]], [[1]], [1000.])
    baseline, new = evaluate(p), evaluate(p, cutoff=1e-12)
    assert not baseline['solver_success'] and new['numerical_optimality']
    assert new['new_lambda'] == pytest.approx(1000.)
    m.warning_gate(new['warnings'])
    assert not evaluate(p, cutoff=1e-11)['solver_success']


def test_synthetic_study_reproduces_dual_issue_without_relaxing_gates():
    result = synthetic_study()
    rows = {(r['case'], r['profile']): r for r in result['rows']}
    bad, new = rows['invented_fixed_U_7', 'default'], rows['invented_fixed_U_7', 'retention_v2']
    assert bad['solver_success'] and not bad['numerical_optimality']
    assert bad['residuals']['dual_equality'] > lp.ALLOWANCES['dual_absolute']
    assert new['numerical_optimality'] and new['residuals']['dual_equality'] < 1e-9
    assert not rows['invented_fixed_U_7', 'shift_only']['numerical_optimality']
    for seed in (7, 19, 71):
        m.warning_gate(rows['invented_fixed_U_'+str(seed), 'retention_v2']['warnings'])
        assert rows['invented_fixed_U_'+str(seed), 'retention_v2']['numerical_optimality']
    for perturb in (0., 2*np.finfo(float).eps):
        for profile in ('default', 'presolve_off', 'shift_only', 'retention_v2'):
            assert rows['redundant_RHS_'+str(perturb), profile]['numerical_optimality']
    # Keeping small entries is not a universal repair: near dependence and
    # cancellation can still make the ORIGINAL guards unqualified.
    for eps in (1e-8, 1e-9):
        assert not rows['near_dependent_'+str(eps), 'retention_v2']['numerical_optimality']
    assert result['molecular_scientific_calls'] == 0


def test_bad_original_duals_still_fail():
    p = lp.problem(np.array([[[1.]], [[2.]]]), np.ones((2, 1, 1)))
    with m.option_scope(), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = lp.solve_once(p)
    r.eqlin.marginals[0] += 1e-6
    d = lp.diagnose(p, r)
    assert not d['numerical_optimality'] and d['strict_improvement'] is None


def test_new_preflight_passive_and_parent_or_input_mismatch(tmp_path, monkeypatch):
    with threadpool_limits(limits=1):
        c = invented(tmp_path)
    monkeypatch.setattr(np, 'load', lambda *a, **k: pytest.fail('passive numerical decode'))
    monkeypatch.setattr(lp, 'solve_once', lambda *a, **k: pytest.fail('passive LP'))
    assert m.preflight(tmp_path, tmp_path/'contract.json') == c
    c['solver_options'] = dict(m.previous.lp.OPTIONS)
    fixtures.write(tmp_path/'contract.json', c)
    with pytest.raises(ValueError, match='solver_options'):
        m.preflight(tmp_path, tmp_path/'contract.json')
    c['solver_options'] = m.OPTIONS
    c['saved_inputs']['export.npz'] = '0'*64
    fixtures.write(tmp_path/'contract.json', c)
    with pytest.raises(ValueError, match='SOURCE_OR_INPUT'):
        m.preflight(tmp_path, tmp_path/'contract.json')
    c['saved_inputs']['export.npz'] = m.io.sha(tmp_path/'export.npz')
    parent = tmp_path/c['records']['parent_LP_result']
    fixtures.write(parent, dict(status='UNRESOLVED_LP', candidate='changed'))
    c['saved_inputs'][c['records']['parent_LP_result']] = m.io.sha(parent)
    fixtures.write(tmp_path/'contract.json', c)
    with pytest.raises(ValueError, match='PARENT_LP_STOP'):
        m.preflight(tmp_path, tmp_path/'contract.json')


@pytest.mark.parametrize('fault', [None, 'dual', 'native_warning', 'tensor'])
def test_synthetic_end_to_end_preserves_old_controls_and_partial_evidence(tmp_path, monkeypatch, fault):
    with threadpool_limits(limits=1):
        c = invented(tmp_path)
    protected, original_options = dict(c['saved_inputs']), lp.OPTIONS
    monkeypatch.setattr(m.subprocess, 'check_output', lambda *a, **k: 'SYNTHETIC_ONLY\n')
    def forbidden(*a, **k):
        pytest.fail('forbidden fit/native/trajectory sampling/physical norm')
    monkeypatch.setattr(m.previous.b, 'build_kernel', forbidden)
    monkeypatch.setattr(m.previous.b, 'independent_occurrence', forbidden)
    monkeypatch.setattr(lp.objective, 'fit', forbidden)
    monkeypatch.setattr(m.previous.old.old.molecule.SectorCAR, 'physical', forbidden)
    original, count = lp.solve_once, []
    def solve(p):
        count.append(1)
        r = original(p)
        if fault == 'dual':
            r.eqlin.marginals[0] += 10
        if fault == 'native_warning':
            # A genuine rejected double value adds another warning. Never
            # accept it just because its text equals the known int warning.
            warnings.warn(m.expected_warnings()[-1], category=OptimizeWarning)
        return r
    monkeypatch.setattr(lp, 'solve_once', solve)
    if fault == 'tensor':
        old_replace = lp.replace_tail
        def changed(*a):
            j, u = old_replace(*a)
            u[0, 0, 0] += .001
            return j, u
        monkeypatch.setattr(lp, 'replace_tail', changed)
    out = tmp_path/m.FRESH
    with threadpool_limits(limits=1):
        if fault in ('native_warning', 'tensor'):
            with pytest.raises(ValueError, match='OPTION_HANDLING' if fault == 'native_warning' else 'U_OR_HEAD'):
                m.run(tmp_path, tmp_path/'contract.json', out)
        else:
            result = m.run(tmp_path, tmp_path/'contract.json', out)
    assert count == [1] and lp.OPTIONS is original_options
    stop, raw = m.io.read(out/'STOP.json'), m.io.read(out/'LP_solver.json')
    assert raw['options'] == m.OPTIONS
    assert (out/'LP_raw.npz').exists() and (out/'solver_option_handling.json').exists()
    assert stop['retry'] == stop['resume'] == 0 and stop['mandatory_stop']
    assert stop['failed'] == (fault in ('native_warning', 'tensor'))
    if fault is None:
        assert stop['scientific_units'] == 50 and stop['call_counts'] == m.LIMITS
        assert result['preservation']['status'] == 'NUMERICAL_PRESERVATION_PASS'
        assert result['actual_CA'] is result['shots'] is result['total_resources'] is None
    elif fault == 'dual':
        assert stop['scientific_units'] == 9 and result['status'] == 'UNRESOLVED_LP'
        assert not (out/'candidate.npz').exists()
    else:
        assert not (out/'PF_RTE.json').exists()
    assert all(m.io.sha(tmp_path/p) == h for p, h in protected.items())
    with pytest.raises(ValueError, match='FRESH_LP_NUMERICS'):
        m.run(tmp_path, tmp_path/'contract.json', out)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Invented matrices only; write reproducible technical study')
    p.add_argument('--synthetic-report', type=Path, required=True)
    args = p.parse_args()
    with args.synthetic_report.open('x') as f:
        json.dump(synthetic_study(), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
