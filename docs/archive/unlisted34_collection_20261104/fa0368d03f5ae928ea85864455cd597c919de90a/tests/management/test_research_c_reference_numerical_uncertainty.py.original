"""Artificial matrices and metadata only; never load saved H4 arrays."""
from fractions import Fraction as F
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from trottertracks import c_reference_numerical_uncertainty as b2

ROOT = Path(__file__).resolve().parents[2]


def test_product_error_contains_known_noncommuting_complex_perturbation():
    a = np.array([[1, 1j], [.25, -.5]], complex)
    b = np.array([[.2j, .7], [1, -.3j]], complex)
    da = np.array([[.01j, 0], [.02, -.03]], complex)
    db = np.array([[0, -.04j], [.01, 0]], complex)
    # Conservative exact rational input guards, not rounded SVD evidence.
    bound = b2.product_error('1/20', '1/20', '2', '2', '0')
    assert np.linalg.norm((a+da)@(b+db)-a@b, 2) < float(bound)
    assert b2.product_error('0', '0', '2', '2', '0') == 0
    assert b2.product_error('0', '0', '2', '2', None) is None


def test_coherent_perturbation_and_squaring_rounding_are_separate():
    epsilon = F(1, 1000)
    rows = b2.round_transport(str(epsilon), '101/100', ['0']*12,
                             state_norm_upper='1', state_error='0', expectation_errors=['0']*13)
    assert [x['m'] for x in rows] == list(range(13))
    for row in rows[:6]:
        exact_difference = (1+epsilon)**row['coherent_occurrences'] - 1
        assert F(row['raw_operator_error']) >= exact_difference
        assert row['power_evaluation_rounding'] == '0'
        assert row['numerical_squarings'] == row['m']
    assert F(rows[-1]['raw_operator_error']) > 2  # no inappropriate unitary cap
    r = b2.round_transport('0', '1', ['1/1000000']*12,
                           state_norm_upper='1', state_error='0', expectation_errors=['0']*13)
    assert r[0]['raw_operator_error'] == '0' and F(r[1]['raw_operator_error']) >= F(1, 1000000)
    assert all(x['kernel_error_amplification'] == '0' for x in r)


def test_state_error_and_complex_XY_conversion_not_operator_identity():
    # Exact |psi_hat|=2 and |psi-psi_hat|=1/10; complex kernel 3i.
    r = b2.round_transport('0', '3', ['0']*12,
                           state_norm_upper='2', state_error='1/10', expectation_errors=['0']*13)[0]
    known_signal_difference = abs((2.1**2-2**2)*3j)
    assert F(r['raw_complex_signal_error']) == b2._up(F(123, 100))
    assert abs(float(F(r['raw_complex_signal_error']))-known_signal_difference) < 1e-14
    assert r['raw_operator_error'] == '0'


def test_normalized_exact_eigenvector_still_needs_ground_oracle_relation():
    # An excited eigenvector has norm 1 and eigen-residual 0, yet is not ground.
    h = np.diag([0,1]); saved = np.array([0,1]); ground = np.array([1,0])
    assert np.vdot(saved,saved) == 1 and np.linalg.norm(h@saved-saved) == 0
    assert abs(np.vdot(saved,1j*h@saved)-np.vdot(ground,1j*h@ground)) == 1
    row = b2.round_transport('0','1',['0']*12,state_norm_upper='1',state_error=None,
                             expectation_errors=['0']*13)[0]
    assert row['raw_complex_signal_error'] is None


@pytest.mark.parametrize('missing', ['kernel', 'norm', 'state_norm', 'state', 'square', 'expectation'])
def test_unknown_local_error_or_oracle_relation_is_never_zero(missing):
    k, a, s, d = '0', '1', '1', '0'
    sq, ex = ['0']*12, ['0']*13
    if missing == 'kernel': k = None
    if missing == 'norm': a = None
    if missing == 'state_norm': s = None
    if missing == 'state': d = None
    if missing == 'square': sq[1] = None
    if missing == 'expectation': ex[4] = None
    r = b2.round_transport(k, a, sq, state_norm_upper=s, state_error=d, expectation_errors=ex)
    m = 2 if missing == 'square' else 4 if missing == 'expectation' else 0
    assert r[m]['raw_complex_signal_error'] is None


@pytest.mark.parametrize('bad', [-1, '-1/10', float('nan'), .1, True])
def test_inexact_or_invalid_bound_input_is_refused(bad):
    with pytest.raises(ValueError): b2.product_error(bad, '0', '1', '1', '0')


def test_subnormal_positive_error_not_silently_flushed_to_zero():
    tiny = F(1, 1 << 1100)
    assert b2.product_error(tiny, '0', '1', '1', '0') >= tiny > 0
    assert float(tiny) == 0
    with pytest.raises(ValueError, match='THIRTEEN'):
        b2.round_transport('0', '1', ['0']*11, state_norm_upper='1', state_error='0', expectation_errors=['0']*13)


def test_existing_exponential_and_high_low_precision_diagnostic():
    from flint import ctx, acb_mat
    from trottertracks.c_pr_aware_df_fixed_car import Arithmetic, Matrix
    from trottertracks import c_pr_aware_df_zsxx_enclosure as z
    from trottertracks.c1_stage3_reference import _ordered_exponentials
    from trottertracks.pr_dictionary_validation.reference import precision_context, exact
    from scipy.linalg import expm
    g = np.diag([.25, -.5]).astype(complex)
    low = expm(-.3j*g)
    values = []
    for digits in (90, 128):
        mp = precision_context(digits)
        value = mp.expm(-mp.j*exact(mp, '3/10')*mp.matrix([[exact(mp, '1/4'), 0], [0, exact(mp, '-1/2')]]))
        values.append([str(value[i, i]) for i in range(2)])
        assert max(abs(complex(value[i, i])-low[i, i]) for i in range(2)) < 1e-15
    assert values[0] != values[1]  # precision comparison is diagnostic, not a bound
    with z._BALL_LOCK, ctx.workprec(192):
        generator = Matrix(z._matrix(g))
        out = _ordered_exponentials([generator], ((0, F(3,10)),), {}, arithmetic=Arithmetic())
        # Independently form exact diagonal exp using the same validated scalar backend.
        from flint import acb
        target = acb_mat([[(-acb(0,1)*z._real(F(3,40))).exp(),0],
                          [0,(acb(0,1)*z._real(F(3,20))).exp()]])
        assert F(z._norm(out.ball-target)['upper_exact']) < F(1, 10**50)
        assert F(z._norm(out.ball-z._matrix(low))['upper_exact']) < F(1, 10**14)


def test_exact_known_matrix_and_Arb_power_signal_preserve_complex_phase():
    from flint import acb_mat, ctx
    from trottertracks.c_pr_aware_df_fixed_car import Matrix
    from trottertracks import c_pr_aware_df_zsxx_enclosure as z
    with z._BALL_LOCK, ctx.workprec(192):
        k = Matrix(acb_mat([[1j, 0], [0, 1]]))
        rows = b2.enclosed_rounds(k, acb_mat([[1],[0]]))
        for row in rows:
            expected = (1, 1j, -1, -1j)[(1 << row['m']) % 4]
            for value, pair in zip((expected.real, expected.imag), row['signal_rectangle'], strict=True):
                assert F(pair[0]) <= F(value) <= F(pair[1])
            assert row['own_ground_oracle_error'] is None
        zero = b2.enclosed_rounds(Matrix(acb_mat([[0]])), acb_mat([[1]]))
        assert all(x['signal_rectangle'] == [['0','0'],['0','0']] for x in zero)
        with pytest.raises(ValueError, match='SHAPE'): b2.enclosed_rounds(k, acb_mat([[1]]))


def test_weighted_event_mean_reuses_K0_signed_events_with_exact_probabilities():
    from flint import acb_mat, ctx
    from trottertracks.c_pr_aware_df_fixed_car import Matrix
    from trottertracks import c_pr_aware_df_zsxx_enclosure as z
    from trotterlib.rte import event_unitary
    ops = dict(z=np.diag([1,-1]).astype(complex), x=np.array([[0,1],[1,0]],complex))
    events = [SimpleNamespace(rotation_component_id=k, rotation_angle=t, product_component_ids=(), phase=1, taylor_order=0)
              for k,t in [('z', .25), ('x', -.125)]]
    numeric = event_unitary(events[0], ops)/3 + 2*event_unitary(events[1], ops)/3
    with z._BALL_LOCK, ctx.workprec(192):
        ball_ops = {k:Matrix(z._matrix(v)) for k,v in ops.items()}
        mean = b2.weighted_events(events, ball_ops, ['1/3', '2/3'])
        assert F(z._norm(mean.ball-z._matrix(numeric))['upper_exact']) < F(1,10**14)
        assert len(b2.enclosed_rounds(mean, acb_mat([[1],[0]]))) == 13
        with pytest.raises(ValueError, match='NORMALIZED'): b2.weighted_events(events, ball_ops, ['1/3','1/3'])
        events[0].taylor_order = 2
        with pytest.raises(ValueError, match='K0_EVENT'): b2.weighted_events(events, ball_ops, ['1/3','2/3'])


def record(quantity='reference_uncertainty', qualification='unknown', values=None, **kw):
    return b2.evidence_record(quantity, [None]*13 if values is None else values,
                             identity=b2.integration.identity(), qualification=qualification,
                             provenance='invented test only;not H4', **kw)


@pytest.mark.parametrize('level', b2.LEVELS[:3])
def test_observations_sensitivity_estimate_do_not_acquire_bound(level):
    r = record(qualification=level, values=['1/100']*13)
    assert all(x['value'] == '1/100' and x['bound'] is None for x in r['rows'])
    assert not r['proofs_verified_by_schema']
    with pytest.raises(ValueError, match='NOT_A_BOUND'):
        record(qualification='validated_numerical_bound', values=['1/100']*13)


@pytest.mark.parametrize('field', ['candidate', 'stage_A_scientific_SOURCE', 'model'])
def test_wrong_candidate_SOURCE_or_randomness_model_refused(field):
    identity = b2.integration.identity(); identity[field] = 'wrong'
    with pytest.raises(ValueError, match='SCIENTIFIC_SOURCE_MODEL'):
        b2.evidence_record('reference_bridge', ['0']*13, identity=identity,
                          qualification='observed_numerical_residual', provenance='artificial')


def test_integration_consumes_disjoint_B2_interface_and_does_not_run_Stage_C():
    from test_research_c_b1_b2_scope_integration import fixture
    p,r,pubs,cov = fixture()
    r.update({q:record(q) for q in b2.TERMS})
    assert set(r['reference_bridge']['included_error_terms']).isdisjoint(r['reference_uncertainty']['included_error_terms'])
    assert all(x['reference_uncertainty_included'] is False for x in r['reference_bridge']['rows'])
    with patch.object(b2.integration.iid, 'qualify', side_effect=AssertionError('UNKNOWN_NOT_A_BOUND')):
        attached,q = b2.integration.integrate(p,r,pubs,cov)
    with patch.object(b2.integration.iid, 'assess', side_effect=AssertionError('NO_STAGE_C')):
        assert b2.integration.assess_if_ready(None,attached,q)['rounds'] == []
    # Even qualified invented B2 evidence cannot fill missing B1 synthesis.
    conditions = dict.fromkeys(b2.OBLIGATIONS, True); conditions['basis'] = 'artificial exact model'
    r.update({name:record(name, 'validated_numerical_bound', ['1/1000000000']*13, conditions=conditions) for name in b2.TERMS})
    r['synthesis'] = None
    _,q = b2.integration.integrate(p,r,pubs,cov)
    assert not q['eligible'] and q['actual_CA_achieved'] is None
    r['reference_uncertainty']['included_error_terms'] = r['reference_bridge']['included_error_terms']
    with pytest.raises(ValueError, match='DOUBLE_COUNT'): b2.integration.integrate(p,r,pubs,cov)


def test_bounds_need_complete_model_and_rigorous_needs_rounding_enclosure():
    c = dict.fromkeys(b2.OBLIGATIONS, True); c['basis'] = 'invented exact model'
    with pytest.raises(ValueError, match='ROUNDING_ENCLOSURE'):
        record(qualification='rigorous_enclosure', values=['0']*13, conditions=c)
    for args in (dict(unresolved=['state']), dict(values=[None]*13)):
        a = dict(qualification='validated_numerical_bound', values=['0']*13, conditions=c); a.update(args)
        with pytest.raises(ValueError, match='NOT_A_BOUND'): record(**a)
    r = record(qualification='rigorous_enclosure', values=['0']*13, conditions=c, rounding_enclosed=True)
    assert all(x['kind'] == 'rigorous_bound' and x['rounding_enclosed'] for x in r['rows'])


def artificial_preflight_root(tmp_path, monkeypatch):
    monkeypatch.setattr(b2.integration, 'preflight', lambda *a:dict(source_files=166,saved_inputs=33))
    source = tmp_path/'SOURCE.json'; source.write_text(json.dumps(dict(files={})))
    c = dict(b2.fixed(), source_manifest='SOURCE.json', source_manifest_sha256=b2.io.sha(source), packages={})
    path = tmp_path/'contract.json'; path.write_text(json.dumps(c))
    return path,c


@pytest.mark.parametrize('field,value', [('status','READY'), ('executable',True), ('retry',1), ('resume',1),
                                        ('mandatory_stop',False), ('maximum_logical_units',38), ('Stage_C','READY')])
def test_contract_scope_retry_resume_STOP_and_old_call_budget_refused(tmp_path,monkeypatch,field,value):
    path,c = artificial_preflight_root(tmp_path,monkeypatch); c[field] = value
    path.write_text(json.dumps(c))
    with pytest.raises(ValueError, match='FIXED_B2_NOT_READY'): b2.preflight(tmp_path,path)


def test_fresh_output_existing_partial_directory_refused_without_modifying_it(tmp_path,monkeypatch):
    path,c = artificial_preflight_root(tmp_path,monkeypatch)
    assert b2.preflight(tmp_path,path)['scientific_calls'] == 0
    out = tmp_path/c['fresh_output']; out.mkdir(parents=True)
    marker = out/'STOP.json'; marker.write_text('artificial partial STOP')
    with pytest.raises(ValueError, match='FRESH_NO_RETRY_RESUME'): b2.preflight(tmp_path,path)
    assert marker.read_text() == 'artificial partial STOP'
    with pytest.raises(PermissionError): b2.run_science()
    with pytest.raises(NotImplementedError, match='NOT_READY'): b2.run_science(execute_science=True)


def test_passive_real_preflight_import_blocks_and_no_science_cli():
    code = '''
import sys
from pathlib import Path
class Block:
 def find_spec(self,fullname,*a):
  if fullname.split('.')[0] in ('numpy','scipy','qiskit','pyscf','openfermion','ffsim','jax','flint','mpmath'):
   raise AssertionError(fullname)
sys.meta_path.insert(0,Block())
from trottertracks import c_reference_numerical_uncertainty as m
r=m.preflight(Path.cwd(),Path(m.CONTRACT))
assert r['scientific_calls']==r['Stage_C_calls']==0 and not r['execution_ready']
assert r['reference_numerical_uncertainty'] is None and r['mandatory_stop']
'''
    subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,check=True)
    result = subprocess.run([sys.executable,'-B',b2.ENTRY,'--execute'],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode == 2 and 'unrecognized arguments' in result.stderr
