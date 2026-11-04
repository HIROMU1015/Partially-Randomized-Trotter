"""Invented arrays/byte fixtures only. No saved molecular input evaluation."""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks import c_near_transfer as m


def system(n=4, rank=8, ld=2):
    return m.System(n, rank, ld, 1.15, 'a'*64)


@pytest.fixture(autouse=True)
def no_saved_array_reads():
    with patch('numpy.load', side_effect=AssertionError('NO_REAL_INPUT_DECODE')):
        yield


@pytest.mark.parametrize('n,dim', [(2, 4), (4, 36), (6, 400)])
def test_geometry_spin_sector_and_memory_metadata(n, dim):
    s = system(n)
    assert s.distance_angstrom == 1.15 and s.sector_dimension == dim
    assert s.dimensions()['sector'] == dim
    assert s.dimensions()['controlled'] == 2**(2*n+1)
    assert s.nalpha == s.nbeta == n//2
    assert s.geometry()[1] == ('H', (0., 0., 1.15))


@pytest.mark.parametrize('change', [dict(ld=8), dict(rank=0), dict(norb=3), dict(sector_dimension=70),
    dict(nalpha=1), dict(input_sha256='wrong'), dict(distance_angstrom=-1), dict(basis='other')])
def test_wrong_rank_mask_sector_identity_rejected(change):
    with pytest.raises(ValueError):
        replace(system(), **change)


@pytest.mark.parametrize('n,rank,ld', [(2, 3, 1), (6, 3, 1)])
def test_new_neutral_scales_and_fair_start_not_old_H4_constants(n, rank, ld):
    from test_research_c_separated_J_U import invented
    from trottertracks import c_near_stable as old
    g, J, U, _ = invented(n=n, rank=rank, ld=ld)
    s = system(n, rank, ld)
    specs = m.objective_specs(g, J, U, s, strength=.1, epsilon=1e-8)
    a, b = specs['PR'], specs['ENTRY']
    assert a.loss_scale == b.loss_scale and a.loss_scale != 2.1839632053132894e-9
    assert float(old.near.pr.smooth_mass(J, a, np))/a.mass_scale == pytest.approx(1.)
    assert float(old.near.mass(J, b, np))/b.mass_scale == pytest.approx(1.)
    assert b.entry_epsilon == pytest.approx(1e-8*(2*n+1)/(1.5*n-.25))
    assert a.ld == b.ld == ld and a.norb == b.norb == n
    with pytest.raises(ValueError, match='TENSOR_SCALE_UNRESOLVED'):
        m.objective_specs(old.stable.linear.reconstruct(J, U), J, U, s, strength=.1, epsilon=1e-8)
    with pytest.raises(ValueError, match='SHAPE'):
        m.validate_arrays(g, J[:-1], U, s)
    bad = U.copy(); bad[0, 0, 0] += .1
    with pytest.raises(ValueError, match='ORTHOGONAL'):
        m.validate_arrays(g, J, bad, s)


def test_six_orbital_entry_gradient_hessian_U_tangent_via_existing_test():
    # Same test function, extended to a previously untested artificial shape.
    from test_research_c_near_stable import test_same_array_value_J_U_gradient_HVP_AD_and_refinement
    test_same_array_value_J_U_gradient_HVP_AD_and_refinement(6, 3, 1)


def test_six_orbital_PR_gradient_hessian_and_same_array_checks():
    from test_research_c_separated_J_U import invented
    from trottertracks import c_near_stable as old
    g, J, U, _ = invented(n=6, rank=3, ld=1)
    spec = m.objective_specs(g, J, U, system(6, 3, 1), strength=.1, epsilon=1e-8)['PR']
    p = old.stable.Coefficients(g, U, spec)
    chart = old.inner.point.CoefficientChart(p, J)
    zero = np.zeros(len(chart.x0)); direction = np.linspace(-.1, .1, len(zero)); h = 1e-6
    value, grad = chart.vg(zero)
    np.testing.assert_allclose(chart.independent(zero)[2], grad, atol=1e-12)
    np.testing.assert_allclose((chart.vg(h*direction)[1]-chart.vg(-h*direction)[1])/(2*h),
                               chart.hess(zero)@direction, atol=1e-7)


def test_fourteen_frame_six_orbital_corrected_codec_preserves_signed_initialization():
    from test_research_c_separated_J_U import invented
    from trottertracks import c1_real_orthogonal_codec as codec
    g, J, U, _ = invented(n=6, rank=14, ld=2)
    restored_J, restored_U, x, report, _ = codec.roundtrip(J, U)
    assert len(x) == 504 and report['norb'] == 6 and report['actual_rank'] == 14
    np.testing.assert_array_equal(J, restored_J)
    assert report['tensor_frobenius'] <= report['tensor_allowance']
    s = system(6, 14, 2)
    specs = m.objective_specs(g, restored_J, restored_U, s, strength=.1, epsilon=1e-8)
    assert specs['PR'].rank == specs['ENTRY'].rank == 14


def test_existing_arm_wrapper_dispatch_no_new_solver_or_global_mutation():
    from test_research_c_separated_J_U import invented
    from trottertracks import c_near_practical as old
    g, J, U, _ = invented(n=6, rank=3, ld=1)
    s = system(6, 3, 1)
    specs = m.objective_specs(g, J, U, s, strength=.1, epsilon=1e-8)
    def trajectory(g2, j, u, spec, physical, **kw):
        assert g2 is g and spec.norb == 6
        assert kw['step'] is old.STEP and kw['policy'] is old.POLICY
        assert kw['maximum_total_points'] == old.MAX_POINTS
        return dict(terminal=None, status='ARTIFICIAL_UNQUALIFIED')
    for name in ['PR', 'ENTRY']:
        summary, point, _ = m.recipe_arm(name, g, dict(J=J, U=U), specs[name], s, None,
                                         lambda k,l,f:f(), lambda *a:None, trajectory=trajectory)
        assert point is None and summary['endpoint_eligible'] is False
        assert summary['total_RZ'] is None and summary['actual_CA_achieved'] is None
    assert old.OBJECTIVE['norb'] == 4 and old.STEP.max_updates == 16


@pytest.mark.parametrize('n', [2, 6])
def test_physical_context_existing_sector_CAR_generalK_scalar_and_budget(n):
    from test_research_c_c1_h2_h6 import artificial
    packet, J, U = artificial(n, rotated=True)
    packet = {k:packet[k] for k in ['h','g','constant']}
    s = system(n, 1, 0)
    ctx = m.PhysicalContext(packet, s, observed_input_sha256='a'*64)
    row, generators = ctx.assess(J, U, candidate_sha256='b'*64, budget_Ha=.00019875)
    assert row['budget_PASS'] and row['sector_error'] < 1e-10
    assert row['representation']['max_entry'] < 1e-10
    assert row['rigorous_enclosure'] is None and row['ground_alias'] is None
    assert row['kappa'] == 'original_g_corrected' and row['scalar'] == 'nuclear_once'
    np.testing.assert_allclose(sum(generators), ctx.physical, atol=1e-10)
    # Existing numerical guard remains positive even for exact toy factorization.
    failed, _ = ctx.assess(J, U, candidate_sha256='b'*64, budget_Ha=row['numerical_guard']/2)
    assert not failed['budget_PASS']
    with pytest.raises(ValueError, match='WRONG_PHYSICAL_INPUT'):
        m.PhysicalContext(packet, s, observed_input_sha256='c'*64)
    with pytest.raises(ValueError, match='SCALAR_SCHEMA'):
        m.PhysicalContext(dict(packet, constant=np.ones(2)), s, observed_input_sha256='a'*64)


def byte_fixture(root):
    (root/'entry.py').write_text('# invented technical source\n')
    (root/'metadata.json').write_text('{}')
    sha = lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    (root/'SOURCE.json').write_text(json.dumps(dict(files={'entry.py':sha(root/'entry.py')})))
    c=dict(status=m.STATUS, executable=False, selected_system=None, science_entry=None,
           maximum_scientific_calls=None, comparison='II_practical_finite_recipe',
           transfer='recipe_not_endpoint_parameters', retry=0, resume=0, mandatory_stop=True,
           technical_SOURCE='SOURCE.json',technical_SOURCE_sha256=sha(root/'SOURCE.json'),
           metadata_inputs={'metadata.json':sha(root/'metadata.json')},future_output='run01')
    (root/'contract.json').write_text(json.dumps(c)); return c


@pytest.mark.parametrize('fault', ['source','metadata','missing','fresh','policy'])
def test_passive_preflight_source_input_drift_missing_fresh_and_policy(tmp_path, fault):
    c = byte_fixture(tmp_path)
    assert m.preflight(tmp_path, tmp_path/'contract.json')['scientific_calls'] == 0
    if fault == 'source':(tmp_path/'entry.py').write_text('changed')
    if fault == 'metadata':(tmp_path/'metadata.json').write_text('changed')
    if fault == 'missing':(tmp_path/'metadata.json').unlink()
    if fault == 'fresh':(tmp_path/'run01').mkdir()
    if fault == 'policy':
        c['executable'] = True;(tmp_path/'contract.json').write_text(json.dumps(c))
    with pytest.raises((ValueError, FileNotFoundError)):m.preflight(tmp_path, tmp_path/'contract.json')


def test_passive_import_and_refusal_even_with_execute_flag():
    code="import sys;sys.path.insert(0,'src');from trottertracks import c_near_transfer;assert not any(n in sys.modules for n in ('numpy','scipy','jax','ffsim','qiskit'))"
    subprocess.run([sys.executable,'-B','-c',code], cwd=ROOT, check=True)
    with pytest.raises(PermissionError, match='DESIGN_NOT_EXECUTABLE'):m.execute(scientific_permission=True)
    r=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/prepare_research_c_near_transfer.py'),
                      '--contract','absent.json','--execute'],capture_output=True,text=True)
    assert r.returncode and 'DESIGN_NOT_EXECUTABLE' in r.stderr


def test_call_model_derivation_and_no_transfer_of_unused_calls():
    assert m.optimization_call_model(accepted_update_cap=1,total_trial_cap=3)['maximum'] == 69
    old = m.optimization_call_model(accepted_update_cap=16,total_trial_cap=128)
    assert old['maximum'] == 1629 and old['per_arm']['inner_J_solve'] == 387
    assert 'all downstream work' in old['excludes']
    with pytest.raises(ValueError):m.optimization_call_model(accepted_update_cap=0,total_trial_cap=3)


@pytest.mark.parametrize('fault', [None, 'input', 'candidate', 'rank', 'budget'])
def test_new_ground_alias_handoff_never_reuses_old_H4_price_or_wrong_candidate(fault):
    s = system(6, 14, 2)
    row = dict(input_sha256='a'*64, candidate_sha256='b'*64, actual_rank=14, budget_PASS=True,
               sector_error=1e-8, numerical_guard=1e-9, budget_Ha=.00019875, rigorous_enclosure=None)
    if fault == 'input':row['input_sha256'] = 'c'*64
    if fault == 'candidate':row['candidate_sha256'] = 'c'*64
    if fault == 'rank':row['actual_rank'] = 8
    if fault == 'budget':row['budget_PASS'] = False
    if fault:
        with pytest.raises(ValueError):m.downstream_handoff(s, 'b'*64, row)
    else:
        h = m.downstream_handoff(s, 'b'*64, row)
        assert h['sector_dimension'] == 400 and h['ld'] == 2
        assert all(h[k] is None for k in ['ground_alias','PF_RTE','shots','W','native_price','total_RZ','actual_CA_achieved'])
