"""Invented arrays only. No repository molecular NPZ/QPY is decoded."""
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from copy import deepcopy
import json
import os
import subprocess
import sys

import numpy as np
import pytest
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks import c_h4_strong_d_relocation as m
from trottertracks import c_h4_strong_d_relocation_execution as run


@pytest.fixture(autouse=True)
def invented_loads_only(monkeypatch, tmp_path):
    original = np.load
    def guarded(path, *args, **kwargs):
        name = Path(path if isinstance(path, (str, os.PathLike)) else path.name).resolve()
        if not name.is_relative_to(tmp_path): pytest.fail('molecular array decoding forbidden: '+str(name))
        return original(path, *args, **kwargs)
    monkeypatch.setattr(np, 'load', guarded)
    with threadpool_limits(limits=1): yield


@pytest.fixture(scope='module')
def car():
    from trottertracks.c1_stage1 import car_bilinears
    with threadpool_limits(limits=1): return car_bilinears(8)


def fixture(kind):
    rng = np.random.default_rng(20261107)
    q = np.linalg.qr(rng.normal(size=(4, 4)))[0]
    kappa = q @ np.diag([-.3, -.1, .05, .2]) @ q.T
    # Exact symmetry is part of the saved number-J schema.
    kappa = (kappa+kappa.T)/2
    if kind == 'CDF':
        a = rng.normal(size=(8, 4, 4))*1e-4
        J = a+a.transpose(0, 2, 1)
        U = np.array([np.linalg.qr(rng.normal(size=(4, 4)))[0] for _ in range(8)])
        return m.prepare_cdf(kappa, .17, J, U, source_identity='invented-CDF', input_identity='invented-input')
    lam = np.array([.006, -.004, .003, -.007, .002, .005, -.003, .004])/20
    a = rng.normal(size=(8, 4, 4))*.07
    g = a+a.transpose(0, 2, 1)
    matrices = np.array([np.kron(x, np.eye(2)) for x in g], complex)
    return m.prepare_xdf(np.kron(kappa, np.eye(2)), lam, matrices, .17, kappa=kappa,
        source_identity='invented-XDF', input_identity='invented-input')


def shells():
    from trottertracks import c_pr_aware_df_stage_b2 as b2
    from trottertracks.c1_h4_native_fair import compile_checked
    from trottertracks.pr_dictionary_validation.fixture import CompilerProfile
    result = {}
    for label in ('head', 'X', 'Y'):
        q = b2.shell_build(8, label)
        result[label] = dict(raw=q, compiled=compile_checked(q, CompilerProfile()))
    return result


def saved(p, car):
    from trottertracks.c1_stage1 import sector_indices
    old, _ = m.generators(p, car); ix = sector_indices(4, 2, 2)
    H = sum(old)[np.ix_(ix, ix)]; values, vectors = eigh(H)
    return dict(candidate_identity=p.before.source_identity, input_identity=p.model.input_identity,
        DF_budget_PASS=True, DF_error_with_guard=1e-12, H=H,
        generators=np.array([a[np.ix_(ix, ix)] for a in old]), energy=values[0],
        sector_state=vectors[:, 0], ground_origin='invented-ground')


def test_import_has_no_numeric_or_circuit_side_effects():
    code = "import sys;sys.path.insert(0,'src');from trottertracks import c_h4_strong_d_relocation,c_h4_strong_d_relocation_execution;assert not any(n in sys.modules for n in ('numpy','scipy','qiskit','ffsim'))"
    subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT, check=True)


@pytest.mark.parametrize('kind', ['CDF', 'XDF'])
def test_full_fock_H_scalar_identity_Z_ZZ_and_order(kind, car):
    p = fixture(kind); old, new = m.generators(p, car)
    np.testing.assert_allclose(sum(old), sum(new), atol=2e-12, rtol=0)
    assert p.model.constant == pytest.approx(.17+p.scalar_shift, abs=1e-15)
    assert [b.block_id for b in p.model.blocks] == ['one_body']+[f'F{i}' for i in range(8)]
    for t, (a, b) in enumerate(zip(p.before.blocks[1:], p.model.blocks[1:])):
        assert a.rotation == b.rotation and a.diagonal.rzz == b.diagonal.rzz
        assert a.diagonal.global_phase == b.diagonal.global_phase
        if t < 2: assert a.diagonal == b.diagonal
        else: assert b.diagonal.rz == ()
    assert not p.one.flags.writeable and not p.delta_one.flags.writeable


def test_XDF_square_core_factor_two_signed_and_spin_lift(car):
    from trottertracks import c1_stage1 as stage
    from trottertracks.c1_h4_strong_baseline import spin_adapted_eigh
    p = fixture('XDF'); old, _ = m.generators(p, car)
    assert any(p.lambdas < 0) and any(p.lambdas > 0)
    for t, (lam, g) in enumerate(zip(p.lambdas, p.matrices)):
        u, eta, report = spin_adapted_eigh(g)
        assert report['spin_order'] == 'mode 2p=alpha,2p+1=beta'
        J = 2*lam*np.outer(eta[::2], eta[::2])
        raw, _ = stage.raw_frame(J, u[::2, ::2], car)
        np.testing.assert_allclose(raw, old[t+1], atol=2e-12, rtol=0)
        assert p.before.blocks[t+1].rotation == tuple(tuple(complex(x) for x in row) for row in u)


@pytest.mark.parametrize('fault', ['spin', 'kappa', 'sign_schema'])
def test_XDF_rejects_invalid_saved_conventions(fault):
    p = fixture('XDF'); one = p.one.copy(); gs = p.matrices.copy(); lam = p.lambdas.copy()
    if fault == 'spin': gs[3, 0, 1] = .01; gs[3, 1, 0] = .01
    if fault == 'kappa': one[0, 0] += .01
    if fault == 'sign_schema': lam = lam.astype(complex)
    with pytest.raises(ValueError):
        m.prepare_xdf(one, lam, gs, .17, kappa=p.one[::2, ::2], source_identity='bad', input_identity='I')


@pytest.mark.parametrize('kind', ['CDF', 'XDF'])
def test_corrupted_scalar_or_frame_is_detected(kind, car):
    p = fixture(kind)
    bad = replace(p, model=replace(p.model, blocks=p.model.blocks[:1]+tuple(reversed(p.model.blocks[1:]))))
    with pytest.raises(ValueError): m.generators(bad, car)
    if kind == 'XDF':
        bad = replace(p, model=replace(p.model, constant=p.model.constant+.01))
        with pytest.raises(ValueError): m.generators(bad, car)


def test_original_XDF_strong_builder_full512(car):
    from trottertracks import c1_stage3_reference as reference, c1_h4_native_fair as fair
    from trottertracks.c1_h4_strong_baseline import StrongBuilder
    p = fixture('XDF'); old, _ = m.generators(p, car)
    raw = StrongBuilder(relative_fusion=True, basis_policy=p.basis_policy).ordered(p.before, Fraction(1, 10), optimized=True)
    actual = fair.decode(raw)
    target = fair.controlled(reference.reference_kernels(old, Fraction(1, 10), ld=8)['pf'])
    np.testing.assert_allclose(actual, target, atol=1e-10, rtol=0)
    assert actual.shape == (512, 512)


@pytest.mark.parametrize('kind', ['CDF', 'XDF'])
def test_two_lane_full512_E2E_13_rounds_inverse_XY_accounting(kind, car, tmp_path):
    from trottertracks.c1_stage3_integrated import Calls
    p = fixture(kind); s = saved(p, car)
    journal = tmp_path/'journal'; journal.mkdir()
    call = Calls(journal, run.LIMITS, sum(run.LIMITS.values()))
    result = m.evaluate(p, s, shells(), tmp_path/'out', call)
    assert result['status'] == 'RELOCATED_STRONG_D_COMPLETE'
    assert call.counts['PF_round'] == call.counts['round_synthesis'] == 13
    assert call.counts['operator_decode'] == 4 and call.counts['adjoint_gate'] == 2
    assert call.counts['kernel_build'] == call.counts['compile'] == 1
    for lane in (result['ideal_exact_native'], result['numerical']):
        assert [r['m'] for r in lane['rounds']] == list(range(13))
        assert sum(2*r['alpha_axis'] for r in lane['rounds']) <= .01
        assert lane['cost']['totals']['rz_count'] > 0
        W = sum(2*r['shots']*(1 << r['m']) for r in lane['rounds'])
        expected = W*result['metrics']['rz_count']+sum(7*r['shots'] for r in lane['rounds'])
        assert lane['cost']['totals']['rz_count'] == expected
    assert result['numerical']['rigorous_enclosure'] is None
    assert result['actual_CA_achieved'] is result['PR_whole_law_qualification'] is None
    report = json.loads((tmp_path/'out/segment.json').read_text())
    assert report['input_dimension'] == report['output_dimension'] == 512
    assert report['global_phase_fit'] is False
    assert json.loads((tmp_path/'out/XY.json').read_text())['status'] == 'PASS'
    # A control-relative phase fault must be rejected without phase fitting.
    from trottertracks.c1_stage3_molecular import component_check
    with np.load(tmp_path/'out/operators/compiled.npz') as f: compiled = f['matrix']
    bad = compiled.copy(); bad[256:] *= np.exp(.02j)
    with pytest.raises(ValueError): component_check(bad, compiled, 1e-10, label='injected control phase')


@pytest.mark.parametrize('fault', ['candidate', 'input', 'ground', 'H', 'DF'])
def test_reference_and_accuracy_binding_rejects_other_candidate(fault, car, tmp_path):
    p = fixture('CDF'); s = saved(p, car)
    if fault == 'candidate': s['candidate_identity'] = 'other'
    if fault == 'input': s['input_identity'] = 'other'
    if fault == 'ground': s['sector_state'] = np.zeros(36)
    if fault == 'H': s['H'] = s['H']+.01*np.eye(36)
    if fault == 'DF': s['DF_error_with_guard'] = .01
    with pytest.raises(ValueError): m.evaluate(p, s, {}, tmp_path/'out', lambda k,l,f:f())


def test_point_comparison_tie_ineligible_and_no_new_statistics():
    def result(point): return dict(status='RELOCATED_STRONG_D_COMPLETE', ideal_exact_native=dict(cost=dict(totals=dict(rz_count=point))))
    r = dict(CDF=result(11.), XDF=result(12.))
    assert run.compare_points(r, 10.)['decision'] == 'E2_POINT_BENEFIT_REMAINS'
    r['XDF'] = result(10.)
    out = run.compare_points(r, 10.)
    assert not out['E2_lower_than']['XDF'] and out['new_CI'] is None and out['new_PR_samples'] == 0
    r['CDF']['status'] = 'STOP_PF_RPE_UNQUALIFIED'
    assert run.compare_points(r, 10.)['comparisons'] is None


def test_worker_exception_saves_STOP_and_no_result(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'load_candidate', lambda *a: (_ for _ in ()).throw(ValueError('invented fault')))
    with pytest.raises(ValueError): run.candidate_worker(tmp_path, {}, 'CDF', tmp_path/'worker')
    stop = json.loads((tmp_path/'worker/STOP.json').read_text())
    assert stop['failed'] and stop['mandatory_stop'] and stop['retry'] == stop['resume'] == 0
    assert not (tmp_path/'worker/evaluation/result.json').exists()


def test_actual_contract_passive_without_any_molecular_decode():
    c = run.preflight(ROOT, 'docs/research/research_c_h4_strong_d_relocation_contract_unapproved_20261107.json')
    assert c['policy'] == run.POLICY and c['target'] == run.TARGET
    assert c['science_executed'] is False and c['scientific_execution_permission'] is False


@pytest.mark.parametrize('field', ['target', 'policy', 'correctness', 'source'])
def test_passive_rejects_scope_or_source_change_without_decode(field, monkeypatch):
    contract = 'docs/research/research_c_h4_strong_d_relocation_contract_unapproved_20261107.json'
    c = deepcopy(run.read(ROOT/contract)); original = run.read
    if field == 'target': c['target']['energy_ha'] *= 2
    if field == 'policy': c['policy']['relocation_ld'] = 8
    if field == 'correctness': c['correctness']['input_columns'] = 36
    if field == 'source': c['SOURCE_sha256'] = '0'*64
    monkeypatch.setattr(run, 'read', lambda p: c if Path(p) == ROOT/contract else original(p))
    with pytest.raises(ValueError): run.preflight(ROOT, contract)


def synthetic_files(root, car):
    """Whitelisted loader/worker fixture; every NPZ/QPY is freshly invented."""
    from qiskit import qpy
    from trottertracks import c1_stage1_execution as io
    cdf, xdf = fixture('CDF'), fixture('XDF')
    arrays = dict(kappa=cdf.one[::2, ::2].real, constant=np.asarray(.17),
        diag_coulomb_mats=cdf.cdf.arm.J, orbital_rotations=cdf.cdf.arm.U)
    np.savez(root/'cdf.npz', **arrays)
    metadata = dict(actual_rank=8, z_representation=False, frame_ids=[f'F{i}' for i in range(8)],
        one_body_role='ffsim_corrected_kappa', arrays={k:io.array_record(a) for k,a in arrays.items()})
    np.savez(root/'original.npz', one_mo=arrays['kappa'], eri_chemist=np.zeros((4,)*4),
        nuclear_repulsion=np.asarray(.17), mo_coeff=np.eye(4), df_one_body=xdf.one.astype(complex),
        df_lambdas=np.r_[xdf.lambdas, 0., 0.],
        df_g_matrices=np.concatenate([xdf.matrices, np.zeros((2,8,8), complex)]))
    metadata['input_sha256'] = run.sha(root/'original.npz')
    (root/'metadata.json').write_text(json.dumps(metadata))
    c = dict(cdf_export='cdf.npz', cdf_metadata='metadata.json', original_input='original.npz',
        saved_inputs={'original.npz':metadata['input_sha256']}, candidates={}, shells={}, SOURCE='source.json', E2_point_RZ=10.)
    (root/'source.json').write_text('{}')
    for arm, p in [('CDF', cdf), ('XDF', xdf)]:
        s = saved(p, car)
        ref = arm+'_reference.npz'; ground = arm+'_ground.npz'
        if arm == 'CDF': np.savez(root/ref, matrix=s['H'])
        else:
            from trottertracks.c1_stage1 import sector_indices
            np.savez(root/ref, generators=s['generators'], sector_indices=sector_indices(4,2,2))
        np.savez(root/ground, energy=s['energy'], sector_state=s['sector_state'])
        c['candidates'][arm] = dict(identity=p.before.source_identity, reference=ref, ground=ground,
            DF_error_with_guard=1e-12)
    for label, qs in shells().items():
        c['shells'][label] = {}
        for kind, qc in qs.items():
            path=label+'_'+kind+'.qpy'
            with (root/path).open('wb') as stream: qpy.dump(qc, stream)
            c['shells'][label][kind] = path
    return c


@pytest.mark.parametrize('arm', ['CDF', 'XDF'])
def test_invented_saved_loader_to_worker_E2E_fresh_STOP(arm, car, tmp_path):
    c = synthetic_files(tmp_path, car)
    result = run.candidate_worker(tmp_path, c, arm, tmp_path/'worker')
    assert result['status'] == 'RELOCATED_STRONG_D_COMPLETE'
    stop = run.read(tmp_path/'worker/STOP.json')
    assert not stop['failed'] and stop['mandatory_stop'] and stop['retry'] == stop['resume'] == 0
    assert stop['call_counts']['candidate_load'] == stop['call_counts']['candidate_prepare'] == 1
    assert sum(stop['call_counts'].values()) <= sum(run.LIMITS.values())
    with pytest.raises(FileExistsError): run.candidate_worker(tmp_path, c, arm, tmp_path/'worker')


def test_root_exception_preserves_STOP_and_has_no_ranking(tmp_path, monkeypatch):
    (tmp_path/'source.json').write_text('{}')
    c = dict(SOURCE='source.json', E2_point_RZ=10.)
    monkeypatch.setattr(run, 'preflight', lambda *a:c)
    with pytest.raises(KeyError): run.execute(tmp_path, 'unused.json')
    stop = run.read(tmp_path/run.FRESH/'STOP.json')
    assert stop['failed'] and stop['mandatory_stop'] and stop['retry'] == stop['resume'] == 0
    assert not (tmp_path/run.FRESH/'comparison.json').exists()


def test_two_process_invented_end_to_end_and_no_root_retry(car, tmp_path, monkeypatch):
    c = synthetic_files(tmp_path, car)
    def preflight(*args):
        if (tmp_path/run.FRESH).exists(): raise FileExistsError('fresh output required')
        return c
    monkeypatch.setattr(run, 'preflight', preflight)
    result = run.execute(tmp_path, 'invented-contract.json')
    assert result['status'] == 'FIXED_TWO_CANDIDATE_IDEAL_POINT_COMPARISON'
    assert result['decision'] == 'E2_POINT_BENEFIT_REMAINS'
    stop = run.read(tmp_path/run.FRESH/'STOP.json')
    assert not stop['failed'] and set(stop['completed_candidates']) == {'CDF', 'XDF'}
    assert stop['retry'] == stop['resume'] == 0 and stop['mandatory_stop']
    for arm in ('CDF', 'XDF'):
        assert not run.read(tmp_path/run.FRESH/arm/'STOP.json')['failed']
    with pytest.raises(FileExistsError): run.execute(tmp_path, 'invented-contract.json')


def test_PF_unqualified_stops_before_native_and_marks_worker_STOP(car, tmp_path, monkeypatch):
    from trottertracks import c1_molecule_extension as molecule
    c = synthetic_files(tmp_path, car)
    monkeypatch.setattr(molecule, 'PF_rounds', lambda *a,**k:[dict(m=m,pf_operator_error=.3) for m in range(13)])
    result = run.candidate_worker(tmp_path, c, 'CDF', tmp_path/'worker')
    assert result['status'] == 'STOP_PF_RPE_UNQUALIFIED' and result['metrics'] is None
    stop = run.read(tmp_path/'worker/STOP.json')
    assert stop['failed'] and stop['call_counts'].get('compile', 0) == 0
    assert run.compare_points({'CDF':result}, 10.)['comparisons'] is None


def test_numerical_lane_unqualified_keeps_ideal_separate_and_STOPS(car, tmp_path, monkeypatch):
    from trottertracks import c_pr_aware_df_m2 as m2
    c = synthetic_files(tmp_path, car)
    monkeypatch.setattr(m2, 'evaluate_accuracy', lambda *a,**k:dict(cost=dict(totals=None)))
    result = run.candidate_worker(tmp_path, c, 'CDF', tmp_path/'worker')
    assert result['status'] == 'STOP_NUMERICAL_LANE_UNQUALIFIED'
    assert result['ideal_exact_native']['cost']['totals']['rz_count'] > 0
    assert run.read(tmp_path/'worker/STOP.json')['failed']
    assert run.compare_points({'CDF':result}, 10.)['comparisons'] is None


def test_native_compile_phase_error_saves_STOP_before_metrics(car, tmp_path, monkeypatch):
    from trottertracks import c1_h4_native_fair as fair
    c = synthetic_files(tmp_path, car); original = fair.compile_checked
    def corrupted(*a,**k):
        qc = original(*a,**k); qc.rz(.02, qc.num_qubits-1); return qc
    monkeypatch.setattr(fair, 'compile_checked', corrupted)
    with pytest.raises(ValueError): run.candidate_worker(tmp_path, c, 'XDF', tmp_path/'worker')
    stop = run.read(tmp_path/'worker/STOP.json')
    assert stop['failed'] and stop['call_counts'].get('metrics', 0) == 0
    assert not (tmp_path/'worker/evaluation/result.json').exists()
