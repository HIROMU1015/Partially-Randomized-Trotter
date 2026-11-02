"""Synthetic arrays only. Never open saved H4 science arrays/QPY."""
from dataclasses import replace
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest
from scipy.linalg import expm
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks import c1_h4_strong_baseline as m
import test_research_c_c1_h4_native_fair as fixtures


def unitary(n, seed):
    rng = np.random.default_rng(seed)
    return np.linalg.qr(rng.normal(size=(n,n))+1j*rng.normal(size=(n,n)))[0]


def model():
    from trotterlib.df_trotter.ops import DiagonalEvolutionPrimitives as D
    blocks = tuple(m.native.NativeBlock('F'+str(i), m.native._rotation(unitary(4, i)),
        D(.13*(i+1), ((0, .3+i*.02), (3, -.17)), ((0,2,.23),))) for i in range(3))
    return m.native.NativeModel('ARTIFICIAL', blocks, .37, 'S', 'I')


def test_passive_import_and_default_route_unchanged():
    subprocess.run([sys.executable, '-c', "import sys;sys.path.insert(0,'src');from trottertracks import c1_h4_strong_baseline;assert not any(k in sys.modules for k in ['numpy','qiskit','ffsim','scipy'])"], cwd=ROOT, check=True)
    a = model()
    old = m.precision.PrecisionBuilder().ordered(a, '1/10', optimized=True)
    new = m.StrongBuilder().ordered(a, '1/10', optimized=True)
    assert old == new
    one = np.kron(np.diag([.3,-.2]), np.eye(2))
    kw = dict(source_identity='S', input_identity='I')
    assert m.prepare_xdf_basis(one, [-.2], [one], .37, **kw)[0] == m.native.prepare_xdf(one, [-.2], [one], .37, **kw)


@pytest.mark.parametrize('complex_basis', [False, True])
def test_spin_basis_preserves_original_G_signed_coefficients_order(complex_basis):
    u = unitary(3, 6)
    if not complex_basis: u = np.linalg.qr(u.real)[0]
    g = (u*np.array([-.7,.2,.2])) @ u.conj().T
    spin = np.kron(g, np.eye(2))
    v, eta, report = m.spin_adapted_eigh(spin)
    assert report['exact_spin_structure'] and np.allclose(v[::2,1::2], 0)
    assert np.allclose((v*eta)@v.conj().T, spin, rtol=0, atol=1e-14)
    x, reports = m.prepare_xdf_basis(spin, [-.3,.8], [spin,2*spin], -.7,
        source_identity='S', input_identity='I', basis_policy=m.SPIN)
    assert x.constant == -.7 and [b.block_id for b in x.blocks] == ['one_body','F0','F1']
    assert [r['signed_lambda'] for r in reports[1:]] == [-.3,.8]


def test_eigenvector_phase_permutation_and_exact_degeneracy_only():
    g = np.diag([1.,1.,3.]).astype(complex)
    u = np.eye(3, dtype=complex); u[:2,:2] = unitary(2, 1)
    u *= np.exp(1j*np.array([.4,-.2,1.3]))
    perm = [2,0,1]
    assert m.basis_certificate(g, u[:,perm], np.array([1.,1.,3.])[perm])['status'] == 'PASS'
    # A non-degenerate rotation is not legalized by closeness of the eigenvalues.
    close = np.diag([1.,1.+1e-6,3.])
    with pytest.raises(ValueError, match='reconstruction'):
        m.basis_certificate(close, u, np.array([1.,1.+1e-6,3.]))


@pytest.mark.parametrize('defect', ['offspin','spinblocks','permutation','nonfinite','nonhermitian'])
def test_ineligible_spin_structure_is_refused_without_projection(defect):
    a = np.kron(np.array([[.3,.11],[.11,-.2]]), np.eye(2)).astype(complex)
    if defect == 'offspin': a[0,1] = 1e-30
    if defect == 'spinblocks': a[1,1] += 1e-15
    if defect == 'permutation': a = a[np.ix_([0,2,1,3], [0,2,1,3])]
    if defect == 'nonfinite': a[0,0] = np.nan
    if defect == 'nonhermitian': a[0,2] += .01; a[1,3] += .01
    with pytest.raises(ValueError): m.spin_adapted_eigh(a)


def test_relative_orientation_complex_conjugation_and_analytic_phase():
    a = model(); builder = m.StrongBuilder(relative_fusion=True)
    i, j = a.blocks[:2]
    bi, bj = builder._basis(a, i), builder._basis(a, j)
    correct = Operator(bj).data.conj().T @ Operator(bi).data
    relative = builder.transition(a, i, j)
    assert np.max(abs(Operator(relative).data-correct)) < 1e-13
    wrong_order, _ = m.precision.gaussian(np.asarray(i.rotation) @ np.asarray(j.rotation).conj().T)
    assert np.max(abs(Operator(wrong_order).data-correct)) > .01
    assert len(builder.relative_cache) == 1
    builder.transition(replace(a, source_identity='OTHER_SOURCE'), i, j)
    assert len(builder.relative_cache) == 2
    with pytest.raises(ValueError, match='VACUUM_PHASE'):
        q = QuantumCircuit(4); q.h(0); m.vacuum_phase(q)
    # Explicit phase branch crossing: the uncorrected synthesis differs by -1.
    i = replace(i, rotation=m.native._rotation(np.diag([np.exp(.8j*np.pi),1,1,1])))
    j = replace(j, rotation=m.native._rotation(np.diag([np.exp(-.8j*np.pi),1,1,1])))
    bi, bj = builder._basis(a,i), builder._basis(a,j)
    target = Operator(bj).data.conj().T @ Operator(bi).data
    corrected = builder.transition(a,i,j)
    wrong, _ = m.precision.gaussian(np.asarray(j.rotation).conj().T @ np.asarray(i.rotation))
    assert np.max(abs(Operator(corrected).data-target)) < 1e-13
    assert np.max(abs(Operator(wrong).data-target)) > 1.9


@pytest.mark.parametrize('time', [Fraction(1,10), Fraction(-1,10)])
def test_full_controlled_evolution_same_basis_fsum_order_adjoint_and_wrapper(time):
    a = model(); a = replace(a, blocks=(*a.blocks[:2], replace(a.blocks[2], rotation=a.blocks[1].rotation)))
    old = m.precision.PrecisionBuilder().ordered(a, time, optimized=True)
    builder = m.StrongBuilder(relative_fusion=True)
    new = builder.ordered(a, time, optimized=True)
    actual, expected = Operator(new).data, Operator(old).data
    assert np.max(abs(actual-expected)) < 1e-13
    assert np.max(abs(actual[:16,:16]-np.eye(16))) < 1e-13
    groups = m.native.runs(a, m.native.ordered_schedule(3, time), optimized=True)
    assert len(builder.fusion_records) == len(groups)-1
    assert new.metadata['scalar_once']
    assert np.max(abs(Operator(m.native.actual_adjoint(new)).data-actual.conj().T)) < 1e-13
    for axis in ('X','Y'):
        wa = m.native.wrapper(new, a, time, axis, optimized=True, occupied=(0,2))
        wb = m.native.wrapper(old, a, time, axis, optimized=True, occupied=(0,2))
        assert np.max(abs(m.fair.decode(wa)-m.fair.decode(wb))) < 1e-13


def synthetic_execute(tmp_path, monkeypatch):
    root = tmp_path/'ARTIFICIAL'; root.mkdir()
    out = root/'output'; out.mkdir()
    for n in ('circuits','operators','candidates','local'): (out/n).mkdir()
    data, _ = fixtures.artificial_data()
    # Use nontrivial frames so all 16 directed boundaries per arm are exercised.
    c = dict(export='e', saved_inputs={'e':'S'}, original_input={'sha256':'I'})
    call = m.integrated.Calls(out, m.LIMITS, m.MAX_CALLS)
    models, reports = m.prepare_models(data, c, call)
    for a in m.fair.ARMS:
        p = root/(a+'_ref.npz')
        gens = data['CDF']
        ref = m.fair.reference._ordered_exponentials(gens, m.native.ordered_schedule(9,'1/10'))
        np.savez_compressed(p, matrix=ref)
    c['references'] = {a:a+'_ref.npz' for a in m.fair.ARMS}; c['shells'] = {}
    for axis in m.fair.SHELLS:
        head, tail = m.fair.connection.readout_segments(data['modes'], 'Y' if axis=='Y' else 'X')
        q = head if axis=='head' else tail
        p = root/(axis+'.npz'); np.savez_compressed(p, matrix=m.fair.decode(q))
        c['shells'][axis] = {k:axis+'.npz' for k in ('raw','compiled')}
    baseline = dict(metrics={a:{k:1 for k in ('rz_count','rz_depth','cx_count','cx_depth','total_depth','circuit_size')} for a in m.fair.ARMS})
    return root, out, data, models, c, call, baseline


def test_synthetic_end_to_end_full_operator_compile_signals_metrics_and_calls(tmp_path, monkeypatch):
    root, out, data, models, c, call, baseline = synthetic_execute(tmp_path, monkeypatch)
    with patch.object(m.fair.stage, 'fit_once', side_effect=AssertionError('NO_FIT')), \
         patch.object(m.fair.synthesis, 'propagate', side_effect=AssertionError('NO_13_ROUND')), \
         patch.object(np.linalg, 'svd', side_effect=AssertionError('NO_SVD')):
        r = m.execute(data, models, c, root, out, call, dict(workers=3), baseline)
    assert r['status'] == 'H4_STRONG_BASELINE_S1_NUMERICAL_COMPLETE'
    assert len(r['candidates']) == 3 and len(r['comparisons']) == 5
    assert len(list((out/'circuits').glob('*.qpy'))) == 21
    assert len(list((out/'operators').glob('*.npz'))) == 6
    for v in r['candidates'].values():
        assert v['segment']['input_dimension'] == v['segment']['output_dimension'] == 32
        assert v['segment']['compiled_reference_frobenius'] < 1e-10
        assert v['XY']['compiled_reference']['error'] < 2e-10
        assert v['CA_achieved'] is v['shots'] is v['total_cost'] is None
    missing = dict(m.LOAD_LIMITS, identity_gate=1, baseline_gate=1)
    assert 0 < call.counts['relative_gaussian'] <= m.LIMITS['relative_gaussian']
    assert sum(call.counts.values()) + sum(missing.values()) <= m.MAX_CALLS
    assert {k:v for k,v in call.counts.items() if k != 'relative_gaussian'} == {k:v for k,v in m.LIMITS.items() if k not in missing and k != 'relative_gaussian'}


@pytest.mark.parametrize('defect', ['phase','direction','order','compile'])
def test_synthetic_wrong_phase_direction_order_compile_stops_before_metrics(tmp_path, monkeypatch, defect):
    root, out, data, models, c, call, baseline = synthetic_execute(tmp_path, monkeypatch)
    if defect in ('phase','direction'):
        original = m.StrongBuilder.transition
        def bad(self, model, previous, following):
            if defect == 'direction': previous, following = following, previous
            q = original(self, model, previous, following).copy()
            if defect == 'phase': q.global_phase += .03
            return q
        monkeypatch.setattr(m.StrongBuilder, 'transition', bad)
    elif defect == 'order':
        original = m.StrongBuilder.ordered
        def bad(self, model, time, **kw):
            return original(self, replace(model, blocks=tuple(reversed(model.blocks))), time, **kw)
        monkeypatch.setattr(m.StrongBuilder, 'ordered', bad)
    else:
        original = m.fair.compile_checked
        def bad(*a):
            q = original(*a); q.global_phase += .03; return q
        monkeypatch.setattr(m.fair, 'compile_checked', bad)
    with pytest.raises((ValueError, m.integrated.Stopped)):
        m.execute(data, models, c, root, out, call, dict(workers=1), baseline)
    assert call.stop.is_set()
    # B1 is independent of relative fusion and may finish before B2 fails.
    assert call.counts['metrics'] == (1 if defect == 'phase' else 0)
    assert (out/'science_calls.jsonl').exists()


def test_unknown_error_is_not_zero_or_rigorous():
    a = np.eye(8,dtype=complex)
    r = m.fair.synthesis.segment(a,a,a,'ARTIFICIAL')
    assert r['rigorous_operator_bound'] is None
    assert not r['floating_point_rounding_enclosed']
    with pytest.raises(ValueError): m.fair.synthesis.segment(a,a*np.nan,a,'ARTIFICIAL')


@pytest.mark.parametrize('change', ['none','source','input','conditions','routes'])
def test_preflight_bytes_only_and_mismatch_before_decode(tmp_path, monkeypatch, change):
    root = tmp_path/'ARTIFICIAL'; root.mkdir()
    paths = ['legacy.json','baseline.json','source.json','stop.json','ref.npz','shell.npz']
    for p in paths: (root/p).write_bytes(b'ARTIFICIAL_BYTES_NOT_ARRAYS')
    (root/'module.py').write_text('ARTIFICIAL_SOURCE')
    m.io.save(root/'SOURCE.json', dict(files={'module.py':m.io.sha(root/'module.py')}))
    c = dict(scope=m.SCOPE, status='UNAPPROVED_SCIENCE_CONTRACT_DRAFT', routes=list(m.ROUTES),
        allowances=m.ALLOWANCES, call_limits=m.LIMITS, maximum_logical_units=m.MAX_CALLS,
        adapter_version=m.VERSION, basis_policy=m.SPIN, source_manifest='SOURCE.json',
        source_manifest_sha256=m.io.sha(root/'SOURCE.json'), saved_inputs={p:m.io.sha(root/p) for p in paths},
        legacy_contract='legacy.json', baseline_result='baseline.json', baseline_SOURCE='source.json', baseline_STOP='stop.json',
        references={a:'ref.npz' for a in m.fair.ARMS}, shells={a:{k:'shell.npz' for k in ('raw','compiled')} for a in m.fair.SHELLS},
        fresh_output=m.FAMILY+'/<execution-date>/run01',
        conditions=dict(m.fair.CONDITIONS, substeps=[1], route='B0 saved; B1 XDF spin; B2 both legacy bases relative fusion'),
        compiler=m.fair.profile_dict(), gaussian=m.fair.GAUSSIAN, compiler_phase=m.fair.PHASE,
        **{k:0 for k in m.FORBIDDEN})
    if change == 'source': (root/'module.py').write_text('ALTERED')
    if change == 'input': (root/'ref.npz').write_bytes(b'ALTERED')
    if change == 'conditions': c['conditions']['substeps'] = [4]
    if change == 'routes': c['routes'].append('B3')
    m.io.save(root/'contract.json', c)
    monkeypatch.setattr(m.fair, 'preflight', lambda *a:dict(ARTIFICIAL=True))
    with patch.object(np, 'load', side_effect=AssertionError('NO_DECODE')), \
         patch.object(m.StrongBuilder, 'ordered', side_effect=AssertionError('NO_BUILD')):
        if change == 'none': assert m.preflight(root,root/'contract.json',root/'packet')[0] == c
        else:
            with pytest.raises(ValueError): m.preflight(root,root/'contract.json',root/'packet')


@pytest.mark.parametrize('fail', [False, True])
def test_full_run_partial_evidence_fresh_and_mandatory_stop(tmp_path, monkeypatch, fail):
    root, _, data, _, c, _, baseline = synthetic_execute(tmp_path, monkeypatch)
    c.update(source_manifest='SOURCE.json', baseline_result='B0.json', baseline_SOURCE='B0_source.json', baseline_STOP='B0_stop.json')
    for p in ('SOURCE.json','B0.json','B0_source.json','B0_stop.json'): m.io.save(root/p, {})
    legacy = dict(export='e', saved_inputs={'e':'S'}, original_input={'sha256':'I'})
    monkeypatch.setattr(m,'preflight',lambda *a:(c,legacy))
    def load(root,legacy,inp,call):
        for k,n in m.LOAD_LIMITS.items():
            for i in range(n): call(k,str(i),lambda:None)
        return data
    monkeypatch.setattr(m.fair,'load_data',load)
    def identity(*a):
        if fail: raise ValueError('ARTIFICIAL_INPUT_IDENTITY_FAIL')
        return dict(status='PASS')
    monkeypatch.setattr(m.fair,'identity',identity)
    monkeypatch.setattr(m,'baseline_gate',lambda *a:baseline)
    original=m.subprocess.check_output
    monkeypatch.setattr(m.subprocess,'check_output',lambda cmd,*a,**k:'ARTIFICIAL\n' if cmd[:2]==['git','rev-parse'] else original(cmd,*a,**k))
    out=root/m.FAMILY/'2026-10-26/run01'
    if fail:
        with pytest.raises(ValueError,match='INPUT_IDENTITY'): m.run(root,root/'contract',root/'input',out)
    else:
        assert m.run(root,root/'contract',root/'input',out)['status']=='H4_STRONG_BASELINE_S1_NUMERICAL_COMPLETE'
    stop=m.io.read(out/'STOP.json')
    assert stop['mandatory_stop'] and stop['failed']==fail and stop['retry']==stop['resume']==0
    assert (out/'SOURCE.json').exists() and (out/'resources.json').exists()
    assert (out/'failure.json').exists()==fail
    with pytest.raises(ValueError,match='FRESH'): m.run(root,root/'contract',root/'input',out)
