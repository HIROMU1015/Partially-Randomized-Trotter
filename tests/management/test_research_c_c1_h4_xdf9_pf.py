"""Invented packets only; reuse the XDF8 fixture and failure-law mocks."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks import c1_h4_xdf9_pf as m

spec = importlib.util.spec_from_file_location('xdf8_artificial',
    ROOT/'tests/management/test_research_c_c1_h4_xdf8_pf.py')
previous_tests = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous_tests)


def fixture(tmp_path, monkeypatch):
    root, c, path = previous_tests.fixture(tmp_path, monkeypatch)
    row = m.io.read(root/'row.json'); row['rank'] = 9
    (root/'row.json').write_text(json.dumps(row))  # invented fixture only
    c.update(scope=m.SCOPE, conditions=copy.deepcopy(m.CONDITIONS),
        allowances=copy.deepcopy(m.ALLOWANCES), call_limits=dict(m.LIMITS),
        maximum_logical_units=m.MAX_CALLS, fresh_output=m.FAMILY+'/<execution-date>/run01',
        saved_inputs={p:m.io.sha(root/p) for p in c['saved_inputs']},
        **{k:0 for k in m.FORBIDDEN})
    (root/'contract.json').write_text(json.dumps(c))
    return root, c, path


def execute(root, path, **kwargs):
    output = root/m.FAMILY/'2026-10-26/run01'
    original = subprocess.check_output
    def git_only(command, *args, **kw):
        return 'ARTIFICIAL_SOURCE\n' if command[:2] == ['git','rev-parse'] else original(command,*args,**kw)
    with patch.object(m.subprocess, 'check_output', side_effect=git_only):
        return m.run(root, root/'contract.json', path, output, **kwargs), output


def test_passive_preflight_without_decode_or_numeric_import(tmp_path, monkeypatch):
    root, c, path = fixture(tmp_path, monkeypatch)
    with patch.object(np, 'load', side_effect=AssertionError('NO DECODE')):
        assert m.preflight(root, root/'contract.json', path) == c
    subprocess.run([sys.executable, '-c',
        "import sys;sys.path.insert(0,sys.argv[1]);from trottertracks import c1_h4_xdf9_pf;assert 'numpy' not in sys.modules",
        str(ROOT/'src')], check=True)
    assert not (root/m.FAMILY).exists()


def test_artificial_e2e_95_calls_and_independent_ordered_reference(tmp_path, monkeypatch):
    root, c, path = fixture(tmp_path, monkeypatch)
    forbidden = AssertionError('NO DF-error/fitter/physical Hamiltonian')
    with patch.object(m.stage1, 'fit_once', side_effect=forbidden), \
         patch.object(m.stage1, 'physical_sector', side_effect=forbidden), \
         patch.object(m.prefix, 'error_norm', side_effect=forbidden):
        result, output = execute(root, path, workers=2)
    stop = m.io.read(output/'STOP.json')
    assert stop['scientific_units'] == m.MAX_CALLS == 95
    assert stop['call_counts'] == m.LIMITS and not stop['failed']
    assert stop['retry'] == stop['resume'] == 0
    assert result['status'] == 'XDF9_STAGE1_COMPLETE'
    assert result['saved_DF']['rank'] == 9 and not result['saved_DF']['recomputed']
    assert result['ground']['state_model'] == m.CONDITIONS['target']
    assert len(list((output/'cells').glob('*.json'))) == 26
    events = [json.loads(line) for line in (output/'science_calls.jsonl').read_text().splitlines()]
    assert sum(e['event'] == 'attempt' for e in events) == 95
    with np.load(output/'generators.npz') as a: factors = a['generators']
    assert factors.shape == (10, 36, 36)
    with np.load(output/'ground_XDF9.npz') as a:
        assert np.linalg.norm(sum(factors)@a['sector_state']-a['energy']*a['sector_state']) < 1e-12
    for s in (1, 4):
        h = .1/s
        schedule = [(i,h/2) for i in range(9)]+[(9,h)]+[(i,h/2) for i in reversed(range(9))]
        W = np.eye(36, dtype=complex)
        for i, t in schedule: W = expm(-1j*t*factors[i])@W
        with np.load(output/'kernels'/f'XDF9_s{s}.npz') as a:
            np.testing.assert_allclose(a['pf'], W, atol=1e-14, rtol=0)
        for index in (0, 12):
            expected = np.linalg.norm(np.linalg.matrix_power(W,s*(1<<index))-
                expm(-1j*.1*(1<<index)*sum(factors)), 2)
            row = result['candidates'][str(s)]['rounds'][index]
            assert abs(row['pf_operator_error']-expected) < 2e-11
            assert row['occurrences'] == s*(1<<index)
            assert row['kernel_time'] == .1/s and row['time'] == .1*(1<<index)
        assert result['candidates'][str(s)]['summary']['status'] == 'POTENTIALLY_FEASIBLE'
        assert result['candidates'][str(s)]['summary']['CA_achieved'] is None


def test_ninth_factor_independent_K_and_unused_tenth(tmp_path, monkeypatch):
    root, c, path = fixture(tmp_path, monkeypatch)
    b = m.load_saved(root,c,path); ctx = m.contexts()
    assert m.identity(b,ctx)['frame_order'] == list(range(9))
    model = m.k_model(b)
    raw = [m.raw_factor(b,ctx,i) for i in range(10)]
    assert len(model.frames) == 9
    for i in range(10): assert m.representation(ctx,model,raw[i],i)[0]['max_entry'] < 1e-12
    p = b['packet']; p['df_lambdas'] = p['df_lambdas'].copy(); p['df_lambdas'][9] = 1e8
    assert m.k_model(b) == model
    np.testing.assert_array_equal(m.raw_factor(b,ctx,9), raw[9])
    p['df_lambdas'][8] *= 2
    np.testing.assert_allclose(m.raw_factor(b,ctx,9), 2*raw[9], atol=0, rtol=0)
    broken = copy.copy(model); object.__setattr__(broken,'frames',model.frames[::-1])
    with pytest.raises(ValueError,match='FRAME_ORDER'): m.representation(ctx,broken,raw[1],1)


def test_original_g_kappa_once_and_scalar_unchanged(tmp_path, monkeypatch):
    root,c,path = fixture(tmp_path,monkeypatch); b = m.load_saved(root,c,path); p = b['packet']; ctx = m.contexts()
    p['eri_chemist'] = p['eri_chemist'].copy(); p['eri_chemist'][0,1,1,0] = .25
    p['df_one_body'] = np.kron(p['one_mo']-np.diag([.125,0,0,0]),np.eye(2)).astype(complex)
    with patch.object(m.stage1,'corrected_kappa',wraps=m.stage1.corrected_kappa) as correction:
        m.identity(b,ctx); model = m.k_model(b)
        for i in range(10): m.representation(ctx,model,m.raw_factor(b,ctx,i),i)
        assert correction.call_count == 1 and model.constant == .25


@pytest.mark.parametrize('error,status',[(.4,'INFEASIBLE_REGISTERED_PHASE_ROOM'),(None,'UNRESOLVED')])
def test_insufficient_unknown_and_no_CA_claim(tmp_path,monkeypatch,error,status):
    root,c,path = fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(m.molecule,'PF_rounds',previous_tests.fake_pf(error))
    result,output = execute(root,path,workers=2)
    assert {x['summary']['status'] for x in result['candidates'].values()} == {status}
    assert not m.io.read(output/'STOP.json')['failed']


@pytest.mark.parametrize('error',[float('nan'),-.1])
def test_invalid_PF_partial_stop(tmp_path,monkeypatch,error):
    root,c,path = fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(m.molecule,'PF_rounds',previous_tests.fake_pf(error))
    with pytest.raises(ValueError,match='CORRECTNESS_STOP'): execute(root,path,workers=1)
    output = root/m.FAMILY/'2026-10-26/run01'; stop = m.io.read(output/'STOP.json')
    assert stop['failed'] and stop['scientific_units'] < 95
    assert (output/'partial_candidates.json').exists() and (output/'failure.json').exists()


def test_ninth_spin_defect_stops_before_model_and_PF(tmp_path,monkeypatch):
    root,c,path = fixture(tmp_path,monkeypatch); original = m.load_saved
    def broken(*args):
        b = original(*args); p = b['packet']; p['df_g_matrices'] = p['df_g_matrices'].copy()
        p['df_g_matrices'][8,0,1] = 1e-20; return b
    monkeypatch.setattr(m,'load_saved',broken)
    with pytest.raises(ValueError,match='SPIN_LIFT'): execute(root,path,workers=1)
    stop = m.io.read(root/m.FAMILY/'2026-10-26/run01/STOP.json')
    assert stop['failed'] and not stop['call_counts'].get('xdf_prefix_model')


def test_unresolved_alias_no_PF(tmp_path,monkeypatch):
    root,c,path = fixture(tmp_path,monkeypatch); original = m.ground_gate
    monkeypatch.setattr(m,'ground_gate',lambda *args: dict(original(*args),status='UNRESOLVED_ALIAS_WINDOW'))
    with patch.object(m.molecule,'PF_rounds',side_effect=AssertionError('NO PF')):
        result,output = execute(root,path,workers=1)
    assert result['status'] == 'UNRESOLVED_GROUND_ALIAS' and result['candidates'] == {}


@pytest.mark.parametrize('name',['code.txt','row.json','artificial.npz'])
def test_identity_fail_before_decode(tmp_path,monkeypatch,name):
    root,c,path = fixture(tmp_path,monkeypatch); (root/name).write_bytes(b'ARTIFICIAL_DRIFT')
    with patch.object(np,'load',side_effect=AssertionError('NO DECODE')):
        with pytest.raises(ValueError,match='SOURCE|INPUT'): m.preflight(root,root/'contract.json',path)


def test_saved_rank_and_source_mismatch_and_no_resume(tmp_path,monkeypatch):
    root,c,path = fixture(tmp_path,monkeypatch); b = m.load_saved(root,c,path)
    b['records']['DF_row']['rank'] = 8
    with pytest.raises(ValueError,match='DF_IDENTITY'): m.saved_df_gate(b,c)
    b['records']['DF_row']['rank'] = 9; b['records']['DF_SOURCE']['execution_commit'] = 'WRONG'
    with pytest.raises(ValueError,match='DF_IDENTITY'): m.saved_df_gate(b,c)
    execute(root,path,workers=2)
    with pytest.raises(ValueError,match='FRESH_OUTPUT'): execute(root,path,workers=1)
    with pytest.raises(ValueError): m.output_path(root,root/m.FAMILY/'2026-10-26/run02')
