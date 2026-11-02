"""Artificial NPZ-to-STOP path; no saved molecular NPZ is opened."""
from dataclasses import asdict
from pathlib import Path
import json,subprocess,sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from trottertracks import c_pr_reduced_limited_pilot as m
from test_research_c_separated_J_U import invented


def fixture(root,monkeypatch):
    g,J,U,s=invented();paths=dict(input='input.npz',initial='initial.npz',scales='scales.json')
    for key,values in [('input',dict(g=g)),('initial',dict(J=J,U=U))]:
        with (root/paths[key]).open('xb') as f:np.savez(f,**values)
    (root/paths['scales']).write_text(json.dumps(dict(initial_tensor_objective=s.loss_scale,initial_smooth_tail_mass=s.mass_scale)))
    entry=root/m.ENTRY;entry.parent.mkdir();entry.write_text('# invented entry\n')
    (root/'SOURCE.json').write_text(json.dumps(dict(files={m.ENTRY:m.io.sha(entry)})))
    c=dict(base_commit=m.BASE,scope=m.SCOPE,entry=m.ENTRY,objective=asdict(s),policy=asdict(m.algorithm.DEFAULT),
        fresh_output=m.FRESH,call_limits=m.LIMITS,maximum_logical_units=m.MAX_CALLS,
        retry=0,resume=0,mandatory_stop=True,source_manifest='SOURCE.json',source_manifest_sha256=m.io.sha(root/'SOURCE.json'),
        input_paths=paths,saved_inputs={p:m.io.sha(root/p) for p in paths.values()},packages={},ffsim_sources={})
    (root/'contract.json').write_text(json.dumps(c));monkeypatch.setattr(m,'OBJECTIVE',asdict(s));return c


def test_default_import_passive_no_numerical_libraries():
    root=Path(__file__).resolve().parents[2]
    code="import sys;sys.path.insert(0,'src');from trottertracks import c_pr_reduced_limited_pilot as m;assert not any(k in sys.modules for k in ('numpy','scipy','jax','ffsim','qiskit'));assert m.MAX_CALLS==54"
    subprocess.run([sys.executable,'-B','-c',code],cwd=root,check=True)


def test_passive_no_decode_no_solver_and_fixed_scope(tmp_path,monkeypatch):
    c=fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('passive decode'))
    assert m.preflight(tmp_path,tmp_path/'contract.json')==c
    c['policy']['max_points']=9;(tmp_path/'contract.json').write_text(json.dumps(c))
    with pytest.raises(ValueError,match='FIXED_PR_REDUCED_LIMITED'):m.preflight(tmp_path,tmp_path/'contract.json')
    assert not (tmp_path/m.FRESH).exists()


@pytest.mark.parametrize('mode',['advance','precision_loss','unqualified','correctness'])
def test_artificial_real_runner_fresh_partial_and_unqualified_STOP(tmp_path,monkeypatch,mode):
    c=fixture(tmp_path,monkeypatch)
    if mode=='precision_loss':
        real=m.algorithm.inner.solve_j
        def stopped(*a,**kw):
            from scipy.optimize import minimize
            def precision(*args,**kwargs):
                r=minimize(*args,**kwargs);r.success=False;r.status=2;r.message='artificial precision loss';return r
            return real(*a,**kw,solver=precision)
        monkeypatch.setattr(m.algorithm.inner,'solve_j',stopped)
    if mode=='unqualified':
        monkeypatch.setattr(m.algorithm.stable.previous,'applicability',lambda *a:dict(admitted=False))
    if mode=='correctness':
        monkeypatch.setattr(m.algorithm.inner,'solve_j',lambda *a,**k:(_ for _ in ()).throw(ValueError('ARTIFICIAL_CONSISTENCY_FAIL')))
        with pytest.raises(ValueError,match='ARTIFICIAL_CONSISTENCY_FAIL'):m.run(tmp_path,tmp_path/'contract.json')
    else:
        result=m.run(tmp_path,tmp_path/'contract.json')
        if mode=='unqualified':
            assert result['status']=='REDUCED_GRADIENT_UNQUALIFIED' and result['completed_U_updates']==0
            assert result['new_outer_optimizations']==0
        else:assert 1<=result['completed_U_updates']<=2
        assert result['DF_budget'] is result['PR_performance'] is result['qualified_tail_comparison'] is None
    out=tmp_path/m.FRESH;stop=m.io.read(out/'STOP.json')
    assert stop['failed']==(mode=='correctness') and stop['scientific_units']<=54
    assert stop['retry']==stop['resume']==0 and stop['mandatory_stop']
    if mode=='unqualified':assert stop['scientific_units']==12
    if mode=='correctness':assert stop['scientific_units']==5 and not (out/'diagnostic_terminal.npz').exists()
    if mode=='precision_loss':
        q=m.io.read(out/'optimization.json')['qualification_by_point']
        assert all(x['status']==2 for p in q for x in p['raw_solver_flags'])
    assert all(m.io.sha(tmp_path/p)==h for p,h in c['saved_inputs'].items())
    with pytest.raises(ValueError,match='FRESH_PR_REDUCED_LIMITED'):m.run(tmp_path,tmp_path/'contract.json')


@pytest.mark.parametrize('fault',['hash','source','scale','dtype'])
def test_input_inconsistency_prevents_dependent_solves(tmp_path,monkeypatch,fault):
    c=fixture(tmp_path,monkeypatch)
    if fault=='hash':(tmp_path/'input.npz').write_bytes(b'invented wrong input')
    elif fault=='source':(tmp_path/m.ENTRY).write_text('# changed source\n')
    else:
        real=m.load
        def bad(*a):
            g,J,U,s=real(*a)
            if fault=='scale':s['initial_tensor_objective']=3.
            else:J=J.astype(np.float32)
            return g,J,U,s
        with pytest.raises(ValueError):m.run(tmp_path,tmp_path/'contract.json',loader=bad)
        assert m.io.read(tmp_path/m.FRESH/'STOP.json')['scientific_units']==2
        return
    with pytest.raises(ValueError,match='SOURCE_OR_INPUT'):m.run(tmp_path,tmp_path/'contract.json')
    assert not (tmp_path/m.FRESH).exists()
