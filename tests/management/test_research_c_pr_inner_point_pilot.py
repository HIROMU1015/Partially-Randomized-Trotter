"""Actual temporary NPZ/JSON fixtures, invented tensors only."""
from dataclasses import asdict
from pathlib import Path
import json,sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from trottertracks import c_pr_inner_point_pilot as m
from test_research_c_separated_J_U import invented


def fixture(root,monkeypatch,fault=None):
    g,J,U,s=invented();paths=dict(input='input.npz',initial='initial.npz',diagnostic='diagnostic.npz',saved_result='old_result.json',saved_identity='old_identity.json')
    for p,values in [(paths['input'],dict(g=g)),(paths['initial'],dict(J=J)),(paths['diagnostic'],dict(U=U))]:
        with (root/p).open('xb') as f:np.savez(f,**values)
    (root/paths['saved_result']).write_text(json.dumps(dict(status='NEW_SAVED_POINT_NUMERICAL_CONSISTENCY_PASS')))
    (root/paths['saved_identity']).write_text(json.dumps(dict(status='FAIL' if fault=='identity' else 'SAVED_TRACE_AND_FAILED_POINT_IDENTITIES_PASS',new_scientific_objective_queries_in_replay=0,accepted_replay_steps=0)))
    (root/'source.py').write_text('# invented source identity\n')
    (root/'SOURCE.json').write_text(json.dumps(dict(files={'source.py':m.io.sha(root/'source.py')})))
    c=dict(base_commit=m.BASE,scope=m.SCOPE,entry='scripts/run_research_c_pr_inner_point.py',objective=asdict(s),policy=asdict(m.algorithm.DEFAULT),fresh_output=m.FRESH,call_limits=m.LIMITS,maximum_logical_units=8,new_inner_solver_invocations=1,new_refinements=0,new_outer_optimizations=0,new_U_updates=0,retry=0,resume=0,mandatory_stop=True,source_manifest='SOURCE.json',source_manifest_sha256=m.io.sha(root/'SOURCE.json'),input_paths=paths,saved_inputs={p:m.io.sha(root/p) for p in paths.values()},packages={},ffsim_sources={})
    (root/'contract.json').write_text(json.dumps(c));monkeypatch.setattr(m.previous,'OBJECTIVE',asdict(s))
    return c


def test_passive_no_decode_and_unchanged_fixed_policy(tmp_path,monkeypatch):
    c=fixture(tmp_path,monkeypatch);monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('passive scientific decode'))
    assert m.preflight(tmp_path,tmp_path/'contract.json')==c
    c['policy']['j_maxiter']=2000;(tmp_path/'contract.json').write_text(json.dumps(c))
    with pytest.raises(ValueError,match='FIXED_PR_INNER_PILOT'):m.preflight(tmp_path,tmp_path/'contract.json')
    assert not (tmp_path/m.FRESH).exists()


@pytest.mark.parametrize('fault',[None,'identity','inner','state'])
def test_artificial_one_solve_E2E_and_partial_STOP(tmp_path,monkeypatch,fault):
    c=fixture(tmp_path,monkeypatch,fault);seen=[];real=m.algorithm.solve_j
    def solve(*args,**kwargs):
        seen.append(True)
        if fault=='inner':raise ValueError('ARTIFICIAL_INNER_CORRECTNESS_FAIL')
        return real(*args,**kwargs)
    monkeypatch.setattr(m.algorithm,'solve_j',solve)
    if fault=='state':
        monkeypatch.setattr(m.algorithm.stable,'state',lambda *a,**k:(_ for _ in ()).throw(ValueError('ARTIFICIAL_STATE_FAIL')))
    if fault:
        with pytest.raises(ValueError):m.run(tmp_path,tmp_path/'contract.json')
    else:m.run(tmp_path,tmp_path/'contract.json')
    out=tmp_path/m.FRESH;stop=m.io.read(out/'STOP.json')
    assert stop['scientific_units']==({'identity':2,'inner':4,'state':5}.get(fault,8))
    assert len(seen)==(0 if fault=='identity' else 1)
    assert stop['failed']==bool(fault) and stop['retry']==stop['resume']==0
    if not fault:
        r=m.io.read(out/'result.json');assert r['status']=='INNER_CORRECTNESS_PASS_J_MET'
        assert r['new_inner_solver_invocations']==1 and r['new_U_updates']==r['new_outer_optimizations']==r['new_refinements']==0
        assert r['DF_budget'] is r['reduced_gradient_qualification'] is None
        with np.load(out/'diagnostic_endpoint.npz') as a, np.load(tmp_path/'diagnostic.npz') as b:np.testing.assert_array_equal(a['U'],b['U'])
    with pytest.raises(ValueError,match='FRESH_PR_INNER'):m.run(tmp_path,tmp_path/'contract.json')
    assert all(m.io.sha(tmp_path/p)==h for p,h in c['saved_inputs'].items())


def test_input_hash_mismatch_prevents_any_solve(tmp_path,monkeypatch):
    fixture(tmp_path,monkeypatch);(tmp_path/'input.npz').write_bytes(b'changed input')
    with pytest.raises(ValueError,match='SOURCE_OR_INPUT'):m.run(tmp_path,tmp_path/'contract.json')
    assert not (tmp_path/m.FRESH).exists()
