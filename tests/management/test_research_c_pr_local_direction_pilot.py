"""Invented NPZ-to-directional-diagnosis E2E, no saved H4 decode."""
from pathlib import Path
from dataclasses import asdict
import json,sys,subprocess
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from trottertracks import c_pr_local_direction_pilot as m
from test_research_c_separated_J_U import invented


def fixture(root,monkeypatch):
    g,J,U,s=invented();paths=dict(input='input.npz',initial='initial.npz',scales='scales.json')
    for name,values in [('input',dict(g=g)),('initial',dict(J=J,U=U))]:
        with (root/paths[name]).open('xb') as f:np.savez(f,**values)
    (root/paths['scales']).write_text(json.dumps(dict(initial_tensor_objective=s.loss_scale,initial_smooth_tail_mass=s.mass_scale)))
    entry=root/m.ENTRY;entry.parent.mkdir();entry.write_text('# synthetic entry\n')
    (root/'SOURCE.json').write_text(json.dumps(dict(files={m.ENTRY:m.io.sha(entry)})))
    c=dict(base_commit=m.BASE,scope=m.SCOPE,entry=m.ENTRY,objective=asdict(s),policy=asdict(m.algorithm.DEFAULT),fractions=list(m.algorithm.FRACTIONS),
        fresh_output=m.FRESH,call_limits=m.LIMITS,maximum_logical_units=m.MAX_CALLS,retry=0,resume=0,mandatory_stop=True,
        source_manifest='SOURCE.json',source_manifest_sha256=m.io.sha(root/'SOURCE.json'),input_paths=paths,saved_inputs={p:m.io.sha(root/p) for p in paths.values()},packages={},ffsim_sources={})
    (root/'contract.json').write_text(json.dumps(c));monkeypatch.setattr(m,'OBJECTIVE',asdict(s));return c


def test_passive_no_decode_and_49_units(tmp_path,monkeypatch):
    c=fixture(tmp_path,monkeypatch);monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('passive decode'))
    assert m.preflight(tmp_path,tmp_path/'contract.json')==c and m.MAX_CALLS==49
    c['fractions'][0]=.25;(tmp_path/'contract.json').write_text(json.dumps(c))
    with pytest.raises(ValueError,match='FIXED_LOCAL_DIRECTION'):m.preflight(tmp_path,tmp_path/'contract.json')
    assert not (tmp_path/m.FRESH).exists()


def test_passive_import_no_numerical_libraries():
    code="import sys;sys.path.insert(0,'src');from trottertracks import c_pr_local_direction_pilot;assert not any(k in sys.modules for k in ('numpy','scipy','jax','ffsim','qiskit'))"
    subprocess.run([sys.executable,'-B','-c',code],cwd=Path(__file__).resolve().parents[2],check=True)


@pytest.mark.parametrize('mode',['success','one_side','initial','correctness'])
def test_artificial_E2E_probe_qualification_partial_STOP_and_counts(tmp_path,monkeypatch,mode):
    c=fixture(tmp_path,monkeypatch);real=m.algorithm.previous.Oracle.evaluate
    def evaluate(self,eta):
        if mode=='correctness' and self.points:raise ValueError('ARTIFICIAL_CORRECTNESS_FAIL')
        p=real(self,eta)
        if mode=='initial' or mode=='one_side' and p['label']=='1':
            p['qualification']['status']='REDUCED_GRADIENT_UNQUALIFIED';raise m.algorithm.stable.Unqualified('REDUCED_GRADIENT_UNQUALIFIED')
        return p
    monkeypatch.setattr(m.algorithm.previous.Oracle,'evaluate',evaluate)
    if mode=='correctness':
        with pytest.raises(ValueError,match='ARTIFICIAL_CORRECTNESS_FAIL'):m.run(tmp_path,tmp_path/'contract.json')
    else:
        r=m.run(tmp_path,tmp_path/'contract.json')
        assert r['accepted_U_updates']==0 and r['outer_BFGS_invocations']==0
        assert r['true_reduced_gradient_qualification'] is r['certified_descent'] is r['DF_budget'] is None
        if mode=='initial':assert r['status']=='INITIAL_UNQUALIFIED_NO_LOCAL_DIAGNOSTICS'
        elif mode=='one_side':assert r['status']=='LOCAL_DIAGNOSTICS_PARTIAL'
        else:assert r['status']=='LOCAL_DIRECTIONAL_DIAGNOSTICS_AVAILABLE'
    out=tmp_path/m.FRESH;stop=m.io.read(out/'STOP.json')
    assert stop['failed']==(mode=='correctness') and stop['scientific_units']<=49
    assert stop['scientific_units']==(10 if mode=='correctness' else 13 if mode=='initial' else 49)
    assert stop['retry']==stop['resume']==0 and stop['mandatory_stop']
    if mode=='correctness':assert not (out/'diagnostic_initial.npz').exists()
    assert all(m.io.sha(tmp_path/p)==h for p,h in c['saved_inputs'].items())
    with pytest.raises(ValueError,match='FRESH_LOCAL_DIRECTION'):m.run(tmp_path,tmp_path/'contract.json')


def test_input_mismatch_prevents_any_probe(tmp_path,monkeypatch):
    fixture(tmp_path,monkeypatch);(tmp_path/'input.npz').write_bytes(b'invented mismatch')
    with pytest.raises(ValueError,match='SOURCE_OR_INPUT'):m.run(tmp_path,tmp_path/'contract.json')
    assert not (tmp_path/m.FRESH).exists()
