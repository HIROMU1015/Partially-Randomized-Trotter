"""Synthetic NPZ end-to-end, passive preflight and one-shot result semantics."""
from dataclasses import asdict
from pathlib import Path
import json,sys,subprocess
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from trottertracks import c_pr_small_step_pilot as m
from test_research_c_separated_J_U import invented


def fixture(root,monkeypatch):
    g,J,U,s=invented();paths=dict(input='input.npz',initial='initial.npz',scales='scales.json')
    for name,arrays in [('input',dict(g=g)),('initial',dict(J=J,U=U))]:
        with (root/paths[name]).open('xb') as f:np.savez(f,**arrays)
    (root/paths['scales']).write_text(json.dumps(dict(initial_tensor_objective=s.loss_scale,initial_smooth_tail_mass=s.mass_scale)))
    entry=root/m.ENTRY;entry.parent.mkdir();entry.write_text('# synthetic entry\n')
    (root/'SOURCE.json').write_text(json.dumps(dict(files={m.ENTRY:m.io.sha(entry)})))
    c=dict(base_commit=m.BASE,scope=m.SCOPE,entry=m.ENTRY,objective=asdict(s),policy=asdict(m.algorithm.DEFAULT),step_policy=asdict(m.algorithm.STEP),
        fresh_output=m.FRESH,call_limits=m.LIMITS,maximum_logical_units=m.MAX_CALLS,retry=0,resume=0,mandatory_stop=True,
        source_manifest='SOURCE.json',source_manifest_sha256=m.io.sha(root/'SOURCE.json'),input_paths=paths,
        saved_inputs={p:m.io.sha(root/p) for p in paths.values()},packages={},ffsim_sources={})
    (root/'contract.json').write_text(json.dumps(c));monkeypatch.setattr(m,'OBJECTIVE',asdict(s));return c


def test_passive_no_decode_fixed19_and_no_import_side_effect(tmp_path,monkeypatch):
    c=fixture(tmp_path,monkeypatch);monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('passive decode'))
    assert m.preflight(tmp_path,tmp_path/'contract.json')==c and m.MAX_CALLS==19
    c['step_policy']['fraction']=1/32;(tmp_path/'contract.json').write_text(json.dumps(c))
    with pytest.raises(ValueError,match='FIXED_PR_SMALL_STEP'):m.preflight(tmp_path,tmp_path/'contract.json')
    code="import sys;sys.path.insert(0,'src');from trottertracks import c_pr_small_step_pilot;assert not any(k in sys.modules for k in ('numpy','scipy','jax','ffsim','qiskit'))"
    subprocess.run([sys.executable,'-B','-c',code],cwd=Path(__file__).resolve().parents[2],check=True)


@pytest.mark.parametrize('mode',['success','initial_unqualified','trial_unqualified','no_decrease','post_unqualified','correctness','post_correctness'])
def test_artificial_E2E_accept_unqualified_decline_FAIL_and_STOP(tmp_path,monkeypatch,mode):
    c=fixture(tmp_path,monkeypatch);real=m.algorithm.Oracle.evaluate;assess=m.algorithm.assess
    def evaluate(self,eta):
        if mode=='correctness' and self.points:raise ValueError('ARTIFICIAL_CORRECTNESS_FAIL')
        p=real(self,eta)
        if mode=='initial_unqualified' or mode=='trial_unqualified' and p['label']=='1':
            p['qualification']['status']='REDUCED_GRADIENT_UNQUALIFIED';raise m.algorithm.stable.Unqualified('REDUCED_GRADIENT_UNQUALIFIED')
        if mode=='no_decrease' and p['label']=='1':p['value']=self.points[0]['value']+1.
        return p
    def assess_modified(*a,**k):
        q=assess(*a,**k)
        if mode=='post_correctness':raise ValueError('INDEPENDENT_TERMINAL_STATE_MISMATCH')
        if mode=='post_unqualified':return dict(q,status='POST_UPDATE_J_UNQUALIFIED',J_numerically_qualified=False)
        return q
    monkeypatch.setattr(m.algorithm.Oracle,'evaluate',evaluate);monkeypatch.setattr(m.algorithm,'assess',assess_modified)
    failed=mode in ('correctness','post_correctness')
    if failed:
        with pytest.raises(ValueError):m.run(tmp_path,tmp_path/'contract.json')
    else:
        r=m.run(tmp_path,tmp_path/'contract.json')
        assert r['accepted_U_updates']==int(mode=='success')
        assert r['trials']==int(mode!='initial_unqualified')
        assert r['DF_budget'] is r['PR_performance'] is r['certified_descent'] is None
        if mode=='success':assert r['terminal_kind']=='accepted_U_update'
        else:
            assert r['terminal_label']=='0'
            if mode!='post_unqualified':assert r['independent_terminal_assessment'] is None
    out=tmp_path/m.FRESH;stop=m.io.read(out/'STOP.json')
    assert stop['failed']==failed and stop['scientific_units']<=19
    assert stop['retry']==stop['resume']==0 and stop['mandatory_stop']
    if mode=='success':assert stop['scientific_units']==19
    if mode=='initial_unqualified':assert stop['scientific_units']==12
    if mode in ('trial_unqualified','no_decrease'):assert stop['scientific_units']==18
    if failed:assert not (out/'diagnostic_terminal.npz').exists()
    assert all(m.io.sha(tmp_path/p)==h for p,h in c['saved_inputs'].items())
    with pytest.raises(ValueError,match='FRESH_PR_SMALL_STEP'):m.run(tmp_path,tmp_path/'contract.json')


def test_input_mismatch_is_passive_STOP(tmp_path,monkeypatch):
    fixture(tmp_path,monkeypatch);(tmp_path/'initial.npz').write_bytes(b'artificial mismatch')
    with pytest.raises(ValueError,match='SOURCE_OR_INPUT'):m.run(tmp_path,tmp_path/'contract.json')
    assert not (tmp_path/m.FRESH).exists()
