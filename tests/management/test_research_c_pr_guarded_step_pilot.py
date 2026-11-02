"""Invented NPZ E2E only; no molecular data are opened."""
from dataclasses import asdict
from pathlib import Path
import json,sys,subprocess
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from trottertracks import c_pr_guarded_step_pilot as m
from test_research_c_separated_J_U import invented


def fixture(root,monkeypatch):
    g,J,U,s=invented();paths=dict(input='input.npz',initial='initial.npz',scales='scales.json')
    for name,values in [('input',dict(g=g)),('initial',dict(J=J,U=U))]:
        with (root/paths[name]).open('xb') as f:np.savez(f,**values)
    (root/paths['scales']).write_text(json.dumps(dict(initial_tensor_objective=s.loss_scale,initial_smooth_tail_mass=s.mass_scale)))
    entry=root/m.ENTRY;entry.parent.mkdir();entry.write_text('# artificial identity only\n')
    (root/'SOURCE.json').write_text(json.dumps(dict(files={m.ENTRY:m.io.sha(entry)})))
    c=dict(base_commit=m.BASE,scope=m.SCOPE,entry=m.ENTRY,objective=asdict(s),policy=asdict(m.algorithm.DEFAULT),search_policy=asdict(m.algorithm.SEARCH),
        fresh_output=m.FRESH,call_limits=m.LIMITS,maximum_logical_units=m.MAX_CALLS,retry=0,resume=0,mandatory_stop=True,
        source_manifest='SOURCE.json',source_manifest_sha256=m.io.sha(root/'SOURCE.json'),input_paths=paths,
        saved_inputs={p:m.io.sha(root/p) for p in paths.values()},packages={},ffsim_sources={})
    (root/'contract.json').write_text(json.dumps(c));monkeypatch.setattr(m,'OBJECTIVE',asdict(s));return c


def test_passive_no_decode_and_budget60(tmp_path,monkeypatch):
    c=fixture(tmp_path,monkeypatch);monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('passive decode'))
    assert m.preflight(tmp_path,tmp_path/'contract.json')==c and m.MAX_CALLS==60
    c['search_policy']['max_trials']=9;(tmp_path/'contract.json').write_text(json.dumps(c))
    with pytest.raises(ValueError,match='FIXED_GUARDED'):m.preflight(tmp_path,tmp_path/'contract.json')
    assert not (tmp_path/m.FRESH).exists()


def test_import_has_no_numerical_side_effects():
    code="import sys;sys.path.insert(0,'src');from trottertracks import c_pr_guarded_step_pilot;assert not any(k in sys.modules for k in ('numpy','scipy','jax','ffsim','qiskit'))"
    subprocess.run([sys.executable,'-B','-c',code],cwd=Path(__file__).resolve().parents[2],check=True)


@pytest.mark.parametrize('mode',['success','first_trial_unqualified','all_trials_unqualified','initial_unqualified','correctness'])
def test_artificial_pipeline_calls_partial_and_fresh_STOP(tmp_path,monkeypatch,mode):
    c=fixture(tmp_path,monkeypatch);real=m.algorithm.Oracle.evaluate
    def evaluate(self,eta):
        # Inject numerical qualification at a registered completed point; do
        # not create made-up gradients or bypass any genuine initial solve.
        if mode=='correctness' and self.points:raise ValueError('ARTIFICIAL_CORRECTNESS_FAIL')
        p=real(self,eta)
        reject=(mode=='initial_unqualified' or mode=='first_trial_unqualified' and p['label']=='1'
                or mode=='all_trials_unqualified' and p['label']!='0')
        if reject:
            p['qualification']['status']='REDUCED_GRADIENT_UNQUALIFIED'
            raise m.algorithm.stable.Unqualified('REDUCED_GRADIENT_UNQUALIFIED')
        return p
    monkeypatch.setattr(m.algorithm.Oracle,'evaluate',evaluate)
    if mode=='correctness':
        with pytest.raises(ValueError,match='ARTIFICIAL_CORRECTNESS_FAIL'):m.run(tmp_path,tmp_path/'contract.json')
    else:
        r=m.run(tmp_path,tmp_path/'contract.json')
        if mode in ('success','first_trial_unqualified'):assert r['status']=='QUALIFIED_ARMIJO_STEP_ACCEPTED' and r['accepted_U_updates']==1
        if mode=='all_trials_unqualified':assert r['status']=='GUARDED_TRIAL_LIMIT_NO_ACCEPT' and r['trials']==8 and r['distinct_U_points']==9
        if mode=='initial_unqualified':assert r['status']=='REDUCED_GRADIENT_UNQUALIFIED' and r['trials']==0
        assert r['DF_budget'] is r['PR_performance'] is None
    out=tmp_path/m.FRESH;stop=m.io.read(out/'STOP.json')
    assert stop['failed']==(mode=='correctness') and stop['scientific_units']<=60
    assert stop['retry']==stop['resume']==0 and stop['mandatory_stop']
    if mode=='all_trials_unqualified':assert stop['scientific_units']==60
    if mode=='initial_unqualified':assert stop['scientific_units']==12
    if mode=='correctness':assert not (out/'diagnostic_terminal.npz').exists()
    assert all(m.io.sha(tmp_path/p)==h for p,h in c['saved_inputs'].items())
    with pytest.raises(ValueError,match='FRESH_GUARDED'):m.run(tmp_path,tmp_path/'contract.json')


def test_input_mismatch_no_science(tmp_path,monkeypatch):
    fixture(tmp_path,monkeypatch);(tmp_path/'input.npz').write_bytes(b'invented mismatch')
    with pytest.raises(ValueError,match='SOURCE_OR_INPUT'):m.run(tmp_path,tmp_path/'contract.json')
    assert not (tmp_path/m.FRESH).exists()
