"""Artificial exact CAR and mocked child fits; no saved molecular data."""
from pathlib import Path
from unittest.mock import patch
import sys,subprocess,json
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c_cdf7_onefit as m


@pytest.mark.parametrize('fault',[None,'final_budget','optimizer','codec','representation'])
def test_onefit_dispatch_initial_NOT_MET_allowed_and_STOP(tmp_path,monkeypatch,fault):
    monkeypatch.setattr(m,'FRESH','fresh')
    c=dict(entry='fake.py',source_manifest='SOURCE.json')
    m.io.save(tmp_path/'SOURCE.json',dict(files={}))
    monkeypatch.setattr(m,'preflight',lambda *a,**k:c)
    monkeypatch.setattr(m,'saved_comparison',lambda *a:{'baseline':{'actual_rank':8}})
    out=tmp_path/'fresh'
    original=m.molecule.representation_gate
    if fault=='representation':
        def fail(*a,**k):
            original(*a,**k)
            raise ValueError('ARTIFICIAL_REPRESENTATION_FAIL')
        monkeypatch.setattr(m.molecule,'representation_gate',fail)
    jobs=[];original_run=m.subprocess.run
    def worker(args,**kwargs):
        if '--worker' not in args:
            if args[:2]==['git','rev-parse']:
                return subprocess.CompletedProcess(args,0,stdout='ARTIFICIAL_SOURCE\n')
            return original_run(args,**kwargs)
        name=args[args.index('--worker')+1];jobs.append(name);child=out/name;child.mkdir()
        calls=m.old.Calls(child,m.PREPARE if name=='prepare' else m.FIT,
            sum((m.PREPARE if name=='prepare' else m.FIT).values()))
        for kind,n in (m.PREPARE if name=='prepare' else m.FIT).items():
            for i in range(n):calls(kind,'ARTIFICIAL_CHILD',lambda:None)
        J=np.repeat((np.eye(4)*.01)[None],7,axis=0);U=np.repeat(np.eye(4)[None],7,axis=0)
        g=np.zeros((4,)*4)
        if name=='prepare':
            m.pilot.arrays(child/'input.npz',h=np.zeros((4,4)),g=g,constant=np.array(.5))
            for label in ('before','initial'):
                current=J*.5 if fault=='codec' and label=='initial' else J
                m.pilot.arrays(child/(label+'.npz'),J=current,U=U)
                m.io.save(child/(label+'_tensor.json'),m.old.tensor_metrics(g,current,U))
        else:
            gate=m.io.read(out/'initial_gate.json')
            assert gate['status']=='PASS' and not any(gate['initial_budget_diagnostic'].values())
            assert gate['initial_budget_is_admission'] is False
            if fault!='final_budget':J=np.zeros_like(J)
            m.pilot.arrays(child/'final.npz',J=J,U=U)
            m.io.save(child/'tensor.json',m.old.tensor_metrics(g,J,U))
            m.io.save(child/'optimizer.json',dict(success=fault!='optimizer',status=0,
                message='ARTIFICIAL',fun=0.,nit=0,nfev=1,njev=1))
        m.pilot.finish(child,calls,m.time.monotonic())
        return subprocess.CompletedProcess(args,0)
    monkeypatch.setattr(m.subprocess,'run',worker)
    if fault in ('codec','representation'):
        with pytest.raises((ValueError,RuntimeError)):
            m.run(tmp_path,tmp_path/'contract.json',out,'fit-python')
        assert jobs==['prepare'] and m.io.read(out/'STOP.json')['failed']
        assert not (out/'fit').exists()
    else:
        m.run(tmp_path,tmp_path/'contract.json',out,'fit-python')
        result=m.io.read(out/'result.json');stop=m.io.read(out/'STOP.json')
        expected='CDF7_DF_ELIGIBLE' if fault is None else 'CDF7_DF_NOT_MET'
        assert result['status']==expected and jobs==['prepare','fit']
        assert result['PF_error'] is result['shots'] is result['native_price'] is None
        assert stop['scientific_units']==(25 if fault is None else 24)
        if fault is None:assert stop['call_counts']==m.LIMITS
        assert not stop['failed'] and stop['retry']==stop['resume']==0
        content=(out/'STOP.json').read_bytes()
        with pytest.raises(ValueError,match='FRESH'):m.run(tmp_path,tmp_path/'contract.json',out,'fit-python')
        assert content==(out/'STOP.json').read_bytes()


def test_passive_import():
    code="import sys;sys.path.insert(0,'src');from trottertracks import c_cdf7_onefit as m;assert m.MAX_CALLS==25;assert not any(x in sys.modules for x in ['numpy','scipy','ffsim','jax','qiskit'])"
    subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,check=True)


def test_source_and_input_preflight_is_byte_only(tmp_path,monkeypatch):
    (tmp_path/'input.npz').write_bytes(b'opaque artificial input')
    (tmp_path/'source.py').write_text('# synthetic')
    records={}
    for key in ['result','STOP','SOURCE','baseline_tensor','baseline_optimizer','proposed_tensor','proposed_optimizer']:
        name=key+'.json';m.io.save(tmp_path/name,{});records[key]=name
    saved={p.name:m.io.sha(p) for p in tmp_path.iterdir() if p.is_file() and p.name!='source.py'}
    monkeypatch.setattr(m.io,'P_H4_SHA256',saved['input.npz'])
    m.io.save(tmp_path/'manifest.json',dict(files={'source.py':m.io.sha(tmp_path/'source.py')}))
    c=dict(scope=m.SCOPE,conditions=m.CONDITIONS,initializer=m.INITIALIZER,call_limits=m.LIMITS,
        maximum_logical_units=25,fresh_output=m.FRESH,allowances=m.ALLOWANCES,
        guard_policy=m.molecule.GUARD_POLICY,codec=m.objective.corrected.codec.VERSION,
        optimizer_settings=dict(m.objective.corrected.SETTINGS,max_vecs=7),
        objective_policy='StageA_gamma0_f_over_max_f0_1e-14_no_penalty',subsequent_PR_execution_authorized=False,
        source_manifest='manifest.json',source_manifest_sha256=m.io.sha(tmp_path/'manifest.json'),
        saved_inputs=saved,original_input='input.npz',comparison_records=records,evaluation_packages={})
    m.io.save(tmp_path/'contract.json',c)
    assert m.preflight(tmp_path,tmp_path/'contract.json')==c
    (tmp_path/'input.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='SOURCE_OR_INPUT'):m.preflight(tmp_path,tmp_path/'contract.json')
