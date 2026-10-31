"""Artificial paired pipeline/STOP and worker integration, no molecular data."""
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import threading
from unittest.mock import patch

import numpy as np
import pytest
import scipy.optimize

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/"src"))
from trottertracks import c_common_convergence_pilot as m


@pytest.mark.parametrize("fault",[None,"unqualified","solver","budget","initial","representation","worker"])
def test_artificial_paired_25_call_route(tmp_path,monkeypatch,fault):
    out=tmp_path/"run01"
    c=dict(fresh_output="run01",source_manifest="SOURCE.json",initial_sector="initial.json",comparators={})
    m.io.save(tmp_path/"SOURCE.json",dict(files={}))
    m.io.save(tmp_path/"initial.json",dict(sector_error=.001 if fault=="initial" else 1e-6,numerical_guard=1e-9))
    monkeypatch.setattr(m,"preflight",lambda *args,**kwargs:c)
    a=dict(J=np.ones((8,4,4)),U=np.repeat(np.eye(4)[None],8,axis=0),x0=np.zeros(128))
    p=dict(h=np.eye(4),g=np.zeros((4,)*4),constant=np.array(.5))
    monkeypatch.setattr(m.previous,"load",lambda *args:(p,a,{}))
    monkeypatch.setattr(m.old,"resource_plan",lambda **kwargs:dict(workers=2,available_memory_bytes=10**10))
    monkeypatch.setattr(m.molecule,"SectorCAR",lambda *args:SimpleNamespace(physical=lambda *a:np.zeros((2,2))))
    def assessment(*args):
        return dict(sector_error=.001 if fault=="budget" else 1e-7,numerical_guard=1e-9),[]
    monkeypatch.setattr(m.molecule,"rank_assessment",assessment)
    def representation(*args):
        if fault=="representation":raise ValueError("ARTIFICIAL_REPRESENTATION_FAIL")
        return dict(status="PASS")
    monkeypatch.setattr(m.molecule,"representation_gate",representation)
    def workers(root,path,output,c,fit_python,number,calls):
        if fault=="worker":raise ValueError("ARTIFICIAL_WORKER_FAIL")
        for role in m.ROLES:
            folder=out/role;folder.mkdir()
            m.existing.arrays(folder/"final.npz",J=a["J"],U=a["U"],x=np.zeros(128),jac=np.zeros(128))
            qualified=not (fault in ("unqualified","solver") and role=="rc2")
            m.io.save(folder/"final_diagnostic.json",dict(components=dict(exact_tail_mass_Ha=2. if role=="rc2" else 1.),
                assessment=dict(qualified=qualified)))
            m.io.save(folder/"optimizer.json",dict(success=fault!="solver"))
            m.io.save(folder/"tensor.json",dict(frobenius=0.))
            m.io.save(folder/"STOP.json",dict(failed=False,call_counts=m.FIT))
    monkeypatch.setattr(m,"launch_workers",workers)
    if fault in ("initial","representation","worker"):
        with pytest.raises(ValueError):m.run(tmp_path,tmp_path/"contract.json",out,"fit-python")
        assert m.io.read(out/"result.json")["status"]=="TECHNICAL_FAILURE"
        assert m.io.read(out/"STOP.json")["failed"]
    else:
        m.run(tmp_path,tmp_path/"contract.json",out,"fit-python")
        result=m.io.read(out/"result.json");stop=m.io.read(out/"STOP.json")
        assert stop["scientific_units"]==25 and stop["call_counts"]==m.LIMITS
        expected="DF_BUDGET_NOT_MET" if fault=="budget" else "ONE_OR_BOTH_UNQUALIFIED" if fault else "BOTH_CONVERGENCE_CRITERIA_MET"
        assert result["status"]==expected
        assert (result["qualified_tail_comparison"] is None)==bool(fault)
        assert result["total_cost"] is None and not stop["failed"]
    with pytest.raises(ValueError,match="FRESH"):m.run(tmp_path,tmp_path/"contract.json",out,"fit-python")
    assert m.io.read(out/"STOP.json")["retry"]==0


@pytest.mark.parametrize("role",["rc2","pr"])
def test_artificial_rank8_worker_real_ffsim_with_mock_optimizer(tmp_path,monkeypatch,role):
    rng=np.random.default_rng(789);a=rng.normal(size=(8,4,4));J=a+a.transpose(0,2,1)
    J*=np.sqrt(m.CONDITIONS["l2_scale"]/np.sum(J*J))
    U=np.repeat(np.eye(4)[None],8,axis=0);U[0,:,0]*=-1;U[1,:2,:2]=-np.eye(2)
    J,U,x,_,_=m.previous.objective.pr.corrected.codec.roundtrip(J,U)
    g=m.previous.objective.pr.corrected.codec.reconstruct(J,U)
    g[0,0,0,0]+=np.sqrt(2*m.CONDITIONS["loss_scale"])
    info=dict(spec={k:m.CONDITIONS[k] for k in ("rank","norb","ld","loss_scale","mass_scale","epsilon")})
    packet=dict(g=g,h=np.eye(4),constant=np.array(.5));initial=dict(J=J,U=U,x0=x)
    monkeypatch.setattr(m.previous,"load",lambda *args:(packet,initial,info))
    m.io.save(tmp_path/"initial_gate.json",dict(status="PASS"))
    def optimizer(fun,x0,**kwargs):
        assert kwargs["options"]==m.policy.OPTIONS
        np.testing.assert_array_equal(x0,x)
        value,grad=fun(x0)
        return SimpleNamespace(x=x0.copy(),jac=grad,fun=value,success=True,status=0,
            message="MOCK_REL_REDUCTION_NO_ITERATIONS",nit=0,nfev=1,njev=1)
    with patch.object(scipy.optimize,"minimize",side_effect=optimizer):
        m.fit_worker(tmp_path,tmp_path/role,{},role)
    stop=m.io.read(tmp_path/role/"STOP.json")
    assert not stop["failed"] and stop["scientific_units"]==7
    assert not m.io.read(tmp_path/role/"final_diagnostic.json")["assessment"]["qualified"]
    assert (tmp_path/role/"optimizer_raw.npz").exists()


def test_passive_import_has_no_numeric_or_quantum_modules():
    code="import sys;sys.path.insert(0,'src');import trottertracks.c_common_convergence_pilot;assert not {'numpy','jax','ffsim','qiskit'}&set(sys.modules)"
    subprocess.run([sys.executable,"-B","-c",code],cwd=ROOT,check=True)


def test_byte_preflight_rejects_changed_policy_and_source(tmp_path,monkeypatch):
    m.io.save(tmp_path/"source.json",dict(files={}))
    (tmp_path/"saved.bin").write_bytes(b"invented metadata; no numerical decode")
    c=dict(scope=m.SCOPE,conditions=m.CONDITIONS,options=m.policy.OPTIONS,
        criteria=m.policy.CRITERIA,call_limits=m.LIMITS,maximum_logical_units=25,
        fresh_output=m.FAMILY+"/run01",source_manifest="source.json",
        source_manifest_sha256=m.io.sha(tmp_path/"source.json"),
        saved_inputs={"saved.bin":m.io.sha(tmp_path/"saved.bin")},
        input_snapshot="saved.bin",initial_snapshot="saved.bin",initial_scales="saved.bin",
        initial_sector="saved.bin",comparators={},evaluation_packages={})
    m.io.save(tmp_path/"contract.json",c)
    m.preflight(tmp_path,tmp_path/"contract.json")
    changed={**c,"options":{**m.policy.OPTIONS,"gtol":1e-3}}
    m.io.save(tmp_path/"changed.json",changed)
    with pytest.raises(ValueError,match="FIXED"):m.preflight(tmp_path,tmp_path/"changed.json")
    (tmp_path/"saved.bin").write_bytes(b"changed")
    with pytest.raises(ValueError,match="SOURCE_OR_INPUT"):m.preflight(tmp_path,tmp_path/"contract.json")


def test_worker_launch_failure_stops_dependents_and_collects_started_tasks(tmp_path,monkeypatch):
    calls=SimpleNamespace(stop=threading.Event());started=[]
    def worker(args,**kwargs):
        role=args[args.index("--fit-worker")+1];started.append(role)
        if role=="rc2":raise subprocess.CalledProcessError(3,args)
    monkeypatch.setattr(m.subprocess,"run",worker)
    with pytest.raises(RuntimeError,match="TECHNICAL_WORKER_FAILURE"):
        m.launch_workers(tmp_path,tmp_path/"c.json",tmp_path,dict(entry="fake.py"),"fake-python",2,calls)
    assert calls.stop.is_set() and "rc2" in started and len(started)<=2
