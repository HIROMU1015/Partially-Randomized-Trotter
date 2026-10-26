"""Synthetic shared-library stress and B2 relaunch integration; no real J/U."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import sys
import threading
import time

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c_pr_aware_df_stage_b2_compat as m


def test_all_three_shared_library_users_exclude_each_other_and_restore(monkeypatch):
    active=0;maximum=0
    def operation():
        nonlocal active,maximum
        active+=1;maximum=max(maximum,active);time.sleep(.001);active-=1;return 7
    monkeypatch.setattr(m.base.b.native,'actual_adjoint',lambda qc:operation())
    original=m.base.b.native.actual_adjoint
    call=SimpleNamespace(stop=threading.Event())
    class Invoke:
        def __call__(self,k,l,fn):return fn()
    shared=m.SharedQiskitCalls(Invoke())
    with shared.adjoint_builds():
        def task(i):
            if i%3==0:return shared('kernel_build',str(i),operation)
            if i%3==1:return shared('compile',str(i),operation)
            return m.base.b.native.actual_adjoint(None)
        with ThreadPoolExecutor(max_workers=12) as pool:assert list(pool.map(task,range(60)))==[7]*60
    assert maximum==1 and m.base.b.native.actual_adjoint is original


def test_real_synthetic_control_build_P0_inverse_are_unchanged():
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import RZZGate
    from qiskit.quantum_info import Operator
    from trottertracks.c1_h4_native_fair import compile_checked
    from trottertracks.pr_dictionary_validation.fixture import CompilerProfile
    shared=m.SharedQiskitCalls(lambda k,l,f:f())
    def task(i):
        def build():
            qc=QuantumCircuit(3);qc.append(RZZGate(.01*(i+1)).control(1),[2,0,1]);qc.p(.03,2);return qc
        raw=shared('kernel_build',str(i),build)
        compiled=shared('compile',str(i),lambda:compile_checked(raw,CompilerProfile()))
        inverse=m.base.b.native.actual_adjoint(raw)
        R,C,A=map(lambda q:Operator(q).data,(raw,compiled,inverse))
        np.testing.assert_allclose(C,R,atol=1e-12);np.testing.assert_allclose(A@R,np.eye(8),atol=1e-12)
    with shared.adjoint_builds():
        with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(task,range(16)))


def test_stop_is_checked_after_shared_lock(tmp_path):
    call=m.base.integrated.Calls(tmp_path,dict(kernel_build=1,compile=1),2)
    shared=m.SharedQiskitCalls(call);call.stop.set()
    with pytest.raises(m.base.integrated.Stopped):shared('kernel_build','waiting',lambda:None)
    assert not call.counts


def test_synthetic_relaunch_preserves_seed_SOURCE_and_previous_evidence(tmp_path,monkeypatch):
    from test_research_c_pr_aware_df_stage_b2 import invented_evidence
    previous=invented_evidence(tmp_path);c=dict(source_manifest='source.json')
    monkeypatch.setattr(m,'preflight',lambda *a:(c,previous))
    monkeypatch.setattr(m.subprocess,'check_output',lambda *a,**k:'ARTIFICIAL_SOURCE\n')
    execute=m.base.execute
    monkeypatch.setattr(m.base,'execute',lambda old,r,o,call:execute(old,r,o,call,samples_per_arm=2))
    before={p:(tmp_path/p).read_bytes() for p in previous['saved_inputs']}
    out=tmp_path/m.FRESH;m.run(tmp_path,'contract','input',out)
    stop=m.base.io.read(out/'STOP.json')
    assert not stop['failed'] and stop['technical_relaunch']==1 and stop['resume']==stop['retry_within_run']==0
    assert stop['cumulative_scientific_units']==429+stop['scientific_units']
    for p,data in before.items():assert (tmp_path/p).read_bytes()==data
    for p in (out/'samples').glob('*_trajectory.json'):
        assert m.base.io.read(p)['SOURCE_sha256']==previous['source_manifest_sha256']
    assert m.base.io.read(out/'result.json')['total_resources'] is None
    with pytest.raises(ValueError,match='FRESH'):m.run(tmp_path,'contract','input',out)
