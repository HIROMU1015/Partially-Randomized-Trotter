"""Artificial compiler contention and P0 equivalence only; no H4 input."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import threading
import time
from unittest.mock import patch

import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
from trottertracks import c1_stage3_cost_compat as fix
from test_research_c_c1_stage3_cost_only import proposal,ArtificialBackend


def test_mutex_blocks_compile_overlap_but_not_independent_work():
    gate=fix.CompileSerialCalls(lambda kind,label,fn:fn())
    lock=threading.Lock();active=0;maximum=0;seen=[]
    def job(i):
        def compile_one():
            nonlocal active,maximum
            with lock:active+=1;maximum=max(maximum,active)
            time.sleep(.005)
            with lock:active-=1;seen.append(i)
            return i
        return gate('compile',str(i),compile_one)
    with ThreadPoolExecutor(max_workers=8) as pool:assert list(pool.map(job,range(8)))==list(range(8))
    assert maximum==1 and sorted(seen)==list(range(8)) and gate.completed==8
    rendezvous=threading.Barrier(2)
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(gate,'metrics',str(i),lambda:rendezvous.wait(timeout=2)) for i in range(2)]
        assert sorted(f.result() for f in jobs)==[0,1]


def test_failure_stops_waiting_compiles_before_attempt_and_keeps_budget(tmp_path):
    from trottertracks.c1_stage3_integrated import Calls,Stopped
    calls=Calls(tmp_path,dict(compile=8),8);guard=fix.CompileSerialCalls(calls)
    entered=threading.Event();release=threading.Event()
    def fail():entered.set();release.wait(timeout=2);raise RuntimeError('ARTIFICIAL_COMPILER_ERROR')
    with ThreadPoolExecutor(max_workers=8) as pool:
        first=pool.submit(guard,'compile','first',fail);assert entered.wait(timeout=2)
        waiting=[pool.submit(guard,'compile','waiting'+str(i),lambda:None) for i in range(7)]
        release.set()
        with pytest.raises(RuntimeError,match='ARTIFICIAL_COMPILER_ERROR'):first.result()
        for future in waiting:
            with pytest.raises(Stopped):future.result()
    assert calls.counts==dict(compile=1) and guard.completed==0


def artificial_circuit(i):
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import XXPlusYYGate
    q=QuantumCircuit(3)
    for j in range(8):
        a=(j+i)%3;b=(a+1)%3
        q.ry((i+1)*(j+1)/173,a);q.rz(-(j+2)/139,b)
        q.cp((i+j+1)/101,a,b);q.append(XXPlusYYGate((j+1)/97,(i+1)/53),[a,b])
    q.p(-(i+1)/19,2)
    return q


def test_real_P0_parallel_requests_match_original_serial_profile_and_phase(tmp_path):
    import numpy as np
    from qiskit.quantum_info import Operator
    from trottertracks.pr_dictionary_validation.gates import compiled_wrapper
    from trottertracks.pr_dictionary_validation.fixture import CompilerProfile
    from trottertracks.pr_dictionary_validation.f2_execution import circuit_metrics
    circuits=[artificial_circuit(i) for i in range(8)];profile=CompilerProfile()
    serial=[compiled_wrapper(q,profile) for q in circuits]
    guard=fix.CompileSerialCalls(lambda kind,label,fn:fn())
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(guard,'compile',str(i),lambda q=q:compiled_wrapper(q,profile)) for i,q in enumerate(circuits)]
        guarded=[f.result() for f in futures]
    for q,a,b in zip(circuits,serial,guarded):
        assert circuit_metrics(a)==circuit_metrics(b)
        assert np.max(abs(Operator(a).data-Operator(b).data))<1e-12
        assert np.max(abs(Operator(q).data-Operator(b).data))<1e-10


def test_real_old_cost_backend_receives_mutex_and_preserves_other_callbacks(tmp_path):
    from trottertracks.c1_stage3_cost_only import CostBackend
    original=lambda kind,label,fn:fn()
    backend=fix.CompatibleBackend(dict(root=tmp_path,contract=dict(proposal=proposal())),tmp_path)
    def fake_prepare(self,call):self.call=call
    with patch.object(CostBackend,'prepare',fake_prepare):backend.prepare(original)
    assert isinstance(backend.call,fix.CompileSerialCalls)
    assert backend.call('model_prepare','A',lambda:'same')=='same'
    assert backend.call('compile','A',lambda:'P0')=='P0'


def test_artificial_full_scope_and_fresh_execute_provenance(tmp_path):
    v=dict(root=tmp_path,contract=dict(proposal=proposal()),source=dict(files={}),runtime=dict(artificial=True))
    backend=ArtificialBackend();backend.compile_calls=fix.CompileSerialCalls(lambda k,l,f:f())
    output=tmp_path/fix.FAMILY/'ARTIFICIAL'
    with patch.object(fix,'preflight',return_value=v),patch.object(fix,'CompatibleBackend',return_value=backend),\
            patch.dict(os.environ,os.environ.copy()),patch.object(sys,'pycache_prefix',sys.pycache_prefix):
        r=fix.execute(tmp_path,tmp_path/'contract',tmp_path/'input',output)
        assert r['scientific_units']==936 and r['call_counts']==fix.base.call_limits()
        assert r['CA_achieved'] is None and not r['guaranteed_total_resource_superiority']
        assert json.loads((output/'compatibility.json').read_text())['P0_compile_only_serialized']
        with pytest.raises(ValueError,match='FRESH_COST_COMPAT_OUTPUT'):fix.execute(tmp_path,tmp_path/'contract',tmp_path/'input',output)


def test_preflight_revision_keeps_old_scientific_conditions_and_inputs(tmp_path):
    old=dict(proposal=proposal(),candidate_ids=[x.id for x in fix.base.candidates()],
        call_limits=fix.base.call_limits(),input_sha256='ARTIFICIAL',saved_inputs={})
    path=tmp_path/'previous.json';path.write_text(json.dumps(old))
    c=dict(**old,compatibility_version=fix.VERSION,entry='scripts/run_research_c_c1_stage3_cost_compat.py',
        previous_contract='previous.json',previous_contract_sha256=fix.base.io.sha(path))
    with patch.object(fix.base,'preflight',return_value=dict(root=tmp_path,contract=c)):
        fix.preflight(tmp_path,'contract','input')
        c['input_sha256']='CHANGED'
        with pytest.raises(ValueError,match='CHANGED_COMPAT_SCIENTIFIC_CONDITIONS'):fix.preflight(tmp_path,'contract','input')
