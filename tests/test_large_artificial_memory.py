"""Artificial generation and native-allocation RSS observer validation."""

import mmap
import multiprocessing as mp
import time

import numpy as np
import pytest
from qiskit.quantum_info import Operator

import benchmark_large_artificial_memory as m


@pytest.fixture(autouse=True)
def no_molecular_inputs(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('Artificial memory test only')
    monkeypatch.setattr(np,'load',forbidden)


@pytest.fixture(scope='module')
def unit():
    return m.prepare_unit(m.compiler_settings())[0]


def test_compiled_template_is_flat_and_control_zero_identity(unit):
    assert unit.num_qubits==3 and unit.num_clbits==0
    assert not set(unit.count_ops())-{'rz','sx','x','cx'}
    u=Operator(unit).data
    assert np.max(abs(u[:4,:4]-np.eye(4)))<1e-10
    assert np.max(abs(u[4:,:4]))<1e-10


def test_materialized_and_streamed_rows_and_canonical_identity_agree(unit):
    a=m.flat_body(unit,3,True);b=m.flat_body(unit,3,False)
    assert len(a.data)==len(b.data)==3*len(unit.data)
    assert a.num_qubits==b.num_qubits==9 and a.num_clbits==b.num_clbits==0
    assert m.instruction_digest(a)==m.instruction_digest(b)
    assert m.streamed_hash(a)==m.compiler_module.canonical_qiskit_circuit_fingerprint(b)
    b.rz(.123,0)
    assert m.instruction_digest(a)!=m.instruction_digest(b)


def test_compile_probe_matches_existing_compiler_and_wrappers(unit):
    settings=m.compiler_settings();body=m.flat_body(unit,1,False)
    for axis in ('cosine','sine'):
        w=m.wrap_controlled_no_prep(body,ancilla_qubit=8,axis=axis,include_measurement=True)
        assert w.num_clbits==1 and w.count_ops()['measure']==1
        direct=m.compile_only(w,settings)
        old=m.compiler_module.transpile_and_measure_cost(w,settings)
        assert m.streamed_hash(direct)==m.streamed_hash(old.transpiled_circuit)
        assert m.measured_metrics(direct)=={k:getattr(old,k) for k in m.measured_metrics(direct)}


def mapped_worker(connection):
    meter=m.StageMeter(connection)
    def mapped():
        area=mmap.mmap(-1,16*1024*1024)
        page=b'\x01'*4096
        for _ in range(4096):area.write(page)
        time.sleep(.08)
        return area
    area=meter.measure('C_memory_probe',mapped)
    area.close()
    meter.measure('short_boundary',lambda:time.sleep(.002))
    connection.send({'event':'result','result':{'status':'PASS_ARTIFICIAL_ONLY'}})
    connection.close()


def test_external_observer_sees_non_tracemalloc_memory():
    context=mp.get_context('spawn');parent,child=context.Pipe()
    process=context.Process(target=mapped_worker,args=(child,));process.start();child.close()
    rows,result,_=m.observe(process,parent);parent.close()
    assert result['status']=='PASS_ARTIFICIAL_ONLY' and process.exitcode==0
    stage=rows[0]
    assert stage['observed_stage_peak_RSS_KiB']-stage['RSS_before_KiB']>8*1024
    assert stage['Python_traced_peak_bytes']<1024*1024
    assert stage['samples']>=2 and stage['maximum_sample_gap_seconds']>0
    for row in rows:
        assert row['observed_stage_wall_seconds']>=row['elapsed_seconds']
        assert row['maximum_sample_gap_seconds']*(row['samples']-1)+1e-6>=row['observed_stage_wall_seconds']
