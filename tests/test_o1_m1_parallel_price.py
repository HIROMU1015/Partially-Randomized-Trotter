"""Invented two-mode circuits only. No saved inputs or molecular execution."""
from dataclasses import asdict
from functools import partial
import importlib
from pathlib import Path
import sys
import time

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import o1_m1_parallel_price as parallel
bridge=parallel._bridge()
adapter=importlib.import_module('trotterlib.o1_m1_matched_split')
rte=importlib.import_module('trotterlib.rte')
from test_o1_m1_matched_split import toy_inputs


def reference_compile(circuit,compiler,*,references,record=False,**kwargs):
    # Generated Pauli Gate names differ between processes. Check the complete
    # operator against the serial circuits, rather than pricing their names.
    from qiskit.quantum_info import Operator
    time.sleep(.1)
    assert circuit.num_clbits==1 and circuit.data[-1].operation.name=='measure'
    assert kwargs['actual_circuit_fingerprint']
    matrix=Operator(circuit.remove_final_measurements(inplace=False)).data
    matches=[i for i,expected in enumerate(references) if np.max(abs(matrix-expected))<1e-12]
    if not matches:
        assert record, 'Parallel operator differs from every serial reference'
        references.append(matrix);matches=[len(references)-1]
    value=100+17*matches[0]
    return rte.CircuitCost(value,value/2,3,2,value+1,value+2,compiler,5,'empirical_compiled_estimate')


def broken_compile(circuit,compiler,**kwargs):
    raise ValueError('invented compiler failure')


def prepared(dictionary):
    inputs=adapter.MatchedAInputs(**asdict(toy_inputs()))
    p=bridge.prepare_matched_resources(adapter.prepare_matched_a(inputs),dictionary=dictionary)
    config,distribution=bridge.configure_law(p,delta_time=.13,r=2,K=2)
    return p,config,distribution


def compiler():
    return rte.CompilerSettings(('rz','sx','x','cx'),None,None,1,None,None,7,'1.3.0')


@pytest.fixture(autouse=True)
def no_saved_inputs(monkeypatch):
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('No saved science input'))


@pytest.mark.parametrize('dictionary',['P','DF'])
def test_spawn_MC_matches_serial_all_metrics_SE_metadata_and_law(dictionary):
    p,c,d=prepared(dictionary)
    before=p.split.inputs.state.copy()
    references=[]
    serial=bridge.MatchedNoPrepCostProvider(compiler(),compile_function=partial(reference_compile,references=references,record=True))
    kwargs=dict(q=2,sample_count=8,seed=37)
    expected=serial.monte_carlo(p,c,d,**kwargs)
    candidate=parallel.ParallelMatchedNoPrepCostProvider(compiler(),max_workers=2,
        compile_function=partial(reference_compile,references=references))
    actual=candidate.monte_carlo(p,c,d,**kwargs)
    assert actual==expected  # all six metrics, both axes, SE, fingerprints, metadata
    assert candidate.last_execution['configured_workers']==2
    assert candidate.last_execution['task_count']==8
    assert len(candidate.last_execution['worker_pids'])>=2
    assert candidate.last_execution['reduction']=='submission order'
    np.testing.assert_array_equal(p.split.inputs.state,before)


def test_duplicate_draws_retained_and_workers_do_not_resample(monkeypatch):
    p,c,d=prepared('P')
    event=next(rte.iter_rte_events(p.tail.components,d))
    def repeated(components,distribution,*,sample_count,seed):
        yield from (event for _ in range(sample_count))
    monkeypatch.setattr(rte,'iter_sample_rte_events',repeated)
    references=[]
    serial=bridge.MatchedNoPrepCostProvider(compiler(),compile_function=partial(reference_compile,references=references,record=True))
    kwargs=dict(q=1,sample_count=4,seed=11)
    expected=serial.monte_carlo(p,c,d,**kwargs)
    candidate=parallel.ParallelMatchedNoPrepCostProvider(compiler(),max_workers=2,
        compile_function=partial(reference_compile,references=references))
    assert candidate.monte_carlo(p,c,d,**kwargs)==expected


def test_parallel_axis_price_matches_actual_serial_compiler_on_toy_gate():
    from qiskit import QuantumCircuit
    circuit=QuantumCircuit(2);circuit.cp(.17,1,0);circuit.p(-.11,1)
    kwargs=dict(context_hash='invented',arm_hash='toy_baseline',q=1,delta_time=.13)
    expected=bridge.MatchedNoPrepCostProvider(compiler()).deterministic(circuit,**kwargs)
    candidate=parallel.ParallelMatchedNoPrepCostProvider(compiler())
    assert candidate.deterministic(circuit,**kwargs)==expected
    assert candidate.last_execution['task_count']==2
    assert candidate.last_execution['configured_workers']==2


@pytest.mark.parametrize('dictionary',['P','DF'])
def test_actual_toy_MC_compiler_matches_all_metrics(dictionary):
    p,_,_=prepared(dictionary)
    c,d=bridge.configure_law(p,delta_time=.13,r=1,K=0)
    kwargs=dict(q=1,sample_count=4,seed=11)
    expected=bridge.MatchedNoPrepCostProvider(compiler()).monte_carlo(p,c,d,**kwargs)
    candidate=parallel.ParallelMatchedNoPrepCostProvider(compiler(),max_workers=2)
    assert candidate.monte_carlo(p,c,d,**kwargs)==expected


@pytest.mark.parametrize('dictionary',['P','DF'])
def test_split_axes_reuse_same_draws_and_order(dictionary):
    p,c,d=prepared(dictionary)
    references=[]
    serial=bridge.MatchedNoPrepCostProvider(compiler(),compile_function=partial(reference_compile,references=references,record=True))
    kwargs=dict(q=1,sample_count=2,seed=19)
    expected=serial.monte_carlo(p,c,d,**kwargs)
    candidate=parallel.ParallelMatchedNoPrepCostProvider(compiler(),max_workers=4,
        compile_function=partial(reference_compile,references=references))
    assert candidate.monte_carlo(p,c,d,**kwargs)==expected
    assert candidate.last_execution['granularity']=='axis'
    assert candidate.last_execution['task_count']==4


@pytest.mark.parametrize('bad',[{'q':0},{'sample_count':1},{'seed':-1}])
def test_invalid_MC_request_fails_before_worker_creation(monkeypatch,bad):
    p,c,d=prepared('P')
    candidate=parallel.ParallelMatchedNoPrepCostProvider(compiler())
    monkeypatch.setattr(candidate,'_map',lambda *a,**k:pytest.fail('Invalid request reached workers'))
    with pytest.raises(ValueError):
        candidate.monte_carlo(p,c,d,**(dict(q=1,sample_count=2,seed=1)|bad))


def test_worker_failure_propagates_without_a_fake_price():
    p,c,d=prepared('P')
    candidate=parallel.ParallelMatchedNoPrepCostProvider(compiler(),max_workers=2,compile_function=broken_compile)
    with pytest.raises(ValueError,match='invented compiler failure'):
        candidate.monte_carlo(p,c,d,q=1,sample_count=2,seed=1)
    assert candidate.last_execution is None


def test_worker_limit_uses_measured_memory_and_cpu_not_a_fixed_small_cap():
    assert parallel.memory_worker_limit(63,64,1_000_000,480_000_000)==63
    assert parallel.memory_worker_limit(63,64,8_000_000,480_000_000)==30
    assert parallel.memory_worker_limit(7,64,1_000_000,480_000_000)==7
    with pytest.raises(RuntimeError,match='headroom'):
        parallel.memory_worker_limit(32,64,8_000_000,10_000_000)


def test_existing_strong_spin_native_circuit_survives_spawn_and_real_compile():
    h=np.kron(np.array([[.2,.03],[.03,-.1]]),np.eye(2))
    factors=np.array([np.kron(np.array([[.4,.2],[.2,-.6]]),np.eye(2)),
                      np.kron(np.array([[-.1,-.21],[-.21,.5]]),np.eye(2))])
    inputs=adapter.MatchedAInputs(.13,h,np.zeros((4,)*4),h,np.array([-.4,.2]),
        factors,np.arange(16)+1j,np.array([3,6,9,12]))
    split=adapter.prepare_matched_a(inputs)
    circuit=bridge.build_strong_df_baseline(split,delta_time=.05,q=1,substeps=1,
        basis_policy='exact_spin_lift_spatial_eigh_v1')
    kwargs=dict(context_hash='invented_spin',arm_hash='strong_toy',q=1,delta_time=.05)
    serial=bridge.MatchedNoPrepCostProvider(compiler()).deterministic(circuit,**kwargs)
    candidate=parallel.ParallelMatchedNoPrepCostProvider(compiler())
    assert candidate.deterministic(circuit,**kwargs)==serial
