"""Compare new exact action path, then reuse isolated saved-H8 product check."""
import argparse
from functools import partial
import time
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import XXPlusYYGate,RZZGate
from qiskit_aer import AerSimulator
import h8_full_h_pure_rte as s
import h8_full_h_price_parallel as parallel
import h8_full_h_parallel_aer as fast


def benchmark():
    s.install()
    import h46_full_h_t20_aer as old
    qc=QuantumCircuit(17)
    for k in range(80):
        for i in range(15):qc.append(XXPlusYYGate(.031*(i+1)+.0001*(k%4),.17),[i,i+1])
        for i in range(0,16,2):
            qc.crz(.007*(k%4+1),16,i)
            qc.append(RZZGate(.009*(k%4+1)).control(),[16,i,i+1])
        for i in reversed(range(15)):qc.append(XXPlusYYGate(-.031*(i+1)-.0001*(k%4),.17),[i,i+1])
    qc.h(16);rng=np.random.default_rng(101)
    state=rng.normal(size=2**17)+1j*rng.normal(size=2**17);state/=np.linalg.norm(state)
    start=time.monotonic();expected=old.simulate_statevector(qc,state)
    baseline=time.monotonic()-start;rows=[]
    for threads in (1,16,32):
        fast.configure(threads);start=time.monotonic();actual=fast.simulate_statevector(qc,state)
        elapsed=time.monotonic()-start;error=float(np.linalg.norm(actual-expected))
        s.require(error<1e-10,'ARTIFICIAL_OLD_VS_PARALLEL_ACTION')
        rows.append(dict(threads=threads,seconds=elapsed,speedup=baseline/elapsed,vector_error=error))
    record=dict(status='PASS',artificial=True,scientific_samples=0,qubits=17,
        native_instructions=len(qc.data),old_definition_expansion_one_thread_seconds=baseline,
        results=rows,full_H8_speedup_guaranteed=False,priced_compiler_unchanged=True,
        technical_findings=['Writable C-order local matrices required by installed Aer conversion',
            'Existing content keys are call-local; no cross-call identity memo reuse'])
    s.save(s.PUBLIC/'performance_check_parallel.json',record);print(record,flush=True)


def saved_product_check():
    parallel.verify_old_sources()
    parent=s.PUBLIC;s.PUBLIC=parent/'parallel_validation';s.PRIVATE=parent/'PRIVATE/parallel_validation'
    s.require(not s.PUBLIC.exists() and not s.PRIVATE.exists(),'FRESH_PARALLEL_VALIDATION')
    s.PUBLIC.mkdir();(s.PUBLIC/'selection.json').write_bytes((parent/'selection.json').read_bytes())
    s.install=parallel.install
    threads,observation=parallel.choose_threads()
    fast.configure(threads,s.PRIVATE/'actions')
    import check_h8_full_h_product_compile as reused
    reused.main()
    result=s.read(s.PUBLIC/'conditional_product_compile_validation.json')
    result.update(action_threads=threads,resource_observation=observation,
        action_verifier='EXACT_LOCAL_OPERATORS_AER_PARALLEL',original_phase_helper_file_unchanged=True,
        original_source_unchanged=True,new_price_samples=0)
    s.save(s.PUBLIC/'parallel_validation.json',result)
    print('SAVED_H8_PARALLEL_VALIDATION_PASS',flush=True)



def spawn_check():
    s.install()
    import h8_compiler_phase_bookkeeping as phase
    qc=QuantumCircuit(17);qc.h(16)
    for i in range(6):qc.append(XXPlusYYGate(.03*(i+1),0),[i,i+1])
    state=np.zeros(2**17,complex);state[0]=1
    fast.configure(32);fast.simulate_statevector(qc,state)
    phase.ProcessPoolExecutor=parallel.spawned_phase_pool
    proof=phase.certify(qc,{},workers=4)
    s.save(s.PUBLIC/'parallel_spawn_check.json',dict(status='PASS',artificial=True,
        scientific_samples=0,start_method='spawn',after_Aer_threads=32,
        certified_unique_gates=len(proof['records']),maximum_operator_error=proof['maximum_error']))
    print('SPAWN_AFTER_MULTITHREADED_AER_PASS',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['benchmark','validate','spawn']);args=p.parse_args()
    if args.command=='benchmark':benchmark()
    elif args.command=='validate':saved_product_check()
    else:spawn_check()
