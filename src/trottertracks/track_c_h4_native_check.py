"""Reproduce compiler qualification checks; scientific H4 calls need permission.

Synthetic minimal repro and actual registered q1/j0 wrappers are separate.
Measure is retained during transpilation and removed only for state evolution.
"""
from .track_c_h4_price import *
from qiskit import qasm2
from qiskit.converters import dag_to_circuit
from qiskit.quantum_info import Operator,Statevector


def compiler_minimal_repro():
    qc=QuantumCircuit(2);qc.crz(1e-5,0,1);target=Operator(qc).data;trace=[]
    def callback(**kwargs):
        circuit=dag_to_circuit(kwargs['dag']);actual=Operator(circuit).data
        phase=np.trace(target.conj().T@actual);phase/=abs(phase)
        trace.append(dict(pass_name=type(kwargs['pass_']).__name__,
            projective_operator_error=float(np.linalg.norm(target-actual/phase,2)),
            counts=dict(circuit.count_ops())))
    native=transpile(qc,basis_gates=['rz','sx','x','cx'],optimization_level=2,
                     seed_transpiler=17,num_processes=1,callback=callback)
    result=dict(input='CRZ(1e-5), two artificial qubits',trace=trace,
        first_pass_above_1e_minus_10=next((x for x in trace if x['projective_operator_error']>1e-10),None),
        native_counts=dict(native.count_ops()),scope='synthetic debugging, not H4 performance')
    save_json(OUT/'COMPILER_MINIMAL_REPRO_FINAL.json',result)
    return result


def native_wrapper_check(task):
    c=common();r=0 if task['ld']==12 else 1;q=1;rare=False;j=0
    group='CP_shared' if task['pilot']=='C-P' else task['name']
    seed=int.from_bytes(hashlib.sha256(f'{group}:{r}:{q}:{rare}:{j}'.encode()).digest()[:8],'big')
    parts,_,_=trajectory_parts(task,q,r,rare,seed)
    source=wrapper(parts,'X');native=native_compile(source)
    directory=OUT/'native_correctness_measured_profile';directory.mkdir(exist_ok=True)
    (directory/f"{task['name']}.qasm").write_text(qasm2.dumps(native)+'\n')
    reverse=np.array([int(f'{int(x):08b}'[::-1],2) for x in c['sector'].basis_indices])
    initial=np.zeros(512,complex);initial[reverse]=c['state']
    # Qiskit evolve can share input storage while applying a global phase.
    a=Statevector(initial.copy()).evolve(source.remove_final_measurements(inplace=False)).data.copy()
    b=Statevector(initial.copy()).evolve(native.remove_final_measurements(inplace=False)).data.copy()
    def readout(v):
        return float(np.vdot(v[:256],v[:256]).real-np.vdot(v[256:],v[256:]).real)
    phase=np.vdot(a,b);phase/=abs(phase)
    result=dict(task=task,seed=seed,q=q,r=r,axis='X',original_stratum=0,j=j,
        projective_state_error=float(np.linalg.norm(a-b/phase)),
        source_X=readout(a),native_X=readout(b),X_error=abs(readout(a)-readout(b)),
        native_resources=metrics(native),resource_names=list(METRICS),
        measurement_retained_during_compile=True,
        scope='full 512-dimensional joint evolution of an actual registered price sample; no intermediate sector projections')
    save_json(directory/f"{task['name']}.json",result)
    return result


def native_check_tasks():
    return [dict(pilot='C-N',formula='S2',h=.02,ld=3,name='CN_S2_h0.02_ld3'),
        dict(pilot='C-N',formula='Y4',h=.02,ld=12,name='CN_Y4_h0.02_ld12'),
        dict(pilot='C-P',P=False,formula='S2',h=.02,ld=3,name='CP_P0')]
