"""Two parallel hypothetical-budget units; synthetic fixtures only."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import multiprocessing
import os
from pathlib import Path
import time
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from m5_algebraic_oneq_prototype import compile_artificial, rational_payload, require, POLICY


def fixtures():
    rows={}
    q=QuantumCircuit(1);q.global_phase=.37
    for _ in range(12):q.rz(.125,0);q.rz(.375,0);q.sx(0);q.sx(0);q.x(0)
    rows['exact_dyadic_merge']=q
    q=QuantumCircuit(1);q.rz(.1,0);q.rz(.2,0);rows['rounded_sum']=q
    q=QuantumCircuit(1)
    for _ in range(256):q.rz(2.**-40,0);q.barrier(0)
    rows['retained_small_turns']=q
    q=QuantumCircuit(1);q.global_phase=.37;q.rz(math.pi,0);q.rz(math.pi,0)
    for _ in range(4):q.sx(0)
    rows['two_pi_not_identity']=q
    q=QuantumCircuit(2);q.global_phase=.37
    for _ in range(8):
        q.rz(.125,0);q.rz(.375,0);q.sx(1);q.sx(1);q.cx(0,1)
        q.rz(.1,1);q.rz(.2,1);q.cx(1,0)
    rows['entangled_blocks']=q
    q=QuantumCircuit(1);q.global_phase=.37;q.rz(.2,0);q.rz(-.2,0);q.sx(0);q.sx(0);q.x(0)
    rows['exact_cancellation']=q
    q=QuantumCircuit(1)
    for _ in range(256):q.rz(.1,0);q.rz(.2,0);q.barrier(0)
    rows['cumulative_rounding']=q
    return rows


def without_barriers(circuit):
    q=QuantumCircuit(circuit.num_qubits);q.global_phase=circuit.global_phase
    for item in circuit.data:
        if item.operation.name!='barrier':q.append(item.operation,[circuit.find_bit(b).index for b in item.qubits])
    return q


def worker(budget):
    started=time.monotonic();rows=[]
    for name,q in fixtures().items():
        out,records,ledger=compile_artificial(q,budget_radians=budget,artificial=True)
        raw=float(np.linalg.norm(Operator(out).data-Operator(q).data))
        controlled=float(np.linalg.norm(Operator(without_barriers(out).to_gate().control()).data-Operator(without_barriers(q).to_gate().control()).data))
        require(math.isfinite(raw) and math.isfinite(controlled) and raw<1e-12 and controlled<1e-12,'ARTIFICIAL_RAW_OPERATOR_DISAGREEMENT:'+name)
        require(float(out.global_phase).hex()==float(q.global_phase).hex(),'WHOLE_PIPELINE_PHASE_BITS')
        rows.append(dict(case=name,hypothetical_angle_budget_radians=budget,
            input_counts=dict(q.count_ops()),output_counts=dict(out.count_ops()),
            raw_operator_Frobenius=raw,raw_controlled_operator_Frobenius=controlled,
            exact_angle_rounding_charge=rational_payload(ledger.spent),
            ideal_operator_norm_bound_from_RZ_rounding=rational_payload(ledger.spent/2),
            scope_of_bound='IDEAL_MATHEMATICAL_GATES_ONLY_EXCLUDES_SIMULATOR_AND_OTHER_PASSES',
            global_phase_bits_unchanged=True,pass_records=records))
    return dict(worker_pid=os.getpid(),rows=rows,wall_seconds=time.monotonic()-started)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-artificial',action='store_true');p.add_argument('--output',type=Path);a=p.parse_args()
    if not a.run_artificial:
        print(json.dumps(dict(status='ARTIFICIAL_SCOPE_ONLY',policy=POLICY,H6_execution_authorized=False)));return
    require(a.output is not None and not a.output.exists(),'NEW_EXCLUSIVE_ARTIFICIAL_OUTPUT')
    require(__import__('qiskit').__version__=='1.3.0','PINNED_QISKIT')
    started=time.monotonic()
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
        units=list(pool.map(worker,[0.0,1e-11]))
    result=dict(status='ARTIFICIAL_PROTOTYPE_VALIDATED_H6_UNTESTED',policy=POLICY,units=units,
        fixture_cases=7,synthetic_compiles=14,productive_work_units=2,workers_joined=True,
        wall_seconds=time.monotonic()-started,H6_compile=0,H6_replay=0,molecular_QPY_NPZ_loads=0,
        valid_grants_issued=0,scientific_budget_adopted=False,shared_runtime_modified=False,
        phase_alignment=False,postcompile_correction=False,cost_accounting=0,mandatory_STOP=True)
    with a.output.open('x') as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status=result['status'],cases=14,wall_seconds=result['wall_seconds'])))


if __name__=='__main__':main()
