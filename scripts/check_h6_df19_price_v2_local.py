"""Limited Phase C diagnostics: actual Gaussian pair, not a full wrapper."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
from qiskit import QuantumCircuit,transpile
from qiskit.quantum_info import Operator
from run_h6_df19_price_v2_pilot import context,PUBLIC,Accounting,write


def main():
    ctx=context();ops=ctx.preps[7].deterministic_blocks[0].runtime_basis_operations
    index=next(i for i,(g,w) in enumerate(ops) if g.name=='xx_plus_yy')
    gate,wires=ops[index];following=ops[index+1:index+3]
    results=[]
    for mode in ('PAIR','PAIR_WITH_PHASES','INVERSE_PAIR','PAIR_CX_INTERFACE'):
        qc=QuantumCircuit(3 if mode=='PAIR_CX_INTERFACE' else 2)
        qc.append(gate,[0,1]);a=Accounting();emitted=[]
        a._native=lambda name,qs,angle=None,role='other':emitted.append((name,tuple(qs),angle))
        a._gate('xx_plus_yy',(0,1),tuple(float(x) for x in gate.params))
        if mode=='PAIR_WITH_PHASES':
            for g,w in following:
                q=wires.index(w[0]);qc.append(g,[q]);a._gate(g.name,(q,),tuple(float(x) for x in g.params))
        elif mode=='INVERSE_PAIR':
            qc.append(gate.inverse(),[0,1]);a._gate('xx_plus_yy',(0,1),(-float(gate.params[0]),float(gate.params[1])))
        elif mode=='PAIR_CX_INTERFACE':qc.cx(2,1);a._gate('cx',(2,1))
        expanded=QuantumCircuit(qc.num_qubits);expanded.global_phase=a.global_phase
        for name,qs,angle in emitted:
            fn=getattr(expanded,name)
            if name=='rz':fn(angle,qs[0])
            else:fn(*qs)
        compiled=transpile(qc,basis_gates=['rz','sx','x','cx'],optimization_level=1,seed_transpiler=17,num_processes=1)
        errors=dict(analytic_expansion=float(np.linalg.norm(Operator(expanded).data-Operator(qc).data)),compiler=float(np.linalg.norm(Operator(compiled).data-Operator(qc).data)))
        overlap=complex(np.trace(Operator(qc).data.conj().T@Operator(compiled).data)/2**qc.num_qubits)
        errors['compiler_phase_only_residual']=float(np.linalg.norm(Operator(compiled).data-overlap*Operator(qc).data))
        if errors['analytic_expansion']>=1e-12 or errors['compiler_phase_only_residual']>=1e-12:raise ValueError(errors)
        counter=Accounting()
        counter._gate('xx_plus_yy',(0,1),tuple(float(x) for x in gate.params))
        if mode=='PAIR_WITH_PHASES':
            for g,w in following:counter._gate(g.name,(wires.index(w[0]),),tuple(float(x) for x in g.params))
        elif mode=='INVERSE_PAIR':counter._gate('xx_plus_yy',(0,1),(-float(gate.params[0]),float(gate.params[1])))
        elif mode=='PAIR_CX_INTERFACE':counter._gate('cx',(2,1))
        pred=counter.finish()
        results.append(dict(case=mode,actual_pair_parameters=[float(x) for x in gate.params],structural_rz=pred['structural_rz'],expanded_rz=pred['expanded_rz_after_basis_cancellation'],compiled_rz=compiled.count_ops().get('rz',0),compiled_cx=compiled.count_ops().get('cx',0),matrix_errors_no_phase_alignment=errors,compiler_relative_global_phase=dict(real=overlap.real,imag=overlap.imag),raw_compiler_matrix_match=errors['compiler']<1e-12))
    write(PUBLIC/'LOCAL_DIAGNOSTICS.json',dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Four local 2/3-qubit gate fragments; no full-wrapper acquisition or new RTE event',rows=results,rule_change=False,full_wrapper_error_attribution=False,global_phase_note='Raw matrix discrepancies are retained; phase-only residual is a diagnostic, not a raw-vector PASS or compiler modification.',prior_boundary_method='No direct H6 m5 signed-piece coefficients; cannot transfer short-step IID two/three-event coefficients'))
    print(json.dumps(results),flush=True)


if __name__=='__main__':main()
