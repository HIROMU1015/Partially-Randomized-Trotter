#!/usr/bin/env python3
"""Technical diagnosis of the saved original B0 first wrapper; no new cases."""
import argparse,base64,gzip,io,json,subprocess,sys
from pathlib import Path

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/'src'))
    import numpy as np
    from qiskit import qpy,transpile
    from qiskit.quantum_info import Operator
    from qiskit.circuit.equivalence import EquivalenceLibrary
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import UnrollCustomDefinitions
    from qiskit.converters import dag_to_circuit
    from trottertracks.pr_dictionary_validation.fixture import CompilerProfile
    record=json.loads(Path(args.input).read_text())['rows'][-1]
    assert record['identity']=='B3:9/50:0:cosine'
    def load(name):
        encoded=record['diagnostics']['circuits'][name]
        return qpy.load(io.BytesIO(gzip.decompress(base64.b64decode(encoded['data']))))[0]
    raw,old=load('raw'),load('actual');profile=CompilerProfile()
    reference=Operator(raw.remove_final_measurements(inplace=False)).data
    rows=[]
    def report(circuit,label):
        actual=Operator(circuit.remove_final_measurements(inplace=False)).data
        rows.append(dict(label=label,spectral_from_raw=float(np.linalg.norm(actual-reference,ord=2)),
            unitarity=float(np.linalg.norm(actual.conj().T@actual-np.eye(8),ord=2)),
            global_phase=float(circuit.global_phase),operations=dict(circuit.count_ops())))
    report(old,'saved_compiled')
    lowered=PassManager([UnrollCustomDefinitions(EquivalenceLibrary(),list(profile.basis_gates)+['u'])]).run(raw)
    report(lowered,'official_definition_lowering')
    def callback(**kw):
        if kw['pass_'].is_transformation_pass:report(dag_to_circuit(kw['dag']),type(kw['pass_']).__name__)
    compiled=transpile(lowered,basis_gates=list(profile.basis_gates),optimization_level=1,seed_transpiler=17,
        backend=None,target=None,coupling_map=None,initial_layout=None,layout_method=None,routing_method=None,
        translation_method=None,scheduling_method=None,approximation_degree=1.,qubits_initially_zero=False,
        callback=callback)
    report(compiled,'final')
    value=dict(identity=record['identity'],source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        input=str(Path(args.input).resolve()),rows=rows,raw_operations=dict(raw.count_ops()),
        extra_transpile_requests=1,operator_evaluations=len(rows)+1,phase_fit=False,no_cost_measurement=True)
    with Path(args.output).open('x') as f:json.dump(value,f,indent=2);f.write('\n')

if __name__=='__main__':main()
