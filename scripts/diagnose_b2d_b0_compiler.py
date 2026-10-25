#!/usr/bin/env python3
"""Technical diagnosis of the saved original B0 first wrapper; no new cases."""
import argparse,base64,gzip,io,json,subprocess,sys
from pathlib import Path

def balanced_operator(circuit):
    """Independent little-endian embedding and balanced complex128 products.

    Primitive gate matrices reuse Qiskit; the full-register accumulation does
    not use Operator(circuit). No phase fit, projection or renormalization.
    """
    import numpy as np
    from qiskit.quantum_info import Operator
    unitary=circuit.remove_final_measurements(inplace=False)
    n=unitary.num_qubits
    if n>3:raise ValueError('Only the fixed B0 small register is supported')
    dimension=1<<n;cache={};factors=[]
    for item in unitary.data:
        op=item.operation
        if op.name=='barrier':continue
        if item.clbits:raise ValueError('Unexpected nonunitary instruction')
        qs=tuple(unitary.find_bit(q).index for q in item.qubits)
        key=(op.name,tuple(map(float,op.params)),qs)
        if key not in cache:
            local=np.asarray(Operator(op).data,dtype=np.complex128)
            matrix=np.zeros((dimension,dimension),dtype=np.complex128)
            mask=sum(1<<q for q in qs)
            for col in range(dimension):
                local_col=sum(((col>>q)&1)<<j for j,q in enumerate(qs))
                for local_row in range(1<<len(qs)):
                    row=(col&~mask)|sum(((local_row>>j)&1)<<q for j,q in enumerate(qs))
                    matrix[row,col]=local[local_row,local_col]
            cache[key]=matrix
        factors.append(cache[key])
    if factors:
        level=np.stack(factors)
        while len(level)>1:
            pairs=np.matmul(level[1::2],level[:len(level)//2*2:2])
            level=np.concatenate((pairs,level[-1:])) if len(level)%2 else pairs
        value=level[0]
    else:value=np.eye(dimension,dtype=np.complex128)
    return np.exp(1j*float(unitary.global_phase))*value

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--stable-phase',action='store_true')
    parser.add_argument('--final-only',action='store_true')
    parser.add_argument('--independent-operator',action='store_true')
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
    assert record['identity'] in ('B3:9/50:0:cosine','D1:9/50:0:cosine')
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
    if args.stable_phase:
        from trottertracks.pr_dictionary_validation.gates import stable_official_lowering
        lowered=stable_official_lowering(raw,list(profile.basis_gates)+['u'])
    else:lowered=PassManager([UnrollCustomDefinitions(EquivalenceLibrary(),list(profile.basis_gates)+['u'])]).run(raw)
    report(lowered,'official_definition_lowering')
    def callback(**kw):
        if kw['pass_'].is_transformation_pass:report(dag_to_circuit(kw['dag']),type(kw['pass_']).__name__)
    if args.stable_phase and args.final_only:
        from trottertracks.pr_dictionary_validation.gates import stable_native_transpile
        compiled=stable_native_transpile(lowered,profile)
    else:compiled=transpile(lowered,basis_gates=list(profile.basis_gates),optimization_level=1,seed_transpiler=17,
        backend=None,target=None,coupling_map=None,initial_layout=None,layout_method=None,routing_method=None,
        translation_method=None,scheduling_method=None,approximation_degree=1.,qubits_initially_zero=False,
        callback=None if args.final_only else callback)
    report(compiled,'final')
    value=dict(identity=record['identity'],source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        input=str(Path(args.input).resolve()),rows=rows,raw_operations=dict(raw.count_ops()),
        extra_transpile_requests=1,operator_evaluations=len(rows)+1,phase_fit=False,no_cost_measurement=True)
    if args.independent_operator:
        independent=np.array([[complex(z['real'],z['imag']) for z in row] for row in record['diagnostics']['reference']],dtype=np.complex128)
        raw_balanced=balanced_operator(raw);compiled_balanced=balanced_operator(compiled)
        compiled_sequential=Operator(compiled.remove_final_measurements(inplace=False)).data
        def norm(x):return float(np.linalg.norm(x,ord=2))
        value['independent_operator']=dict(precision='complex128',
            raw_sequential_reference_error=norm(reference-independent),
            raw_balanced_reference_error=norm(raw_balanced-independent),
            compiled_sequential_reference_error=norm(compiled_sequential-independent),
            compiled_balanced_reference_error=norm(compiled_balanced-independent),
            compiled_accumulation_difference=norm(compiled_sequential-compiled_balanced),
            compiled_balanced_raw_error=norm(compiled_balanced-raw_balanced),
            relative_diagonal=[dict(real=float(z.real),imag=float(z.imag)) for z in np.diag(compiled_balanced@raw_balanced.conj().T)])
        def save(qc):
            stream=io.BytesIO();qpy.dump(qc,stream)
            return dict(encoding='gzip+base64+qpy',data=base64.b64encode(gzip.compress(stream.getvalue(),mtime=0)).decode())
        value['circuits']=dict(raw=save(raw),compiled=save(compiled))
        value['independent_operator_evaluations']=2
        value['operator_evaluations']+=1
    with Path(args.output).open('x') as f:json.dump(value,f,indent=2);f.write('\n')

if __name__=='__main__':main()
