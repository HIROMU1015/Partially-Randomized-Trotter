"""Fixed small synthetic evidence only; no molecular data file arguments."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tests'))


def evidence():
    import numpy as np
    import qiskit
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit import Gate, Parameter
    from qiskit.circuit.equivalence import EquivalenceLibrary
    from qiskit.quantum_info import Operator
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import BasisTranslator
    from test_m5_phase_repair_prototype import toy, bytes_metadata
    from m5_phase_repair_prototype import (
        ArtificialPassRecorder, lower_native_once, lower_p_once,
    )
    settings = dict(basis_gates=['rz','sx','x','cx'], optimization_level=1,
                    seed_transpiler=17, num_processes=1)
    original = toy()
    p_only, p_receipt = lower_p_once(original, artificial=True)
    native, native_receipt = lower_native_once(original, artificial=True)
    original_bytes = bytes_metadata(original)
    reference = Operator(original).data
    comparisons = []
    recorder = ArtificialPassRecorder(artificial=True)
    recorder.boundary(native, 'INPUT')
    for name, circuit in [('ORIGINAL',original), ('P_ONLY',p_only), ('EXPLICIT_NATIVE',native)]:
        compiled = transpile(circuit, callback=recorder if name=='EXPLICIT_NATIVE' else None,
                             **settings)
        error = float(np.linalg.norm(reference-Operator(compiled).data))
        comparisons.append(dict(case=name, raw_whole_operator_error=error,
                                artificial_raw_atol=1e-12, raw_check_PASS=error<1e-12,
                                layout=str(compiled.layout), counts=dict(compiled.count_ops())))
        if name=='EXPLICIT_NATIVE':
            recorder.boundary(compiled, 'OUTPUT')
    assert bytes_metadata(original)==original_bytes
    probes = []
    for mode in ('constant_numeric', 'parameter_expression', 'zero'):
        theta = Parameter('theta')
        definition = QuantumCircuit(1)
        definition.global_phase = .123 if mode=='constant_numeric' else theta/2 if mode=='parameter_expression' else 0.
        definition.rz(theta, 0)
        library = EquivalenceLibrary()  # Never change the shared Session library.
        library.add_equivalence(Gate('phase_probe', 1, [theta]), definition)
        circuit = QuantumCircuit(3); circuit.append(Gate('phase_probe', 1, [.2]), [0])
        compiled = PassManager(BasisTranslator(library, ['rz'])).run(circuit)
        ideal = QuantumCircuit(3)
        ideal.compose(definition.assign_parameters({theta:.2}), [0], inplace=True)
        probes.append(dict(case=mode, expected_phase=float(ideal.global_phase),
                           output_phase=float(compiled.global_phase),
                           raw_error=float(np.linalg.norm(Operator(ideal).data-Operator(compiled).data))))
    assert comparisons[0]['raw_check_PASS'] is False
    assert comparisons[1]['raw_check_PASS'] is False
    assert comparisons[2]['raw_check_PASS'] is True
    assert probes[0]['raw_error']>.1 and all(p['raw_error']<1e-12 for p in probes[1:])
    return dict(status='ARTIFICIAL_ONLY_NOT_A_MOLECULAR_REPAIR_PASS',
                SOURCE=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
                qiskit=qiskit.__version__, compiler=settings, comparisons=comparisons,
                explicit_before_compile_error=float(np.linalg.norm(reference-Operator(native).data)),
                explicit_controlled_error=float(np.linalg.norm(Operator(original.to_gate().control()).data-
                                                               Operator(native.to_gate().control()).data)),
                minimal_phase_probes=probes, P_only_receipt=p_receipt,
                native_receipt=native_receipt, callback_trace=recorder.rows,
                real_QPY_or_NPZ_reads=0, molecular_compile=0, molecular_replay=0,
                valid_grants_issued=0, production_installed=False,
                postcompile_phase_correction=False, old_STOP_changed=False,
                mandatory_STOP=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    path = Path(args.output)
    if path.exists():
        raise SystemExit('NEW_ARTIFICIAL_OUTPUT_ONLY')
    value = evidence()
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=value['status'], comparisons=value['comparisons'],
                          minimal_phase_probes=value['minimal_phase_probes'])))
