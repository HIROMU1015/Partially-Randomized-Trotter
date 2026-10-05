"""Four predeclared synthetic full-operator checks, final 128-call budget slots."""
import json
from pathlib import Path
import sys
import time
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import XXPlusYYGate
from qiskit.quantum_info import Operator
from synthetic_fixture import compile_one, own_limits, validate_environment


def main():
    directory = Path(sys.argv[1])
    validate_environment()
    own_limits()
    benchmark = json.loads((directory / "benchmark_result_v0.json").read_text())
    tests = json.loads((directory / "synthetic_tests_v0.json").read_text())
    assert benchmark["actual_transpile_calls"] + tests["actual_transpiles"] == 124
    definition = {"cases": [{"phase": phase, "axis": axis} for phase in [0.0, 0.173] for axis in ["cosine", "sine"]],
                  "new_transpile_cap": 4, "prior_transpiles": 124, "total_cap": 128,
                  "science_access": False, "reason": "full wrapper operator comparison beyond fixed-state axis expectation checks"}
    with (directory / "operator_check_definition_v0.json").open("x") as handle:
        json.dump(definition, handle, indent=2)
        handle.write("\n")
    records = []
    start = time.perf_counter()
    for case in definition["cases"]:
        system = QuantumCircuit(2)
        system.append(XXPlusYYGate(0.71, -0.29), [0, 1])
        system.rz(-0.37, 0)
        system.rz(0.23, 1)
        system.global_phase = case["phase"]
        wrapper = QuantumCircuit(3, 1)
        wrapper.h(2)
        wrapper.append(system.to_gate().control(1), [2, 0, 1])
        if case["axis"] == "sine": wrapper.sdg(2)
        wrapper.h(2)
        wrapper.measure(2, 0)
        compiled = compile_one(wrapper)
        reference = Operator(wrapper.remove_final_measurements(inplace=False)).data
        actual = Operator(compiled.remove_final_measurements(inplace=False)).data
        residual = float(np.max(np.abs(actual - reference)))
        assert residual < 1e-12
        records.append({**case, "exact_operator_max_absolute_residual": residual,
                        "global_phase_retained": True, "actual_transpile_calls": 1})
    result = {"status": "SYNTHETIC_FULL_OPERATOR_CHECKS_PASS", "records": records,
              "actual_transpile_calls": len(records), "total_synthetic_transpile_calls": 124 + len(records),
              "wall_s": time.perf_counter() - start, "science_wrapper_records": 0,
              "molecular_runtime_gpu_access": 0}
    with (directory / "compiled_operator_checks_v0.json").open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
