"""Acquire the two missing direct prices for the saved H6 Full PF round 7."""

import argparse
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parent
CANDIDATE = {
    "system": "H6", "method": "FULL", "formula": "4th(m5_best)",
    "prefix": 19, "T": 12.8, "q": 8, "delta": 1.6,
    "R_outer": None, "K": None, "id": "5cc99f967c6a176c43fb",
}
COMPILER_HASH = "1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b"


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, record):
    with path.open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")


def require(condition, label):
    if not condition:
        raise ValueError(label)


def check():
    manifest = read(ROOT / "MANIFEST.json")
    for file, expected in manifest["sha256"].items():
        require(digest(ROOT / file) == expected, f"Staged bytes differ: {file}")
    price_inputs = read(ROOT / "PRICE-INPUTS.json")
    plan = read(ROOT / "m7-PLAN.json")
    selected = [row for row in plan["rows"] if row["candidate"] == CANDIDATE]
    require(len(selected) == 1 and selected[0]["m"] == 7, "Saved round binding")
    for key, value in plan["scientific_binding"].items():
        require(price_inputs["scientific_binding"][key] == value, f"H6 binding: {key}")
    for name, expected in price_inputs["input_sha256"].items():
        require(digest(ROOT / "inputs" / name) == expected, f"Saved H6 input: {name}")
    require(price_inputs["compiler_hash"] == COMPILER_HASH, "Saved compiler binding")

    sys.path[:0] = [str(ROOT / "provider/src"), str(ROOT / "dependency/scripts")]
    import run_h6_df19_price_v2_q4 as q4
    from trotterlib import rte_compiled_cost as costs
    from trottertracks.resource_applicability import ax2b_native_df_v5 as native
    from trottertracks.resource_applicability.h6_hash_reuse_v1 import use_fingerprint_reuse

    provider = ROOT / "provider"
    p = q4.p
    require(Path(p.trottertracks.__file__).resolve().is_relative_to(provider), "Provider import path")
    require(Path(native.__file__).resolve().is_relative_to(provider), "Native import path")
    require(Path(costs.__file__).resolve().is_relative_to(provider), "Compiler import path")
    require((p.np.__version__, p.qiskit.__version__) == ("1.26.4", "1.3.0"), "Fixed runtime")
    p.PROVIDER_ROOT = provider
    sources = {p.CARRIER: ROOT / "carrier", p.PROVIDER: provider}
    p.blob = lambda ref, path: (sources[ref] / path).read_bytes()
    helpers = p.helpers()
    prep = load_bound_preparation(price_inputs)
    costs.transpile = partial(p.qiskit.transpile, num_processes=1)
    runtime = {
        "numpy": p.np.__version__, "qiskit": p.qiskit.__version__,
        "affinity_cpus": len(os.sched_getaffinity(0)),
        "manifest_sha256": digest(ROOT / "MANIFEST.json"),
    }
    return q4, helpers, use_fingerprint_reuse, price_inputs, prep, runtime


def load_bound_preparation(inputs):
    from qiskit.circuit.library import PhaseGate, XXPlusYYGate
    from trotterlib.df_partial_s2 import (
        DFDeterministicFragmentSpec, DFDeterministicOneBodySpec, _block_hash_payload,
    )
    from trotterlib.df_rte_tail import DFBasisRegistry
    from trottertracks.resource_applicability.ax2b_h6_saved_completion_loader_v1 import load_h6_snapshot

    saved = ROOT / "inputs"
    ham, *_ = load_h6_snapshot(
        saved / "H6_snapshot.npz", read(saved / "H6_snapshot_receipt.json"),
        read(saved / "H6_df_receipt.json"),
    )
    record = read(ROOT / "Full19.preparation.json")
    payload = record["payload"]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    expected = inputs["preparation_hash"]["19"]
    require(hashlib.sha256(encoded).hexdigest() == record["preparation_hash"] == expected,
            "Full19 archived preparation hash")
    n = record["num_system_qubits"]
    require(n == ham.n_qubits == 12 and record["scalar"] == ham.constant, "Saved scalar and size")
    require(payload["hamiltonian_hash"] == inputs["scientific_binding"]["hamiltonian"],
            "Hamiltonian hash")
    rows = payload["deterministic_blocks"]
    require([row["original_fragment_index"] for row in rows] == [None, *range(19)],
            "Saved fragment order")
    registry = DFBasisRegistry()
    blocks = []
    gates = {"p": PhaseGate, "xx_plus_yy": XXPlusYYGate}
    types = {"one_body": DFDeterministicOneBodySpec, "df_fragment": DFDeterministicFragmentSpec}
    for row in rows:
        ops = tuple((gates[op["name"]](*(float(p) for p in op["parameters"])), tuple(op["qubits"]))
                    for op in row["basis_change_operations"])
        basis = registry.register(ops, num_system_qubits=n)
        require(basis.basis_hash == row["basis_hash"] and basis.basis_id == row["basis_id"],
                f"Restored basis differs: {row['block_id']}")
        block = types[row["block_kind"]](**dict(
            row, basis_change_operations=basis.metadata.operations,
            runtime_basis_operations=ops,
        ))
        require(json.loads(json.dumps(_block_hash_payload(block))) == row,
                f"Restored block differs: {row['block_id']}")
        blocks.append(block)
    return SimpleNamespace(
        num_system_qubits=n, deterministic_blocks=tuple(blocks),
        preparation_hash=expected, scalar=ham.constant,
    )


def acquire_axis(axis, output, evolution, fingerprints):
    from qiskit import qpy
    from trotterlib.rte import CompilerSettings
    from trotterlib.rte_compiled_cost import transpile_and_measure_cost
    from trottertracks.resource_applicability import ax2b_native_df_v5 as native

    wrapper = native.build_native_hadamard_wrapper(
        evolution, axis=axis, include_measurement=True,
        max_instructions=len(evolution.circuit.data) + 4,
    )
    fingerprint = fingerprints.fingerprint(wrapper)
    compiler = CompilerSettings(("rz", "sx", "x", "cx"), None, None, 1, None, None, 17, "1.3.0")
    write(output / f"{axis}_compile_started.json", {"fingerprint": fingerprint, "compiler": COMPILER_HASH})
    print(f"COMPILE_START {axis} {fingerprint}", flush=True)
    cost = transpile_and_measure_cost(wrapper, compiler, actual_circuit_fingerprint=fingerprint)
    require(cost.compiler_settings_hash == COMPILER_HASH, "Compiled compiler hash")
    require(fingerprints.fingerprint(wrapper) == fingerprint, "Compiler mutated input")
    metrics = {name: getattr(cost, name) for name in (
        "rz_count", "rz_depth", "cx_count", "cx_depth", "total_depth", "circuit_size",
    )}
    receipt = {"metrics": metrics, "fingerprint": fingerprint, "compiler": COMPILER_HASH}
    write(output / f"{axis}_compiled_metrics.json", receipt)
    circuit_file = output / f"{axis}_compiled.qpy"
    temporary = circuit_file.with_suffix(".tmp")
    with temporary.open("xb") as stream:
        qpy.dump(cost.transpiled_circuit, stream)
    temporary.replace(circuit_file)
    receipt.update(
        cache_hit=False, qpy=circuit_file.name, qpy_sha256=digest(circuit_file),
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    )
    write(output / f"{axis}_price.json", receipt)
    print(f"COMPILE_DONE {axis} RZ={metrics['rz_count']}", flush=True)
    return receipt


def execute(output, q4, helpers, use_fingerprint_reuse, inputs, prep, runtime):
    require(runtime["affinity_cpus"] >= 2, "Two CPU cores required for X/Y workers")
    started = time.monotonic()
    output.mkdir(mode=0o700, exist_ok=False)
    write(output / "STARTED.json", {
        "candidate": CANDIDATE, "round": 7, "axes": list(q4.p.AXES),
        "scientific_binding": inputs["scientific_binding"], "runtime": runtime,
        "pbs_job_id": os.environ.get("PBS_JOBID"),
    })
    try:
        print("BUILD_START Full PF m=7 qPF=8", flush=True)
        q4.price_axis = acquire_axis
        with use_fingerprint_reuse() as fingerprints:
            evolution = helpers["compose"](CANDIDATE, prep, [], prep.scalar, fingerprints.fingerprint)
            write(output / "native.json", {
                "instructions": len(evolution.circuit.data),
                "instruction_bound": evolution.instruction_upper_bound,
                "fingerprint": evolution.circuit_fingerprint,
                "preparation_hash": prep.preparation_hash,
            })
            axes = q4.parallel_axes(output, evolution, fingerprints)
        require(len({receipt["fingerprint"] for receipt in axes.values()}) == 2, "Distinct X/Y fingerprints")
        write(output / "result.json", {
            "status": "PASS_PRICE_ACQUIRED", "candidate": CANDIDATE, "round": 7,
            "axes": axes, "wall_seconds": time.monotonic() - started,
            "signals_generated": 0, "q3_calls": 0,
        })
        print("PASS_PRICE_ACQUIRED Full PF m=7", flush=True)
    except BaseException as error:
        write(output / "STOP.json", {"error": str(error), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--execute", type=Path, metavar="NEW_OUTPUT_DIRECTORY")
    args = parser.parse_args()
    context = check()
    if args.check:
        print(json.dumps({"candidate": CANDIDATE, "runtime": context[-1],
                          "preparation_hash": context[-2].preparation_hash, "science_calls": 0}))
    else:
        execute(args.execute, *context)
