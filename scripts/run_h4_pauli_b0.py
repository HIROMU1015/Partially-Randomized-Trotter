#!/usr/bin/env python3
"""Dedicated finite matched B0 runner; no old grant or approval machinery."""
import argparse
from dataclasses import asdict
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
pkg = types.ModuleType("h4_b0_execution_private"); pkg.__path__ = [str(ROOT / "src/trotterlib")]
sys.modules[pkg.__name__] = pkg
E = importlib.import_module(pkg.__name__ + ".h4_pauli_b0_execution")
OUT = ROOT / "artifacts/non_df_h4_pauli_b0_execution/2026-10-25"
RUN = OUT / "run01"
COMPILER_PYTHON = "/home/AbeHiromu/projects/h6-environments/20261012/native-dependency-v2/venv/bin/python"
ARB_PYTHON = "/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/non-df-h4-c-reference-fix-20261023/.c-reference-fix-venv/bin/python"


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inputs():
    c = read(OUT / "active_contract.json")
    if c["execution_enabled"] is not True:
        raise ValueError("EXECUTION_DISABLED")
    refs = read(ROOT / "artifacts/non_df_h4_pauli_b0_preparation/2026-10-25/evidence_references.json")
    for entry in refs["entries"]:
        if sha(ROOT / entry["path"]) != entry["sha256"]:
            raise ValueError("SAVED_INPUT_CHANGED:" + entry["path"])
    return c, refs


def verify_source():
    source = read(RUN / "SOURCE.json")
    if sha(OUT/"active_contract.json") != source["contract_sha256"]:
        raise ValueError("EXECUTION_CONTRACT_CHANGED")
    for path, digest in source["code_sha256"].items():
        if sha(ROOT / path) != digest:
            raise ValueError("EXECUTING_SOURCE_CHANGED:" + path)
    inputs()
    return source


def prepare():
    c, refs = inputs()
    RUN.mkdir()  # an existing attempt is preserved; never overwrite/retry it
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    paths = ["scripts/run_h4_pauli_b0.py", "src/trotterlib/h4_pauli_b0_execution.py", "src/trotterlib/h4_pauli_b0.py", "src/trotterlib/h4_d2_pauli_control.py", "src/trotterlib/h4_arb_bridge.py", "src/trotterlib/h4_arb_pr0_preparation.py", "src/trotterlib/h4_ball_evidence.py", "src/trotterlib/h4_conditional_cng.py", "src/trotterlib/rte_compiled_cost.py", "src/trotterlib/fixed_signal_evaluation.py"]
    for path in paths:
        if subprocess.check_output(["git", "show", source + ":" + path], cwd=ROOT) != (ROOT/path).read_bytes():
            raise ValueError("SOURCE_NOT_COMMITTED:" + path)
    save(RUN/"SOURCE.json", dict(commit=source, code_sha256={p:sha(ROOT/p) for p in paths},
                                 contract_sha256=sha(OUT/"active_contract.json"),
                                 runtime=dict(compiler_python=COMPILER_PYTHON, qiskit=importlib.metadata.version("qiskit"),
                                              arb_python=ARB_PYTHON),
                                 input_reference_sha256=sha(ROOT/"artifacts/non_df_h4_pauli_b0_preparation/2026-10-25/evidence_references.json")))
    from h4_b0_execution_private import h4_pauli_b0 as B
    from qiskit import qpy
    if importlib.metadata.version("qiskit") != "1.3.0":
        raise ValueError("COMPILER_VERSION")
    model = read(ROOT/c["references"]["C_model"])
    receipt = read(ROOT/c["references"]["receipt"])
    prior = read((ROOT/c["references"]["retained_body"]).with_name("input.json"))
    with (ROOT/c["references"]["retained_body"]).open("rb") as stream:
        bodies = qpy.load(stream)
    if len(bodies) != 1 or model["n_qubits"] != 8 or receipt["prefixes"]["pauli_jw"]["L_D"] != 27:
        raise ValueError("FIXED_H4_RETAINED_INPUT")
    check = E.validate_retained_body(bodies[0], model, receipt["prefixes"]["pauli_jw"], prior)
    save(RUN/"body_validation.json", check)
    wrappers, control = B.central_wrappers(bodies[0])
    save(RUN/"control_mapping.json", control.sidecar)
    for axis, circuit in wrappers.items():
        directory = RUN/axis; directory.mkdir()
        with (directory/"input.qpy").open("xb") as stream:
            qpy.dump(circuit, stream)
    save(RUN/"build_counts.json", dict(saved_body_validation=1, new_uncontrolled_body_builds=0,
                                       central_lifts=1, axis_wrappers=2))


def compile_axis(axis):
    verify_source()
    if (RUN/axis/"output.qpy").exists() or (RUN/axis/"metrics.json").exists():
        raise FileExistsError("COMPILE_ALREADY_SAVED_NO_RETRY")
    from h4_b0_execution_private.h4_d1_adapter import compile_saved
    metrics = compile_saved(RUN/axis)
    save(RUN/axis/"metrics.json", metrics)


def target(bits):
    verify_source()
    if (RUN/("target_"+str(bits)+".json")).exists():
        raise FileExistsError("TARGET_ALREADY_SAVED_NO_RETRY")
    c = read(OUT/"active_contract.json")
    result = E.target_enclosure(read(ROOT/c["references"]["C_model"]),
                                read(ROOT/c["references"]["C_reference"]), bits)
    result["python_flint"] = importlib.metadata.version("python-flint")
    result["precision_decisions"] = E.precision_decisions(result)
    save(RUN/("target_"+str(bits)+".json"), result)


def launch():
    if RUN.exists():
        raise FileExistsError("EXISTING_ATTEMPT_PRESERVED_NO_RETRY")
    children = []; handles = []
    def start(label, python, args):
        out = (RUN/(label+".stdout.log")).open("xb")
        err = (RUN/(label+".stderr.log")).open("xb")
        handles.extend((out, err))
        command = [python, str(Path(__file__).resolve()), *args]
        p = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), stdout=out, stderr=err)
        row = dict(label=label, pid=p.pid, command=command, started_unix=time.time(), native_exit=None, reaped=False)
        children.append((p, row)); save(RUN/"children.json", [r for _, r in children])
        return p, row
    def finish(pair):
        p, row = pair; row.update(native_exit=p.wait(), reaped=True, ended_unix=time.time())
        save(RUN/"children.json", [r for _, r in children])
        if row["native_exit"] != 0:
            raise RuntimeError("NATIVE_CHILD_FAILED:"+row["label"])
    started = time.time()
    try:
        prepare()
        save(RUN/"parent_start.json", dict(pid=os.getpid(), started_unix=started))
        compiles = [start(axis, COMPILER_PYTHON, ["compile", "--axis", axis]) for axis in E.AXES]
        target_pair = start("target_64", ARB_PYTHON, ["target", "--bits", "64"])
        finish(target_pair)
        bits = 64
        for next_bits in (96, 128):
            tier = read(RUN/("target_"+str(bits)+".json"))
            if not any(r["status"] == "WIDTH_AMBIGUITY" for r in tier["precision_decisions"]):
                break
            # Do not spend a higher tier if the fixed saved reference is already
            # wider than every new raw-signal component.
            keys = ("real", "imag")
            if all(Qwidth(tier["raw_signal"][k]) <= Qwidth(tier["saved_full_reference"][k]) for k in keys):
                break
            finish(start("target_"+str(next_bits), ARB_PYTHON, ["target", "--bits", str(next_bits)]))
            bits = next_bits
        for pair in compiles:
            finish(pair)
        verify_source()
        target_result = read(RUN/("target_"+str(bits)+".json"))
        costs = {a:read(RUN/a/"metrics.json") for a in E.AXES}
        contract = read(OUT/"active_contract.json")
        results = E.conditional_results(target_result, costs, read(ROOT/contract["references"]["PR_conditional_results"]))
        results.update(SOURCE=read(RUN/"SOURCE.json")["commit"], selected_bits=bits, costs=costs,
                       target_state_actions=sum(r["label"].startswith("target_") for _, r in children),
                       top_level_compiles=2, compiled_state_actions=0, elapsed_seconds=time.time()-started)
        save(RUN/"results.json", results)
        inputs()
        save(RUN/"STOP.json", dict(status="COMPLETED_MANDATORY_STOP", pid=os.getpid(),
                                   ended_unix=time.time(), all_children_reaped=all(r["reaped"] for _, r in children),
                                   native_child_exits=[r["native_exit"] for _, r in children],
                                   new_scientific_grants=0, actual_precision_certified=False))
        print(json.dumps(dict(status="COMPLETED_MANDATORY_STOP", selected_bits=bits,
                              rows=[{k:r[k] for k in ("epsilon_sig", "N_total", "G_exact", "rho_zero_order", "independent_guard_order")} for r in results["rows"]]),ensure_ascii=False))
    except BaseException as error:
        # Reap only this runner's direct children; never touch other jobs. A
        # successful child is retained even if its sibling failed. No retry.
        for p, row in children:
            if not row["reaped"]:
                row.update(native_exit=p.wait(), reaped=True, ended_unix=time.time())
        if RUN.exists():
            save(RUN/"children.json", [r for _, r in children])
            save(RUN/"STOP.json", dict(status="FAILED_STOP_NO_RETRY", error=repr(error),
                                       all_children_reaped=all(r["reaped"] for _, r in children)))
        raise
    finally:
        for stream in handles:
            stream.close()


def Qwidth(interval):
    from fractions import Fraction
    return Fraction(interval["upper"]) - Fraction(interval["lower"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("run", "compile", "target"))
    parser.add_argument("--axis", choices=E.AXES)
    parser.add_argument("--bits", type=int, choices=(64,96,128))
    a = parser.parse_args()
    if a.mode == "run": launch()
    elif a.mode == "compile": compile_axis(a.axis)
    else: target(a.bits)
