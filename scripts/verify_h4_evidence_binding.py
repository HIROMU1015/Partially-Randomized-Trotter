#!/usr/bin/env python3
"""Sealed static H4 binding and historical-compiler audit; no science calls."""
import argparse
import ast
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("h4_evidence_adapter", ROOT / "src/trotterlib/h4_evidence_binding.py")
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
H = "artifacts/non_df_h4_proof_binding/2026-10-23"
FILES = {"src/trotterlib/h4_evidence_binding.py", "scripts/verify_h4_evidence_binding.py",
         "tests/test_h4_evidence_binding.py", H + "/evidence_manifest.json"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
        stream.write("\n")


def source_site(store, role, needle):
    raw = store.raw[role].decode()
    lines = [i for i, line in enumerate(raw.splitlines(), 1) if needle in line]
    b.require(bool(lines), "SOURCE_SITE_MISSING:" + role + ":" + needle)
    return dict(role=role, **store.manifest["files"][role], lines=lines, needle=needle)


def inspect_compiler(store):
    """Inspect only known case files and explicitly sealed runtime source paths."""
    seal = store.document("C_seal")
    paths = {
        "level1": "qiskit/transpiler/preset_passmanagers/level1.py",
        "plugins": "qiskit/transpiler/preset_passmanagers/builtin_plugins.py",
        "common": "qiskit/transpiler/preset_passmanagers/common.py",
        "translation": "qiskit/transpiler/passes/basis/basis_translator.py",
        "one_qubit": "qiskit/transpiler/passes/optimization/optimize_1q_decomposition.py",
        "inverse_cancel": "qiskit/transpiler/passes/optimization/inverse_cancellation.py",
        "unitary_synthesis": "qiskit/transpiler/passes/synthesis/unitary_synthesis.py",
        "controlled": "qiskit/circuit/controlledgate.py",
        "equivalences": "qiskit/circuit/library/standard_gates/equivalence_library.py",
        "accelerated_binary": "qiskit/_accelerate.abi3.so",
    }
    runtime = {}
    for name, suffix in paths.items():
        matches = [p for p in seal["legacy_runtime_files"] if p.endswith("/" + suffix)]
        b.require(len(matches) == 1, "RUNTIME_COMPILER_FILE_MISSING:" + suffix)
        path = Path(matches[0])
        b.require(path.is_file() and not path.is_symlink() and sha(path) == seal["legacy_runtime_files"][str(path)],
                  "RUNTIME_COMPILER_HASH_MISMATCH")
        raw = None if name == "accelerated_binary" else path.read_text()
        runtime[name] = dict(path=str(path), sha256=seal["legacy_runtime_files"][str(path)],
                             lines=len(raw.splitlines()) if raw else None, source=raw)
    # Inspect only named source paths, without importing or instantiating Qiskit.
    def site(name, needle):
        text = runtime[name]["source"]
        lines = [i for i, line in enumerate(text.splitlines(), 1) if needle in line]
        b.require(bool(lines), "COMPILER_SOURCE_ANCHOR_MISSING")
        return dict(path=runtime[name]["path"], sha256=runtime[name]["sha256"], lines=lines, needle=needle)
    rules = [
        dict(topic="1Q decomposition", status="MISSING_EVIDENCE", evidence=site("plugins", "Optimize1qGatesDecomposition("),
             gap="Floating Euler decomposition uses accelerated kernels; no verified error/pass receipt or coverage in T's symbolic RZ inverse rule."),
        dict(topic="basis translation", status="MISSING_EVIDENCE", evidence=site("common", "BasisTranslator(sel, basis_gates, target)"),
             gap="Session equivalence library and DAG rewrite coverage are not a verified whole-compiler proof."),
        dict(topic="controlled/high-level synthesis", status="MISSING_EVIDENCE", evidence=site("common", "HighLevelSynthesis("),
             gap="All controlled decompositions, phase accumulation and runtime branches require a separate proof."),
        dict(topic="unitary synthesis", status="MISSING_EVIDENCE", evidence=site("common", "UnitarySynthesis("),
             gap="Path exists in translator stage; invocation for each historical program is not recorded. No synthesis error proof is attached."),
        dict(topic="inverse cancellation", status="EVIDENCE_EXISTS_UNCHECKED", evidence=site("plugins", "InverseCancellation("),
             gap="T proves adjacent exact-symbol cancellation only; matching, tolerance, DAG mutation and all supported pairs are not covered."),
        dict(topic="global phase", status="MISSING_EVIDENCE", evidence=site("one_qubit", "out_dag.global_phase"),
             gap="Phase transfer is visible in source; historical branch/rounding correctness and controlled relative phase remain unproved."),
        dict(topic="layout/routing", status="EVIDENCE_EXISTS_UNCHECKED", evidence=site("level1", "if coupling_map or initial_layout:"),
             gap="No coupling map/backend/layout was configured; this simplifies intended paths but does not identify every pass executed historically."),
    ]
    cases = []
    extensions = {".qpy", ".qasm", ".qasm3", ".dag"}
    for name in b.T.CASES:
        ref = store.manifest["files"][name + "_old"]
        directory = store.root / Path(ref["path"]).parent
        known = [p for p in directory.rglob("*") if p.is_file()]
        program_candidates = [p for p in known if p.suffix.lower() in extensions]
        result = store.document(name + "_old")
        cases.append(dict(case=name, case_root=str(directory.relative_to(store.root)),
                          files_inspected=[str(p.relative_to(store.root)) for p in sorted(known)],
                          program_serialization_candidates=[str(p.relative_to(store.root)) for p in program_candidates],
                          compiler=result["compiler"], old_result_sha256=ref["sha256"],
                          postcompiled_program_hash="NOT_SAVED_IN_INSPECTED_EVIDENCE",
                          executed_pass_receipt="NOT_SAVED_IN_INSPECTED_EVIDENCE",
                          guarantee_status="MISSING_EVIDENCE"))
    return dict(kind="HISTORICAL_COMPILED_PROGRAM_FEASIBILITY_STATIC_AUDIT", cases=cases,
                runtime_files=[{k: v for k, v in x.items() if k != "source"} for x in runtime.values()],
                candidate_source_paths=rules,
                actual_historical_pass_sequence="UNKNOWN_NO_CALLBACK_OR_PASS_RECEIPT",
                historical_program_recovery="NOT_FOUND_WITHIN_FOUR_KNOWN_CASE_ROOTS; global absence is not claimed",
                source_only_full_compiler_equivalence="NOT_PROVED",
                empirical_checks=store.document("compiler_audit"),
                fingerprint_site=source_site(store, "old_evaluator_source", "fingerprints[axis] = cost.actual_circuit_fingerprint"),
                cache_site=source_site(store, "old_compiler_source", "persistent_path: str | Path | None = None"),
                safety_scope="No process or filesystem-wide search; no new compile/import/program evaluation",
                compile_calls=0, numeric_reanalysis=False)


def validate_seal(root, seal):
    b.T.keys(seal, {"kind", "technical_SOURCE", "base_commit", "files", "runtime",
                    "execution_authorized", "grant_valid", "dispatch_enabled",
                    "formal_certificate_accepted", "N_G_authorized"}, "BINDING_TECHNICAL_SEAL")
    b.require(seal["kind"] == "H4_BINDING_TECHNICAL_SOURCE_NO_SCIENTIFIC_AUTHORITY"
              and seal["base_commit"] == b.FIXED_COMMITS["base"], "BINDING_SEAL_KIND")
    b.T.hex_digest(seal["technical_SOURCE"], 40)
    b.require(all(seal[k] is False for k in ("execution_authorized", "grant_valid", "dispatch_enabled",
                                            "formal_certificate_accepted", "N_G_authorized")), "BINDING_SEAL_NO_AUTHORITY")
    b.T.keys(seal["files"], FILES, "BINDING_SOURCE_SET")
    raw = {p: b.T.checked_file(root, p, seal["files"][p]) for p in sorted(FILES)}
    for p, data in raw.items():
        b.require(data == subprocess.check_output(["git", "show", seal["technical_SOURCE"] + ":" + p], cwd=root),
                  "BINDING_SOURCE_COMMIT_MISMATCH")
    executable = Path(sys.executable).resolve()
    b.require(seal["runtime"] == dict(executable=str(executable), sha256=sha(executable), version=sys.version),
              "BINDING_RUNTIME_MISMATCH")


def run(args):
    seal = b.T.read_json_bytes(Path(args.seal).read_bytes())
    validate_seal(ROOT, seal)
    manifest = b.T.read_json_bytes((ROOT / H / "evidence_manifest.json").read_bytes())
    store = b.EvidenceStore(ROOT, manifest)
    for role, ref in manifest["files"].items():
        b.require(store.raw[role] == subprocess.check_output(["git", "show", ref["source_commit"] + ":" + ref["path"]], cwd=ROOT),
                  "FIXED_EVIDENCE_COMMIT_MISMATCH:" + role)
    bindings = b.produce_bindings(store)
    verification = b.verify_bindings(store, bindings)
    compiler = inspect_compiler(store)
    output = Path(args.output).absolute()
    b.require((ROOT / "artifacts/non_df_h4_proof_binding").resolve() in output.resolve().parents,
              "NEW_BINDING_NAMESPACE_ONLY")
    b.require(not output.is_symlink(), "NO_OUTPUT_SYMLINK")
    output.mkdir(parents=True, exist_ok=False)
    save(output / "gate_bindings.json", bindings)
    save(output / "verification.json", verification)
    save(output / "historical_compiler.json", compiler)
    with (output / "gate_matrix.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["case", "epsilon_sig", "axis", "gate", "scope", "saved_claim", "binding_status", "proof_status", "obligations"])
        for row in bindings["rows"]:
            for gate in row["gates"].values():
                writer.writerow([*[row["target"][k] for k in ("case", "epsilon_sig", "axis")], gate["gate"], gate["scope"],
                                 gate["saved_claim"], gate["binding_status"], gate["proof_status"], " | ".join(gate["obligations"])])
    save(output / "STOP.json", dict(kind="STATIC_BINDING_CHECK_STOP", mandatory_STOP=True,
        execution_authorized=False, grant_valid=False, dispatch_enabled=False,
        formal_certificate_accepted=False, N_G_authorized=False, science_calls=0))
    print(json.dumps(verification))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, TypeError, KeyError, OSError, subprocess.CalledProcessError) as error:
        print("STATIC_BINDING_STOP: " + str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
