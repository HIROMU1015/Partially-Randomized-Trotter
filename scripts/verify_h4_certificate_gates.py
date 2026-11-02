#!/usr/bin/env python3
"""Phase-T sealed metadata preflight; no scientific evaluation or acceptance.

Exit 0 means the technical check completed and wrote its blockers. It never
means that a scientific certificate has been accepted. Exit 2 means a schema,
integrity or authority mismatch. Existing output paths are never overwritten.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "src/trotterlib/h4_certificate_gates.py"
spec = importlib.util.spec_from_file_location("h4_gate_technical_checker", MODULE)
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)  # Avoid scientific imports from trotterlib.__init__.
INVENTORY = "artifacts/non_df_h4_certificate_gate_audit/2026-10-23/result_inventory.json"
OUTPUT_NAMESPACE = "artifacts/non_df_h4_certificate_gate_foundation"
SOURCE_FILES = {"src/trotterlib/h4_certificate_gates.py",
                "scripts/verify_h4_certificate_gates.py",
                "tests/test_h4_certificate_gates.py"}


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def validate_seal(seal, root):
    g.keys(seal, {"kind", "technical_SOURCE", "base_audit_commit", "files", "runtime",
                  "audit_inventory_sha256", "scientific_execution_authorized",
                  "certificate_acceptance_authorized", "N_G_authorized"}, "TECHNICAL_SEAL_SCHEMA")
    g.require(seal["kind"] == "H4_GATE_PHASE_T_SOURCE_NO_SCIENTIFIC_AUTHORITY", "TECHNICAL_SEAL_KIND")
    g.hex_digest(seal["technical_SOURCE"], 40)
    g.require(seal["base_audit_commit"] == "0369fa2a05aac7820e1c0f7aa7e55f9d048a0efa",
              "BASE_AUDIT_COMMIT")
    g.require(all(seal[k] is False for k in ("scientific_execution_authorized",
              "certificate_acceptance_authorized", "N_G_authorized")), "SEAL_NO_SCIENCE_AUTHORITY")
    g.keys(seal["files"], SOURCE_FILES, "TECHNICAL_SOURCE_FILE_SET")
    contents = {name: g.checked_file(root, name, seal["files"][name])
                for name in sorted(SOURCE_FILES)}
    for name, raw in contents.items():
        committed = subprocess.check_output(["git", "show", seal["technical_SOURCE"] + ":" + name], cwd=root)
        g.require(raw == committed, "TECHNICAL_SOURCE_COMMIT_MISMATCH")
    g.checked_file(root, INVENTORY, seal["audit_inventory_sha256"])
    g.keys(seal["runtime"], {"executable", "sha256", "version"}, "TECHNICAL_RUNTIME_SCHEMA")
    executable = Path(sys.executable).resolve()
    g.require(str(executable) == seal["runtime"]["executable"]
              and sys.version == seal["runtime"]["version"]
              and hashlib.sha256(executable.read_bytes()).hexdigest() == seal["runtime"]["sha256"],
              "TECHNICAL_RUNTIME_MISMATCH")


def run(args):
    seal_path = Path(args.seal).resolve()
    seal = g.read_json_bytes(seal_path.read_bytes())
    validate_seal(seal, ROOT)
    output = Path(args.output).absolute()
    namespace = ROOT / OUTPUT_NAMESPACE
    g.require(namespace.resolve() in output.resolve().parents and not output.is_symlink(),
              "OWN_FRESH_TECHNICAL_NAMESPACE_REQUIRED")
    # Validate all inputs before claiming a new output directory.
    bundles = g.bundles_from_saved_audit(ROOT, INVENTORY)
    results = [g.validate_bundle(bundle, ROOT) for bundle in bundles]
    output.mkdir(parents=True, exist_ok=False)
    write(output / "bundles.json", dict(schema=g.SCHEMA, bundles=bundles))
    write(output / "preflight.json", dict(
        kind="TECHNICAL_PROOF_PREFLIGHT_NOT_CERTIFICATE", technical_SOURCE=seal["technical_SOURCE"],
        input_seal_sha256=hashlib.sha256(seal_path.read_bytes()).hexdigest(),
        status="PASS_TECHNICAL_CHECK_SCIENTIFIC_GATES_BLOCKED", axes_checked=len(results),
        results=results, scientific_calls=0, scientific_reanalysis=False, compile_calls=0,
        sampling_calls=0, RNG_replay_calls=0, high_precision_or_interval_calls=0,
        certificates_generated=0, formal_certificates_accepted=0, N_G_calls=0,
        original_native_completion="UNKNOWN_UNCHANGED", C_native_completion="EXISTING_PASS_UNCHANGED",
        mandatory_STOP=True))
    write(output / "STOP.json", dict(kind="PHASE_T_METADATA_PREFLIGHT_STOP", mandatory_STOP=True,
          scientific_execution_authorized=False, formal_certificate_accepted=False, N_G_authorized=False,
          old_STOP_or_native_evidence_replaced=False))
    print(json.dumps(dict(status="PASS_TECHNICAL_CHECK_SCIENTIFIC_GATES_BLOCKED",
                          axes=len(results), scientific_calls=0, formal_certificates_accepted=0)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, KeyError, TypeError, OSError, subprocess.CalledProcessError) as error:
        print("TECHNICAL_STOP: " + str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
