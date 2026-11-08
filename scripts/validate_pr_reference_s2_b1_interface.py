"""Authorized B1 only: one saved record, four fixed local fixtures.

No synthesis, strict_guard, compile, RZ-target guard, RPE, or T aggregation.
Unchanged functions are AST-loaded to avoid unrelated scientific initializers.
"""
from __future__ import annotations
import ast
import hashlib
import importlib.metadata
import io
import json
import lzma
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import mpmath as mp
import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from pygridsynth.domega_unitary import DOmegaUnitary as Exact
from trottertracks.pr_reference_s2_record_overlay import SingleSavedRecordOverlay
from trottertracks.pr_reference_common_word_consumer import consume_native

BASE = "ca1e71b3624705d7a942572b91b0b50ce4eeb713"
OUT = ROOT / "artifacts/pr_reference_s2_b1_interface/2026-11-08"
PLAN = "docs/research/pr_reference_s3_s2_reuse_20261108/NEXT_VALIDATION_PLAN.md"
ARCHIVE = "artifacts/track_c2_b_pilot/2026-11-06/raw_synthesis.tar.xz"
MEMBER = "synthesis/1c223a1ac0ccd6b7e2729dae93da988f4bf9d3bbc10ac1bba0500ed4ebc3029f.json"
LITERAL = "0x1.0b1d9883f9213p+1"
WORD_HASH = "981736be8f2688787aa2e1da4598b9d42b016abf32171fcc27038f88f55c0155"
TOLERANCE = 1e-10  # Reuse N4-B's emitter diagnostic threshold; not a strict guard.
INPUTS = {}
FUNCTIONS = []


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def blob(path):
    raw = subprocess.check_output(["git", "show", BASE + ":" + path], cwd=ROOT)
    INPUTS[path] = dict(bytes=len(raw), sha256=sha(raw))
    return raw


def function(path, name, namespace):
    raw = blob(path)
    assert (ROOT / path).read_bytes() == raw
    node = next(n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef) and n.name == name)
    assert not node.decorator_list
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(ROOT / path), "exec"), namespace)
    FUNCTIONS.append(dict(path=path, function=name,
        function_source_sha256=sha(ast.get_source_segment(raw.decode(), node).encode())))
    return namespace[name]


def save(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, sort_keys=True,
        ensure_ascii=False, allow_nan=False) + "\n")


def fixture_definitions():
    ret = []
    for kind in ("standalone", "joint"):
        for sign in (1, -1):
            angle = float.fromhex(LITERAL) * sign
            rz = ["rz", [0 if kind == "standalone" else 2], [], [angle.hex()]]
            stream = [rz] if kind == "standalone" else [
                ["reset", [0], [], []], ["reset", [1], [], []], ["reset", [2], [], []],
                ["sx", [0], [], []], ["cx", [0, 2], [], []], rz,
                ["cx", [0, 2], [], []], ["sx", [0], [], []], ["measure", [0], [0], []]]
            ret.append(dict(id=kind + ("_positive" if sign > 0 else "_negative"),
                kind=kind, sign=sign, qubits=1 if kind == "standalone" else 3,
                clbits=0 if kind == "standalone" else 1,
                global_phase_hex=0.0.hex(), time_ordered_native=stream))
    return ret


def build_native(f):
    qc = QuantumCircuit(f["qubits"], f["clbits"], name=f["id"])
    for name, qs, cs, params in f["time_ordered_native"]:
        if name == "measure": qc.measure(qs[0], cs[0])
        else: getattr(qc, name)(*[float.fromhex(p) for p in params], *qs)
    assert float(qc.global_phase).hex() == f["global_phase_hex"]
    return qc


def unitary_body(qc):
    # Fixtures define phase=0 on input. Output's stored W phase is retained.
    body = QuantumCircuit(qc.num_qubits, global_phase=float(qc.global_phase))
    for item in qc.data:
        if item.operation.name in ("reset", "measure"):
            continue
        body.append(item.operation, [qc.find_bit(q).index for q in item.qubits], copy=False)
    return body


def main():
    assert not OUT.exists(), "Preserve previous B1 outputs; do not overwrite/retry"
    started = time.monotonic()
    assert qiskit.__version__ == "1.3.0"
    assert mp.__version__ == "1.3.0"
    assert importlib.metadata.version("pygridsynth") == "2.0.0"
    mp.mp.dps = mp.iv.dps = 100  # Existing N3-B1b-0 written-word diagnostic precision.
    blob(PLAN)
    for path in ("src/trottertracks/pr_reference_common_word_consumer.py",
                 "src/trottertracks/pr_reference_dyadic_dictionary.py"):
        assert (ROOT / path).read_bytes() == blob(path)
    raw_archive = blob(ARCHIVE)
    with tarfile.open(fileobj=io.BytesIO(raw_archive), mode="r:xz") as t:
        raw_record = t.extractfile(MEMBER).read()
    record = json.loads(raw_record)
    assert record["theta_hex"] == LITERAL and record["sequence_sha256"] == WORD_HASH
    assert record["global_W"] == 7
    metadata = json.loads(lzma.decompress(blob(
        "artifacts/pr_reference_s3_s2_static/2026-11-08/STATIC_FINDINGS.json.xz")))
    entry = next(r for r in metadata["candidates_43"] if r["literal"] == LITERAL)
    candidate = next(r for r in entry["candidate_metadata"]
        if r["archive"] == ARCHIVE and r["member"] == MEMBER)
    assert candidate["spec"] == record["spec"] and candidate["sequence_sha256"] == WORD_HASH
    assert candidate["strict_error_upper_saved"] == record["strict_error_upper"]
    provider = SingleSavedRecordOverlay(record)
    ns = dict(QuantumCircuit=QuantumCircuit, math=math)
    lower = function("src/trottertracks/track_c2_b_verify.py", "lower_native", ns)
    exact_matrix = function("src/trottertracks/track_c_h4_synthesis.py", "exact_word_matrix", dict(mp=mp))
    ns = dict(PROVIDER=provider)
    ins = function("scripts/validate_pr_reference_n4b_consumer.py", "ins", ns)
    expected_stream = function("scripts/validate_pr_reference_n4b_consumer.py", "expected_stream", ns)
    resources = dict(hostname=platform.node(), python=sys.version, executable=sys.executable,
        qiskit=qiskit.__version__, numpy=np.__version__, mpmath=mp.__version__,
        pygridsynth=importlib.metadata.version("pygridsynth"),
        available_cpu_affinity=len(os.sched_getaffinity(0)), load_average=os.getloadavg(),
        memory=[r for r in Path("/proc/meminfo").read_text().splitlines()
                if r.startswith(("MemTotal:", "MemAvailable:"))],
        execution="Four tiny dependent fixtures; reuse standalone +/- outputs in joint references. No resource caps or shared changes.")
    fixtures = fixture_definitions()
    sources = dict(base_commit=BASE, saved_inputs=INPUTS, reused_functions=FUNCTIONS,
        new_SOURCE={p: sha((ROOT / p).read_bytes()) for p in (
            "src/trottertracks/pr_reference_s2_record_overlay.py",
            "scripts/validate_pr_reference_s2_b1_interface.py",
            "tests/test_pr_reference_s2_record_overlay.py")})
    inputs = dict(scope="NEXT_VALIDATION_PLAN section 3 B1 ONLY", base_commit=BASE,
        archive=ARCHIVE, member=MEMBER, record_bytes_sha256=sha(raw_record),
        original_record_unchanged=record, fixed_fixtures=fixtures,
        SOURCE_correspondence_quoted=next(h for h in entry["saved_template_hits"]
            if h["arm"] == "Bprime" and h["archive_member"] == "catalogue/rare0_Y0.json"
            and h["native_position"] == 423),
        numerical_operator_diagnostic_tolerance=TOLERANCE,
        written_word_reference="Unchanged exact_word_matrix -> interval midpoint; no RZ target/guard computation",
        reference_interval_decimal_precision=100, environment=resources, source=sources)
    OUT.mkdir(parents=True)
    save("INPUTS.json", inputs)  # Fix inputs/SOURCE before local numeric evaluation.
    rows = []
    try:
        positive = provider.provide(float.fromhex(LITERAL))
        negative = provider.provide(-float.fromhex(LITERAL))
        assert Exact.from_gates((positive.positive_sequence + negative.signed_sequence()).replace(
            "t", "TTTTTTT")) == Exact.identity()
        im = exact_matrix(positive.positive_sequence)
        word = np.array([[complex(float(z.real.mid), float(z.imag.mid)) for z in row] for row in im])
        standalone = {}
        for f in fixtures:
            native = build_native(f)
            theta = f["sign"] * float.fromhex(LITERAL)
            local = word if f["sign"] > 0 else word.conj().T
            if f["kind"] == "standalone":
                out, certificate = provider.emit(theta, lower)
                audit = dict(certificate=certificate.summary(), source="unchanged lower_native",
                    transport_is_sequence_only=True, negative_inversion="once inside lower_native")
                standalone[f["sign"]] = out
                expected = local
                independent = None
            else:
                out, audit = consume_native(native, provider, lower)
                independent = QuantumCircuit(native.qubits, native.clbits)
                for reg in (*native.qregs, *native.cregs): independent.add_register(reg)
                for j in range(3): independent.reset(j)
                independent.sx(0); independent.cx(0, 2)
                independent.compose(standalone[f["sign"]], qubits=[2], inplace=True)
                independent.cx(0, 2); independent.sx(0); independent.measure(0, 0)
                assert ins(out) == ins(independent)
                assert float(out.global_phase).hex() == float(independent.global_phase).hex()
                assert out.qubits == native.qubits and out.clbits == native.clbits
                assert out.qregs == native.qregs and out.cregs == native.cregs
                assert audit["copying_compose"] == audit["new_QPY_roundtrips"] == audit["new_controls"] == 0
                assert len(audit["segments"]) == 1 and audit["extra_segment_joins"] == 1
                assert all(s["segment_input_phase_hex"] == 0.0.hex() for s in audit["segments"])
                prefix = QuantumCircuit(3); prefix.sx(0); prefix.cx(0, 2)
                suffix = QuantumCircuit(3); suffix.cx(0, 2); suffix.sx(0)
                expected = Operator(suffix).data @ np.kron(local, np.eye(4)) @ Operator(prefix).data
            actual_stream = ins(out)
            assert actual_stream == list(expected_stream(native))
            fences = lambda q: [(n, qs, cs) for n, qs, cs, _ in ins(q) if n in ("reset", "measure")]
            assert fences(native) == fences(out)
            mat = Operator(unitary_body(out)).data
            mismatch = float(np.linalg.norm(mat - expected, 2))
            reference_mismatch = (float(np.linalg.norm(mat - Operator(unitary_body(independent)).data, 2))
                if independent is not None else None)
            assert mismatch < TOLERANCE
            if reference_mismatch is not None: assert reference_mismatch < TOLERANCE
            phase = float(out.global_phase)
            expected_phase = (7 if f["sign"] > 0 else 1) * (math.pi / 4)
            assert phase.hex() == expected_phase.hex()
            rows.append(dict(id=f["id"], status="PASS", sign=f["sign"],
                named_gates_wires_and_fences_match=True,
                literal_and_record_selection_equal_for_both_arms=True,
                emitted_phase_hex=phase.hex(), integer_W_phase_residue=7 if f["sign"] > 0 else 1,
                positive_record_spec_retained=record["spec"],
                W_including_operator_diagnostic_spectral=mismatch,
                independent_emitter_consumer_operator_diagnostic_spectral=reference_mismatch,
                numerical_diagnostics_are_strict_certificates=False,
                output_named_instruction_stream=actual_stream, audit=audit))
        results = dict(status="B1_FOUR_LOCAL_FIXTURES_CORRECTNESS_PASS", cases=rows,
            fixed_record_only=True, exact_complete_word_adjoint_identity=True,
            saved_guard_quoted_not_reexecuted=record["strict_error_upper"],
            original_backend_exponent=-13, transport_exponent=-11,
            same_arm_independent_selector=True,
            qualification="One saved record and four artificial local fixtures; not a full-policy, SOURCE phase, m17 or RPE certificate",
            strict_emitter="Existing named-gate/IEEE phase-path proof conditions reused; finite operator diagnostics are not new strict guards",
            calls=dict(saved_record_reads=1, exact_written_word_reference=1,
                logical_fixture_configurations=4, new_synthesis=0, strict_guard=0,
                native_compile=0, RZ_target_guard=0, new_T_price_or_aggregate=0,
                A1_basis_defect=0, S1_S2_reexecution=0, m17_or_18round_RPE=0),
            SOURCE=sources, wall_seconds=time.monotonic()-started)
        save("RESULTS.json", results)
        save("STOP.json", dict(status="MANDATORY_STOP", reason="Authorized B1 scope complete",
            completed_cases=4, next_owner="GPT/user", A1_executed=False,
            new_policy_adopted=False, whole_RPE_or_T_qualified=False))
        print(json.dumps({"status":results["status"], "diagnostics":[
            {k:r[k] for k in ("id", "W_including_operator_diagnostic_spectral", "emitted_phase_hex")}
            for r in rows], "STOP":"MANDATORY_STOP"}))
    except Exception as e:
        save("RESULTS.json", dict(status="B1_UNQUALIFIED_STOP", completed_cases=rows,
            exception_type=type(e).__name__, exception=str(e), SOURCE=sources))
        save("STOP.json", dict(status="MANDATORY_STOP", reason="Fixed B1 qualification failed; no automatic expansion or retry"))
        raise


if __name__ == "__main__": main()
