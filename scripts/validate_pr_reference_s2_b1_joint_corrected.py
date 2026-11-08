"""Complete the SAME B1 joint fixtures after a reference-only harness repair.

The initial runner/inputs/results/STOP are immutable. Reuse its two successful
standalone outputs, not new word selection or repeats of completed fixtures.
"""
from __future__ import annotations
import importlib.util
import json
import math
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("b1_initial", ROOT / "scripts/validate_pr_reference_s2_b1_interface.py")
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
INITIAL = b.OUT
OUT = INITIAL / "verified_final"


def restored_standalone(row):
    q = b.QuantumCircuit(1, global_phase=float.fromhex(row["emitted_phase_hex"]))
    for name, qs, cs, params in row["output_named_instruction_stream"]:
        assert not cs and name in ("h", "s", "sdg", "t", "tdg", "x")
        getattr(q, name)(*params, *qs)
    return q


def reference_joint(native, standalone):
    q = b.QuantumCircuit(native.qubits, native.clbits)
    for reg in (*native.qregs, *native.cregs): q.add_register(reg)
    for j in range(3): q.reset(j)
    # Correct the initial reference's native SX: lower_native emits H,S,H.
    q.h(0); q.s(0); q.h(0); q.cx(0, 2)
    q.compose(standalone, qubits=[2], inplace=True)
    q.cx(0, 2); q.h(0); q.s(0); q.h(0); q.measure(0, 0)
    return q


def main():
    started = time.monotonic()
    assert not OUT.exists(), "Preserve corrected outputs"
    fixed = json.loads((INITIAL / "INPUTS.json").read_text())
    first = json.loads((INITIAL / "RESULTS.json").read_text())
    assert first["status"] == "B1_UNQUALIFIED_STOP" and first["exception_type"] == "AssertionError"
    assert (INITIAL / "STOP.json").exists()
    assert fixed["base_commit"] == b.BASE and fixed["fixed_fixtures"] == b.fixture_definitions()
    assert [r["id"] for r in first["completed_cases"]] == ["standalone_positive", "standalone_negative"]
    for path, digest in fixed["source"]["new_SOURCE"].items():
        assert b.sha((ROOT / path).read_bytes()) == digest
    record = fixed["original_record_unchanged"]
    provider = b.SingleSavedRecordOverlay(record)
    ns = dict(QuantumCircuit=b.QuantumCircuit, math=math)
    lower = b.function("src/trottertracks/track_c2_b_verify.py", "lower_native", ns)
    exact_matrix = b.function("src/trottertracks/track_c_h4_synthesis.py", "exact_word_matrix", dict(mp=b.mp))
    ns = dict(PROVIDER=provider)
    ins = b.function("scripts/validate_pr_reference_n4b_consumer.py", "ins", ns)
    expected_stream = b.function("scripts/validate_pr_reference_n4b_consumer.py", "expected_stream", ns)
    b.mp.mp.dps = b.mp.iv.dps = 100
    OUT.mkdir()
    correction_inputs = dict(base_commit=b.BASE, unchanged_fixed_fixtures=fixed["fixed_fixtures"],
        original_record_unchanged=record, numerical_operator_diagnostic_tolerance=b.TOLERANCE,
        preserved_initial_files={name:b.sha((INITIAL/name).read_bytes())
            for name in ("INPUTS.json", "RESULTS.json", "STOP.json")},
        correction_SOURCE_sha256=b.sha(Path(__file__).read_bytes()),
        initial_SOURCE=fixed["source"], reused_functions=b.FUNCTIONS,
        repair="Independent reference SX -> unchanged lower_native H,S,H named stream. Consumer/native inputs/word unchanged.",
        repeated_successful_standalone_fixture_evaluations=0,
        remaining_fixed_fixture_ids=[f["id"] for f in fixed["fixed_fixtures"][2:]])
    b.OUT = OUT
    b.save("INPUTS.json", correction_inputs)
    rows = list(first["completed_cases"])
    for r in rows: r["evidence_location"] = "Preserved initial RESULTS.json; not rerun"
    try:
        # Reuse completed signed emissions by their saved named streams and phase.
        standalone = {r["sign"]:restored_standalone(r) for r in rows}
        im = exact_matrix(record["sequence"])
        word = b.np.array([[complex(float(z.real.mid), float(z.imag.mid)) for z in row] for row in im])
        for f in fixed["fixed_fixtures"][2:]:
            native = b.build_native(f)
            out, audit = b.consume_native(native, provider, lower)
            independent = reference_joint(native, standalone[f["sign"]])
            actual_stream = ins(out)
            assert actual_stream == ins(independent) == list(expected_stream(native))
            assert float(out.global_phase).hex() == float(independent.global_phase).hex()
            assert out.qubits == native.qubits and out.clbits == native.clbits
            assert out.qregs == native.qregs and out.cregs == native.cregs
            fences = lambda q: [(n, qs, cs) for n, qs, cs, _ in ins(q) if n in ("reset", "measure")]
            assert fences(native) == fences(out)
            assert audit["copying_compose"] == audit["new_QPY_roundtrips"] == audit["new_controls"] == 0
            assert len(audit["segments"]) == audit["extra_segment_joins"] == 1
            assert all(s["segment_input_phase_hex"] == 0.0.hex() for s in audit["segments"])
            local = word if f["sign"] > 0 else word.conj().T
            prefix = b.QuantumCircuit(3); prefix.sx(0); prefix.cx(0, 2)
            suffix = b.QuantumCircuit(3); suffix.cx(0, 2); suffix.sx(0)
            expected = b.Operator(suffix).data @ b.np.kron(local, b.np.eye(4)) @ b.Operator(prefix).data
            mat = b.Operator(b.unitary_body(out)).data
            mismatch = float(b.np.linalg.norm(mat - expected, 2))
            reference_mismatch = float(b.np.linalg.norm(mat - b.Operator(b.unitary_body(independent)).data, 2))
            assert max(mismatch, reference_mismatch) < b.TOLERANCE
            phase = float(out.global_phase)
            assert phase.hex() == ((7 if f["sign"] > 0 else 1)*(math.pi/4)).hex()
            rows.append(dict(id=f["id"], status="PASS", sign=f["sign"],
                named_gates_wires_and_fences_match=True,
                same_registers=True, no_arm_or_role_selection_argument=True,
                emitted_phase_hex=phase.hex(), integer_W_phase_residue=7 if f["sign"] > 0 else 1,
                positive_record_spec_retained=record["spec"],
                W_including_operator_diagnostic_spectral=mismatch,
                independent_emitter_consumer_operator_diagnostic_spectral=reference_mismatch,
                numerical_diagnostics_are_strict_certificates=False,
                output_named_instruction_stream=actual_stream, audit=audit))
        result = dict(status="B1_FOUR_LOCAL_FIXTURES_CORRECTNESS_PASS", cases=rows,
            exact_complete_word_adjoint_identity=first["completed_cases"][0]["status"] == "PASS",
            exact_adjoint_evidence="Initial runner exact identity assertion precedes its two completed standalone cases",
            saved_guard_quoted_not_reexecuted=record["strict_error_upper"],
            original_backend_exponent=-13, transport_exponent=-11,
            same_arm_independent_selector=True, fixed_record_only=True,
            qualification="One saved record/four artificial fixtures; no full SOURCE/global-phase, word-policy, RPE or T-cost qualification",
            strict_emitter="Existing named-gate/IEEE phase-path conditions retained; diagnostics are not newly computed strict guards",
            repair=correction_inputs["repair"], initial_evidence_preserved=True,
            calls_cumulative=dict(distinct_logical_fixture_configurations=4,
                successful_standalone_fixtures_reused=2, corrected_joint_fixtures=2,
                initial_joint_attempt_stopped_at_reference_stream_comparison=1,
                exact_written_word_reference_evaluations=2,
                new_synthesis=0, strict_guard=0, native_compile=0, RZ_target_guard=0,
                new_T_price_or_aggregate=0, A1_basis_defect=0, S1_S2_reexecution=0,
                m17_or_18round_RPE=0),
            SOURCE=correction_inputs, wall_seconds=time.monotonic()-started)
        b.save("RESULTS.json", result)
        b.save("STOP.json", dict(status="MANDATORY_STOP", reason="B1 fixed four fixtures completed after reference-only technical repair",
            initial_STOP_preserved=True, next_owner="GPT/user", A1_executed=False,
            new_policy_adopted=False, whole_RPE_or_T_qualified=False))
        print(json.dumps(dict(status=result["status"], cases=[
            {k:r[k] for k in ("id", "W_including_operator_diagnostic_spectral", "emitted_phase_hex")}
            for r in rows], STOP="MANDATORY_STOP")))
    except Exception as e:
        b.save("RESULTS.json", dict(status="B1_UNQUALIFIED_STOP", completed_cases=rows,
            exception_type=type(e).__name__, exception=str(e), SOURCE=correction_inputs))
        b.save("STOP.json", dict(status="MANDATORY_STOP", reason="Corrected fixed B1 qualification failed"))
        raise


if __name__ == "__main__": main()
