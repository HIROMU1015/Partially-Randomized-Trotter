"""N5-B1: frozen saved-word, two-qubit emitter/consumer diagnostics only.

No synthesizer, guard, production provider, PF builder, compiler, price or RPE
runner is loaded. Unmodified function bodies are reused from the fixed commit.
The literal lookup below binds saved fixture words; it computes no certificate
or word and cannot service any key outside the ten approved inputs.
"""
from __future__ import annotations

import ast
from functools import lru_cache
import hashlib
import importlib.metadata as metadata
import inspect
import io
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
import time
from types import SimpleNamespace

import mpmath as mp
import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

ROOT = Path(__file__).resolve().parents[1]
BASE = "624646b9a4782a65e90561ea29df6beccea2a42c"
OUT = ROOT / "artifacts/pr_reference_n5b1_scalar/2026-11-08"
FIXED = "artifacts/pr_reference_n5b0_static/2026-11-08/NEXT_DIAGNOSTIC_INPUTS.json"
N3 = "artifacts/pr_reference_n3b1b0_runtime/2026-11-08/"
N4 = "artifacts/pr_reference_n4b_consumer/2026-11-08/"
SOURCES, FUNCTIONS = {}, []
TOL = 1e-10  # Reuse the old LOCAL diagnostic tolerance, never a strict bound.


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def blob(path):
    raw = subprocess.check_output(["git", "show", BASE + ":" + path], cwd=ROOT)
    SOURCES[path] = dict(sha256=sha(raw), bytes=len(raw))
    return raw


def read(path):
    return json.loads(blob(path))


def save(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True,
        ensure_ascii=False, allow_nan=False) + "\n")


def functions(path, names, namespace):
    raw = blob(path)
    tree = ast.parse(raw)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    for node in nodes:
        FUNCTIONS.append(dict(path=path, name=node.name,
            function_source_sha256=sha(ast.get_source_segment(raw.decode(), node).encode()),
            decorators_removed=[ast.unparse(d) for d in node.decorator_list]))
        node.decorator_list = []
    exec(compile(ast.Module(body=nodes, type_ignores=[]), BASE + ":" + path, "exec"), namespace)
    return namespace


def records(fixed):
    selected = []
    path = fixed["C0_representatives"][0]["archive"]
    with tarfile.open(fileobj=io.BytesIO(blob(path)), mode="r:xz") as tar:
        for spec in fixed["C0_representatives"]:
            assert spec["archive"] == path
            raw = tar.extractfile(spec["member"]).read()
            record = json.loads(raw)
            assert record["key"] == spec["record_key"]
            assert record["theta_hex"] == spec["theta_hex"]
            assert record["sequence_sha256"] == spec["sequence_sha256"]
            selected.append(dict(spec, sequence=record["sequence"],
                precision_exponent=record["precision_exponent"], record_sha256=sha(raw),
                binding_kind="saved_C0_record"))
    roles = read(N4 + "ROLE_WORD_CERTIFICATES.json")["rows"]
    for spec in fixed["existing_composites"]:
        row = next(r for r in roles if r["positive_literal_hex"] == spec["theta_hex"])
        assert row["positive"]["positive_sequence_sha256"] == spec["sequence_sha256"]
        selected.append(dict(spec, sequence=row["positive_sequence"],
            saved_certificate=row["positive"], binding_kind="saved_N4B_complete_composite"))
    assert len(selected) == 10 and len({r["theta_hex"] for r in selected}) == 10
    for r in selected:
        assert sha(r["sequence"].encode()) == r["sequence_sha256"]
        assert r["sequence"].count("W") == r["global_W"]
        assert r["global_W"] % 8 == r["W_residue"]
        assert set(r["sequence"]) <= set("HTtSXW")
    assert [r["W_residue"] for r in selected[:8]] == list(range(8))
    assert selected[-1]["global_W"] == 72
    return selected


def environment():
    old = read(N3 + "ENVIRONMENT.json")
    assert qiskit.__version__ == old["qiskit_version"] == "1.3.0"
    assert np.__version__ == old["numpy"] and mp.__version__ == old["mpmath"]
    assert metadata.version("pygridsynth") == old["pygridsynth"] == "2.0.0"
    site = Path(inspect.getfile(QuantumCircuit)).resolve().parents[2]
    matches = []
    for r in old["python_source_matches"]:
        if "/qpy/" in r["path"] or r["path"].endswith("parameterexpression.py"):
            continue  # Neither path is used by these numeric, in-memory fixtures.
        raw = (site / r["path"]).read_bytes()
        actual = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        assert actual == r["git_blob_sha"]
        matches.append(dict(path=r["path"], git_blob_sha=actual, matched=True))
    binary = old["acceleration_binary"]
    assert sha(Path(binary["path"]).read_bytes()) == binary["sha256"]
    return dict(python=sys.version, executable=sys.executable,
        qiskit=qiskit.__version__, numpy=np.__version__, mpmath=mp.__version__,
        pygridsynth=metadata.version("pygridsynth"), hostname=platform.node(),
        affinity=len(os.sched_getaffinity(0)),
        memory_available_kib=next(s for s in Path("/proc/meminfo").read_text().splitlines()
                                  if s.startswith("MemAvailable:")),
        pinned_source_matches=matches, same_acceleration_binary_as_N3=binary,
        reused_runtime_probes=N3 + "ENVIRONMENT.json; C1-C6 results not rerun",
        execution_model="Named ideal logical gates; binary64 diagnostics are not certificates",
        shared_environment_changes=False, new_QPY_roundtrips=0,
        acceleration="Cache exact written-word matrices and repeated named-stream Operators; 4x4 kernels only, no JIT/cache files")


def real_float(x):
    n, d = float(x).as_integer_ratio()
    return mp.mpf(n) / d


def scalar(phi):
    return complex(mp.exp(mp.j * phi))


def fixture(theta, phase_label, position, phase_value):
    q = QuantumCircuit(2, 3, name="frozen_joint_scalar_fixture")
    # Preserve the saved negative-subnormal setter path, including stored tau.
    q.global_phase = (-math.ulp(0.0) if phase_label == "tau_via_negative_subnormal"
                      else phase_value(phase_label))
    q.reset(1)
    q.sx(0)
    q.sx(1)
    if position == "before_CX":
        q.rz(theta, 1)
    q.cx(0, 1)
    if position == "after_CX":
        q.rz(theta, 1)
    q.sx(1)
    q.measure(1, 2)
    q.measure(0, 0)
    return q


def body(q):
    out = QuantumCircuit(2)
    for inst in q.data:
        if inst.operation.name not in ("reset", "measure", "barrier"):
            out.append(inst.operation, [q.find_bit(z).index for z in inst.qubits], copy=False)
    return out


def main():
    started = time.monotonic()
    assert not OUT.exists(), "Use a fresh output area; do not overwrite prior results"
    fixed = read(FIXED)
    assert fixed["signs"] == [1, -1]
    assert fixed["prefix_phase_labels_from_existing_N3"] == [
        "zero", "one_tenth", "prev_tau", "tau_via_negative_subnormal"]
    assert fixed["local_positions"] == ["before_CX", "after_CX"]
    selected = records(fixed)
    env = environment()
    old_tests = read(N3 + "PHASE_TESTS.json")
    old_local = read(N4 + "LOCAL_CORRECTNESS.json")
    old_whole = read(N4 + "SAVED_WHOLE_INTEGRATION.json")
    assert old_tests["status"] == "PASS" and old_local["status"] == "LOCAL_CORRECTNESS_PASS"
    assert all(r["passed"] for r in old_local["cases"] + old_whole["rows"])
    save("INPUTS.json", dict(base_commit=BASE, diagnostic_input=FIXED,
        diagnostic_input_sha256=SOURCES[FIXED]["sha256"],
        fixed_diagnostic_inputs=fixed,
        fixture_template="2q/3c: prefix phase; reset(1); SX(0); SX(1); [RZ(theta,1) before CX]; CX(0,1); [RZ(theta,1) after CX]; SX(1); measure(1,2); measure(0,0)",
        fixture_input="|00>; prefix reset(1) leaves it unchanged",
        X_Y_observables="I(q0) tensor X(q1), I(q0) tensor Y(q1), Qiskit little endian",
        local_fixture_cases=160, numerical_diagnostic_tolerance=TOL,
        comparison="Saved logical word versus emitted word; NOT word versus RZ(theta)",
        no_phase_fit=True, no_new_angles_words_precision_or_scientific_policy=True))
    save("ENVIRONMENT.json", env)
    # Reuse fixed AST bodies only; never execute old runner/module main.
    lower = functions("src/trottertracks/track_c2_b_verify.py", ["lower_native"],
        dict(QuantumCircuit=QuantumCircuit, math=math))["lower_native"]
    exact_ns = functions("src/trottertracks/track_c_h4_synthesis.py", ["exact_word_matrix"], dict(mp=mp))
    local_ns = functions("scripts/verify_pr_reference_n3b1b0_runtime.py", ["matrix", "phase_value"],
        dict(exact_word_matrix=exact_ns["exact_word_matrix"], np=np, QuantumCircuit=QuantumCircuit,
             math=math, P=math.pi/4, TAU=2*math.pi))
    mp.mp.dps = mp.iv.dps = 100
    matrix_ns = dict(local_ns)
    matrix_ns["apply_batch"] = functions("src/trottertracks/pr_reference_ct1.py", ["apply_batch"],
                                        dict(np=np))["apply_batch"]
    binding = {r["theta_hex"]: r for r in selected}

    def saved_literal(theta):
        r = binding[abs(float(theta)).hex()]  # Lookup only, no generation/rounding policy.
        return SimpleNamespace(positive_sequence=r["sequence"],
            global_W_signed=(-1 if theta < 0 else 1) * r["global_W"],
            summary=lambda: dict(kind="saved_fixture_binding_only", theta_hex=float(theta).hex(),
                sequence_sha256=r["sequence_sha256"], guard_evaluated=False))

    consumer_ns = functions("src/trottertracks/pr_reference_common_word_consumer.py", ["consume_native"],
        dict(math=math, QuantumCircuit=QuantumCircuit,
             UNITARY=frozenset(("rz", "sx", "x", "cx")),
             FENCES=frozenset(("reset", "measure", "barrier"))))
    ns = functions("scripts/validate_pr_reference_n4b_consumer.py",
        ["ins", "expected_stream", "check_consumer", "word_matrix", "compressed"],
        dict(PROVIDER=SimpleNamespace(provide=saved_literal), NS=dict(lower_native=lower, **matrix_ns),
             CONSUMER=SimpleNamespace(**consumer_ns), math=math, sha=sha, json=json))
    # Restore the original word_matrix caching after isolated AST loading.
    ns["word_matrix"] = lru_cache(None)(ns["word_matrix"])
    named_ops = {}
    rows = []
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    observables = dict(X=np.kron(X, np.eye(2)), Y=np.kron(Y, np.eye(2)))
    for rec in selected:
        for sign in fixed["signs"]:
            theta = sign * float.fromhex(rec["theta_hex"])
            for phase_label in fixed["prefix_phase_labels_from_existing_N3"]:
                for position in fixed["local_positions"]:
                    label = ":".join((rec["theta_hex"], str(sign), phase_label, position))
                    native = fixture(theta, phase_label, position, local_ns["phase_value"])
                    out, stream = ns["check_consumer"](native, label)
                    emitted_body = body(out)
                    key = stream["output_stream_sha256"]
                    if key not in named_ops:
                        # global_phase is deliberately read BEFORE any Operator conversion.
                        # The matrix body has phase zero; include the stored scalar explicitly.
                        named_ops[key] = Operator(emitted_body).data
                    E = scalar(real_float(out.global_phase)) * named_ops[key]
                    L = scalar(real_float(native.global_phase)) * ns["compressed"](body(native), True)
                    signed_W = (-1 if theta < 0 else 1) * rec["global_W"]
                    chi = real_float(out.global_phase) - real_float(native.global_phase) - signed_W * mp.pi/4
                    predicted = scalar(chi)  # SOURCE-derived, not inferred from E or L.
                    residual = float(np.linalg.norm(E - predicted * L, 2))
                    unadjusted = float(np.linalg.norm(E - L, 2))
                    xy = {}
                    for axis, M in observables.items():
                        e = np.vdot(E[:, 0], M @ E[:, 0])
                        l = np.vdot(L[:, 0], M @ L[:, 0])
                        xy[axis] = dict(emitted_real=float(e.real), logical_real=float(l.real),
                            observed_absolute_difference=float(abs(e-l)),
                            imaginary_part_diagnostic=float(max(abs(e.imag), abs(l.imag))))
                    passed = residual < TOL and all(v["observed_absolute_difference"] < TOL for v in xy.values())
                    row = dict(label=label, theta_hex=theta.hex(), word_sha256=rec["sequence_sha256"],
                        sign=sign, W_signed=signed_W, W_residue=signed_W % 8,
                        phase_label=phase_label, position=position,
                        prefix_stored_phase_hex=float(native.global_phase).hex(),
                        emitted_stored_phase_hex=float(out.global_phase).hex(),
                        source_scalar=dict(real=predicted.real, imag=predicted.imag,
                            formula="exp(i*(stored_outer_phase - stored_native_phase - signed_W*pi/4))"),
                        observed_source_factored_operator_residual=residual,
                        observed_unadjusted_operator_difference=unadjusted,
                        X_Y=xy, stream=stream, passed=passed,
                        diagnostic_is_strict_certificate=False)
                    rows.append(row)
                    if not passed:
                        save("LOCAL_RUNTIME_CHECKS.json", dict(status="RELATIVE_PHASE_OR_GATE_MISMATCH",
                            rows=rows, no_phase_fit=True))
                        raise AssertionError("SOURCE-predicted scalar/gate diagnostic mismatch: " + label)
    assert len(rows) == 160
    assert len({r["label"] for r in rows}) == 160
    save("LOCAL_RUNTIME_CHECKS.json", dict(status="LOCAL_RUNTIME_COMPATIBLE", rows=rows,
        cases=160, all_passed=True, named_stream_Operator_cache_entries=len(named_ops),
        maximum_observed_source_factored_operator_residual=max(r["observed_source_factored_operator_residual"] for r in rows),
        maximum_observed_X_Y_difference=max(v["observed_absolute_difference"] for r in rows for v in r["X_Y"].values()),
        diagnostic_tolerance=TOL, diagnostics_are_strict_certificates=False,
        no_phase_fit=True, no_RZ_target_error_evaluated=True,
        whole_H4_or_m17_simulations=0, elapsed_seconds=time.monotonic()-started))
    save("SOURCE_INPUTS.json", dict(base_commit=BASE, files=SOURCES,
        functions_reused_unmodified=FUNCTIONS,
        helper="scripts/verify_pr_reference_n5b1_scalar.py", helper_sha256=sha(Path(__file__).read_bytes()),
        saved_words=[{k:v for k,v in r.items() if k not in ("sequence", "saved_certificate")} for r in selected],
        reuse=dict(N3_standalone_W_word_carrier_tests="quoted PHASE_TESTS, not rerun",
            N4B_standalone_composite_and_fence_tests="quoted LOCAL_CORRECTNESS, not rerun",
            N4B_saved_whole_projections="quoted only; NOT repeated or transferred to m17",
            new_fixture_reason="The joint single-word/CX/prefix/readout template and SOURCE-derived scalar comparison are absent from old fixtures"),
        fixture_binding_is_scientific_provider=False, old_SOURCE_or_certificate_changes=False))
    save("CHECKS.json", dict(status="PASS_LIMITED_TECHNICAL_SCOPE", fixed_word_composites=10,
        joint_fixture_cases=160, named_gate_wire_and_fence_checks=True,
        positive_and_complete_actual_inverse=True, source_predicted_scalar=True,
        new_post_synthesis_control=False, strict_guard_calls=0, synthesis_calls=0,
        new_precision_policy=False, native_PF_build_compile_calls=0,
        RPE_reaccounting_calls=0, new_T_price_calls=0, physical_simulation_sampling_calls=0,
        evidence_status=["JOINT_SCALAR_FACTORIZATION_SOURCE_SUPPORTED", "LOCAL_RUNTIME_COMPATIBLE",
                         "CONDITIONAL_ON_FULL_CONSUMER_PATH", "M17_RPE_UNQUALIFIED"],
        elapsed_seconds=time.monotonic()-started))
    save("STOP.json", dict(status="MANDATORY_STOP", reason="Approved frozen N5-B1 local scope complete",
        m17_RPE="UNQUALIFIED", next_owner="GPT/user", automatic_precision_provider_RPE_or_T_work=False))
    print(json.dumps(dict(status="LOCAL_RUNTIME_COMPATIBLE", cases=len(rows),
        maximum_residual=max(r["observed_source_factored_operator_residual"] for r in rows),
        elapsed_seconds=time.monotonic()-started)))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        if not (OUT / "STOP.json").exists():
            save("STOP.json", dict(status="MANDATORY_STOP", reason=type(exc).__name__ + ": " + str(exc),
                m17_RPE="UNQUALIFIED", next_owner="GPT/user"))
        raise
