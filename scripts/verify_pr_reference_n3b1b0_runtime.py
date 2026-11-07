"""N3-B1b-0 local technical tests. No acquisition, guard, PF or price runner.

Reuse unchanged function ASTs to avoid executing unrelated scientific module
initializers. Artificial record carriers are not certified RZ approximations.
"""
from __future__ import annotations

import ast
import base64
import ctypes
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
import sysconfig
import tarfile
import time
from fractions import Fraction as F

import mpmath as mp
import numpy as np
import qiskit
import qiskit._accelerate as accelerate
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from pygridsynth.domega_unitary import DOmegaUnitary as Exact

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/pr_reference_n3b1b0_runtime/2026-11-08"
BASE = "4abf269d40b14636ffe0f7e2607aa293d4948b0a"
INPUT = json.loads((OUT / "TEST_INPUTS.json").read_text())
U = F(1, 2**53)
P = math.pi / 4
TAU = 2 * math.pi
PROVENANCE = []


def blob(path):
    return subprocess.check_output(["git", "show", BASE + ":" + path], cwd=ROOT)


def load_function(path, name, namespace):
    data = blob(path)
    assert (ROOT / path).read_bytes() == data
    node = next(n for n in ast.parse(data).body
                if isinstance(n, ast.FunctionDef) and n.name == name)
    assert not node.decorator_list
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(ROOT / path),
                 "exec"), namespace)
    PROVENANCE.append(dict(path=path, function=name,
        file_sha256=hashlib.sha256(data).hexdigest(),
        function_source_sha256=hashlib.sha256(
            ast.get_source_segment(data.decode(), node).encode()).hexdigest()))
    return namespace[name]


lower_native = load_function("src/trottertracks/track_c2_b_verify.py",
    "lower_native", dict(QuantumCircuit=QuantumCircuit, math=math))
exact_word_matrix = load_function("src/trottertracks/track_c_h4_synthesis.py",
    "exact_word_matrix", dict(mp=mp))
ct1 = ast.parse(blob("src/trottertracks/pr_reference_ct1.py"))
wm = next(n for n in ct1.body if isinstance(n, ast.FunctionDef)
          and n.name == "word_matrices")
adj_map = ast.literal_eval(next(n.value for n in wm.body
    if isinstance(n, ast.Assign)
    and any(isinstance(t, ast.Name) and t.id == "inverse" for t in n.targets)))


def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def emitter(word, sign=1, theta=1.0, phase=0.0):
    """A local test fixture, never a basic-key dictionary/provider."""
    qc = QuantumCircuit(1, global_phase=phase)
    qc.rz(sign * theta, 0)
    key = abs(theta).hex()
    return lower_native(qc, [dict(role="fixture", angle=key, exponent=-11)],
        {(key, -11): dict(sequence=word)}, "fixture")


def matrix(word):
    # The reused exact cyclotomic/interval routine is evaluated only for
    # written-word comparison, never against an RZ target or a strict guard.
    im = exact_word_matrix(word)
    return np.array([[complex(float(z.real.mid), float(z.imag.mid))
                      for z in row] for row in im])


def same_float(a, b):
    # Signed zeros have the same exact phase; retain their bits in the log.
    return a.hex() == b.hex() or a == b == 0.0


def fhex(x):
    return float(x).hex()


def circle_binary(a, b):
    d = abs(F(a) - F(b))
    return min(d, abs(d - F(TAU)), abs(d + F(TAU)))


def phase_value(label):
    if label == "zero":
        return 0.0
    if label == "P":
        return P
    if label == "7P":
        return 7 * P
    if label == "one_tenth":
        return 0.1
    if label == "prev_tau":
        return math.nextafter(TAU, 0.0)
    if label == "tau_via_negative_subnormal":
        q = QuantumCircuit(1)
        q.global_phase = -math.ulp(0.0)
        assert q.global_phase == TAU
        return float(q.global_phase)
    if label == "small_positive":
        return 2.0**-50
    if label == "prev_P":
        return math.nextafter(P, 0.0)
    raise ValueError(label)


def runtime_checks():
    old = json.loads(blob("artifacts/pr_reference_n3b1a_emitter_static/"
                          "2026-11-08/SOURCE_REFERENCES.json"))
    site = Path(inspect.getfile(QuantumCircuit)).resolve().parents[2]
    matches = []
    diagnostic_sources = [
        dict(path="qiskit/quantum_info/operators/operator.py",
             sha="d549a130ae3ab8d4e2995215846714b187cc5123"),
        dict(path="qiskit/converters/circuit_to_instruction.py",
             sha="db314eee267eca716a5ff7d7078618ec85988cec"),
    ]
    for row in old["qiskit"]["python_source_byte_matches"] + diagnostic_sources:
        raw = (site / row["path"]).read_bytes()
        sha = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        matches.append(dict(path=row["path"], git_blob_sha=sha,
                            matches_pinned_source=sha == row["sha"]))
    assert all(x["matches_pinned_source"] for x in matches)
    assert qiskit.__version__ == "1.3.0"
    libm = ctypes.CDLL("libm.so.6")
    libm.fegetround.restype = ctypes.c_int
    rounding = libm.fegetround()  # Read only: never call fesetround.
    assert rounding == 0  # glibc x86-64 FE_TONEAREST
    assert (sys.float_info.radix, sys.float_info.mant_dig) == (2, 53)
    assert 1.0 + 2.0**-53 == 1.0
    assert (1.0 + 2.0**-52) + 2.0**-53 == 1.0 + 2.0**-51
    assert math.pi.hex() == "0x1.921fb54442d18p+1"
    assert P.hex() == "0x1.921fb54442d18p-1"
    period = []
    q = QuantumCircuit(1)
    for x in [math.nextafter(TAU, 0.0), TAU, math.nextafter(TAU, math.inf)]:
        q._data.global_phase = x
        got = float(q._data.global_phase)
        want = math.fmod(x, TAU)
        period.append(dict(input_hex=x.hex(), output_hex=got.hex(),
                           expected_hex=want.hex(), passed=same_float(got, want)))
    assert all(x["passed"] for x in period)
    ext = Path(accelerate.__file__).resolve()
    binary = ext.read_bytes()
    digest = base64.urlsafe_b64encode(hashlib.sha256(binary).digest()).decode().rstrip("=")
    dist = metadata.distribution("qiskit")
    item = next(x for x in dist.files
                if Path(dist.locate_file(x)).resolve() == ext)
    assert item.hash.mode == "sha256" and item.hash.value == digest
    flags = str(sysconfig.get_config_var("CFLAGS"))
    unsafe = [x for x in ("-ffast-math", "-Ofast", "-funsafe-math-optimizations")
              if x in flags]
    assert not unsafe
    return dict(python=sys.version, executable=sys.executable,
        qiskit_version=qiskit.__version__, pygridsynth=metadata.version("pygridsynth"),
        mpmath=mp.__version__, numpy=np.__version__, machine=platform.machine(),
        hostname=platform.node(), affinity=len(os.sched_getaffinity(0)),
        memory_available_kib=next(x for x in Path("/proc/meminfo").read_text().splitlines()
                                  if x.startswith("MemAvailable:")),
        float_info=dict(radix=sys.float_info.radix, mant_dig=sys.float_info.mant_dig,
                        rounds=sys.float_info.rounds), fegetround=rounding,
        tie_to_even_probes_passed=True, pi_hex=math.pi.hex(), P_hex=P.hex(),
        tau_hex=TAU.hex(), rust_period_runtime_probes=period,
        rust_PI_evidence="Pinned Rust SOURCE plus compiled setter wrap at tau and both adjacent binary64 values; PI is not independently exported.",
        circuit_data_type=str(type(q._data)), python_source_matches=matches,
        acceleration_binary=dict(path=str(ext), sha256=hashlib.sha256(binary).hexdigest(),
            installed_distribution_RECORD_matches=True),
        python_CFLAGS=flags, unsafe_python_flags_detected=unsafe,
        rust_build_flags="Not exposed by installed wheel; no claim of machine-code formal verification.",
        qpy=INPUT["qpy"],
        mathematical_model="Standard IEEE f64 meaning, named ideal Clifford+T gates. SOURCE plus finite runtime conformance, not proof of every binary machine instruction.")


def w_tests():
    rows = []
    for k in INPUT["w_counts"]:
        q = emitter("W" * k)
        expected = (k % 8) * P
        inv = q.inverse()
        neg = emitter("W" * k, -1)
        target_neg = ((-k) % 8) * P
        reinit = QuantumCircuit(1, global_phase=q.global_phase)
        direct = QuantumCircuit(1, global_phase=-(k % 8) * P)
        joined = q.compose(emitter("W"))
        target_join = ((k + 1) % 8) * P
        values = [same_float(q.global_phase, expected),
                  same_float(inv.global_phase, target_neg),
                  same_float(neg.global_phase, target_neg),
                  same_float(direct.global_phase, target_neg),
                  same_float(reinit.global_phase, expected),
                  same_float(joined.global_phase, target_join)]
        rows.append(dict(W_count=k, residue=k % 8,
            expected_hex=expected.hex(), emitted_hex=fhex(q.global_phase),
            inverse_hex=fhex(inv.global_phase), negative_emission_hex=fhex(neg.global_phase),
            negative_setter_hex=fhex(direct.global_phase),
            reapplied_hex=fhex(reinit.global_phase), compose_W_hex=fhex(joined.global_phase),
            passed=all(values)))
    return rows


def carrier_tests():
    rows = []
    # Positive arithmetic model uses exact Fraction inputs and an exact
    # binary-period reference; true-period differences remain in the proof.
    for a in INPUT["phase_fixtures"]:
        for b in INPUT["phase_fixtures"]:
            x = phase_value(a)
            y = phase_value(b)
            left = QuantumCircuit(1)
            right = QuantumCircuit(1)
            # To retain tau, use the documented negative-tiny setter path.
            left.global_phase = -math.ulp(0.0) if a == "tau_via_negative_subnormal" else x
            right.global_phase = -math.ulp(0.0) if b == "tau_via_negative_subnormal" else y
            inv = float(left.inverse().global_phase)
            inv_err = circle_binary(inv, (-F(x)) % F(TAU))
            left.compose(right, inplace=True)
            z = float(left.global_phase)
            err = circle_binary(z, (F(x) + F(y)) % F(TAU))
            reset = float(QuantumCircuit(1, global_phase=x).global_phase)
            reset_expected = 0.0 if x == TAU else x
            ok = (err < 16 * U and inv_err < 16 * U
                  and same_float(reset, reset_expected))
            rows.append(dict(left=a, right=b, left_hex=x.hex(), right_hex=y.hex(),
                composed_hex=z.hex(), binary_period_rounding_error=str(err),
                inverse_hex=inv.hex(), inverse_binary_rounding_error=str(inv_err),
                reset_hex=reset.hex(), reset_expected_hex=reset_expected.hex(),
                passed=ok))
    return rows


def word_test(label, word, theta=1.0):
    adj = "".join(adj_map[g] for g in reversed(word))
    assert Exact.from_gates((word + adj).replace("t", "TTTTTTT")) == Exact.identity()
    expected = matrix(word)
    pos_names = [dict(H="h", S="s", T="t", t="tdg", X="x")[c]
                 for c in reversed(word) if c != "W"]
    inv_names = dict(h="h", s="sdg", sdg="s", t="tdg", tdg="t", x="x")
    rows = []
    for sign in INPUT["signs"]:
        q = emitter(word, sign, theta)
        names = [x.operation.name for x in q.data]
        wanted = pos_names if sign == 1 else [inv_names[n] for n in reversed(pos_names)]
        op = Operator(q).data
        wanted_matrix = expected if sign == 1 else expected.conj().T
        mismatch = float(np.linalg.norm(op - wanted_matrix, 2))
        rows.append(dict(sign=sign, emitted_phase_hex=fhex(q.global_phase),
            gate_order_and_named_inverses_match=names == wanted,
            observed_strict_operator_mismatch=mismatch,
            passed=names == wanted and mismatch < INPUT["numerical_operator_diagnostic_tolerance"]))
    return dict(label=label, word_sha256=hashlib.sha256(word.encode()).hexdigest(),
        exact_whole_adjoint_identity=True, fixtures=rows)


def composite_tests():
    chunks = [INPUT["artificial_chunk_pattern"][i % len(INPUT["artificial_chunk_pattern"])]
              for i in range(INPUT["max_chunks"])]
    word = "".join(chunks)
    expected = matrix(word)
    rows = []
    for sign in INPUT["signs"]:
        direct = emitter(word, sign)
        carrier = QuantumCircuit(1)
        for chunk in reversed(chunks):
            carrier.compose(emitter(chunk), inplace=True)
        if sign < 0:
            carrier = carrier.inverse()
        fixture = QuantumCircuit(2, global_phase=0.1)
        fixture.x(0)
        fixture.cx(0, 1)  # A pre-existing local fixture, not control of a word.
        prefix = Operator(fixture).data
        fixture.compose(carrier, qubits=[1], inplace=True)
        joint = fixture
        target = expected if sign > 0 else expected.conj().T
        mismatch = float(np.linalg.norm(Operator(carrier).data - target, 2))
        joint_mismatch = float(np.linalg.norm(Operator(joint).data
                                  - np.kron(target, np.eye(2)) @ prefix, 2))
        whole_mismatch = float(np.linalg.norm(Operator(direct).data - target, 2))
        rows.append(dict(sign=sign, chunks=len(chunks),
            separate_carrier_mismatch=mismatch, single_emission_mismatch=whole_mismatch,
            two_qubit_strict_embedding_mismatch=joint_mismatch,
            global_phase_hex=fhex(carrier.global_phase),
            passed=max(mismatch, joint_mismatch, whole_mismatch)
                < INPUT["numerical_operator_diagnostic_tolerance"]))
    # A noncommuting two-chunk counterexample ensures the test distinguishes
    # whole-product adjoint from chunkwise adjoints in the original order.
    a, b = INPUT["artificial_chunk_pattern"][:2]
    whole = matrix(a + b).conj().T
    wrong = matrix(a).conj().T @ matrix(b).conj().T
    witness = float(np.linalg.norm(whole - wrong, 2))
    assert witness > 1e-3
    return dict(fixtures=rows, whole_word=word, wrong_chunk_adjoint_witness=witness,
        bookkeeping=dict(positive_chunk_carriers=43, zero_phase_local_joins=43,
            nontrivial_chunk_joins=43, final_join=1, whole_inverse_at_most=1,
            additional_nonzero_phase_settings=1, QPY_reloads=0),
        note="Each zero-phase emitter join is exact on the local 8-state lattice. Forty-three 64u charges include each local W conversion and its nontrivial carrier join; the final join is charged separately.")


def main():
    start = time.monotonic()
    mp.mp.dps = mp.iv.dps = 100
    env = runtime_checks()
    w = w_tests()
    carriers = carrier_tests()
    words = [word_test("artificial_" + str(i), s, 0.0 if not s else 1.0)
             for i, s in enumerate(INPUT["artificial_words"])]
    archive = blob(INPUT["word_archive"])
    saved = []
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:xz") as tar:
        for member in INPUT["saved_members"]:
            raw = tar.extractfile(member).read()
            r = json.loads(raw)
            assert hashlib.sha256(r["sequence"].encode()).hexdigest() == r["sequence_sha256"]
            words.append(word_test(member, r["sequence"], float.fromhex(r["theta_hex"])))
            saved.append(dict(member=member, record_sha256=hashlib.sha256(raw).hexdigest(),
                key=r["key"], sequence_sha256=r["sequence_sha256"],
                strict_guard="Saved record only: not reexecuted"))
    composite = composite_tests()
    copy_source = QuantumCircuit(1, global_phase=-math.ulp(0.0))
    copy_dest = copy_source.copy()
    assert copy_source.global_phase == TAU and copy_dest.global_phase == 0.0
    copy_boundary = dict(before_hex=fhex(copy_source.global_phase),
        after_copy_hex=fhex(copy_dest.global_phase),
        observed_strict_operator_difference=float(np.linalg.norm(
            Operator(copy_source).data - Operator(copy_dest).data, 2)),
        direct_stored_phase_scalar_diagnostic=float(abs(
            np.exp(1j * float(copy_source.global_phase)) - 1)),
        Operator_conversion_note="Operator(circuit) calls to_instruction; circuit_to_instruction reassigns numeric global_phase through the setter. Its zero difference is not evidence that exp(i*tau)=1 exactly.",
        passed=True,
        source="QuantumCircuit.copy -> copy_empty_like plus CircuitData.copy/new -> set_global_phase; retained data is renormalized.",
        qualification="Separate diagnostic. Copying compose requires its reinitializations to be charged; the certified composite path uses inplace=True.")
    bound = ((43 + 1) * 64 + 32 + 2 * 16) * U
    budget = json.loads(blob("artifacts/pr_reference_n3b0_numeric/2026-11-08/ERROR_BUDGET.json"))
    residual = F(budget["emitter_remaining"]["lower"])
    bound_ok = bound < F(1, 2**41) < residual
    all_pass = (all(x["passed"] for x in w + carriers)
        and all(y["passed"] for x in words for y in x["fixtures"])
        and all(x["passed"] for x in composite["fixtures"]) and bound_ok)
    assert all_pass
    assessment = dict(
        C1=dict(status="PASS", evidence="f64 format, read-only fegetround=FE_TONEAREST, ties-to-even probes, Python flags and pinned f64 SOURCE; binary formal proof not claimed"),
        C2=dict(status="PASS", evidence="pi/P hex and direct compiled CircuitData tau/neighbor probes agree with the pinned Rust setter SOURCE"),
        C3=dict(status="PASS", evidence="Named gate order/inverses checked against unchanged lower_native and exact written words"),
        C4=dict(status="PASS", evidence="Fixed W residues, wrap and multiple-wrap fixtures match; general closure relies on N3-B1a algebra plus these runtime checks"),
        C5=dict(status="PASS", evidence="Only numeric carrier phases in [0,tau], whole-product inverse and fixed compose path; no symbolic/control-flow/post-control"),
        C6=dict(status="PASS", evidence="Fixed in-memory path; 43 nontrivial chunk joins + one final join, at most one whole inverse, one nonzero phase initialization, zero QPY loads"),
    )
    source_refs = {
        "C1": ["N3-B1a PHASE_ERROR_BOUND.md sections 2-4",
               "CPython 3.12.3 mathmodule.c/pymath.h; pinned Rust f64 SOURCE"],
        "C2": ["Qiskit 896d1c8b72888a52a46081a6d1f4de8c56db8aea: crates/circuit/src/circuit_data.rs set_global_phase",
               "N3-B1a SOURCE_REFERENCES.json"],
        "C3": ["track_c2_b_verify.lower_native", "track_c_h4_synthesis.exact_word_matrix",
               "pr_reference_ct1.word_matrices adjoint map", "Pinned Qiskit h/s/t/x.py"],
        "C4": ["N3-B1a PHASE_ERROR_BOUND.md section 3", "track_c2_b_verify.lower_native W branch"],
        "C5": ["Pinned Qiskit quantumcircuit.py compose/inverse/global_phase",
               "Pinned circuit_data.rs numeric Float branch"],
        "C6": ["TEST_INPUTS.json adopted_paths", "verify_pr_reference_n3b1b0_runtime.py composite_tests",
               "N3-B1a PHASE_ERROR_BOUND.md section 5",
               "Pinned QuantumCircuit.copy/copy_empty_like and CircuitData.copy/new"],
    }
    observations = {
        "C1": dict(fegetround=env["fegetround"], ties_to_even=True,
                   unsafe_Python_flags=env["unsafe_python_flags_detected"],
                   Rust_build_flags_available=False, binary_formal_proof=False),
        "C2": dict(pi_hex=env["pi_hex"], P_hex=env["P_hex"],
                   compiled_setter_tau_and_neighbors_passed=True),
        "C3": dict(word_sign_fixtures=16, composite_sign_fixtures=2,
                   all_named_order_and_adjoint_checks_passed=True),
        "C4": dict(W_fixtures=len(w), all_passed=True,
                   finite_tests_are_not_the_general_closure_proof=True),
        "C5": dict(carrier_fixtures=len(carriers), all_passed=True,
                   symbolic_or_post_control_paths_tested=False),
        "C6": composite["bookkeeping"],
    }
    for key, row in assessment.items():
        row["source_basis"] = source_refs[key]
        row["runtime_result"] = observations[key]
    write("ENVIRONMENT.json", env)
    write("SOURCE.json", dict(base_commit=BASE, test_source=str(Path(__file__).relative_to(ROOT)),
        test_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reused_function_AST=PROVENANCE, CT1_adjoint_map=adj_map,
        saved_word_archive=dict(path=INPUT["word_archive"],
            sha256=hashlib.sha256(archive).hexdigest(), records=saved),
        input_sha256=hashlib.sha256((OUT / "TEST_INPUTS.json").read_bytes()).hexdigest()))
    write("PHASE_TESTS.json", dict(status="PASS", W_tests=w,
        carrier_phase_tests=carriers, written_word_tests=words, composite=composite,
        copy_boundary=copy_boundary,
        qpy=INPUT["qpy"], operator_diagnostics_are_certificates=False,
        binary_period_rounding_errors_exclude_true_2pi_difference=True,
        true_period_difference="Charged analytically in the N3-B1a SOURCE bound; zero binary-period error does not imply zero strict operator error.",
        no_phase_fit=True, no_strict_guard_executed=True,
        wall_seconds=time.monotonic() - start))
    write("RUNTIME_COMPATIBILITY.json", dict(status="RUNTIME_EMITTER_COMPATIBLE",
        qualification="Fixed observed runtime and adopted in-memory paths, under ordinary IEEE f64 semantics and named ideal logical gates; no real 43-word or whole CT RPE qualification.",
        C_phi=assessment, declared_symbolic_Frobenius_upper=str(bound),
        coarse_bound="1/2199023255552", saved_residual_lower=str(residual),
        strict_rational_comparisons_passed=bound_ok,
        actual_43_basic_words="NOT_ACQUIRED", production_provider="NOT_IMPLEMENTED",
        qpy=INPUT["qpy"], new_guard_or_synthesis_calls=0, new_scientific_price_calls=0,
        forbidden_physics_or_estimator_runs=0))
    print(json.dumps(dict(status="RUNTIME_EMITTER_COMPATIBLE",
        W_fixtures=len(w), carrier_fixtures=len(carriers),
        word_sign_fixtures=sum(len(x["fixtures"]) for x in words),
        composite_sign_fixtures=len(composite["fixtures"]),
        bound=str(bound), wall_seconds=time.monotonic() - start)))


if __name__ == "__main__":
    main()
