"""Authorized N4-B saved-input/local-only validation. No guard or backend calls."""
from __future__ import annotations
import ast
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction as F
from functools import lru_cache
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = "547532cbdc51de0721d333bcc8dfc9dbfab1f76c"
OUT = ROOT / "artifacts/pr_reference_n4b_consumer/2026-11-08"
A4 = "artifacts/pr_reference_n4a_coverage/2026-11-08/"
D3 = "artifacts/pr_reference_n3b1b1_dictionary/2026-11-08/"
C4 = "artifacts/pr_reference_ct4/2026-11-07/"
INPUTS = {}
NS = PROVIDER = CONSUMER = None


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def blob(path):
    raw = subprocess.check_output(["git", "show", BASE + ":" + path], cwd=ROOT)
    INPUTS[path] = dict(sha256=sha(raw), bytes=len(raw))
    return raw


def read(path):
    return json.loads(blob(path))


def save(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True,
        ensure_ascii=False, allow_nan=False) + "\n")


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def function(path, names, namespace):
    raw = blob(path)
    assert (ROOT / path).read_bytes() == raw
    nodes = [n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    for node in nodes:
        node.decorator_list = []
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ROOT/path), "exec"), namespace)
    return namespace


def members(path):
    with tarfile.open(fileobj=io.BytesIO(blob(path)), mode="r:xz") as t:
        return {m.name: t.extractfile(m).read() for m in t if m.isfile()}


def load_provider():
    fixed = read(D3 + "INPUTS.json")
    manifest = read(D3 + "BASIC_WORDS_MANIFEST.json")
    archive = members(D3 + "basic_records.tar.xz")
    ns = function("src/trottertracks/pr_reference_ct0.py", ["positive", "validate_record"],
        dict(math=math, F=F, hashlib=hashlib))
    basic = {}
    for row in manifest["rows"]:
        raw = archive[row["record"]]
        assert sha(raw) == row["record_sha256"] and row["status"] == "PASS"
        record = json.loads(raw)
        ns["validate_record"](record)
        basic[row["j"]] = record
    cache = {}
    adopted = read(C4 + "KEY_REUSE_AND_SYNTHESIS.json")
    expected = {r["key"] for r in adopted["records"]}
    for raw in members(C4 + "strict_records.tar.xz").values():
        record = json.loads(raw)
        ns["validate_record"](record)
        assert record["key"] in expected and record["guard_method"] is not None
        assert record["theta_hex"] not in cache
        cache[record["theta_hex"]] = record
    assert len(basic) == 43 and len(cache) == len(expected) == 299
    mod = module("src/trottertracks/pr_reference_dyadic_dictionary.py", "n4b_dictionary")
    assert (ROOT/"src/trottertracks/pr_reference_dyadic_dictionary.py").read_bytes() == blob(
        "src/trottertracks/pr_reference_dyadic_dictionary.py")
    provider = mod.DictionaryProvider(fixed["fixed_parameters"], fixed["error_budget"], basic, cache)
    return provider, mod, cache


def certificate_stage(coverage, mod):
    keys = coverage["non_C0_domain_candidate_keys"]
    assert len(keys) == len(set(keys)) == 85
    results = []
    for key in keys:
        theta = float.fromhex(key)
        assert theta.hex() == key and key not in PROVIDER.cache and theta <= PROVIDER.limit
        positive = PROVIDER.provide(theta)
        negative = PROVIDER.provide(-theta)
        assert positive.route == negative.route == "dyadic_composite_certificate"
        assert positive.positive_sequence == negative.positive_sequence
        assert negative.signed_sequence() == mod.actual_adjoint(positive.positive_sequence)
        assert positive.strict_upper == negative.strict_upper < F(1, 10**11)
        assert positive.global_W_signed == positive.positive_sequence.count("W")
        assert negative.global_W_signed == -positive.global_W_signed
        bindings = [dict(arm=r["arm"], role=r["role"], axes=r["axes"])
                    for r in coverage["roles"] if key in r["positive_saved_keys"]]
        assert bindings
        results.append(dict(positive_literal_hex=key, roles=bindings,
            positive=positive.summary(), negative=negative.summary(),
            positive_sequence=positive.positive_sequence,
            negative_sequence_sha256=sha(negative.signed_sequence().encode()),
            negative_rule="actual adjoint of COMPLETE positive written word; lower_native inverses once",
            strict_qualified=True, new_guard=False))
    save("ROLE_WORD_CERTIFICATES.json", dict(status="ROLE_WORD_COVERAGE_QUALIFIED",
        base_commit=BASE, scope="fixed 85 saved numeric superset, not trajectory enumeration",
        rows=results, count=85, frame_keys=84, Bprime_b2_keys=1))
    cache_checks = []
    for key in sorted(PROVIDER.cache):
        cert = PROVIDER.provide(float.fromhex(key))
        assert cert.route == "fixed_C0_priority"
        assert cert.positive_sequence == PROVIDER.cache[key]["sequence"]
        cache_checks.append(dict(key=key, sequence_sha256=sha(cert.positive_sequence.encode()),
            preserved=True))
    # The provider signature and the consumer have no arm/role selection input.
    paired = []
    for key in keys:
        a = PROVIDER.provide(float.fromhex(key))
        b = PROVIDER.provide(float.fromhex(key))
        assert a == b
        paired.append(dict(key=key, same_certificate=True,
            sequence_sha256=sha(a.positive_sequence.encode())))
    save("COMMON_POLICY_VERIFICATION.json", dict(status="COMMON_POLICY_IMPLEMENTED",
        fixed_C0_priority=cache_checks, paired_arm_checks=paired,
        arm_or_role_selection_argument=False, new_word_acquisitions=0,
        literal_modulo_identification=False,
        preserved_N1_anchors=dict(K=160222, d0=106, d2=114, B_P=6072,
            threshold=173790, provenance="quoted N1/N3; no new price aggregate")))


def ins(q):
    return [(i.operation.name, [q.find_bit(z).index for z in i.qubits],
        [q.find_bit(z).index for z in i.clbits], list(map(float, i.operation.params))) for i in q.data]


def expected_stream(native):
    mapping = dict(H="h", S="s", T="t", t="tdg", X="x")
    inverse = dict(h="h", s="sdg", t="tdg", tdg="t", x="x")
    for name, qs, cs, params in ins(native):
        if name == "rz":
            cert = PROVIDER.provide(params[0])
            names = [mapping[g] for g in reversed(cert.positive_sequence) if g != "W"]
            if params[0] < 0:
                names = [inverse[g] for g in reversed(names)]
            for n in names:
                yield (n, qs, [], [])
        elif name == "sx":
            for n in ("h", "s", "h"):
                yield (n, qs, [], [])
        else:
            yield (name, qs, cs, params)


def check_consumer(native, label):
    out, audit = CONSUMER.consume_native(native, PROVIDER, NS["lower_native"])
    actual = ins(out)
    wanted = list(expected_stream(native))
    assert actual == wanted
    assert out.qubits == native.qubits and out.clbits == native.clbits
    assert out.qregs == native.qregs and out.cregs == native.cregs
    fences = [(n, qs, cs) for n, qs, cs, p in ins(native) if n in CONSUMER.FENCES]
    assert fences == [(n, qs, cs) for n, qs, cs, p in actual if n in CONSUMER.FENCES]
    assert len(audit["RZ_positions"]) == sum(n == "rz" for n, qs, cs, p in ins(native))
    assert all(x["segment_input_phase_hex"] == 0.0.hex() for x in audit["segments"])
    # Independent integer-W residue reference for the binary phase path.
    # This checks implementation conformance, not true-pi error certification.
    tau, p = 2*math.pi, math.pi/4
    phase = math.fmod(float(native.global_phase), tau)
    assert phase.hex() == audit["initial_outer_phase_hex"]
    native_instructions = ins(native)
    for segment in audit["segments"]:
        local = 0.0
        for name, qs, cs, params in native_instructions[
                segment["native_start"]:segment["native_end_exclusive"]]:
            if name == "rz":
                w = PROVIDER.provide(params[0]).global_W_signed
                local = math.fmod(local + (w % 8)*p, tau)
        assert local.hex() == segment["segment_emitted_phase_hex"]
        assert phase.hex() == segment["outer_before_hex"]
        phase = math.fmod(phase + local, tau)
        assert phase.hex() == segment["outer_after_hex"]
    assert phase.hex() == audit["output_global_phase_hex"]
    return out, dict(label=label, passed=True, all_gate_order_and_wire_checks=True,
        reset_readout_barrier_order_and_clbits=True, same_registers=True, audit=audit,
        integer_W_binary_phase_reference_matches=True,
        output_stream_sha256=sha(json.dumps(actual, separators=(",", ":")).encode()))


@lru_cache(None)
def word_matrix(key):
    cert = PROVIDER.provide(float.fromhex(key))
    return NS["matrix"](cert.positive_sequence)


def compressed(q, ct):
    """Reuse CT-1 apply_batch on local gate products; no whole simulation."""
    import numpy as np
    codes, q0, q1, mats = [], [], [], []
    for name, qs, cs, params in ins(q):
        assert name in CONSUMER.UNITARY
        codes.append(int(name == "cx")); q0.append(qs[0]); q1.append(qs[1] if name == "cx" else 0)
        if name == "rz":
            if ct:
                mat = word_matrix(abs(params[0]).hex())
                if params[0] < 0: mat = mat.conj().T
            else: mat = np.diag(np.exp(np.array([-.5j, .5j]) * params[0]))
        elif name == "sx": mat = np.array([[1+1j, 1-1j], [1-1j, 1+1j]])/2
        elif name == "x": mat = np.array([[0, 1], [1, 0]], complex)
        else: mat = np.eye(2)
        mats.append(mat)
    args = [np.asarray(x, dtype=d) for x, d in zip((codes, q0, q1, mats),
            (np.int64, np.int64, np.int64, np.complex128))]
    return NS["apply_batch"](np.eye(2**q.num_qubits, dtype=complex), *args)


def local_stage(targets):
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    output = []
    # Only the specified frame, b2, zero/sign/phase boundaries and saved bindings.
    selected = [float.fromhex("0x1.08ba32d420f2cp+2"),
                float.fromhex("0x1.921fb54442d18p+2"), 0.0, -0.0]
    for theta in selected[:2]:
        cert = PROVIDER.provide(theta)
        adj = PROVIDER.provide(-theta)
        assert NS["Exact"].from_gates((cert.positive_sequence + adj.signed_sequence()).replace(
            "t", "TTTTTTT")) == NS["Exact"].identity()
    for theta in selected + [-x for x in selected[:2]]:
        q = QuantumCircuit(1); q.rz(theta, 0)
        out, row = check_consumer(q, "signed_RZ:" + theta.hex())
        expected = word_matrix(abs(theta).hex())
        if theta < 0: expected = expected.conj().T
        delta = float(np.linalg.norm(Operator(out).data - expected, 2))
        assert delta < 1e-10
        row.update(exact_word_emitter_operator_diagnostic=delta,
            emitted_T_metadata_matches=(sum(i.operation.name in ("t", "tdg") for i in out.data)
                == PROVIDER.provide(theta).T))
        assert row["emitted_T_metadata_matches"]
        output.append(row)
    for target in targets["targets"]:
        raw = members(target["archive"])[target["member"]]
        batch = json.loads(raw)
        assert batch["request"] == target["request"]
        for fixed in target["records"]:
            rec = next(x for x in batch["records"] if x["name"] == fixed["name"])
            # Matches N4-A's canonical JSON record hashing, not missing global phase.
            assert sha(json.dumps(rec, sort_keys=True).encode()) == fixed["record_sha256"]
            if rec["name"] == "p_symbolic":
                assert rec["lowering"] == fixed["symbolic_literal_template"]
                values = selected + [-x for x in selected[:2]]
            else: values = [None]
            for value in values:
                q = QuantumCircuit(rec["n"])
                # This defines the local native GATE PRODUCT only. No claim
                # that absent SOURCE/global phase metadata equals zero.
                for name, qs, params in rec["lowering"]:
                    params = [value if p == "symbolic_correction" else p for p in params]
                    getattr(q, name)(*params, *qs)
                out, row = check_consumer(q, target["member"] + ":" + rec["name"] + ":" + str(value))
                a, c = compressed(q, False), compressed(q, True)
                # Frobenius diagnostic is cheap at dimension 256; the rigorous
                # joint guarantee uses spectral telescoping, never joint F.
                diagnostic = float(np.linalg.norm(c-a, "fro"))
                assert diagnostic < 1e-8
                row.update(native_gate_product_Frobenius_diagnostic=diagnostic,
                    SOURCE_global_phase="NOT_SAVED; unchanged arbitrary prefactor cancels in replacement",
                    SOURCE_unitary_reconstructed=False)
                output.append(row)
    # Small explicit shell fixture checks nontrivial clbit permutation, barriers
    # and a reset between two unitary segments (not just prefix resets).
    boundaries = [0.0, math.pi/4, math.nextafter(2*math.pi, 0.0), 2*math.pi]
    boundaries += [float.fromhex(r["global_phase_hex"]) for r in targets["whole_native_metadata"]]
    for phase in boundaries:
        q = QuantumCircuit(2, 3); q.global_phase = -math.ulp(0.0) if phase == 2*math.pi else phase
        q.reset(1); q.rz(selected[0], 1); q.cx(1, 0); q.barrier(1, 0)
        q.rz(-selected[1], 0); q.reset(0); q.sx(1); q.rz(-0.0, 1)
        q.measure(1, 2); q.measure(0, 0)
        out, row = check_consumer(q, "shell_phase:" + float(q.global_phase).hex())
        output.append(row)
    rejection = QuantumCircuit(1); rejection.h(0)
    try: CONSUMER.consume_native(rejection, PROVIDER, NS["lower_native"])
    except ValueError: rejected = True
    else: raise AssertionError("Unsupported native gate accepted")
    return dict(status="LOCAL_CORRECTNESS_PASS", cases=output,
        unsupported_gate_rejected=rejected, dense_whole_operator=False,
        diagnostics_are_strict_certificates=False)


def saved_whole_worker(row):
    """Read saved QPY; project only fixed local windows, never lower full body."""
    from qiskit import QuantumCircuit, qpy
    raw_archive = subprocess.check_output(["git", "show", BASE+":"+C4+"native_qpy.tar.xz"], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(raw_archive), mode="r:xz") as t:
        raw = t.extractfile("native/"+row["name"]+".qpy").read()
    assert sha(raw) == row["native_qpy_sha256"]
    q = qpy.load(io.BytesIO(raw))[0]
    assert q.num_qubits == row["qubits"] and dict(q.count_ops()) == row["operations"]
    assert float(q.global_phase).hex() == row["global_phase_hex"]
    # Actual native phase is strictly inside [0,tau), so the QPY constructor's
    # rem_euclid leaves this binary value unchanged. No new QPY output exists.
    assert 0 <= float(q.global_phase) < 2*math.pi
    alphabet, nonunitary, selected, literals = set(), [], set(), Counter()
    for j, inst in enumerate(q.data):
        name = inst.operation.name; alphabet.add(name)
        assert name in CONSUMER.UNITARY | CONSUMER.FENCES
        if name == "rz":
            literals[float(inst.operation.params[0]).hex()] += 1
        if name in CONSUMER.FENCES:
            selected.add(j)
            nonunitary.append(dict(position=j, name=name,
                qubits=[q.find_bit(x).index for x in inst.qubits],
                clbits=[q.find_bit(x).index for x in inst.clbits]))
    assert dict(literals) == row["expected_signed_RZ_occurrences"]
    frozen = set(PROVIDER.cache) | set(read(A4+"ROLE_COVERAGE.json")["non_C0_domain_candidate_keys"])
    assert {abs(float.fromhex(k)).hex() for k in literals} <= frozen
    # Fixed projection rule: original first/last four operations + ALL fences.
    # It preserves actual X/Y tail, prefix resets, wiring and global phase.
    selected.update(range(min(4, len(q.data))))
    selected.update(range(max(0, len(q.data)-4), len(q.data)))
    small = QuantumCircuit(q.qubits, q.clbits, global_phase=float(q.global_phase))
    for register in (*q.qregs, *q.cregs): small.add_register(register)
    for j in sorted(selected):
        inst = q.data[j]
        small.append(inst.operation, inst.qubits, inst.clbits, copy=False)
    _, result = check_consumer(small, "saved_whole_projection:" + row["name"])
    result.update(name=row["name"], native_qpy_sha256=sha(raw),
        native_qubits=q.num_qubits, native_clbits=q.num_clbits,
        native_operations=dict(q.count_ops()), native_global_phase_hex=float(q.global_phase).hex(),
        QPY_phase_matches_saved_metadata=True, nonunitary=nonunitary,
        selected_original_positions=sorted(selected), full_CT_materialization=False,
        all_saved_literal_counts_match=True, all_saved_keys_within_fixed_C0_or_85=True,
        SOURCE_whole_simulation=False, observed_native_alphabet=sorted(alphabet))
    return result


def main():
    global NS, PROVIDER, CONSUMER
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    NS = runpy.run_path(str(ROOT/"scripts/verify_pr_reference_n3b1b0_runtime.py"),
        run_name="n4b_runtime_reuse")
    NS["mp"].mp.dps = NS["mp"].iv.dps = 100
    environment = NS["runtime_checks"]()
    environment["inherited_N3_probe_qpy_note"] = environment.pop("qpy")
    environment["N4B_QPY_path"] = dict(saved_inputs=8, new_roundtrips=0,
        consumer_uses_QPY=False,
        qualification="saved canonical phase bits checked against metadata; no general QPY path qualification")
    save("ENVIRONMENT.json", environment)
    import numpy as np
    from numba import njit
    ns = function("src/trottertracks/pr_reference_ct1.py", ["apply_batch"], {"np": np})
    NS["apply_batch"] = njit(cache=False)(ns["apply_batch"])
    PROVIDER, mod, cache = load_provider()
    CONSUMER = module("src/trottertracks/pr_reference_common_word_consumer.py", "n4b_consumer")
    coverage, targets = read(A4+"ROLE_COVERAGE.json"), read(A4+"INTEGRATION_TARGETS.json")
    certificate_stage(coverage, mod)
    save("LOCAL_CORRECTNESS.json", local_stage(targets))
    blob(C4+"native_qpy.tar.xz")  # provenance once; children read same fixed blob
    old_native = {r["name"]: r for r in read(C4+"NATIVE_INPUT_CHECKS.json")["rows"]}
    tasks = [dict(t, expected_signed_RZ_occurrences=old_native[t["name"]]["signed_RZ_occurrences"])
             for t in targets["whole_native_metadata"]]
    with ProcessPoolExecutor(max_workers=min(len(tasks), len(os.sched_getaffinity(0)))) as pool:
        whole = list(pool.map(saved_whole_worker, tasks))
    save("SAVED_WHOLE_INTEGRATION.json", dict(status="LOCAL_CORRECTNESS_PASS", rows=whole,
        QPY_input_only=True, new_QPY_roundtrips=0, full_CT_materializations=0,
        projection_rule="first four + last four native instructions + all nonunitary fences"))
    provenance = {}
    for path in ["src/trottertracks/pr_reference_dyadic_dictionary.py",
        "src/trottertracks/pr_reference_common_word_consumer.py",
        "scripts/validate_pr_reference_n4b_consumer.py",
        "scripts/verify_pr_reference_n3b1b0_runtime.py",
        "src/trottertracks/track_c2_b_verify.py", "src/trottertracks/track_c_h4_synthesis.py",
        "src/trottertracks/pr_reference_ct0.py", "src/trottertracks/pr_reference_ct1.py"]:
        raw = (ROOT/path).read_bytes()
        provenance[path] = dict(sha256=sha(raw), bytes=len(raw))
    # Verify every read saved input still equals its fixed git object.
    assert all(sha(subprocess.check_output(["git", "show", BASE+":"+p], cwd=ROOT)) == v["sha256"]
               for p, v in INPUTS.items())
    # Read-only references for the next scope; do not invoke their geometry.
    read(C4+"ERROR_AND_SHOTS.json")
    read("artifacts/pr_reference_ct0/2026-11-07/PRECISION_FEASIBILITY.json")
    save("SOURCE_INPUTS.json", dict(base_commit=BASE, files=INPUTS, SOURCE=provenance,
        reused_functions=NS["PROVENANCE"], fixed_conditions=read(D3+"INPUTS.json")["fixed_parameters"],
        input_archives_read_only=True))
    save("CHECKS.json", dict(status="PASS", role_certificates=85, frame=84, b2=1,
        basic_words_unchanged=43, C0_unchanged=299, common_policy=True,
        local_correctness=True, saved_whole_metadata_cases=len(whole),
        consumer_phase="QUALIFIED_PER_RZ_PLUS_SEPARATE_SHELL_TERMS",
        compiler_context="CONDITIONAL_ON_COMPILER_CONTEXT",
        full_support_CT_RPE="FULL_SUPPORT_CT_RPE_UNQUALIFIED",
        new_synthesis_calls=0, strict_guard_calls=0, new_native_compile_calls=0,
        new_RPE_reanalysis_calls=0, new_T_aggregate_calls=0,
        elapsed_seconds=time.monotonic()-started))
    save("STOP.json", dict(status="MANDATORY_STOP", reason="Fixed N4-B scope complete",
        full_support_CT_RPE="UNQUALIFIED; new RPE numerical accounting not authorized",
        next_owner="GPT/user", next_action="Decide fixed m7/q128/r1 error-accounting scope; no automatic execution"))
    print(json.dumps(json.loads((OUT/"CHECKS.json").read_text())))


if __name__ == "__main__":
    main()
