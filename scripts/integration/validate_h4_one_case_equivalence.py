"""Independent saved-H4 verification; public P1 scientific CLI stays disabled.

Never generate chemistry/state/events/circuits. Run once with an already fixed
contract and private locations. Source snapshots and dependencies are external.
"""
from __future__ import annotations
import argparse
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import time

ITEMS = ("DF_HAMILTONIAN", "SCALAR_ONE_BODY", "FRAGMENT_ORDER", "FIXED_PARTITION",
         "S2_PREPARATION", "FINITE_RTE_LAW", "EXPLICIT_EVENT_REQUEST",
         "PHASE_SIGN_CONTROL", "REAL_INPUT_CONTEXT_PROPAGATION", "SAVED_NUMERICAL_REFERENCE")
EXPECTED_NPZ = "06672860ed8f9298053b4d84693fbab6d13e38c3166c27dbabcc161f33673c3d"
EXPECTED_FREEZE = "75d7ddc8dc71ebeec03a6c173397a9b941b492b74e4dc80814d613d83ce56c69"


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_json(value):
    if type(value) is float:
        return {"real64_hex": value.hex()}
    if isinstance(value, dict):
        return {key: exact_json(v) for key, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [exact_json(v) for v in value]
    return value


def fingerprint(domain, payload):
    data = json.dumps(exact_json({"domain": domain, **payload}), sort_keys=True,
                      ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def match_class(*, passed, independent=False, wiring=False, metadata=False):
    if not passed:
        return "MISMATCH"
    if independent:
        return "MATCH_INDEPENDENT"
    if wiring:
        return "MATCH_WIRING_ONLY"
    if metadata:
        return "MATCH_METADATA_ONLY"
    return "INCONCLUSIVE"


def array_comparison(actual, reference, *, atol, rtol):
    import numpy as np
    a, b = np.asarray(actual), np.asarray(reference)
    if a.shape != b.shape:
        return {"passed": False, "shape_mismatch": True, "max_abs_error": None}
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return {"passed": False, "nonfinite": True, "max_abs_error": None}
    error = np.abs(a - b)
    return {"passed": bool(np.all(error <= atol + rtol * np.abs(b))),
            "max_abs_error": float(np.max(error, initial=0)),
            "atol": atol, "rtol": rtol, "shape": list(a.shape)}


def make_context(qualification):
    from trottertracks.resource_applicability.pipeline import (
        ScientificContext, EvidenceLocator, CompilerProfile)
    identity = qualification["candidate_identity"]
    # Compiler values are identity/context only: no compilation/accounting call.
    return ScientificContext(
        study_id="H4_ONE_CASE_DETERMINISTIC_EQUIVALENCE_20261028",
        cohort_id="INDEPENDENT_SAVED_ARRAY_VERIFICATION_NOT_OLD_RESTART",
        result_version="v1", source=EvidenceLocator(qualification["P1_commit"],
            "src/trottertracks/resource_applicability/h4_geometry_pipeline.py", "SOURCE"),
        evidence=(EvidenceLocator("821cc0f6ced57168293ea166644024867b064ffb",
            "artifacts/resource_applicability/track_a_h4_geometry_scalar_diagnostic/2026-10-13/source_input_provenance_v1.json", "INPUT_METADATA"),),
        representation="DF", geometry="linear-H4:0.90Angstrom", basis="STO-3G",
        sector="N=4,Sz=0;8-spin-orbitals", state_id=identity["state"], input_id=identity["input"],
        df_rank=12, rank_policy="frozen_geometry_rank12_generated_order", ld=3,
        deterministic_indices=(0, 1, 2), randomized_indices=tuple(range(3, 12)),
        formula="2nd", q=1,
        stage_order=("deterministic_half", "finite_rte", "reverse_deterministic_half"),
        evolution_time=.8, rte_steps=4, finite_taylor_order=4,
        rte_law="rte_even_taylor_paired", normalization_policy="once_per_complete_occurrence",
        seed=20261006, precision="binary64/complex128",
        applicable_domain="one fixed saved non-B3 DF H4 case; no science CLI/provider",
        estimand="fixed_complex_signal", metric="rz_count",
        cost_scope="single_hadamard_interrogation_without_state_preparation",
        compiler=CompilerProfile(("rz", "cx", "sx", "x"), None, None, 1,
                                 None, None, 17, "1.3.0"),
        source_stop_status="PRESERVED_NOT_RESTARTED", input_kind="frozen_scientific")


class ForbiddenCall(RuntimeError):
    pass


def install_tripwires():
    """Deny prohibited entry points in this process; never replace a science output."""
    import numpy as np
    import scipy.linalg
    import scipy.sparse.linalg
    import qiskit
    from qiskit import QuantumCircuit
    import trotterlib.df_partial_s2 as s2
    import trotterlib.rte as rte
    import trotterlib.df_hamiltonian as dh
    import trotterlib.df_trotter.ops as ops
    def deny(*args, **kwargs):
        raise ForbiddenCall("prohibited science/circuit/random/solver entry point")
    # Input/state/operator calls stay original. Only prohibited entry points deny.
    QuantumCircuit.__init__ = deny
    qiskit.transpile = deny
    s2.prepare_df_partial_s2 = deny
    ops.U_to_qiskit_ops_jw = deny
    ops.U_to_qiskit_ops_jw_givens = deny
    dh.solve_df_ground_state = deny
    dh.build_df_h_d_from_molecule = deny
    dh.df_hamiltonian_from_integrals = deny
    np.linalg.eigh = deny
    np.linalg.eigvalsh = deny
    scipy.linalg.eigh = deny
    scipy.sparse.linalg.eigsh = deny
    scipy.sparse.linalg.lobpcg = deny
    np.random.default_rng = deny
    np.random.Generator = deny
    np.random.PCG64 = deny
    for name in ("iter_sample_rte_events", "sample_rte_events", "sample_event_mean_operator"):
        setattr(rte, name, deny)
    return deny


def run(locations_path, contract_path, qualification_path, output):
    import numpy as np
    from trottertracks.resource_applicability.h4_geometry_pipeline import H4GeometryAdapter, FrozenH4Input
    from trotterlib.df_hamiltonian import PhysicalSector, df_linear_operator
    from trotterlib.df_partial_randomized_pf import RankedDFFragment, DFFragmentPartition
    from trotterlib.df_rte_tail import (uncontrolled_identity_evolution_operator,
                                      controlled_identity_evolution_operator)
    locations = json.loads(Path(locations_path).read_text())
    contract_bytes = Path(contract_path).read_bytes()
    contract = json.loads(contract_bytes)
    qualification = json.loads(Path(qualification_path).read_text())
    output = Path(output)
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    record = {"contract_sha256": hashlib.sha256(contract_bytes).hexdigest(),
              "items": [], "calls": {}, "forbidden_calls": 0,
              "status": "STARTED", "scientific_API_retries": 0}
    start = time.monotonic()
    def flush():
        path = output / "DETAILED_RESULTS_PRIVATE.json"
        path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        path.chmod(0o600)
    def call(name, function, *args, **kwargs):
        cap = contract["API_call_caps"][name]
        record["calls"][name] = record["calls"].get(name, 0) + 1
        if record["calls"][name] > cap:
            raise ForbiddenCall("fixed call budget exceeded: " + name)
        flush()
        return function(*args, **kwargs)
    def item(name, status, evidence, limitations=""):
        record["items"].append({"item": name, "status": status,
                                "evidence": evidence, "limitations": limitations})
        flush()
    try:
        limits = contract["resource_caps"]
        resource.setrlimit(resource.RLIMIT_AS, (limits["address_space_bytes"], limits["address_space_bytes"]))
        resource.setrlimit(resource.RLIMIT_CPU, (limits["CPU_seconds"], limits["CPU_seconds"]))
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("fixed wall limit")))
        signal.alarm(limits["wall_seconds"])
        assert record["contract_sha256"] == Path(contract_path).with_name("CONTRACT_SHA256").read_text().strip()
        npz_path = Path(locations["NPZ"])
        freeze_path = npz_path.with_name("generation-freeze.json")
        assert file_sha(npz_path) == EXPECTED_NPZ
        assert file_sha(freeze_path) == EXPECTED_FREEZE
        freeze = json.loads(freeze_path.read_text())
        assert fingerprint("h4-generation-freeze-v1", freeze) == "0fd1de52bce01c351c99efe0535913b2b3fd9a359ee72be36bc75cf847d193e7"
        assert fingerprint("h4-input-freeze-v1", freeze["identities"]["0.90"]) == qualification["candidate_identity"]["input"]
        previous = Path(locations["prior_discovery"]) / "private/FULL_ARRAY_MANIFEST_VERIFICATION_PRIVATE.json"
        audit = json.loads(previous.read_text())
        assert len(audit["records"]) == 32 and all(x["matches_freeze"] for x in audit["records"].values())
        with np.load(npz_path, allow_pickle=False) as saved:
            assert len(saved.files) == 32
            arrays = {key: saved[key] for key in saved.files}
        for key, value in arrays.items():
            assert value.dtype.kind in "fciu" and value.dtype.str == audit["records"][key]["dtype"]
            assert list(value.shape) == audit["records"][key]["NPY_shape"]
            value.flags.writeable = False
        record["runtime32_header_checks"] = True
        install_tripwires()
        context = make_context(qualification)
        adapter = H4GeometryAdapter()
        core = call("H4GeometryAdapter.bind_core", adapter.bind_core, context)
        # Proxy weights are metadata only, never an exact finite-RTE lambda.
        fragments = tuple(RankedDFFragment(i, i, float(arrays["lambdas"][i]),
                           abs(float(arrays["lambdas"][i])), "abs_lambda") for i in range(12))
        partition = call("DFFragmentPartition_record", DFFragmentPartition, 3,
                         fragments[:3], fragments[3:], (0,1,2), tuple(range(3,12)),
                         math.fsum(x.weight for x in fragments[3:]), "abs_lambda")
        payload = FrozenH4Input(arrays, context.input_id, context.state_id, partition, context.binding)
        # DFHamiltonian is the actual constructor called by this single adapter call.
        result = call("H4GeometryAdapter.input", adapter.input, context, payload, core)
        record["calls"]["DFHamiltonian"] = 1
        assert result.status == "UNQUALIFIED"  # P1 qualification flag is deliberately unchanged.
        h = result.payload["hamiltonian"]
        wiring = (result.binding == context.binding and h.metadata["input_id"] == context.input_id
                  and h.metadata["state_id"] == context.state_id and result.payload["partition"] is partition)
        numerical_data = (np.array_equal(h.one_body, arrays["one"])
                          and np.array_equal(h.lambdas, arrays["lambdas"])
                          and all(np.array_equal(g, old) for g, old in zip(h.g_matrices, arrays["G"], strict=True)))
        assert numerical_data
        item("REAL_INPUT_CONTEXT_PROPAGATION", match_class(passed=wiring, wiring=True),
             {"actual_adapter_status": result.status, "binding_exact": wiring,
              "constructor_forwarding_exact": numerical_data},
             "No independent scientific provider; public composer scientific/real/resume modes stay disabled")
        item("FRAGMENT_ORDER", "MATCH_METADATA_ONLY",
             {"stored_indices": list(range(12)), "original_raw_indices": arrays["raw_indices"].tolist(),
              "permutation_prefix_matches": bool(np.array_equal(arrays["raw_indices"], arrays["permutation"][:12])),
              "P1_lambdas_G_byte_positions_exact": numerical_data},
             "No reranking; metadata/input equality is not a numerical S2 operator result")
        item("FIXED_PARTITION", "MATCH_METADATA_ONLY",
             {"deterministic_stored_indices": list(partition.deterministic_block_indices),
              "randomized_stored_indices": list(partition.randomized_block_indices),
              "raw_deterministic_indices": arrays["raw_indices"][:3].tolist(),
              "raw_randomized_indices": arrays["raw_indices"][3:].tolist()},
             "Explicit partition accepted by input adapter; abs_lambda proxy not used as exact RTE norm; no prepare call")
        t = contract["tolerances"]["scalar_one_body"]
        one_check = array_comparison(h.one_body, arrays["hspin"] + arrays["correction"], **t)
        scalar_exact = float(h.constant) == float(arrays["nuclear"])
        item("SCALAR_ONE_BODY", match_class(passed=one_check["passed"] and scalar_exact, independent=True),
             {"one_body_vs_original_hspin_plus_correction": one_check,
              "nuclear_exactly_forwarded": scalar_exact},
             "Once-only nuclear application also tested by full original stored H comparison")
        sector = PhysicalSector(8, np.arange(256, dtype=np.int64))
        operator, counter = call("df_linear_operator_python", df_linear_operator, h, sector, backend="python")
        dense = call("LinearOperator.matmat_identity256", operator.matmat, np.eye(256, dtype=np.complex128))
        record["calls"]["internal_python_matvec"] = counter["count"]
        assert counter["count"] <= contract["API_call_caps"]["internal_python_matvec"]
        t = contract["tolerances"]["hamiltonian"]
        h_check = array_comparison(dense, arrays["H"], **t)
        item("DF_HAMILTONIAN", match_class(passed=h_check["passed"], independent=True),
             {"full_fock_256_matrix_vs_original_saved_H": h_check},
             "Independent historical stored matrix vs existing Python fermion-action core; one molecule/candidate only")
        if not h_check["passed"]:
            item("SAVED_NUMERICAL_REFERENCE", "INCONCLUSIVE", {}, "Dependent H/state checks stopped after matrix mismatch")
        else:
            record["calls"]["state_scalar_checks_batch"] = 1
            state = arrays["state"]
            si = arrays["sector"].astype(np.int64, copy=False)
            expected_sector = np.array([i for i in range(256) if
                sum((i >> (7-p)) & 1 for p in (0,2,4,6)) == 2 and
                sum((i >> (7-p)) & 1 for p in (1,3,5,7)) == 2], dtype=np.int64)
            bitrev = np.array([int(format(i, "08b")[::-1], 2) for i in range(256)], dtype=np.int64)
            norm_error = float(abs(np.vdot(state, state) - 1))
            embedded_error = float(np.max(np.abs(state[si] - arrays["sector_state"])))
            outside = np.ones(256, dtype=bool); outside[si] = False
            outside_error = float(np.max(np.abs(state[outside]), initial=0))
            bit_error = float(np.max(np.abs(arrays["qiskit_state"] - state[bitrev])))
            psiH = dense @ state
            energy = float(arrays["energy"])
            residual = float(np.linalg.norm(psiH - energy * state))
            energy_error = float(abs(np.vdot(state, psiH) - energy))
            local = arrays["sector_state"]
            pivot = local[int(np.argmax(np.abs(local)))]
            phase_error = float(abs(pivot.imag))
            sector_matrix_check = array_comparison(dense[np.ix_(si,si)], arrays["sector_H"], **t)
            checks = {"norm_abs_error": norm_error, "sector_embedding_abs_error": embedded_error,
                      "outside_sector_abs_error": outside_error, "bit_reversal_abs_error": bit_error,
                      "stored_energy_abs_error": energy_error, "H_state_residual_L2": residual,
                      "phase_pivot_imag_abs_error": phase_error,
                      "phase_pivot_positive": bool(pivot.real > 0),
                      "sector_order_exact": bool(np.array_equal(si, expected_sector)),
                      "sector_H": sector_matrix_check}
            passed = (norm_error <= 1e-12 and max(embedded_error, outside_error, bit_error, phase_error) <= 1e-12
                      and residual <= 1e-9 and energy_error <= 1e-9 and pivot.real > 0
                      and checks["sector_order_exact"] and sector_matrix_check["passed"])
            item("SAVED_NUMERICAL_REFERENCE", match_class(passed=passed, independent=True), checks,
                 "Independent saved H/sector/state/energy only; sampled finite-trajectory signal/cost/N/G/SE are not compared")
        coefficient = float(arrays["nuclear"])
        rawU = call("uncontrolled_identity_evolution_operator", uncontrolled_identity_evolution_operator,
                    coefficient, .8, 256)
        rawC = call("controlled_identity_evolution_operator", controlled_identity_evolution_operator,
                    coefficient, .8, 256)
        angle = -.8 * coefficient
        phase = complex(math.cos(angle), math.sin(angle))
        ident = np.eye(256, dtype=np.complex128)
        phase_ref = phase * ident
        controlled_ref = np.zeros((512,512), dtype=np.complex128)
        controlled_ref[:256,:256] = ident; controlled_ref[256:,256:] = phase_ref
        t = contract["tolerances"]["identity_phase_control"]
        uc = array_comparison(rawU, phase_ref, **t); cc = array_comparison(rawC, controlled_ref, **t)
        item("PHASE_SIGN_CONTROL", match_class(passed=uc["passed"] and cc["passed"], independent=True),
             {"nuclear_identity_global_phase": uc, "nuclear_controlled_diag_I_U": cc},
             "Only identity scalar negative phase and ancilla-major diag(I,U); full S2/event control not executed")
        for name, reason in contract["not_evaluated"].items():
            item(name, "INCONCLUSIVE", {"actual_API_calls": 0}, reason)
        assert {x["item"] for x in record["items"]} == set(ITEMS)
        record["status"] = "COMPLETED_WITH_ITEM_LIMITS"
        record["real_arrays_readonly"] = all(not a.flags.writeable for a in arrays.values())
        # Opaque original matrices and newly reconstructed result stay private.
        private_matrix = output / "reconstructed_H_PRIVATE.npy"
        np.save(private_matrix, dense, allow_pickle=False); private_matrix.chmod(0o600)
        record["classification_counts"] = {s: sum(r["status"] == s for r in record["items"])
             for s in sorted({r["status"] for r in record["items"]})}
    except BaseException as exc:
        if isinstance(exc, ForbiddenCall):
            record["forbidden_calls"] += 1
        record["status"] = "STOPPED"
        record["error_type"] = type(exc).__name__
        record["error_private"] = str(exc)
        record["elapsed_wall_seconds"] = time.monotonic() - start
        flush()
        raise
    finally:
        signal.alarm(0)
        record["elapsed_wall_seconds"] = time.monotonic() - start
        record["peak_RSS_KiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        flush()
    print(json.dumps({k: record[k] for k in ("status", "classification_counts", "calls", "forbidden_calls", "elapsed_wall_seconds", "peak_RSS_KiB")}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--locations", required=True)
    p.add_argument("--contract", required=True)
    p.add_argument("--qualification", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    run(a.locations, a.contract, a.qualification, a.output)


if __name__ == "__main__":
    main()
