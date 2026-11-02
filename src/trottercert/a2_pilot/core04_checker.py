"""Independent SPEC-04--07 recomputation with a shared SPEC-01--03 gate."""

from . import verifier_algebra as algebra
from ..a2_spec04_07 import verifier_steps as steps
from .scope04 import (
    ContractError, LIMITS, NAMESPACE, SCOPE, VERSION, content_id, detached, failure,
    keys_exact, node, partial_identity, policy_validate, prerequisite, read_rat,
    require, root_precision, source_identity, trial, unpack,
)


def _same(actual, expected, path):
    require(actual == expected, "RECOMPUTATION_MISMATCH", path,
            "Independent recomputation or exact context/record comparison differs")


def check_spec04_07(raw, base_evidence, evidence, *, expected_particle_number,
                    expected_occupied_modes, expected_sqrt_bits,
                    expected_comparison_policy=None, expected_fragment_order=None,
                    expected_model_id=None, stage_observer=None):
    """Reconstruct evidence from fixed inputs, not generator flags or outputs.

    Expected trial, policy and precision must be independently supplied by caller.
    A successful core check is not A2 adoption or scientific authorization.
    """
    stage = "SPEC01_03_PREREQUISITE"
    partial = partial_identity(raw, base_evidence, evidence)
    try:
        base, model, report = prerequisite(raw, base_evidence, expected_particle_number,
                                           expected_fragment_order, expected_model_id)
        n, c, q, squares, fragments, caps = unpack(model, base)
        determinant = trial(expected_occupied_modes, n, expected_particle_number)
        precision = root_precision(expected_sqrt_bits)
        policy = policy_validate(expected_comparison_policy)
        parents = {**report["context"], "base_evidence_id": detached(base_evidence)["id"],
                   "raw_input_id": report["raw_input_id"]}
        partial["context"] = parents
        stage = "EVIDENCE_STRUCTURE"
        if stage_observer is not None:
            stage_observer(stage)
        evidence = detached(evidence)
        keys_exact(evidence, ("id", "payload"), "evidence")
        body = evidence["payload"]
        fields = ("schema", "scope", "completion", "source", "context", "sqrt_bits", "comparison_policy",
                  "upper_certificate", "ksos_certificate", "commutator_certificate", "step_comparison", "limitations", "cost_ledger")
        require(type(body) is dict and set(fields) <= set(body), "INCOMPLETE_CERTIFICATE", "bundle", "Missing required SPEC04--07 records")
        keys_exact(body, fields, "bundle")
        _same((body["schema"], body["scope"], body["completion"]),
              (VERSION, SCOPE, "SPEC04_07_FIXED_DF_PILOT"), "scope")
        require(body["context"] == parents, "CONTEXT_MISMATCH", "context", "Different model/sector/partition/PF/raw/base")
        require((body["sqrt_bits"], body["comparison_policy"]) == (expected_sqrt_bits, policy),
                "CONTEXT_MISMATCH", "task_policy", "Expected resolution and task must match")
        for key in ("upper_certificate", "ksos_certificate", "commutator_certificate", "step_comparison"):
            record = body[key]
            keys_exact(record, ("id", "payload"), key)
            require(type(record["payload"]) is dict and type(record["payload"].get("data")) is dict,
                    "INCOMPLETE_CERTIFICATE", key, "Missing payload/data")
        stage = "SPEC04_UPPER"
        if stage_observer is not None:
            stage_observer(stage)
        u = algebra.upper(c, q, squares, expected_occupied_modes, read_rat(report["L"]))
        require(read_rat(u["W"]) >= 0, "UPPER_LOWER_INCONSISTENT", "W", "U below L")
        up = node("upper_certificate_v1", {"kind": "UpperCertificate", "parents": parents,
                  "trial": determinant, "data": u})
        _same(body["upper_certificate"], up, "upper_certificate")
        stage = "SPEC05_KSOS"
        if stage_observer is not None:
            stage_observer(stage)
        kdata = body["ksos_certificate"]["payload"]["data"]
        k = algebra.verify_k_sos(kdata, caps, [f["fragment_id"] for f in fragments], precision)
        _same(body["ksos_certificate"], node("ksos_certificate_v1", {"kind": "KSOSCertificate", "parents": parents, "data": kdata}), "ksos_certificate")
        stage = "SPEC06_COMMUTATOR"
        if stage_observer is not None:
            stage_observer(stage)
        cb = algebra.commutators(q, squares, fragments, caps)
        cc = node("commutator_certificate_v1", {"kind": "CommutatorCertificate", "parents": parents, "data": cb})
        _same(body["commutator_certificate"], cc, "commutator_certificate")
        stage = "SPEC07_STEPS"
        if stage_observer is not None:
            stage_observer(stage)
        comparison = steps.compare(read_rat(report["Lambda"]), read_rat(u["W"]), k, read_rat(cb["Cbar2"]), policy)
        cmp = node("step_comparison_v1", {"kind": "ComparisonRecord", "parents": parents,
                   "certificate_refs": [up["id"], body["ksos_certificate"]["id"], cc["id"]],
                   "conditional_on_exact_ground_access": True, "trial_is_ground_state": False, "data": comparison})
        _same(body["step_comparison"], cmp, "step_comparison")
        stage = "HASH_SOURCE_LIMITS"
        if stage_observer is not None:
            stage_observer(stage)
        _same(body["source"], source_identity(), "source")
        _same(body["limitations"], dict(LIMITS), "limitations")
        _same(body["cost_ledger"], {"status": "NOT_MEASURED_PER_ARTIFICIAL_CALL", "generator_cost": None, "checker_cost": None,
              "reason": "See test execution measurements and documented symbolic complexity; unknown is not zero",
              "information_pool": "ALL_VERIFIED_CERTIFICATES_SHARED_BY_BOTH_ROUTES"}, "cost_ledger")
        _same(evidence["id"], content_id(NAMESPACE, body), "evidence.id")
        return {"schema": VERSION, "scope": SCOPE, "verdict": "SPEC04_07_VERIFIED_EXACT_FIXED_DF",
                "evidence_id": evidence["id"], "source": source_identity(), "context": parents,
                "L": report["L"], "U": u["U"], "W": u["W"], "Lambda": report["Lambda"],
                "K_SOS_upper": kdata["K_SOS_upper"], "Cbar2": cb["Cbar2"],
                "verified_quantities": ["determinant_U", "W", "K0_raw_dictionary", "sqrt_upper_witnesses",
                                        "K_SOS_row_upper", "grouped_commutators", "centered_and_structural_Cbar2"]
                                       + (["step_predicates_and_enclosures"] if policy is not None else []),
                "step_comparison": comparison, "limitations": dict(LIMITS)}
    except ContractError as exc:
        return failure(exc, stage, partial)
    except OSError as exc:
        return failure(ContractError("SOURCE_IO", "source", str(exc)), stage, partial)
