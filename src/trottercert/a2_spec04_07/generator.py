"""SPEC-04--07 proposals for explicit artificial inputs; no CLI/file loader."""

from . import producer_algebra as algebra
from . import producer_steps as steps
from .common import (
    ContractError, LIMITS, NAMESPACE, SCOPE, VERSION, detached, failure, node,
    partial_identity, policy_validate, prerequisite, read_rat, require,
    root_precision, source_identity, trial, unpack,
)


def generate_spec04_07(raw, base_evidence, *, particle_number, occupied_modes,
                       sqrt_bits, comparison_policy=None, fragment_order=None,
                       expected_model_id=None):
    """Generate unverified evidence; supplied policy/resolution are not adopted.

    Synthetic provenance is a scope guard, not a security attestation. This API
    does not authorize use on scientific data or create an execution grant.
    """
    stage, partial = "SPEC01_03_PREREQUISITE", partial_identity(raw, base_evidence)
    try:
        base, model, report = prerequisite(raw, base_evidence, particle_number, fragment_order, expected_model_id)
        n, c, q, squares, fragments, caps = unpack(model, base)
        partial["context"] = report["context"]
        stage = "TRIAL_AND_POLICY"
        determinant = trial(occupied_modes, n, particle_number)
        precision = root_precision(sqrt_bits)
        policy = policy_validate(comparison_policy)
        parents = {**report["context"], "base_evidence_id": detached(base_evidence)["id"],
                   "raw_input_id": report["raw_input_id"]}
        stage = "SPEC04_UPPER"
        u = algebra.upper(c, q, squares, occupied_modes, read_rat(report["L"]))
        require(read_rat(u["W"]) >= 0, "UPPER_LOWER_INCONSISTENT", "W", "U must not be below L")
        upper = node("upper_certificate_v1", {"kind": "UpperCertificate", "parents": parents,
                     "trial": determinant, "data": u})
        partial["upper_certificate"] = upper
        stage = "SPEC05_KSOS"
        k = algebra.k_sos(caps, [f["fragment_id"] for f in fragments], precision)
        ksos = node("ksos_certificate_v1", {"kind": "KSOSCertificate", "parents": parents, "data": k})
        partial["ksos_certificate"] = ksos
        stage = "SPEC06_COMMUTATOR"
        cb = algebra.commutators(q, squares, fragments, caps)
        commutator = node("commutator_certificate_v1", {"kind": "CommutatorCertificate", "parents": parents, "data": cb})
        partial["commutator_certificate"] = commutator
        stage = "SPEC07_STEPS"
        comparison = steps.compare(read_rat(report["Lambda"]), read_rat(u["W"]),
                                   read_rat(k["K_SOS_upper"]), read_rat(cb["Cbar2"]), policy)
        comp = node("step_comparison_v1", {"kind": "ComparisonRecord", "parents": parents,
                    "certificate_refs": [upper["id"], ksos["id"], commutator["id"]],
                    "conditional_on_exact_ground_access": True,
                    "trial_is_ground_state": False, "data": comparison})
        payload = {"schema": VERSION, "scope": SCOPE, "completion": "SPEC04_07_SYNTHETIC_ONLY",
                   "source": source_identity(), "context": parents,
                   "sqrt_bits": sqrt_bits, "comparison_policy": policy,
                   "upper_certificate": upper, "ksos_certificate": ksos,
                   "commutator_certificate": commutator, "step_comparison": comp,
                   "limitations": dict(LIMITS),
                   "cost_ledger": {"status": "NOT_MEASURED_PER_ARTIFICIAL_CALL",
                                   "generator_cost": None, "checker_cost": None,
                                   "reason": "See test execution measurements and documented symbolic complexity; unknown is not zero",
                                   "information_pool": "ALL_VERIFIED_CERTIFICATES_SHARED_BY_BOTH_ROUTES"}}
        return {"schema": VERSION, "scope": SCOPE, "verdict": "EVIDENCE_GENERATED_UNVERIFIED",
                "evidence": node(NAMESPACE, payload), "limitations": dict(LIMITS)}
    except ContractError as exc:
        return failure(exc, stage, partial)
    except OSError as exc:
        return failure(ContractError("SOURCE_IO", "source", str(exc)), stage, partial)
