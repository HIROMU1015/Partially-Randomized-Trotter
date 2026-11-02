"""Explicit pilot scope with the scope-versioned prerequisite checker."""

from ..a2_spec04_07.common import *
from ..a2_spec04_07 import common as frozen
from .core01_checker import check_spec01_03
from .identity import source_identity

VERSION = "A2_PILOT_SPEC04_07_V1"
SCOPE = "FIXED_DF_PILOT_CORE_ONLY_NOT_EXECUTION_AUTHORITY"
NAMESPACE = "a2_pilot_spec04_07_evidence_v1"


def prerequisite(raw, base, number, order, model_id):
    report = check_spec01_03(raw, base, expected_particle_number=number,
                            expected_fragment_order=order, expected_model_id=model_id)
    if report["verdict"] == "IDENTITY_MISMATCH_INCOMPARABLE":
        raise ContractError("CONTEXT_MISMATCH", "SPEC01_03", "Prerequisite context differs")
    require(report["verdict"] == "SPEC01_03_VERIFIED_EXACT_FIXED_DF", "BASE_NOT_VERIFIED",
            "SPEC01_03", "Prerequisite rejected: " + report["verdict"])
    body = detached(base)["payload"]
    return body, body["intake"]["model"]["payload"], report


def failure(exc, stage, partial):
    record = frozen.failure(exc, stage, partial)
    record.update(schema=VERSION, scope=SCOPE)
    return record
