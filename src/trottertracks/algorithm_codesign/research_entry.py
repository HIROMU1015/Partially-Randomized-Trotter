"""Track B development/reference entry with independent mathematical scopes.

No generator, interval bound, saved-result analysis or science runner is called.
Only G6 exact descriptor serialization and existing JSON storage are shared.
"""
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from .g6_f3_contract import Evidence
from .df_correction_comparison import storage
from .m1s_saved_signal_gamma_v1 import spec as gamma_spec
from .m1s_gamma_pi3_v1 import spec as pi3_spec

FROZEN_BASE = "5f750a23f023d75c8c4fbcabc6937d6953ce61f2"
M1_REFERENCES = {'gamma': {'reference_commit': 'e7791c9ffe59577528e6095da5b2419bf6fadda4', 'scientific_source': 'f2a5611ca9fc7461c64f2deb74a1f624ac8aa7c5', 'contract': 'artifacts/track_b_m1s_v1_gamma_formal_adoption/2026-10-17/v1/execution_contract_adopted.json', 'status': 'ADOPTED_FOR_SEPARATE_EXECUTION_APPROVAL', 'evidence': 'CONDITIONAL_ENGINEERING_NOT_CERTIFIED', 'input': {'path': 'artifacts/track_b_m1s_finite_signal_v2_result/2026-10-16/completion_attempt_02/result.json', 'commit': '025c5a90888b02b73d2815fbb10d6bd1d2823602', 'bytes': 454422, 'sha256': 'ff6c5192d8cca769cd61a1478cea01671d058a85a8f6a4cdcdc9baa672a56fe0', 'schema': 'm1s_finite_signal_result_v2', 'status': 'PILOT_COMPLETE_AWAITING_SCIENTIFIC_JUDGMENT'}, 'one_shot': True, 'retry': False, 'mandatory_STOP': True}, 'pi3': {'reference_commit': 'b17b056deec60820413c0881978f7b77caf0861b', 'scientific_source': '2f800f179c7b29589b2f9a7e532408b16b8fe6d2', 'contract': 'artifacts/track_b_m1s_gamma_pi3_formal_adoption/2026-10-17/v1/execution_contract_adopted.json', 'status': 'ADOPTED_FOR_SEPARATE_EXECUTION_APPROVAL', 'evidence': 'CONDITIONAL_ENGINEERING_NOT_CERTIFIED', 'input': {'path': 'artifacts/track_b_m1s_saved_signal_gamma_v2_result/2026-10-17/completion_attempt_02/result.json', 'commit': 'b34572770e7c2105e16da003af050d789f490662', 'bytes': 146764, 'sha256': 'bb2d09e6963fdc8b8871170ba28e98c36f4e353f452e50870103a7e97bf2e55d', 'schema': 'm1s_saved_signal_gamma_v1_result', 'status': 'GAMMA_CONDITIONS_COMPLETE_AWAITING_SCIENTIFIC_JUDGMENT', 'provenance': {'S': '677dd6957e0e1913b70ca5f999458835ae8a2e23', 'C': '2cb5160d9051a6568b8acc0cfb29fc169bdfca09', 'A': '161ee8c3dc82c7fa19e0e1cfb79d71d3c236a03c'}}, 'one_shot': True, 'retry': False, 'mandatory_STOP': True}}


class EntryError(ValueError):
    """Unbound or incompatible research context."""


def describe(study, *, stage="reference"):
    if stage not in ("reference", "descriptor", "mock"):
        raise EntryError("Only reference/descriptor/mock; scientific execution is closed")
    if not isinstance(study, str) or study not in ("G6", "M1-S-gamma", "M1-S-pi3"):
        raise EntryError("Select one independent study")
    common = {"schema": "track_b_research_entry_v1", "study": study, "stage": stage,
              "scientific_execution": "DEFERRED_SEPARATE_CONTRACT_AND_PERMISSION",
              "scientific_GO": None, "numerical_equivalence": "NOT_TESTED",
              "stop": "MANDATORY_STOP_NO_AUTOMATIC_EXTENSION"}
    if study == "G6":
        return {**common, "reference_commit": FROZEN_BASE,
                "scientific_source": "1aea9b8c228c5b940f2fcba0f94493b3ffdec46e",
                "implementation": "trottertracks.algorithm_codesign.g6_f3_generator:prepare",
                "descriptor_api": "trottertracks.algorithm_codesign.g6_f3_contract:Evidence",
                "scope": "FINITE_POSITIVE_RATIONAL_P_RATIONAL_0_LT_X_LE_1_ODD_M",
                "guarantee": "CONDITIONAL_SUPPLIED_PRECISION_NOT_PHYSICAL_PROVIDER_CERTIFICATION",
                "record_locator": "docs/research/track_b_g6_v1_2_draft_review_integration_20261022/handoff.md",
                "stop_locator": "docs/research/track_b_g6_v1_2_draft_review_integration_20261022/completion_state.json",
                "result_locator": "docs/research/track_b_g6_v1_2_draft_review_integration_20261022/source_inventory.json",
                "generator": "trottertracks.algorithm_codesign.g6_f3_generator:GeneratorSession",
                "native_inventory": "trottertracks.algorithm_codesign.df_correction_comparison.native",
                "construction": "trottertracks.algorithm_codesign.return_aggregation",
                "historical_state": "FIXED_DRAFT_AND_PER_STUDY_STOP_PRESERVED"}
    key, spec = (("gamma", gamma_spec) if study == "M1-S-gamma" else ("pi3", pi3_spec))
    fixed = deepcopy(M1_REFERENCES[key])
    # Specs retain their original candidate defaults. The fixed adopted contract,
    # not spec.template(), is the normative execution-condition locator.
    return {**common, **fixed, "implementation": spec.PACKAGE,
            "descriptor_api": spec.PACKAGE.replace("src/", "").replace("/", ".") + ".spec",
            "scope": ("CONTINUOUS_GAMMA_SAVED_IDEAL_SIGNAL_CONDITIONAL_RADII"
                      if key == "gamma" else "SAVED_GAMMA_BOUNDARY_PI_OVER_THREE_INTERSECTION"),
            "guarantee": spec.EVIDENCE, "forecast_kind": spec.FORECAST,
            "input_locator": deepcopy(spec.INPUT),
            "result_locator_status": "UPSTREAM_INPUT_EVIDENCE_NOT_RESULT_OF_NEW_ENTRY",
            "stop_locator": fixed["contract"] + "#/mandatory_STOP",
            "record_locator": fixed["contract"],
            "runner": "scripts/tracks/algorithm_codesign/run_" +
                      ("m1s_saved_signal_gamma_v1.py" if key == "gamma" else "m1s_gamma_pi3_v1.py"),
            "numerical_api": ("m1s_saved_signal_gamma_v1.analysis" if key == "gamma"
                              else "m1s_gamma_pi3_v1.compare:evaluate"),
            "historical_state": fixed["status"], "physical_success_guaranteed": False}


@dataclass(frozen=True)
class MockContext:
    study: str
    scientific_source: str
    scope: str
    cohort: str = "ARTIFICIAL_ONLY"
    status: str = "READY"


def artificial_context(study):
    descriptor = describe(study)
    return MockContext(study, descriptor["scientific_source"], descriptor["scope"])


def run_mock(context, *, stage="mock"):
    if type(context) is not MockContext or stage != "mock":
        raise EntryError("Explicit artificial context and mock stage required")
    descriptor = describe(context.study, stage=stage)
    if (context.scientific_source != descriptor["scientific_source"]
            or context.scope != descriptor["scope"]
            or context.cohort != "ARTIFICIAL_ONLY" or context.status != "READY"):
        raise EntryError("Source/scope/cohort/status mismatch; no fallback or resume")
    # Exact descriptor serialization only: not G6 mathematical evidence for M1.
    evidence = Evidence.create({"study": context.study, "source": context.scientific_source,
                                "scope": context.scope, "cohort": context.cohort,
                                "payload_kind": "ARTIFICIAL_DESCRIPTOR_NO_SCIENTIFIC_VALUES"})
    return {**descriptor, "technical_status": "MOCK_DESCRIPTOR_ROUNDTRIP_ONLY",
            "artificial_record": evidence.read(), "descriptor_digest": evidence.digest,
            "scientific_values": None, "physical_success_guaranteed": False}


def write_record(directory, record):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "record.json"
    storage.dump_json(path, record, replace_existing=False)
    if storage.load_json(path) != record:
        raise EntryError("Existing storage round-trip mismatch")
    return path
