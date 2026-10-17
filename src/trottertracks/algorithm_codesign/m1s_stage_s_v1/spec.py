"""Preregisterable proposals, not adoption or scientific authorization."""
from itertools import product

BASE = "fb5519bf3f44392dc050feb9db44057dbc5207b1"
ART = "artifacts/track_b_m1s_stage_s_preparation/2026-10-17/v1"
PACKAGE = "src/trottertracks/algorithm_codesign/m1s_stage_s_v1"
REVIEW = "docs/research/m1s_stage_s_scientific_conditions_review_20261017.md"
REVIEW_SHA256 = "f815e886d9e4ff458a2a1398c3b41c4be2e4187e357ba58069eea9c3136b492e"
OUTPUT_PREFIX = "artifacts/track_b_m1s_stage_s_result/"
AUTH_PREFIX = "artifacts/track_b_m1s_stage_s_execution_authorization/"
EVIDENCE = "CONDITIONAL_ENGINEERING_NOT_CERTIFIED"
SOURCE_FILES = [PACKAGE + "/" + n for n in [
    "__init__.py", "spec.py", "arithmetic.py", "kernels.py", "adapter.py",
    "control.py", "analysis.py"]] + [
    "scripts/tracks/algorithm_codesign/run_m1s_stage_s_v1.py",
    ART + "/result_schema.json"]
INPUTS = {
    "V1": {
        "path": "artifacts/track_b_m1s_saved_signal_gamma_v2_result/2026-10-17/completion_attempt_02/result.json",
        "commit": "b34572770e7c2105e16da003af050d789f490662",
        "sha256": "bb2d09e6963fdc8b8871170ba28e98c36f4e353f452e50870103a7e97bf2e55d",
        "bytes": 146764,
        "schema": "m1s_saved_signal_gamma_v1_result",
        "status": "GAMMA_CONDITIONS_COMPLETE_AWAITING_SCIENTIFIC_JUDGMENT",
        "provenance": {"S": "677dd6957e0e1913b70ca5f999458835ae8a2e23",
                       "C": "2cb5160d9051a6568b8acc0cfb29fc169bdfca09",
                       "A": "161ee8c3dc82c7fa19e0e1cfb79d71d3c236a03c"},
    },
    "PI3": {
        "path": "artifacts/track_b_m1s_gamma_pi3_result/2026-10-17/attempt_01/result.json",
        "commit": "81d6103ec89d485cb7336c3a90bd95dad66992aa",
        "sha256": "693ce941bf7793a6104af28443746f43e05d4903118c0ebbb6f09729f389fadf",
        "bytes": 37054,
        "schema": "m1s_gamma_pi3_v1_result",
        "status": "PI3_COMPARISON_COMPLETE_AWAITING_SCIENTIFIC_JUDGMENT",
        "provenance": {"S": "2f800f179c7b29589b2f9a7e532408b16b8fe6d2",
                       "C": "bf97fcb2607904f8b96a625c5b64440a78bbbfdc",
                       "A": "a1dd87a1f81467bea0ce27cacf9d21e49cff7908"},
    },
}
# These are proposed values. The candidate's selected_science remains null.
PROPOSED_SCIENCE = {
    "task": "TASK_A_SAVED_RAW_H_IDEAL_MEAN_ONLY",
    "gamma_policy": "OPEN_PI_OVER_THREE_CAP_LEFT_LIMIT_NO_GRID",
    "confidence_scope": "PER_METHOD_ROW",
    "alpha_row": "1/20",
    "statistics": "FRESH_IID_HOEFFDING_EQUAL_X_Y",
    "coverage": "SAVED_CONDITIONAL_RADII_IDEAL_MEAN_ONLY",
    "arithmetic_bits": 128,
    "integer_policy": "FLOOR_AT_CAP_PLUS_ONE_BOTH_INTEGERS_RESOLVED",
    "comparison": "ALL_48_REGISTRY_24_POSTHOC_ELIGIBLE_8_NEGATIVE_16_G9_CARRY",
    "scientific_decision": "DESCRIPTIVE_ONLY_NO_AUTOMATIC_GO",
}
RESOURCES = {
    "cpu_time_cap_seconds": None, "wall_time_cap_seconds": None,
    "cpu_core_cap": None, "worker_cap": None, "gpu_cap": None,
    "memory_cap_bytes": None,
    "worker_policy": "AUTO_AFFINITY_AND_INDEPENDENT_CELL_COUNT",
    "backend": "EXACT_RATIONAL_CPU_NO_FLOAT_BLAS_GPU_KERNEL",
    "input_bytes_per_file": 16 * 1024**2,
    "result_bytes": 16 * 1024**2, "log_bytes": 16 * 1024**2,
    "max_rational_bits": 65536, "max_rational_text_chars": 25000,
    "max_fraction_fields": 1024, "max_series_terms_per_call": 512,
    "max_bound_steps": 100000, "max_comparisons": 48,
    "max_evaluated_contrasts": 24, "max_cell_jobs": 12,
    "max_margin_evaluations": 36, "max_sin_endpoint_calls": 72,
    "max_log_calls": 1, "max_refinements": 0,
}


def row_keys():
    return list(product(["G3", "G6", "G9"], [2, 4], ["1/8", "1/16"],
                        ["0", "1", "4"], ["shared_pm", "ordinary", "independent_pm"]))


def contrast_keys():
    return list(product(["G3", "G6", "G9"], [2, 4], ["1/8", "1/16"],
                        ["1", "4"], ["ordinary", "independent_pm"]))


def old_status(key):
    if key[0] == "G9":
        return "G9_ABSOLUTE_GAIN_UNRESOLVED_NOT_TESTED"
    if key[0] == "G6" and key[3] == "1":
        return "NOT_SATISFIED"
    return "SATISFIED"
