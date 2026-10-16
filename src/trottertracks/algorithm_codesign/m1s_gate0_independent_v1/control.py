"""Metadata-only interface. Execution remains closed in this preparation release.

No authorization, permit, marker, or numerical input is created here. A future
explicitly authorized integration must bind the published SOURCE and candidate
contract, input and saved comparator before enabling an isolated one-shot runner.
"""
import argparse
import json

BASE = "47da54279838a6b3e1497f36204ba7e53c4857f9"
INPUT_PATH = "artifacts/pr2_s0_s1_validation/2026-09-28/h4_1p00_rank12_development_v1.npz"
INPUT_SHA = "3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a"
H_SHA = "de7a549238e3a21f15a84018bef28440c345b31030282c01cf874f3d1d212424"
SCOPE = "M1S_GATE0_V5_INDEPENDENT_CORRECTNESS_V1_ONLY"
POLICY = {
    "schema": "m1s_gate0_independent_contract_candidate_v1",
    "status": "CANDIDATE_NOT_ADOPTED",
    "execution_enabled": False,
    "base_commit": BASE,
    "scope": SCOPE,
    "input": {"path": INPUT_PATH, "sha256": INPUT_SHA, "bytes": 8652},
    "model": {"molecule": "linear_H4", "distance_angstrom": "1.00",
              "basis": "sto-3g", "modes": 8, "df_rank": 12,
              "splits": [3, 6, 9], "delta_window": None,
              "mode_order": "mode0_MSB_alpha_even_beta_odd"},
    "arithmetic": {"coefficient": "Gaussian_integer_dyadic_full_Fock_Walsh_trace",
                   "state_metrics": "Decimal_80_digits_ROUND_HALF_EVEN",
                   "state_metric_evidence": "CONDITIONAL_NOT_CERTIFIED"},
    "tolerances": {"exact_algebra": "0", "basis_embedding": "0",
        "state_norm2": "2^-40", "probability_mass": "2^-46",
        "reference_metric_allowance": "2^-80", "W": "2^-24",
        "v": "2^-24", "rho": "2^-12",
        "coefficient_L1_Ha": "gamma_(713+tail_rank)*S_R",
        "support_or_sign_discrepancy": "STOP_FOR_INVESTIGATION_EVEN_IF_L1_SMALL",
        "rounded_law": "exact_binary64_rounding_relation",
        "rho_domain": "W_minus_reference_allowance_positive_and_interval_resolved"},
    "resources": {"workers": 1, "threads_per_library": 1, "gpu": False,
        "memory_bytes": 2147483648, "output_bytes": 134217728,
        "log_bytes": 16777216, "wall_time_seconds": None,
        "cpu_time_seconds": None, "max_exact_integer_bits": 16384,
        "max_primitive_events": 200000000,
        "max_coefficient_cells": 65536, "max_input_decode_bytes": 1048576},
    "future_comparator": {"saved_v5_result_sha256": None,
        "saved_v5_result_commit": None, "policy": "bind_before_reference_evaluation",
        "missing_metrics": "UNDETERMINED_NEVER_REEXECUTE_V5"},
    "requires_separate_explicit_user_execution_approval": True,
    "requires_new_execution_adapter_release": True,
    "mandatory_STOP": True,
}


class ExecutionClosed(PermissionError):
    pass


def require_execution(authorization=None):
    """Fail before input decoding or importing any reference numerical module.

    Deliberately no bypass flag or permit issuer in this preparation release.
    Even an old v5 authorization, or a new JSON claiming permission, cannot run it.
    """
    raise ExecutionClosed("PREPARATION_ONLY: separate explicit user approval and "
                          "a newly frozen execution adapter are required; old v5 authority is invalid")


def validate_contract(value):
    # Full literal equality rejects extra fields and result-dependent thresholds.
    return {"valid": type(value) is dict and value == POLICY,
            "execution_enabled": False, "mandatory_STOP": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", default="status",
                        choices=("status", "schema", "validate-config", "run"))
    parser.add_argument("--contract")
    parser.add_argument("--authorization")
    args = parser.parse_args(argv)
    if args.command == "run":
        try:
            require_execution(args.authorization)
        except ExecutionClosed as exc:
            print(json.dumps({"status": "REFUSED", "reason": str(exc), "mandatory_STOP": True}))
            return 2
    if args.command == "schema":
        print(json.dumps(POLICY, indent=2))
    elif args.command == "validate-config":
        from pathlib import Path
        def pairs(items):
            out = {}
            for key, value in items:
                if key in out:
                    raise ValueError("duplicate JSON key")
                out[key] = value
            return out
        raw = Path(args.contract).read_bytes()
        if len(raw) > 65536:
            raise ValueError("contract size limit")
        result = validate_contract(json.loads(raw, object_pairs_hook=pairs))
        print(json.dumps(result))
        return 0 if result["valid"] else 1
    else:
        print(json.dumps({"status": "IMPLEMENTED_UNEXECUTED_PREPARATION_ONLY",
                          "execution_enabled": False, "mandatory_STOP": True}))
    return 0
