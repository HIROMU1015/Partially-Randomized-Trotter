"""Read-only metadata preflight. No projection of scientific values or model calls."""
from __future__ import annotations

import csv
import io
import json

from .ax1b_contract import AXES, FLAGS, METRICS, digest, require, safe_path, select_fields, sha256
from .ax1b_data import canonical_compiler_identity, check_membership, join_m1, pm1_features, validate_candidate_scope


class MetadataReader:
    """Explicit metadata permit; cannot be passed to the analysis VerifiedReader."""
    def __init__(self, root, allowlist, metadata_only=False):
        require(metadata_only is True, "AUTHORIZATION", "explicit metadata-only preflight required")
        self.root = root
        self.entries = {e["path"]: e for e in allowlist["entries"]}
        require(len(self.entries) == len(allowlist["entries"]), "SCHEMA", "duplicate allowlist path")
        self.audit = []

    def read(self, path):
        require(path in self.entries, "INPUT_IDENTITY", "preflight path outside exact allowlist")
        entry = self.entries[path]
        raw = safe_path(self.root, path).read_bytes()
        record = dict(path=path, sha256=sha256(raw), hash_matches=sha256(raw) == entry["sha256"],
                      schema_checked=False, embedded_paths_followed=False)
        self.audit.append(record)
        require(record["hash_matches"], "INPUT_IDENTITY", "preflight byte hash differs")
        schema = entry["schema"]
        value = None
        if schema["format"] == "json":
            value = json.loads(raw)
            require(type(value) is dict and sorted(value) == schema["root_keys"]
                    and value.get("schema_version") == schema["schema_version"], "SCHEMA", "preflight JSON root/version")
            for field in entry["planned_fields"]:
                matches = select_fields(value, field["selector"])
                require(not field["required"] or bool(matches), "SCHEMA", "preflight required field: " + field["selector"])
                if "static_presence" in field:
                    presence = field["static_presence"]
                    require(len(matches) == presence["matched_value_count"]
                            and sum(x is None for x in matches) == presence["null_value_count"], "SCHEMA", "preflight field cardinality")
        elif schema["format"] == "csv":
            require(next(csv.reader(io.StringIO(raw.decode()))) == schema["ordered_header"], "SCHEMA", "preflight CSV header")
        else:
            require(schema["format"] in {"markdown", "python_source_text", "toml"}, "SCHEMA", "unregistered input format")
        record["schema_checked"] = True
        return value


def _numeric_shape(value):
    # No range/accuracy calculation, averaging, normalization, or cost use.
    return type(value) in (int, float)


def _signal_shape(candidate, signal):
    required = ("normalization_multiplier", "axis_bias")
    require(all(k in signal for k in required), "SCHEMA", "signal reference field missing")
    require(signal["normalization_multiplier"] is None or _numeric_shape(signal["normalization_multiplier"]),
            "SCHEMA", "normalization field type")
    require(type(signal["axis_bias"]) is dict and set(signal["axis_bias"]) == {"real", "imag"}
            and all(v is None or _numeric_shape(v) for v in signal["axis_bias"].values()), "SCHEMA", "axis bias field shape")
    fields = {}
    for name in ("n_det", "n_fixed"):
        value = signal.get(name)
        require(value is None or _numeric_shape(value), "SCHEMA", "action metadata type")
        fields[name] = "SAVED_FIELD_AVAILABLE" if value is not None else "N_A_SAVED_FIELD_MISSING"
    if candidate["method"] in {"B0", "B1"}:
        fields["E_rand"] = "DETERMINISTIC_SOURCE_ZERO"
    elif signal.get("expected_random_applications_exact") is not None:
        require(_numeric_shape(signal["expected_random_applications_exact"]), "SCHEMA", "expected action field type")
        fields["E_rand"] = "SAVED_FIELD_AVAILABLE"
    else:
        distribution = signal.get("finite_distribution") or {}
        orders, probabilities = distribution.get("orders"), distribution.get("order_probabilities")
        available = type(orders) is list and type(probabilities) is list and bool(orders) and len(orders) == len(probabilities)
        if available:
            require(all(type(n) is int for n in orders) and all(_numeric_shape(p) for p in probabilities),
                    "SCHEMA", "finite distribution field type")
        fields["E_rand"] = "STRUCTURAL_FALLBACK_AVAILABLE_NOT_EVALUATED" if available else "N_A_INPUTS_MISSING"
    return fields


def _axis_shapes(axes):
    require(type(axes) is dict and set(axes) == set(AXES), "SCHEMA", "two compiled axes required")
    for axis in AXES:
        row = axes[axis]
        require(row.get("state_preparation_included") is False and row.get("measurement_included") is True
                and row.get("backend_execution_included") is False, "INPUT_IDENTITY", "preflight wrapper scope")
        require(row.get("trajectory_records_truncated") is False, "PAIR_IDENTITY", "truncated trajectory metadata")
        require(type(row.get("cost")) is dict and all(m in row["cost"] and _numeric_shape(row["cost"][m]) for m in METRICS),
                "SCHEMA", "compiled cost field shape")
        require(type(row.get("retained_trajectory_records")) is list, "SCHEMA", "trajectory records missing")
        for pair in row["retained_trajectory_records"]:
            require(all(k in pair for k in ("trajectory_index", "trajectory_seed", "step_seeds", "evolution_circuit_semantics_fingerprint", "cost")),
                    "PAIR_IDENTITY", "trajectory identity field missing")
            require(type(pair["cost"]) is dict and all(m in pair["cost"] and _numeric_shape(pair["cost"][m]) for m in METRICS),
                    "SCHEMA", "trajectory cost field shape")


def inspect_metadata(values, allowlist):
    """Identity/membership and field shape only; never call project_saved/analyze/fit."""
    def by_schema(name):
        matches = [values[e["path"]] for e in allowlist["entries"] if e["schema"].get("schema_version") == name]
        require(len(matches) == 1, "SCHEMA", "unique saved input schema required")
        return matches[0]

    A = by_schema("pr2_matched_accuracy_m1_a_result_v2")
    B = by_schema("pr2_matched_accuracy_m1_b1_result_v2")
    P = by_schema("track_a_pm1_discard_result_v1")
    M = by_schema("pr2_matched_accuracy_m2_transfer_result_v2")
    anchors = join_m1(A, B)
    membership = allowlist["membership"]
    check_membership([x[0] for x in anchors], membership["TRAIN_M1_210"]["rows"], "M1")
    check_membership([r["candidate"] for r in P["candidate_records"]], membership["DIAG_PM1_8"]["rows"], "PM1")
    dataset_rows = {name: [] for name in ("TRAIN_M1_210", "DIAG_PM1_8", "DIAG_M2_5")}
    pm1 = []
    for candidate, signal, compiled in anchors:
        validate_candidate_scope(candidate)
        shape = _signal_shape(candidate, signal)
        _axis_shapes(compiled["compiled_axes"])
        dataset_rows["TRAIN_M1_210"].append(dict(candidate_fingerprint=candidate["candidate_fingerprint"], feature_fields=shape))
    for row in P["candidate_records"]:
        candidate, signal = row["candidate"], row["signal"]
        require(candidate["candidate_fingerprint"] == signal["candidate_fingerprint"] == row["signal_cost_candidate_fingerprint"],
                "INPUT_IDENTITY", "PM1 signal/cost fingerprint differs")
        before = digest(candidate)
        validate_candidate_scope(candidate)
        comparison = canonical_compiler_identity(candidate["compiler_identity"])
        # Only I1 action metadata n_fixed is shared; no reference or cost fields enter.
        features, provenance = pm1_features(candidate, anchors)
        shape = _signal_shape(candidate, signal)
        shape.update(n_det="SOURCE_INTEGER_ACTION_INVARIANT_AVAILABLE", E_rand="DETERMINISTIC_SOURCE_ZERO",
                     n_fixed=provenance["n_fixed_status"])
        _axis_shapes(row["axes"])
        require(digest(candidate) == before, "INPUT_IDENTITY", "raw PM1 candidate changed")
        pm1.append(dict(candidate_fingerprint=candidate["candidate_fingerprint"], raw_compiler_identity=candidate["compiler_identity"],
                        canonical_comparison_identity=comparison, raw_identity_preserved=True,
                        anchor_fingerprints=provenance["anchor_fingerprints"], n_fixed_status=provenance["n_fixed_status"],
                        n_fixed_shared=features.n_fixed is not None, anchor_input_layer="TRAIN_M1_210_ONLY",
                        other_identity_checks_unchanged=True, no_truth_or_cost_imputation=True))
        dataset_rows["DIAG_PM1_8"].append(dict(candidate_fingerprint=candidate["candidate_fingerprint"], feature_fields=shape))
    m2_candidates = []
    development = {x[0]["candidate_fingerprint"] for x in anchors}
    for row in M["candidate_results"]:
        signal = row["signal"]
        raw = signal["candidate"]
        require(raw["candidate_fingerprint"] == signal["candidate_fingerprint"] == row["execution_candidate_fingerprint"],
                "INPUT_IDENTITY", "M2 execution fingerprint differs")
        require(row["development_candidate_fingerprint"] in development, "INPUT_IDENTITY", "M2 development fingerprint missing")
        snapshot = M["input_snapshot_identity"]
        candidate = dict(raw, **{k: snapshot[k] for k in ("hamiltonian_hash", "state_hash", "state_vector_hash")},
                         snapshot_sha256=snapshot["file_sha256"], T_hex=None, delta_hex=None, T_literal=raw["T"], delta_literal=raw["delta"])
        m2_candidates.append(candidate)
        validate_candidate_scope(candidate)
        shape = _signal_shape(candidate, signal)
        means = row["axis_one_shot_compiled_means"]
        require(type(means) is dict and set(means) == set(AXES)
                and all(type(means[a]) is dict and all(m in means[a] and _numeric_shape(means[a][m]) for m in METRICS) for a in AXES),
                "SCHEMA", "M2 compiled mean field shape")
        pairs = row["compiled"]["paired_trajectory_rows"]
        require(type(pairs) is list, "PAIR_IDENTITY", "M2 paired rows missing")
        for pair in pairs:
            require(all(k in pair for k in ("trajectory_index", "trajectory_seed", "axes")) and set(pair["axes"]) == set(AXES),
                    "PAIR_IDENTITY", "M2 paired identity missing")
            for axis in AXES:
                node = pair["axes"][axis]
                require(all(k in node for k in ("step_seeds", "shared_evolution_fingerprint", "metrics"))
                        and all(m in node["metrics"] and _numeric_shape(node["metrics"][m]) for m in METRICS), "SCHEMA", "M2 pair field shape")
        dataset_rows["DIAG_M2_5"].append(dict(candidate_fingerprint=candidate["candidate_fingerprint"], feature_fields=shape))
    check_membership(m2_candidates, membership["DIAG_M2_5"]["rows"], "M2")
    fingerprints = [r["candidate_fingerprint"] for rows in dataset_rows.values() for r in rows]
    require(len(fingerprints) == len(set(fingerprints)), "INPUT_IDENTITY", "cross-dataset duplicate fingerprint")
    for name, rows in dataset_rows.items():
        require(len(rows) == len(membership[name]["rows"]), "INPUT_IDENTITY", "dataset cardinality")
    return dict(precheck_status="PRECHECK_PASS", datasets={k: dict(count=len(v), status="PRECHECK_PASS", candidates=v) for k, v in dataset_rows.items()},
                pm1_anchor_audit=pm1, fingerprint_check="EXACT_FROZEN_MEMBERSHIP_AND_CROSS_RECORD_MATCH_NO_RECOMPUTATION",
                actual_feature_values_exported=False,
                declared_unverified=[dict(check=name, status="PRECHECK_UNVERIFIED") for name in (
                    "finite_normalization_value_reconciliation", "PM2_reaccounting", "model_fit_and_NNLS", "cost_prediction_errors",
                    "shot_and_eligibility_calculation", "regret", "paired_cost_statistics", "independent_recompile_gate_sequence_identity")])


def run_preflight(root, request, input_audit):
    """Request is results-prior metadata authorization, with all analysis flags false."""
    from .ax1b_contract import load_contract
    from .ax1b_execution import environment
    require(request.get("schema_version") == "track_a_ax1b_metadata_precheck_request_v1"
            and request.get("metadata_only_preflight_authorized") is True
            and all(request.get(k) is v for k, v in FLAGS.items()), "AUTHORIZATION", "preflight-only permission required")
    for entry in request["source_files"]:
        require(sha256(safe_path(root, entry["path"]).read_bytes()) == entry["sha256"], "IMPLEMENTATION", "preflight source bytes differ")
    allow, plan = load_contract(root)
    require(request["model_configuration_sha256"] == plan["model_configuration_sha256"], "CONTRACT_CONFLICT", "preflight model configuration")
    require(request["environment"] == environment(), "ENVIRONMENT", "preflight environment differs")
    proof_ref = request["synthetic_test_audit"]
    proof_raw = safe_path(root, proof_ref["path"]).read_bytes()
    require(sha256(proof_raw) == proof_ref["sha256"], "IMPLEMENTATION", "preflight synthetic proof hash")
    proof = json.loads(proof_raw)
    require(proof.get("environment") == request["environment"] and proof.get("exit_code") == 0 and proof.get("passed", 0) > 0
            and all(proof.get(k) == 0 for k in ("failed", "skipped", "protected_access_attempts", "scientific_import_attempts"))
            and proof.get("real_data_fit_executed") is False and proof["source_files_after_successful_test"] == request["source_files"],
            "IMPLEMENTATION", "same-source/environment synthetic proof differs")
    reader = MetadataReader(root, allow, metadata_only=True)
    # Share the partial audit even if a later hash/schema/identity gate fails.
    reader.audit = input_audit
    values = {e["path"]: reader.read(e["path"]) for e in allow["entries"]}
    result = inspect_metadata(values, allow)
    for entry in allow["entries"] + request["source_files"]:
        require(sha256(safe_path(root, entry["path"]).read_bytes()) == entry["sha256"], "INPUT_IDENTITY", "bytes changed during preflight")
    return result
