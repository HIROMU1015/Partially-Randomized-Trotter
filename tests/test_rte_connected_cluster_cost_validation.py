from __future__ import annotations

import json
from copy import deepcopy

import numpy as np
import pytest
import qiskit

from trotterlib.df_hamiltonian import DFHamiltonian
from trotterlib.rte import CompilerSettings
from trotterlib.rte_connected_cluster_cost_validation import (
    CONNECTED_CLUSTER_TASK_IMPLEMENTATION_VERSION,
    _fingerprint,
    _merge_sample_statistics,
    _run_tasks,
    _sample_chunks,
    _sequence_form,
    calibrate_and_validate_connected_cluster_k4,
    calibrate_connected_cluster_cost_model,
    diagnose_connected_cluster_k4_extrapolation,
    load_connected_cluster_calibration,
    load_connected_cluster_hamiltonian_snapshot,
    predict_connected_cluster_cost,
    supplement_connected_cluster_holdout_precision,
    validate_connected_cluster_payload,
    validate_connected_cluster_calibration_holdout,
    validate_connected_cluster_calibration_payload,
    validate_connected_cluster_k4_extrapolation_payload,
    validate_connected_cluster_k4_calibration_payload,
    validate_connected_cluster_supplement_payload,
    validate_connected_cluster_transfer_payload,
    validate_operational_connected_cluster_cost_model,
    write_connected_cluster_hamiltonian_snapshot,
    write_connected_cluster_calibration,
    write_connected_cluster_k4_extrapolation_diagnostic,
    write_connected_cluster_k4_calibration,
    write_connected_cluster_validation,
    write_connected_cluster_transfer_validation,
)
from trotterlib.rte_cost_angle_invariance_validation import (
    validate_rte_cost_angle_invariance,
    validate_rte_cost_angle_invariance_payload,
)


def _compiler() -> CompilerSettings:
    return CompilerSettings(
        basis_gates=("rz", "sx", "x", "cx"),
        backend_name=None,
        coupling_map=None,
        optimization_level=1,
        layout_method=None,
        routing_method=None,
        transpiler_seed=17,
        qiskit_version=qiskit.__version__,
    )


def _hamiltonian() -> DFHamiltonian:
    return DFHamiltonian(
        constant=0.11,
        one_body=np.asarray([[0.2]], dtype=np.complex128),
        lambdas=np.asarray([0.7]),
        g_matrices=(np.asarray([[1.0]], dtype=np.complex128),),
        metadata={"name": "connected-cluster-cost-validation-toy"},
    )


def test_sequence_form_counts_every_local_window() -> None:
    form = _sequence_form((0, 2, 0, 0))

    assert form["k1:0"] == 3.0
    assert form["k1:2"] == 1.0
    assert form["k2:0,2"] == 1.0
    assert form["k2:2,0"] == 1.0
    assert form["k2:0,0"] == 1.0
    assert form["k3:0,2,0"] == 1.0
    assert form["k3:2,0,0"] == 1.0


def test_sample_chunks_use_stable_fixed_indices() -> None:
    assert _sample_chunks(5, 2) == ((0, 0, 2), (1, 2, 4), (2, 4, 5))
    assert _sample_chunks(7, 2) == (
        (0, 0, 2),
        (1, 2, 4),
        (2, 4, 6),
        (3, 6, 7),
    )


def test_merge_sample_statistics_matches_raw_samples() -> None:
    first_values = [1.0, 3.0, 5.0]
    second_values = [2.0, 8.0]

    def summary(values):
        array = np.asarray(values)
        return {
            "mean": float(np.mean(array)),
            "unbiased_sample_variance": float(np.var(array, ddof=1)),
            "standard_error": float(np.std(array, ddof=1) / np.sqrt(array.size)),
            "minimum": float(np.min(array)),
            "maximum": float(np.max(array)),
        }

    merged = _merge_sample_statistics(
        summary(first_values),
        len(first_values),
        summary(second_values),
        len(second_values),
    )
    expected = summary(first_values + second_values)
    assert merged == pytest.approx(expected)


def test_connected_cluster_hamiltonian_snapshot_round_trip(tmp_path) -> None:
    path = tmp_path / "hamiltonian.npz"
    write_connected_cluster_hamiltonian_snapshot(_hamiltonian(), path)

    restored = load_connected_cluster_hamiltonian_snapshot(path)

    assert restored.constant == _hamiltonian().constant
    assert np.array_equal(restored.one_body, _hamiltonian().one_body)
    assert np.array_equal(restored.lambdas, _hamiltonian().lambdas)
    assert np.array_equal(restored.g_matrices[0], _hamiltonian().g_matrices[0])
    assert restored.metadata == _hamiltonian().metadata


def test_checkpoint_recomputes_after_implementation_version_change(tmp_path) -> None:
    task = {
        "hamiltonian": _hamiltonian(),
        "compiler": _compiler(),
        "cluster_length": 1,
        "payload": "fixed",
    }
    calls = []

    def execute(current):
        calls.append(current["payload"])
        return {"result": len(calls)}

    first = _run_tasks(
        [task],
        1,
        execute,
        checkpoint_directory=tmp_path,
        checkpoint_stage="audit",
    )
    checkpoint = tmp_path / "audit_k1.json"
    envelope = json.loads(checkpoint.read_text(encoding="utf-8"))
    envelope["task_implementation_version"] = "obsolete-implementation"
    checkpoint.write_text(json.dumps(envelope), encoding="utf-8")

    second = _run_tasks(
        [task],
        1,
        execute,
        checkpoint_directory=tmp_path,
        checkpoint_stage="audit",
    )

    assert first == [{"result": 1}]
    assert second == [{"result": 2}]
    assert calls == ["fixed", "fixed"]
    assert len(list(tmp_path.glob("audit_k1*.json"))) == 2


def test_operational_connected_cluster_uses_disjoint_stages_and_holdouts(
    tmp_path,
) -> None:
    checkpoint_directory = tmp_path / "checkpoints"
    payload = validate_operational_connected_cluster_cost_model(
        _hamiltonian(),
        ld=0,
        reference_delta_time=0.08,
        reference_rte_steps=4,
        compiler=_compiler(),
        pilot_sample_count=2,
        minimum_production_sample_count=2,
        maximum_production_sample_count=4,
        prediction_relative_standard_error_target=0.05,
        allocation_safety_factor=1.0,
        holdout_zero_sample_count=2,
        holdout_single_rare_sample_count=2,
        seed=701,
        maximum_workers=2,
        checkpoint_directory=checkpoint_directory,
        provenance={"test": True},
    )

    assert set(payload["pilot"]) == {"1", "2", "3"}
    assert set(payload["production"]) == {"1", "2", "3"}
    assert set(payload["holdout"]) == {"4", "6", "8"}
    assert len(payload["production"]["3"]["strata"]) == 8
    assert len(payload["holdout"]["8"]["strata"]) == 9
    all_seeds = {
        stratum["seed"]
        for role in ("pilot", "production", "holdout")
        for stage in payload[role].values()
        for stratum in stage["strata"].values()
    }
    assert len(all_seeds) == 14 + 14 + 5 + 7 + 9
    comparison = payload["holdout_comparisons"]["8"][
        "exactly_one_order2_condition"
    ]["metrics"]["rz_count"]
    assert comparison["prediction_minus_actual"] == pytest.approx(
        comparison["prediction"] - comparison["actual"]
    )
    validate_connected_cluster_payload(payload)
    checkpoints = sorted(checkpoint_directory.glob("*.json"))
    assert len(checkpoints) == 9
    checkpoint_mtimes = {path: path.stat().st_mtime_ns for path in checkpoints}
    resumed = validate_operational_connected_cluster_cost_model(
        _hamiltonian(),
        ld=0,
        reference_delta_time=0.08,
        reference_rte_steps=4,
        compiler=_compiler(),
        pilot_sample_count=2,
        minimum_production_sample_count=2,
        maximum_production_sample_count=4,
        prediction_relative_standard_error_target=0.05,
        allocation_safety_factor=1.0,
        holdout_zero_sample_count=2,
        holdout_single_rare_sample_count=2,
        seed=701,
        maximum_workers=2,
        checkpoint_directory=checkpoint_directory,
        provenance={"test": True},
    )
    assert resumed["pilot"] == payload["pilot"]
    assert resumed["production"] == payload["production"]
    assert resumed["holdout"] == payload["holdout"]
    assert checkpoint_mtimes == {
        path: path.stat().st_mtime_ns for path in checkpoints
    }
    output = tmp_path / "connected.json"
    write_connected_cluster_validation(payload, output)
    assert output.exists()

    tampered = deepcopy(payload)
    tampered["configuration"]["finite_taylor_order"] = 0
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_connected_cluster_payload(tampered)

    supplement = supplement_connected_cluster_holdout_precision(
        payload,
        _hamiltonian(),
        compiler=_compiler(),
        additional_zero_sample_count=2,
        additional_single_rare_sample_count=2,
        seed=702,
        maximum_workers=1,
        provenance={"test": True},
    )
    assert supplement["source_validation_fingerprint"] == payload[
        "validation_fingerprint"
    ]
    assert supplement["combined_holdout"]["8"]["strata"]["0,0,0,0,0,0,0,0"][
        "sample_count"
    ] == 4
    validate_connected_cluster_supplement_payload(supplement)


def test_split_calibration_prediction_and_transfer_holdout_are_independent(
    tmp_path,
) -> None:
    calibration = calibrate_connected_cluster_cost_model(
        _hamiltonian(),
        ld=0,
        reference_delta_time=0.08,
        reference_rte_steps=4,
        compiler=_compiler(),
        pilot_sample_count=2,
        minimum_production_sample_count=2,
        maximum_production_sample_count=4,
        prediction_relative_standard_error_target=0.05,
        allocation_safety_factor=1.0,
        seed=801,
        target_event_counts=(4, 6),
        maximum_workers=2,
        checkpoint_directory=tmp_path / "calibration-checkpoints",
        persistent_cache_path=tmp_path / "compiled-cost.sqlite",
        cost_aware_allocation=True,
        chunk_patterns=True,
        provenance={"test": True},
    )
    validate_connected_cluster_calibration_payload(calibration)
    assert set(calibration["production"]) == {"1", "2", "3"}
    assert calibration["configuration"]["target_event_counts"] == [4, 6]
    assert calibration["allocation"]["allocation_rule"] == (
        "neyman_deterministic_relative_work_cost"
    )
    assert len(list((tmp_path / "calibration-checkpoints").glob("*.json"))) == 28
    assert calibration["condition_fingerprint"] == _fingerprint(
        calibration["condition"]
    )
    assert calibration["performance"]["pilot_stage_elapsed_seconds"] > 0.0
    assert calibration["performance"]["production_stage_elapsed_seconds"] > 0.0
    for role in ("pilot", "production"):
        for stage in calibration[role].values():
            assert stage["role"] == role
            assert stage["performance"]["maximum_chunk_seconds"] > 0.0
            assert stage["performance"]["aggregate_worker_seconds"] >= stage[
                "performance"
            ]["maximum_chunk_seconds"]
    checkpoint_envelope = next(
        (tmp_path / "calibration-checkpoints").glob("*.json")
    ).read_text(encoding="utf-8")
    assert CONNECTED_CLUSTER_TASK_IMPLEMENTATION_VERSION in checkpoint_envelope
    calibration_path = tmp_path / "calibration.json"
    write_connected_cluster_calibration(calibration, calibration_path)
    assert load_connected_cluster_calibration(calibration_path) == calibration

    extended = calibrate_connected_cluster_cost_model(
        _hamiltonian(),
        ld=0,
        reference_delta_time=0.08,
        reference_rte_steps=4,
        compiler=_compiler(),
        pilot_sample_count=3,
        minimum_production_sample_count=2,
        maximum_production_sample_count=5,
        prediction_relative_standard_error_target=0.05,
        allocation_safety_factor=1.0,
        seed=801,
        target_event_counts=(4, 6),
        maximum_workers=1,
        checkpoint_directory=tmp_path / "calibration-checkpoints",
        persistent_cache_path=tmp_path / "compiled-cost.sqlite",
        cost_aware_allocation=True,
        chunk_patterns=True,
        provenance={"test": True, "extended": True},
    )
    persistent_hits = sum(
        stage["performance"].get("persistent_cache_hits", 0)
        for role in ("pilot", "production")
        for stage in extended[role].values()
    )
    assert persistent_hits > 0
    assert len(list((tmp_path / "calibration-checkpoints").glob("*.json"))) > 28

    iid = predict_connected_cluster_cost(calibration, event_count=6)
    conditioned = predict_connected_cluster_cost(
        calibration,
        event_count=4,
        order_pattern=(0, 2, 0, 0),
    )
    assert iid["requires_qiskit_transpile"] is False
    assert iid["metrics"]["rz_count"]["mean"] >= 0.0
    assert conditioned["prediction_kind"] == "conditioned_order_pattern"
    assert conditioned["prediction_form"]["k1:2"] == 1.0

    transfer = validate_connected_cluster_calibration_holdout(
        calibration,
        _hamiltonian(),
        compiler=_compiler(),
        holdout_lengths=(4, 6),
        holdout_zero_sample_count=2,
        holdout_single_rare_sample_count=2,
        seed=802,
        maximum_workers=2,
        checkpoint_directory=tmp_path / "holdout-checkpoints",
        persistent_cache_path=tmp_path / "compiled-cost.sqlite",
        chunk_patterns=True,
        provenance={"test": True},
    )
    validate_connected_cluster_transfer_payload(transfer)
    transfer_path = tmp_path / "transfer.json"
    write_connected_cluster_transfer_validation(transfer, transfer_path)
    assert transfer_path.exists()
    assert set(transfer["holdout"]) == {"4", "6"}
    assert set(transfer["holdout_comparisons"]["4"]) == {
        "zero_order2_condition",
        "exactly_one_order2_condition",
    }
    assert transfer["calibration_fingerprint"] == calibration[
        "calibration_fingerprint"
    ]
    assert transfer["calibration_reference"]["condition"] == calibration[
        "condition"
    ]
    assert transfer["performance"]["holdout_stage_elapsed_seconds"] > 0.0

    k4_diagnostic = diagnose_connected_cluster_k4_extrapolation(
        transfer,
        fit_length=4,
        test_length=6,
        provenance={"test": True},
    )
    validate_connected_cluster_k4_extrapolation_payload(k4_diagnostic)
    k4_path = tmp_path / "k4-diagnostic.json"
    write_connected_cluster_k4_extrapolation_diagnostic(k4_diagnostic, k4_path)
    zero_metric = "rz_count"
    zero_fit = transfer["holdout_comparisons"]["4"]["zero_order2_condition"][
        "metrics"
    ][zero_metric]
    zero_test = transfer["holdout_comparisons"]["6"]["zero_order2_condition"][
        "metrics"
    ][zero_metric]
    expected = zero_test["prediction"] + 3.0 * (
        zero_fit["actual"] - zero_fit["prediction"]
    )
    observed = k4_diagnostic["test_comparisons"]["zero_order2_condition"][
        "metrics"
    ][zero_metric]["adjusted_prediction"]
    assert observed == pytest.approx(expected)

    calibrated_k4 = calibrate_and_validate_connected_cluster_k4(
        transfer,
        _hamiltonian(),
        compiler=_compiler(),
        sample_count_per_pattern=2,
        seed=803,
        maximum_workers=2,
        checkpoint_directory=tmp_path / "k4-checkpoints",
        persistent_cache_path=tmp_path / "compiled-cost.sqlite",
        sample_chunk_size=2,
        provenance={"test": True},
    )
    validate_connected_cluster_k4_calibration_payload(calibrated_k4)
    assert set(calibrated_k4["k4_calibration"]["strata"]) == {
        "0,0,0,0",
        "2,0,0,0",
        "0,2,0,0",
        "0,0,2,0",
        "0,0,0,2",
    }
    assert calibrated_k4["holdout_comparisons"]["6"][
        "zero_order2_condition"
    ]["k4_form"] == {"0,0,0,0": 3.0}
    k4_calibration_path = tmp_path / "k4-calibration.json"
    write_connected_cluster_k4_calibration(
        calibrated_k4, k4_calibration_path
    )
    assert k4_calibration_path.exists()

    tampered_k4 = deepcopy(k4_diagnostic)
    tampered_k4["test_length"] = 8
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_connected_cluster_k4_extrapolation_payload(tampered_k4)

    semantic_tamper = deepcopy(calibration)
    semantic_tamper["condition"]["ld"] = 1
    semantic_tamper.pop("calibration_fingerprint")
    semantic_tamper["calibration_fingerprint"] = _fingerprint(semantic_tamper)
    with pytest.raises(ValueError, match="condition mismatch"):
        validate_connected_cluster_calibration_payload(semantic_tamper)

    transfer_tamper = deepcopy(transfer)
    transfer_tamper["holdout"]["4"]["preparation_hash"] = "wrong"
    transfer_tamper.pop("validation_fingerprint")
    transfer_tamper["validation_fingerprint"] = _fingerprint(transfer_tamper)
    with pytest.raises(ValueError, match="preparation hash mismatch"):
        validate_connected_cluster_transfer_payload(transfer_tamper)


def test_chunking_preserves_connected_cluster_scientific_results(tmp_path) -> None:
    common = {
        "hamiltonian": _hamiltonian(),
        "ld": 0,
        "reference_delta_time": 0.08,
        "reference_rte_steps": 4,
        "compiler": _compiler(),
        "pilot_sample_count": 2,
        "minimum_production_sample_count": 2,
        "maximum_production_sample_count": 4,
        "prediction_relative_standard_error_target": 0.05,
        "allocation_safety_factor": 1.0,
        "seed": 811,
        "target_event_counts": (4, 6),
        "maximum_workers": 1,
        "persistent_cache_path": tmp_path / "chunk-equivalence.sqlite",
        "cost_aware_allocation": False,
        "provenance": {"test": True},
    }
    unchunked = calibrate_connected_cluster_cost_model(
        **common,
        checkpoint_directory=tmp_path / "unchunked",
        chunk_patterns=False,
    )
    chunked = calibrate_connected_cluster_cost_model(
        **common,
        checkpoint_directory=tmp_path / "chunked",
        chunk_patterns=True,
    )

    def scientific_fields(value):
        if isinstance(value, dict):
            return {
                key: scientific_fields(item)
                for key, item in value.items()
                if key != "performance" and not key.endswith("_seconds")
            }
        if isinstance(value, list):
            return [scientific_fields(item) for item in value]
        return value

    assert unchunked["allocation"] == chunked["allocation"]
    for role in ("pilot", "production"):
        assert scientific_fields(unchunked[role]) == scientific_fields(
            chunked[role]
        )

    holdout_common = {
        "calibration": unchunked,
        "hamiltonian": _hamiltonian(),
        "compiler": _compiler(),
        "holdout_lengths": (4,),
        "holdout_zero_sample_count": 2,
        "holdout_single_rare_sample_count": 2,
        "seed": 812,
        "maximum_workers": 1,
        "persistent_cache_path": tmp_path / "chunk-equivalence.sqlite",
        "provenance": {"test": True},
    }
    unchunked_holdout = validate_connected_cluster_calibration_holdout(
        **holdout_common,
        checkpoint_directory=tmp_path / "unchunked-holdout",
        chunk_patterns=False,
    )
    chunked_holdout = validate_connected_cluster_calibration_holdout(
        **holdout_common,
        checkpoint_directory=tmp_path / "chunked-holdout",
        chunk_patterns=True,
    )
    assert scientific_fields(unchunked_holdout["holdout"]) == scientific_fields(
        chunked_holdout["holdout"]
    )
    assert unchunked_holdout["holdout_comparisons"] == chunked_holdout[
        "holdout_comparisons"
    ]


def test_sample_chunk_resume_extension_and_worker_invariance(tmp_path) -> None:
    calibration = calibrate_connected_cluster_cost_model(
        _hamiltonian(),
        ld=0,
        reference_delta_time=0.08,
        reference_rte_steps=4,
        compiler=_compiler(),
        pilot_sample_count=2,
        minimum_production_sample_count=2,
        maximum_production_sample_count=4,
        prediction_relative_standard_error_target=0.05,
        allocation_safety_factor=1.0,
        seed=921,
        target_event_counts=(4,),
        maximum_workers=2,
        checkpoint_directory=tmp_path / "calibration-chunks",
        persistent_cache_path=tmp_path / "compiled-cost.sqlite",
        cost_aware_allocation=False,
        sample_chunk_size=2,
        adaptive_production_rounds=1,
        provenance={"test": True},
    )
    chunk_files = list((tmp_path / "calibration-chunks").glob("*_c*.json"))
    assert chunk_files
    assert calibration["allocation"]["adaptive_history"]
    assert calibration["configuration"]["adaptive_production_rounds"] == 1
    for role in ("pilot", "production"):
        for stage in calibration[role].values():
            for stratum in stage["strata"].values():
                if stratum["estimate_kind"] == "exact_conditional_order0_enumeration":
                    continue
                assert stratum["sampling_stream_scheme"].endswith("_v3")
                assert stratum["sample_ranges"][0][0] == 0

    common = {
        "calibration": calibration,
        "hamiltonian": _hamiltonian(),
        "compiler": _compiler(),
        "holdout_lengths": (4,),
        "seed": 922,
        "persistent_cache_path": tmp_path / "compiled-cost.sqlite",
        "provenance": {"test": True},
    }
    legacy = validate_connected_cluster_calibration_holdout(
        **common,
        holdout_zero_sample_count=5,
        holdout_single_rare_sample_count=5,
        maximum_workers=2,
        checkpoint_directory=tmp_path / "legacy-holdout",
        chunk_patterns=True,
        sample_chunk_size=None,
    )
    legacy_file_count = len(list((tmp_path / "legacy-holdout").glob("*.json")))
    restored_legacy = validate_connected_cluster_calibration_holdout(
        **common,
        holdout_zero_sample_count=5,
        holdout_single_rare_sample_count=5,
        maximum_workers=2,
        checkpoint_directory=tmp_path / "legacy-holdout",
        sample_chunk_size=2,
    )
    assert legacy["holdout"] == restored_legacy["holdout"]
    assert len(list((tmp_path / "legacy-holdout").glob("*.json"))) == (
        legacy_file_count
    )
    assert not list((tmp_path / "legacy-holdout").glob("*_c*.json"))

    first = validate_connected_cluster_calibration_holdout(
        **common,
        holdout_zero_sample_count=5,
        holdout_single_rare_sample_count=5,
        maximum_workers=2,
        checkpoint_directory=tmp_path / "holdout-chunks",
        sample_chunk_size=2,
    )
    zero = first["holdout"]["4"]["strata"]["0,0,0,0"]
    assert zero["sample_count"] == 5
    assert zero["sample_ranges"] == [[0, 2], [2, 4], [4, 5]]
    original_chunk = tmp_path / "holdout-chunks" / "holdout_L4_p0-0-0-0_c000000.json"
    original_bytes = original_chunk.read_bytes()

    extended = validate_connected_cluster_calibration_holdout(
        **common,
        holdout_zero_sample_count=7,
        holdout_single_rare_sample_count=7,
        maximum_workers=2,
        checkpoint_directory=tmp_path / "holdout-chunks",
        sample_chunk_size=2,
    )
    extended_zero = extended["holdout"]["4"]["strata"]["0,0,0,0"]
    assert extended_zero["sample_count"] == 7
    assert extended_zero["sample_ranges"] == [[0, 2], [2, 4], [4, 6], [6, 7]]
    assert original_chunk.read_bytes() == original_bytes

    independent = validate_connected_cluster_calibration_holdout(
        **common,
        holdout_zero_sample_count=7,
        holdout_single_rare_sample_count=7,
        maximum_workers=1,
        checkpoint_directory=tmp_path / "holdout-independent",
        sample_chunk_size=2,
    )

    def scientific_fields(value):
        if isinstance(value, dict):
            return {
                key: scientific_fields(item)
                for key, item in value.items()
                if key != "performance" and not key.endswith("_seconds")
            }
        if isinstance(value, list):
            return [scientific_fields(item) for item in value]
        return value

    assert scientific_fields(extended["holdout"]) == scientific_fields(
        independent["holdout"]
    )
    assert extended["holdout_comparisons"] == independent["holdout_comparisons"]

    tampered = deepcopy(extended)
    tampered["holdout"]["4"]["strata"]["0,0,0,0"]["chunk_seeds"][0] += 1
    tampered.pop("validation_fingerprint")
    tampered["validation_fingerprint"] = _fingerprint(tampered)
    with pytest.raises(ValueError, match="chunk seeds mismatch"):
        validate_connected_cluster_transfer_payload(tampered)


def test_angle_invariance_validator_keeps_structural_reuse_disabled(tmp_path) -> None:
    payload = validate_rte_cost_angle_invariance(
        _hamiltonian(),
        ld=0,
        short_step_times=(0.01, 0.02),
        compiler=_compiler(),
        sample_count_per_pattern=2,
        seed=901,
        cluster_lengths=(1, 2),
        persistent_cache_path=tmp_path / "angle-cache.sqlite",
        provenance={"test": True},
    )
    validate_rte_cost_angle_invariance_payload(payload)
    assert payload["coverage"]["metric_comparison_count"] == 12
    assert payload["summary"]["sampled_metric_invariance_passed"] is True
    assert payload["summary"]["structural_cache_reuse_enabled"] is False
