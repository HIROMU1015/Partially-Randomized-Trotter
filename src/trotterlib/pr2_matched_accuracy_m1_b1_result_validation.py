"""Read-only validation and post-hoc analysis of the completed PR-2 M1-B1 map.

This module never loads either molecular NPZ.  It validates the frozen M1-A,
M1-B1 plan, authorization, result, checkpoints, and candidate-scoped SQLite
caches, then summarizes the already-computed compiled resource map.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import statistics
import subprocess
from typing import Any, Iterable, Mapping, Sequence


METRICS = (
    "rz_count",
    "rz_depth",
    "cx_count",
    "cx_depth",
    "total_depth",
    "circuit_size",
)
COMPLETE_STATUS = "M1_B1_COMPILE_MAP_COMPLETE_AWAITING_REVIEW"
REVIEW_DECISION = "CONTINUE_RESOURCE_STUDY"
SOURCE_COMMIT = "33f436bb3a7d5b9cefa23604bb22c8d1fb17cd62"
M1_A_SHA256 = "1f960d7a33296e2dcb74d497e360572b26409dc9aeae01522335e7b91ed81086"
M1_A_FINGERPRINT = "422f898bba1e3849d0f45830082b76d4f42da436e2b49796e562cd79fc716c9e"
PLAN_SHA256 = "5afc94fac0571b38c74b0b00cfcf68e34e491a5fc65ef579d3ec884d079e5aa5"
PLAN_FINGERPRINT = "17c91d41e77d7c085629b60ea470abc87e9590f448ba9bcdfa4342410cd89607"
AUTHORIZATION_SHA256 = "7d188f782354609fec1ea83e289e9872c6a801c3e0e9493f93753fe7cae9d93a"
RESULT_SCHEMA_SHA256 = "035d9b1e48d8d02718f8c7218ea49d28297b3b3f29c39c4232cdef83c7c1d081"
RESULT_SHA256 = "71278113c32b26af0dbf6144a626237a0087478212f8a93fc908de3d4d52aee4"
RESULT_FINGERPRINT = "504d9c9089726800a291a8259e87b2d37c1fdea046263db6c9582bb659c77975"
MARKER_SHA256 = "c076d80bd584e646bce7877df3b13dac9aec5e93a52f7ed9b1464fa6a1e0e52c"

M1_A_RELATIVE = Path(
    "artifacts/pr2_matched_accuracy_m1_execution/2026-09-30/"
    "pr2_matched_accuracy_m1_a_result_v1.json"
)
CONTRACT_ROOT_RELATIVE = Path("artifacts/pr2_matched_accuracy_m1_b1_contract/2026-09-30")
PLAN_RELATIVE = CONTRACT_ROOT_RELATIVE / "pr2_matched_accuracy_m1_b1_execution_plan_v2.json"
AUTHORIZATION_RELATIVE = (
    CONTRACT_ROOT_RELATIVE / "pr2_matched_accuracy_m1_b1_execution_authorization_v1.json"
)
RESULT_SCHEMA_RELATIVE = (
    CONTRACT_ROOT_RELATIVE / "pr2_matched_accuracy_m1_b1_result_schema_v2.json"
)
EXECUTION_ROOT_RELATIVE = Path("artifacts/pr2_matched_accuracy_m1_b1_execution/2026-09-30")
RESULT_RELATIVE = EXECUTION_ROOT_RELATIVE / "pr2_matched_accuracy_m1_b1_compile_map_result_v2.json"
MARKER_RELATIVE = EXECUTION_ROOT_RELATIVE / "M1_B1_COMPLETE.json"
S2_RELATIVE = Path(
    "artifacts/pr2_v4_s2_development/2026-09-29/"
    "pr2_s2_development_resource_result_parallel_v1.json"
)


def canonical_json(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def fingerprint(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _close(actual: float, expected: float, *, name: str) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-8):
        raise ValueError(f"{name} differs: {actual!r} != {expected!r}")


def _git_blob_sha256(root: Path, commit: str, relative: str) -> str:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
    )
    return hashlib.sha256(completed.stdout).hexdigest()


def _rankdata(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        rank = (start + stop - 1) / 2.0 + 1.0
        for index in order[start:stop]:
            ranks[index] = rank
        start = stop
    return ranks


def _correlation(left: Sequence[float], right: Sequence[float]) -> float:
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = math.fsum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=True)
    )
    denominator = math.sqrt(
        math.fsum((x - left_mean) ** 2 for x in left)
        * math.fsum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    return _correlation(_rankdata(left), _rankdata(right))


def _metric_work(record: Mapping[str, Any], metric: str) -> float:
    work = record["matched_accuracy_compiled_work_no_state_preparation"]
    if not isinstance(work, Mapping):
        raise ValueError("ineligible record has no compiled work")
    return float(work[metric])


def _candidate_summary(record: Mapping[str, Any]) -> dict[str, Any]:
    candidate = record["candidate"]
    return {
        "candidate_id": candidate["candidate_id"],
        "candidate_fingerprint": candidate["candidate_fingerprint"],
        "method": candidate["method"],
        "rank": candidate["rank"],
        "q": candidate["q"],
        "delta": candidate["delta"],
        "r": candidate["r"],
        "K": candidate["K"],
        "total_shots": sum(int(value) for value in record["axis_shots"].values()),
        "work_by_metric": {
            metric: _metric_work(record, metric) for metric in METRICS
        },
    }


def _trajectory_values(record: Mapping[str, Any], metric: str) -> list[float]:
    cosine = record["compiled_axes"]["cosine"]["retained_trajectory_records"]
    sine = record["compiled_axes"]["sine"]["retained_trajectory_records"]
    real_shots = int(record["axis_shots"]["real"])
    imag_shots = int(record["axis_shots"]["imag"])
    _require(len(cosine) == len(sine), "axis trajectory counts differ")
    if len(cosine) > 1:
        _require(
            [item["trajectory_seed"] for item in cosine]
            == [item["trajectory_seed"] for item in sine],
            "cosine and sine do not share trajectories",
        )
    return [
        real_shots * float(cosine_item["cost"][metric])
        + imag_shots * float(sine_item["cost"][metric])
        for cosine_item, sine_item in zip(cosine, sine, strict=True)
    ]


def _paired_statistics(record: Mapping[str, Any], metric: str) -> dict[str, float | int]:
    values = _trajectory_values(record, metric)
    mean = statistics.fmean(values)
    standard_error = (
        statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    )
    _close(mean, _metric_work(record, metric), name="paired work mean")
    return {
        "samples": len(values),
        "mean": mean,
        "standard_error": standard_error,
        "relative_standard_error": standard_error / mean if mean else 0.0,
    }


def _validate_metric_statistics(axis: Mapping[str, Any]) -> None:
    records = axis["retained_trajectory_records"]
    for metric in METRICS:
        values = [float(item["cost"][metric]) for item in records]
        statistics_record = axis["metric_statistics"][metric]
        mean = statistics.fmean(values)
        variance = statistics.variance(values) if len(values) > 1 else 0.0
        standard_error = math.sqrt(variance / len(values)) if values else 0.0
        _close(statistics_record["mean"], mean, name=f"{metric} mean")
        _close(statistics_record["minimum"], min(values), name=f"{metric} minimum")
        _close(statistics_record["maximum"], max(values), name=f"{metric} maximum")
        if len(values) == 1:
            _require(
                statistics_record["unbiased_sample_variance"] is None
                and statistics_record["standard_error"] is None,
                f"exact {metric} uncertainty must be null",
            )
        else:
            _close(
                statistics_record["unbiased_sample_variance"],
                variance,
                name=f"{metric} sample variance",
            )
            _close(
                statistics_record["standard_error"],
                standard_error,
                name=f"{metric} standard error",
            )


def _cache_key(payload: Mapping[str, Any]) -> str:
    return fingerprint(
        {
            "actual_circuit_fingerprint": payload["actual_circuit_fingerprint"],
            "compiler_settings_hash": payload["compiler_settings_hash"],
            "backend_fingerprint": payload["backend_fingerprint"],
            "cacheable": True,
            "bypass_object_identity": None,
            "cache_key_policy": "actual_circuit_compiler_backend_v2",
        }
    )


def _validate_cache(
    cache_path: Path,
    compile_record: Mapping[str, Any],
) -> tuple[int, int]:
    uri = f"file:{cache_path.resolve()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        _require(
            connection.execute("PRAGMA integrity_check").fetchone() == ("ok",),
            f"SQLite integrity failed: {cache_path}",
        )
        rows = connection.execute(
            "SELECT cache_key, schema_version, payload_json "
            "FROM compiled_cost_metric_cache"
        ).fetchall()
    finally:
        connection.close()
    payload_by_circuit: dict[str, dict[str, Any]] = {}
    for key, schema_version, encoded in rows:
        _require(
            schema_version == "compiled_cost_metric_cache_v1",
            f"cache schema mismatch: {cache_path}",
        )
        payload = json.loads(encoded)
        _require(key == _cache_key(payload), f"cache key mismatch: {cache_path}")
        actual = str(payload["actual_circuit_fingerprint"])
        _require(actual not in payload_by_circuit, f"duplicate circuit cache row: {cache_path}")
        payload_by_circuit[actual] = payload

    wrapper_records = [
        item
        for axis in ("cosine", "sine")
        for item in compile_record["compiled_axes"][axis]["retained_trajectory_records"]
    ]
    unique_actual = {str(item["actual_circuit_fingerprint"]) for item in wrapper_records}
    _require(
        unique_actual == set(payload_by_circuit),
        f"cache/result circuit set mismatch: {cache_path}",
    )
    for item in wrapper_records:
        payload = payload_by_circuit[str(item["actual_circuit_fingerprint"])]
        for metric in METRICS:
            _close(item["cost"][metric], payload[metric], name=f"cache {metric}")
    return len(wrapper_records), len(rows)


def _proxy_frontier(signal_records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    eligible = [
        item
        for item in signal_records
        if item["accuracy_eligible"] and item["candidate"]["method"] in {"B2", "B3"}
    ]
    metrics = ("total_shots", "n_det", "n_rand")
    return [
        item
        for item in eligible
        if not any(
            all(int(other[name]) <= int(item[name]) for name in metrics)
            and any(int(other[name]) < int(item[name]) for name in metrics)
            for other in eligible
            if other is not item
        )
    ]


def _pareto_frontier(records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        item
        for item in records
        if not any(
            all(_metric_work(other, metric) <= _metric_work(item, metric) for metric in METRICS)
            and any(_metric_work(other, metric) < _metric_work(item, metric) for metric in METRICS)
            for other in records
            if other is not item
        )
    ]


def _lower_envelope(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    lines = [
        (
            _metric_work(item, "rz_count"),
            sum(int(value) for value in item["axis_shots"].values()),
            item,
        )
        for item in records
    ]
    intersections = {0.0}
    for index, (left_intercept, left_slope, _left) in enumerate(lines):
        for right_intercept, right_slope, _right in lines[index + 1 :]:
            if left_slope == right_slope:
                continue
            point = (right_intercept - left_intercept) / (left_slope - right_slope)
            if point >= 0.0 and math.isfinite(point):
                intersections.add(point)
    points = sorted(intersections)
    interval_winners: list[tuple[float, tuple[float, int, Mapping[str, Any]]]] = []
    for index, start in enumerate(points):
        if index + 1 < len(points):
            probe = (start + points[index + 1]) / 2.0
        else:
            probe = start + max(1.0, abs(start))
        winner = min(lines, key=lambda line: line[0] + line[1] * probe)
        if not interval_winners or winner[2]["candidate"]["candidate_fingerprint"] != (
            interval_winners[-1][1][2]["candidate"]["candidate_fingerprint"]
        ):
            interval_winners.append((start, winner))
    output: list[dict[str, Any]] = []
    previous: tuple[float, int, Mapping[str, Any]] | None = None
    for approximate_start, current in interval_winners:
        if previous is None:
            start = 0.0
        else:
            start = (current[0] - previous[0]) / (previous[1] - current[1])
            _require(start >= 0.0, "negative lower-envelope breakpoint")
        output.append(
            {
                "start_P_inclusive": start,
                "candidate": _candidate_summary(current[2]),
                "rz_intercept": current[0],
                "shot_slope": current[1],
                "scan_start_diagnostic": approximate_start,
            }
        )
        previous = current
    return output


def _best_by_method_rank(
    records: Sequence[Mapping[str, Any]], *, q: int | None = None
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for item in records:
        if q is None or int(item["candidate"]["q"]) == q:
            groups[(str(item["candidate"]["method"]), int(item["candidate"]["rank"]))].append(item)
    output = []
    for (method, rank), group in sorted(groups.items()):
        winner = min(group, key=lambda item: _metric_work(item, "rz_count"))
        output.append({"method": method, "rank": rank, **_candidate_summary(winner)})
    return output


def _s2_summary(s2: Mapping[str, Any]) -> dict[str, Any]:
    def find(method: str, rank: int, r: int, cutoff: int) -> Mapping[str, Any]:
        for collection in (
            "deterministic_candidates",
            "B2_candidates",
            "B3_candidates",
            "rank3_rank9_controls",
        ):
            for item in s2[collection]:
                if (
                    item["method"],
                    int(item["rank"]),
                    int(item["r"]),
                    int(item["K"]),
                ) == (method, rank, r, cutoff):
                    return item
        raise ValueError(f"missing S2 record: {(method, rank, r, cutoff)}")

    selected_b2 = find("B2", 6, 1, 2)
    selected_b3 = find("B3", 0, 32, 4)
    rank3_control = find("B2", 3, 1, 2)
    return {
        "status": s2["status"],
        "result_fingerprint": s2["result_fingerprint"],
        "fixed_q": 8,
        "primary_material_frontier": s2["decision"]["material_frontier"],
        "selected_B2_rank6_r1_K2_rz_work": selected_b2["resource"][
            "total_work_no_preparation"
        ],
        "selected_B3_rank0_r32_K4_rz_work": selected_b3["resource"][
            "total_work_no_preparation"
        ],
        "rank3_control_r1_K2_rz_work": rank3_control["resource"][
            "total_work_no_preparation"
        ],
    }


def validate_and_analyze(project_root: Path) -> dict[str, Any]:
    """Validate all completed local evidence and return a compact review artifact."""
    root = project_root.resolve()
    m1_a_path = root / M1_A_RELATIVE
    plan_path = root / PLAN_RELATIVE
    authorization_path = root / AUTHORIZATION_RELATIVE
    schema_path = root / RESULT_SCHEMA_RELATIVE
    result_path = root / RESULT_RELATIVE
    marker_path = root / MARKER_RELATIVE
    s2_path = root / S2_RELATIVE

    expected_hashes = {
        str(M1_A_RELATIVE): M1_A_SHA256,
        str(PLAN_RELATIVE): PLAN_SHA256,
        str(AUTHORIZATION_RELATIVE): AUTHORIZATION_SHA256,
        str(RESULT_SCHEMA_RELATIVE): RESULT_SCHEMA_SHA256,
        str(RESULT_RELATIVE): RESULT_SHA256,
        str(MARKER_RELATIVE): MARKER_SHA256,
    }
    for relative, expected in expected_hashes.items():
        _require(file_sha256(root / relative) == expected, f"SHA-256 mismatch: {relative}")

    m1_a = load_json(m1_a_path)
    plan = load_json(plan_path)
    authorization = load_json(authorization_path)
    result = load_json(result_path)
    marker = load_json(marker_path)
    s2 = load_json(s2_path)
    _require(m1_a["result_fingerprint"] == M1_A_FINGERPRINT, "M1-A fingerprint mismatch")
    _require(plan["plan_fingerprint"] == PLAN_FINGERPRINT, "plan fingerprint mismatch")
    _require(
        fingerprint({key: value for key, value in plan.items() if key != "plan_fingerprint"})
        == PLAN_FINGERPRINT,
        "recomputed plan fingerprint mismatch",
    )
    _require(result["result_fingerprint"] == RESULT_FINGERPRINT, "result fingerprint mismatch")
    _require(
        fingerprint({key: value for key, value in result.items() if key != "result_fingerprint"})
        == RESULT_FINGERPRINT,
        "recomputed result fingerprint mismatch",
    )
    _require(
        marker == {"status": COMPLETE_STATUS, "result_fingerprint": RESULT_FINGERPRINT},
        "completion marker mismatch",
    )
    _require(result["status"] == COMPLETE_STATUS, "unexpected result status")
    _require(result["source_commit"] == SOURCE_COMMIT, "source commit mismatch")
    _require(result["authorization_sha256"] == AUTHORIZATION_SHA256, "authorization mismatch")
    _require(result["execution_plan_sha256"] == PLAN_SHA256, "plan SHA mismatch")
    _require(result["execution_plan_fingerprint"] == PLAN_FINGERPRINT, "plan identity mismatch")
    _require(result["m1_a_result_sha256"] == M1_A_SHA256, "M1-A SHA mismatch")
    _require(result["m1_a_result_fingerprint"] == M1_A_FINGERPRINT, "M1-A identity mismatch")
    _require(authorization["source_commit"] == SOURCE_COMMIT, "authorization source mismatch")
    _require(plan["source_commit"] == SOURCE_COMMIT, "plan source mismatch")
    _require(authorization["execution_plan_sha256"] == PLAN_SHA256, "authorization plan mismatch")
    _require(
        authorization["execution_plan_fingerprint"] == PLAN_FINGERPRINT,
        "authorization plan fingerprint mismatch",
    )
    _require(
        authorization["result_schema"]["sha256"] == RESULT_SCHEMA_SHA256,
        "authorization schema mismatch",
    )
    for relative, expected in authorization["source_hashes"].items():
        _require(file_sha256(root / relative) == expected, f"working source differs: {relative}")
        _require(
            _git_blob_sha256(root, SOURCE_COMMIT, relative) == expected,
            f"source commit blob differs: {relative}",
        )

    expected_caps = authorization["resource_caps"]
    counts = result["resource_counts"]
    for name in (
        "random_cells",
        "trajectories_per_random_cell",
        "random_trajectories",
        "random_full_wrappers",
        "baseline_cells",
        "baseline_full_wrappers",
        "total_full_wrappers",
        "extension_trajectories",
    ):
        _require(counts[name] == expected_caps[name], f"resource cap mismatch: {name}")
    _require(counts["process_workers"] == 6, "worker count mismatch")
    _require(counts["blas_threads_per_worker"] == 1, "BLAS thread count mismatch")
    _require(result["signal_reevaluations"] == 0, "signal was re-evaluated")
    _require(result["held_out_accessed"] is False, "held-out was accessed")
    _require(result["additional_96_trajectories_executed"] is False, "extension was executed")
    _require(result["transfer_executed"] is False, "transfer was executed")
    _require(result["s3_authorized"] is False, "S3 was authorized")
    _require(result["research_decision"] is None, "science runner made a research decision")
    _require(result["automatic_next_stage"] is None, "science runner selected a next stage")
    _require(result["execution"]["gpu_queries"] == 0, "GPU query count is nonzero")
    _require(result["execution"]["gpu_allocations"] == 0, "GPU allocation count is nonzero")
    _require(result["execution"]["gpu_kernels"] == 0, "GPU kernel count is nonzero")

    plan_cells = [*plan["random_cells"], *plan["baseline_cells"]]
    compile_map = result["compile_map"]
    _require(len(plan_cells) == len(compile_map) == 210, "cell count mismatch")
    checkpoint_root = root / EXECUTION_ROOT_RELATIVE / ".runtime" / "checkpoints"
    cache_root = root / EXECUTION_ROOT_RELATIVE / ".runtime" / "cache" / SOURCE_COMMIT
    run_identity = load_json(root / EXECUTION_ROOT_RELATIVE / ".runtime" / "run_identity.json")
    _require(
        run_identity
        == {
            "schema_version": "pr2_matched_accuracy_m1_b1_run_identity_v1",
            "source_commit": SOURCE_COMMIT,
            "authorization_sha256": AUTHORIZATION_SHA256,
            "execution_plan_sha256": PLAN_SHA256,
            "execution_plan_fingerprint": PLAN_FINGERPRINT,
            "workers": 6,
        },
        "run identity mismatch",
    )
    checkpoint_files = sorted(checkpoint_root.glob("*.json"))
    cache_files = sorted(cache_root.glob("*.sqlite3"))
    _require(len(checkpoint_files) == len(cache_files) == 210, "runtime file count mismatch")
    _require(
        {path.stem for path in cache_files}
        == {str(cell["candidate_fingerprint"]) for cell in plan_cells},
        "candidate-scoped cache file set mismatch",
    )

    signal_by_fingerprint = {
        str(item["candidate"]["candidate_fingerprint"]): item
        for item in m1_a["signal_records"]
    }
    random_seed_set: set[int] = set()
    wrapper_records = 0
    unique_transpiles = 0
    cache_row_distribution: Counter[int] = Counter()
    method_counts: Counter[str] = Counter()
    eligibility_counts: Counter[str] = Counter()
    for ordinal, (cell, compile_record) in enumerate(zip(plan_cells, compile_map, strict=True)):
        candidate = compile_record["candidate"]
        candidate_fingerprint = str(cell["candidate_fingerprint"])
        _require(
            candidate["candidate_fingerprint"] == candidate_fingerprint,
            f"candidate order mismatch at {ordinal}",
        )
        _require(candidate["candidate_id"] == cell["candidate_id"], f"candidate id mismatch at {ordinal}")
        method_counts[str(candidate["method"])] += 1
        eligibility_counts[f"{candidate['method']}:{bool(compile_record['accuracy_eligible'])}"] += 1
        signal = signal_by_fingerprint[candidate_fingerprint]
        _require(
            compile_record["signal_record_fingerprint"] == fingerprint(signal),
            f"signal record fingerprint mismatch at {ordinal}",
        )
        checkpoint_path = checkpoint_root / f"{ordinal:03d}_{candidate_fingerprint}.json"
        checkpoint = load_json(checkpoint_path)
        checkpoint_body = {
            key: value for key, value in checkpoint.items() if key != "checkpoint_fingerprint"
        }
        _require(
            checkpoint["checkpoint_fingerprint"] == fingerprint(checkpoint_body),
            f"checkpoint fingerprint mismatch at {ordinal}",
        )
        _require(checkpoint["task_fingerprint"] == cell["task_fingerprint"], f"task mismatch at {ordinal}")
        _require(checkpoint["candidate_fingerprint"] == candidate_fingerprint, f"checkpoint cell mismatch at {ordinal}")
        _require(
            checkpoint["result"]["axes"] == compile_record["compiled_axes"],
            f"checkpoint/result axes mismatch at {ordinal}",
        )
        random_cell = candidate["method"] in {"B2", "B3"}
        expected_count = 32 if random_cell else 1
        for axis in ("cosine", "sine"):
            axis_record = compile_record["compiled_axes"][axis]
            retained = axis_record["retained_trajectory_records"]
            _require(axis_record["status"] == "complete", f"axis incomplete at {ordinal}")
            _require(axis_record["sample_count"] == (32 if random_cell else None), f"sample count mismatch at {ordinal}")
            _require(len(retained) == expected_count, f"retained count mismatch at {ordinal}")
            _require(axis_record["quantum_shots_executed"] == 0, f"quantum shots at {ordinal}")
            _require(axis_record["state_preparation_included"] is False, f"state preparation at {ordinal}")
            _require(axis_record["measurement_included"] is True, f"measurement missing at {ordinal}")
            _require(axis_record["transpile_completed"] is True, f"transpile incomplete at {ordinal}")
            _validate_metric_statistics(axis_record)
        if random_cell:
            expected_seeds = list(cell["sampled_trajectory_seeds"])
            _require(len(expected_seeds) == len(set(expected_seeds)) == 32, f"seed repetition at {ordinal}")
            _require(random_seed_set.isdisjoint(expected_seeds), f"cross-cell seed collision at {ordinal}")
            random_seed_set.update(expected_seeds)
            for axis in ("cosine", "sine"):
                axis_record = compile_record["compiled_axes"][axis]
                _require(axis_record["sampled_trajectory_seeds"] == expected_seeds, f"axis seed mismatch at {ordinal}")
                _require(
                    [item["trajectory_seed"] for item in axis_record["retained_trajectory_records"]]
                    == expected_seeds,
                    f"retained seed mismatch at {ordinal}",
                )
        if compile_record["accuracy_eligible"]:
            for metric in METRICS:
                _paired_statistics(compile_record, metric)
        wrappers, cache_rows = _validate_cache(cache_root / f"{candidate_fingerprint}.sqlite3", compile_record)
        wrapper_records += wrappers
        unique_transpiles += cache_rows
        cache_row_distribution[cache_rows] += 1

    _require(len(random_seed_set) == 6208, "random trajectory seed count mismatch")
    _require(wrapper_records == 12448, "wrapper record count mismatch")
    _require(unique_transpiles == 12128, "unique transpile count mismatch")

    eligible = [item for item in compile_map if item["matched_accuracy_frontier_eligible"]]
    _require(len(eligible) == 206, "eligible count mismatch")
    pareto = _pareto_frontier(eligible)
    primary_winner = min(eligible, key=lambda item: _metric_work(item, "rz_count"))
    material_rz = [
        item
        for item in eligible
        if _metric_work(item, "rz_count") <= 1.1 * _metric_work(primary_winner, "rz_count")
    ]
    selected_fingerprints = {
        str(item["candidate_fingerprint"]) for item in m1_a["compile_selection"]["selected"]
    }
    proxy_frontier = _proxy_frontier(m1_a["signal_records"])
    proxy_frontier_fingerprints = {
        str(item["candidate_fingerprint"]) for item in proxy_frontier
    }
    _require(len(proxy_frontier) == 64, "recomputed proxy frontier count mismatch")
    _require(
        proxy_frontier_fingerprints - selected_fingerprints
        == set(m1_a["compile_selection"]["unselected_proxy_frontier_fingerprints"]),
        "proxy frontier audit mismatch",
    )

    actual_vs_selected = {}
    for metric in METRICS:
        actual = min(eligible, key=lambda item: _metric_work(item, metric))
        selected = min(
            (
                item
                for item in eligible
                if item["candidate"]["candidate_fingerprint"] in selected_fingerprints
            ),
            key=lambda item: _metric_work(item, metric),
        )
        actual_vs_selected[metric] = {
            "actual": _candidate_summary(actual),
            "old_selector_best": _candidate_summary(selected),
            "old_selector_regret_fraction": (
                _metric_work(selected, metric) / _metric_work(actual, metric) - 1.0
            ),
        }

    random_records = [
        item for item in eligible if item["candidate"]["method"] in {"B2", "B3"}
    ]
    proxy_correlations = {}
    for metric in METRICS:
        actual_values = [_metric_work(item, metric) for item in random_records]
        action_values = [
            float(signal_by_fingerprint[item["candidate"]["candidate_fingerprint"]]["W_action"])
            for item in random_records
        ]
        tail_values = [
            float(signal_by_fingerprint[item["candidate"]["candidate_fingerprint"]]["W_tail"])
            for item in random_records
        ]
        proxy_correlations[metric] = {
            "spearman_W_action": _spearman(action_values, actual_values),
            "spearman_W_tail": _spearman(tail_values, actual_values),
            "pearson_log_W_action_log_actual": _correlation(
                [math.log(value) for value in action_values],
                [math.log(value) for value in actual_values],
            ),
        }

    best_method_rank = _best_by_method_rank(eligible)
    fixed_q8_best = _best_by_method_rank(eligible, q=8)
    by_method = {
        method: min(
            (item for item in eligible if item["candidate"]["method"] == method),
            key=lambda item: _metric_work(item, "rz_count"),
        )
        for method in ("B0", "B1", "B2", "B3")
    }
    method_comparisons = {
        method: {
            "best": _candidate_summary(item),
            "primary_winner_fraction_lower": (
                1.0
                - _metric_work(primary_winner, "rz_count")
                / _metric_work(item, "rz_count")
            ),
        }
        for method, item in by_method.items()
    }
    uncertainty = []
    for item in sorted(eligible, key=lambda record: _metric_work(record, "rz_count"))[:10]:
        uncertainty.append(
            {
                "candidate": _candidate_summary(item),
                "paired_rz_work": _paired_statistics(item, "rz_count"),
            }
        )
    runner_up = sorted(eligible, key=lambda item: _metric_work(item, "rz_count"))[1]
    winner_stats = _paired_statistics(primary_winner, "rz_count")
    runner_stats = _paired_statistics(runner_up, "rz_count")
    gap_se = math.sqrt(
        float(winner_stats["standard_error"]) ** 2
        + float(runner_stats["standard_error"]) ** 2
    )
    winner_gap = _metric_work(runner_up, "rz_count") - _metric_work(primary_winner, "rz_count")

    payload: dict[str, Any] = {
        "schema_version": "pr2_matched_accuracy_m1_b1_result_validation_v1",
        "status": "M1_B1_RESULT_VALIDATED_RESEARCH_REVIEW_COMPLETE",
        "scope": {
            "system": "H4 linear 1.00 Angstrom",
            "basis": "STO-3G",
            "df_rank": 12,
            "sector_qubits": 8,
            "total_time": 0.8,
            "q_values": [1, 2, 4, 8],
            "delta_values": [0.8, 0.4, 0.2, 0.1],
            "split_ranks": [0, 3, 6, 9, 12],
            "compiler": plan["compiler_identity"],
            "state_preparation_excluded_from_primary": True,
            "local_validation_only": True,
        },
        "input_identity": {
            "source_commit": SOURCE_COMMIT,
            "m1_a_sha256": M1_A_SHA256,
            "m1_a_fingerprint": M1_A_FINGERPRINT,
            "plan_sha256": PLAN_SHA256,
            "plan_fingerprint": PLAN_FINGERPRINT,
            "authorization_sha256": AUTHORIZATION_SHA256,
            "result_schema_sha256": RESULT_SCHEMA_SHA256,
            "result_sha256": RESULT_SHA256,
            "result_fingerprint": RESULT_FINGERPRINT,
            "completion_marker_sha256": MARKER_SHA256,
            "s2_sha256": file_sha256(s2_path),
        },
        "integrity": {
            "all_gates_passed": True,
            "compile_map_cells": len(compile_map),
            "checkpoint_files": len(checkpoint_files),
            "candidate_scoped_sqlite_files": len(cache_files),
            "wrapper_records": wrapper_records,
            "unique_actual_circuit_transpiles": unique_transpiles,
            "within_cell_semantically_identical_cache_reuses": wrapper_records
            - unique_transpiles,
            "cache_row_count_distribution": {
                str(count): cells for count, cells in sorted(cache_row_distribution.items())
            },
            "random_trajectory_seeds": len(random_seed_set),
            "random_seed_collisions": 0,
            "method_counts": dict(sorted(method_counts.items())),
            "eligibility_counts": dict(sorted(eligibility_counts.items())),
            "resource_counts": counts,
            "forbidden_actions": {
                "signal_reevaluations": result["signal_reevaluations"],
                "extension_trajectories": counts["extension_trajectories"],
                "held_out_accessed": result["held_out_accessed"],
                "transfer_executed": result["transfer_executed"],
                "s3_authorized": result["s3_authorized"],
                "gpu_queries": result["execution"]["gpu_queries"],
                "gpu_allocations": result["execution"]["gpu_allocations"],
                "gpu_kernels": result["execution"]["gpu_kernels"],
            },
        },
        "analysis": {
            "accuracy_eligible_cells": len(eligible),
            "actual_six_metric_pareto": [_candidate_summary(item) for item in pareto],
            "primary_rz_point_winner": _candidate_summary(primary_winner),
            "primary_rz_within_10_percent": [
                _candidate_summary(item)
                for item in sorted(material_rz, key=lambda record: _metric_work(record, "rz_count"))
            ],
            "best_by_method_rank": best_method_rank,
            "fixed_q8_best_by_method_rank": fixed_q8_best,
            "method_comparisons": method_comparisons,
            "state_preparation_rz_lower_envelope": _lower_envelope(eligible),
            "proxy_correlations_random_194": proxy_correlations,
            "old_selector_actual_comparison": {
                "selected_cells": len(selected_fingerprints),
                "proxy_frontier_cells": len(proxy_frontier),
                "actual_frontier_cells": len(pareto),
                "actual_frontier_selected_by_old_selector": sum(
                    item["candidate"]["candidate_fingerprint"] in selected_fingerprints
                    for item in pareto
                ),
                "actual_frontier_in_old_proxy_frontier": sum(
                    item["candidate"]["candidate_fingerprint"]
                    in proxy_frontier_fingerprints
                    for item in pareto
                ),
                "by_metric": actual_vs_selected,
            },
            "paired_monte_carlo_uncertainty_top10_rz": uncertainty,
            "rz_point_winner_vs_runner_up": {
                "runner_up": _candidate_summary(runner_up),
                "relative_point_gap": winner_gap / _metric_work(primary_winner, "rz_count"),
                "independent_candidate_gap_standard_error": gap_se,
                "gap_z_diagnostic": winner_gap / gap_se,
                "formal_confidence_interval": False,
            },
            "fixed_q8_s2_reference": _s2_summary(s2),
        },
        "external_research_review": {
            "decision": REVIEW_DECISION,
            "reasons": [
                "Intermediate B2 rank 3 is the complete-grid point winner for RZ count and circuit size and remains on the six-metric Pareto frontier.",
                "The second six-metric Pareto point was outside the old sixteen-cell selector, confirming that the proxy cap was not frontier-complete.",
                "Allowing matched-accuracy q changes the design conclusion relative to the fixed-q=8 S2 framing; q=1 rank-3 B2 is the no-preparation resource region.",
                "The state-preparation RZ lower envelope remains entirely within intermediate B2 candidates over nonnegative P; endpoints do not recover the envelope.",
                "The exact r/K point winner is not resolved by 32 trajectories, but all candidates within ten percent of the RZ minimum are B2 rank 3, q=1, so the method/split conclusion is stable.",
            ],
            "limitations": [
                "This is one local H4 1.00 Angstrom, STO-3G, DF-rank-12, Qiskit-1.3.0 optimization-level-1 development map.",
                "The analysis does not authorize additional trajectories, held-out access, transfer, S3, or final winner refinement.",
                "No final total-cost, backend-noise, state-preparation implementation, H12, or general resource-optimality claim is made.",
            ],
            "next_stop": "DRAFT_SEPARATE_RESULT_PRIOR_HELD_OUT_TRANSFER_REVIEW; DO_NOT_ACCESS_HELD_OUT_YET",
        },
    }
    payload["validation_fingerprint"] = fingerprint(payload)
    return payload


def write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite validation artifact: {path}")
    path.write_bytes(canonical_json(payload) + b"\n")
