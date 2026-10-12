"""Read saved H6 JSON metadata only; never launch science or decode NPZ arrays.

Run in the matching SOURCE worktree with Python -I -P -B. Writes exclusively
to this report directory and refuses to replace any generated artifact.
"""
from collections import Counter, defaultdict
from fractions import Fraction
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import statistics
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RUN = ROOT / "artifacts/resource_applicability/track_a_h6_confirmation_fast_v1/2026-10-12/launch_confirmation_fast_v1"
OLD = ROOT / "artifacts/resource_applicability/track_a_h6_exploration_summary_v1/2026-10-12"
EVIDENCE = Path("/home/AbeHiromu/projects/h6-handoff-evidence/20261012/confirmation-fast-handoff-v1")
SOURCE = "87b6b9f46b7e8ba9abe132ee21478c114b6f46f7"


def read(path):
    path = Path(path)
    if path.is_symlink() or path.stat().st_size > 16 * 2**20:
        raise ValueError("REPORT_INPUT_PATH_OR_SIZE")
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def write(name, value):
    data = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    if len(data) > 16 * 2**20:
        raise ValueError("REPORT_OUTPUT_LIMIT")
    with (HERE / name).open("xb") as stream:
        stream.write(data)


def close(a, b):
    if a is None or b is None:
        assert a is b
    else:
        assert math.isclose(a, b, rel_tol=2e-15, abs_tol=1e-9), (a, b)


def main():
    resource.setrlimit(resource.RLIMIT_AS, (2 * 2**30, 2 * 2**30))
    signal.alarm(180)
    os.nice(10)
    sys.path.insert(0, str(ROOT / "src"))
    from trottertracks.resource_applicability.ax2a_preparation import digest
    from trottertracks.resource_applicability.h6_confirmation_fast_execution_v1 import admit_confirmation_group, complete_confirmation
    m = read(RUN / "frozen_manifest.json")
    terminal = read(RUN / "execution_terminal.json")
    receipt = read(RUN / "confirmation_complete_receipt.json")
    resources = read(RUN / "confirmation_resource_summary.json")
    native_exit = read(EVIDENCE / "coordinator_exit.json")
    assert m["source_commit"] == terminal["source_commit"] == receipt["source_commit"] == SOURCE
    assert terminal["status"] == "H6_CONFIRMATION_FAST_COMPLETE_MANDATORY_STOP" and terminal["reason"] is None
    assert terminal["resource_summary"] == resources and native_exit["exit_code"] == 0
    assert resources["all_dispatched_reaped"] and resources["ownership_cleanup_verified"]
    assert resources["log_and_journal_closed"] and not resources["cleanup_errors"]
    assert resources["first_stop_reason"] is None and resources["pending"] == 0
    assert all(x["reaped"] and x["exit_code"] == 0 for x in resources["retired"])
    assert not terminal["next_stage_authorized"] and terminal["mandatory_stop"]
    for path, expected in m["source_hashes"].items():
        assert sha(ROOT / path) == expected, path
    for path, expected in m["exploration_binding"]["report_hashes"].items():
        assert sha(ROOT / path) == expected, path
    profile = m["base_manifest"]["profile"]
    assert sha(ROOT / profile["repository_path"]) == profile["sha256"]
    original = m["exploration_binding"]["selection"]["registrations"]
    reused = {x["index"]: x for x in m["confirmation_reuse"]["accepted"]}
    groups, provenance, orders, hash_stats = {}, [], defaultdict(Counter), []
    for registration in original:
        i, task = registration["index"], registration["task"]
        parent = reused.get(i)
        directory = Path(parent["directory"]) if parent else RUN / f"cost_confirmation_fast_{i:04d}"
        source = m["confirmation_reuse"]["parent_source"] if parent else SOURCE
        result_path = directory / "cost_result.json"
        if parent:
            assert sha(result_path) == parent["cost_result_sha256"]
        groups[i] = admit_confirmation_group(directory, task, source=source)
        tr = read(directory / "trajectory.json")
        orders[task["cell_id"]].update(str(x) for x in tr["observed_orders"])
        files = sorted(p for p in directory.glob("*.json") if not p.name.startswith(("diagnostic_", "progress_", "compile_attempt_")))
        provenance.append(dict(index=i, origin="reused159" if parent else "new97", source_commit=source,
            task_digest=digest(task), directory=str(directory), file_sha256={p.name: sha(p) for p in files}))
        if not parent:
            hash_stats.append(read(directory / "hash_reuse_statistics.json"))
    rows = complete_confirmation(dict(m, registrations=original), groups)
    assert len(groups) == receipt["group_count"] == 256 and len(rows) == receipt["wrapper_count"] == 512
    assert digest(rows) == receipt["confirmation_rows_digest"]
    assert digest(m["exploration_binding"]["selection"]) == receipt["selection_digest"]
    deterministic, baselines = [], []
    tasks = m["base_manifest"]["reuse_registry"]["original_tasks"]
    for g in read(OLD / "integrated_exploration_v1.json")["groups"]:
        if tasks[g["index"]]["cell"]["method"] == "B2":
            continue
        assert sha(g["cost_result_path"]) == g["cost_result_sha256"]
        deterministic.extend(read(g["cost_result_path"])["rows"])
        baselines.append(g)
    assert len(baselines) == 16
    raw_by_cell = defaultdict(list)
    for r in deterministic + rows:
        if r["control"] == "symmetric_directional":
            raw_by_cell[r["cell_id"]].append(r)
    maps = read(RUN / "confirmation_resource_map.json")
    # Independent integer/Fraction means and paired trajectory SE; no science API.
    for r in maps["rows"]:
        raw = raw_by_cell[r["cell_id"]]
        paired = defaultdict(dict)
        for axis, stats in r["axes"].items():
            values = [x["metrics"][r["metric"]] for x in raw if x["axis"] == axis]
            assert len(values) == stats["n"]
            close(float(Fraction(sum(values), len(values))), stats["mean"])
            assert min(values) == stats["min"] and max(values) == stats["max"]
            sd = statistics.stdev(values) if len(values) > 1 else None
            close(sd, stats["sample_sd"])
            close(sd / math.sqrt(len(values)) if sd is not None else None, stats["standard_error"])
        for x in raw:
            paired[x["replica"]][x["axis"]] = x["metrics"][r["metric"]]
        totals = [sum(r["axes"][a]["shots"] * x[a] for a in ("cosine", "sine")) for x in paired.values()]
        close(float(Fraction(sum(totals), len(totals))), r["G_point"])
        close(statistics.stdev(totals) / math.sqrt(len(totals)) if len(totals) > 1 else None, r["G_standard_error"])
        assert r["N_total"] == sum(x["shots"] for x in r["axes"].values())
    comparisons = []
    old_maps = read(OLD / "exploration_resource_map_v1.json")
    for factor in ("1.0", "10.0"):
        for eps in (0.05, 0.01, 0.005, 0.001):
            def candidates(method, metric="RZ", source_maps=maps):
                return sorted((r for r in source_maps["rows"] if r["epsilon"] == eps and r["metric"] == metric
                    and r["allowance_factor"] == factor and r["method"] == method), key=lambda r: (r["G_point"], r["cell_id"]))
            family = {method: candidates(method) for method in ("B0", "B1", "B2")}
            pr = family["B2"][0]
            second = family["B2"][1]
            gapsd = math.sqrt(pr["G_standard_error"]**2 + second["G_standard_error"]**2)
            crossings = {}
            for method in ("B0", "B1"):
                baseline = family[method][0]
                delta_n = pr["N_total"] - baseline["N_total"]
                value = (baseline["G_point"] - pr["G_point"]) / delta_n if delta_n else None
                crossings[method] = dict(slope_difference=delta_n, crossing_P=value if value is not None and value >= 0 else None)
            comparisons.append(dict(epsilon=eps, allowance_factor=factor,
                family_best_point={k:v[0] for k,v in family.items()}, B2_top2=family["B2"][:2],
                eligible_counts={k:len(v) for k,v in family.items()},
                PR_to_baseline_point_ratio={k:pr["G_point"]/family[k][0]["G_point"] for k in ("B0", "B1")},
                B2_top2_descriptive_two_SE_separated=second["G_point"]-pr["G_point"] > 2*gapsd,
                exploration_best_B2=candidates("B2", source_maps=old_maps)[0],
                metric_best={metric:{k:candidates(k,metric)[0] for k in family} for metric in ("CX","depth","size")},
                common_preparation_crossings=crossings))
    rare = read(RUN / "confirmation_order_diagnostics.json")
    for r in rare["rows"]:
        assert r["observed_counts"] == {k:orders[r["cell_id"]][k] for k in r["orders"]}
        assert sum(r["observed_counts"].values()) == 32*r["R"]
        p = math.fsum(v for k,v in r["orders"].items() if r["observed_counts"][k] == 0)
        r["stable_probability_trajectory_contains_unseen_order"] = -math.expm1(r["R"] * math.log1p(-p))
    coverage = read(RUN / "confirmation_coverage.json")
    assert not coverage["exploration_B2_in_confirmation_mean"] and len(coverage["unconfirmed_B2_ids"]) == 64
    calls = read(RUN / "aggregate_calls.json")
    assert calls["historical_unknown_actual_totals"] is None and not calls["failed_calls_refunded"] and not calls["budget_reset"]
    for k in ("compile", "trajectory"):
        assert calls["known_calls"][k]+calls["reservations"][k] <= m["cumulative_accounting"]["proposed_caps"][k]
    summed_hash = {k:sum(s[k] for s in hash_stats) for k in ("hash_calls","final_digest_hits","operation_byte_hits","operation_byte_misses","evictions")}
    summed_hash["workers_sampled"] = len(hash_stats)
    summed_hash["operation_byte_hit_fraction"] = summed_hash["operation_byte_hits"] / (summed_hash["operation_byte_hits"]+summed_hash["operation_byte_misses"])
    parent_resources = read(Path(m["confirmation_reuse"]["parent_run"]) / "confirmation_resource_summary.json")
    summary = dict(schema="h6_independent_confirmation_result_v1", source_commit=SOURCE,
        science_source=m["base_manifest"]["science_source"], environment=m["base_manifest"]["environment"], profile=profile,
        input_identity=m["base_manifest"]["input_identity"], selection_digest=receipt["selection_digest"],
        completion=receipt, native_exit=native_exit, resources=resources,
        parent_supervisor_wall_seconds=parent_resources["wall_seconds"],
        two_supervisor_wall_sum_seconds=parent_resources["wall_seconds"]+resources["wall_seconds"],
        comparisons=comparisons, coverage=coverage, calls=calls, hash_reuse_statistics=summed_hash,
        scope="linear H6/1.00 Angstrom/STO-3G/tol-only DF19/sector400/T0.8/prefix5/10/15/full19/delta0.1-0.8; P=0, empirical allowance; eight selected cells, 32 paired trajectories each",
        formal_accuracy_certified=False, population_winner="UNDETERMINED_POPULATION_MEAN", next_stage_authorized=False)
    write("result_summary_v1.json", summary)
    write("confirmation_resource_map_v1.json", maps)
    write("rare_order_diagnostics_v1.json", rare)
    write("integrated_confirmation_v1.json", dict(groups=provenance, reused_deterministic_baselines=baselines))
    with (HERE / "wrapper_metrics_v1.csv").open("x", newline="") as stream:
        fields=["cell_id","phase","replica","seed","axis","control","event_digest","compiler_hash","fingerprint","RZ","CX","RZ_depth","CX_depth","depth","size"]
        w=csv.DictWriter(stream, fieldnames=fields, lineterminator="\n");w.writeheader()
        for r in rows:
            w.writerow({k:r["metrics"][k] if k in r["metrics"] else r[k] for k in fields})
    write("summary_audit_v1.json", dict(status="LOCAL_SAVED_METADATA_AND_INTEGER_ACCOUNTING_VERIFIED",
        source_commit=SOURCE, source_files_verified=len(m["source_hashes"]), groups_verified=256,
        wrapper_records_verified=512, resource_map_rows_independently_recomputed=len(maps["rows"]),
        formal_scientific_review=False, CI=False, scientific_array_decodes=0, new_science_calls=0, new_transpiles=0,
        report_process=dict(address_space_cap_bytes=2*2**30, wall_cap_seconds=180, threads=1, nice=10, output_cap_per_file_bytes=16*2**20),
        native_file_hashes={name:sha(RUN/name) for name in ("frozen_manifest.json","execution_terminal.json","confirmation_complete_receipt.json","confirmation_resource_summary.json","confirmation_resource_map.json","confirmation_order_diagnostics.json","confirmation_coverage.json","aggregate_calls.json")},
        native_exit_sha256=sha(EVIDENCE/"coordinator_exit.json"),
        generated_file_hashes={p.name:sha(p) for p in HERE.iterdir() if p.is_file()}))
    print(json.dumps(dict(groups=256, wrappers=512, map_rows=len(maps["rows"]), primary=[{k:r[k] for k in ("epsilon","PR_to_baseline_point_ratio","B2_top2_descriptive_two_SE_separated")} for r in comparisons[:4]], hash_statistics=summed_hash)))


if __name__ == "__main__":
    main()
