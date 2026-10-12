"""H6 saved-JSON aggregation only; never launch a worker or read scientific arrays.

Run from the matching worktree with Python -P -B. Existing frozen metadata
admission/accounting functions supply the checks and original shot definition.
The report is an exploration summary, not a confirmation grant or selection seal.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import resource
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = "78a39051d8b6d41c33d592ecfe172f8824adfaeb"
RUN = ROOT / "artifacts/resource_applicability/track_a_h6_roundoff_resume01/2026-10-12/launch_parallel_v1"


def read(path):
    path = Path(path)
    if path.is_symlink() or path.stat().st_size > 16 * 2**20:
        raise ValueError("SUMMARY_INPUT_PATH_OR_SIZE")
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    resource.setrlimit(resource.RLIMIT_AS, (2**30, 2**30))
    sys.path.insert(0, str(ROOT / "src"))
    from trottertracks.resource_applicability.ax2a_preparation import digest
    from trottertracks.resource_applicability.h6_matched_reuse_v3 import admit_group, complete_exploration
    from trottertracks.resource_applicability.h6_parallel_execution_v1 import admit_parallel_group
    from trottertracks.resource_applicability.h6_matched_execution_v2 import resource_maps
    from trottertracks.resource_applicability.h6_matched_accounting_v1 import resource_row, order_diagnostic, precision_rows

    manifest = read(RUN / "frozen_manifest.json")
    base = manifest["base_manifest"]
    registry = base["reuse_registry"]
    plan = base["plan"]
    terminal = read(RUN / "execution_terminal.json")
    completion = read(RUN / "exploration_complete_receipt.json")
    summary = read(RUN / "parallel_resource_summary.json")
    if (manifest["source_commit"] != SOURCE
            or terminal["status"] != "H6_PARALLEL_COMPLETE_MANDATORY_STOP"
            or terminal["reason"] is not None
            or terminal["source_commit"] != SOURCE
            or terminal["resource_summary"] != summary
            or completion["group_count"] != 592 or completion["wrapper_count"] != 1188
            or not completion["selection_not_performed"] or not completion["confirmation_not_started"]
            or not summary["all_dispatched_reaped"] or summary["cleanup_errors"]
            or not summary["ownership_cleanup_verified"] or not summary["log_and_journal_closed"]):
        raise ValueError("SUMMARY_COMPLETION_IDENTITY")
    for path, expected in manifest["source_hashes"].items():
        if sha(ROOT / path) != expected:
            raise ValueError("SUMMARY_SOURCE_CHANGED:" + path)

    registered = registry["original_tasks"]
    groups, paths, provenance = {}, {}, []

    def add(index, directory, source, origin, old=False, expected_sha=None):
        if index in groups:
            raise ValueError("SUMMARY_DUPLICATE_GROUP")
        directory = Path(directory)
        result_path = directory / "cost_result.json"
        result_sha = sha(result_path)
        if expected_sha is not None and result_sha != expected_sha:
            raise ValueError("SUMMARY_SAVED_RESULT_CHANGED")
        groups[index] = (admit_group if old else admit_parallel_group)(
            directory, registered[index], source=source)
        paths[index] = directory
        provenance.append(dict(index=index, origin=origin, source_commit=source,
            task_digest=digest(registered[index]), cost_result_path=str(result_path),
            cost_result_sha256=result_sha, wrapper_count=len(groups[index]["rows"])))

    for row in registry["accepted"]:
        rp = row["directory"] + "/cost_result.json"
        add(row["index"], ROOT / row["directory"], registry["source_commit"], "legacy449", True,
            registry["raw_hashes"][rp])
    phase = manifest["phase_a_receipt"]
    add(447, Path(phase["run"]) / "cost_exploration_0447", phase["source_commit"], "server_phase_a",
        expected_sha=phase["cost_result_sha256"])
    for row in manifest["verified_reuse"]["accepted"]:
        add(row["index"], row["directory"], row["source_commit"], "saved_roundoff7",
            expected_sha=row["cost_result_sha256"])
    for row in manifest["registrations"]:
        add(row["index"], RUN / f'cost_exploration_{row["index"]:04d}', SOURCE, "resume01_new135")
    rows = complete_exploration(registered, groups)
    if len(groups) != 592 or len(rows) != 1188:
        raise ValueError("SUMMARY_COMPLETE_SET")
    primary = [row for row in rows if row["control"] == "symmetric_directional"]
    if len(primary) != 1184 or len(rows) - len(primary) != 4:
        raise ValueError("SUMMARY_WRAPPER_COVERAGE")

    old_run = ROOT / registry["original_raw_path"]
    signal_path = old_run / "signal/signal_summary.json"
    b3_path = old_run / "signal/b3_normalization_diagnostic.json"
    for path in (signal_path, b3_path):
        if sha(path) != registry["raw_hashes"][str(path.relative_to(ROOT))]:
            raise ValueError("SUMMARY_SIGNAL_BYTES")
    signals = read(signal_path)["signals"]
    b3 = read(b3_path)
    if len(signals) != 92 or len(b3["rows"]) != 84:
        raise ValueError("SUMMARY_SIGNAL_COVERAGE")
    by_cell = {s["cell"]["id"]: s for s in signals}
    if len(by_cell) != 92 or set(by_cell) != {c["id"] for c in plan["cells"]}:
        raise ValueError("SUMMARY_SIGNAL_IDS")
    for s in signals:
        signal_record = old_run / "signal" / (s["cell"]["id"] + "_signal.json")
        if (sha(signal_record) != registry["raw_hashes"][str(signal_record.relative_to(ROOT))]
                or read(signal_record) != s or precision_rows(s, plan) != s["precision_rows"]):
            raise ValueError("SUMMARY_PRECISION_ACCOUNTING")

    cell_rows = defaultdict(list)
    for row in rows:
        cell_rows[row["cell_id"]].append(row)
    maps = resource_maps(signals, rows, plan, "exploration")
    comparisons = []
    for factor in ("1.0", "10.0"):
        for epsilon in plan["epsilons"]:
            family_best, eligible_counts, family_top2 = {}, {}, {}
            for method in ("B0", "B1", "B2"):
                candidates = [r for r in maps["rows"] if r["epsilon"] == epsilon
                    and r["metric"] == "RZ" and r["allowance_factor"] == factor and r["method"] == method]
                candidates.sort(key=lambda r: (r["G_point"], r["cell_id"]))
                family_best[method] = candidates[0] if candidates else None
                eligible_counts[method] = len(candidates)
                family_top2[method] = candidates[:2]
            b2 = family_best["B2"]
            ratios, prep_crossings = {}, {}
            for method in ("B0", "B1"):
                baseline = family_best[method]
                if b2 and baseline:
                    ratios[method] = b2["G_point"] / baseline["G_point"]
                    delta_n = b2["N_total"] - baseline["N_total"]
                    crossing = ((baseline["G_point"] - b2["G_point"]) / delta_n) if delta_n else None
                    prep_crossings[method] = dict(crossing_P=crossing if crossing is not None and crossing >= 0 else None,
                        units="RZ per common state preparation", slope_difference=delta_n,
                        intercept_difference=b2["G_point"] - baseline["G_point"])
            full_pf_by_order = {}
            for formula in ("2nd", "4th"):
                candidates = [r for r in maps["rows"] if r["epsilon"] == epsilon and r["metric"] == "RZ"
                    and r["allowance_factor"] == factor and r["method"] == "B1"
                    and by_cell[r["cell_id"]]["cell"]["formula"] == formula]
                full_pf_by_order[formula] = min(candidates, key=lambda r: (r["G_point"], r["cell_id"])) if candidates else None
            comparisons.append(dict(epsilon=epsilon, allowance_factor=factor,
                eligible_counts=eligible_counts, family_best_point=family_best,
                family_top2_point=family_top2, full_pf_best_by_order=full_pf_by_order,
                PR_to_baseline_point_ratio=ratios, common_preparation_crossings=prep_crossings,
                candidate_selection_sealed=False, population_winner="UNDETERMINED_POPULATION_MEAN"))

    trajectories = defaultdict(list)
    for index, directory in sorted(paths.items()):
        tr = read(directory / "trajectory.json")
        if tr["finite_distribution"] is not None:
            trajectories[registered[index]["cell_id"]].append(tr)
    orders = []
    for cell_id, trs in sorted(trajectories.items()):
        if len(trs) != 8:
            raise ValueError("SUMMARY_RANDOM_SAMPLE_COUNT")
        distributions = {digest(tr["finite_distribution"]): tr["finite_distribution"] for tr in trs}
        law = trs[0]["finite_distribution"]
        counts = Counter(k for tr in trs for k in tr["observed_orders"])
        unseen_probability = math.fsum(p for k, p in zip(law["orders"], law["order_probabilities"], strict=True) if counts[k] == 0)
        unseen_per_trajectory = -math.expm1(trs[0]["task"]["cell"]["R"] * math.log1p(-unseen_probability)) if unseen_probability < 1 else 1.0
        orders.append(dict(cell_id=cell_id, phase="exploration", distribution_variants=len(distributions),
            saved_distributions=distributions, nonzero_observations=sum(v for k,v in counts.items() if k),
            unseen_single_draw_mass_stable=unseen_probability,
            unseen_trajectory_probability_stable=unseen_per_trajectory,
            **order_diagnostic(law, [k for tr in trs for k in tr["observed_orders"]],
                R=trs[0]["task"]["cell"]["R"], trajectories=len(trs))))

    ordinary = []
    for index, group in sorted(groups.items()):
        if not registered[index]["ordinary"]:
            continue
        for axis in ("cosine", "sine"):
            pairs = {row["control"]: row for row in group["rows"] if row["axis"] == axis}
            ordinary.append(dict(cell_id=registered[index]["cell_id"], replica=0, axis=axis,
                symmetric_directional=pairs["symmetric_directional"]["metrics"], ordinary=pairs["ordinary"]["metrics"],
                ordinary_to_primary_RZ=pairs["ordinary"]["metrics"]["RZ"] / pairs["symmetric_directional"]["metrics"]["RZ"],
                scope="paired replica0 only; not all-trajectory control equivalence"))

    signal_family = {}
    for method in ("B0", "B1", "B2"):
        ss = [s for s in signals if s["cell"]["method"] == method]
        signal_family[method] = dict(cell_count=len(ss), cost_acquired_cells=sum(bool(cell_rows[s["cell"]["id"]]) for s in ss),
            absolute_signal_bias_range=[min(s["error_decomposition"]["total_absolute"] for s in ss), max(s["error_decomposition"]["total_absolute"] for s in ss)],
            B_range=[math.exp(min(s["log_B"] for s in ss)), math.exp(max(s["log_B"] for s in ss))])

    loaded = [name for name in ("numpy", "scipy", "qiskit", "pyscf", "cupy") if name in sys.modules]
    if loaded:
        raise ValueError("SUMMARY_SCIENCE_IMPORT:" + str(loaded))
    result = dict(schema="h6_saved_exploration_summary_v1", status="EXPLORATION_COMPLETE_CONFIRMATION_PENDING",
        source_commit=SOURCE, science_source=base["science_source"], target=plan["target"], T=plan["T"], rank=plan["actual_rank"],
        df_policy=plan["df_policy"], compiler=plan["compiler_proposed"], compiler_hash=rows[0]["compiler_hash"],
        environment=base["environment"], profile=base["profile"], input_identity=base["input_identity"],
        normalization_and_shot_policy=plan["shot_rule"], common_preparation_cost=plan["common_preparation_cost"],
        delta_window=sorted({plan["T"]/c["q"] for c in plan["cells"]}),
        coverage=dict(signals=92, B3_combinations=84, exploration_groups=592, primary_wrappers=1184,
            ordinary_sensitivity_wrappers=4, total_wrappers=1188, B2_cost_cells=72, B2_trajectories_per_cell=8,
            B0_B1_deterministic_cost_cells=16, missing_registered_groups=0,
            provenance_counts=dict(Counter(row["origin"] for row in provenance)), independent_confirmation_groups=0),
        resources=dict(resume01=summary, proposed=manifest["proposed_resources"]), signal_family=signal_family,
        comparisons=comparisons, ordinary_sensitivity=ordinary,
        B3_scope=dict(signal_evaluated=b3["signal_evaluated"], cost_evaluated=b3["cost_evaluated"],
            family_excluded=b3["family_excluded"], log_B_range=[min(r["log_B"] for r in b3["rows"]),max(r["log_B"] for r in b3["rows"])]),
        roundoff_policy=manifest["roundoff_preparation_binding"],
        accounting=dict(known_calls=completion["known_calls"], unknown_old_aborted_actual_totals=None,
            reservations=completion["reservations"], reservations_are_not_actual_totals=True, budget_reset=False),
        confirmation_started=False, candidate_selection_sealed=False, new_grant_created=False,
        population_winner="UNDETERMINED_POPULATION_MEAN", formal_accuracy_certified=False,
        compiler_output_bit_equivalence_to_old_host_verified=False, ground_state_certified=False,
        new_signal_calls=0, new_trajectory_calls=0, new_compile_calls=0, scientific_arrays_read=False,
        science_libraries_imported=loaded, agent_push=False,
        saved_execution_terminal_sha256=sha(RUN/"execution_terminal.json"),
        saved_completion_receipt_sha256=sha(RUN/"exploration_complete_receipt.json"))

    output_names = ("result_summary_v1.json", "exploration_resource_map_v1.json", "integrated_exploration_v1.json",
        "cost_order_diagnostics_v1.json", "candidate_summary_v1.csv", "wrapper_metrics_v1.csv", "summary_audit_v1.json")
    if any((HERE/name).exists() for name in output_names):
        raise ValueError("SUMMARY_OUTPUT_ALREADY_EXISTS")

    def write_json(name, value):
        with (HERE/name).open("x") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, separators=(",",":"), allow_nan=False)
            stream.write("\n")

    write_json("result_summary_v1.json", result)
    write_json("exploration_resource_map_v1.json", maps)
    write_json("integrated_exploration_v1.json", dict(group_count=592, wrapper_count=1188,
        groups=sorted(provenance,key=lambda r:r["index"]), exploration_rows_digest=digest(rows)))
    write_json("cost_order_diagnostics_v1.json", dict(rows=orders, population_mean_certified=False,
        stable_probability_fields="log1p/expm1 avoid subtraction of rounded p0 from one; finite-law values remain binary64",
        component_sequence_support_fully_covered=False, cost_mean_tail_bound=None))

    columns = ["cell_id","method","prefix","formula","q","r","R","K","epsilon","allowance_factor","eligible",
        "absolute_signal_bias","bias_real","bias_imag","B","allowance_axis","N_total","cosine_shots","sine_shots",
        "paired_cost_trajectories","cosine_RZ_mean","sine_RZ_mean","G_RZ_point","G_RZ_standard_error",
        "G_CX_point","G_depth_point","G_size_point"]
    with (HERE/"candidate_summary_v1.csv").open("x",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=columns,lineterminator="\n");writer.writeheader()
        for s in sorted(signals,key=lambda s:s["cell"]["id"]):
            c=s["cell"];cr=cell_rows[c["id"]]
            for epsilon in plan["epsilons"]:
                for factor in ("1.0","10.0"):
                    acc=s["precision_rows"][str(epsilon)][factor]
                    rz=resource_row(s,cr,epsilon,factor=factor)
                    row=dict(cell_id=c["id"],**{key:c[key] for key in ("method","prefix","formula","q","r","R","K")},
                        epsilon=epsilon,allowance_factor=factor,eligible=acc["cost_acquisition_eligible"],
                        absolute_signal_bias=s["error_decomposition"]["total_absolute"],
                        bias_real=s["error_decomposition"]["total_signed"]["real"],
                        bias_imag=s["error_decomposition"]["total_signed"]["imag"],B=math.exp(s["log_B"]),
                        allowance_axis=float(factor)*s["allowance"]["real"],N_total=acc["N_total"],
                        cosine_shots=acc["axes"]["real"]["shots"],sine_shots=acc["axes"]["imag"]["shots"],
                        paired_cost_trajectories=len({r["replica"] for r in cr if r["control"]=="symmetric_directional"}))
                    if rz:
                        row.update(cosine_RZ_mean=rz["axes"]["cosine"]["mean"],sine_RZ_mean=rz["axes"]["sine"]["mean"],
                            G_RZ_point=rz["G_point"],G_RZ_standard_error=rz["G_standard_error"])
                        for metric in ("CX","depth","size"):
                            row["G_"+metric+"_point"]=resource_row(s,cr,epsilon,metric=metric,factor=factor)["G_point"]
                    writer.writerow(row)
    columns=["index","source_commit","origin","cell_id","phase","replica","seed","control","axis", "compiler_hash", "event_digest", "fingerprint",
        "RZ","CX","RZ_depth","CX_depth","depth","size","classical_wall_seconds","worker_peak_rss_bytes"]
    sources={r["index"]:r for r in provenance}
    with (HERE/"wrapper_metrics_v1.csv").open("x",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=columns,lineterminator="\n");writer.writeheader()
        for index,group in sorted(groups.items()):
            for r in group["rows"]:
                writer.writerow(dict(index=index,source_commit=sources[index]["source_commit"],origin=sources[index]["origin"],
                    **{key:r[key] for key in columns[3:12]},**r["metrics"],
                    classical_wall_seconds=r.get("classical_wall_seconds"),worker_peak_rss_bytes=r.get("worker_peak_rss_bytes")))
    write_json("summary_audit_v1.json",dict(status="LOCAL_SAVED_METADATA_AUDIT_PASS", source_commit=SOURCE,
        source_closure_file_count=len(manifest["source_hashes"]), admitted_groups=592, paired_primary_wrappers=1184,
        sensitivity_wrappers=4, signal_records=92, all_precision_rows_recomputed_equal=True,
        numeric_arrays_read=False, new_compile_calls=0,new_trajectory_calls=0,new_signal_calls=0,
        independent_review_or_CI=False,new_grant_created=False,candidate_selection_sealed=False,
        raw_execution_unchanged=True, scientific_source_unchanged=True,
        input_bindings={str(path):dict(bytes=path.stat().st_size,sha256=sha(path)) for path in
            (RUN/"frozen_manifest.json",RUN/"exploration_complete_receipt.json",RUN/"execution_terminal.json",signal_path,b3_path)},
        output_files={name:dict(bytes=(HERE/name).stat().st_size,sha256=sha(HERE/name)) for name in output_names[:-1]},
        aggregation_script_sha256=sha(__file__),script_memory_cap_bytes=2**30))
    print(json.dumps(dict(status="LOCAL_SAVED_METADATA_AUDIT_PASS",groups=592,wrappers=1188,
        total_report_data_bytes=sum((HERE/name).stat().st_size for name in output_names),
        best=[dict(epsilon=c["epsilon"],PR=c["family_best_point"]["B2"]["cell_id"],
            ratios=c["PR_to_baseline_point_ratio"]) for c in comparisons if c["allowance_factor"]=="1.0"])))


if __name__ == "__main__":
    main()
