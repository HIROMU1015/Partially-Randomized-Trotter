#!/usr/bin/env python3
"""Render existing G_RZ tokens only; no science modules, ranking, or accounting.

The selectors below copy saved comparison records, not minima computed from rows.
Python's standard library is sufficient for --check. Matplotlib is needed to build.
"""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
BASE = "c0808b254aa02a94c7482f5c6e353b54532a54f8"
OUT = ROOT / "artifacts/resource_applicability/track_a_pr_pf_absolute_cost_v2/2026-10-19"
EPSILONS = ("0.05", "0.01", "0.005", "0.001", "0.0005")
PROTOCOL = "docs/research/track_a_pr_resource_evaluation_protocol_v2_DRAFT.md"
V2 = "artifacts/resource_applicability/track_a_protocol_v2_draft/2026-10-19/"
PAPER = "artifacts/resource_applicability/track_a_h468_paper_display_methods/2026-10-19/"
NEW_A_COMMIT = "64c85f7a833b5950ef2ccae623708b73081edcd6"
NEW_A_DIR = "artifacts/resource_applicability/track_a_pr_new_a_total_cost_results/2026-10-18/"
SPECS = {
    "protocol": (BASE, PROTOCOL),
    "protocol_manifest": (BASE, V2 + "protocol_manifest.json"),
    "system_conditions": (BASE, V2 + "system_conditions.json"),
    "stage_status": (BASE, V2 + "stage_status.csv"),
    "main_table": (BASE, PAPER + "main_table.csv"),
    "paper_manifest": (BASE, PAPER + "publication_manifest.json"),
    "input_exceptions": (BASE, PAPER + "input_exceptions.csv"),
    "h2_a": (
        "829d697ae7905b3ea90a61da9d2d4da15d8b757a",
        "artifacts/resource_applicability/track_a_h2_pr_resource/2026-10-19/A_completed_run01/result.json",
    ),
    "h2_a_audit": (
        "829d697ae7905b3ea90a61da9d2d4da15d8b757a",
        "artifacts/resource_applicability/track_a_h2_pr_resource/2026-10-19/A_completed_run01/completion_audit.json",
    ),
    "h4_a": (
        "85cb5b4d1808434165c421115ed09dab3b63dd00",
        "artifacts/resource_applicability/track_a_h4_tol_a_completed/2026-10-19/result.json",
    ),
    "h4_a_audit": (
        "85cb5b4d1808434165c421115ed09dab3b63dd00",
        "artifacts/resource_applicability/track_a_h4_tol_a_completed/2026-10-19/completion_audit.json",
    ),
    "h68_a": (NEW_A_COMMIT, NEW_A_DIR + "comparisons.csv"),
    "h68_a_points": (NEW_A_COMMIT, NEW_A_DIR + "new_pr_cost_points.csv"),
    "h68_a_completion": (NEW_A_COMMIT, NEW_A_DIR + "completion_summary.json"),
    "h68_a_report": (NEW_A_COMMIT, "docs/research/track_a_pr_new_a_total_cost_results_20261018.md"),
}
COLUMNS = (
    "comparison_id", "epsilon", "panel", "system", "hydrogen_atoms", "geometry",
    "basis", "DF_policy", "DF_rank", "T", "P", "bias_mode", "allowance_factor",
    "PR_id", "PF_id", "L_D", "q", "delta", "r", "K", "n", "PR_cohort",
    "G_PR_exact", "G_PR_paired_SE_saved", "G_PF_exact", "PF_cost_class",
    "PF_cost_status_saved", "PF_selection_scope", "saved_direction_status",
    "audit_status", "result_commit", "result_file", "result_record",
    "scientific_SOURCE", "hamiltonian_hash", "state_hash", "snapshot_hash",
    "compiler_settings_hash", "mechanism_alias", "limits",
)
LIMITS = (
    "FINITE_SAMPLE_POINT;NO_POPULATION_MEAN_GUARANTEE;UNOBSERVED_RARE_COST_UNKNOWN;"
    "EMPIRICAL_BIAS_ALLOWANCE;ANALYTIC_SHOTS;COMPILED_RZ_NOT_FT;P0;"
    "NO_GLOBAL_OPTIMUM;INPUT_RULE_EQUIVALENCE_NOT_VERIFIED;NO_SIZE_SCALING_CLAIM;"
    "NO_COHORT_POOLING;NOT_FINAL_FREEZE"
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_bytes(commit: str, path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT)


def json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def csv_bytes(rows, columns) -> bytes:
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode()


def csv_rows(data: bytes) -> list[dict]:
    return list(csv.DictReader(io.StringIO(data.decode())))


def load_evidence():
    raw = {label: git_bytes(*spec) for label, spec in SPECS.items()}
    provenance = [dict(label=label, commit=SPECS[label][0], path=SPECS[label][1],
                       sha256=sha(data), bytes=len(data)) for label, data in raw.items()]
    current_protocol = ROOT / PROTOCOL
    if current_protocol.read_bytes() != raw["protocol"]:
        raise ValueError("The pinned v2 protocol differs from this checkout")
    paper = json.loads(raw["paper_manifest"])
    assert sha(raw["main_table"]) == paper["main_table"]["csv"]["sha256"]
    conditions = {s["system"]: s for s in json.loads(raw["system_conditions"])["systems"]}
    stages = {(r["system"], r["stage"]): r for r in csv_rows(raw["stage_status"])}
    h2, h4 = json.loads(raw["h2_a"]), json.loads(raw["h4_a"])
    assert h2["status"] == h4["status"] == "COMPLETE_MANDATORY_STOP"
    assert h2["FP_PR_cost_pooled_with_A"] is False
    assert h4["registered_set_complete"] is True
    completion = json.loads(raw["h68_a_completion"])
    for s in ("H6", "H8"):
        assert completion["systems"][s]["status"] == "COMPLETE_MANDATORY_STOP"
        assert completion["systems"][s]["native_exit_code"] == 0
        assert completion["systems"][s]["reaped"] is True

    selected = []
    for index, record in enumerate(csv_rows(raw["main_table"])):
        selected.append(("main_table", f"CSV data row {index + 1}", "independent", record))
    for label, obj, pointer in (("h2_a", h2["accounting"], "/accounting/comparisons/"),
                                 ("h4_a", h4, "/comparisons/")):
        for index, record in enumerate(obj["comparisons"]):
            selected.append((label, pointer + str(index), "exploratory", record))
    scopes = {"H6": "strengthened_S2_S4", "H8": "strengthened184"}
    cost_points = {(r["system"], r["cell_id"], r["bias_mode"], r["epsilon"], r["factor"]): r
                   for r in csv_rows(raw["h68_a_points"])}
    for index, record in enumerate(csv_rows(raw["h68_a"])):
        if (record["system"] in scopes and record["bias_mode"] == "USUAL_SIGNED"
                and record["factor"] == "1.0"
                and record["deterministic_scope"] == scopes[record["system"]]):
            assert record["new_domain_complete"] == "True"
            assert record["unresolved_new_domain_ids"] == ""
            selected.append(("h68_a", f"CSV data row {index + 1}", "exploratory", record))

    table, source_records = [], []
    for label, pointer, panel, original in selected:
        if label == "main_table":
            s = original["system"]
            eps, pr_id, pf_id = original["epsilon"], original["PR_id"], original["PF_id"]
            g_pr, g_pf = original["G_PR_exact"], original["G_PF_exact"]
            se, n, cohort = original["paired_SE"], original["n"], original["cohort"]
            cost_status, pf_scope = original["PF_cost_class"], "PREREGISTERED_FIXED_P_CONTROL"
            direction, audit = original["saved_direction_status"], original["audit_status"]
            result_commit, result_file, result_record = (
                original["result_commit"], original["source_file"], original["source_record"])
            source = stages[(s, "P")]["scientific_SOURCE"]
            alias = original["mechanism_core_alias"]
        elif label in ("h2_a", "h4_a"):
            s = "H2" if label == "h2_a" else "H4"
            pr, pf = original["PR"], original["PF"]
            assert pr["eligible"] and pf["eligible"]
            eps, pr_id, pf_id = str(original["epsilon"]), pr["cell_id"], pf["cell_id"]
            g_pr, g_pf, se = pr["G_RZ_exact"], pf["G_RZ_exact"], str(pr["G_RZ_SE"])
            n, cohort = str(pr["n"]), pr["cohort"]
            assert n == "8" and cohort == "EXPLORATION_N8"
            cost_status = "DIRECT_MEASURED"
            pf_scope = "D70_SAVED_WINNER" if s == "H2" else "D190_SAVED_WINNER"
            direction = original.get("status", "FINITE_SAMPLE_POINT")
            audit = "SAVED_PUBLIC_COMPLETION_AUDIT;NOT_REPLAYED"
            result_commit, result_file = SPECS[label]
            result_record = pointer
            source = h2["SOURCE"] if s == "H2" else h4["SOURCE"]
            binding = h2["binding"] if s == "H2" else h4["binding"]
            assert binding["rank"] == conditions[s]["actual_rank"]
            assert binding["hamiltonian"] == conditions[s]["hamiltonian_hash"]
            assert binding["state"] == conditions[s]["state_hash"]
            alias = ""
        else:
            s, eps = original["system"], original["epsilon"]
            pr_id, pf_id = original["new_best_id"], original["PF_best_available_id"]
            g_pr, g_pf, se = original["new_n8_G_exact"], original["PF_G_exact"], original["new_n8_G_SE"]
            point = cost_points[(s, pr_id, "USUAL_SIGNED", eps, "1.0")]
            assert point["status"] == "FINITE_SAMPLE_POINT"
            assert point["G_RZ_exact"] == g_pr and point["G_RZ_SE"] == se
            n, cohort = point["n_used"], point["cohort"]
            assert n == "8" and cohort == "EXPLORATION_N8"
            cost_status, pf_scope = original["PF_cost_status"], original["deterministic_scope"]
            direction, audit = "FINITE_SAMPLE_POINT", "SAVED_PUBLIC_COMPLETION_SUMMARY;NOT_REPLAYED"
            result_commit, result_file = SPECS[label]
            result_record, source, alias = pointer, original["SOURCE"], ""
        assert eps in EPSILONS
        if any(v in (None, "", "null") for v in (g_pr, g_pf, se, pr_id, pf_id)):
            raise ValueError("Missing selected saved value; do not fill or reconstruct")
        assert Fraction(g_pr) > 0 and Fraction(g_pf) > 0 and float(se) >= 0
        model = cost_status == "VALIDATED_MODEL_ESTIMATE"
        assert model == (panel == "exploratory" and s == "H6" and eps in EPSILONS[1:4])
        cell = re.fullmatch(r"B2_p(\d+)_q(\d+)_2nd_r(\d+)_K(\d+)", pr_id)
        assert cell is not None
        ld, q, r, k = cell.groups()
        # Literal T=.8 schedule already specified by v2; no signal/shot computation.
        delta = {"1": "0.8", "2": "0.4", "4": "0.2", "8": "0.1", "16": "0.05"}[q]
        c = conditions[s]
        row = dict(comparison_id=f"{panel}_{s}_eps_{eps}", epsilon=eps, panel=panel,
                   system=s, hydrogen_atoms=s[1:], geometry=c["molecule_geometry"],
                   basis=c["basis"], DF_policy=c["DF_policy"], DF_rank=str(c["actual_rank"]),
                   T=c["T"], P=str(c["P"]), bias_mode="USUAL_SIGNED", allowance_factor="1",
                   PR_id=pr_id, PF_id=pf_id, L_D=ld, q=q, delta=delta, r=r, K=k, n=n,
                   PR_cohort=cohort, G_PR_exact=g_pr, G_PR_paired_SE_saved=se, G_PF_exact=g_pf,
                   PF_cost_class="VALIDATED_MODEL_ESTIMATE" if model else "DIRECT_MEASURED",
                   PF_cost_status_saved=cost_status, PF_selection_scope=pf_scope,
                   saved_direction_status=direction, audit_status=audit,
                   result_commit=result_commit, result_file=result_file, result_record=result_record,
                   scientific_SOURCE=source, hamiltonian_hash=c["hamiltonian_hash"],
                   state_hash=c["state_hash"], snapshot_hash=c["snapshot_hash"],
                   compiler_settings_hash=c["compiler_settings_hash"], mechanism_alias=alias,
                   limits=LIMITS + (";H2_P_NOT_EXECUTED" if s == "H2" else "")
                   + (";H8_FULL_PUBLIC_PR_AUDIT_NOT_REVERIFIED" if s == "H8" else ""))
        table.append(row)
        source_records.append(dict(comparison_id=row["comparison_id"], source_label=label,
                                   source_commit=SPECS[label][0], source_file=SPECS[label][1],
                                   record_pointer=pointer, saved_record=original))
    # Ordering is by requested epsilon, evidence panel and atom count, never cost.
    table.sort(key=lambda r: (EPSILONS.index(r["epsilon"]), r["panel"], int(r["hydrogen_atoms"])))
    assert len(table) == 26 and len({r["comparison_id"] for r in table}) == 26
    assert {r["DF_rank"] for r in table if r["system"] == "H4"} == {"10"}
    for eps in EPSILONS:
        assert {r["system"] for r in table if r["epsilon"] == eps and r["panel"] == "exploratory"} == {"H2", "H4", "H6", "H8"}
        expected = {"H4", "H6", "H8"} if eps in (EPSILONS[0], EPSILONS[-1]) else set()
        assert {r["system"] for r in table if r["epsilon"] == eps and r["panel"] == "independent"} == expected
    return table, source_records, provenance


def points_from(table):
    points = []
    for r in table:
        for method in ("PR", "PF"):
            token = r[f"G_{method}_exact"]
            coordinate = float(Fraction(token))  # representation conversion only
            if not math.isfinite(coordinate) or coordinate <= 0:
                raise ValueError("Log plot requires a finite positive saved value")
            points.append(dict(comparison_id=r["comparison_id"], epsilon=r["epsilon"],
                               panel=r["panel"], system=r["system"], hydrogen_atoms=r["hydrogen_atoms"],
                               method=method, G_RZ_exact=token, G_RZ_plot=repr(coordinate),
                               display_text=format(coordinate, ".3g"),
                               saved_paired_SE=r["G_PR_paired_SE_saved"] if method == "PR" else "",
                               cost_class=r["PF_cost_class"] if method == "PF" else "FINITE_SAMPLE_MEAN",
                               PR_cohort=r["PR_cohort"], n=r["n"],
                               result_commit=r["result_commit"], result_file=r["result_file"],
                               result_record=r["result_record"]))
    return points


def write_display_tables(table):
    lines = ["# 描画に使った保存費用（閲覧用）", "",
             "数値は保存exact tokenの表示用変換（9桁）。新しい会計・平均・順位計算なし。"
             "exact値・候補・出典・入力hashは[comparisons.csv](comparisons.csv)を参照。", ""]
    for eps in EPSILONS:
        for panel in ("independent", "exploratory"):
            rows = [r for r in table if r["epsilon"] == eps and r["panel"] == panel]
            if not rows:
                continue
            lines.extend([f"## ε={eps} / {panel}", "",
                          "|系・DF|n|G_PR|保存paired SE|G_PF|PF証拠|",
                          "|---|---:|---:|---:|---:|---|"])
            for r in rows:
                gpr = format(float(Fraction(r["G_PR_exact"])), ".9g")
                gpf = format(float(Fraction(r["G_PF_exact"])), ".9g")
                lines.append(f"|{r['system']} DF{r['DF_rank']}|{r['n']}|{gpr}|"
                             f"{r['G_PR_paired_SE_saved']}|{gpf}|{r['PF_cost_class']}|")
            lines.append("")
    (OUT / "data/comparisons_display.md").write_text("\n".join(lines), encoding="utf-8")
    decisions = []
    for r in table:
        reason = "Saved fixed P comparison; retain original cohort and audit status" if r["panel"] == "independent" else "Saved A comparison; exploration n8 only; no new winner selection"
        if r["system"] == "H2":
            reason += "; H2 A complete, P incomplete, exploratory panel only"
        if r["PF_cost_class"] == "VALIDATED_MODEL_ESTIMATE":
            reason += "; PF model estimate shown by open square"
        if r["system"] == "H8":
            reason += "; H8 full public PR receipts/profile bytes not reverified"
        decisions.append(dict(decision="INCLUDE_DISPLAY_ONLY_WITH_RECORDED_LIMITS", system=r["system"],
                              epsilon=r["epsilon"], comparison_id=r["comparison_id"], panel=r["panel"], reason=reason))
    excluded = [
        ("H2", "0.05;0.0005", "independent", "H2 P not executed/not sealed at pinned v2; no independent value"),
        ("H4;H6;H8", "0.01;0.005;0.001", "independent", "No independent P comparison in adopted saved main table; no filling from exploration"),
        ("H4", "all5", "all", "Legacy fixed DF12 is a different input series; use current tol-only DF10"),
        ("H4;H6;H8", "all5", "exploratory", "Old PR n32/union n32/FP n32 kept separate; no unregistered union winner"),
        ("H2", "all5", "exploratory", "FP diagnostic n1 not pooled with A n8"),
        ("H6;H8", "all5", "exploratory", "Weaker old32 PF domain not substituted for saved strengthened comparison"),
        ("all", "all5", "all", "Noncancellation and allowance factor10 outside requested usual-signed/factor1 scope; missing stays unknown"),
        ("all", "other", "all", "Additional epsilon values and deferred additional PF formulas outside scope"),
    ]
    for s, eps, panel, reason in excluded:
        decisions.append(dict(decision="EXCLUDE_FROM_THIS_FIGURE_SET", system=s, epsilon=eps,
                              comparison_id="", panel=panel, reason=reason))
    (OUT / "data/selection_decisions.csv").write_bytes(csv_bytes(decisions, list(decisions[0])))


def render(table, points):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".plot-cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogLocator, LogFormatterSciNotation, NullFormatter

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "svg.hashsalt": "pr-pf-absolute-v2",
                         "savefig.facecolor": "white", "pdf.fonttype": 42})
    colors = {"PR": "#0072B2", "PF": "#D55E00"}
    figures = []
    for eps in EPSILONS:
        panels = ["independent", "exploratory"] if eps in (EPSILONS[0], EPSILONS[-1]) else ["exploratory"]
        fig, axes = plt.subplots(1, len(panels), figsize=(12, 7), squeeze=False, sharey=True)
        fig.subplots_adjust(left=.087, right=.98, bottom=.25, top=.76, wspace=.17)
        fig.suptitle(r"PR vs deterministic PF  |  $\epsilon = " + eps + "$", fontsize=19, weight="bold", y=.975)
        fig.text(.5, .909, "Linear H-chains, 1 Å  ·  STO-3G  ·  T = 0.8  ·  P = 0  ·  usual signed bias / factor 1",
                 ha="center", fontsize=10.5, color="#444444")
        legend = [Line2D([], [], color=colors["PR"], marker="o", lw=1.8, ms=6, label="PR: saved mean ± 1 SE"),
                  Line2D([], [], color=colors["PF"], marker="s", ls="--", lw=1.7, ms=7, label="Deterministic PF")]
        if len(panels) == 1:
            legend.append(Line2D([], [], color=colors["PF"], marker="s", mfc="white", ls="None", ms=7,
                                 label="Open square: PF model estimate"))
        fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.53, .875), ncol=len(legend), frameon=False, fontsize=10)
        values = [float(p["G_RZ_plot"]) for p in points if p["epsilon"] == eps]
        # Limits and label offsets are layout, not new scientific estimators.
        low, high = min(values) / 2.2, max(values) * 2.2
        for ax, panel in zip(axes[0], panels):
            rows = [r for r in table if r["epsilon"] == eps and r["panel"] == panel]
            xs = [int(r["hydrogen_atoms"]) for r in rows]
            for method in ("PF", "PR"):
                ys = [float(Fraction(r[f"G_{method}_exact"])) for r in rows]
                ax.plot(xs, ys, color=colors[method], ls="--" if method == "PF" else "-", lw=1.7, alpha=.85, zorder=2)
                for x, y, row in zip(xs, ys, rows):
                    model = method == "PF" and row["PF_cost_class"] == "VALIDATED_MODEL_ESTIMATE"
                    ax.plot(x, y, marker="s" if method == "PF" else "o", ms=8 if method == "PF" else 6,
                            mfc="white" if model else colors[method], mec=colors[method], mew=1.6, zorder=3 if method == "PF" else 4)
                    if method == "PR":
                        ax.errorbar(x, y, yerr=float(row["G_PR_paired_SE_saved"]), fmt="none", ecolor=colors[method],
                                    capsize=3, lw=1.1, zorder=4)
                    other = float(Fraction(row["G_PF_exact" if method == "PR" else "G_PR_exact"]))
                    above = y >= other
                    ax.annotate(format(y, ".3g"), (x, y), xytext=(0, 12 if above else -15), textcoords="offset points",
                                ha="center", va="bottom" if above else "top", fontsize=9, color=colors[method],
                                bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none", alpha=.94))
            ax.set_yscale("log")
            ax.set_ylim(low, high)
            ax.set_xlim(min(xs) - .6, max(xs) + .6)
            ax.set_xticks(xs, [r["system"] + ("†" if r["system"] == "H8" else "") + "\nDF" + r["DF_rank"] for r in rows])
            ax.set_xlabel("Number of hydrogen atoms", labelpad=9)
            ax.yaxis.set_major_locator(LogLocator(base=10, numticks=8))
            ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.grid(axis="y", which="major", color="#DFE4E8", lw=.8, zorder=0)
            ax.tick_params(axis="x", length=0, pad=9)
            if panel == "independent":
                ax.set_title("Independent confirmation P", fontsize=12, weight="bold", pad=10)
                ns = ", ".join(f"{r['system']} n={r['n']}" for r in rows)
                ax.text(.02, .97, ns, transform=ax.transAxes, va="top", fontsize=8.5, color="#555555")
            else:
                ax.set_title("Exploratory A  |  n = 8 per selected candidate", fontsize=12, weight="bold", pad=10)
                ax.text(.02, .97, "Saved selections; no independent P at this precision" if len(panels) == 1
                        else "Separate A cohorts; H2 P is incomplete", transform=ax.transAxes,
                        va="top", fontsize=8.5, color="#555555")
        axes[0, 0].set_ylabel(r"Shot-inclusive total RZ cost  $G_{\mathrm{RZ}}$  (log scale)", labelpad=12)
        footer = (
            "Stored analytic shots × compiled RZ; PR bars are saved paired ±1 SE, not confidence intervals.\n"
            "† H8: full public PR execution receipts not reverified; original profile bytes remain unverified.\n"
            "Input-generation procedures differ. Lines guide the eye across discrete chains; no scaling or population-mean claim."
        )
        if len(panels) == 1:
            footer += "\nH6 PF is a validated model estimate; the registered H6/H8 PF domains retain model dependence."
        else:
            footer += "\nFixed P controls are directly measured; the exploratory H6/H8 registered PF domains retain model dependence."
        fig.text(.087, .035, footer, fontsize=8.5, linespacing=1.5, va="bottom", color="#444444")
        stem = "absolute_rz_eps_" + eps.replace(".", "p")
        for suffix in ("png", "svg", "pdf"):
            p = OUT / "figures" / (stem + "." + suffix)
            metadata = {"Date": None} if suffix == "svg" else {"CreationDate": None, "ModDate": None} if suffix == "pdf" else {}
            fig.savefig(p, dpi=300, metadata=metadata)
            if suffix == "svg":
                # Matplotlib emits trailing spaces in path attributes. Preserve
                # the XML token separators while keeping the Git diff clean.
                p.write_text("\n".join(line.rstrip() for line in p.read_text().splitlines()) + "\n")
            figures.append(str(p.relative_to(OUT)))
        plt.close(fig)
    return figures, matplotlib.__version__


def check(table, records, provenance):
    assert (OUT / "data/comparisons.csv").read_bytes() == csv_bytes(table, COLUMNS)
    assert (OUT / "data/source_records.json").read_bytes() == json_bytes(records)
    assert (OUT / "data/input_sources.json").read_bytes() == json_bytes(provenance)
    points = points_from(table)
    assert (OUT / "data/plot_points.csv").read_bytes() == csv_bytes(points, list(points[0]))
    manifest = json.loads((OUT / "manifest.json").read_bytes())
    assert manifest["input_sources"] == provenance
    assert manifest["comparison_count"] == 26 and manifest["plotted_point_count"] == 52
    assert len(manifest["figure_files"]) == 15
    assert {Path(p).suffix for p in manifest["figure_files"]} == {".png", ".svg", ".pdf"}
    for path, token in manifest["output_files_sha256"].items():
        assert sha((OUT / path).read_bytes()) == token, path
    for path in manifest["figure_files"]:
        data = (OUT / path).read_bytes()
        if path.endswith(".png"):
            assert data[:8] == b"\x89PNG\r\n\x1a\n"
            assert struct.unpack(">II", data[16:24]) == (3600, 2100)
        elif path.endswith(".svg"):
            assert ET.fromstring(data).tag == "{http://www.w3.org/2000/svg}svg"
        else:
            assert data.startswith(b"%PDF-") and data.rstrip().endswith(b"%%EOF")
    # Check only this worktree against its base; no observations of other jobs.
    changed = subprocess.check_output(["git", "diff", "--name-only", "-z", BASE], cwd=ROOT).decode().split("\0")
    changed = [path for path in changed if path]
    allowed = {".gitignore", "PROJECT_MAP.md", "VALIDATION_STATUS.md", "artifacts/validation_manifest.json",
               "docs/research/研究概要・現状.md", "docs/README.md", "scripts/README.md"}
    assert set(changed) <= allowed | {
        "scripts/plot_pr_pf_absolute_cost_v2.py", "docs/research/track_a_pr_pf_absolute_cost_v2_20261019.md",
        "docs/research/研究ノート/2026-10-19_pr_pf_absolute_cost_v2.md"
    } | {str(p.relative_to(ROOT)) for p in OUT.rglob("*") if p.is_file()}
    return dict(status="PASS_SAVED_TOKEN_AND_STRUCTURE_CHECKS", comparisons=26, points=52,
                precisions=list(EPSILONS), figure_files=15, scientific_modules_imported=False,
                figure_formats="5 PNG 3600x2100; 5 parsed SVG; 5 PDF header/EOF checks",
                scientific_calls=0, new_shot_or_cost_accounting=False, new_ranking=False,
                original_sources_or_results_modified=False, check_author="Codex primary agent; not an independent reviewer")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify existing outputs against pinned saved records; no rendering")
    args = parser.parse_args()
    table, records, provenance = load_evidence()
    if args.check:
        print(json.dumps(check(table, records, provenance), ensure_ascii=False, indent=2))
        return
    for subdir in ("data", "figures"):
        (OUT / subdir).mkdir(parents=True, exist_ok=True)
    (OUT / "data/comparisons.csv").write_bytes(csv_bytes(table, COLUMNS))
    (OUT / "data/source_records.json").write_bytes(json_bytes(records))
    (OUT / "data/input_sources.json").write_bytes(json_bytes(provenance))
    points = points_from(table)
    (OUT / "data/plot_points.csv").write_bytes(csv_bytes(points, list(points[0])))
    write_display_tables(table)
    figures, version = render(table, points)
    manifest = dict(schema="PR_PF_ABSOLUTE_RZ_SAVED_VALUES_V2_1", status="DISPLAY_ONLY_NOT_SCIENTIFIC_ACCEPTANCE_OR_FREEZE",
                    base_commit=BASE, protocol=PROTOCOL, protocol_status_as_saved="DRAFT_V2_NOT_ADOPTED_NO_EXECUTION_AUTHORIZATION",
                    authorization="User selected v2 and all five precisions for existing-result plots; local commit only",
                    epsilons=list(EPSILONS), comparison_count=len(table), plotted_point_count=len(points),
                    independent_comparisons=6, exploratory_comparisons=20,
                    H2_exploratory_only=True, H2_added_to_existing_paper_main_table=False,
                    H4_legacy_DF12_included=False, scientific_calls=0, new_shot_or_cost_accounting=False,
                    new_ranking=False, new_statistics=False, numeric_conversion="Exact fraction to float for plotting and labels only",
                    error_bars="Copied saved paired SE; ±1 SE visualization, not CI or support-tail bound",
                    population_mean_certified=False, COMMON_CORE_READY=False, final_evidence_freeze=False,
                    original_Figure2_modified=False, push_authorized=False,
                    input_sources=provenance, figure_files=figures,
                    rendering=dict(python=sys.version.split()[0], matplotlib=version, backend="Agg", dpi=300,
                                   shared_environment_modified=False, environment=".plot-venv (local, ignored)"),
                    output_files_sha256={str(p.relative_to(OUT)): sha(p.read_bytes())
                                         for p in sorted(OUT.rglob("*")) if p.is_file()
                                         and p.name not in ("manifest.json", "verification.json")})
    (OUT / "manifest.json").write_bytes(json_bytes(manifest))
    verification = check(table, records, provenance)
    (OUT / "verification.json").write_bytes(json_bytes(verification))
    print(json.dumps(verification, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
