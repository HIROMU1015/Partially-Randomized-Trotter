#!/usr/bin/env python3
"""Display pinned S4-corrected totals/ratios; no science or accounting code.

Stdlib JSON/CSV/SHA helpers and rendering conventions reuse
c5ad7945:scripts/plot_pr_pf_absolute_cost_v2.py. Existing scientific
h46_cost_summary.py is deliberately not imported or executed.
"""
from __future__ import annotations

import argparse
import ast
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
BASE = "6e2b7eb6a0afc4ee2c5f57d3b7baf89b4ed237e1"
SAVED = "artifacts/resource_applicability/track_a_h46_s4_counter_native_compatibility/2026-10-18/"
OUT = ROOT / "artifacts/resource_applicability/track_a_h46_theory_compiled_total_plots/2026-10-19"
EPS = ("0.05", "0.01", "0.005", "0.001", "0.0005")
FACTORS = ("1.0", "10.0")
MODEL = "NATIVE_MIXED"
PANEL = {
    "H4": ("B1_p12_q1_2nd", "B1_p12_q1_4th", "B2_p3_q1_2nd_r4_K2", "B2_p3_q8_2nd_r2_K4"),
    "H6": ("B1_p19_q8_4th", "B2_p10_q8_2nd_r1_K2", "B2_p15_q8_2nd_r2_K4"),
}
SPECS = {name: SAVED + "stage3/" + name for name in (
    "shot_weighted_totals.json", "pf_pr_comparisons.json", "preserved_shots.json",
    "direction_audit.json", "terminal.json", "result_summary.json")}
SPECS["corrected_result_report"] = "docs/research/track_a_h46_s4_counter_native_compatibility_results_20261018.md"
SPECS["stage2_acceptance"] = SAVED + "stage2/acceptance.json"
SPECS["original_model_document"] = "docs/research/track_a_pr_theory_vs_compiled_completed_20261018.md"
ABS_FIELDS = ("system", "candidate_id", "method", "epsilon", "factor", "model", "metric", "status",
              "theory_G_lower", "theory_G_upper", "compiled_G_saved", "sample_count", "sample_phase",
              "shots_cosine_saved", "shots_sine_saved", "N_total_saved", "DF_rank", "DF_policy",
              "L_D", "q", "delta", "r", "K", "hamiltonian_hash", "state_hash", "input_sha256",
              "accounting_SOURCE", "counter_SOURCE", "result_commit", "source_path", "json_pointer", "shot_pointer")
RATIO_FIELDS = ("system", "PR", "PF", "epsilon", "factor", "model", "metric", "status",
                "theory_PR_PF_lower", "theory_PR_PF_upper", "compiled_PR_PF",
                "PR_sample_count", "PF_sample_count", "sample_phase", "saved_theory_direction",
                "saved_compiled_direction", "saved_theory_vs_compiled_disagree", "population_mean_certified",
                "accounting_SOURCE", "result_commit", "source_path", "json_pointer")


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


def coordinate(token):
    """Only representation conversion; no ratios, means, midpoints or totals."""
    if token is None:
        return None
    value = float(Fraction(token))
    if not math.isfinite(value) or value <= 0:
        raise ValueError("A saved coordinate must be finite and positive")
    return value


def load_saved():
    raw = {label: git_bytes(BASE, path) for label, path in SPECS.items()}
    sources = [dict(label=label, commit=BASE, path=SPECS[label], sha256=sha(data), bytes=len(data))
               for label, data in raw.items()]
    # Reject changed copies of these own inputs, not unrelated worktree activity.
    for label, path in SPECS.items():
        assert (ROOT / path).read_bytes() == raw[label], path
    terminal = json.loads(raw["terminal.json"])
    assert terminal["status"] == "COMPLETE_MANDATORY_STOP" and terminal["first_stop"] is None
    assert terminal["SOURCE"] == "da8eb64ba8e88c219a6b707e1c72d130f5f37fcb"
    acceptance = json.loads(raw["stage2_acceptance"])
    assert len(acceptance) == 8 and all(r["exact_integer_match"] for r in acceptance)
    summary = json.loads(raw["result_summary.json"])
    assert summary["pf_pr_comparison_rows"] == 360 and summary["direction_changes"] == 0
    totals = json.loads(raw["shot_weighted_totals.json"])
    ratios = json.loads(raw["pf_pr_comparisons.json"])
    assert len(totals) == 420 and len(ratios) == 360
    shots = json.loads(raw["preserved_shots.json"])
    shot_lookup = {(r["system"], r["candidate_id"], str(r["epsilon"]), r["factor"]): (i, r)
                   for i, r in enumerate(shots)}
    directions = {r["index"]: r for r in json.loads(raw["direction_audit.json"])}
    provenance = {}
    for r in ratios:
        for cid, p in r["input_and_compiled_provenance"].items():
            key = (r["system"], cid)
            if key in provenance:
                assert provenance[key] == p
            provenance[key] = p
    absolute_rows, ratio_rows, selected_records = [], [], []
    for index, saved in enumerate(totals):
        if saved["model"] != MODEL or saved["metric"] != "RZ":
            continue
        system, cid, epsilon, factor = saved["system"], saved["candidate_id"], str(saved["epsilon"]), saved["factor"]
        assert cid in PANEL[system] and epsilon in EPS and factor in FACTORS
        shot_index, shot = shot_lookup[(system, cid, epsilon, factor)]
        assert saved["status"] == shot["status"]
        good = saved["status"] == "ELIGIBLE_SAVED_EMPIRICAL_SHOTS"
        fields = ("theory_G_lower", "theory_G_upper", "compiled_G_saved")
        assert all(saved.get(k) is not None for k in fields) if good else all(saved.get(k) is None for k in fields)
        p = provenance[(system, cid)]
        assert p["mean_sample_size"] == shot["sample_count"]
        row = {k: saved.get(k) for k in ABS_FIELDS if k in saved}
        cell = re.fullmatch(r"B[12]_p(\d+)_q(\d+)_(2nd|4th)(?:_r(\d+)_K(\d+))?", cid)
        ld, q, formula, r, k = cell.groups()
        row.update(method="PR" if cid.startswith("B2") else "PF", epsilon=epsilon,
                   sample_count=shot["sample_count"], sample_phase=shot["sample_phase"],
                   shots_cosine_saved=shot["shots"]["cosine"] if good else None,
                   shots_sine_saved=shot["shots"]["sine"] if good else None, N_total_saved=shot["N_total"],
                   DF_rank=12 if system == "H4" else 19,
                   DF_policy="LEGACY_FIXED_DF12" if system == "H4" else "TOL_ONLY_1E-8",
                   L_D=ld, q=q, delta={"1": "0.8", "8": "0.1"}[q], r=r, K=k,
                   hamiltonian_hash=p["hamiltonian_hash"], state_hash=p["state_hash"], input_sha256=p["input_sha256"],
                   accounting_SOURCE=terminal["SOURCE"], result_commit=BASE,
                   source_path=SPECS["shot_weighted_totals.json"], json_pointer=f"/{index}", shot_pointer=f"/{shot_index}")
        # Copy the evaluation identity already attached to the saved pair.
        matches = [r for r in ratios if r["system"] == system and r["model"] == MODEL and cid in (r["PR"], r["PF"])]
        row["counter_SOURCE"] = matches[0]["SOURCE_counter_evaluation"][cid]
        assert all(r["SOURCE_counter_evaluation"][cid] == row["counter_SOURCE"] for r in matches)
        row = {k: row.get(k) for k in ABS_FIELDS}
        absolute_rows.append(row)
        selected_records.append(dict(kind="absolute", source_path=row["source_path"], json_pointer=f"/{index}", record=saved))
    for index, saved in enumerate(ratios):
        if saved["model"] != MODEL or saved["metric"] != "RZ":
            continue
        system, pr, pf = saved["system"], saved["PR"], saved["PF"]
        assert pr in PANEL[system] and pf in PANEL[system]
        good = saved["status"] == "CONDITIONAL_SAVED_COST_COMPARISON"
        fields = ("theory_PR_PF_lower", "theory_PR_PF_upper", "compiled_PR_PF")
        assert all(saved.get(k) is not None for k in fields) if good else all(saved.get(k) is None for k in fields)
        row = {k: saved.get(k) for k in RATIO_FIELDS if k in saved}
        d = directions.get(index)
        assert good == (d is not None)
        row.update(epsilon=str(saved["epsilon"]), PR_sample_count=provenance[(system, pr)]["mean_sample_size"],
                   PF_sample_count=provenance[(system, pf)]["mean_sample_size"],
                   sample_phase=provenance[(system, pr)]["sample_phase"],
                   saved_theory_direction=d["new_direction"] if d else None,
                   saved_compiled_direction=d["compiled_direction"] if d else None,
                   saved_theory_vs_compiled_disagree=d["theory_vs_compiled_disagree"] if d else None,
                   accounting_SOURCE=saved["SOURCE_accounting"], result_commit=BASE,
                   source_path=SPECS["pf_pr_comparisons.json"], json_pointer=f"/{index}")
        ratio_rows.append({k: row.get(k) for k in RATIO_FIELDS})
        selected_records.append(dict(kind="ratio", source_path=row["source_path"], json_pointer=f"/{index}", record=saved))
    absolute_rows.sort(key=lambda r: (r["system"], FACTORS.index(r["factor"]), PANEL[r["system"]].index(r["candidate_id"]), EPS.index(r["epsilon"])))
    ratio_rows.sort(key=lambda r: (r["system"], FACTORS.index(r["factor"]), r["PR"], r["PF"], EPS.index(r["epsilon"])))
    assert len(absolute_rows) == 70 and len(ratio_rows) == 60
    assert sum(r["status"] == "ELIGIBLE_SAVED_EMPIRICAL_SHOTS" for r in absolute_rows) == 50
    assert sum(r["status"] == "CONDITIONAL_SAVED_COST_COMPARISON" for r in ratio_rows) == 32
    for system, ids in PANEL.items():
        for factor in FACTORS:
            for cid in ids:
                assert {r["epsilon"] for r in absolute_rows if (r["system"], r["factor"], r["candidate_id"]) == (system, factor, cid)} == set(EPS)
    return absolute_rows, ratio_rows, selected_records, sources


def plot_tokens(absolute_rows, ratio_rows):
    rows = []
    for kind, records, fields in (("absolute", absolute_rows, ("theory_G_lower", "theory_G_upper", "compiled_G_saved")),
                                  ("ratio", ratio_rows, ("theory_PR_PF_lower", "theory_PR_PF_upper", "compiled_PR_PF"))):
        for r in records:
            for field in fields:
                token = r.get(field)
                value = coordinate(token)
                rows.append(dict(kind=kind, system=r["system"], candidate_id=r.get("candidate_id", ""),
                                 PR=r.get("PR", ""), PF=r.get("PF", ""), epsilon=r["epsilon"], factor=r["factor"],
                                 field=field, exact_token=token, plot_coordinate=repr(value) if value is not None else None,
                                 display_text=format(value, ".5g") if value is not None else None,
                                 status=r["status"], source_path=r["source_path"], json_pointer=r["json_pointer"]))
    assert len(rows) == 390 and sum(r["exact_token"] is not None for r in rows) == 246
    return rows


def display_table(absolute_rows, ratio_rows):
    lines = ["# 保存値の閲覧用表示", "", "小数はexact tokenから表示用に9桁へ変換。比・総費用・中点・統計量の新規計算なし。", ""]
    def show(token):
        value = coordinate(token)
        return "—" if value is None else format(value, ".9g")
    for system in PANEL:
        for factor in FACTORS:
            lines += [f"## {system} / allowance factor {factor}", "", "### 絶対RZ総費用", "",
                      "|候補|ε|theory lower|theory upper|compiled|状態|", "|---|---:|---:|---:|---:|---|"]
            for r in absolute_rows:
                if r["system"] == system and r["factor"] == factor:
                    lines.append(f"|{r['candidate_id']}|{r['epsilon']}|{show(r['theory_G_lower'])}|{show(r['theory_G_upper'])}|{show(r['compiled_G_saved'])}|{r['status']}|")
            lines += ["", "### PR/PF費用比（保存ratio）", "", "|PR|PF|ε|theory lower|theory upper|compiled|状態|", "|---|---|---:|---:|---:|---:|---|"]
            for r in ratio_rows:
                if r["system"] == system and r["factor"] == factor:
                    lines.append(f"|{r['PR']}|{r['PF']}|{r['epsilon']}|{show(r['theory_PR_PF_lower'])}|{show(r['theory_PR_PF_upper'])}|{show(r['compiled_PR_PF'])}|{r['status']}|")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render(absolute_rows, ratio_rows):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".plot-cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogLocator, LogFormatterSciNotation, NullFormatter

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "svg.hashsalt": "h46-s4-corrected-total",
                         "pdf.fonttype": 42, "savefig.facecolor": "white"})
    theory_color, compiled_color = "#6F4A9E", "#00838F"
    figure_files = []
    for system in PANEL:
        for factor in FACTORS:
            for kind, all_rows in (("absolute", absolute_rows), ("ratio", ratio_rows)):
                selected = [r for r in all_rows if r["system"] == system and r["factor"] == factor]
                if kind == "absolute":
                    groups = [(cid, [r for r in selected if r["candidate_id"] == cid]) for cid in PANEL[system]]
                else:
                    pairs = [(pr, pf) for pr in PANEL[system] if pr.startswith("B2") for pf in PANEL[system] if pf.startswith("B1")]
                    groups = [(pair, [r for r in selected if (r["PR"], r["PF"]) == pair]) for pair in pairs]
                ncols, nrows = (2, 2) if system == "H4" else (len(groups), 1)
                fig, axes = plt.subplots(nrows, ncols, figsize=(13.2, 9.2 if nrows == 2 else 6.5), squeeze=False, sharey=True)
                fig.subplots_adjust(left=.075, right=.982, top=.76 if nrows == 1 else .80,
                                    bottom=.26 if nrows == 1 else .19, hspace=.54, wspace=.16)
                factor_label = "1" if factor == "1.0" else "10"
                title = "Shot-inclusive RZ totals" if kind == "absolute" else "PR / PF shot-inclusive RZ cost ratio"
                fig.suptitle(f"{system}  |  {title}", fontsize=19, weight="bold", y=.982)
                sub = "legacy DF12 · PR anchor n=32" if system == "H4" else "tol-only DF19 · PR exploratory n=8"
                fig.text(.5, .922 if nrows == 2 else .906,
                         f"{sub}  |  NATIVE_MIXED  |  allowance factor {factor_label}  |  S4 corrected",
                         ha="center", fontsize=10.5, color="#444444")
                legend = [Line2D([], [], color=theory_color, marker="D", mfc="white", ms=6, ls="--", lw=1.5,
                                 label="Theory: saved lower / upper"),
                          Line2D([], [], color=compiled_color, marker="o", ms=6, lw=1.5,
                                 label="Compiled: saved finite-sample cost"),
                          Line2D([], [], color="#888888", marker="x", ms=6, ls="None",
                                 label="Ineligible / no common eligibility")]
                fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.53, .898 if nrows == 2 else .869), ncol=3, frameon=False, fontsize=10)
                fields = ("theory_G_lower", "theory_G_upper", "compiled_G_saved") if kind == "absolute" else ("theory_PR_PF_lower", "theory_PR_PF_upper", "compiled_PR_PF")
                values = [coordinate(r.get(k)) for r in selected for k in fields if r.get(k) is not None]
                if kind == "ratio":
                    values.append(1.0)
                low, high = min(values) / (7.0 if kind == "absolute" else 1.8), max(values) * (2.2 if kind == "absolute" else 1.8)
                for ax, (group, rows) in zip(axes.flat, groups):
                    rows = sorted(rows, key=lambda r: EPS.index(r["epsilon"]))
                    assert len(rows) == 5
                    lower = [coordinate(r.get(fields[0])) for r in rows]
                    upper = [coordinate(r.get(fields[1])) for r in rows]
                    compiled = [coordinate(r.get(fields[2])) for r in rows]
                    xs = list(range(5))
                    # NaN breaks lines across missing points; absent values get
                    # only a marker in axes coordinates, never a zero cost.
                    theory_x = [x - .045 for x in xs]
                    compiled_x = [x + .045 for x in xs]
                    ax.plot(theory_x, [math.nan if v is None else v for v in lower], color=theory_color, marker="D",
                            mfc="white", mec=theory_color, ms=5.5, ls="--", lw=1.5, zorder=3)
                    ax.plot(theory_x, [math.nan if v is None else v for v in upper], color=theory_color,
                            ls="--", lw=1.0, zorder=2)
                    ax.plot(compiled_x, [math.nan if v is None else v for v in compiled], color=compiled_color,
                            marker="o", ms=5.5, lw=1.5, zorder=4)
                    for x, lo, hi, co, r in zip(xs, lower, upper, compiled, rows):
                        if lo is None:
                            ax.axvspan(x - .16, x + .16, color="#F3F4F5", zorder=0)
                            ax.plot(x, .055, marker="x", color="#888888", ms=6, transform=ax.get_xaxis_transform(), clip_on=False)
                            continue
                        # Endpoints only; deliberately no interval midpoint.
                        ax.vlines(x - .045, lo, hi, color=theory_color, lw=2, zorder=3)
                        ax.plot(x - .045, hi, marker="_", color=theory_color, ms=6, zorder=3)
                        if kind == "absolute" or system == "H4" or x in (0, 4):
                            compiled_above = kind == "ratio" and co >= hi
                            label_offset = 11 if compiled_above else -10 if kind == "absolute" else -15
                            ax.annotate(format(co, ".3g"), (x + .045, co), xytext=(0, label_offset), textcoords="offset points",
                                        ha="center", va="bottom" if compiled_above else "top", fontsize=8, color=compiled_color,
                                        bbox=dict(fc="white", ec="none", pad=.3, alpha=.9))
                        if kind == "ratio" and (system == "H4" or x in (0, 4)):
                            theory_above = lo > co
                            ax.annotate(format(lo, ".3g"), (x - .045, lo), xytext=(0, 11 if theory_above else -15), textcoords="offset points",
                                        ha="center", va="bottom" if theory_above else "top", fontsize=8, color=theory_color,
                                        bbox=dict(fc="white", ec="none", pad=.3, alpha=.9))
                        if kind == "ratio" and r["saved_theory_vs_compiled_disagree"]:
                            ax.annotate("Saved direction differs", (x, 1), xytext=(38, 31), textcoords="offset points",
                                        fontsize=8.2, color="#A1351E", arrowprops=dict(arrowstyle="->", color="#A1351E", lw=.8))
                    ax.set_yscale("log")
                    ax.set_ylim(low, high)
                    ax.set_xlim(-.43, 4.43)
                    ax.set_xticks(xs, EPS)
                    ax.set_xlabel(r"Required signal accuracy $\epsilon$", labelpad=8)
                    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=7))
                    ax.yaxis.set_major_formatter(LogFormatterSciNotation())
                    ax.yaxis.set_minor_formatter(NullFormatter())
                    ax.grid(axis="y", which="major", color="#DEE3E8", lw=.8, zorder=0)
                    ax.tick_params(axis="x", labelsize=9, pad=6)
                    if kind == "absolute":
                        method = "PR" if group.startswith("B2") else "PF"
                        count = 32 if system == "H4" and method == "PR" else 8 if method == "PR" else 1
                        ax.set_title(f"{method}  |  {group}\nSaved compiled sample n = {count}", fontsize=10.5,
                                     weight="bold", color="#0072B2" if method == "PR" else "#B34D00", pad=11)
                    else:
                        pr, pf = group
                        ax.set_title(f"PR  {pr}\nPF  {pf}", fontsize=10.3, weight="bold", pad=11)
                        ax.axhline(1, color="#666666", lw=1.0, ls=":", zorder=1)
                for row in range(nrows):
                    axes[row, 0].set_ylabel(r"Total RZ cost $G_{\mathrm{RZ}}$ (log)" if kind == "absolute" else r"$G_{\mathrm{PR}}/G_{\mathrm{PF}}$ (log)", labelpad=11)
                footer = "Linear 1 Å / STO-3G; T=0.8; P=0; cos+sin full Hadamard wrapper; saved empirical shots for each candidate.\n"
                footer += "Theory is an expected cost under the declared model; compiled PR is a finite-sample mean. No CI or population guarantee.\n"
                footer += "Lower/upper are stored endpoints, not a confidence interval. Gaps are saved ineligibility; no values are filled.\n"
                footer += "RZ only; excludes state preparation and FTQC. "
                footer += "Ratio < 1 favours PR; > 1 favours PF." if kind == "ratio" else "Fixed candidates; no new ranking or cost accounting."
                fig.text(.075, .028, footer, fontsize=8.4, va="bottom", linespacing=1.45, color="#444444")
                stem = f"{system.lower()}_{kind}_rz_factor_{factor_label}"
                for suffix in ("png", "svg", "pdf"):
                    path = OUT / "figures" / f"{stem}.{suffix}"
                    metadata = {"Date": None} if suffix == "svg" else {"CreationDate": None, "ModDate": None} if suffix == "pdf" else {}
                    fig.savefig(path, dpi=250, metadata=metadata)
                    if suffix == "svg":
                        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
                    figure_files.append(str(path.relative_to(OUT)))
                plt.close(fig)
    return figure_files, matplotlib.__version__


def check(absolute_rows, ratio_rows, records, sources):
    tokens = plot_tokens(absolute_rows, ratio_rows)
    assert (OUT / "data/absolute_totals.csv").read_bytes() == csv_bytes(absolute_rows, ABS_FIELDS)
    assert (OUT / "data/pf_pr_ratios.csv").read_bytes() == csv_bytes(ratio_rows, RATIO_FIELDS)
    assert (OUT / "data/source_records.json").read_bytes() == json_bytes(records)
    assert (OUT / "data/input_sources.json").read_bytes() == json_bytes(sources)
    assert (OUT / "data/plot_coordinates.csv").read_bytes() == csv_bytes(tokens, list(tokens[0]))
    assert (OUT / "data/display_tables.md").read_text() == display_table(absolute_rows, ratio_rows)
    manifest = json.loads((OUT / "manifest.json").read_text())
    assert manifest["input_sources"] == sources and len(manifest["figure_files"]) == 24
    for path, receipt in manifest["output_sha256"].items():
        assert sha((OUT / path).read_bytes()) == receipt, path
    for name in manifest["figure_files"]:
        data = (OUT / name).read_bytes()
        if name.endswith(".png"):
            assert data.startswith(b"\x89PNG\r\n\x1a\n")
            width, height = struct.unpack(">II", data[16:24])
            assert width == 3300 and height in (2300, 1625)
        elif name.endswith(".svg"):
            assert ET.fromstring(data).tag == "{http://www.w3.org/2000/svg}svg"
        else:
            assert data.startswith(b"%PDF-") and data.rstrip().endswith(b"%%EOF")
    tree = ast.parse(Path(__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(n.name.split(".")[0] not in {"trotterlib", "scipy", "qiskit"} for n in node.names)
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in {"trotterlib", "scipy", "qiskit"}
        assert not (isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Div, ast.Mult))
                    and any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "Fraction" for n in ast.walk(node)))
    report = ROOT / "docs/research/track_a_h46_theory_compiled_total_plots_20261019.md"
    note = ROOT / "docs/research/研究ノート/2026-10-19_h46_theory_compiled_total_plots.md"
    link_count = 0
    for p in [report, note, *OUT.rglob("*.md")]:
        for link in re.findall(r"\]\(([^)]+)\)", p.read_text()):
            if "://" not in link and not link.startswith("#"):
                assert (p.parent / link.split("#")[0]).exists(), (p, link)
                link_count += 1
    changed = subprocess.check_output(["git", "diff", "--name-only", "-z", BASE], cwd=ROOT).decode().split("\0")
    changed = {r for r in changed if r}
    allowed = {".gitignore", "PROJECT_MAP.md", "VALIDATION_STATUS.md", "artifacts/validation_manifest.json",
               "docs/README.md", "docs/research/研究概要・現状.md", "scripts/README.md", "src/trotterlib/README.md",
               str(Path(__file__).relative_to(ROOT)), str(report.relative_to(ROOT)), str(note.relative_to(ROOT))}
    allowed |= {str(p.relative_to(ROOT)) for p in OUT.rglob("*") if p.is_file()}
    assert changed <= allowed, changed - allowed
    for name in ("PROJECT_MAP.md", "VALIDATION_STATUS.md", "docs/README.md", "docs/research/研究概要・現状.md", "scripts/README.md", "src/trotterlib/README.md"):
        assert (ROOT / name).read_bytes().endswith(git_bytes(BASE, name)), name
    original = json.loads(git_bytes(BASE, "artifacts/validation_manifest.json"))
    current = json.loads((ROOT / "artifacts/validation_manifest.json").read_text())
    current.pop("track_a_h46_theory_compiled_total_plots_20261019")
    assert original == current
    return dict(status="PASS_DISPLAY_TOKEN_STRUCTURE_AND_PRESERVATION_CHECKS", absolute_conditions=70,
                absolute_eligible=50, absolute_ineligible=20, ratio_conditions=60, ratio_eligible=32,
                ratio_no_common_eligibility=28, coordinate_slots=390, nonmissing_coordinates=246,
                png=8, svg=8, pdf=8, local_links=link_count, scientific_modules_imported=False,
                new_sampling=0, new_compile=0, new_bootstrap=0, new_shot_cost_ratio_or_statistical_analysis=0,
                interval_midpoints_computed=False, saved_inputs_and_old_artifacts_unchanged=True,
                index_history_and_existing_manifest_entries_preserved=True,
                reviewer="Codex primary agent technical checks; no independent science review")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Read-only verification; stdlib only")
    args = parser.parse_args()
    absolute_rows, ratio_rows, records, sources = load_saved()
    if args.check:
        print(json.dumps(check(absolute_rows, ratio_rows, records, sources), ensure_ascii=False, indent=2))
        return
    for name in ("data", "figures"):
        (OUT / name).mkdir(parents=True, exist_ok=True)
    (OUT / "data/absolute_totals.csv").write_bytes(csv_bytes(absolute_rows, ABS_FIELDS))
    (OUT / "data/pf_pr_ratios.csv").write_bytes(csv_bytes(ratio_rows, RATIO_FIELDS))
    (OUT / "data/source_records.json").write_bytes(json_bytes(records))
    (OUT / "data/input_sources.json").write_bytes(json_bytes(sources))
    tokens = plot_tokens(absolute_rows, ratio_rows)
    (OUT / "data/plot_coordinates.csv").write_bytes(csv_bytes(tokens, list(tokens[0])))
    (OUT / "data/display_tables.md").write_text(display_table(absolute_rows, ratio_rows))
    figures, mpl = render(absolute_rows, ratio_rows)
    manifest = dict(status="SAVED_DATA_PLOTS_ONLY_NOT_EVIDENCE_FREEZE", result_commit=BASE,
                    model=MODEL, metric="RZ", epsilons=list(EPS), allowance_factors=list(FACTORS),
                    input_sources=sources, figure_files=figures, candidate_panel=PANEL,
                    counts=dict(absolute=70, absolute_eligible=50, ratios=60, ratio_eligible=32),
                    authorization="User requested existing-data plots, dedicated commit and push of this scope",
                    no_new_scientific_analysis=True, intervals="Saved lower and upper; no midpoint or CI",
                    saved_ratio_tokens_used_directly=True, new_shot_or_cost_products=False,
                    compiled_PR_estimand="Saved finite-sample mean, H4 anchor n32 / H6 exploration n8",
                    theory_PR_estimand="Expectation in declared NATIVE_MIXED model",
                    state_preparation_included=False, FTQC_included=False,
                    original_Figure2_or_main_tables_modified=False, detailed_science_review=False,
                    final_evidence_freeze=False, old_STOPs_preserved=True,
                    rendering=dict(python=sys.version.split()[0], matplotlib=mpl, backend="Agg", dpi=250,
                                   environment="Reused prior project-only .plot-venv; local .plot-cache",
                                   reused_helpers_commit="c5ad7945", shared_environment_changed=False),
                    output_sha256={str(p.relative_to(OUT)): sha(p.read_bytes()) for p in sorted(OUT.rglob("*"))
                                   if p.is_file() and p.name not in {"manifest.json", "verification.json"}})
    (OUT / "manifest.json").write_bytes(json_bytes(manifest))
    (OUT / "verification.json").write_bytes(json_bytes({"status": "CHECK_IN_PROGRESS"}))
    result = check(absolute_rows, ratio_rows, records, sources)
    (OUT / "verification.json").write_bytes(json_bytes(result))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
