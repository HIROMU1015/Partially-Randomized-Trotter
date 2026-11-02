"""Export manuscript figures from an explicit, immutable saved-value allowlist.

No science module, molecular snapshot, runtime, compiler, or sampler is used.
Selection/formatting is for display, not a new precision analysis or winner test.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import platform
import re
import subprocess
from collections import defaultdict
from pathlib import Path

EVIDENCE_COMMIT = "5a1adffad780f0ec4272f5e8bb94713f9ff0f2bc"
DESIGN_COMMIT = "d45c4006d440ba053517d6f646844a61b18dfd15"
PM2 = "artifacts/resource_applicability/pr2_pm2_precision_analysis/2026-10-05/"
PM0 = "artifacts/resource_applicability/pr2_post_m2_evidence_attribution/2026-10-04/"
INPUTS = {
    PM2 + "precision_ledger.csv": "ed4ff1f9192f18b9dd3cd700b84a49d525bbd7e78f464c85ea57559702a51749",
    PM2 + "representative_decomposition.csv": "8900b769b6a3211833ffd32c1e7e8d1be057c852c2e98a7f8bd3453fe5f60839",
    PM2 + "eligibility_boundaries.csv": "d6b45e88fa60238091d3c514e9aada98a907a1b4a07ed306e6e0411a78826882",
    PM2 + "P_envelope.csv": "3eb0791687c1fbe7c56bbfde99fba4256071aac126ad0ef90ae415b6831c1a8c",
    PM2 + "summary.json": "7b026d4cc657cf43ad23fd7d6e10d5aa31cd58af555649b7a8e858ea12155845",
    PM2 + "manifest.json": "546cdfaf8c77f349f6f55b346e93749ce5956c9a605281d169887257377c843f",
    PM2 + "claim_audit.json": "3ca528aa8061bcf45a63ffeb58f41e722b097fa05b27370889859eb44ba7fecd",
    PM0 + "same_R_candidate_comparison.csv": "0e202cc7c913d06f4124b671780eba8245f33f4343739f87bc1f1f33cb9d65e5",
    PM0 + "summary.json": "182d525116a10cda335f696a89d58204f989b3075fd06a3c303677d9f861a313",
}
FIG1_IDS = (
    "PM1-B0-rank5-q1-r0-K0", "B1-rank12-q1-r0-K0",
    "B2-rank3-q1-r4-K2", "B3-rank0-q8-r32-K4",
)
TRANSFER_IDS = (
    "B2-rank3-q1-r4-K2", "B2-rank3-q1-r8-K2", "B0-rank6-q1-r0-K0",
    "B1-rank12-q1-r0-K0", "B3-rank0-q8-r32-K4",
)
COLORS = {"B0": "#777777", "B1": "#d55e00", "B2": "#0072b2", "B3": "#009e73"}
DEFAULT_OUTPUT = "artifacts/resource_applicability/track_a_manuscript_v0_1/2026-10-05"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def method(candidate):
    return re.search(r"B[0-3]", candidate).group()


def eligible(row):
    if row["accuracy_eligible"] not in ("True", "False"):
        raise ValueError("Invalid eligibility flag")
    return row["accuracy_eligible"] == "True"


def value(row, field):
    if not eligible(row):
        return math.nan  # MISSING is never zero, and plots must break here.
    result = float(row[field])
    if not math.isfinite(result):
        raise ValueError("Eligible row has missing/nonfinite value")
    return result


def csv_rows(data):
    return list(csv.DictReader(io.StringIO(data.decode("utf-8"))))


def verify_inputs(root):
    data, audit = {}, []
    for relative, expected in INPUTS.items():
        raw = (root / relative).read_bytes()
        blob = subprocess.check_output(["git", "show", EVIDENCE_COMMIT + ":" + relative], cwd=root)
        if sha(raw) != expected or raw != blob:
            raise ValueError("Frozen input identity mismatch: " + relative)
        data[relative] = raw
        audit.append({"path": relative, "bytes": len(raw), "sha256": expected, "commit_blob_identical": True})
    summary = json.loads(data[PM2 + "summary.json"])
    claims = json.loads(data[PM2 + "claim_audit.json"])
    if summary["status"] != "PM2_PRECISION_RESOURCE_MAP_COMPLETE_AWAITING_REVIEW":
        raise ValueError("Not a complete saved precision map")
    if claims["formal_CI"] or claims["next_stage_authorized"] or not claims["M2_only_original_five"]:
        raise ValueError("Saved scope gate failed")
    return data, audit


def display_selection(data):
    ledger = csv_rows(data[PM2 + "precision_ledger.csv"])
    domains = defaultdict(lambda: defaultdict(list))
    for row in ledger:
        domains[row["dataset"]][float(row["epsilon"])].append(row)
    if set(domains) != {"development", "transfer_fixed_five"} or len(ledger) != 67346:
        raise ValueError("Unexpected saved domain")
    for domain, expected in (("development", 218), ("transfer_fixed_five", 5)):
        if len(domains[domain]) != 302 or set(domains[domain]) != set(domains["development"]):
            raise ValueError("Unexpected frozen precision grid")
        for rows in domains[domain].values():
            ids = {r["candidate_id"] for r in rows}
            if len(rows) != expected or len(ids) != expected:
                raise ValueError("Missing/duplicate candidate")
            if domain == "transfer_fixed_five" and ids != set(TRANSFER_IDS):
                raise ValueError("Changed transfer set")
    original = {r["candidate_id"]: r for r in domains["development"][.05]}
    representatives = {(r["dataset"], float(r["epsilon"]), r["candidate_id"]): r
                       for r in csv_rows(data[PM2 + "representative_decomposition.csv"])}
    fig1 = []
    for candidate in FIG1_IDS:
        row = dict(original[candidate])
        rep = representatives[("development", .05, candidate)]
        row.update({k: rep[k] for k in ("mean_RZ_cosine", "mean_RZ_sine")})
        work = int(row["N_real"]) * float(row["mean_RZ_cosine"]) + int(row["N_imag"]) * float(row["mean_RZ_sine"])
        if not math.isclose(work, value(row, "primary_RZ_P0"), rel_tol=1e-12):
            raise ValueError("Figure 1 axis/work mismatch")
        fig1.append(row)
    minima, global_min = [], []
    for epsilon, rows in sorted(domains["development"].items()):
        valid = [r for r in rows if eligible(r)]
        global_min.append(min(valid, key=lambda r: (value(r, "primary_RZ_P0"), r["candidate_id"])))
        for family in COLORS:
            candidates = [r for r in valid if method(r["candidate_id"]) == family]
            if candidates:
                minima.append(dict(min(candidates, key=lambda r: (value(r, "primary_RZ_P0"), r["candidate_id"])), method=family))
            else:
                minima.append({"dataset": "development", "epsilon": str(epsilon), "method": family,
                               "candidate_id": "", "accuracy_eligible": "False", "primary_RZ_P0": "MISSING", "primary_SE": "MISSING"})
    fig4 = [r for r in csv_rows(data[PM0 + "same_R_candidate_comparison.csv"])
            if r["stage"] == "M1" and r["method"] == "B2" and r["rank"] == "3"
            and r["K"] == "2" and r["R"] == "8" and float(r["T"]) == .8]
    fig4.sort(key=lambda r: int(r["q"]))
    if [int(r["q"]) for r in fig4] != [1, 2, 4, 8]:
        raise ValueError("Changed fixed same-R group")
    for field in ("tau", "normalization", "expected_random_applications"):
        if not all(math.isclose(float(r[field]), float(fig4[0][field]), rel_tol=1e-12) for r in fig4):
            raise ValueError("Same-R invariant failed")
    return domains, fig1, minima, global_min, fig4


def export_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def affine_segment_display(row, intercept, slope, points=101):
    """Evaluate a saved affine segment for rendering; do not find a new envelope."""
    lower = max(1, float(row["P_min"]))
    upper = min(1e7, float(row["P_max"])) if row["P_max"] not in (None, "MISSING", "") else 1e7
    if lower > upper:
        return [], []
    xs = [math.exp(math.log(lower) + (math.log(upper)-math.log(lower))*i/(points-1)) for i in range(points)]
    return xs, [float(row[intercept]) + float(row[slope])*p for p in xs]


def draw(out, data, domains, fig1, minima, global_min, fig4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap
    from matplotlib.ticker import NullLocator

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.hashsalt": "track-a-manuscript-v0.1"})

    def save(fig, stem, footer):
        fig.get_layout_engine().set(rect=(0, .09, 1, .91))
        fig.text(.01, .01, footer, fontsize=8, color="#444444")
        for extension in ("png", "svg", "pdf"):
            metadata = {"CreationDate": None, "ModDate": None} if extension == "pdf" else {"Date": None} if extension == "svg" else {}
            fig.savefig(out / (stem + "." + extension), dpi=180, bbox_inches="tight", metadata=metadata)
        plt.close(fig)

    labels = ["B0\nL=5\nq=1", "B1\nL=12\nq=1", "B2\nL=3, q=1\nr=4, K=2", "B3\nL=0, q=8\nr=32, K=4"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8), layout="constrained")
    x = np.arange(4)
    for ax, keys, title, unit in (
        (axes[0], ("N_real", "N_imag"), "a  Analytic shot burden", "Shots"),
        (axes[1], ("mean_RZ_cosine", "mean_RZ_sine"), "b  One-shot compiled cost", "RZ / wrapper"),
    ):
        for shift, key, axis, hatch in ((-.19, keys[0], "Cosine / real", ""), (.19, keys[1], "Sine / imag", "//")):
            ax.bar(x + shift, [float(r[key]) for r in fig1], width=.36,
                   color=[COLORS[method(r["candidate_id"])] for r in fig1], hatch=hatch, label=axis)
        ax.set_title(title, loc="left"); ax.set_ylabel(unit); ax.set_xticks(x, labels)
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=.2)
    axes[2].bar(x, [value(r, "primary_RZ_P0") / 1e8 for r in fig1],
                color=[COLORS[method(r["candidate_id"])] for r in fig1])
    for i, row in enumerate(fig1):
        if method(row["candidate_id"]) in ("B2", "B3"):
            axes[2].errorbar(i, value(row, "primary_RZ_P0") / 1e8, yerr=2 * value(row, "primary_SE") / 1e8,
                             color="black", capsize=4, fmt="none")
    axes[2].set_xticks(x, labels); axes[2].set_title("c  Shot-weighted work", loc="left")
    axes[2].set_ylabel(r"$G_{RZ}$ / $10^8$"); axes[2].grid(axis="y", alpha=.2)
    save(fig, "figure_1_development_cost_components", "H4 1.00 A | STO-3G | DF rank 12 | T=0.8 | epsilon=0.05 | P=0 | random bars: engineering +/-2SE")

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), height_ratios=(3, 1.3), layout="constrained", sharex=True)
    for family in COLORS:
        rows = [r for r in minima if r["method"] == family]
        eps = np.array([float(r["epsilon"]) for r in rows])
        work = np.array([value(r, "primary_RZ_P0") for r in rows])
        se = np.array([value(r, "primary_SE") for r in rows])
        axes[0].plot(eps, work, color=COLORS[family], label=family)
        if family in ("B2", "B3"):
            axes[0].fill_between(eps, work - 2*se, work + 2*se, color=COLORS[family], alpha=.15)
    axes[0].set(xscale="log", yscale="log", ylabel=r"$G_{RZ}(P=0)$", title="a  Methodwise eligible point minima; development 218")
    axes[0].legend(ncol=4); axes[0].grid(alpha=.2)
    eps = [float(r["epsilon"]) for r in global_min]
    q = [int(re.search(r"-q(\d+)-", r["candidate_id"])[1]) for r in global_min]
    axes[1].scatter(eps, q, color=COLORS["B2"], s=12)
    axes[1].set(yticks=[1, 2, 4], ylabel="Point-minimum q", xlabel=r"Required complex-signal accuracy $\epsilon$",
                title="b  Configuration changes within B2, not a method switch", ylim=(.6, 4.6))
    for left, right in zip(global_min, global_min[1:]):
        if left["candidate_id"] != right["candidate_id"]:
            for ax in axes:
                ax.axvspan(float(left["epsilon"]), float(right["epsilon"]), color="#777777", alpha=.2)
    for pos, height, label in ((.0052,4,"r=4, K=4"),(.011,2,"r=4, K=4"),(.05,1,"r=4, K=2")):
        axes[1].text(pos, height+.27, label, ha="left" if height==4 else "center", fontsize=9)
    for ax in axes:
        ax.axvline(.05, color="#333333", linestyle=":"); ax.set_xlim(.005, .1)
    save(fig, "figure_2_development_precision", "Saved 302-point sensitivity | lines are visual guides | shaded cost: conditional engineering +/-2SE, not simultaneous CI")

    fig, axes = plt.subplots(2, 1, figsize=(9, 6.8), height_ratios=(1.4, 3), layout="constrained", sharex=True)
    grid = sorted(domains["transfer_fixed_five"])
    lookup = {(float(r["epsilon"]),r["candidate_id"]):r for rows in domains["transfer_fixed_five"].values() for r in rows}
    edges = np.exp(np.r_[math.log(grid[0]), (np.log(grid[:-1])+np.log(grid[1:]))/2, math.log(grid[-1])])
    flags = [[int(eligible(lookup[(e,c)])) for e in grid] for c in TRANSFER_IDS]
    axes[0].pcolormesh(edges, np.arange(6)-.5, flags, cmap=ListedColormap(["#dddddd", "#436c9c"]), vmin=0, vmax=1, shading="flat")
    short = ["B2 L3 q1 r4 K2", "B2 L3 q1 r8 K2", "B0 L6 q1", "B1 L12 q1", "B3 L0 q8 r32 K4"]
    axes[0].set(yticks=range(5), yticklabels=short, title="a  Fixed-five eligibility: blue eligible, gray ineligible")
    axes[0].invert_yaxis()
    for c, label in zip(TRANSFER_IDS, short):
        rows = [lookup[(e,c)] for e in grid]
        color = COLORS[method(c)]
        axes[1].plot(grid, [value(r,"primary_RZ_P0") for r in rows], color=color,
                     linestyle="--" if "-r8-" in c else "-", label=label)
    boundary = float(lookup[(.05,TRANSFER_IDS[0])]["epsilon_min"])
    axes[1].axvline(boundary, color="#777777", linestyle="--", linewidth=1)
    axes[1].axvspan(.006679381313664138, .006746414238367818, color="#777777", alpha=.25)
    axes[1].text(boundary*1.06, 1.3e14, "B2 r4 eligibility boundary", rotation=90, fontsize=8, va="top")
    axes[1].set(xscale="log", yscale="log", ylabel=r"$G_{RZ}(P=0)$", xlabel=r"Required complex-signal accuracy $\epsilon$",
                title="b  Posthoc cost sensitivity; dotted line: original transfer accuracy")
    axes[1].legend(fontsize=8, loc="upper right"); axes[1].grid(alpha=.2)
    for ax in axes:
        ax.set_xscale("log"); ax.axvline(.05, color="#333333", linestyle=":"); ax.set_xlim(.005,.1)
    save(fig, "figure_3_frozen_transfer_precision", "H4 1.30 A | fixed five / posthoc / symmetric-axis Hoeffding rule | no held-out reoptimization | gaps are ineligible")

    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), layout="constrained")
    qs = [int(r["q"]) for r in fig4]
    for ax, key, title, ylabel, scale in (
        (axes[0,0],"total_bias_abs","a  Total complex bias","Absolute signal bias",1),
        (axes[0,1],"total_shots","b  Analytic shot burden","Shots",1),
        (axes[1,0],"effective_rz_cost","c  Shot-weighted one-shot cost","Effective RZ / wrapper",1),
        (axes[1,1],"rz_count","d  Total work",r"$G_{RZ}$ / $10^8$",1e8),
    ):
        ax.plot(qs,[float(r[key])/scale for r in fig4],"o-",color=COLORS["B2"])
        ax.set(xscale="log", xticks=qs, xticklabels=["1 / 8","2 / 4","4 / 2","8 / 1"],
               xlabel="q / r (R=qr=8)",ylabel=ylabel,title=title)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.grid(alpha=.2)
    axes[0,0].set_yscale("log")
    save(fig,"figure_4_same_R_competition","H4 1.00 A | DF rank 12 | B2 L_D=3, K=2, T=0.8, epsilon=0.05 | same tau, normalization, expected random actions")

    # Secondary preparation sensitivity: plot the existing linear segments only.
    p_rows = [r for r in csv_rows(data[PM2+"P_envelope.csv"]) if r["dataset"]=="development" and float(r["epsilon"])==.05]
    pm0 = json.loads(data[PM0+"summary.json"])
    panels = [("Development 218", p_rows, "G_RZ_P0", "N_total"),
              ("Development common five", pm0["state_preparation_point_lower_envelopes"]["M1_common5"], "rz_intercept", "shot_slope"),
              ("Transfer common five", pm0["state_preparation_point_lower_envelopes"]["M2_common5"], "rz_intercept", "shot_slope")]
    fig, axes = plt.subplots(1,3,figsize=(12,4.2),layout="constrained",sharey=True)
    for ax,(title,rows,intercept,slope) in zip(axes,panels):
        seen=set()
        for row in rows:
            ps, work = affine_segment_display(row, intercept, slope)
            if not ps: continue
            family=method(row["candidate_id"])
            ax.plot(ps, work,
                    color=COLORS[family],label=family if family not in seen else None)
            seen.add(family)
        ax.set(xscale="log",yscale="log",title=title,xlabel="Hypothetical P (RZ-equivalent / shot)",xlim=(1,1e7))
        ax.legend(); ax.grid(alpha=.2)
    axes[0].set_ylabel(r"Point lower envelope $G_{RZ}(P)$")
    save(fig,"figure_S1_preparation_sensitivity","epsilon=0.05 | secondary hypothetical common-P model | candidate-domain differences are not geometry-only effects")
    return matplotlib.__version__


def build(root, output):
    data, before = verify_inputs(root)
    domains, fig1, minima, global_min, fig4 = display_selection(data)
    # Identity/selection failures leave no output. Never overwrite a prior bundle.
    output.mkdir(parents=True, exist_ok=False)
    export_csv(output/"figure_1_values.csv",fig1)
    export_csv(output/"figure_2_method_minima.csv",minima)
    export_csv(output/"figure_2_point_minimum_settings.csv",global_min)
    export_csv(output/"figure_3_fixed_five.csv",[r for e in sorted(domains["transfer_fixed_five"]) for r in domains["transfer_fixed_five"][e]])
    export_csv(output/"figure_4_values.csv",fig4)
    export_csv(output/"original_precision_223_candidates.csv",domains["development"][.05]+domains["transfer_fixed_five"][.05])
    version = draw(output,data,domains,fig1,minima,global_min,fig4)
    _, after = verify_inputs(root)
    if before != after:
        raise ValueError("Input changed during rendering")
    audit={"status":"MANUSCRIPT_DISPLAY_EXPORT_COMPLETE_NOT_NEW_SCIENCE", "evidence_commit":EVIDENCE_COMMIT,
           "design_commit":DESIGN_COMMIT, "input_identity":before,"input_identity_unchanged_after":True,
           "python":platform.python_version(),"matplotlib":version,"builder_sha256":sha(Path(__file__).read_bytes()),
           "development_candidates":218,"transfer_candidates":5,"saved_precision_points":302,
           "new_signal":0,"new_trajectory":0,"new_compile":0,"molecular_data_access":0,"GPU_operations":0,
           "scope_note":"Counts describe this display-only program, not OS-sandbox certification or a scientific rerun.",
           "interpolation":"Visual guides only; no new precision points or crossover search",
           "preparation_display":"Saved affine segments evaluated for plotting only; no new envelope or crossover search",
           "uncertainty":"Conditional engineering +/-2SE, not formal or simultaneous CI", "scientific_next_stage_authorized":False}
    (output/"display_audit.json").write_text(json.dumps(audit,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    files=[{"path":p.name,"bytes":p.stat().st_size,"sha256":sha(p.read_bytes())} for p in sorted(output.iterdir()) if p.is_file()]
    (output/"manifest.json").write_text(json.dumps({"kind":"MANUSCRIPT_ASSET_MANIFEST_NOT_SCIENCE_MANIFEST","files":files},indent=2)+"\n",encoding="utf-8")
    return {"output":str(output),"files":len(files),"main_figures":4,"supplementary_figures":1,"new_science":0}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root",type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir",type=Path,default=Path(DEFAULT_OUTPUT))
    args=parser.parse_args()
    print(json.dumps(build(args.project_root,args.output_dir if args.output_dir.is_absolute() else args.project_root/args.output_dir)))
