"""Render three published scalar values; no scientific modules or raw analysis.

Only two fixed-commit JSON records are read via git show. No fitting, circuit
construction, sampling, statistical estimation, pooling, or rate calculation.
See REPORT.md for candidate identities, qualifications, and source references.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from decimal import Decimal
import json
from pathlib import Path
import subprocess


CDF0_COMMIT = "cbc517e75ad47054e4246748351f56269dac3a0c"
PE_COMMIT = "3015d4fb342d4d98b58e41d57553e75539a07655"
CDF0_PATH = "artifacts/research_c_cdf0_relocated/2026-11-08/science/run02/CDF0.json"
PE_PATH = "artifacts/research_c_h4_handoff/2026-11-06/science/run01/comparison.json"
NOTICE = (
    "保存済みの異なる実験から取得した点値の記述的比較。"
    "統一した最適化条件での新しい三者実験ではない。"
)


@dataclass(frozen=True)
class Point:
    label: str
    detail: str
    r: int
    total: Decimal


def read_record(repo: Path, commit: str, path: str) -> dict:
    raw = subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=repo)
    return json.loads(raw, parse_float=Decimal, parse_int=Decimal)


def load_points(repo: Path) -> tuple[Point, ...]:
    cdf0 = read_record(repo, CDF0_COMMIT, CDF0_PATH)
    pe = read_record(repo, PE_COMMIT, PE_PATH)
    primary = pe["comparisons"]["B_fair_retuned"]
    if cdf0["candidate_id"] != "CDF0_STAGE_A_X0_V2_GAMMA0_FINAL_R8":
        raise ValueError("Unexpected CDF0 candidate")
    if cdf0["selection"]["selected_r"] != 64:
        raise ValueError("Unexpected saved CDF0 r")
    if any(pe["selection"]["tuned_r"][arm] != 16 for arm in ("P", "E2")):
        raise ValueError("Unexpected saved P/E2 r")
    values = (
        cdf0["comparison"]["total_points"]["PR"],  # Legacy PR key means CDF0.
        primary["total_points"]["PR"],
        primary["total_points"]["ENTRY"],
    )
    expected = tuple(map(Decimal, ("231951425282.5", "87514417699", "75621395471")))
    if values != expected or cdf0["comparison"]["total_points"]["ENTRY"] != values[2]:
        raise ValueError("Published points or reused E2 do not match")
    if cdf0["comparison"]["decision"] != "UNRESOLVED_PRECISION_TARGET":
        raise ValueError("CDF0 precision qualification changed")
    if primary["decision"] != "NOMINAL_MODEL_ENTRY_LOWER":
        raise ValueError("Registered comparison B qualification changed")
    return (
        Point("通常CDF0", "Stage A / γ = 0", 64, values[0]),
        Point("PR-aware DF", "C0 / strength 0.1", 16, values[1]),
        Point("ENTRY DF", "保存E2 / strength 0.03", 16, values[2]),
    )


def render(points: tuple[Point, ...], output: Path, font: Path | None = None):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager, pyplot as plt
    from matplotlib.ticker import MaxNLocator

    if font is not None:
        font_manager.fontManager.addfont(str(font))
        family = font_manager.FontProperties(fname=font).get_name()
    else:
        family = "Noto Sans CJK JP"
        font_manager.findfont(family, fallback_to_default=False)
    plt.rcParams.update({
        "font.family": family, "font.size": 11, "axes.unicode_minus": False,
        "svg.fonttype": "path", "svg.hashsalt": "research-c-three-method-20261108",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "text.color": "#202b36", "axes.labelcolor": "#202b36",
        "xtick.color": "#202b36", "ytick.color": "#202b36",
    })
    fig, ax = plt.subplots(figsize=(11.8, 8.5))
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.34, top=0.81)
    fig.text(0.06, 0.942, "通常CDF・PR-aware DF・ENTRY DFの条件付き総RZ比較",
             fontsize=17, weight="bold")
    fig.text(0.06, 0.898,
             "H4 1.00 Å / STO-3G / CDF8 / LD2 / Z relocation / PR",
             fontsize=12, color="#4e6070")
    heights = [float(p.total / Decimal("1e9")) for p in points]  # Display units only.
    bars = ax.bar(range(3), heights, width=0.58,
                  color=["#929eab", "#477a9b", "#167d75"],
                  edgecolor=["#546371", "#285773", "#0a4c48"], linewidth=1.0)
    bars[2].set_hatch("///")  # Saved E2 is also identifiable in grayscale.
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#e0e5e9", linewidth=0.8)
    ax.set_ylim(0, max(heights) * 1.19)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.set_ylabel("条件付き総native RZ（$10^9$ RZ）", labelpad=13)
    ax.set_xlabel("DF最適化方式", labelpad=16)
    ax.set_xticks(range(3), [f"{p.label}\n{p.detail}\n選択 r = {p.r}" for p in points])
    ax.tick_params(axis="x", length=0, pad=12)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#9da8b1")
    for bar, p in zip(bars, points):
        label = f"{p.total:,.1f}" if p.total % 1 else f"{p.total:,.0f}"
        ax.annotate(label + " RZ", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 10), textcoords="offset points", ha="center", va="bottom",
                    fontsize=11, weight="bold")
    notes = [
        "各方式の保存gridから選択したrを使用。共通r固定ではない。各final価格標本 N = 64。",
        "Ideal finite-RTE IID law / exact native synthesis / finite-bit価格proxy / 状態準備除外。",
        "P–E2：登録主比較Bの名目97.5%モデルでENTRY低費用。CDF0–E2：精度目標未達。",
        "保存済みの異なる実験から取得した点値の記述的比較。",
        "統一した最適化条件での新しい三者実験ではない。共通の統計区間がないためエラーバーなし。",
    ]
    for y, note in zip((0.196, 0.164, 0.132, 0.078, 0.046), notes):
        fig.text(0.06, y, note, fontsize=9.4, color="#364858")
    output.mkdir(parents=True, exist_ok=True)
    description = f"{NOTICE} Sources: {CDF0_COMMIT}:{CDF0_PATH}; {PE_COMMIT}:{PE_PATH}."
    fig.savefig(output / "comparison_total_rz.png", dpi=300,
                metadata={"Description": description})
    svg_path = output / "comparison_total_rz.svg"
    fig.savefig(svg_path, metadata={"Date": None, "Description": description})
    # Matplotlib leaves line-end spaces in SVG paths; keep the Git text clean.
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text(
        encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font", type=Path, help="Japanese font file, e.g. NotoSansCJKjp-Regular.otf")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    render(load_points(repo), args.output_dir, args.font)


if __name__ == "__main__":
    main()
