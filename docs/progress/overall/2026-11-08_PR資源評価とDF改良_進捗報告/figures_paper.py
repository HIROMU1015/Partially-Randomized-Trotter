"""Restyle saved presentation values; no scientific computation.

Reads figures_paper/figure_values.csv only. Values, cohorts, saved SEs and
citations stay unchanged. Fractions and axis units are converted for display.
No estimates, ratios, intervals, scientific optimization or research imports.
The DF precision figure includes H8's two published fixed PF/B2 confirmations;
missing precision/method records are left blank.
Run with the repository's .venv311/bin/python.
"""
from pathlib import Path
from fractions import Fraction
import csv
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures_paper"
FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams.update({
    "font.family": font_manager.FontProperties(fname=FONT).get_name(),
    "font.size": 9, "mathtext.fontset": "stixsans", "axes.unicode_minus": False,
    "axes.labelsize": 10, "axes.titlesize": 10, "axes.titlepad": 7,
    "axes.linewidth": 0.75, "axes.edgecolor": "#222222",
    "axes.labelcolor": "#222222", "text.color": "#222222",
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": "#222222", "ytick.color": "#222222",
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.major.width": 0.65, "ytick.major.width": 0.65,
    "xtick.minor.size": 1.8, "ytick.minor.size": 1.8,
    "legend.fontsize": 9, "legend.frameon": False,
    "legend.handlelength": 2, "legend.handletextpad": 0.5,
    "lines.linewidth": 1.35, "lines.markersize": 4.6,
    "lines.markeredgewidth": 0.85, "hatch.linewidth": 0.5,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "svg.fonttype": "path", "svg.hashsalt": "pr-progress-paper-20261110",
    "savefig.facecolor": "white",
})
GRAY, BLUE, GREEN, ORANGE, VERMILLION = (
    "#666A70", "#0072B2", "#009E73", "#C58A12", "#C65D32"
)
EPS = ["0.05", "0.005", "0.0005"]
EPS_MATH = {"0.05": r"5\times10^{-2}", "0.005": r"5\times10^{-3}",
            "0.0005": r"5\times10^{-4}"}
with (OUT / "figure_values.csv").open(encoding="utf-8", newline="") as stream:
    ROWS = list(csv.DictReader(stream))
FIGURES = []


def rows(name, **filters):
    return [r for r in ROWS if r["figure"] == name
            and all(r[k] == v for k, v in filters.items())]


def number(token):
    return float(Fraction(token))


def value(name, **filters):
    found = rows(name, **filters)
    if len(found) != 1:
        raise ValueError(f"Expected one saved record: {name}, {filters}")
    return number(found[0]["saved_value"])


def grid(ax, axis="y"):
    ax.set_axisbelow(True)
    ax.grid(axis=axis, which="major", color="#DADDE0", linewidth=0.5, alpha=0.75)


def panel(ax, index):
    ax.text(0, 1.045, f"({chr(97 + index)})", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=10, fontweight="bold")


def logarithmic(ax, values):
    ax.set_yscale("log")
    # Choosing axis ticks is layout only.
    span = math.log10(max(values) / min(values))
    ax.yaxis.set_major_locator(ticker.LogLocator(
        base=10, subs=(1, 2, 5) if span < 1.8 else (1,), numticks=7))
    ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation(
        base=10, labelOnlyBase=False, minor_thresholds=(math.inf, math.inf)))
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=(2, 3, 4, 5, 6, 7, 8, 9)))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.margins(y=0.15)
    grid(ax)


def legend(fig, handles, labels, columns):
    fig.legend(handles, labels, loc="upper center", ncol=columns,
               bbox_to_anchor=(0.53, 1), borderaxespad=0, columnspacing=1.6)


def save(fig, name):
    sources = sorted({r["source_commit"] + ":" + r["source_path"] for r in rows(name)})
    description = ("Display-only redraw of saved values; no new scientific "
                   "evaluation or inference. Conditions and qualifications "
                   "are in the companion manuscript. Sources: " + "; ".join(sources))
    fig.savefig(OUT / (name + ".png"), dpi=600, bbox_inches="tight",
                pad_inches=0.055, metadata={"Description": description})
    fig.savefig(OUT / (name + ".svg"), bbox_inches="tight", pad_inches=0.055,
                metadata={"Date": None, "Description": description})
    plt.close(fig)
    FIGURES.append(name)


def precision(name, molecules, specifications):
    five_series = len(specifications) > 4
    h8_partial = name == "df_precision" and "H8" in molecules
    height = 3.6 if h8_partial else (3.1 if five_series else 2.95)
    fig, axes = plt.subplots(1, 3, figsize=(7.5, height))
    fig.subplots_adjust(left=0.09, right=0.995, bottom=0.23 if h8_partial else 0.16,
                        top=0.75 if h8_partial else (0.71 if five_series else 0.77),
                        wspace=0.36)
    for p, (ax, epsilon) in enumerate(zip(axes, EPS)):
        plotted = []
        for k, (series, label, color, marker, filled) in enumerate(specifications):
            points = rows(name, panel=epsilon, series=series, metric="G")
            xs = [molecules.index(r["x"]) + (k - (len(specifications) - 1) / 2) * 0.075
                  for r in points]
            ys = [number(r["saved_value"]) for r in points]
            ax.scatter(xs, ys, s=26, marker=marker, edgecolors=color,
                       facecolors=color if filled else "white", linewidths=0.9,
                       label=label, zorder=3)
            plotted.extend(ys)
        logarithmic(ax, plotted)
        ax.set_xticks(range(len(molecules)),
                      [r"$\mathrm{H}_{" + m[1:] + "}$" for m in molecules])
        ax.set_xlim(-0.35, len(molecules) - 0.65)
        ax.set_title(r"$\epsilon_{\rm sig}=" + EPS_MATH[epsilon] + "$")
        panel(ax, p)
    axes[0].set_ylabel(r"総RZ費用 $G$ [RZ]")
    legend(fig, *axes[0].get_legend_handles_labels(),
           3 if five_series else len(specifications))
    fig.text(0.995, 0.985, r"$T=2.0$" if name == "df_full_h_t20" else r"$T=0.8$",
             ha="right", va="top", fontsize=8.5, color="#555555")
    if h8_partial:
        fig.text(0.09, 0.09,
                 "H4/H6：方式別の保存選択値。H8：固定PF対照・PR独立確認（各n=48）。",
                 fontsize=8)
        fig.text(0.09, 0.025,
                 "H8は両端の2精度のみ。Full PF・二体因子RTE・全H純RTEの点は本図の出典にない。",
                 fontsize=8)
    save(fig, name)


def ncg():
    fig, axes = plt.subplots(2, 3, figsize=(7.5, 4.8))
    fig.subplots_adjust(left=0.09, right=0.995, bottom=0.08, top=0.92,
                        hspace=0.65, wspace=0.48)
    cases = [("H4・ε=.05", "4", "0.05"), ("H6・ε=.0005", "6", "0.0005")]
    scales = [(1e4, 1e2, 1e7), (1e8, 1e3, 1e12)]
    powers = [(4, 2, 7), (8, 3, 12)]
    for i, (case, molecule, epsilon) in enumerate(cases):
        for j, metric in enumerate(("N", "C", "G")):
            ax = axes[i, j]
            ys = [value("ncg_cases", panel=case, series=s, metric=metric) / scales[i][j]
                  for s in ("Hybrid PR", "純RTE")]
            bars = ax.bar([0, 1], ys, width=0.5, color=[BLUE, GREEN],
                          edgecolor="#333333", linewidth=0.55, zorder=3)
            for bar, y in zip(bars, ys):
                ax.annotate(f"{y:.3g}", (bar.get_x() + bar.get_width() / 2, y),
                            xytext=(0, 3), textcoords="offset points", ha="center",
                            va="bottom", fontsize=8.5)
            units = ("shots", "RZ/shot", "RZ")
            ax.set_ylabel("$" + metric + "$ " + r"[$10^{"
                          + str(powers[i][j]) + r"}$ " + units[j] + "]")
            ax.set_xticks([0, 1], ["Hybrid PR", "純RTE"], fontsize=8.5)
            ax.set_ylim(0, max(ys) * 1.27)
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
            ax.set_title(r"$\mathrm{H}_{" + molecule + r"},\ \epsilon_{\rm sig}="
                         + EPS_MATH[epsilon] + "$", fontsize=9)
            grid(ax)
            panel(ax, i * 3 + j)
    save(fig, "ncg_cases")


def high_order():
    specifications = [
        ("二次PF", "PF（二次）", GRAY, "--", "s", "white"),
        ("四次PF m5", "PF（四次m5）", GRAY, "-", "s", GRAY),
        ("二次Hybrid PR", "PR（二次）", BLUE, "--", "o", "white"),
        ("四次Hybrid PR m5", "PR（四次m5）", BLUE, "-", "o", BLUE),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 4.9))
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.09, top=0.86,
                        hspace=0.62, wspace=0.28)
    for i, epsilon in enumerate(("0.05", "0.0005")):
        for j, molecule in enumerate(("H4", "H6")):
            ax = axes[i, j]
            plotted = []
            for series, label, color, style, marker, face in specifications:
                points = rows("high_order_absolute", panel=molecule + "/epsilon=" + epsilon,
                              series=series, metric="G")
                xs = [float(r["x"]) for r in points]
                ys = [number(r["saved_value"]) for r in points]
                ax.plot(xs, ys, color=color, linestyle=style, marker=marker,
                        markerfacecolor=face, label=label, zorder=3)
                plotted.extend(ys)
            logarithmic(ax, plotted)
            ax.set_xticks([0.8, 1.2, 1.6, 2.0])
            ax.set_xlabel(r"発展時間 $T$")
            if j == 0:
                ax.set_ylabel(r"総RZ費用 $G$ [RZ]")
            ax.set_title(r"$\mathrm{H}_{" + molecule[1:] + r"}$ · DF"
                         + ("10" if molecule == "H4" else "19")
                         + r", $\epsilon_{\rm sig}=" + EPS_MATH[epsilon] + "$")
            panel(ax, i * 2 + j)
    legend(fig, *axes[0, 0].get_legend_handles_labels(), 4)
    save(fig, "high_order_absolute")


def ratios():
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.8), sharey=True)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.18, top=0.80, wspace=0.16)
    for i, (ax, molecule) in enumerate(zip(axes, ("H4", "H6"))):
        ax.axhline(1, color="#555555", lw=0.8, ls=(0, (3, 3)), zorder=1)
        for series, label, color, marker, style in [
            ("PF", "決定論PF", GRAY, "s", "--"),
            ("PR", "Hybrid PR", BLUE, "o", "-"),
        ]:
            points = rows("higher_order_time", panel=molecule, series=series)
            xs = [float(r["x"]) for r in points]
            ys = [number(r["saved_value"]) for r in points]
            ax.plot(xs, ys, marker=marker, color=color, ls=style, label=label,
                    markerfacecolor="white" if series == "PF" else color)
            if series == "PR":
                ax.annotate(f"{ys[-1]:.4f}", (xs[-1], ys[-1]), xytext=(-2, -15),
                            textcoords="offset points", ha="right", color=color, fontsize=8.5)
        ax.set(xlabel=r"発展時間 $T$", xlim=(0.74, 2.08), ylim=(0.1, 1.18))
        ax.set_xticks([0.8, 1.2, 1.6, 2.0])
        ax.set_yticks([0.25, 0.5, 0.75, 1])
        ax.set_title(r"$\mathrm{H}_{" + molecule[1:] + r"}$ · DF"
                     + ("10" if molecule == "H4" else "19"))
        panel(ax, i)
        grid(ax)
    axes[0].set_ylabel(r"総RZ費用比 $G_{\rm m5}/G_{\rm S2}$")
    legend(fig, *axes[0].get_legend_handles_labels(), 2)
    save(fig, "higher_order_time")


def relocation():
    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    ax.set(xlim=(0, 10), ylim=(0, 3.65))
    ax.axis("off")
    fig.subplots_adjust(left=0.025, right=0.99, bottom=0.13, top=0.98)
    for y, before in ((2.35, True), (0.78, False)):
        ax.text(0.25, y + 1.03, "(a) 再配置前" if before else "(b) 再配置後",
                fontsize=10, va="center", fontweight="bold")
        for x, color, title, body in (
            (0.6, BLUE, r"$\hat H_D$" if before else r"$\hat H_D+L$",
             "一体項 ＋ 保持因子" if before else "一体項 ＋ 保持因子 ＋ 線形Z"),
            (5.65, GREEN, r"$\hat H_R$" if before else r"$\hat H_R-L$",
             "線形Z ＋ ZZ" if before else "ZZ"),
        ):
            ax.add_patch(Rectangle((x, y), 3.65, 0.83, facecolor="#FAFBFC",
                                   edgecolor=color, linewidth=0.95))
            ax.text(x + 1.825, y + 0.59, title, ha="center", va="center",
                    fontsize=13, color=color)
            ax.text(x + 1.825, y + 0.22, body, ha="center", va="center", fontsize=8.8)
    ax.annotate("", xy=(3.55, 1.70), xytext=(6.5, 2.25),
                arrowprops={"arrowstyle": "-|>", "color": "#555555", "lw": 1})
    ax.text(5.7, 1.81, "線形Zと対応する定数を移す", ha="left", va="center", fontsize=8.5)
    ax.text(5, 0.28, r"$\hat H_D+\hat H_R=(\hat H_D+L)+(\hat H_R-L)$",
            ha="center", va="center", fontsize=12)
    fig.text(0.5, 0.035, "I成分は既知の位相として別処理。J・Uと全Hamiltonianを保持する。",
             ha="center", fontsize=8.3)
    save(fig, "z_relocation")


def three_methods():
    fig, ax = plt.subplots(figsize=(6.7, 3.35))
    fig.subplots_adjust(left=0.12, right=0.99, bottom=0.21, top=0.87)
    labels = ["通常CDF0\n$r=64$", "PR-aware\n$r=16$", "ENTRY E2\n$r=16$"]
    series = ["通常CDF0", "PR-aware P", "ENTRY E2"]
    ys = [value("df_three_method_z_relocated", series=s, metric="G") / 1e9 for s in series]
    bars = ax.bar(range(3), ys, width=0.52, color=[GRAY, BLUE, GREEN],
                  edgecolor="#333333", linewidth=0.6, zorder=3)
    bars[-1].set_hatch("////")
    for bar, y in zip(bars, ys):
        ax.annotate(f"{y:.3f}", (bar.get_x() + bar.get_width() / 2, y),
                    xytext=(0, 4), textcoords="offset points", ha="center",
                    va="bottom", fontsize=9)
    ax.set_xticks(range(3), labels)
    ax.set_ylim(0, 265)
    ax.set_yticks([0, 50, 100, 150, 200, 250])
    ax.set_ylabel(r"条件付き総RZ費用 $G$ [$10^9$ RZ]")
    ax.set_title(r"$\mathrm{H}_4$ · CDF8 · $L_D=2$ · Z再配置後", loc="left")
    fig.text(0.12, 0.02, "保存点値（各n=64）。rは候補別に選択。状態準備費用除外。",
             fontsize=8)
    grid(ax)
    save(fig, "df_three_method_z_relocated")


def before_after():
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.9), sharey=True)
    fig.subplots_adjust(left=0.09, right=0.995, bottom=0.20, top=0.84, wspace=0.16)
    panels = ["Z再配置前・各目的でrを選択", "Z再配置後・各目的でrを選択"]
    for i, (ax, name) in enumerate(zip(axes, panels)):
        points = rows("dictionary_comparison", panel=name, metric="G")
        ys = [number(r["saved_value"]) / 1e9 for r in points]
        bars = ax.bar(range(2), ys, width=0.5, color=[BLUE, GREEN],
                      edgecolor="#333333", linewidth=0.6, zorder=3)
        bars[-1].set_hatch("////")
        for bar, y in zip(bars, ys):
            ax.annotate(f"{y:.3f}", (bar.get_x() + bar.get_width() / 2, y),
                        xytext=(0, 4), textcoords="offset points", ha="center",
                        va="bottom", fontsize=9)
        ax.set_xticks([0, 1], [r["series"] for r in points])
        ax.set_ylim(0, 200)
        ax.set_yticks([0, 50, 100, 150, 200])
        ax.set_title("Z再配置前" if i == 0 else "Z再配置後")
        panel(ax, i)
        grid(ax)
    axes[0].set_ylabel(r"条件付き総RZ費用 $G$ [$10^9$ RZ]")
    save(fig, "dictionary_comparison")


def strong_d():
    fig, ax = plt.subplots(figsize=(7.5, 2.45))
    fig.subplots_adjust(left=0.24, right=0.98, bottom=0.23, top=0.88)
    points = rows("strong_d")
    ys = [number(r["saved_value"]) / 1e9 for r in points]
    bars = ax.barh(range(3), ys, height=0.48, color=[GREEN, "#A8ADB2", GRAY],
                   edgecolor="#333333", linewidth=0.6, zorder=3)
    for i in (1, 2):
        bars[i].set_hatch("//" if i == 1 else "....")
    for i, y in enumerate(ys):
        ax.text(y + 1.6, i, f"{y:.3f}", va="center", fontsize=9)
    ax.set_yticks(range(3), ["ENTRY E2 + PR ($r=16$)", "XDF8 ＋ 強D", "CDF8 ＋ 強D"])
    ax.set_xlim(0, 134)
    ax.set_xlabel(r"条件付き総RZ費用 $G$ [$10^9$ RZ]")
    ax.set_title(r"$\mathrm{H}_4$, 共通 $L_D=2$ でZ再配置", loc="left")
    ax.invert_yaxis()
    grid(ax, "x")
    save(fig, "strong_d")


def overview():
    fig, axes = plt.subplots(6, 2, figsize=(12, 14.4))
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99,
                        wspace=0.03, hspace=0.15)
    for ax, name in zip(axes.ravel(), FIGURES):
        ax.imshow(plt.imread(OUT / (name + ".png")))
        ax.set_title(name, fontsize=9, loc="left", pad=3)
        ax.axis("off")
    for ax in axes.ravel()[len(FIGURES):]:
        ax.axis("off")
    fig.savefig(OUT / "overview.png", dpi=160, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    precision("pauli_precision", ["H2", "H4", "H6"], [
        ("Full PF", "Full PF", GRAY, "s", False),
        ("切捨てPF", "切捨てPF", ORANGE, "D", False),
        ("Hybrid PR", "Hybrid PR", BLUE, "o", True),
        ("純RTE", "純RTE", GREEN, "^", True),
    ])
    precision("df_precision", ["H4", "H6", "H8"], [
        ("Full PF", "Full PF", GRAY, "s", False),
        ("決定論PF対照", "決定論PF対照", ORANGE, "D", False),
        ("Hybrid B2", "PR（一部因子を保持）", BLUE, "o", True),
        ("DF-native B3", "二体因子RTE（一体項PF）", VERMILLION, "^", True),
        ("全H純RTE（固定候補）", "全H純RTE（固定）", GREEN, "v", True),
    ])
    ncg()
    high_order()
    ratios()
    relocation()
    three_methods()
    before_after()
    strong_d()
    precision("df_full_h_t20", ["H4", "H6"], [
        ("Full PF", "Full PF", GRAY, "s", False),
        ("決定論PF対照", "決定論PF対照", ORANGE, "D", False),
        ("Hybrid B2", "PR（一部因子を保持）", BLUE, "o", True),
        ("DF-native B3", "二体因子RTE（一体項PF）", VERMILLION, "^", True),
        ("全H純RTE（探索の点最良）", "全H純RTE（探索）", GREEN, "v", True),
    ])
    overview()
    print(f"Rendered {len(FIGURES)} figures as 600-dpi PNG and outlined SVG.")
