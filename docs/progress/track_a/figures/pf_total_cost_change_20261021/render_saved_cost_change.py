"""Display only: transcribe the 100 saved total_change_percent values; no accounting."""
from pathlib import Path
import hashlib
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

ROOT = Path(__file__).resolve().parent
source_path = ROOT / "source/figure_data.json"
source = json.loads(source_path.read_text())
bindings = json.loads((ROOT / "display_points.json").read_text())["points"]
assert len(bindings) == len(source["points"]) == 100
for i, b in enumerate(bindings):
    assert b["point"] == source["points"][i]
    assert b["display_index"] == i + 1

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})
fig, ax = plt.subplots(figsize=(13.2, 5.1))
fig.subplots_adjust(left=0.095, right=0.985, top=0.86, bottom=0.24)
groups = [
    ("H6_DF19", "VERBATIM_PREVIOUS70", "#2478a5", "o", "H6 signed / previous 70"),
    ("H8_DF28", "VERBATIM_PREVIOUS70", "#c26b12", "s", "H8 / previous 70"),
    ("H6_DF19", "NEW_STAGE_B_30", "#008b76", "^", "H6 noncancel / new 30"),
]
for lo, hi, label, tint in [
    (0.5, 30.5, "H6 / DF19", "#edf4f8"),
    (30.5, 70.5, "H8 / DF28", "#fbf4eb"),
    (70.5, 100.5, "H6 / DF19", "#edf7f4"),
]:
    ax.axvspan(lo, hi, color=tint, zorder=0)
    ax.text((lo + hi) / 2, 1.035, label, transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=12)
for x in [30.5, 70.5]:
    ax.axvline(x, color="#b1b9be", linewidth=0.8, linestyle="--", zorder=1)
ax.axhline(0, color="#444444", linewidth=1.0, zorder=1)
ax.grid(axis="y", color="#d4dade", linewidth=0.65, zorder=0)

artists = []
for system, origin, color, marker, label in groups:
    chosen = [b for b in bindings
              if b["point"]["system"] == system and b["point"]["origin"] == origin
              and b["winner_total_changed"]]
    x = [b["display_index"] for b in chosen]
    y = [b["point"]["total_change_percent"] for b in chosen]
    artist = ax.scatter(x, y, s=35, marker=marker, c=color, edgecolors="white",
                        linewidths=0.35, zorder=3, label=label)
    artists.append((artist, chosen))
unchanged = [b for b in bindings if not b["winner_total_changed"]]
zero_artist = ax.scatter([b["display_index"] for b in unchanged],
                        [b["point"]["total_change_percent"] for b in unchanged],
                        s=24, marker="o", facecolors="white", edgecolors="#7d858b",
                        linewidths=0.85, zorder=3, label="Unchanged")
artists.append((zero_artist, unchanged))
changed = [b for b in bindings if b["best_candidate_changed"]]
star_artist = ax.scatter([b["display_index"] for b in changed],
                        [b["point"]["total_change_percent"] for b in changed],
                        s=125, marker="*", c="#9f3377", edgecolors="white",
                        linewidths=0.5, zorder=5, label="Candidate changed")

ax.set_xlim(0, 101)
ax.set_ylim(-0.27, 0.385)  # display bounds only; no inferred extrema or statistics
ax.set_xticks([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
ax.set_yticks([-0.2, -0.1, 0, 0.1, 0.2, 0.3])
ax.yaxis.set_major_formatter(FormatStrFormatter("%+.1f"))
ax.set_xlabel("Scope index (saved order)")
ax.set_ylabel("Best PF total cost change (%)")
handles = [Line2D([], [], linestyle="none", marker=m, color=c, markersize=6, label=l)
           for _, _, c, m, l in groups]
handles += [
    Line2D([], [], linestyle="none", marker="o", markerfacecolor="white",
           markeredgecolor="#7d858b", markersize=6, label="Unchanged"),
    Line2D([], [], linestyle="none", marker="*", color="#9f3377",
           markersize=10, label="Candidate changed"),
]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.54, 0.015),
           ncol=3, frameon=False, columnspacing=1.9, handletextpad=0.45,
           fontsize=10.4)

# Audit actual scatter coordinates against the saved tokens, not a recomputed metric.
coordinate_checks = []
for artist, expected in artists:
    offsets = artist.get_offsets()
    assert len(offsets) == len(expected)
    for offset, b in zip(offsets, expected):
        assert float(offset[0]) == b["display_index"]
        assert float(offset[1]) == b["point"]["total_change_percent"]
        coordinate_checks.append(b["point"]["condition_id"])
assert len(coordinate_checks) == len(set(coordinate_checks)) == 100
for offset, b in zip(star_artist.get_offsets(), changed):
    assert float(offset[0]) == b["display_index"]
    assert float(offset[1]) == b["point"]["total_change_percent"]

exports = []
for ext in ["png", "svg", "pdf"]:
    out = ROOT / ("PF_total_cost_change100_clean." + ext)
    metadata = {"Date": None} if ext in ("svg", "pdf") else {}
    fig.savefig(out, dpi=300, facecolor="white", metadata=metadata)
    if ext == "svg":
        # Serialization hygiene only: newline remains an SVG path separator.
        out.write_text("\n".join(line.rstrip() for line in out.read_text().splitlines()) + "\n")
    exports.append({"path": out.name, "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
                    "bytes": out.stat().st_size})
plt.close(fig)
receipt = {
    "scope": "Existing saved scalar fields plotted only; no science accounting or reanalysis",
    "source_field": "figure_data.json:points[].total_change_percent",
    "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
    "matplotlib_version": matplotlib.__version__,
    "python": sys.version.split()[0],
    "ordering": "100 original saved row indices; no sorting, binning or jitter",
    "primary_scatter_points": 100,
    "highlight_overlays": 2,
    "coordinate_binding": "PASS: all 100 scalar y coordinates identical to saved fields",
    "zero_and_nonzero_style": "Saved winner_total_changed flag, not recalculated costs",
    "new_costs_percentages_statistics_ranks": False,
    "exports": exports,
}
(ROOT / "render_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(receipt, ensure_ascii=False))
