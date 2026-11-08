"""Post-run checks and outward display of the fixed D0 result; no new policy."""
from pathlib import Path
from fractions import Fraction as F
import hashlib
import json
import lzma
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/pr_reference_cost_structure/2026-11-08/D0_run01"
SOURCE = "c9d5b55729064882afaa9ae5d816283902f1f53c"
BASE = "654b8501c1348cd0694daeefd19391df6b928354"
FIELDS = ("Bprime_cost", "reference_cost", "change")
ANCHORS = ("Bprime_shot_anchor", "reference_shot_anchor")


def pair(value):
    return tuple(F(x) for x in value)


def add(values):
    values = list(values)
    return tuple(sum((x[k] for x in values), F()) for k in (0, 1))


def aggregate(rows):
    rows = list(rows)
    d = {k: add(pair(x[k]) for x in rows) for k in FIELDS}
    for anchor in ANCHORS:
        d[anchor] = {k: add(pair(x[anchor][k]) for x in rows) for k in ("price", "shots")}
    return d


def decode_split(d):
    return {**{k: pair(d[k]) for k in FIELDS},
            **{a: {k: pair(d[a][k]) for k in ("price", "shots")} for a in ANCHORS}}


def check_split(d, nb, nr, b, r):
    # Compute every expression at shared price endpoints, without using the
    # prepared implementation's split_cost or aggregate_splits functions.
    values = {k: [] for k in FIELDS}
    anchor_values = {a: {"price": [], "shots": []} for a in ANCHORS}
    for pb in b:
        for pr in r:
            change = nr * pr - nb * pb
            values["Bprime_cost"].append(nb * pb)
            values["reference_cost"].append(nr * pr)
            values["change"].append(change)
            candidates = ((nb * (pr - pb), (nr - nb) * pr),
                          (nr * (pr - pb), (nr - nb) * pb))
            for anchor, terms in zip(ANCHORS, candidates):
                assert terms[0] + terms[1] == change
                for key, value in zip(("price", "shots"), terms):
                    anchor_values[anchor][key].append(value)
    expected = {k: (min(v), max(v)) for k, v in values.items()}
    for anchor in ANCHORS:
        expected[anchor] = {k: (min(v), max(v)) for k, v in anchor_values[anchor].items()}
    assert decode_split(d) == expected


def display(value, upper=False):
    q = F(value) * 1000
    v = -((-q.numerator) // q.denominator) if upper else q.numerator // q.denominator
    return ("-" if v < 0 else "") + str(abs(v) // 1000) + "." + str(abs(v) % 1000).zfill(3)


def display_pair(value):
    return [display(value[0]), display(value[1], True)]


def ready(v):
    if isinstance(v, F):
        return str(v)
    if isinstance(v, tuple):
        return [ready(x) for x in v]
    if isinstance(v, list):
        return [ready(x) for x in v]
    if isinstance(v, dict):
        return {k: ready(x) for k, x in v.items()}
    return v


def main():
    if (OUT / "CHECKS.json").exists():
        raise FileExistsError("Existing post-run checks preserved")
    source = json.loads((OUT / "SOURCE.json").read_text())
    source_path = source["source_path"]
    source_raw = (ROOT / source_path).read_bytes()
    assert source_raw == subprocess.check_output(["git", "show", SOURCE + ":" + source_path], cwd=ROOT)
    assert hashlib.sha256(source_raw).hexdigest() == source["source_sha256"]
    originals = {}
    for path, blob in source["fixed_inputs"].items():
        assert subprocess.check_output(["git", "rev-parse", BASE + ":" + path], cwd=ROOT, text=True).strip() == blob
        raw = subprocess.check_output(["git", "show", BASE + ":" + path], cwd=ROOT)
        originals[path] = json.loads(lzma.decompress(raw) if path.endswith(".xz") else raw)
    b = originals["artifacts/pr_reference_bprime/2026-11-07/TOTALS_18ROUND.json"]
    r = originals["artifacts/pr_reference_transfer_qualification/2026-11-06/STAGE4_RPE.json"]
    v = originals["artifacts/pr_reference_rotation_v3/2026-11-07/ANGLE_KEY_MULTIPLICITIES.json.xz"]
    raw = (OUT / "RESULTS.json").read_bytes()
    result = json.loads(raw)
    assert result["input_commit"] == BASE and set(result["input_paths"].values()) == set(originals)
    assert result["next_action"] == "MANDATORY_STOP"
    assert len(result["rows"]) == 36
    by_key = {(x["m"], x["axis"]): x for x in result["rows"]}
    assert len(by_key) == 36 and set(by_key) == {(m, a) for m in range(18) for a in "XY"}
    bm, rm = {x["m"]: x for x in b["rows"]}, {x["m"]: x for x in r["rows"]}
    vm = {(x["m"], x["arm"], x["axis"]): x for x in v["rows"]}
    roles = tuple(result["role_totals"])
    assert len(roles) == 9
    nonzero_shot_cells = []
    for (m, axis), row in by_key.items():
        nb, nr = bm[m]["shots_per_axis"], rm[m]["arms"]["reference"]["shots_per_axis"]
        assert row["Bprime_shots"] == nb and row["reference_shots"] == nr
        assert (row["q"], row["r"], row["bits"]) == tuple(bm[m][k] for k in ("q", "r", "bits"))
        assert tuple(rm[m][k] for k in ("q", "r", "bits")) == tuple(row[k] for k in ("q", "r", "bits"))
        p = bm[m]["price_per_axis"][axis]
        bp, rp = pair((p["Bprime_RZ_lower"], p["Bprime_RZ_upper"])), pair((p["reference_RZ_lower"], p["reference_RZ_upper"]))
        check_split(row, nb, nr, bp, rp)
        for role in roles:
            prices = []
            for arm, n in (("Bprime", nb), ("reference", nr)):
                item = vm[m, arm, axis]
                assert item["shots"] == n
                assert tuple(item[k] for k in ("q", "r", "bits")) == tuple(row[k] for k in ("q", "r", "bits"))
                assert set(item["parts"]) == set(roles)
                count = sum((F(x) for x in item["parts"][role].values()), F())
                prices.append((F() if arm == "reference" and role == "RTE_products" else count, count))
            check_split(row["roles"][role], nb, nr, *prices)
        for field in FIELDS:
            assert add(pair(x[field]) for x in row["roles"].values()) == pair(row[field])
        if nr != nb:
            nonzero_shot_cells.append([m, axis])
    assert nonzero_shot_cells == [[17, "X"], [17, "Y"]]
    assert aggregate(result["rows"]) == decode_split(result["totals"])
    for role in roles:
        assert aggregate(x["roles"][role] for x in result["rows"]) == decode_split(result["role_totals"][role])
    assert len(result["round_costs"]) == 18
    for row in result["round_costs"]:
        assert aggregate(x for x in result["rows"] if x["m"] == row["m"]) == decode_split(row)
    totals = decode_split(result["totals"])
    assert totals["Bprime_cost"] == pair((b["exact_Bprime_total_fractions"]["RZ_lower"], b["exact_Bprime_total_fractions"]["RZ_upper"]))
    assert totals["reference_cost"][0] == F(b["exact_reference_RZ_lower_fraction"])
    for arm, field in (("Bprime", "Bprime_cost"), ("reference", "reference_cost")):
        assert F.from_float(b["totals"][arm]["total_RZ_lower"]) <= totals[field][0]
        assert totals[field][1] <= F.from_float(b["totals"][arm]["total_RZ_upper"])
    assert F.from_float(b["comparisons"]["RZ"]["guaranteed_absolute_increase_lower"]) <= totals["change"][0]
    before = aggregate(x for x in result["rows"] if x["m"] < 17)
    terminal = decode_split(result["round_costs"][17])
    order = {}
    for field in ("Bprime_cost", "reference_cost"):
        rows = sorted(result["round_costs"], key=lambda x: F(x[field][0]), reverse=True)
        assert all(F(x[field][0]) > F(y[field][1]) for x, y in zip(rows, rows[1:]))
        order[field] = [x["m"] for x in rows]
        assert terminal[field][0] > before[field][1]
    basis = pair(result["role_totals"]["RTE_basis_frames"]["change"])
    assert all(basis[0] > F(value["change"][1])
               for role, value in result["role_totals"].items() if role != "RTE_basis_frames")
    d_net = add(pair(result["role_totals"][role]["change"])
                for role in ("D_unfused", "D_boundary_fusion_difference"))
    archive = OUT / "RESULTS.json.xz"
    if archive.exists():
        raise FileExistsError("Existing archive preserved")
    compressed = lzma.compress(raw)
    assert lzma.decompress(compressed) == raw
    archive.write_bytes(compressed)
    checks = {
        "status": "EXACT_D0_POSTRUN_CHECKS_PASS", "execution_source": SOURCE,
        "source_and_three_inputs_unchanged": True, "fixed_source_CLI_invocations": 1,
        "CLI_exit_code": 0, "CLI_wall_seconds_reported_by_exec": 0.240103571,
        "postrun_checker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "verified_round_axis_cells": 36, "verified_roles_per_cell": 9,
        "endpoint_and_anchor_identities": True, "role_round_and_global_sums": True,
        "original_exact_totals_and_outward_bounds": True,
        "shot_difference_cells": nonzero_shot_cells,
        "RZ_round_order_proven_by_disjoint_bounds": order,
        "m17_RZ_cost_exceeds_prior17_combined": True,
        "largest_positive_net_role_change": "RTE_basis_frames",
        "m0_to_m16": before, "m17": terminal, "D_net_role_change": d_net,
        "display_RZ": {
            "totals": {k: display_pair(v) for k, v in totals.items() if k in FIELDS},
            "Bprime_shot_anchor": {k: display_pair(v) for k, v in totals["Bprime_shot_anchor"].items()},
            "reference_shot_anchor": {k: display_pair(v) for k, v in totals["reference_shot_anchor"].items()},
            "prior17_change": display_pair(before["change"]), "m17_change": display_pair(terminal["change"]),
            "D_net_role_change": display_pair(d_net),
        },
        "raw_result_bytes": len(raw), "raw_result_sha256": hashlib.sha256(raw).hexdigest(),
        "archive_bytes": len(compressed), "archive_sha256": hashlib.sha256(compressed).hexdigest(),
        "archive_decodes_to_original_raw_bytes": True,
        "native_phase_shots_circuits_or_T_policy_changed": False,
        "independent_scientific_review": False, "next_action": "MANDATORY_STOP",
    }
    with (OUT / "CHECKS.json").open("x") as stream:
        json.dump(ready(checks), stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(ready(checks), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
