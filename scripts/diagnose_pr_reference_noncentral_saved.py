"""Prepared S1/S2 saved-value reanalysis. Main needs scientific permission.

It is NOT run by the static record audit or artificial tests. No circuit,
synthesis, guard, RPE evaluator, global T aggregation or optimizer is loaded.
"""
from collections import Counter, defaultdict
from fractions import Fraction as F
from functools import lru_cache
import hashlib
import io
import json
import lzma
from pathlib import Path
import subprocess
import tarfile
from audit_pr_reference_noncentral_reuse import weighted_template_max

ROOT = Path(__file__).resolve().parents[1]
BASE = "800554fbb64c21e584bb29a1a8f52490ab4163e0"
META = "artifacts/pr_reference_noncentral_reuse/2026-11-08/RECORD_METADATA.json.xz"
META_SHA = "396f11326d2a5bc86316d0997c744a315b60a856ce5912a1fab064676f285027"
OUT = ROOT / "artifacts/pr_reference_noncentral_saved_numeric/2026-11-08"
TR = "artifacts/pr_reference_transfer_qualification/2026-11-06/"
BP = "artifacts/pr_reference_bprime/2026-11-07/"
VR = "artifacts/pr_reference_depth_v5_reinforcement/2026-11-07/"
INPUTS = {}


@lru_cache(None)
def blob(path):
    raw = subprocess.check_output(["git", "show", BASE + ":" + path], cwd=ROOT)
    INPUTS[path] = dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    return raw


def read(path):
    return json.loads(blob(path))


def archive(path):
    with tarfile.open(fileobj=io.BytesIO(blob(path)), mode="r:xz") as t:
        return [json.load(t.extractfile(m)) for m in t if m.isfile()]


def template_weights(batches, guards, inverse=False):
    counts = defaultdict(list)
    allowed = {"left_all", "right_all", "left_no0", "right_no0", "boundary_control",
               "cz", "z", "cswap", "block:0:full", *(f"block:{j}:half" for j in range(4))}
    for batch in batches:
        for r in batch["records"]:
            name = r["name"]
            if name not in allowed and not name.startswith(("basis:", "basis_inverse:")):
                continue
            native = r["native"] if inverse else r
            keys = Counter(abs(float(p[0])).hex() for op, qs, p in native["lowering"] if op == "rz")
            counts[name].append(keys)
    return {name: weighted_template_max(contexts, guards) for name, contexts in counts.items()}


def weighted_parts(q, r, b, ref, inv, alphabet):
    """Only CT-0's noncentral expressions; scalar '+1' is NOT a norm."""
    def pair_max(a, ids):
        pairs = [a["basis:" + bid] + a["basis_inverse:" + bid] for bid in ids]
        if not pairs:
            raise ValueError("Missing SOURCE basis alphabet")
        return max(pairs)
    def lane(n, a):
        return (2 * a["block:0:half"] + (n - 1) * a["block:0:full"]
                + 2 * n * sum(a[f"block:{j}:half"] for j in (1, 2, 3)))
    assert q > 1 and q % 2 == 0 and r > 0
    if any(v is None for a in (b, ref, inv) for v in a.values()):
        return None
    return dict(Bprime=dict(
        D=b["left_all"] + b["right_all"] + (q - 1) * (
            b["left_no0"] + b["right_no0"] + b["boundary_control"]),
        frames=3 * q * r * pair_max(b, alphabet["Bprime"]),
        products=4 * q * r * b["cz"], CSWAP=F(0)),
        reference=dict(D=lane(q // 2, ref) + lane(q // 2, inv),
        frames=3 * q * r * pair_max(ref, alphabet["reference"]),
        products=4 * q * r * ref["z"], CSWAP=16 * ref["cswap"]))


def pareto_ids(offers):
    """Finite guard/T metadata comparison only; do not choose a policy."""
    for r in offers:
        if r["guard"] < 0 or not isinstance(r["T"], int) or r["T"] < 0:
            raise ValueError("Invalid saved guard/T offer")
    return [a["id"] for a in offers if not any(
        b["guard"] <= a["guard"] and b["T"] <= a["T"] and
        (b["guard"] < a["guard"] or b["T"] < a["T"]) for b in offers)]


def rational_json(value):
    if isinstance(value, F):
        return str(value)
    raise TypeError(type(value).__name__)


def main():
    assert not OUT.exists(), "Fresh numeric output required"
    raw = (ROOT / META).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == META_SHA
    meta = json.loads(lzma.decompress(raw))
    assert meta["base_commit"] == BASE and not meta["current_policy_changed"]
    c0path = "artifacts/pr_reference_ct4/2026-11-07/strict_records.tar.xz"
    c0 = archive(c0path)
    assert hashlib.sha256(blob(c0path)).hexdigest() == meta["source_inputs"][c0path]["sha256"]
    guards = {r["theta_hex"]: F(r["strict_error_upper"]) for r in c0}
    for e in meta["entries"]:
        current = e["current_word"]
        if current["kind"] == "N4_85_composite":
            guards[e["target_literal"]] = F(current["Q_saved"]) + F(current["G_saved"])
    b = template_weights(archive(BP + "Bprime_local_catalogue.tar.xz"), guards)
    ref = template_weights([x for x in archive(TR + "local_catalogue_boundary.tar.xz")
                            if x["request"][0] == "reference"], guards)
    inv = template_weights(archive(VR + "stage_A_raw.tar.xz"), guards, inverse=True)
    condition = read("artifacts/pr_reference_n5b0_static/2026-11-08/M17_INPUTS.json")["condition"]
    assert (condition["m"], condition["q"], condition["r"], condition["bits"]) == (17, 131072, 32, 88)
    assert {a: r["saved_shots_per_axis"] for a, r in condition["arms"].items()} == {"Bprime": 626, "reference": 630}
    tail_ids = tuple(read(TR + "STAGE1_STRUCTURE.json")["source_basis_probabilities"])
    full_ids = {arm: tuple(k[6:] for k in a if k.startswith("basis:")) for arm, a in (("Bprime", b), ("reference", ref))}
    assert all(set(tail_ids) <= set(ids) for ids in full_ids.values())
    old_rows = read("artifacts/pr_reference_post_v0_v1_budget/2026-11-08/RESULTS.json")["fixed_noncentral_loads"]
    rows = []
    for variant, alphabet in (("weighted_all_catalogue_basis", full_ids),
                              ("weighted_SOURCE_RTE_basis", {a: tail_ids for a in full_ids})):
        parts = weighted_parts(131072, 32, b, ref, inv, alphabet)
        for arm in ("Bprime", "reference"):
            for axis in "XY":
                total = None if parts is None else sum(parts[arm].values(), F(0))
                budget = F.from_float(condition["arms"][arm]["joint_synthesis_error_retaining_saved_N_inclusive_upper"])
                old = F(next(x["known_noncentral_certificate_load"] for x in old_rows
                             if (x["arm"], x["axis"]) == (arm, axis)))
                rows.append(dict(variant=variant, arm=arm, axis=axis,
                    parts=None if parts is None else parts[arm], noncentral_upper=total,
                    saved_V1_noncentral_upper=old,
                    no_worse_than_saved_V1=None if total is None else total <= old,
                    saved_joint_budget=budget, residual=None if total is None else budget-total,
                    central_event_scalar_emitter_shell="UNQUALIFIED_NOT_ZERO", m17_RPE="UNQUALIFIED"))
    assert len(rows) == 8
    for arm in ("Bprime", "reference"):
        for axis in "XY":
            a, b_row = [x for x in rows if (x["arm"], x["axis"]) == (arm, axis)]
            if a["noncentral_upper"] is not None and b_row["noncentral_upper"] is not None:
                assert b_row["noncentral_upper"] <= a["noncentral_upper"]
    pareto = []
    for e in meta["entries"]:
        current = e["current_word"]
        g = F(current["strict_error_upper_saved"]) if current["kind"] == "C0_fixed_priority" else (
            F(current["Q_saved"]) + F(current["G_saved"]))
        offers = [dict(id="fixed_current", sequence_sha256=current["sequence_sha256"], guard=g, T=current["existing_T_quoted"])]
        for r in e["exact_literal_records"]:
            if r["status"] == "PASS" and r["guard_method_documented"] and r["strict_spec"]:
                offers.append(dict(id=r["archive"] + ":" + r["member"],
                    sequence_sha256=r["sequence_sha256"], guard=F(r["strict_error_upper_saved"]), T=r["existing_T_quoted"]))
        pareto.append(dict(target_literal=e["target_literal"], offers=offers, nondominated_ids=pareto_ids(offers), policy_chosen=False))
    OUT.mkdir(parents=True)
    values = dict(RESULTS=dict(status="S1_S2_FINITE_SAVED_DIAGNOSTIC_COMPLETE", rows=rows,
        template_weights=dict(Bprime=b, reference=ref, actual_SOURCE_inverse=inv), pareto=pareto,
        full_joint_RPE="UNQUALIFIED", global_T_total="NOT_EVALUATED", new_policy_adopted=False),
        SOURCE_INPUTS=dict(base_commit=BASE, metadata_sha256=META_SHA, fixed_files=INPUTS,
                          runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
        STOP=dict(status="MANDATORY_STOP", next_owner="GPT/user", new_guard_synthesis_compile_RPE_calls=0))
    for name, value in values.items():
        (OUT / (name + ".json")).write_text(json.dumps(value, indent=2, sort_keys=True,
            default=rational_json, allow_nan=False) + "\n")
    print(json.dumps(dict(status=values["RESULTS"]["status"], weighted_rows=len(rows), literal_offer_sets=len(pareto))))


if __name__ == "__main__":
    main()
