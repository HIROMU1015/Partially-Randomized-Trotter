"""Seal saved input copies and UNAPPROVED plans. Never evaluate science.

Private input locations are supplied through a local map, never published.
Old one-shot adapters/shot/cost/binomial functions are not imported or invoked.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import csv
import gzip
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PREP = "d0fc0d24893330066f36b1c5c83860f8c823ecb1"
H6 = "f89c40af73289af9f4c2b70ed0b3966d95d47b06"
SIX = "96e69a01d1fae7a402aac2c4efa570a59f8cc7a0"
EPS = "41ba315b2ad369825577ad671c79b69c77e1f3cc"
H6STEM = "artifacts/resource_applicability/track_a_v1_reaccount_result/2026-10-16/"
PREPSTEM = "artifacts/resource_applicability/pr_alpha_followup_preparation/2026-10-18/"
OUTSTEM = "artifacts/resource_applicability/pr_alpha_input_binding/2026-10-18"
EXECUTION_FILES = ["src/trotterlib/pr_alpha_followup.py", "src/trotterlib/pr_alpha_scalar_input.py",
    "scripts/pr_alpha_followup_runtime.py", "scripts/run_pr_alpha_binomial.py",
    "scripts/run_pr_alpha_sensitivity.py", "scripts/d0_frozen_shot_rule_v1.py"]
PREPARATION_FILES = ["scripts/bind_pr_alpha_saved_inputs.py", "tests/test_pr_alpha_scalar_input.py"]
spec = importlib.util.spec_from_file_location("alpha_copy_only", ROOT / "src/trotterlib/pr_alpha_scalar_input.py")
codec = importlib.util.module_from_spec(spec)
spec.loader.exec_module(codec)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def strict(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            codec.require(key not in result, "DUPLICATE_JSON_KEY")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("NONFINITE_JSON")))


def git(commit, path):
    return subprocess.check_output(["git", "-C", str(ROOT), "show", commit + ":" + path])


def decode(raw, path):
    if path.endswith(".gz"):
        raw = gzip.decompress(raw)
    if ".jsonl" in path:
        return [strict(line) for line in raw.splitlines()]
    if path.endswith(".csv"):
        return list(csv.DictReader(io.StringIO(raw.decode())))
    return strict(raw)


class Store:
    def __init__(self):
        self.receipts = []
        self.private_used = []

    def public(self, role, commit, path):
        raw = git(commit, path)
        blob = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", commit + ":" + path], text=True).strip()
        self.receipts.append(dict(role=role, origin=commit + ":" + path,
            git_blob=blob, bytes=len(raw), sha256=sha(raw), projection_fields=["records.source_fields:" + role]))
        return decode(raw, path)

    def private(self, role, path, receipt):
        path = Path(path)
        codec.require(path.is_absolute() and not path.is_symlink(), "PRIVATE_INPUT_PATH")
        raw = path.read_bytes()
        codec.require(len(raw) == receipt["bytes"] and sha(raw) == receipt["sha256"], "FIXED_PRIVATE_BYTES:" + role)
        self.private_used.append((path, len(raw), sha(raw)))
        self.receipts.append(dict(role=role, origin="PRIVATE_SAVED_SCALAR_BOUND_BY_FIXED_RECEIPT:" + role,
            bytes=len(raw), sha256=sha(raw), projection_fields=["records.source_fields:" + role]))
        return strict(raw)

    def post_check(self):
        for path, size, digest in self.private_used:
            raw = path.read_bytes()
            codec.require(len(raw) == size and sha(raw) == digest, "USED_PRIVATE_INPUT_CHANGED")


def pair_costs(rows, ids):
    """Copy paired RZ integers with original identity; do not average them."""
    grouped = {}
    for index, row in enumerate(rows):
        if row["cell_id"] not in ids or row["phase"] != "confirmation" or row["control"] != "symmetric_directional":
            continue
        key = row["cell_id"], row["replica"]
        pair = grouped.setdefault(key, {})
        codec.require(row["axis"] in ("cosine", "sine") and row["axis"] not in pair, "PAIRED_AXIS_IDENTITY")
        pair[row["axis"]] = (index, row)
    costs = {}
    for cid in ids:
        samples = []
        for (cell, replica), pair in grouped.items():
            if cell != cid:
                continue
            codec.require(set(pair) == {"cosine", "sine"}, "MISSING_PAIRED_AXIS")
            ri, real = pair["cosine"]; ii, imag = pair["sine"]
            for field in ("seed", "event_digest", "compiler_hash"):
                codec.require(real[field] == imag[field], "PAIRED_PROVENANCE:" + field)
            samples.append(dict(replica=int(replica), seed=int(real["seed"]),
                event_digest=real["event_digest"], compiler_hash=real["compiler_hash"],
                real=int(real["RZ"]), imag=int(imag["RZ"]), CSV_data_rows=dict(real=ri, imag=ii),
                source_real=real.get("source"), source_imag=imag.get("source")))
        codec.require(len(samples) == 32 and len({p["replica"] for p in samples}) == 32 and len({p["seed"] for p in samples}) == 32, "CONFIRMATION_N32")
        costs[cid] = dict(kind="PR_PAIRED_MEASURED", phase="confirmation", sample_count=32,
                          pairs=sorted(samples, key=lambda r: r["replica"]))
    return costs


def deterministic_cost(axes):
    if not axes:
        return dict(kind="MISSING")
    if isinstance(axes, list):
        axes = {r["axis"]: r for r in axes}
    codec.require(set(axes) == {"cosine", "sine"}, "COST_AXES")
    classes = {a: "MEASURED" if axes[name]["measured"] else "MODEL"
               for a, name in (("real", "cosine"), ("imag", "sine"))}
    kind = "DETERMINISTIC_MEASURED" if set(classes.values()) == {"MEASURED"} else "DETERMINISTIC_MODEL" if set(classes.values()) == {"MODEL"} else "DETERMINISTIC_MIXED"
    return dict(kind=kind, real=axes["cosine"]["RZ"], imag=axes["sine"]["RZ"],
                axis_classes=classes, original_axis_costs=deepcopy(axes))


def deterministic_axis_index(rows):
    units, origins = {}, {}
    for i, row in enumerate(rows):
        if row["cell_id"].startswith("B2_"):
            continue  # aggregate n8/n32 PR means are never used as paired costs
        key = "H6", row["cell_id"]
        codec.require(row["axis"] not in units.get(key, {}), "DUPLICATE_H6_UNIT_AXIS")
        units.setdefault(key, {})[row["axis"]] = row
        origins.setdefault(key, dict(receipt="H6_unit_cost", pointers={}))['pointers'][row['axis']] = f"/rows/{i}"
    return units, origins


def unit_signature(axes):
    return {axis: (row["RZ"], row["measured"]) for axis, row in axes.items()}


def put(path, value, *, private=False):
    raw = canonical(value) + b"\n"
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o400 if private else 0o644)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
    return dict(bytes=len(raw), sha256=sha(raw))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--local-input-map", type=Path, required=True)
    parser.add_argument("--private-directory", type=Path, required=True)
    args = parser.parse_args()
    source = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--verify", args.source + "^{commit}"], text=True).strip()
    closure = []
    for path in EXECUTION_FILES + PREPARATION_FILES:
        raw = (ROOT / path).read_bytes()
        codec.require(raw == git(source, path), "UNSEALED_SOURCE:" + path)
        closure.append(dict(path=path, bytes=len(raw), sha256=sha(raw)))
    target = args.private_directory
    codec.require(target.is_absolute() and not target.exists() and not target.is_relative_to(ROOT), "NEW_PRIVATE_OUTPUT_ONLY")
    mapping = strict(args.local_input_map.read_bytes())
    s = Store()
    plans = {stage: s.public(stage + "_proposal", PREP, PREPSTEM + stage + "_plan_UNAPPROVED.json") for stage in ("B1", "B2")}
    h6binding = s.public("H6_input_receipts", PREP, "artifacts/resource_applicability/track_a_v1_reaccount_readiness/2026-10-16/input_binding.json")
    h6receipts = {r["logical_name"]: r for r in h6binding["input_receipts"]}
    h6old = s.private("H6_old_signal", mapping["H6_old_signal"], h6receipts["old_signals"])
    h6b0i = s.private("H6_B0I_signal", mapping["H6_B0I_signal"], h6receipts["b0i_signals"])
    h6additional = s.public("H6_additional", PREP, "artifacts/resource_applicability/track_a_v1_phase_a_run02/2026-10-16/signal_results_v1.json.gz")
    h6baseline = s.public("H6_baseline", H6, H6STEM + "candidate_accounting.json.gz")["rows"]
    h6axis = s.public("H6_unit_cost", H6, H6STEM + "axis_costs.json")["rows"]
    h6wrappers = s.public("H6_pairs", "321cd8c3066194fe98741cbebbcdba1515d1deaa", "artifacts/resource_applicability/track_a_h6_confirmation_summary_v1/2026-10-12/wrapper_metrics_v1.csv")
    h8manifest = s.public("H8_manifest", PREP, "references/h8_noncancellation_inputs_v1.json")
    old_receipt = next(r for r in h8manifest["files"] if r["role"] == "old_signals")
    h8old = s.private("H8_old_signal", mapping["H8_old_signal"], old_receipt)
    h8b0i = s.public("H8_B0I_signal", "5f79257f432bc7e0317f122090f62fec379e2a1b", "artifacts/resource_applicability/track_a_h8_native_b0i_full_family_result_v1/2026-10-16/case_scalar_signal_receipts_v1.json")
    h8additional = s.public("H8_additional", "01309059636f2432e29b663bcd2eb0e400f5ae0a", "artifacts/resource_applicability/track_a_h8_phase_a_run01/2026-10-16/H8_signal_complete_v1.json.gz")
    h8D0 = s.public("H8_D0_baseline", "3d910ee1de4597b8895e8506500553dbad5a5085", "artifacts/resource_applicability/track_a_h8_d0_saved_value_execution_v1/2026-10-15/candidate_accounting_v1.json")
    h8b0ia = s.public("H8_B0I_baseline", "5f79257f432bc7e0317f122090f62fec379e2a1b", "artifacts/resource_applicability/track_a_h8_native_b0i_full_family_result_v1/2026-10-16/case_accounting_v1.json")
    h8six = s.public("H8_six_costs", SIX, "artifacts/resource_applicability/track_a_h8_six_axis_reaccount_run01/2026-10-17/accounting_rows.jsonl.gz")
    h8eps = s.public("H8_epsilon_baseline", EPS, "artifacts/resource_applicability/track_a_h8_epsilon_boundary_run01/2026-10-18/accounting_rows.jsonl.gz")
    h8env = s.public("H8_saved_bias_magnitudes", "3f902229cef9da26cb6feeff09eb35860b6115a6", "artifacts/resource_applicability/track_a_h8_noncancellation_shots_run01/2026-10-17/axis_envelopes_v1.csv")
    h8wrappers = s.public("H8_pairs", "681ec23c93fdd1090d325908f4b155a79115afdf", "artifacts/resource_applicability/track_a_h8_resource_map_result_v1/2026-10-14/wrapper_metrics.csv")
    signals, origin = {}, {}
    def add(system, records, role, prefix):
        for index, record in enumerate(records):
            key = system, record["cell"]["id"]
            codec.require(key not in signals, "DUPLICATE_SIGNAL_ID")
            signals[key] = record
            origin[key] = dict(receipt=role, pointer=prefix + str(index))
    add("H6", h6old["signals"], "H6_old_signal", "/signals/")
    add("H6", h6b0i, "H6_B0I_signal", "/")
    add("H6", h6additional["signals"], "H6_additional", "/signals/")
    add("H8", h8old["signals"], "H8_old_signal", "/signals/")
    # B0I entries contain a saved signal under /<index>/signal.
    for index, wrapped in enumerate(h8b0i):
        record = wrapped["signal"]; key = "H8", record["cell"]["id"]
        codec.require(key not in signals, "DUPLICATE_SIGNAL_ID")
        signals[key] = record; origin[key] = dict(receipt="H8_B0I_signal", pointer=f"/{index}/signal")
    add("H8", h8additional["signals"], "H8_additional", "/signals/")
    baseline, baseline_origin = {}, {}
    def set_base(key, value, role, pointer):
        codec.require(key not in baseline, "DUPLICATE_BASELINE")
        baseline[key] = value; baseline_origin[key] = dict(receipt=role, pointer=pointer)
    for i, row in enumerate(h6baseline):
        if row["factor"] != 1 or (row["method"] == "B2" and row.get("sample_scope") != "H6_CONFIRMATION_UNION8"):
            continue
        mode = "USUAL_SIGNED_SAVED" if row["bias_rule"] == "signed_signal" else "EMPIRICAL_NONCANCELLATION"
        set_base(("H6", row["cell_id"], row["epsilon"], mode), row["accounting"], "H6_baseline", f"/rows/{i}/accounting")
    for scope, rows in h8D0.items():
        if scope not in ("H8_EXPLORATION92", "H8_CONFIRMATION_UNION6"):
            continue
        for i, row in enumerate(rows):
            if row["factor"] != 1 or (row["method"] == "B2" and scope != "H8_CONFIRMATION_UNION6"):
                continue
            set_base(("H8", row["cell_id"], row["epsilon"], "USUAL_SIGNED_SAVED"), row["accounting"], "H8_D0_baseline", f"/{scope}/{i}/accounting")
    for i, row in enumerate(h8b0ia):
        if row["factor"] == 1:
            set_base(("H8", row["cell_id"], row["epsilon"], "USUAL_SIGNED_SAVED"), row["accounting"], "H8_B0I_baseline", f"/{i}/accounting")
    for i, record in enumerate(h8additional["signals"]):
        for e, variants in record["precision_rows"].items():
            for factor, a in variants.items():
                if float(factor) == 1:
                    set_base(("H8", record["cell"]["id"], float(e), "USUAL_SIGNED_SAVED"), a, "H8_additional", f"/signals/{i}/precision_rows/{e}/{factor}")
    units, unit_origin = deterministic_axis_index(h6axis)
    h8_case_units, h8_case_origin, h8_unit_signatures = {}, {}, {}
    for i, entry in enumerate(h8six):
        row = entry["updated"]
        if row["factor"] != 1:
            continue
        if entry["bias_kind"] == "EMPIRICAL_NONCANCELLATION":
            saved = entry["saved"]
            set_base(("H8", row["cell_id"], row["epsilon"], entry["bias_kind"]), saved["accounting"], "H8_six_costs", f"/{i}/saved/accounting")
        case = row["cell_id"], row["epsilon"], entry["bias_kind"]
        h8_case_units[case] = row.get("axis_costs")
        h8_case_origin[case] = dict(receipt="H8_six_costs", pointer=f"/{i}/updated/axis_costs")
        if row.get("axis_costs"):
            cid = row["cell_id"]
            signature = unit_signature(row["axis_costs"])
            codec.require(cid not in h8_unit_signatures or h8_unit_signatures[cid] == signature, "UNIT_COST_CHANGED_ACROSS_SAVED_ROWS")
            h8_unit_signatures[cid] = signature
    for i, row in enumerate(h8eps):
        if row["epsilon"] == .000625:
            set_base(("H8", row["cell_id"], row["epsilon"], row["bias_kind"]), row["accounting"], "H8_epsilon_baseline", f"/{i}/accounting")
            case = row["cell_id"], row["epsilon"], row["bias_kind"]
            h8_case_units[case] = row.get("axis_costs")
            h8_case_origin[case] = dict(receipt="H8_epsilon_baseline", pointer=f"/{i}/axis_costs")
            if row.get("axis_costs"):
                codec.require(h8_unit_signatures[row["cell_id"]] == unit_signature(row["axis_costs"]), "EPSILON_UNIT_COST_CHANGED")
    noncancellation = {}
    for i, row in enumerate(h8env):
        codec.require(row["status"] == "EMPIRICAL_NONCANCELLATION_ENVELOPE", "H8_ENVELOPE_STATUS")
        key = "H8", row["candidate_id"]
        noncancellation.setdefault(key, {})[row["axis"]] = float(row["empirical_noncancellation_bias"])
    pairs = {}
    for system, rows in (("H6", h6wrappers), ("H8", h8wrappers)):
        ids = [cid for cid in plans["B2"]["scope"]["candidate_domains"][system] if cid.startswith("B2_")]
        pairs.update({(system, cid): value for cid, value in pair_costs(rows, ids).items()})
    all_records = {}
    for system, cid, e, mode, factor in plans["B2"]["scope"]["case_keys"]:
        key = system, cid, e, mode
        signal = signals.get((system, cid))
        cost = pairs.get((system, cid)) if cid.startswith("B2_") else deterministic_cost(
            units.get((system, cid)) if system == "H6" else h8_case_units.get((cid, e, mode)))
        all_records[key] = codec.make_record(system=system, cell_id=cid, epsilon=e, mode=mode,
            signal=signal, baseline=baseline.get(key), cost=cost,
            noncancellation=noncancellation.get((system, cid)) if mode == "EMPIRICAL_NONCANCELLATION" else None,
            lineage=dict(signal=origin.get((system, cid)), baseline=baseline_origin.get(key),
                costs=dict(receipt=system + "_pairs", row_locator="cost.pairs[*].CSV_data_rows") if cid.startswith("B2_") else
                    (unit_origin.get((system, cid), dict(status="MISSING")) if system == "H6" else h8_case_origin.get((cid, e, mode), dict(status="MISSING"))),
                saved_magnitude=dict(receipt="H8_saved_bias_magnitudes") if system == "H8" and mode == "EMPIRICAL_NONCANCELLATION" else None))
    s.post_check()
    target.mkdir(mode=0o700)
    public = ROOT / OUTSTEM
    public.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for stage, proposal in plans.items():
        records = [deepcopy(all_records[tuple(key[:4])]) for key in proposal["scope"]["case_keys"]]
        payload = dict(schema=codec.SCHEMA, payload_kind="REAL", records=records,
            original_source_receipts=s.receipts, value_policy="COPY_SAVED_BITS_NO_SCIENTIFIC_EVALUATION; recipes evaluated only after independent approval")
        input_path = target / (stage + "_bound_input.json")
        receipt = put(input_path, payload, private=True)
        plan = deepcopy(proposal)
        plan.update(schema="PR_ALPHA_FOLLOWUP_PLAN_V1", status="UNAPPROVED_INPUTS_SEALED",
            sealed=True, inputs_bound=True, source_commit=source,
            source_files=[r for r in closure if r["path"] in EXECUTION_FILES],
            preparation_source_files=[r for r in closure if r["path"] in PREPARATION_FILES],
            input=dict(path=str(input_path), **receipt, schema=codec.SCHEMA), input_blockers=[])
        plan["budget"]["input_recipe_decodes_maximum"] = len(records)
        plan["budget"]["binomial_normalization_decodes_maximum"] = len(records) if stage == "B1" else 0
        plan_path = target / (stage + "_plan_UNAPPROVED.json")
        plan_receipt = put(plan_path, plan, private=True)
        missing = [dict(system=r["system"], cell_id=r["cell_id"], epsilon=r["epsilon"], bias_kind=r["bias_kind"], missing_fields=r["missing_fields"]) for r in records if r.get("missing_fields")]
        summaries[stage] = dict(stage=stage, execution_authorized=False, inputs_bound=True,
            source_commit=source, input=receipt, private_plan=plan_receipt,
            private_plan_canonical_sha256=sha(canonical(plan)), rows=len(records), complete_rows=len(records)-len(missing),
            missing_rows=missing, scope=proposal["scope"], budget=plan["budget"],
            source_files=plan["source_files"], runtime=plan["runtime"],
            cost_classes=dict(Counter(r.get("cost", {}).get("kind", "WHOLE_ROW_MISSING") for r in records)),
            raw_policies=dict(Counter(r.get("raw_policy", "WHOLE_ROW_MISSING") for r in records)),
            grant_issued=False, mandatory_stop=True, real_scientific_calls=0)
        put(public / (stage + "_input_binding_UNAPPROVED.json"), summaries[stage])
    s.post_check()
    put(public / "source_input_receipts.json", dict(receipts=s.receipts,
        metadata_only=True, old_fixed_inputs_modified=False, science_evaluations=0,
        scalar_transform_evaluations_during_preparation=0, private_paths_published=False))
    put(public / "preparation_terminal.json", dict(status="INPUTS_SEALED_UNAPPROVED_MANDATORY_STOP",
        source_commit=source, real_binomial=0, real_shot_cost_rank_reanalysis=0,
        grant_issued=False, B1_authorized=False, B2_authorized=False,
        automatic_retry=False, automatic_next_stage=False))
    print(json.dumps(dict(status="INPUTS_SEALED_UNAPPROVED_MANDATORY_STOP", stages={k:dict(rows=v["rows"], complete=v["complete_rows"], missing=len(v["missing_rows"])) for k,v in summaries.items()}, real_scientific_calls=0)))


if __name__ == "__main__":
    main()
