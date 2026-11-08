"""Reuse qualification for the selected central q4 fixture, not new simulation."""
import hashlib
import json


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",", ":")).encode()).hexdigest()


def check_request(record,decision,rows):
    if (record["sample_id"],record["q"],record["sample"],decision["selected"])!=("q4_sample00",4,0,2):
        raise ValueError("CENTRAL_FIXED_PATH_ONLY")
    keys=[(r["method"],r["axis"]) for r in rows]
    if len(keys)!=4 or set(keys)!={(m,a) for m in ("symmetric_directional","reference") for a in ("X","Y")}:
        raise ValueError("FOUR_SAVED_WRAPPERS_REQUIRED")
    for row in rows:
        request=row["request"]
        expected=dict(q=4,sample=0,seeds=record["seeds"],events_sha256=digest(record["ordered_events"]),
            execution_source=record["execution_source"],cut_after_complete_step=2,
            science_source="ba0e2bf2d1d6f0dc78616d3392f3ff5b6ba7d5b7",endpoint=record["endpoint"],
            input_packet_sha256=record["input_packet_sha256"],law_sha256=record["law_sha256"],cohort=record["cohort"])
        if request!=expected or row["request_fingerprint"]!=digest(expected):raise ValueError("SAME_EVENT_REQUEST")
        if row["execution_source"]!=record["execution_source"] or row["optimization_policy"]!="strong_whole_path_half_sweep_relative_basis_scalar_aggregation_v1":
            raise ValueError("SOURCE_OR_STRONG_POLICY")
        if row["precision"]!=dict(component_abs=1e-10,observable_frobenius=2e-10,rtol=0,global_phase_fit=False):
            raise ValueError("PRECISION_OR_PHASE")
        if (row["cost_scope"],row["state_preparation"],row["topology"])!=("single_hadamard_interrogation_without_state_preparation","EXCLUDED","free"):
            raise ValueError("FULL_SCOPE")
        q=row["qualification"]
        if any(q[k]!="PASS" for k in ("raw","postcheck","paired_XY_phase")):raise ValueError("OLD_PASS_REQUIRED")


def checked_errors(rows):
    if not rows:raise ValueError("EMPTY_OLD_CHECKS")
    for row in rows:
        if (row["tolerance"]!=1e-10 or row.get("rtol")!=0 or not 0<=row["max_entry"]<=1e-10):
            raise ValueError("OLD_COMPONENT_CHECK")
        if "observable" in row["label"] or row["label"].startswith("compiled_paired"):
            if not 0<=row["frobenius"]<=2e-10:raise ValueError("OLD_OBSERVABLE_CHECK")
