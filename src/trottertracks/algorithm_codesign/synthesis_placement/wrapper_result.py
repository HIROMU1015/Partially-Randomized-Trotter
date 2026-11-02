"""Inventory/STOP validation only; never computes a signal or resource score."""
from __future__ import annotations

import re


def validate_result(result, contract):
    if (result.get("mandatory_STOP") is not True or result.get("retries") != 0
            or result.get("runs") != 1 or result.get("next_stage_authorized") is not False):
        raise ValueError("one-shot STOP receipt missing")
    for key in ("source_commit", "authorization_commit"):
        if not re.fullmatch(r"[0-9a-f]{40}", result.get(key, "")):
            raise ValueError("full source/authorization identity required")
    if result.get("status") not in contract["terminal_statuses"]:
        raise ValueError("unregistered terminal status")
    if result["status"] == "INCONCLUSIVE_MANDATORY_STOP_NO_RETRY":
        return  # Preserve partial evidence; no complete-map claim.
    if any(result.get(k) != 0 for k in ("synthesis_calls", "sampling_calls", "GPU_query_use")):
        raise ValueError("SP-1 prohibited work receipt")
    domain = contract["domain"]
    expected = {(t, n, m) for t in domain["templates"] for n in domain["sizes"]
                for m in domain["masks"]}
    rows = result.get("rows", [])
    actual = [(r["template"], r["n"], r["mask"]) for r in rows]
    if len(actual) != domain["mask_rows"] or len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError("complete result has missing/duplicate/unregistered mask row")
    if sum(len(r["axes"]) for r in rows) != domain["axis_rows"]:
        raise ValueError("complete result axis inventory mismatch")
    for row in rows:
        if {a["axis"] for a in row["axes"]} != set(domain["axes"]) or len(row["axes"]) != 2:
            raise ValueError("Re/Im axis inventory mismatch")
        duplicate = (("NONE" if row["mask"] == "R" else "D" if row["mask"] == "DR" else None)
                     if row["template"] in ("A", "B") else None)
        if row["duplicate_of_mask"] != duplicate:
            raise ValueError("duplicate mask provenance mismatch")
        positive = (row["classification"] == "MATERIAL_GAIN" and row["mask"] != "NONE"
                    and duplicate is None)
        if row["count_as_independent_positive"] is not positive:
            raise ValueError("duplicate/non-gain counted as independent positive")
        for axis in row["axes"]:
            if axis["status"] == "ELIGIBLE" and (type(axis["shots_sufficient"]) is not int
                    or not 1 <= axis["shots_sufficient"] <= contract["caps"]["shot_cap_per_axis"]):
                raise ValueError("eligible axis has missing/clipped/cap-hit shot count")
    if result["summary"].get("research_GO") is not None or result["summary"].get("automatic_next_stage") is not None:
        raise ValueError("runner cannot authorize research continuation")
