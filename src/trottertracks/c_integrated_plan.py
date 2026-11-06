"""Passive integrated research-C plan and selection of accounted training rows.

No optimizer, array loader, molecular engine or compiler is imported here.
The command-line entry only checks a proposal; it cannot launch science.
"""
import hashlib
import json
import math
from pathlib import Path
from statistics import median

STRENGTHS = (.03, .1, .3)
R_VALUES = (16, 32, 64)
STARTS = ("C0", "P1", "P2")
BUDGETS = (16, 64)
ARMS = ("PR", "ENTRY")
SCOPE = "C_INTEGRATED_ROBUSTNESS_FAIR_RETUNING_TRANSFER_V1"


def _cost(row):
    if row["correctness"] != "PASS" or row["samples"] != 32:
        raise ValueError("QUALIFIED_SELECTION_COHORT_REQUIRED")
    if type(row["eligible"]) is not bool:
        raise ValueError("BOOLEAN_ELIGIBILITY_REQUIRED")
    if row["eligible"] is not True:
        if row["total_RZ"] is not None:
            raise ValueError("INELIGIBLE_COST_MUST_BE_NULL")
        return math.inf
    value = row["total_RZ"]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("QUALIFIED_ACCOUNTED_TOTAL_REQUIRED")
    return value


def select_training(rows, *, strengths=STRENGTHS, r_values=R_VALUES,
                    common_r=32, tie_order=(32, 16, 64)):
    """Select independently at common r32 and over the same strength x r grid.

    Supply exactly one arm/start/update checkpoint. Ineligible policies remain
    explicit rows. No missing observation becomes a free zero-cost candidate.
    Caller supplies existing serial-accounting totals, never raw objectives.
    """
    rows = list(rows)
    if (not r_values or any(type(r) is not int or r < 1 for r in r_values)
            or len(set(r_values)) != len(r_values) or common_r not in r_values
            or len(tie_order) != len(r_values) or set(tie_order) != set(r_values)):
        raise ValueError("FIXED_COMMON_AND_RETUNING_R_GRID")
    expected = {(g, r) for g in strengths for r in r_values}
    keys = [(x["strength"], x["r"]) for x in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("COMPLETE_EQUAL_STRENGTH_R_OPPORTUNITIES")
    if any(x["cohort_role"] != "selection" for x in rows):
        raise ValueError("FINAL_DATA_MUST_NOT_SELECT")
    for field in ("geometry", "input_sha256", "arm", "start", "budget"):
        if len({x[field] for x in rows}) != 1:
            raise ValueError("MIXED_SELECTION_GROUP:" + field)
    for strength in strengths:
        block = [x for x in rows if x["strength"] == strength]
        if len({x["candidate_sha256"] for x in block}) != 1:
            raise ValueError("R_SEARCH_CHANGED_DF_CANDIDATE")
    out = {}
    common_name = f"common_r{common_r}"
    for name, candidates in ((common_name, [x for x in rows if x["r"] == common_r]),
                             ("fair_retuned", rows)):
        eligible = [x for x in candidates if math.isfinite(_cost(x))]
        out[name] = (min(eligible, key=lambda x: (_cost(x), x["strength"], tie_order.index(x["r"])))
                     if eligible else None)
    return out


def freeze_transfer_strength(rows):
    """Development-only median training cost at update64, minimized over r.

    A strength must have qualified data at all three registered starts for
    BOTH arms. This keeps one method's missing start out of a favorable subset.
    The selected strength may differ between methods. No transfer row is read.
    """
    rows = list(rows)
    if any(x["geometry"] != "H4_100" or x["cohort_role"] != "selection"
           or x["budget"] != 64 for x in rows):
        raise ValueError("DEVELOPMENT_UPDATE64_TRAINING_ONLY")
    if len({x["input_sha256"] for x in rows}) != 1:
        raise ValueError("ONE_DEVELOPMENT_PHYSICAL_INPUT_REQUIRED")
    expected = {(a, s, g, r) for a in ARMS for s in STARTS
                for g in STRENGTHS for r in R_VALUES}
    keys = [(x["arm"], x["start"], x["strength"], x["r"]) for x in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("COMPLETE_DEVELOPMENT_GRID_REQUIRED")
    groups = {(a, s): select_training([x for x in rows if (x["arm"], x["start"]) == (a, s)])
              for a in ARMS for s in STARTS}
    # select_training above also checks group/input/candidate binding.
    del groups
    scores = {a: {} for a in ARMS}
    for g in STRENGTHS:
        costs = {a: [min(_cost(x) for x in rows
                        if (x["arm"], x["start"], x["strength"]) == (a, s, g))
                     for s in STARTS] for a in ARMS}
        if all(math.isfinite(c) for values in costs.values() for c in values):
            for a in ARMS:
                scores[a][g] = median(costs[a])
    if any(not v for v in scores.values()):
        raise ValueError("NO_COMMON_COMPLETE_STRENGTH_FOR_TRANSFER")
    return {a: min(scores[a], key=lambda g: (scores[a][g], g)) for a in ARMS}


def final_tasks(selections):
    """Deduplicate only an identical input/candidate/r, retaining claim labels."""
    tasks = {}
    for label, row in selections.items():
        if row is None:
            continue
        key = (row["input_sha256"], row["candidate_sha256"], row["r"])
        if key not in tasks:
            tasks[key] = dict(input_sha256=key[0], candidate_sha256=key[1], r=key[2],
                              samples=64, cohort_role="final", labels=[])
        tasks[key]["labels"].append(label)
    return list(tasks.values())


def downstream_call_limits(candidates, final_policies, deterministic_candidates):
    """Existing logical work accounting, including both N32 final batches.

    Each selected policy uses 2 native32 batches, one combined calibration and
    the existing 33-unit affine accounting. Ground/PF is shared only per input
    and candidate; r16/r64 add fresh RTE/round/shot work (80 units each).
    """
    if any(type(n) is not int or n < 0 for n in (candidates, final_policies, deterministic_candidates)):
        raise ValueError("NONNEGATIVE_INTEGER_TASK_COUNTS")
    return dict(candidate_setup=14*candidates, fixed_r32_references=93*candidates,
                additional_r_references=2*80*candidates,
                selection_native_and_accounting=(35+3*(289+33))*candidates,
                final_native_and_accounting=(2*289+1+33)*final_policies,
                same_JU_strong_D=141*deterministic_candidates,
                common_selection_comparison_summary=7+12+12+1)


def checked_path(root, relative):
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root):
        raise ValueError("REPOSITORY_RELATIVE_PATH_REQUIRED")
    return path


def preflight(root, contract_path):
    """Bytes/JSON/paths only. No NPZ decode, fit, RPE or native work."""
    root = Path(root).resolve()
    c = json.loads(Path(contract_path).read_text())
    if (c["scope"] != SCOPE or c["status"] != "UNAPPROVED_SCIENTIFIC_PROPOSAL"
            or c["science_executed"] != 0 or c["retry"] != 0 or c["resume"] != 0
            or c["mandatory_stop"] is not True):
        raise ValueError("PROPOSAL_SCOPE_OR_STOP")
    if (c["grid"]["starts"] != list(STARTS) or c["grid"]["strengths"] != list(STRENGTHS)
            or c["grid"]["accepted_update_checkpoints"] != list(BUDGETS)
            or c["execution_policy"]["r_candidates"] != list(R_VALUES)):
        raise ValueError("REGISTERED_GRID")
    source = json.loads(checked_path(root, c["technical_source"]).read_text())
    for group in (source["files"], c["saved_development_inputs"]):
        for name, digest in group.items():
            if hashlib.sha256(checked_path(root, name).read_bytes()).hexdigest() != digest:
                raise ValueError("SOURCE_OR_INPUT_DRIFT:" + name)
    if source["code_commit"] != c["source_code_commit"]:
        raise ValueError("SOURCE_COMMIT_BINDING")
    for name in c["fresh_outputs"]:
        if checked_path(root, name).exists():
            raise ValueError("FRESH_NO_RETRY_RESUME")
    actual = {k: sum(v.values()) for k, v in c["maximum_calls"]["components"].items()}
    if actual != c["maximum_calls"]["stage_totals"] or sum(actual.values()) != c["maximum_calls"]["total"]:
        raise ValueError("CALL_SUM_MISMATCH")
    return dict(status="PASSIVE_PREPARATION_PASS", scientific_calls=0,
                scientific_input_arrays_decoded=0, molecular_qualification="NOT_RUN",
                source_code_commit=source["code_commit"], maximum_calls=c["maximum_calls"]["total"],
                mandatory_stop=True)
