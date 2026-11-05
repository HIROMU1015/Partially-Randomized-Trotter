"""Invented scalar rows only; never read saved molecular evidence."""
from copy import deepcopy
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from trottertracks import c_integrated_plan as m


def rows():
    return [dict(geometry="H4_100", input_sha256="a"*64, arm=a, start=s, budget=64,
                 strength=g, r=r, candidate_sha256=f"{a}-{s}-{g}", cohort_role="selection",
                 eligible=True, correctness="PASS", samples=32, total_RZ=1000 + 10*gi + ri)
            for a in m.ARMS for s in m.STARTS
            for gi, g in enumerate(m.STRENGTHS) for ri, r in enumerate(m.R_VALUES)]


def group():
    return [x for x in rows() if (x["arm"], x["start"]) == ("PR", "C0")]


def test_common_r32_and_tuned_select_independently_over_strength():
    data = group()
    data[4]["total_RZ"] = 10
    data[-1]["total_RZ"] = 1
    result = m.select_training(data)
    assert result["common_r32"]["strength"] == .1
    assert (result["fair_retuned"]["strength"], result["fair_retuned"]["r"]) == (.3, 64)
    assert len(m.final_tasks(result)) == 2


@pytest.mark.parametrize("change", ["missing", "duplicate", "final", "input", "candidate", "nan"])
def test_selection_rejects_leakage_incomplete_opportunities_and_wrong_binding(change):
    data = group()
    if change == "missing": data.pop()
    if change == "duplicate": data[-1] = deepcopy(data[0])
    if change == "final": data[0]["cohort_role"] = "final"
    if change == "input": data[0]["input_sha256"] = "b"*64
    if change == "candidate": data[0]["candidate_sha256"] = "changed-by-r"
    if change == "nan": data[0]["total_RZ"] = float("nan")
    with pytest.raises(ValueError): m.select_training(data)


def test_ineligible_rows_are_missing_results_and_never_free():
    data = group()
    for x in data: x.update(eligible=False, total_RZ=None)
    assert m.select_training(data) == dict(common_r32=None, fair_retuned=None)
    data[4].update(eligible=True, total_RZ=3)
    selection = m.select_training(data)
    assert selection["common_r32"] is selection["fair_retuned"]
    assert len(m.final_tasks(selection)) == 1


def test_transfer_freezes_using_development_and_both_arms_complete_starts():
    data = rows()
    # The apparent best strength fails one ENTRY start; neither arm can select
    # that strength by silently restricting its aggregate to favorable starts.
    for x in data:
        if (x["arm"], x["start"], x["strength"]) == ("ENTRY", "P2", .03):
            x.update(eligible=False, total_RZ=None)
    assert m.freeze_transfer_strength(data) == dict(PR=.1, ENTRY=.1)
    data[0]["geometry"] = "H4_115"
    with pytest.raises(ValueError, match="DEVELOPMENT"): m.freeze_transfer_strength(data)


def test_final_dedup_requires_input_candidate_and_r():
    a = group()[0]
    b = dict(a, input_sha256="b"*64)
    assert len(m.final_tasks(dict(fixed=a, tuned=b))) == 2


def test_downstream_bound_accounts_for_strength_changing_selected_candidate():
    a = m.downstream_call_limits(36, 24, 24)
    b = m.downstream_call_limits(12, 24, 12)
    assert a["same_JU_strong_D"] == 24*141
    assert a["final_native_and_accounting"] == b["final_native_and_accounting"]
    assert sum(a.values()) == 63752
    assert sum(b.values()) == 31628


def test_passive_checks_bytes_and_fresh_output_without_science_imports(tmp_path):
    import hashlib
    import json
    import subprocess
    c = dict(scope=m.SCOPE, status="UNAPPROVED_SCIENTIFIC_PROPOSAL", science_executed=0,
             retry=0, resume=0, mandatory_stop=True, grid=dict(starts=list(m.STARTS),
             strengths=list(m.STRENGTHS), accepted_update_checkpoints=list(m.BUDGETS)),
             execution_policy=dict(r_candidates=list(m.R_VALUES)), technical_source="SOURCE.json",
             source_code_commit="invented", saved_development_inputs={}, fresh_outputs=["fresh/run01"],
             maximum_calls=dict(components=dict(stage=dict(unit=2)), stage_totals=dict(stage=2), total=2))
    p = tmp_path / "contract.json"
    p.write_text(json.dumps(c))
    (tmp_path / "SOURCE.json").write_text(json.dumps(dict(code_commit="invented",
        files={"contract.json": hashlib.sha256(p.read_bytes()).hexdigest()})))
    root = Path(__file__).resolve().parents[2]
    code = ("import sys;sys.path.insert(0,sys.argv[1]);from trottertracks.c_integrated_plan import preflight;"
            "r=preflight(sys.argv[2],sys.argv[3]);assert r['scientific_calls']==0;"
            "assert not any(x in sys.modules for x in ['numpy','scipy','pyscf','ffsim','qiskit'])")
    subprocess.run([sys.executable, "-B", "-c", code, str(root/"src"), str(tmp_path), str(p)], check=True)
    (tmp_path / "fresh/run01").mkdir(parents=True)
    with pytest.raises(ValueError, match="FRESH_NO_RETRY_RESUME"): m.preflight(tmp_path, p)
    p.write_text(p.read_text()+" ")
    with pytest.raises(ValueError, match="DRIFT"): m.preflight(tmp_path, p)
