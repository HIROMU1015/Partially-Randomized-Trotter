"""Synthetic PM0 bookkeeping tests and one read-only saved-JSON regression.

Never call original science/validation runners (they may inspect runtime/cache).
"""
import ast
import copy
import csv
import io
import json
from pathlib import Path
import sys

import pytest

from trottertracks.resource_applicability import pm0_evidence_attribution as pm0


def line(name, n, g, **metrics):
    return {"candidate_id":name,"total_shots":n,"rz_count":g,**metrics}


def test_primary_minimum_is_distinct_from_pareto():
    rows=[line("a",3,4,cx_count=8),line("b",4,5,cx_count=7),line("c",5,6,cx_count=9)]
    assert pm0.minimum(rows)["candidate_id"] == "a"
    assert {r["candidate_id"] for r in pm0.frontier(rows,("rz_count","cx_count"))} == {"a","b"}


def test_equal_points_both_remain_pareto():
    rows=[line("a",2,3),line("b",2,3)]
    assert len(pm0.frontier(rows,("rz_count",))) == 2


def test_affine_crossing_without_P_grid():
    out=pm0.lower_envelope([line("a",5,10),line("b",2,19),line("dominated",6,20)])
    assert [(r["candidate_id"],r["P_min"],r["P_max"]) for r in out] == [("a",0,3),("b",3,None)]


def test_identical_lines_and_zero_width_boundary_are_retained():
    out=pm0.lower_envelope([line("a",4,10),line("duplicate",4,10),line("b",2,20),line("tie",3,15)])
    assert {r["candidate_id"] for r in out} == {"a","duplicate","b","tie"}
    tie=next(r for r in out if r["candidate_id"]=="tie")
    assert tie["P_min"] == tie["P_max"] == 5


def test_parallel_more_expensive_line_is_rejected():
    assert [r["candidate_id"] for r in pm0.lower_envelope([line("cheap",3,10),line("expensive",3,11)])] == ["cheap"]


def test_input_identity_mismatch_fails_closed():
    with pytest.raises(ValueError,match="differs"):
        pm0.verified_bytes(b"edited",b"original")


def test_bias_decomposition_keeps_signed_cancellation():
    signal={"candidate":{"method":"B2"},"corrected_mean":{"real":1,"imag":0},
            "exact_target":{"real":0,"imag":0},"pf_exact_tail_signal":{"real":2,"imag":0}}
    result=pm0.complex_bias(signal)
    assert result["outer_bias_real_signed"] == 2
    assert result["finite_bias_real_signed"] == -1
    assert result["total_bias_abs"] == 1


def test_discard_does_not_fabricate_exact_truncated_signal():
    result=pm0.complex_bias({"candidate":{"method":"B0"},"corrected_mean":{"real":1,"imag":0},
                            "exact_target":{"real":0,"imag":0}})
    assert result["pure_discard_pf_bias_abs"] is None
    assert result["discard_bias_abs"] is None


def test_csv_distinguishes_missing_and_zero():
    text=pm0.csv_text([{"missing":None,"zero":0}])
    row=list(csv.DictReader(io.StringIO(text)))[0]
    assert row == {"missing":"MISSING","zero":"0"}


def test_same_R_rejects_normalization_mismatch():
    row={"method":"B2","rank":3,"K":2,"T":.8,"R":8,"q":1,
         "tau":.1,"normalization":1.1,"expected_random_applications":8.2,"candidate_id":"a"}
    other={**row,"q":2,"candidate_id":"b","normalization":1.2}
    with pytest.raises(ValueError,match="same R"):
        pm0.same_R_groups([row,other])


def test_source_namespace_has_no_science_dependency():
    source=Path(pm0.__file__).read_text()
    tree=ast.parse(source)
    imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    imports += [a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names]
    assert not any(x and x.startswith(("numpy","scipy","qiskit","trotterlib","cupy")) for x in imports)
    assert len(pm0.INPUTS) == 9
    assert all(p.endswith((".json",".py")) and ".runtime" not in p for p in pm0.INPUTS.values())


@pytest.fixture(scope="module")
def saved():
    root=Path(__file__).parents[3]
    data,audit=pm0.load_inputs(root)
    return data,audit,pm0.analyze(data,audit)


def test_saved_result_regression_and_stops(saved):
    _,_,(s,rows,_) = saved
    assert len(rows) == 215
    assert s["counts"]["m1_eligible"] == 206
    assert s["counts"]["actual_frontier_in_selector"] == 1
    assert s["counts"]["actual_frontier_in_proxy"] == 2
    assert s["common_domains"]["M1_q8"]["best_by_method"]["B2"]["candidate_id"] == "B2-rank3-q8-r2-K2"
    assert next(r for r in s["selector_regret"] if r["metric"]=="rz_count")["regret_fraction"] == 0
    assert s["pm1_authorized"] is s["next_stage_authorized"] is False
    assert s["access_audit"]["npz_resolve_stat_hash_load"] == s["access_audit"]["runtime_cache_reads"] == 0


def test_saved_fingerprint_and_common_P_domains(saved):
    _,_,(s,_,_) = saved
    payload={k:v for k,v in s.items() if k!="summary_fingerprint"}
    assert pm0.digest(payload) == s["summary_fingerprint"]
    for domain in ("M1_common5","M2_common5"):
        envelope=s["state_preparation_point_lower_envelopes"][domain]
        assert envelope[-1]["candidate_id"] == "B1-rank12-q1-r0-K0"
    assert all(r["candidate_id"].startswith("B2") for r in s["state_preparation_point_lower_envelopes"]["M1_all"])


@pytest.mark.parametrize("mutation",["candidate","signal","axis_shots"])
def test_saved_candidate_and_signal_identity_rejects_changes(saved,mutation):
    original,audit,_=saved
    data=copy.deepcopy(original)
    if mutation=="candidate":
        data["m1b"]["compile_map"][0]["candidate"]["r"] += 1
    elif mutation=="signal":
        data["m1a"]["signal_records"][0]["axis_bias"]["real"] += .01
    else:
        data["m1b"]["compile_map"][0]["axis_shots"]["real"] += 1
    with pytest.raises(ValueError):
        pm0.analyze(data,audit)


def test_saved_R8_and_missing_rank_grid(saved):
    _,_,(s,_,matrix)=saved
    group=next(g for g in s["same_R_groups"] if (g["method"],g["rank"],g["K"],g["R"]) == ("B2",3,2,8))
    assert len(group["candidate_ids"]) == 4
    assert all(r["status"]=="NOT_REGISTERED" for r in matrix if r["method"]=="B0" and r["rank"] in (4,5))


def test_loader_accesses_only_allowlisted_names(monkeypatch):
    root=Path("/synthetic-pm0-root")
    seen=[]
    def read(path):
        seen.append(str(path.relative_to(root)))
        assert seen[-1] in pm0.INPUTS.values()
        return b"{}" if str(path).endswith(".json") else b"# synthetic source\n"
    monkeypatch.setattr(Path,"read_bytes",read)
    monkeypatch.setattr(pm0.subprocess,"check_output",lambda *a,**k: read(root/a[0][-1].split(":",1)[1]))
    pm0.load_inputs(root)
    assert len(seen)==18 and set(seen)==set(pm0.INPUTS.values())
