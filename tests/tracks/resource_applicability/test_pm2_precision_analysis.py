"""Pure synthetic tests. Never load real saved JSON, NPZ or old science runners."""
import ast
import copy
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics

import jsonschema
import pytest

from trottertracks.resource_applicability import pm2_precision_analysis as a
from trottertracks.resource_applicability import pm2_precision_contract as c


def fixture(name="synthetic", *, random=False, bias=None, B=1.0, dataset="development", cost=20):
    bias = {"real": 0.0, "imag": 0.01} if bias is None else bias
    shots = {axis: None if 0.05/math.sqrt(2)-b <= 0 else
             math.ceil(2*B**2/(0.05/math.sqrt(2)-b)**2*math.log(80)) for axis, b in bias.items()}
    eligible = all(v is not None for v in shots.values())
    fp = hashlib.sha256(name.encode()).hexdigest()
    signal = {"candidate_fingerprint": fp, "axis_bias": bias, "normalization_multiplier": B,
              "axis_shots": shots, "accuracy_eligible": eligible}
    entry = {"dataset": dataset, "candidate_id": name, "candidate_fingerprint": fp,
        "parameters": {"method": "B2" if random else "B0", "rank": 3 if random else 5, "q": 1, "r": 4 if random else 0, "K": 2 if random else 0, "T": 0.8, "delta": 0.8},
        "signal_record_fingerprint": c.fingerprint(signal), "reference_accuracy_eligible": eligible}
    pairs = [{"cosine": {m: cost+i for m in c.METRICS}, "sine": {m: cost+3+2*i for m in c.METRICS}}
             for i in range(32 if random else 1)]
    means = {axis: {m: statistics.mean(p[axis][m] for p in pairs) for m in c.METRICS} for axis in c.AXES.values()}
    work = {m: shots["real"]*means["cosine"][m]+shots["imag"]*means["sine"][m] for m in c.METRICS} if eligible else None
    return a.project_candidate(entry, signal, means, pairs, work)


def test_fixed_epsilon_design_not_adaptive():
    points = a.precision_points()
    assert len(points) <= 302 and points == sorted(set(points))
    assert points[0] == 0.005 and points[-1] == 0.1 and 0.05 in points
    assert set(0.005*20**(i/300) for i in range(1,300)) <= set(points)


@pytest.mark.parametrize("epsilon", [0, -1, math.nan, math.inf, True])
def test_invalid_epsilon_rejected(epsilon):
    with pytest.raises(ValueError):
        a.shot_accounting({"real": 0, "imag": 0}, 1, epsilon)


@pytest.mark.parametrize("B", [0.5, -1, math.nan, math.inf, True])
def test_invalid_normalization_rejected(B):
    with pytest.raises(ValueError):
        a.shot_accounting({"real": 0, "imag": 0}, B, 0.05)


def test_strict_equality_and_next_float_boundary():
    bias = {"real": 0.01, "imag": 0.02}
    boundary = math.sqrt(2)*0.02
    equal = a.shot_accounting(bias, 1, boundary)
    assert not equal["accuracy_eligible"] and equal["axis_shots"]["imag"] is None
    above = a.shot_accounting(bias, 1, math.nextafter(boundary, math.inf))
    assert above["accuracy_eligible"] and above["N_total"] > 0


def test_axis_null_and_work_null_not_zero():
    row = a.evaluate_candidate(fixture(bias={"real": 0.1, "imag": 0}), 0.05)
    assert row["N_real"] is None and row["N_imag"] is not None
    assert row["N_total"] is None and row["primary_SE"] is None
    assert not row["point_frontier"]
    assert all(row[m] is None for m in c.METRICS)


def test_normalization_squared_and_ceil():
    base = a.shot_accounting({"real": 0, "imag": 0}, 1, 0.05)
    norm = a.shot_accounting({"real": 0, "imag": 0}, 2, 0.05)
    expected = math.ceil(2/(0.05/math.sqrt(2))**2*math.log(80))
    assert base["axis_shots"] == {"real": expected, "imag": expected}
    assert norm["N_total"] != 4*base["N_total"]  # ceil applies after scaling
    assert norm["axis_shots"]["real"] == math.ceil(8/(0.05/math.sqrt(2))**2*math.log(80))


def test_axis_weighting_not_fixed_effective_cost():
    candidate = fixture()
    first, second = (a.evaluate_candidate(candidate, e) for e in (0.05, 0.1))
    assert second["primary_RZ_P0"] == second["N_real"]*20+second["N_imag"]*23
    assert not math.isclose(second["primary_RZ_P0"], second["N_total"]*first["primary_RZ_P0"]/first["N_total"], rel_tol=1e-9)


def test_paired_covariance_positive_and_no_resampling():
    candidate = fixture(random=True)
    row = a.evaluate_candidate(candidate, 0.05)
    stat = candidate["statistics"]["rz_count"]
    nc, ns = row["N_real"], row["N_imag"]
    expected = math.sqrt((nc*nc*stat["cc"]+ns*ns*stat["ss"]+2*nc*ns*stat["cs"])/32)
    independent = math.sqrt((nc*nc*stat["cc"]+ns*ns*stat["ss"])/32)
    assert math.isclose(row["primary_SE"], expected, rel_tol=1e-12)
    assert row["primary_SE"] > independent
    assert len(candidate["pairs"]) == 32


def test_anticorrelated_pairs_cancel_exactly_without_negative_variance():
    candidate = fixture(random=True, bias={"real": 0, "imag": 0})
    for i, pair in enumerate(candidate["pairs"]):
        pair["cosine"] = {m: i for m in c.METRICS}
        pair["sine"] = {m: 31-i for m in c.METRICS}
    candidate["statistics"] = a.sample_statistics(candidate["pairs"], True)
    candidate["means"] = {axis: {m: 15.5 for m in c.METRICS} for axis in c.AXES.values()}
    assert a.evaluate_candidate(candidate, 0.05)["primary_SE"] == 0


def test_missing_random_samples_not_variance_zero():
    with pytest.raises(ValueError, match="missing cost samples"):
        a.sample_statistics(fixture(random=True)["pairs"][:-1], True)


def test_deterministic_cost_is_exact():
    assert a.evaluate_candidate(fixture(), 0.05)["primary_SE"] == 0


@pytest.mark.parametrize("field", ["candidate_fingerprint", "signal_record_fingerprint"])
def test_projection_identity_rejects_changed_fields(field):
    projected = fixture()
    signal = {"candidate_fingerprint": projected["candidate_fingerprint"], "axis_bias": projected["bias"],
              "normalization_multiplier": projected["B"], "axis_shots": projected["reference_axis_shots"], "accuracy_eligible": True}
    projected[field] = "changed"
    with pytest.raises(ValueError, match="identity|record"):
        a.project_candidate(projected, signal, projected["means"], projected["pairs"], projected["reference_work"])


def test_reference_gate_synthetic_positive():
    assert a.reference_gate([fixture(), fixture("bad-accuracy", bias={"real": 0.1, "imag": 0})])["passed"]


@pytest.mark.parametrize("field", ["reference_accuracy_eligible", "reference_axis_shots", "reference_work"])
def test_reference_gate_rejects_mismatch(field):
    row = fixture()
    if field == "reference_accuracy_eligible":
        row[field] = False
    elif field == "reference_axis_shots":
        row[field]["real"] += 1
    else:
        row[field]["rz_count"] += 1
    with pytest.raises(ValueError, match="reference"):
        a.reference_gate([row])


def test_reference_failure_precedes_epsilon_sweep(monkeypatch):
    row = fixture()
    row["reference_axis_shots"]["real"] += 1
    monkeypatch.setattr(a, "precision_points", lambda: pytest.fail("sweep reached before reference gate"))
    with pytest.raises(ValueError, match="reference"):
        a.analyze([row])


def line(name, intercept, slope, *, dataset="development"):
    return {"dataset": dataset, "epsilon": 0.05, "candidate_id": name, "accuracy_eligible": True,
            "N_total": slope, "primary_RZ_P0": intercept, **{m: intercept for m in c.METRICS}}


def test_pareto_strict_and_ties_kept():
    rows = [line("a",1,1),line("tie",1,1),line("dominated",2,1)]
    a.mark_frontier(rows)
    assert [r["candidate_id"] for r in rows if r["point_frontier"]] == ["a","tie"]


def test_envelope_crossing_and_unbounded_end():
    rows = [line("cheap",10,5), line("few-shots",20,1), line("dominated",40,10)]
    result = a.lower_envelope(rows)
    assert [(r["candidate_id"],r["P_min"],r["P_max"]) for r in result] == [("cheap",0,2.5),("few-shots",2.5,None)]
    assert all(r["boundary_tie"] for r in result)


def test_envelope_identical_lines_and_zero_width_ties():
    rows = [line("a",0,3),line("duplicate",0,3),line("middle",1,2),line("b",2,1)]
    result = {r["candidate_id"]: r for r in a.lower_envelope(rows)}
    assert result["middle"]["P_min"] == result["middle"]["P_max"] == 1
    assert result["duplicate"]["P_max"] == result["a"]["P_max"] == 1


def test_envelope_equal_slope_discards_only_higher_intercept():
    assert [r["candidate_id"] for r in a.lower_envelope([line("a",1,2),line("b",2,2)])] == ["a"]
    assert a.lower_envelope([]) == []


def test_negative_crossing_has_no_nonnegative_interval():
    result = a.lower_envelope([line("dominated",20,5),line("best",10,1)])
    assert len(result) == 1 and result[0]["candidate_id"] == "best"


def test_representatives_keep_all_minimum_ties():
    candidates = [fixture("a"),fixture("tie"),fixture("expensive",cost=40)]
    rows = [a.evaluate_candidate(candidate,0.05) for candidate in candidates]
    output = a.representatives(rows,{r["candidate_id"]:r for r in candidates})
    assert {r["candidate_id"] for r in output} == {"a","tie"}


def test_domains_remain_separate_even_with_identical_id(monkeypatch):
    monkeypatch.setattr(a,"precision_points",lambda:[0.05,0.1])
    candidates = [fixture("same",dataset="development"),fixture("same",dataset="transfer_fixed_five",cost=100)]
    tables = a.analyze(candidates)
    assert len(tables["precision_ledger.csv"]) == 4
    assert all(r["point_frontier"] for r in tables["precision_ledger.csv"])
    assert {r["dataset"] for r in tables["P_envelope.csv"]} == {"development","transfer_fixed_five"}
    a.validate_analysis_tables(tables,candidates)


def test_missing_ledger_record_rejected(monkeypatch):
    monkeypatch.setattr(a,"precision_points",lambda:[0.05])
    candidates=[fixture()]
    tables=a.analyze(candidates)
    tables["precision_ledger.csv"]=[]
    with pytest.raises(ValueError,match="coverage"):
        a.validate_analysis_tables(tables,candidates)


def test_csv_missing_and_exact_fields():
    text=a.render_csv([{"a":None,"b":0}], ["a","b"])
    assert list(csv.DictReader(io.StringIO(text))) == [{"a":"MISSING","b":"0"}]
    with pytest.raises(ValueError,match="fields"):
        a.render_csv([{"a":1}], ["a","b"])


def test_launch_flag_fails_before_any_input_or_git(monkeypatch,tmp_path):
    monkeypatch.setattr(c,"load_inputs",lambda *args:pytest.fail("real input boundary reached"))
    monkeypatch.setattr(a,"git",lambda *args:pytest.fail("git reached without launch"))
    with pytest.raises(ValueError,match="explicit"):
        a.run(tmp_path,"a"*40)
    assert not (tmp_path/a.OUTPUT).exists()


@pytest.mark.parametrize("source", [None,"short","g"*40])
def test_launch_requires_full_source_identity(source,tmp_path):
    with pytest.raises(ValueError,match="source commit"):
        a.validate_launch(tmp_path,source,True)


def test_launch_rejects_environment_before_git(monkeypatch,tmp_path):
    monkeypatch.delenv("OPENBLAS_NUM_THREADS",raising=False)
    monkeypatch.setattr(a,"git",lambda *args:pytest.fail("git reached before environment gate"))
    with pytest.raises(ValueError,match="environment"):
        a.validate_launch(tmp_path,"a"*40,True)


def test_launch_rejects_source_blob_change(monkeypatch,tmp_path):
    for name in a.SOURCE_FILES:
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text("synthetic source")
    monkeypatch.setattr(a,"git",lambda root,*args: b"changed" if args[0]=="show" else b"")
    with pytest.raises(ValueError,match="source differs"):
        a.validate_launch(tmp_path,"a"*40,True)


def mock_gate_tree(tmp_path, monkeypatch, settings=None):
    for name in a.SOURCE_FILES:
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text("synthetic frozen source")
    directory=tmp_path/a.PREPARATION;directory.mkdir(parents=True)
    settings=c.contract_settings() if settings is None else settings
    blob=json.dumps(settings).encode()
    (directory/"contract_settings_v1.json").write_bytes(blob)
    manifest=json.dumps({"files":[{"path":"contract_settings_v1.json","bytes":len(blob),"sha256":hashlib.sha256(blob).hexdigest()}]}).encode()
    (directory/"manifest.json").write_bytes(manifest)
    monkeypatch.setattr(a,"PREPARATION_SHA",hashlib.sha256(manifest).hexdigest())
    def git(root,*args):
        return (root/args[1].split(":",1)[1]).read_bytes() if args[0]=="show" else b""
    monkeypatch.setattr(a,"git",git)
    return directory


def test_synthetic_positive_source_preparation_gate(monkeypatch,tmp_path):
    mock_gate_tree(tmp_path,monkeypatch)
    audit=a.validate_launch(tmp_path,"a"*40,True)
    assert set(audit)==set(a.SOURCE_FILES)


def test_preparation_manifest_change_rejected(monkeypatch,tmp_path):
    directory=mock_gate_tree(tmp_path,monkeypatch)
    (directory/"manifest.json").write_text("{}")
    with pytest.raises(ValueError,match="manifest changed"):
        a.validate_launch(tmp_path,"a"*40,True)


def test_correctly_hashed_but_changed_contract_is_rejected(monkeypatch,tmp_path):
    settings=c.contract_settings();settings["accounting"]["alpha_axis"]=0.02
    mock_gate_tree(tmp_path,monkeypatch,settings)
    with pytest.raises(ValueError,match="contract settings"):
        a.validate_launch(tmp_path,"a"*40,True)


@pytest.mark.parametrize("source",["m1_signal","pm1","m2"])
def test_saved_format_projection_on_synthetic_records(source):
    candidate=fixture()
    signal={"candidate_fingerprint":candidate["candidate_fingerprint"],"axis_bias":candidate["bias"],
        "normalization_multiplier":candidate["B"],"axis_shots":candidate["reference_axis_shots"],"accuracy_eligible":True}
    entry={k:candidate[k] for k in ("dataset","candidate_id","candidate_fingerprint","parameters","signal_record_fingerprint","reference_accuracy_eligible")}
    entry.update(signal_source=source,signal_row_index=1 if source=="m1_signal" else 0,compile_row_index=0)
    axes={axis:{"retained_trajectory_records":[{"trajectory_index":0,"trajectory_seed":None,"cost":candidate["pairs"][0][axis]}],
        "metric_statistics":{m:{"mean":candidate["means"][axis][m]} for m in c.METRICS}} for axis in c.AXES.values()}
    if source=="m1_signal":
        values={"m1_signal":{"signal_records":[{},signal]},"m1_compile":{"compile_map":[{"compiled_axes":axes,"matched_accuracy_compiled_work_no_state_preparation":candidate["reference_work"]}]}}
    elif source=="pm1":
        values={"pm1":{"candidate_records":[{"signal":signal,"axes":axes,"work":candidate["reference_work"]}]}}
    else:
        values={"m2":{"candidate_results":[{"signal":signal,"axis_one_shot_compiled_means":candidate["means"],"work_by_metric":candidate["reference_work"],
            "compiled":{"paired_trajectory_rows":[{"axes":{axis:{"metrics":candidate["pairs"][0][axis]} for axis in c.AXES.values()}}]}}]}}
    projected=a.project_inputs(values,{"development":[entry]})
    assert a.reference_gate(projected)["passed"] and projected[0]["candidate_id"]==candidate["candidate_id"]


def test_precision_cap_not_extended(monkeypatch):
    monkeypatch.setattr(a,"precision_points",lambda:[0.05]*303)
    with pytest.raises(ValueError,match="cap exceeded"):
        a.analyze([fixture()])


def summary_fixture():
    return {"schema_version":"track_a_pm2_precision_result_v1","status":c.COMPLETE,
        "analysis_label":"POSTHOC_SAVED_VALUES_ONLY","preparation_manifest_sha256":a.PREPARATION_SHA,
        "input_identity":{key:{} for key in c.INPUTS},"domain_counts":{"development":218,"transfer_fixed_five":5},
        "reference_reproduction":{"passed":True},"output_files":[f for f in c.contract_settings()["future_outputs"] if f not in {"summary.json","manifest.json"}],
        "failure_reason":None,"new_science_counts":dict(a.ZERO_SCIENCE),"mandatory_stop":True,"next_stage_authorized":False,"research_decision":None}


def test_terminal_summary_agrees_with_reserved_schema():
    summary=summary_fixture()
    a.validate_summary(summary)
    jsonschema.validate(summary,c.reserved_result_schema())
    summary.update(status=c.FAILURE,failure_reason="synthetic failure",reference_reproduction=None,input_identity={},output_files=[])
    a.validate_summary(summary)
    jsonschema.validate(summary,c.reserved_result_schema())


@pytest.mark.parametrize("field,value",[("status","CONTINUE_RESOURCE_STUDY"),("mandatory_stop",False),
    ("next_stage_authorized",True),("research_decision","GO"),("reference_reproduction",{"passed":False}),
    ("output_files",[]),("input_identity",{})])
def test_terminal_invariants_reject_changes(field,value):
    summary=summary_fixture();summary[field]=value
    with pytest.raises(ValueError):a.validate_summary(summary)


def test_terminal_new_science_rejected():
    summary=summary_fixture();summary["new_science_counts"]["compile"]=1
    with pytest.raises(ValueError,match="science"):a.validate_summary(summary)


def test_failure_output_is_a_failure_not_negative_science(monkeypatch,tmp_path):
    monkeypatch.setattr(a,"validate_launch",lambda *args:{})
    def reject(*args):raise ValueError("synthetic input mismatch")
    monkeypatch.setattr(c,"load_inputs",reject)
    result=a.run(tmp_path,"a"*40,execute_saved_analysis=True)
    assert result["status"]==c.FAILURE and result["research_decision"] is None
    manifest=json.loads((tmp_path/a.OUTPUT/"manifest.json").read_text())
    assert manifest["partial_output_must_not_be_used"] is True
    assert result["reference_reproduction"] is None and "mismatch" in result["failure_reason"]


def test_existing_output_not_overwritten(monkeypatch,tmp_path):
    monkeypatch.setattr(a,"validate_launch",lambda *args:{})
    (tmp_path/a.OUTPUT).mkdir(parents=True)
    with pytest.raises(ValueError,match="already exists"):
        a.run(tmp_path,"a"*40,execute_saved_analysis=True)


def test_full_synthetic_runner_outputs_only_fixed_artifacts(monkeypatch,tmp_path):
    monkeypatch.setattr(a,"validate_launch",lambda *args:{"synthetic":"identity"})
    monkeypatch.setattr(c,"load_inputs",lambda *args:({}, {key:{} for key in c.INPUTS}))
    monkeypatch.setattr(c,"candidate_inventory",lambda values:{"synthetic":[]})
    candidates=[fixture(str(i),dataset="development" if i<218 else "transfer_fixed_five") for i in range(223)]
    monkeypatch.setattr(a,"project_inputs",lambda *args:candidates)
    monkeypatch.setattr(a,"precision_points",lambda:[0.05])
    prep=tmp_path/a.PREPARATION;prep.mkdir(parents=True)
    (prep/"candidate_inventory_v1.json").write_text(json.dumps({"inventory_fingerprint":c.fingerprint({"synthetic":[]})}))
    result=a.run(tmp_path,"a"*40,execute_saved_analysis=True)
    assert result["status"]==c.COMPLETE, result["failure_reason"]
    jsonschema.validate(result,c.reserved_result_schema())
    output=tmp_path/a.OUTPUT
    assert {p.name for p in output.iterdir()}==set(c.contract_settings()["future_outputs"])
    manifest=json.loads((output/"manifest.json").read_text())
    for record in manifest["files"]:
        data=(output/record["path"]).read_bytes()
        assert len(data)==record["bytes"] and hashlib.sha256(data).hexdigest()==record["sha256"]
    assert result["next_stage_authorized"] is False and result["research_decision"] is None


def test_report_states_uncertainty_and_no_general_claim(monkeypatch):
    monkeypatch.setattr(a,"precision_points",lambda:[0.05])
    text=a.report(a.analyze([fixture("sample",random=True)]))
    assert "not a formal CI" in text and "overlapping" in text
    assert "Mandatory STOP" in text and "No general optimum" in text


def test_stdlib_only_no_science_import_or_cached_effective_cost():
    tree=ast.parse(Path(a.__file__).read_text())
    imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    imports += [name.name for n in ast.walk(tree) if isinstance(n,ast.Import) for name in n.names]
    assert not any(name and name.startswith(("numpy","scipy","qiskit","cupy","trotterlib")) for name in imports)
    assert set(a.ZERO_SCIENCE)==set(c.ZERO_ACTIONS[4:])
