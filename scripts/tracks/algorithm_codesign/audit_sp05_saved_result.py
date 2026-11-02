#!/usr/bin/env python3
"""Audit SP05 saved fields with stdlib only; no synthesis, PAI, J, or guard rerun."""
from collections import Counter
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
PREP = ROOT / "artifacts/track_b_sp05_economics_preparation/2026-10-06"
RESULT = ROOT / "artifacts/track_b_sp05_economics_result/2026-10-06/v1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def key(a):
    return f'{a["unit"]}:{a["numerator"]}/{a["denominator"]}'


def scaled(a, sign):
    f = Fraction(a["numerator"], a["denominator"])*Fraction(sign,2)
    return dict(unit=a["unit"],numerator=f.numerator,denominator=f.denominator)


def endpoints(b):
    lo,hi=Decimal(b["lo"]),Decimal(b["hi"])
    assert lo.is_finite() and hi.is_finite() and lo <= hi
    return lo,hi


def audit_saved_relations(rows):
    """Exact rational enclosure checks on saved fields; no new J evaluation."""
    def band(b):
        return tuple(Fraction(Decimal(b[k])) for k in ("lo","hi"))
    def mul(a,b):
        assert a[0]>=0 and b[0]>=0
        return a[0]*b[0],a[1]*b[1]
    def overlap(a,b):
        return max(a[0],b[0])<=min(a[1],b[1])
    passed=[]
    for row in rows:
        moment,mean=(Fraction(1),Fraction(1)),(Fraction(0),Fraction(0))
        for interp,costs in zip(row["interpolations"],row["notch_T_counts"]):
            gamma=band(interp["gamma"])
            moment=mul(moment,mul(gamma,gamma))
            ps=[band(p) for p in interp["p"]]
            gs=[band(g) for g in interp["g"]]
            assert sum(p[0] for p in ps)<=1<=sum(p[1] for p in ps)
            assert sum(g[0] for g in gs)<=1<=sum(g[1] for g in gs)
            for p,g in zip(ps,gs):
                absolute=(Fraction(0) if g[0]<=0<=g[1] else min(abs(g[0]),abs(g[1])),max(abs(g[0]),abs(g[1])))
                assert overlap(mul(p,gamma),absolute)
            mean=(mean[0]+sum(p[0]*c for p,c in zip(ps,costs)),
                  mean[1]+sum(p[1]*c for p,c in zip(ps,costs)))
        if row["deterministic_T_count"]:
            assert overlap(moment,band(row["weight_second_moment"]))
            assert overlap(mean,band(row["expected_T_count"]))
            # Compare stored J*Cdet and stored moment*mean; no ratio or score is generated.
            det=Fraction(row["deterministic_T_count"])
            assert overlap(mul(band(row["J"]),(det,det)),
                           mul(band(row["weight_second_moment"]),band(row["expected_T_count"])))
        else:
            assert moment==(1,1) and mean==(0,0) and row["J"] is None
        passed.append({"target_id":row["target_id"],"primitive":row["primitive"],
                       "saved_field_relations_pass":True})
    return {"status":"SAVED_INTERVAL_RELATIONS_PASSED","rows":passed,
            "method":"exact rational enclosure overlaps for saved g/p/gamma, moment, mean cost, and J*Cdet; no synthesis/trigonometry/division/objective calls",
            "new_J_evaluation":False,"primary_rows_modified":False,"science_rerun":False}


def main():
    c,a,r,m,p,e = (load(path) for path in (
        PREP/"contract_v1.json",PREP/"authorization.json",RESULT/"result.json",
        RESULT/"one_shot_consumed.json",RESULT/"prelaunch.json",RESULT/"execution_receipt.json"))
    source,auth=r["source_commit"],r["authorization_commit"]
    assert source==a["source_commit"]==m["source_commit"]==p["source_commit"]==e["source_commit"]
    assert auth==m["authorization_commit"]==p["authorization_commit"]==e["authorization_commit"]
    parents=subprocess.check_output(["git","-C",str(ROOT),"show","-s","--format=%P",auth],text=True).split()
    changed=set(subprocess.check_output(["git","-C",str(ROOT),"diff","--name-only",source,auth],text=True).splitlines())
    assert parents==[source]
    assert changed=={c["authorization_path"],c["optional_receipt_path"]}
    assert a["status"]=="APPROVED_FOR_ONE_SP05_RUN" and a["science_execution_authorized"] is True
    assert a["runs"]==1 and a["retries"]==0 and a["mandatory_STOP"] is True
    assert c["wrapper_pilot_authorized"] is False and a["wrapper_pilot_authorized"] is False
    assert r["wrapper_pilot_authorized"] is False and r["retry"] is False
    assert e["runner_run_invocations"]==1 and e["registered_gate_runs"]==1
    assert e["retries"]==0 and e["subsequent_science_execution"] is False
    assert e["mandatory_STOP_reached"] is True and r["mandatory_STOP"] is True
    assert a["contract_sha256"]==r["contract_sha256"]==m["contract_sha256"]==p["contract_sha256"]==digest(PREP/"contract_v1.json")
    assert p["authorization_sha256"]==digest(PREP/"authorization.json")
    assert p["tool_identity_sha256"]==digest(PREP/"tool_identity_v1.json")
    assert p["source_hashes_verified"]==11 and p["runtime_identity_verified"] is True
    assert p["clean_direct_child_verified"] is True and p["marker_absent"] is True
    manifest=load(PREP/"preparation_manifest_v1.json")
    assert all(digest(ROOT/f)==h for f,h in manifest["source_sha256"].items())
    instruction=a["explicit_execution_instruction"]
    receipt=(ROOT/a["review_receipt"]["path"]).read_text()
    assert receipt.split("<!-- BEGIN USER EXECUTION INSTRUCTION -->\n",1)[1].split("\n<!-- END USER EXECUTION INSTRUCTION -->",1)[0]==instruction
    assert hashlib.sha256(instruction.encode()).hexdigest()==a["explicit_execution_instruction_sha256"]
    assert digest(ROOT/a["review_receipt"]["path"])==a["review_receipt"]["sha256"]

    synth=r["synthesis_rows"]
    assert len(synth)==len(p["planned_keys"])==23
    assert [x["key"] for x in synth]==p["planned_keys"]
    assert len({x["key"] for x in synth})==23
    by_key={x["key"]:x for x in synth}
    for row in synth:
        seq=row["sequence"]
        assert row["key"]==key(row["angle"])
        assert set(seq)<=set("HTtSXW")
        assert hashlib.sha256(seq.encode()).hexdigest()==row["sequence_sha256"]
        assert seq.count("T")+seq.count("t")==row["T_count"]
        assert sum(seq.count(g) for g in "HSX")==row["Clifford_count"]
        assert seq.count("W")==row["scalar_W_count"]
        guard=row["error_guard"]
        assert row["error_pass"] is True
        assert Decimal(guard["projective_operator_upper"])<=Decimal(c["operator_epsilon"])
        assert Decimal(guard["channel_diamond_upper"])<=2*Decimal(c["operator_epsilon"])
        assert 0<=guard["phase_witness_pi_over_8"]<16
        assert row["cpu_seconds"]<=c["caps"]["per_key_cpu_seconds"]
        assert row["wall_seconds"]<=c["caps"]["per_key_wall_seconds"]
        assert len(seq)<=c["caps"]["sequence_characters"]

    rows=r["economics_rows"]
    target_by_id={x["id"]:x["angle"] for x in c["targets"]}
    assert len(rows)==16 and len(target_by_id)==8
    assert {(x["target_id"],x["primitive"]) for x in rows}=={
        (t,k) for t in target_by_id for k in ("ordinary_Rz","controlled_pair")}
    for row in rows:
        natives=([target_by_id[row["target_id"]]] if row["primitive"]=="ordinary_Rz" else
                 [scaled(target_by_id[row["target_id"]],1),scaled(target_by_id[row["target_id"]],-1)])
        assert row["native_angles"]==natives
        assert len(row["interpolations"])==len(natives)
        det=sum(by_key[key(x)]["T_count"] for x in natives)
        assert det==row["deterministic_T_count"]
        for interpolation,costs in zip(row["interpolations"],row["notch_T_counts"]):
            assert len(interpolation["g"])==len(interpolation["p"])==len(costs)==3
            assert costs==[by_key[key(c["catalogue"]["angles"][k])]["T_count"] for k in interpolation["notch_indices"]]
            for b in interpolation["g"]+interpolation["p"]+[interpolation["gamma"]]:
                endpoints(b)
        if det==0:
            assert row["J"] is None and row["classification"]=="ZERO_COST_BASELINE_NO_STRICT_GAIN"
        else:
            lo,hi=endpoints(row["J"])
            assert (hi<1 if row["classification"]=="STRICT_TRADEOFF" else
                    lo>=1 if row["classification"]=="NO_STRICT_TRADEOFF" else lo<1<=hi)
            endpoints(row["expected_T_count"]); endpoints(row["weight_second_moment"])

    labels=Counter(x["classification"] for x in rows)
    # Validate the stored classification predicates; never call an objective or alter a row.
    assert r["status"]=="PRIMITIVE_TRADEOFF_EXISTS" and labels["STRICT_TRADEOFF"]>=1
    assert "failure" not in r
    assert r["wall_seconds"]<c["caps"]["total_wall_seconds"]
    assert r["worker_cpu_seconds"]+r["parent_cpu_seconds"]<c["caps"]["total_cpu_seconds"]
    assert (RESULT/"result.json").stat().st_size<c["caps"]["output_bytes"]
    resource_text=(RESULT/"process_resources.txt").read_text()
    assert "Exit status: 0" in resource_text
    peak_kib=int(next(line.split(":",1)[1] for line in resource_text.splitlines() if "Maximum resident set size" in line))
    # At most one parent and one worker: 2*max individual RSS is a conservative combined upper.
    assert 2*peak_kib<c["caps"]["combined_RSS_MiB"]*1024
    assert all(row["peak_RSS_KiB"]<=peak_kib for row in synth)
    zero=next(row for row in rows if row["deterministic_T_count"]==0)
    assert zero["interpolations"][0]["gamma"]=={"lo":"1","hi":"1"}
    assert zero["interpolations"][0]["p"]==[{"lo":"1","hi":"1"},{"lo":"0","hi":"0"},{"lo":"0","hi":"0"}]
    assert zero["notch_T_counts"][0][0]==0
    audit={"status":"SAVED_RESULT_AUDIT_PASSED","source_commit":source,"authorization_commit":auth,
        "synthesis_keys":23,"primitive_rows":16,"classification_counts":dict(labels),
        "synthesis_routes":dict(Counter(x["route"] for x in synth)),
        "source_and_authorization_identity_pass":True,"all_sequence_count_guard_saved_fields_pass":True,
        "all_row_coverage_and_saved_classification_predicates_pass":True,
        "error_guard_recomputed":False,"PAI_or_J_objective_called":False,
        "science_rerun":False,"primary_classification_changed":False,
        "zero_cost_baseline_supplement":{"target_id":zero["target_id"],"primitive":zero["primitive"],
            "original_J":None,"original_moment_and_expected_cost_fields":"omitted by source early return",
            "exact_moment_from_saved_single_branch":"1","exact_expected_T_count_from_saved_single_branch":"0",
            "derivation":"saved gamma=1, p=(1,0,0), active notch T=0; descriptive supplement only"},
        "runtime":{"runner_wall_seconds":r["wall_seconds"],"worker_cpu_seconds":r["worker_cpu_seconds"],
            "parent_cpu_seconds":r["parent_cpu_seconds"],"max_individual_RSS_KiB":peak_kib,
            "conservative_combined_RSS_upper_KiB":2*peak_kib,"cap_hit":False,
            "max_saved_operator_error_upper":str(max(Decimal(x["error_guard"]["projective_operator_upper"]) for x in synth))},
        "numeric_inconclusive_rows":labels["NUMERIC_INCONCLUSIVE"],"runtime_error_numeric_failure":None,
        "literal_keys_include_rational_duplicate":{"keys":["pi:1/2","pi:2/4"],
            "note":"source key policy preserved; 23 registered literal keys are not 23 independent physical angles"},
        "runner_run_invocations":1,"retries":0,"mandatory_STOP":True,"wrapper_pilot_authorized":False,
        "result_sha256":digest(RESULT/"result.json"),"marker_sha256":digest(RESULT/"one_shot_consumed.json")}
    saved_audit=RESULT/"saved_value_audit_v1.json"
    if saved_audit.exists():
        assert load(saved_audit)==audit, "published audit differs from checked saved fields"
    else:
        with saved_audit.open("x") as f:
            json.dump(audit,f,indent=2); f.write("\n")
    relations=RESULT/"saved_interval_relations_v1.json"
    if relations.exists():
        assert load(relations)==audit_saved_relations(rows)
    print(json.dumps({k:audit[k] for k in ("status","synthesis_keys","primitive_rows","classification_counts","retries","mandatory_STOP")},indent=2))


if __name__=="__main__":
    main()
