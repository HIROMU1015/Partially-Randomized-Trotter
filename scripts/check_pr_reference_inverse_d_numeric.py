"""Exact post-run bookkeeping, separate D1/D2 checks. No runner invocation."""
from collections import Counter,defaultdict
from fractions import Fraction as F
import ast
import hashlib
import json
import lzma
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
BASE="06895797e032bc236562da5ee541b33ce8eb5050"
SOURCE="e8cb9351704219297b6c46321fa9b063f14eeddd"
AREA=ROOT/"artifacts/pr_reference_inverse_d/2026-11-08/numeric"


def sha(raw):return hashlib.sha256(raw).hexdigest()


def blob(commit,path):return subprocess.check_output(["git","show",commit+":"+path],cwd=ROOT)


def read(path):return json.loads(blob(BASE,path))


def exact_functions():
    old="scripts/check_pr_reference_frame_impact_numeric.py"
    raw=blob(BASE,old)
    node=next(n for n in ast.walk(ast.parse(raw)) if isinstance(n,ast.FunctionDef) and n.name=="check_witness")
    ns=dict(F=F);exec(compile(ast.Module(body=[node],type_ignores=[]),old,"exec"),ns)
    raw2=blob(SOURCE,"scripts/diagnose_pr_reference_inverse_d.py")
    node2=next(n for n in ast.parse(raw2).body if isinstance(n,ast.FunctionDef) and n.name=="d_multiplicities")
    exec(compile(ast.Module(body=[node2],type_ignores=[]),"frozen_d_coefficients","exec"),ns)
    return ns,dict(witness_checker_SOURCE=old,witness_checker_sha256=sha(raw),
        d_coefficients_SOURCE_sha256=sha(raw2))


def check_attribution(arm,attr,details,expected,coefficients):
    grouped=defaultdict(F)
    assert {(t["mapping"],t["template"]) for t in attr["terms"]}==set(coefficients)
    for term in attr["terms"]:
        mapping,name=term["mapping"],term["template"]
        assert term["SOURCE_multiplicity"]==coefficients[(mapping,name)]
        contexts=[r for r in details if r["mapping"]==mapping and r["template"]==name]
        maximum=max(F(r["logical_guard"]) for r in contexts)
        ties=sorted(r["context"] for r in contexts if F(r["logical_guard"])==maximum)
        assert term["tied_contexts"]==ties and term["selected_context"]==ties[0]
        assert F(term["template_maximum"])==maximum
        chosen=next(r for r in contexts if r["context"]==ties[0])
        assert term["guard_contributions"]==chosen["target_key_contributions"]
        assert sum((F(r["guard"]) for r in term["guard_contributions"].values()),F(0))==maximum
        for key,item in term["guard_contributions"].items():
            grouped[key]+=coefficients[(mapping,name)]*F(item["guard"])
    assert dict(grouped)=={k:F(v) for k,v in attr["key_upper_contributions"].items()}
    assert F(attr["D_upper"])==sum(grouped.values(),F(0))==F(expected)
    assert attr["keys_by_decreasing_guard"]==sorted(grouped,key=lambda k:(-grouped[k],k))


def main():
    functions,reuse=exact_functions()
    gf=read("artifacts/pr_reference_frame_impact/2026-11-08/numeric/RESULTS.json")
    fixed=next(r for r in gf["policies"] if r["policy"]=="G")
    frontier={r["target_literal"]:r for r in read("artifacts/pr_reference_noncentral_saved_numeric/2026-11-08/RESULTS.json")["pareto"]}
    expected_keys=json.loads(lzma.decompress(blob(SOURCE,
        "artifacts/pr_reference_inverse_d/2026-11-08/STATIC_BINDINGS.json.xz")))["fixed_keysets"]
    for mode in ("D1","D2"):
        out=AREA/mode
        assert not (out/"CHECKS.json").exists()
        inp=json.loads((out/"INPUTS.json").read_text())
        raw=(out/"RESULTS.json").read_bytes() if (out/"RESULTS.json").exists() else lzma.decompress((out/"RESULTS.json.xz").read_bytes())
        data=json.loads(raw)
        assert inp["base_commit"]==BASE and data["mode"]==mode
        assert data["status"]=="FINITE_INVERSE_D_DIAGNOSTIC_COMPLETE"
        assert json.loads((out/"STOP.json").read_text())["status"]=="MANDATORY_STOP"
        assert not (out/"FAILURE.json").exists()
        env=json.loads((out/"ENVIRONMENT.json").read_text())
        assert env["SOURCE_commit"]==SOURCE
        for path,field in (("scripts/diagnose_pr_reference_inverse_d.py","SOURCE_sha256"),
                           ("scripts/audit_pr_reference_inverse_d_static.py","static_helper_sha256")):
            b=blob(SOURCE,path);assert b==(ROOT/path).read_bytes() and sha(b)==inp[field]
            assert env["SOURCE_sha256"][path]==inp[field]
        for path,info in {**inp["saved_inputs"],**inp["legacy_inputs"]}.items():
            b=blob(BASE,path);assert len(b)==info["bytes"] and sha(b)==info["sha256"]
        assert inp["fixed_G_forward_choices"]==fixed["choices"]
        assert data["baseline_G_rows_quoted"]==fixed["rows"]
        assert set(inp["targets"])==set(expected_keys["D1_inverse" if mode=="D1" else "D2_D"])
        assert inp["shared_keys"]==expected_keys["inverse_D_shared"]
        assert not inp["role_dependent_word_selector"] and not inp["old_G_aggregate_recomputed"]
        previous={(r["mapping"],r["template"],r["context"]):r for r in fixed["context_details"]}
        current={(r["mapping"],r["template"],r["context"]):r for r in data["baseline_details"]}
        assert set(current)==set(previous) and len(current)==180
        for key,row in current.items():
            assert row["logical_guard"]==previous[key]["logical_guard"]
            assert row["local_template_T_from_saved_prices"]==previous[key]["local_template_T_from_saved_prices"]
        baseline={(r["arm"],r["axis"]):r for r in fixed["rows"]}
        if mode=="D2":
            for arm,attr in data["D_baseline_attribution"].items():
                check_attribution(arm,attr,data["baseline_details"],baseline[(arm,"X")]["parts"]["D"],functions["d_multiplicities"](131072,arm))
        summary=[];seen=set()
        for p in data["policies"]:
            if "alias" in p:
                assert p["alias"] in seen;summary.append(dict(policy=p["policy"],alias=p["alias"]));continue
            seen.add(p["policy"])
            assert set(p["choices"])==set(inp["targets"])
            changes=Counter()
            for key,c in p["choices"].items():
                offers={r["id"]:r for r in frontier[key]["offers"]}
                assert c["record_id"] in frontier[key]["nondominated_ids"] or (p["policy"]=="NT" and c["record_id"]=="fixed_current")
                new,old=offers[c["record_id"]],offers["fixed_current"]
                assert c["sequence_sha256"]==new["sequence_sha256"]
                assert F(c["saved_guard"])==F(new["guard"]) and c["saved_T"]==new["T"]
                assert F(c["guard_change"])==F(new["guard"])-F(old["guard"])
                assert c["T_change_per_occurrence"]==new["T"]-old["T"]
                changes["guard_down" if F(c["guard_change"])<0 else "guard_up" if F(c["guard_change"])>0 else "guard_same"]+=1
                changes["T_down" if c["T_change_per_occurrence"]<0 else "T_up" if c["T_change_per_occurrence"]>0 else "T_same"]+=1
            assert len(p["context_details"])==180
            for row in p["context_details"]:
                old=current[(row["mapping"],row["template"],row["context"])]
                before,after=old["target_key_contributions"],row["target_key_contributions"]
                assert set(before)==set(after)
                assert all(before[k]["multiplicity"]==after[k]["multiplicity"] for k in before)
                assert F(row["logical_guard"])-F(old["logical_guard"])==sum((F(after[k]["guard"])-F(before[k]["guard"]) for k in before),F(0))
                assert row["local_template_T_from_saved_prices"]-old["local_template_T_from_saved_prices"]==sum(after[k]["local_T"]-before[k]["local_T"] for k in before)
            for row in p["rows"]:
                old=baseline[(row["arm"],row["axis"])]
                assert sum(map(F,row["parts"].values()),F(0))==F(row["noncentral_upper"])
                assert F(row["residual"])==F(row["saved_joint_budget"])-F(row["noncentral_upper"])
                assert row["saved_joint_budget"]==old["saved_joint_budget"]
                assert all(F(v)==F(row["parts"][k])-F(old["parts"][k]) for k,v in row["parts_change_from_fixed_G"].items())
                assert row["m17_RPE"]=="UNQUALIFIED" and row["domain_central_emitter_shell_terminal"]=="UNQUALIFIED_NOT_ZERO"
                functions["check_witness"](row["arm"],row["frame_witness"],p["context_details"],row["parts"]["frames"])
                if mode=="D2":check_attribution(row["arm"],p["D_attribution"][row["arm"]],p["context_details"],row["parts"]["D"],functions["d_multiplicities"](131072,row["arm"]))
            assert not p["new_policy_adopted"]
            summary.append(dict(policy=p["policy"],key_changes=dict(changes)))
        assert data["full_RPE"]=="UNQUALIFIED" and data["global_T"]=="NOT_EVALUATED"
        assert not data["frontier_recomputed"] and not data["scientific_word_policy_adopted"]
        compressed=(out/"RESULTS.json.xz").read_bytes()
        assert lzma.decompress(compressed)==raw
        checks=dict(status="EXACT_POSTRUN_CHECKS_PASS",mode=mode,SOURCE_commit=SOURCE,
            checker_SOURCE_sha256=sha(Path(__file__).read_bytes()),reused_checking_SOURCE=reuse,
            fixed_SOURCE_inputs_and_G_baseline_match=True,scope_independent=True,
            contexts=180,target_keys=len(inp["targets"]),policies=summary,
            exact_parts_residuals_local_T_and_witnesses=True,
            D_attribution_verified=mode=="D2",no_new_science_runner_calls=True,
            raw_RESULTS_sha256=sha(raw),raw_RESULTS_bytes=len(raw),
            published_RESULTS_xz_sha256=sha(compressed),published_RESULTS_xz_bytes=len(compressed),
            compressed_copy_exactly_preserves_raw_bytes=True,
            environment_sha256=sha((out/"ENVIRONMENT.json").read_bytes()),
            author_checks_not_independent_scientific_review=True,STOP="MANDATORY_STOP")
        (out/"CHECKS.json").write_text(json.dumps(checks,sort_keys=True,indent=2)+"\n")
        print(json.dumps(checks))


if __name__=="__main__":main()
