"""Post-run exact bookkeeping checks only; never invoke the science runner."""
from collections import Counter
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"artifacts/pr_reference_frame_impact/2026-11-08/numeric"
SOURCE_COMMIT="4ebbc3458e13fe0fa6a03f1e2b3bd82e6c3d686e"
RUNNER="scripts/diagnose_pr_reference_frame_impact.py"


def sha(raw):return hashlib.sha256(raw).hexdigest()


def main():
    assert not (OUT/"CHECKS.json").exists()
    inp=json.loads((OUT/"INPUTS.json").read_text())
    data=json.loads((OUT/"RESULTS.json").read_text())
    stop=json.loads((OUT/"STOP.json").read_text())
    assert data["status"]=="FINITE_FRAME_SAVED_DIAGNOSTIC_COMPLETE"
    assert stop["status"]=="MANDATORY_STOP"
    raw=subprocess.check_output(["git","show",SOURCE_COMMIT+":"+RUNNER],cwd=ROOT)
    assert raw==(ROOT/RUNNER).read_bytes()
    assert sha(raw)==inp["runner_sha256"]==data["SOURCE"]
    assert inp["base_commit"]=="340670387dbe58bdda5606fdf222408774be1932"
    fixed={**inp["saved_inputs"],**inp["legacy_inputs"]}
    for path,info in fixed.items():
        value=subprocess.check_output(["git","show",inp["base_commit"]+":"+path],cwd=ROOT)
        assert sha(value)==info["sha256"] and len(value)==info["bytes"]
    def read_input(path):
        return json.loads(subprocess.check_output(["git","show",inp["base_commit"]+":"+path],cwd=ROOT))
    saved=read_input("artifacts/pr_reference_noncentral_s1_coverage_fix/2026-11-08/science/run01/RESULTS.json")
    quoted=[r for r in saved["rows"] if r["variant"]=="weighted_SOURCE_RTE_basis"]
    assert quoted==data["baseline_rows_quoted"]
    baseline={(r["arm"],r["axis"]):r for r in quoted}
    frontier={r["target_literal"]:r for r in read_input(
        "artifacts/pr_reference_noncentral_saved_numeric/2026-11-08/RESULTS.json")["pareto"]}
    current={(r["mapping"],r["template"],r["context"]):r for r in data["current_context_details"]}
    assert len(current)==len(data["current_context_details"])
    counts=Counter(r["mapping"] for r in current.values())
    def check_witness(arm,witness,details,frame):
        mapping="b" if arm=="Bprime" else "ref"
        per=witness["per_template_context_witnesses"]
        for name,item in per.items():
            rows=[r for r in details if r["mapping"]==mapping and r["template"]==name]
            assert rows and all(r["logical_guard"] is not None for r in rows)
            assert max(F(r["logical_guard"]) for r in rows)==F(item["maximum"])
            assert item["tied_contexts"]==[r["context"] for r in rows if F(r["logical_guard"])==F(item["maximum"])]
        pairs=witness["per_basis_pair"]
        assert len(pairs)==7
        for bid,value in pairs.items():
            assert F(value)==F(per["basis:"+bid]["maximum"])+F(per["basis_inverse:"+bid]["maximum"])
        assert F(witness["pair_maximum"])==max(map(F,pairs.values()))
        assert witness["tied_basis_ids"]==[bid for bid in pairs if F(pairs[bid])==F(witness["pair_maximum"])]
        assert F(frame)==3*131072*32*F(witness["pair_maximum"])
    for arm,w in data["current_witness"].items():
        check_witness(arm,w,list(current.values()),baseline[(arm,"X")]["parts"]["frames"])
    signatures={};summary=[];evaluated=0
    for policy in data["policies"]:
        mode=policy["policy"]
        if "identical_to_policy" in policy:
            assert policy["identical_to_policy"] in signatures
            assert mode=="NT" and policy["identical_to_policy"]=="G"
            summary.append(dict(policy=mode,alias_of=policy["identical_to_policy"]));continue
        evaluated+=1
        assert set(policy["choices"])==set(inp["targets"]) and len(policy["choices"])==43
        key_changes=Counter()
        for key,choice in policy["choices"].items():
            pool=frontier[key]
            assert choice["record_id"] in pool["nondominated_ids"] or (mode=="NT" and choice["record_id"]=="fixed_current")
            offers={r["id"]:r for r in pool["offers"]}
            item,old=offers[choice["record_id"]],offers["fixed_current"]
            assert choice["sequence_sha256"]==item["sequence_sha256"]
            assert F(choice["guard"])==F(item["guard"]) and choice["T_saved"]==item["T"]
            assert F(choice["guard_change"])==F(item["guard"])-F(old["guard"])
            assert choice["T_change_per_occurrence"]==item["T"]-old["T"]
            key_changes["guard_lower" if F(choice["guard_change"])<0 else "guard_equal_or_higher"]+=1
            key_changes["T_lower" if choice["T_change_per_occurrence"]<0 else "T_equal_or_higher"]+=1
        signatures[mode]=tuple(policy["choices"][k]["record_id"] for k in sorted(policy["choices"]))
        assert len(policy["context_details"])==len(current)
        affected=0
        for row in policy["context_details"]:
            old=current[(row["mapping"],row["template"],row["context"])]
            before,after=old["target_key_contributions"],row["target_key_contributions"]
            assert set(before)==set(after)
            assert all(before[k]["multiplicity"]==after[k]["multiplicity"] for k in before)
            assert F(row["logical_guard"])-F(old["logical_guard"])==sum(
                (F(after[k]["guard"])-F(before[k]["guard"]) for k in before),F(0))
            assert row["local_template_T_from_saved_prices"]-old["local_template_T_from_saved_prices"]==sum(
                after[k]["local_T"]-before[k]["local_T"] for k in before)
            if before:
                affected+=1;assert row["template"].startswith("basis:")
        for row in policy["rows"]:
            old=baseline[(row["arm"],row["axis"])]
            assert F(row["noncentral_upper"])==sum(map(F,row["parts"].values()),F(0))
            assert F(row["residual"])==F(row["saved_joint_budget"])-F(row["noncentral_upper"])
            assert F(row["saved_joint_budget"])==F(old["saved_joint_budget"])
            assert F(row["frame_change_from_saved"])==F(row["parts"]["frames"])-F(old["parts"]["frames"])
            for role in ("D","products","CSWAP"):assert row["parts"][role]==old["parts"][role]
            assert row["m17_RPE"]=="UNQUALIFIED"
            assert row["central_domain_emitter_shell_terminal"]=="UNQUALIFIED_NOT_ZERO"
            check_witness(row["arm"],row["frame_witness"],policy["context_details"],row["parts"]["frames"])
        assert not policy["new_policy_adopted"]
        summary.append(dict(policy=mode,contexts=len(current),affected_contexts=affected,key_changes=dict(key_changes)))
    assert data["full_RPE"]=="UNQUALIFIED" and data["global_T"]=="NOT_EVALUATED"
    assert not data["frontier_recomputed"]
    checks=dict(status="EXACT_POSTRUN_CHECKS_PASS",SOURCE_commit=SOURCE_COMMIT,runner_sha256=sha(raw),
        source_and_fixed_input_hashes_match=True,baseline_rows_unchanged=True,
        context_counts=dict(counts),policies=summary,distinct_evaluated_maps=evaluated,
        exact_parts_residuals_local_T_deltas_and_witnesses=True,
        unchanged_nonframe_parts=True,old_outputs_changed=False,
        no_new_guard_synthesis_compile_RPE_global_T=True,
        author_checks_not_independent_scientific_review=True,
        checker_SOURCE_sha256=sha(Path(__file__).read_bytes()),
        environment=json.loads((OUT/"ENVIRONMENT.json").read_text()),
        STOP="MANDATORY_STOP")
    (OUT/"CHECKS.json").write_text(json.dumps(checks,sort_keys=True,indent=2)+"\n")
    print(json.dumps(checks))


if __name__=="__main__":main()
