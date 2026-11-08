"""Prepared D1/D2 only; saved scientific operands must NOT be run yet.

Reuse frozen weighted-template/finite-choice functions. No quantum, synthesis,
strict guard, operator/defect, provider or RPE implementation is imported.
"""
from collections import Counter,defaultdict
from fractions import Fraction as F
from functools import lru_cache
import ast
import copy
import hashlib
import io
import json
import lzma
from pathlib import Path
import subprocess
import sys
import tarfile
import time

import audit_pr_reference_inverse_d_static as a

ROOT=Path(__file__).resolve().parents[1]
BASE=a.BASE
PARENT_RUNNER="scripts/diagnose_pr_reference_frame_impact.py"
GF=a.GF
POLICIES=("G","T","NT")


def reuse():
    raw=a.blob(PARENT_RUNNER)
    wanted={"load_arithmetic","choose_offers","context_details","witnesses"}
    nodes=[n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name in wanted]
    assert {n.name for n in nodes}==wanted
    ns={k:globals()[k] for k in ("Counter","defaultdict","F","lru_cache","ast","copy",
        "hashlib","io","json","lzma","Path","subprocess","tarfile","ROOT","BASE")}
    ns.update(blob=a.blob,LEGACY="scripts/diagnose_pr_reference_noncentral_s1_coverage_fix.py")
    exec(compile(ast.Module(body=nodes,type_ignores=[]),BASE+":"+PARENT_RUNNER,"exec"),ns)
    return ns


def d_multiplicities(q,arm):
    """SOURCE coefficients only; artificial tests check the old D expression."""
    if q<=1 or q%2:raise ValueError("Fixed diagnostic needs even q>1")
    if arm=="Bprime":
        return {("b",n):c for n,c in (("left_all",1),("right_all",1),
            ("left_no0",q-1),("right_no0",q-1),("boundary_control",q-1))}
    if arm=="reference":
        return {(m,n):c for m in ("ref","inv") for n,c in (
            ("block:0:half",2),("block:0:full",q//2-1),
            ("block:1:half",q),("block:2:half",q),("block:3:half",q))}
    raise ValueError("Unknown arm")


def d_attribution(arm,q,weights,details):
    """Future science: decompose the sum of template maxima, NOT a mean path."""
    terms=[];by_key=defaultdict(F)
    for (mapping,name),coefficient in d_multiplicities(q,arm).items():
        value=weights[mapping].get(name)
        if value is None:raise ValueError("Missing D guard")
        rows=[r for r in details if r["mapping"]==mapping and r["template"]==name]
        if not rows or any(r["logical_guard"] is None for r in rows):raise ValueError("Missing D context")
        assert max(r["logical_guard"] for r in rows)==value
        ties=sorted(r["context"] for r in rows if r["logical_guard"]==value)
        selected=next(r for r in rows if r["context"]==ties[0])
        contributions=selected["target_key_contributions"]
        assert sum((F(r["guard"]) for r in contributions.values()),F(0))==value
        for key,item in contributions.items():by_key[key]+=coefficient*F(item["guard"])
        terms.append(dict(mapping=mapping,template=name,SOURCE_multiplicity=coefficient,
            template_maximum=value,tied_contexts=ties,selected_context=ties[0],
            guard_contributions=contributions))
    return dict(terms=terms,key_upper_contributions=dict(by_key),
        keys_by_decreasing_guard=sorted(by_key,key=lambda k:(-by_key[k],k)),
        D_upper=sum(by_key.values(),F(0)),
        interpretation="Decomposition of conservative separate-context maxima; not a reachable joint trajectory or actual error")


def write(out,name,value):
    def serial(v):
        if isinstance(v,F):return str(v)
        raise TypeError(type(v).__name__)
    (out/name).write_text(json.dumps(value,sort_keys=True,indent=2,default=serial,allow_nan=False)+"\n")


def main(mode):
    if mode not in ("D1","D2"):raise ValueError("Only separately approved D1/D2 are prepared; D3 is not implemented")
    out=ROOT/"artifacts/pr_reference_inverse_d/2026-11-08/numeric"/mode
    assert not out.exists(),"Preserve previous numeric outputs"
    started=time.monotonic();source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    meta=a.read(a.META);s2={r["target_literal"]:r for r in a.read(a.S2)["pareto"]}
    s3=a.read(a.S3);ids=tuple(s3["source_RTE_basis_ids"])
    gf=a.read(GF);g=next(r for r in gf["policies"] if r["policy"]=="G")
    assert next(r for r in gf["policies"] if r["policy"]=="NT")["identical_to_policy"]=="G"
    f=reuse();control,legacy,arms,evidence=f["load_arithmetic"]()
    alphabet={arm:ids for arm in ("Bprime","reference")};required={}
    for arm in alphabet:required.update(control["required_templates"](arm,ids))
    batches={key:[dict(batch,records=[r for r in batch["records"] if r["name"] in required[key]])
        for _,batch in a.archive(path) if key!="ref" or batch["request"][0]=="reference"]
        for key,path in (("b",a.BP),("ref",a.REF),("inv",a.INV))}
    target_inverse={abs(float(p[0])).hex() for groups in batches.values() for batch in groups
        for rec in batch["records"] if rec["name"].startswith("basis_inverse:")
        for op,qs,p in rec["lowering"] if op=="rz"}
    target_D={abs(float(p[0])).hex() for m,groups in batches.items() for batch in groups
        for rec in batch["records"] if rec["name"] in a.D_NAMES[m]
        for op,qs,p in (rec["native"] if m=="inv" else rec)["lowering"] if op=="rz"}
    assert len(target_inverse)==109 and len(target_D)==164 and len(target_inverse&target_D)==5
    targets=tuple(sorted(target_inverse if mode=="D1" else target_D))
    assert not set(g["choices"]) & set(targets)
    assert all("fixed_current" in s2[k]["nondominated_ids"] for k in targets)
    c0=legacy["archive"]("artifacts/pr_reference_ct4/2026-11-07/strict_records.tar.xz")
    guards={r["theta_hex"]:F(r["strict_error_upper"]) for r in c0};prices={r["theta_hex"]:r["T"] for r in c0}
    for e in meta["entries"]:
        c=e["current_word"]
        if c["kind"]=="N4_85_composite":
            guards[e["target_literal"]]=F(c["Q_saved"])+F(c["G_saved"]);prices[e["target_literal"]]=c["existing_T_quoted"]
    for key,choice in g["choices"].items():
        offer=next(r for r in s2[key]["offers"] if r["id"]==choice["record_id"])
        assert F(offer["guard"])==F(choice["guard"]) and offer["T"]==choice["T_saved"]
        guards[key]=F(offer["guard"]);prices[key]=offer["T"]
    for key in targets:
        current=next(r for r in s2[key]["offers"] if r["id"]=="fixed_current")
        assert F(current["guard"])==guards[key] and current["T"]==prices[key]
    # A single literal map, not a role-specific assignment. Shared keys propagate.
    baseline_details=f["context_details"](batches,guards,prices,legacy["weighted_template_max"],targets)
    previous={(r["mapping"],r["template"],r["context"]):r for r in g["context_details"]}
    for row in baseline_details:
        old=previous[(row["mapping"],row["template"],row["context"])]
        assert row["logical_guard"]==F(old["logical_guard"])
        assert row["local_template_T_from_saved_prices"]==old["local_template_T_from_saved_prices"]
    quoted=g["rows"];baseline={(r["arm"],r["axis"]):r for r in quoted}
    out.mkdir(parents=True)
    write(out,"INPUTS.json",dict(base_commit=BASE,mode=mode,targets=targets,policies=POLICIES,
        fixed_G_forward_choices=g["choices"],SOURCE_sha256=source,
        reused_frame_SOURCE_sha256=hashlib.sha256(a.blob(PARENT_RUNNER)).hexdigest(),
        static_helper_sha256=hashlib.sha256(Path(a.__file__).read_bytes()).hexdigest(),
        saved_inputs=a.INPUTS,legacy_inputs=legacy["INPUTS"],arithmetic=evidence,
        shared_keys=sorted(target_inverse&target_D),old_G_aggregate_recomputed=False,
        fixed_m17=dict(q=131072,r=32,bits=88,N_B=626,N_R=630),
        role_dependent_word_selector=False))
    try:
        output=[];signatures={}
        for policy in POLICIES:
            choices={k:f["choose_offers"](s2[k],policy) for k in targets}
            signature=tuple(choices[k]["id"] for k in targets)
            if signature in signatures:
                output.append(dict(policy=policy,alias=signatures[signature]));continue
            signatures[signature]=policy
            cg,cp=dict(guards),dict(prices)
            for k,r in choices.items():cg[k]=F(r["guard"]);cp[k]=r["T"]
            weights={m:legacy["template_weights"](groups,cg,inverse=m=="inv") for m,groups in batches.items()}
            details=f["context_details"](batches,cg,cp,legacy["weighted_template_max"],targets)
            rows=[];attribution={}
            for arm in alphabet:
                case=control["scoped_parts"](131072,32,weights["b"],weights["ref"],weights["inv"],alphabet,arm,arms)
                if case["missing_templates"] or case["parts"] is None:raise ValueError("Required guard coverage missing")
                parts=case["parts"];total=sum(parts.values(),F(0))
                witness=f["witnesses"](weights,details,"b" if arm=="Bprime" else "ref",ids)
                if witness is None:raise ValueError("Missing frame witness")
                if mode=="D2":
                    attribution[arm]=d_attribution(arm,131072,weights,details)
                    assert attribution[arm]["D_upper"]==parts["D"]
                for axis in "XY":
                    old=baseline[(arm,axis)];budget=F(old["saved_joint_budget"])
                    rows.append(dict(arm=arm,axis=axis,parts=parts,noncentral_upper=total,
                        saved_joint_budget=budget,residual=budget-total,frame_witness=witness,
                        parts_change_from_fixed_G={k:v-F(old["parts"][k]) for k,v in parts.items()},
                        m17_RPE="UNQUALIFIED",domain_central_emitter_shell_terminal="UNQUALIFIED_NOT_ZERO"))
            output.append(dict(policy=policy,choices={k:dict(record_id=r["id"],sequence_sha256=r["sequence_sha256"],
                saved_guard=F(r["guard"]),saved_T=r["T"],guard_change=F(r["guard"])-guards[k],
                T_change_per_occurrence=r["T"]-prices[k]) for k,r in choices.items()},
                rows=rows,context_details=details,D_attribution=attribution,new_policy_adopted=False))
        if mode=="D2":
            baseline_weights={m:legacy["template_weights"](groups,guards,inverse=m=="inv") for m,groups in batches.items()}
            base_attribution={arm:d_attribution(arm,131072,baseline_weights,baseline_details) for arm in alphabet}
            for arm,value in base_attribution.items():assert value["D_upper"]==F(baseline[(arm,"X")]["parts"]["D"])
        else:base_attribution={}
        assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==source
        write(out,"RESULTS.json",dict(status="FINITE_INVERSE_D_DIAGNOSTIC_COMPLETE",mode=mode,
            baseline_G_rows_quoted=quoted,baseline_details=baseline_details,D_baseline_attribution=base_attribution,
            policies=output,full_RPE="UNQUALIFIED",global_T="NOT_EVALUATED",frontier_recomputed=False,
            scientific_word_policy_adopted=False,elapsed_seconds=time.monotonic()-started))
        write(out,"STOP.json",dict(status="MANDATORY_STOP",reason="Fixed diagnostic complete",next_owner="GPT/user"))
    except Exception as error:
        write(out,"FAILURE.json",dict(type=type(error).__name__,message=str(error)))
        write(out,"STOP.json",dict(status="MANDATORY_STOP",reason="Fixed diagnostic unqualified",next_owner="GPT/user"))
        raise


if __name__=="__main__":
    if len(sys.argv)!=2 or sys.argv[1] not in ("D1","D2"):
        raise SystemExit("Prepared only; after scientific permission: python3 -B scripts/diagnose_pr_reference_inverse_d.py D1|D2")
    mode=sys.argv[1];out=ROOT/"artifacts/pr_reference_inverse_d/2026-11-08/numeric"/mode
    if out.exists():raise RuntimeError("Existing results preserved")
    try:main(mode)
    except Exception as error:
        if not (out/"STOP.json").exists():
            out.mkdir(parents=True,exist_ok=True)
            write(out,"FAILURE.json",dict(type=type(error).__name__,message=str(error)))
            write(out,"STOP.json",dict(status="MANDATORY_STOP",reason="Input/spec failure",next_owner="GPT/user"))
        raise
