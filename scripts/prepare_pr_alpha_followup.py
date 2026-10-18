"""Prepare UNAPPROVED B1/B2 plans using saved IDs/schema only, never numbers.

No probability, headroom, shot, cost or ranking function is called here.
Existing winner IDs are copied from fixed result tables, not selected anew.
Canonical scalar input projections remain unbound until their original private
sources and exact field mapping can be sealed before an approved execution.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/resource_applicability/pr_alpha_followup_preparation/2026-10-18"
AUDIT = "e62ddb59173d100c113be7a33581515d78f152ee"
H6_RESULT = "f89c40af73289af9f4c2b70ed0b3966d95d47b06"
H8_RESULT = "96e69a01d1fae7a402aac2c4efa570a59f8cc7a0"
EPS_RESULT = "41ba315b2ad369825577ad671c79b69c77e1f3cc"
SOURCE_PATHS = ["src/trotterlib/pr_alpha_followup.py", "scripts/pr_alpha_followup_runtime.py",
                "scripts/run_pr_alpha_binomial.py", "scripts/run_pr_alpha_sensitivity.py",
                "scripts/prepare_pr_alpha_followup.py", "scripts/d0_frozen_shot_rule_v1.py",
                "tests/test_pr_alpha_followup.py"]
MODES = ["USUAL_SIGNED_SAVED", "EMPIRICAL_NONCANCELLATION"]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def blob(commit, path):
    return subprocess.check_output(["git", "-C", str(ROOT), "show", commit+":"+path])


def saved(commit, path, catalog):
    raw = blob(commit,path)
    catalog.append(dict(commit=commit,path=path,bytes=len(raw),sha256=sha(raw)))
    content = gzip.decompress(raw) if path.endswith(".gz") else raw
    return [json.loads(line) for line in content.splitlines()] if ".jsonl" in path else json.loads(content)


def dump(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",required=True)
    args=parser.parse_args()
    args.source=subprocess.check_output(["git","-C",str(ROOT),"rev-parse","--verify",args.source+"^{commit}"],text=True).strip()
    OUT.mkdir(parents=True,exist_ok=True)
    closure=[]
    for path in SOURCE_PATHS:
        raw=(ROOT/path).read_bytes()
        if raw != blob(args.source,path):raise ValueError("SOURCE_NOT_SEALED:"+path)
        closure.append(dict(path=path,bytes=len(raw),sha256=sha(raw)))
    catalog=[]
    audit=saved(AUDIT,"artifacts/resource_applicability/pr_alpha_contract_audit/2026-10-18/technical_validation.json",catalog)
    h6stem="artifacts/resource_applicability/track_a_v1_reaccount_result/2026-10-16/"
    h8stem="artifacts/resource_applicability/track_a_h8_six_axis_reaccount_run01/2026-10-17/"
    epsstem="artifacts/resource_applicability/track_a_h8_epsilon_boundary_run01/2026-10-18/"
    h6rows=saved(H6_RESULT,h6stem+"candidate_accounting.json.gz",catalog)["rows"]
    h6comparison=saved(H6_RESULT,h6stem+"comparisons.json",catalog)["rows"]
    h8comparison=saved(H8_RESULT,h8stem+"comparisons.json",catalog)["comparisons"]
    epsrows=saved(EPS_RESULT,epsstem+"accounting_rows.jsonl.gz",catalog)
    epscomparison=saved(EPS_RESULT,epsstem+"comparisons.json",catalog)
    h6ids=sorted({r["cell_id"] for r in h6rows if r["method"]!="B2" or r.get("sample_scope")=="H6_CONFIRMATION_UNION8"})
    h8ids=sorted({r["cell_id"] for r in epsrows})
    assert len(h6ids)==144 and len(h8ids)==190
    representatives=[]
    for r in h6comparison:
        if r["factor"]==1 and r["bias_rule"]=="signed_signal" and r["scope"]=="fine_grid_S2_plus_truncated_S4" and r["epsilon"] in (.05,.001,.0005):
            for family,key in (("PF","deterministic"),("PR","confirmed_PR")):
                representatives.append(dict(system="H6",epsilon=r["epsilon"],family=family,
                    cell_id=r[key]["best"]["cell_id"],selection="COPY_SAVED_POINT_WINNER_ID_NOT_RERANKING"))
    for r in h8comparison:
        if r["factor"]==1 and r["bias_kind"]==MODES[0] and r["scope"]=="strengthened184" and r["epsilon"] in (.05,.001,.0005):
            for family,key in (("PF","after"),("PR","saved_PR")):
                representatives.append(dict(system="H8",epsilon=r["epsilon"],family=family,
                    cell_id=r[key]["best"]["cell_id"],selection="COPY_SAVED_POINT_WINNER_ID_NOT_RERANKING"))
    for r in epscomparison:
        if r["epsilon"]==.000625 and r["bias_kind"]==MODES[0]:
            for family,key in (("PF","deterministic"),("PR","PR")):
                representatives.append(dict(system="H8",epsilon=r["epsilon"],family=family,
                    cell_id=r[key]["best"]["cell_id"],selection="COPY_SAVED_POINT_WINNER_ID_NOT_RERANKING"))
    assert len(representatives)==14
    runtime=dict(python_version=sys.version.split()[0],packages=[])
    for name in ("numpy","scipy"):
        dist=importlib.metadata.distribution(name)
        runtime["packages"].append(dict(name=name,version=dist.version,record_sha256=sha(dist.read_text("RECORD").encode())))
    common=dict(schema="PR_ALPHA_FOLLOWUP_PLAN_V1",status="UNAPPROVED_PREPARATION_ONLY",
        approved_by_user=False,execution_authorized=False,sealed=False,code_sealed=True,source_commit=args.source,
        source_files=closure,runtime=runtime,payload_kind="REAL",inputs_bound=False,
        input=dict(path=None,bytes=None,sha256=None,schema="PR_ALPHA_BOUND_NATIVE_SCALARS_V1"),
        audit_commit=AUDIT,mandatory_stop=True,cross_stage_dispatch=False,retry=False,
        resource_policy="No arbitrary CPU/RAM cap; parse once, reuse fixed snapshots/pairs, optimized SciPy CDF/SF; no shared configuration changes",
        input_blockers=["Canonical scalar projection and exact field-to-original-source mapping not sealed",
            "H6 original signal raw is private: SHA256 b9f6cb3dc1a168ddc409833c78a617a7ddd05a3a41e0665e9dec47346d516dfe; current path unbound",
            "H8 original PR signal raw is private: SHA256 dac86904b9a9d5d72146956ce2ed7aa3f9da52545169f5bbecb2af45c8ae6706; current path unbound",
            "Saved raw/corrected, B_corr/B_budget, signed/noncancellation components, old baseline and paired costs must be joined without new scientific values"],
        forbidden_calls=["new_signal","Hamiltonian_DF_state_generation","trajectory_sampling","measurement_sampling","circuit_build","native_compile","nonuniform_alpha_optimization","RPE_energy_analysis"])
    b1=dict(common,stage="B1",scope=dict(contract="TRACK_A_NATIVE_SIGNAL_V1",allocation="EQUAL_TWO_AXES",
        alpha_total_grid=["0.05"],bias_modes=[MODES[0]],allowance_factors=[1.],representatives=representatives,
        case_keys=[[r["system"],r["cell_id"],r["epsilon"],MODES[0],1.] for r in representatives],
        failure_event="ge",boundary_policy="Exact rational boundaries for stored binary64; no snapping; numeric binomial evaluation separately labelled"),
        budget=dict(input_rows=14,binomial_axis_evaluations=28,maximum_CDF_SF_calls=56,
                    maximum_logCDF_logSF_calls=56,shot_recalculations=0,cost_reaccounting=0,Monte_Carlo=0),
        approval_required=["14 representative signal/precision conditions and fixed headroom/N input references",
            "Exact input hashes/projection, numeric runtime, independent B1 one-shot grant/output/marker"])
    eps=dict(H6=[.05,.01,.005,.001,.0005],H8=[.05,.01,.005,.001,.000625,.0005])
    domains=dict(H6=h6ids,H8=h8ids)
    keys=[[system,cid,e,mode,1.] for system,ids in domains.items() for cid in ids for e in eps[system] for mode in MODES]
    b2=dict(common,stage="B2",scope=dict(contract="TRACK_A_NATIVE_SIGNAL_V1",allocation="EQUAL_TWO_AXES",
        alpha_total_grid=["0.01","0.05","0.10"],alpha_grid_status="PROPOSED_NOT_APPROVED",epsilons=eps,
        candidate_domains=domains,bias_modes=MODES,allowance_factors=[1.],case_keys=keys,
        selection="Exact rational point cost, lexical ID tie break, full domain required; no population winner",
        H6_missing_noncancellation="Retain missing old B0I as unresolved; never drop from full-domain minimum"),
        budget=dict(input_rows=len(keys),accounting_rows=3*len(keys),baseline_accounting_rows=len(keys),
                    maximum_pair_weighted_cost_evaluations=3*len(keys),binomial_axis_evaluations=0,
                    new_sampling=0,new_compile=0),
        approval_required=["Proposed total-alpha grid 0.01/0.05/0.10 and equal two-axis allocation",
            "H6 144/H8 190 fixed candidate domains, epsilon sets, both bias modes, allowance factor1",
            "Exact input projection/hashes, source/runtime, independent B2 one-shot grant/output/marker"])
    assert b2["budget"]["input_rows"]==3720 and b2["budget"]["accounting_rows"]==11160
    for plan in (b1,b2):
        stage=plan["stage"]
        dump(OUT/(stage+"_plan_UNAPPROVED.json"),plan)
        dump(OUT/(stage+"_grant_INVALID.json"),dict(stage=stage,approved_by_user=False,runtime_authorization=False,
            plan_sha256=None,source_commit=args.source,input_sha256=None,one_shot=True,retry=False,
            auto_next_stage=False,output=None,marker=None))
    protected=[]
    for r in audit["source_catalog"]:
        git_blob=subprocess.check_output(["git","-C",str(ROOT),"rev-parse",r["commit"]+":"+r["path"]],text=True).strip()
        assert git_blob==r["git_blob"]
        protected.append(dict(commit=r["commit"],path=r["path"],git_blob=git_blob,status="FIXED_A_EVIDENCE_REUSED_NO_REAUDIT"))
    dump(OUT/"preparation_metadata.json",dict(schema="PR_ALPHA_FOLLOWUP_PREPARATION_METADATA_V1",
        audit_reused=AUDIT,prior_AST_checks_reused=11,prior_artificial_checks_reused=16,
        metadata_inputs=catalog,protected_fixed_objects=protected,representative_IDs_copied=14,
        real_binomial_evaluations=0,real_shot_recalculations=0,real_cost_reaccounting=0,
        B1_execution_authorized=False,B2_execution_authorized=False,private_raw_published=False))
    print(json.dumps(dict(status="PREPARATION_PLANS_WRITTEN_NOT_APPROVED",B1_axes=28,B2_rows=11160,
                          inputs_bound=False,real_scientific_calls=0)))


if __name__=="__main__":
    main()
