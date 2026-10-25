#!/usr/bin/env python3
"""Finite registered PR batches. SOURCE/input and native exits reuse B0 style.

No approval-proof machinery. Each batch is a fresh directory and cannot reopen
a completed or failed batch. Adaptive choices live in separately fixed plans.
"""
import argparse
from collections import Counter
import concurrent.futures
from dataclasses import asdict
from fractions import Fraction as Q
import gzip
import hashlib
import importlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types

ROOT=Path(__file__).resolve().parents[1]
pkg=types.ModuleType("pr_search_execution");pkg.__path__=[str(ROOT/"src/trotterlib")];sys.modules[pkg.__name__]=pkg
S=importlib.import_module(pkg.__name__+".h4_pauli_pr_search")
M=importlib.import_module(pkg.__name__+".h4_pauli_pr_mean")
F=importlib.import_module(pkg.__name__+".h4_pauli_b0_comparison")
E=importlib.import_module(pkg.__name__+".h4_pauli_b0_execution")
A=importlib.import_module(pkg.__name__+".h4_conditional_cng")
OUT=ROOT/"artifacts/non_df_h4_pauli_pr_search/2026-10-25"
COMPILER="/home/AbeHiromu/projects/h6-environments/20261012/native-dependency-v2/venv/bin/python"
ARB="/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/non-df-h4-c-reference-fix-20261023/.c-reference-fix-venv/bin/python"
MODEL="artifacts/non_df_h4_c_reference_evidence/2026-10-23/run01/pauli_jw_model.json"
REFERENCE="artifacts/non_df_h4_c_reference_evidence/2026-10-23/run01/pauli_jw_result.json"
OLD_PR="artifacts/non_df_h4_a_axes/2026-10-24/run01/paired_costs.json"
CODE=["scripts/run_h4_pauli_pr_search.py","src/trotterlib/h4_pauli_pr_search.py",
      "src/trotterlib/h4_pauli_pr_mean.py","src/trotterlib/h4_pauli_b0_comparison.py",
      "src/trotterlib/h4_pauli_b0_execution.py","src/trotterlib/h4_conditional_cng.py",
      "src/trotterlib/h4_arb_bridge.py","src/trotterlib/h4_arb_pr0_preparation.py",
      "src/trotterlib/h4_ball_evidence.py","src/trotterlib/pauli_signal_adapter.py",
      "src/trotterlib/fixed_signal_evaluation.py","src/trotterlib/rte.py",
      "src/trotterlib/rte_compiled_cost.py","src/trotterlib/rte_event_audit.py",
      "src/trotterlib/h4_d2_pauli_control.py"]


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+".tmp");t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+"\n");t.replace(p)
def spec_id(c):return S.candidate_id(c["L_D"],c["q"],c["r"],c["K"])


def bind(stage):
    folder=OUT/stage;plan=read(folder/"plan.json");source=read(folder/"run01/SOURCE.json")
    if sha(folder/"plan.json")!=source["plan_sha256"] or sha(OUT/"active_scope.json")!=source["scope_sha256"]:
        raise ValueError("BATCH_SCOPE_CHANGED")
    for p,h in source["code_sha256"].items():
        if sha(ROOT/p)!=h:raise ValueError("FIXED_SCIENTIFIC_SOURCE_CHANGED:"+p)
    for p,h in source["input_sha256"].items():
        if sha(ROOT/p)!=h:raise ValueError("FIXED_INPUT_CHANGED:"+p)
    return plan,folder/"run01"


def hamiltonian():
    from pr_search_execution.pauli_signal_adapter import PauliHamiltonian
    m=read(ROOT/MODEL)
    h=PauliHamiltonian(m["n_qubits"],m["hamiltonian"]["constant"],tuple(tuple(x) for x in m["hamiltonian"]["terms"]),"{}")
    if h.hamiltonian_hash!="e0dbbe9a2b6e86f81004307cd337ba83ee1e25642f04a30b02365be4d2dae795":raise ValueError("FROZEN_HAMILTONIAN")
    return m,h


def descriptor_job(stage):
    plan,run=bind(stage);m,h=hamiltonian()
    if importlib.metadata.version("qiskit")!="1.3.0":raise ValueError("FROZEN_QISKIT")
    for c in plan["candidates"]:
        cid=spec_id(c);p=run/"candidates"/cid/"law.json"
        if p.exists():raise FileExistsError("LAW_ALREADY_SAVED")
        d=S.law_descriptor(h,S.config(c["L_D"],c["q"],c["r"],c["K"]))
        if c.get("reuse_saved_fixed_candidate"):
            old=m["cases"]["PR"]
            for new,prior in ((d["angles"],old["law"]["angles"]),(d["component_mapping"]["cdf"],old["law"]["component_cdf"]),(d["order_mapping"]["cdf"],old["law"]["order_cdf"]),(d["B_saved_binary64"],old["saved_normalization"]["B"]),(d["scalar_step_binary64"],sum(old["native_scalar_phases"]))):
                if new!=prior:raise ValueError("FIXED_PR_SOURCE_CORRESPONDENCE")
            d["saved_fixed_descriptor_correspondence"]="PASS"
        save(p,d)


def reused_fixed_target(model,reference):
    # Saved C already divides this field by the same exact raw norm.
    old=reference["implemented_law"]["PR"]["same_ray_physical_bias"]
    nu=sum((Q(x)**2+Q(y)**2 for x,y in model["state"]),Q(0))
    t=dict(bits=reference["bits"],state_actions=0,raw_norm_exact=str(nu),state_renormalized=False,
           axes={},bias=old,reuse="SAVED_IMPLEMENTED_C_WHOLE_BIAS_INTERVAL",actual_rho=None)
    for axis,key in zip(F.AXES,("real","imag")):
        center,radius=E.center_radius(old[key]);t["axes"][axis]=dict(signed_bias_exact=str(center),u_enc_exact=str(radius),interval_width_exact=str(2*radius),bias_interval=old[key])
    return t


def mean_job(stage,cid):
    plan,run=bind(stage);cs=[c for c in plan["candidates"] if spec_id(c)==cid]
    if len(cs)!=1:raise ValueError("UNREGISTERED_MEAN")
    c=cs[0];folder=run/"candidates"/cid
    if (folder/"target.json").exists():raise FileExistsError("PAST_TARGET_PRESERVED")
    model=read(ROOT/MODEL);reference=read(ROOT/REFERENCE);d=read(folder/"law.json")
    if c.get("reuse_saved_fixed_candidate"):
        t=reused_fixed_target(model,reference)
    else:
        for bits in (64,96,128):
            t=M._mean_enclosure(d,model["state"],reference,bits)
            save(folder/("target_"+str(bits)+".json"),t)
            if not any(x["status"]=="WIDTH_AMBIGUITY" for x in E.precision_decisions(t)):break
    t["state_actions_this_candidate"]=sum((folder/("target_"+str(b)+".json")).exists() for b in (64,96,128))
    save(folder/"target.json",t)


class Capture:
    """Observe existing audited sampler return values, without RNG replay."""
    def observe(self,adapter,sequences,step_seeds):
        self.events=[dict(event_position=i,outer_step_index=i//adapter.r,inner_RTE_position=i%adapter.r,**event.to_dict())
                     for i,event in enumerate(e for s in sequences for e in s)]
        self.step_seeds=step_seeds


def price_job(stage,job_index):
    plan,run=bind(stage);j=plan["jobs"][job_index];c=j["candidate"]
    cid=spec_id(c);cohort=j["cohort"];index=j["index"]
    folder=run/"prices"/cid/cohort/f"{index:04d}"
    if folder.exists():raise FileExistsError("PAST_TRAJECTORY_PRESERVED")
    folder.mkdir(parents=True)
    _,h=hamiltonian();cfg=S.config(c["L_D"],c["q"],c["r"],c["K"])
    seed=S.trajectory_seed(c["L_D"],c["q"],c["r"],c["K"],cohort,index)
    observer=Capture();body,wrapped,sidecar=S.central_trajectory(h,cfg,seed=seed,audit=observer)
    if len(observer.events)!=c["q"]*c["r"]:raise ValueError("EVENT_JOIN_COUNT")
    from pr_search_execution.fixed_signal_evaluation import default_compiler
    from pr_search_execution.rte_compiled_cost import TranspiledCircuitCostCache
    from qiskit import qpy
    cache=TranspiledCircuitCostCache(retain_transpiled_circuits=True)
    metrics={};compiler=default_compiler()
    if compiler.qiskit_version!="1.3.0":raise ValueError("QISKIT_VERSION")
    for axis in F.AXES:
        cost,_,hit=cache.get_or_transpile(wrapped[axis],compiler,
            circuit_fingerprint=f"{cid}:{cohort}:{index}:{axis}")
        out=io.BytesIO();qpy.dump(cost.transpiled_circuit,out);payload=out.getvalue()
        metrics[axis]=dict(RZ=cost.rz_count,CX=cost.cx_count,depth=cost.total_depth,rz_depth=cost.rz_depth,cx_depth=cost.cx_depth,input_fingerprint=cost.actual_circuit_fingerprint,compiled_QPY_sha256=hashlib.sha256(payload).hexdigest(),cache_hit=hit)
        if index==0:
            (folder/(axis+"_compiled.qpy.gz")).write_bytes(gzip.compress(payload,mtime=0))
            inp=io.BytesIO();qpy.dump(wrapped[axis],inp)
            (folder/(axis+"_input.qpy.gz")).write_bytes(gzip.compress(inp.getvalue(),mtime=0))
    save(folder/"trajectory.json",dict(candidate_id=cid,cohort=cohort,index=index,seed=seed,
        events=observer.events,step_seeds=observer.step_seeds,order_counts=dict(Counter(str(e["taylor_order"]) for e in observer.events)),
        event_record_hash=hashlib.sha256(json.dumps(observer.events,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest(),
        source_global_phase=float(body.global_phase),central_phase_owner=sidecar["phase_owner"],
        compiler=asdict(compiler),metrics=metrics,RZ={a:metrics[a]["RZ"] for a in F.AXES},
        event_costs_additive=False,paired_axes=True,sampler_calls=1,body_builds=1,central_lifts=1,top_level_compiles=2,
        compiled_QPY_storage="Both axes saved at index0 of each cohort; other compiled-byte SHA retained with metrics/events/seed/source",actual_rho=None))


def execute(stage):
    folder=OUT/stage;run=folder/"run01";plan=read(folder/"plan.json")
    scope=read(OUT/"active_scope.json")
    if not scope["execution_enabled"] or not plan["execution_enabled"]:raise ValueError("BATCH_DISABLED")
    if run.exists():raise FileExistsError("PAST_BATCH_ATTEMPT_PRESERVED")
    source=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    for p in CODE:
        if subprocess.check_output(["git","show",source+":"+p],cwd=ROOT)!=(ROOT/p).read_bytes():raise ValueError("COMMIT_SOURCE_FIRST:"+p)
    refs=read(OUT/"input_references.json")["files"]
    for p,h in refs.items():
        if sha(ROOT/p)!=h:raise ValueError("INPUT_CHANGED:"+p)
    run.mkdir()
    save(run/"SOURCE.json",dict(commit=source,code_sha256={p:sha(ROOT/p) for p in CODE},input_sha256=refs,
        plan_sha256=sha(folder/"plan.json"),scope_sha256=sha(OUT/"active_scope.json"),runtimes=dict(compiler=COMPILER,Arb=ARB)))
    receipts=[];start=time.time()
    def child(job):
        label,python,args=job
        with (run/(label+".stdout.txt")).open("xb") as out,(run/(label+".stderr.txt")).open("xb") as err:
            begin=time.time();p=subprocess.Popen([python,str(Path(__file__).resolve()),*args,"--stage",stage],cwd=ROOT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1"),stdout=out,stderr=err)
            code=p.wait()
        return dict(label=label,pid=p.pid,native_exit=code,reaped=True,started_unix=begin,elapsed_seconds=time.time()-begin)
    def wave(jobs):
        if not jobs:return
        print(json.dumps(dict(stage=stage,wave_tasks=len(jobs),affinity=len(os.sched_getaffinity(0)))),flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(jobs),len(os.sched_getaffinity(0)))) as pool:
            futures=[pool.submit(child,j) for j in jobs]
            for f in concurrent.futures.as_completed(futures):
                r=f.result();receipts.append(r);save(run/"native_children.json",receipts)
                if len(receipts)%10==0:print(json.dumps(dict(completed=len(receipts),last=r["label"])),flush=True)
        if any(x["native_exit"]!=0 for x in receipts):raise RuntimeError("BATCH_CHILD_FAILURE_PRESERVED")
    try:
        bind(stage)
        if plan["kind"]=="CANDIDATES":
            wave([("descriptors",COMPILER,["descriptors"])])
            wave([(spec_id(c)+"_mean",ARB,["mean","--candidate",spec_id(c)]) for c in plan["candidates"]])
            results=[]
            for c in plan["candidates"]:
                cid=spec_id(c);p=run/"candidates"/cid;d=read(p/"law.json");t=read(p/"target.json")
                acc=[dict(S.account(t,{a:None for a in F.AXES},e,d),eligibility=F.joint_status(t,e)) for e in F.EPSILONS]
                results.append(dict(candidate=c,law_path=str((p/"law.json").relative_to(ROOT)),target_path=str((p/"target.json").relative_to(ROOT)),rows=acc,state_actions=t["state_actions_this_candidate"]))
            counts=dict(candidates=len(results),mean_state_actions=sum(x["state_actions"] for x in results),trajectory_samples=0,top_level_compiles=0)
            result=dict(candidates=results)
        elif plan["kind"]=="PRICES":
            wave([(f"price_{i:05d}",COMPILER,["price","--job",str(i)]) for i in range(len(plan["jobs"]))])
            counts=dict(candidates=0,mean_state_actions=0,trajectory_samples=len(plan["jobs"]),top_level_compiles=2*len(plan["jobs"]))
            result=dict(trajectory_paths=[str(p.relative_to(ROOT)) for p in sorted((run/"prices").glob("*/*/*/trajectory.json"))])
        else:raise ValueError("BATCH_KIND")
        for key,limit in plan["call_limits"].items():
            if counts[key]>limit:raise ValueError("BATCH_CALL_SCOPE:"+key)
        bind(stage)
        save(run/"stage_results.json",dict(stage=stage,SOURCE=source,plan_sha256=sha(folder/"plan.json"),counts=counts,**result))
        save(run/"STOP.json",dict(status="BATCH_COMPLETE_NO_REOPEN",all_children_reaped=True,children=len(receipts),native_child_exits=[x["native_exit"] for x in receipts],elapsed_seconds=time.time()-start,counts=counts))
        print(json.dumps(dict(stage=stage,status="COMPLETE",elapsed_seconds=time.time()-start,counts=counts)),flush=True)
    except BaseException as err:
        save(run/"FAILED_STOP.json",dict(status="FAILED_BATCH_PRESERVED",error=repr(err),native_receipts=receipts));raise


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("mode",choices=["run","descriptors","mean","price"]);p.add_argument("--stage",required=True);p.add_argument("--candidate");p.add_argument("--job",type=int);a=p.parse_args()
    if a.mode=="run":execute(a.stage)
    elif a.mode=="descriptors":descriptor_job(a.stage)
    elif a.mode=="mean":mean_job(a.stage,a.candidate)
    else:price_job(a.stage,a.job)
