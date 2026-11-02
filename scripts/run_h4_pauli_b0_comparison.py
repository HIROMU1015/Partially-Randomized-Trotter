#!/usr/bin/env python3
"""Finite independently fixed batches for Pauli B0/B0I prefix/q comparison."""
import argparse
import concurrent.futures
import copy
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
PACKAGE="h4_b0_comparison_private"
pkg=types.ModuleType(PACKAGE);pkg.__path__=[str(ROOT/"src/trotterlib")];sys.modules[PACKAGE]=pkg
F=importlib.import_module(PACKAGE+".h4_pauli_b0_comparison")
E=importlib.import_module(PACKAGE+".h4_pauli_b0_execution")
OUT=ROOT/"artifacts/non_df_h4_pauli_b0_comparison/2026-10-25"
COMPILER="/home/AbeHiromu/projects/h6-environments/20261012/native-dependency-v2/venv/bin/python"
ARB="/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/non-df-h4-c-reference-fix-20261023/.c-reference-fix-venv/bin/python"
CODE=("scripts/run_h4_pauli_b0_comparison.py","src/trotterlib/h4_pauli_b0_comparison.py",
      "src/trotterlib/h4_pauli_b0_execution.py","src/trotterlib/h4_pauli_b0.py",
      "src/trotterlib/h4_arb_bridge.py","src/trotterlib/h4_arb_pr0_preparation.py",
      "src/trotterlib/h4_ball_evidence.py","src/trotterlib/h4_conditional_cng.py",
      "src/trotterlib/rte_compiled_cost.py","src/trotterlib/fixed_signal_evaluation.py",
      "src/trotterlib/pauli_signal_adapter.py","src/trotterlib/h4_d2_pauli_control.py")


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n");tmp.replace(path)


def batch_paths(batch):
    folder=OUT/batch
    return folder,folder/"run01"


def verify(batch):
    folder,run=batch_paths(batch);s=read(run/"SOURCE.json")
    if sha(folder/"plan.json")!=s["plan_sha256"]:raise ValueError("FIXED_BATCH_PLAN_CHANGED")
    for path,h in s["code_sha256"].items():
        if sha(ROOT/path)!=h:raise ValueError("BATCH_SOURCE_CHANGED:"+path)
    refs=read(OUT/"input_references.json")
    if sha(OUT/"input_references.json")!=s["input_reference_sha256"]:raise ValueError("INPUT_REFERENCE_CHANGED")
    for path,h in refs["files"].items():
        if sha(ROOT/path)!=h:raise ValueError("SCIENTIFIC_INPUT_CHANGED:"+path)
    return read(folder/"plan.json"),run


def case_spec(plan,case_id):
    matches=[c for c in plan["candidates"] if c["candidate_id"]==case_id]
    if len(matches)!=1:raise ValueError("UNREGISTERED_CANDIDATE")
    return matches[0]


def target_job(batch,case_id):
    plan,run=verify(batch);c=case_spec(plan,case_id);directory=run/"cases"/case_id
    if (directory/"target.json").exists():raise FileExistsError("SAVED_TARGET_NO_RETRY")
    base=read(ROOT/plan["inputs"]["C_model"]);reference=read(ROOT/plan["inputs"]["C_reference"])
    model=F.candidate_model(base,c["prefix"],c["q"])
    for bits in plan["precision_bits"]:
        result=E.target_enclosure(model,reference,bits)
        result["candidate_id"]=case_id;result["prefix"]=c["prefix"];result["q"]=c["q"]
        result["precision_decisions"]=E.precision_decisions(result)
        save(directory/(f"target_{bits}.json"),result)
        if not any(d["status"]=="WIDTH_AMBIGUITY" for d in result["precision_decisions"]):break
    result["state_actions_this_candidate"]=sum((directory/f"target_{b}.json").exists() for b in plan["precision_bits"])
    save(directory/"target.json",result)


def build_job(batch,case_id):
    plan,run=verify(batch);c=case_spec(plan,case_id);directory=run/"cases"/case_id
    if (directory/"build.json").exists():raise FileExistsError("SAVED_BUILD_NO_RETRY")
    from h4_b0_comparison_private import h4_pauli_b0 as B
    from h4_b0_comparison_private.pauli_signal_adapter import PauliHamiltonian
    from h4_b0_comparison_private.fixed_signal_evaluation import FixedSignalConfig
    from qiskit import qpy
    base=read(ROOT/plan["inputs"]["C_model"])
    h=PauliHamiltonian(base["n_qubits"],base["hamiltonian"]["constant"],tuple(tuple(x) for x in base["hamiltonian"]["terms"]),"{}")
    if h.hamiltonian_hash!=plan["Hamiltonian_hash"]:raise ValueError("SAVED_PAULI_DESCRIPTOR_MISMATCH")
    cfg=B.matched_config(FixedSignalConfig("pauli_jw",method="PR",T=.8,q=c["q"],L_D=c["prefix"],r=4,K=4,seed=17))
    body,adapter=B.build_body(h,cfg);wrappers,controlled=B.central_wrappers(body)
    record=dict(candidate_id=case_id,prefix=c["prefix"],q=c["q"],constant_phase=float(body.global_phase),
                retained_labels=adapter.metadata()["retained_labels"],B0I_equivalent_to_B0=True,
                body_builds=1,central_lifts=1,axis_wrappers=2,inputs={})
    for axis,circuit in wrappers.items():
        folder=directory/axis;folder.mkdir(parents=True,exist_ok=True)
        stream=io.BytesIO();qpy.dump(circuit,stream);raw=stream.getvalue()
        (folder/"input.qpy.gz").write_bytes(gzip.compress(raw,mtime=0))
        record["inputs"][axis]=dict(uncompressed_sha256=hashlib.sha256(raw).hexdigest(),gzip_sha256=sha(folder/"input.qpy.gz"))
    save(directory/"build.json",record)


def compile_job(batch,case_id,axis):
    plan,run=verify(batch);case_spec(plan,case_id);folder=run/"cases"/case_id/axis
    if (folder/"metrics.json").exists():raise FileExistsError("SAVED_COMPILE_NO_RETRY")
    from h4_b0_comparison_private.fixed_signal_evaluation import default_compiler
    from h4_b0_comparison_private.rte_compiled_cost import transpile_and_measure_cost
    from qiskit import qpy
    if importlib.metadata.version("qiskit")!="1.3.0":raise ValueError("QISKIT_VERSION")
    raw=gzip.decompress((folder/"input.qpy.gz").read_bytes())
    if hashlib.sha256(raw).hexdigest()!=read(folder.parent/"build.json")["inputs"][axis]["uncompressed_sha256"]:raise ValueError("INPUT_QPY_CHANGED")
    circuits=qpy.load(io.BytesIO(raw))
    if len(circuits)!=1:raise ValueError("SINGLE_INPUT_CIRCUIT")
    cost=transpile_and_measure_cost(circuits[0],default_compiler())
    stream=io.BytesIO();qpy.dump(cost.transpiled_circuit,stream);output=stream.getvalue()
    (folder/"output.qpy.gz").write_bytes(gzip.compress(output,mtime=0))
    from dataclasses import asdict
    metrics=dict(RZ=cost.rz_count,CX=cost.cx_count,depth=cost.total_depth,rz_depth=cost.rz_depth,cx_depth=cost.cx_depth,
                 pretranspile_histogram=dict(cost.pretranspile_gate_counts),posttranspile_histogram=dict(cost.posttranspile_gate_counts),
                 input_fingerprint=cost.actual_circuit_fingerprint,compiler=asdict(cost.compiler),
                 output_sha256=hashlib.sha256(output).hexdigest(),output_gzip_sha256=sha(folder/"output.qpy.gz"))
    save(folder/"metrics.json",metrics)


def run_batch(batch):
    folder,run=batch_paths(batch)
    if run.exists():raise FileExistsError("PAST_BATCH_ATTEMPT_PRESERVED")
    plan=read(folder/"plan.json")
    if not plan["execution_enabled"]:raise ValueError("BATCH_DISABLED")
    run.mkdir();source=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    for p in CODE:
        if subprocess.check_output(["git","show",source+":"+p],cwd=ROOT)!=(ROOT/p).read_bytes():raise ValueError("COMMIT_SOURCE_FIRST:"+p)
    save(run/"SOURCE.json",dict(commit=source,code_sha256={p:sha(ROOT/p) for p in CODE},plan_sha256=sha(folder/"plan.json"),
                               input_reference_sha256=sha(OUT/"input_references.json"),runtimes=dict(compiler=COMPILER,Arb=ARB)))
    receipts=[]
    def execute(job):
        label,python,args=job;started=time.time()
        with (run/(label+".stdout.txt")).open("xb") as out,(run/(label+".stderr.txt")).open("xb") as err:
            command=[python,str(Path(__file__).resolve()),*args,"--batch",batch]
            p=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1"),stdout=out,stderr=err)
            code=p.wait()
        return dict(label=label,pid=p.pid,command=command,native_exit=code,reaped=True,started_unix=started,ended_unix=time.time())
    def wave(jobs):
        if not jobs:return
        # Parallelism follows the finite task count and actual CPU affinity.
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(jobs),len(os.sched_getaffinity(0)))) as pool:
            futures=[pool.submit(execute,j) for j in jobs]
            for future in concurrent.futures.as_completed(futures):
                row=future.result();receipts.append(row);save(run/"native_children.json",receipts)
        if any(x["native_exit"]!=0 for x in receipts):raise RuntimeError("BATCH_CHILD_FAILURE_NO_AUTOMATIC_RETRY")
    started=time.time()
    try:
        verify(batch)
        wave([(c["candidate_id"]+"_target",ARB,["target","--candidate",c["candidate_id"]]) for c in plan["candidates"] if c["target_kind"]=="NEW"])
        # Resolve aliases after all new independent targets exist. The scientific
        # basis is diagonal commutation and exact T/q for powers-of-two q.
        for c in plan["candidates"]:
            directory=run/"cases"/c["candidate_id"];kind=c["target_kind"]
            if kind=="NEW":continue
            if kind=="SAVED":target=read(ROOT/c["target_path"])
            elif kind=="FULL_PF_SAVED":
                old=read(ROOT/plan["inputs"]["old_B0_target"])
                target=F.reuse_full_pf_target(read(ROOT/plan["inputs"]["C_reference"]),old["raw_norm_exact"])
            elif kind=="ALIAS":
                if not F.commuting_prefix(read(ROOT/plan["inputs"]["C_model"]),c["prefix"]):raise ValueError("ALIAS_REQUIRES_DIAGONAL_COMMUTATION")
                target=read(run/"cases"/c["alias_candidate"]/"target.json")
            else:raise ValueError("TARGET_REUSE_KIND")
            target=copy.deepcopy(target);target.update(candidate_id=c["candidate_id"],prefix=c["prefix"],q=c["q"],state_actions_this_candidate=0,reuse=dict(kind=kind,path=c.get("target_path"),alias_candidate=c.get("alias_candidate")))
            save(directory/"target.json",target)
        builds=[];eligible=[]
        for c in plan["candidates"]:
            target=read(run/"cases"/c["candidate_id"]/"target.json")
            if any(F.joint_status(target,e)=="CONDITIONAL_GUARD_ELIGIBLE" for e in F.EPSILONS):
                eligible.append(c["candidate_id"])
                if not c.get("price_reuse"):builds.append(c)
        wave([(c["candidate_id"]+"_build",COMPILER,["build","--candidate",c["candidate_id"]]) for c in builds])
        wave([(c["candidate_id"]+"_"+axis,COMPILER,["compile","--candidate",c["candidate_id"],"--axis",axis]) for c in builds for axis in F.AXES])
        results=[]
        stat=read(ROOT/plan["inputs"]["axis_statistics"])
        for c in plan["candidates"]:
            directory=run/"cases"/c["candidate_id"];target=read(directory/"target.json");metrics={}
            if c.get("price_reuse")=="SAVED_B0":metrics=read(ROOT/plan["inputs"]["old_B0_results"])["costs"]
            elif c.get("price_reuse")=="SAVED_FULL_PF":metrics={a:stat["PF"][a] for a in F.AXES}
            elif c["candidate_id"] in eligible:metrics={a:read(directory/a/"metrics.json") for a in F.AXES}
            prices={a:metrics.get(a,{}).get("RZ") for a in F.AXES}
            rows=[]
            for eps in F.EPSILONS:
                row=F.account(target,prices,eps);row.update(candidate_id=c["candidate_id"],prefix=c["prefix"],q=c["q"],family=c["family"],eligibility=F.joint_status(target,eps));rows.append(row)
            record=dict(candidate=c,metrics=metrics,prices=prices,precision_bits=target["bits"],target_path=str((directory/"target.json").relative_to(ROOT)),rows=rows)
            save(directory/"accounting.json",record);results.append(record)
        state_calls=sum(read(run/"cases"/c["candidate_id"]/"target.json").get("state_actions_this_candidate",0) for c in plan["candidates"])
        counts=dict(candidates=len(results),new_target_state_actions=state_calls,new_body_builds=len(builds),new_central_lifts=len(builds),new_axis_wrappers=2*len(builds),new_top_level_compiles=2*len(builds),eligible_at_any_precision=len(eligible))
        assert counts["new_target_state_actions"]<=plan["call_limits"]["target_state_actions"]
        assert counts["new_top_level_compiles"]<=plan["call_limits"]["top_level_compiles"]
        verify(batch)
        save(run/"batch_results.json",dict(batch=batch,SOURCE=source,plan_sha256=sha(folder/"plan.json"),counts=counts,candidates=results,elapsed_seconds=time.time()-started))
        save(run/"BATCH_COMPLETE.json",dict(status="BATCH_COMPLETE_NO_RETRY",all_children_reaped=True,child_count=len(receipts),native_child_exits=[x["native_exit"] for x in receipts],counts=counts))
        print(json.dumps(dict(batch=batch,counts=counts,elapsed_seconds=time.time()-started),ensure_ascii=False))
    except BaseException as error:
        save(run/"FAILED_STOP.json",dict(status="FAILED_BATCH_PRESERVED",error=repr(error),native_receipts=receipts));raise


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("mode",choices=("run-batch","target","build","compile"));p.add_argument("--batch",required=True);p.add_argument("--candidate");p.add_argument("--axis",choices=F.AXES);a=p.parse_args()
    if a.mode=="run-batch":run_batch(a.batch)
    elif a.mode=="target":target_job(a.batch,a.candidate)
    elif a.mode=="build":build_job(a.batch,a.candidate)
    else:compile_job(a.batch,a.candidate,a.axis)
