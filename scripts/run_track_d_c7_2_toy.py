#!/usr/bin/env python3
"""Explicit one-shot C7-2 artificial probes; never loads H4 data."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
PLAN="docs/research/track_d_c7_2_cx_structure_20261107/PLAN.md"
CASES="docs/research/track_d_c7_2_cx_structure_20261107/CASES.json"
OUTPUT="artifacts/track_d_c7_2_toy/2026-11-07/run01"
EXEC=(PLAN,CASES,"scripts/run_track_d_c7_2_toy.py","src/trottertracks/track_d_c7_2_toy.py")
def write(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+"\n")
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def git(*args):
    return subprocess.check_output(["git","-C",str(ROOT),*args])
def compile_one(name):
    from trottertracks import track_d_c7_2_toy as toy
    from trottertracks.c1_h4_native_fair import compile_checked
    from trottertracks.pr_dictionary_validation.fixture import CompilerProfile
    out=ROOT/OUTPUT; raw=toy.c3.base.load_qpy(out/"local"/(name+"_raw.qpy"))
    compiled=compile_checked(raw,CompilerProfile())
    clean=lambda c:c.remove_final_measurements(inplace=False) if c.num_clbits else c
    check=toy.check(toy.Operator(clean(compiled)).data,toy.Operator(clean(raw)).data,"P0_full_unitary_phase")
    toy.c3.base.save_qpy(out/"local"/(name+"_compiled.qpy"),compiled)
    record=dict(name=name,status="PASS",postcheck=check,raw=toy.structure(raw),compiled=toy.structure(compiled))
    write(out/"cases"/(name+".json"),record)
    return name
def main():
    p=argparse.ArgumentParser();p.add_argument("--origin",required=True);a=p.parse_args()
    source=git("rev-parse","HEAD").decode().strip()
    # The pre-existing frozen origin is an export, not a Git worktree.
    # Identify its scientific bytes against the named commit in ROOT's repo.
    for path in EXEC:
        if (ROOT/path).read_bytes()!=git("show",source+":"+path):raise ValueError("UNCOMMITTED_SOURCE:"+path)
    boot=load("c7_2_boot",ROOT/"scripts/run_track_d_h4_reference.py");boot.bootstrap(a.origin)
    from trottertracks import track_d_c7_2_toy as toy
    from trottertracks.pr_dictionary_validation.fixture import CompilerProfile
    import importlib.metadata
    if importlib.metadata.version("qiskit")!="1.3.0":raise ValueError("P0_VERSION")
    spec=json.loads((ROOT/CASES).read_text());out=ROOT/OUTPUT;out.mkdir(parents=True,exist_ok=False)
    for leaf in ("local","cases"): (out/leaf).mkdir()
    (out/"local/.gitignore").write_text("*\n")
    dependencies=["src/trottertracks/track_d_c3_whole_path.py","src/trottertracks/track_d_h4_reference.py",
        "src/trottertracks/resource_applicability/ax2b_native_df_v5.py","scripts/native_df_qiskit_validation_support.py"]
    origin_files=["src/trottertracks/c1_h4_strong_baseline.py","src/trottertracks/c1_h6_native_precision.py",
        "src/trottertracks/c1_h6_compiler_precision.py","src/trottertracks/pr_dictionary_validation/circuits.py",
        "src/trottertracks/pr_dictionary_validation/gates.py","src/trotterlib/df_trotter/ops.py"]
    hashes={path:hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in (*EXEC,*dependencies)}
    origin_hashes={path:hashlib.sha256((Path(a.origin)/path).read_bytes()).hexdigest() for path in origin_files}
    for path in dependencies:
        if (ROOT/path).read_bytes()!=git("show",source+":"+path):raise ValueError("UNCOMMITTED_DEPENDENCY:"+path)
    for path in origin_files:
        if (Path(a.origin)/path).read_bytes()!=git("show","ba0e2bf2d1d6f0dc78616d3392f3ff5b6ba7d5b7:"+path):
            raise ValueError("ORIGIN_SOURCE_BYTES:"+path)
    write(out/"SOURCE.json",dict(source_commit=source,origin_commit="ba0e2bf2d1d6f0dc78616d3392f3ff5b6ba7d5b7",
        files=hashes,origin_files=origin_hashes,compiler=asdict(CompilerProfile()),topology="free",
        phase_bits=160,molecular_input_loaded=False,scientific_scope="artificial-only"))
    support=load("c7_2_exterior_support",ROOT/"scripts/native_df_qiskit_validation_support.py")
    try:
        cases,bundles,checks=toy.make_cases(spec,support.exterior_fock_lift)
        for case in cases:
            if case["expected"] is not None:
                checks[case["name"]]=[toy.check(toy.Operator(case["raw"]).data,case["expected"],"raw_independent_unitary")]
            toy.c3.base.save_qpy(out/"local"/(case["name"]+"_raw.qpy"),case["raw"])
        write(out/"raw_checks.json",dict(status="PASS",checks=checks,
            fixtures={k:b.identity for k,b in bundles.items()},cases=[{k:v for k,v in c.items() if k not in ("raw","expected")} for c in cases]))
        workers=min(len(cases),len(os.sched_getaffinity(0)))
        write(out/"local/resources.json",dict(workers=workers,affinity=len(os.sched_getaffinity(0))))
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("fork")) as pool:
            futures=[pool.submit(compile_one,c["name"]) for c in cases]
            for future in as_completed(futures):print("PASS",future.result(),flush=True)
        # Compiled X/Y observables and paired coherence, vacuum/reference leakage.
        post=[]
        for name,bundle in bundles.items():
            pair={}
            for method in ("symmetric_directional","reference"):
                obs={}
                for axis in ("X","Y"):
                    c=toy.c3.base.load_qpy(out/"local"/(name+"_"+method+"_"+axis+"_compiled.qpy"))
                    obs[axis]=toy.c3.base.artificial.induced_observable(c)
                    post.append(toy.check(obs[axis],toy.c3.base.artificial.expected_observable(bundle.reference,axis),name+method+axis,observable=True))
                    if method == "reference":
                        matrix=toy.Operator(c.remove_final_measurements(inplace=False)).data
                        rows=[i for i in range(32) if i & 12]
                        import numpy as np
                        post.append(toy.check(matrix[rows,:4],np.zeros((len(rows),4)),name+"_compiled_reference_leakage"))
                pair[method]=obs["X"]+1j*obs["Y"]
                post.append(toy.check(pair[method],bundle.reference,name+method+"_complex_coherence",observable=True))
            post.append(toy.check(pair["reference"],pair["symmetric_directional"],name+"_paired",observable=True))
            # Full-unitary postchecks above cover the entire 5-qubit space,
            # including vacuum-reference columns and all leakage amplitudes.
        write(out/"postchecks.json",dict(status="PASS",checks=post,
            leakage_coverage="all raw/compiled unitary entries, including vacuum-reference columns; no phase fit"))
        preserved=all(hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha for path,sha in hashes.items()) and all(
            hashlib.sha256((Path(a.origin)/path).read_bytes()).hexdigest()==sha for path,sha in origin_hashes.items())
        if not preserved:raise ValueError("SOURCE_CHANGED")
        members=[dict(path=p.name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted((out/"local").glob("*.qpy"))]
        write(out/"payload_index.json",dict(public_payload=False,kind="artificial-only QPY",members=members))
        write(out/"STOP.json",dict(status="TECHNICAL_PASS_MANDATORY_STOP",mandatory_stop=True,compile_calls=len(cases),
            postcheck_pass=len(cases),raw_cases=len(cases),source_input_preserved=preserved,molecular_sampling=0,
            molecular_circuit_build=0,molecular_compile=0,price_refit=0,q32_execution=0,retry=0,resume=0,next_stage_authorized=False))
    except BaseException as exc:
        write(out/"STOP_FAILURE.json",dict(status="FAIL_MANDATORY_STOP",mandatory_stop=True,error=str(exc),retry=0,resume=0))
        raise
if __name__=="__main__":main()
