#!/usr/bin/env python3
"""Explicit frozen artificial-only cut experiment; no molecular input or RNG."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import multiprocessing
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time
import traceback
import unittest

ROOT=Path(__file__).resolve().parents[1]
PLAN="docs/research/track_d_cut_artificial_20261108/PLAN.json"

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n")

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module

def sha(data):return hashlib.sha256(data).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",required=True);a=p.parse_args()
    out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();compile_calls=0;rows=[]
    try:
        plan=json.loads((ROOT/PLAN).read_text())
        boot=load("cut_boot",ROOT/"scripts/native_df_qiskit_validation_support.py")
        boot.package("trotterlib",ROOT/"src/trotterlib")
        boot.package("trotterlib.df_trotter",ROOT/"src/trotterlib/df_trotter")
        sys.path.insert(0,str(ROOT/"src"))
        from trottertracks import track_d_cut_artificial as art
        from trottertracks import track_d_cut_selection as selector
        from trottertracks.c1_h4_native_fair import compile_checked
        from trottertracks.pr_dictionary_validation.fixture import CompilerProfile
        from trottertracks.pr_dictionary_validation.gates import compile_cache_key
        profile=CompilerProfile();profile.validate()
        if importlib.metadata.version("qiskit")!=profile.qiskit:raise ValueError("P0_VERSION")
        feature_bytes=subprocess.check_output(["git","-C",str(ROOT),"show",
            plan["feature_source_commit"]+":"+plan["feature_source_path"]])
        # Read immutable Git bytes; no fit/old campaign entry point is invoked.
        with tempfile.TemporaryDirectory(prefix="cut-feature-") as temporary:
            fixed=Path(temporary)/"features.py";fixed.write_bytes(feature_bytes)
            features=load("trottertracks._cut_frozen_features",fixed).event_features
        tests=load("cut_artificial_tests",ROOT/"tests/check_track_d_cut_selection.py")
        tests.CutTests.features=staticmethod(features)
        paths={PLAN,plan["reuse_toy_spec"],"scripts/run_track_d_cut_artificial.py",
               "tests/check_track_d_cut_selection.py","scripts/native_df_qiskit_validation_support.py"}
        # Snapshot actual repository Python dependencies, including compiler and
        # the toy builder; no saved molecular payload is opened or hashed.
        for module in list(sys.modules.values()):
            filename=getattr(module,"__file__",None)
            if filename:
                path=Path(filename).resolve()
                if path.is_relative_to(ROOT) and path.suffix==".py":paths.add(str(path.relative_to(ROOT)))
        hashes={name:sha((ROOT/name).read_bytes()) for name in sorted(paths)}
        versions={name:importlib.metadata.version(name) for name in
                  ("numpy","qiskit","scipy","mpmath","symengine")}
        # threadpoolctl is unused by this artificial Operator/compile route.
        try:versions["threadpoolctl"]=importlib.metadata.version("threadpoolctl")
        except importlib.metadata.PackageNotFoundError:versions["threadpoolctl"]="NOT_INSTALLED_UNUSED"
        resources=dict(affinity=len(os.sched_getaffinity(0)),load=list(os.getloadavg()),
            meminfo=Path("/proc/meminfo").read_text(),own_pid=os.getpid(),
            process_changes="only this runner and its new artificial compile children")
        write(out/"SOURCE.json",dict(scope="ARTIFICIAL_ONLY",base=plan["parent"],
            source_sha256=hashes,plan_sha256=hashes[PLAN],
            feature=dict(commit=plan["feature_source_commit"],path=plan["feature_source_path"],
                         sha256=sha(feature_bytes)),python=sys.version,executable=sys.executable,
            versions=versions,profile=asdict(profile),phase_compiler="existing MPFR160 declared-global-phase",
            command=[str(ROOT/"scripts/run_track_d_cut_artificial.py"),"--output",str(out)],
            scientific_boundary="No H4 I/O, saved-price reanalysis, state/Born/RNG/shot, fit or N accounting"))
        write(out/"RESOURCE_START.json",resources)
        stream=io.StringIO()
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(tests.CutTests))
        (out/"TESTS.txt").write_text(stream.getvalue())
        if not result.wasSuccessful():raise ValueError("ARTIFICIAL_UNIT_TESTS")
        spec=json.loads((ROOT/plan["reuse_toy_spec"]).read_text())
        bundles={};checks={};decisions={};descriptors={};lanes={};raws={}
        for case in plan["cases"]:
            name=case["id"]
            _,steps,center,adjusted,decision,initial=art.fixture(spec,case,boot.exterior_fock_lift,features)
            bundles[name]=(center,adjusted);decisions[name]=asdict(decision)
            descriptors[name]=selector.descriptors([s["events"] for s in steps])
            checks[name]=dict(basis=initial,center=art.raw_checks(center,steps,decision.center),
                selected=art.raw_checks(adjusted,steps,decision.selected),invariance=[
                    art.toy.check(center.reference,adjusted.reference,"cut_independent_ordered_V"),
                    art.toy.check(art.Operator(center.cores["symmetric_directional"]).data,
                                  art.Operator(adjusted.cores["symmetric_directional"]).data,
                                  "unchanged_strong_baseline")])
            for method,bundle,arm in (("strong",center,"symmetric_directional"),
                ("reference_center",center,"reference"),("reference_selected",adjusted,"reference")):
                for axis in plan["axes"]:raws[name,method,axis]=bundle.wrappers[arm,axis]
            lanes[name]={label:dict(A_inverse=art.toy.structure(b.later_adjoint),
                                    B=art.toy.structure(b.earlier),cut=b.identity["cut"],
                                    compiled=False) for label,b in (("center",center),("selected",adjusted))}
        # These are durable and complete BEFORE the first price/compile call.
        write(out/"DECISIONS.json",decisions);write(out/"DESCRIPTORS.json",descriptors)
        write(out/"RAW_CHECKS.json",dict(status="PASS",fixtures=checks))
        write(out/"RAW_LANES.json",lanes)
        jobs={};keys={};(out/"qpy").mkdir()
        for label,raw in raws.items():
            key=compile_cache_key(raw,profile,dict(scope="O5_frozen_artificial",case=label[0],
                plan_sha256=hashes[PLAN],phase="MPFR160"))
            keys[label]=key;jobs.setdefault(key,raw)
            art.toy.c3.base.save_qpy(out/"qpy"/("_".join(label)+"_raw.qpy"),raw)
        if len(raws)!=plan["result_rows"] or len(jobs)>plan["compile_cap"]:raise ValueError("FINITE_SCOPE")
        workers=min(len(jobs),len(os.sched_getaffinity(0)))
        write(out/"COMPILE_PLAN.json",dict(raw_rows=len(raws),unique_ir=len(jobs),workers=workers,
            cache_hits=len(raws)-len(jobs),keys={"/".join(k):v for k,v in keys.items()}))
        compiled={};compile_records={}
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("fork")) as pool:
            futures={pool.submit(art.compile_one,raw):key for key,raw in jobs.items()}
            compile_calls=len(futures)
            for future in as_completed(futures):
                key=futures[future];c,record=future.result()
                compiled[key]=c;compile_records[key]=record
                write(out/"compile"/(key+".json"),record)
                art.toy.c3.base.save_qpy(out/"qpy"/(key+"_compiled.qpy"),c)
                print("P0 PASS",len(compiled),"/",len(jobs),flush=True)
        post={}
        for case in plan["cases"]:
            name=case["id"];post[name]={};V=bundles[name][0].reference
            for method in plan["methods"]:
                pair={axis:compiled[keys[name,method,axis]] for axis in plan["axes"]}
                post[name][method]=art.post_checks(None,pair,V,method!="strong")
                for axis in plan["axes"]:
                    label=(name,method,axis);key=keys[label]
                    row=dict(case=name,method=method,axis=axis,cache_key=key,
                        raw=art.toy.structure(raws[label]),compiled=art.toy.structure(compiled[key]),
                        compile_record=compile_records[key])
                    rows.append(row);write(out/"rows"/("_".join(label)+".json"),row)
        write(out/"POST_CHECKS.json",dict(status="PASS",fixtures=post))
        comparisons=[]
        for case in plan["cases"]:
            for axis in plan["axes"]:
                table={r["method"]:r["compiled"]["metrics"] for r in rows if r["case"]==case["id"] and r["axis"]==axis}
                comparisons.append(dict(case=case["id"],axis=axis,metrics=table,
                    selected_minus_center={m:table["reference_selected"][m]-table["reference_center"][m] for m in art.toy.METRICS},
                    selected_minus_strong={m:table["reference_selected"][m]-table["strong"][m] for m in art.toy.METRICS}))
        write(out/"COMPARISONS.json",comparisons)
        if any(sha((ROOT/name).read_bytes())!=value for name,value in hashes.items()):raise ValueError("SOURCE_CHANGED")
        write(out/"STOP.json",dict(status="TECHNICAL_PASS_MANDATORY_STOP",mandatory_stop=True,
            unit_tests=result.testsRun,artificial_fixtures=len(bundles),result_rows=len(rows),compile_calls=compile_calls,
            cache_hits=len(raws)-len(jobs),source_preserved=True,molecular_input=0,molecular_sampling=0,
            molecular_build=0,molecular_compile=0,price_refit=0,N_recompute=0,other_job_operations=0,
            positive_artificial_depth_cases=sorted({r["case"] for r in comparisons if r["selected_minus_center"]["total_depth"]<0}),
            next_science_authorized=False,wall_seconds=time.perf_counter()-start,
            parent_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            max_child_peak_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss))
    except BaseException as exc:
        write(out/"STOP_FAILURE.json",dict(status="FAIL_MANDATORY_STOP",mandatory_stop=True,
            error=str(exc),traceback=traceback.format_exc(),compile_calls=compile_calls,result_rows=len(rows)))
        raise

if __name__=="__main__":main()
