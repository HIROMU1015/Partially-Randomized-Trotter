#!/usr/bin/env python3
"""One fixed C1 artificial experiment; no H4 data or ongoing-job access."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import importlib.metadata
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DOC = Path("docs/research/track_d_o5_c1_head_tail_artificial_20261108")
OUT = Path("artifacts/track_d_head_tail_artificial/2026-11-08/run02_metadata_completion")
PRIOR = Path("artifacts/track_d_head_tail_artificial/2026-11-08/run01")
OLD = Path("artifacts/track_d_c7_2_toy/2026-11-07/run01")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved-artificial", type=Path, required=True)
    args = parser.parse_args()
    out = ROOT/OUT
    out.mkdir(parents=True, exist_ok=False)
    private = out/"local"; private.mkdir(mode=0o700); (private/".gitignore").write_text("*\n")
    os.environ["MPLCONFIGDIR"] = str(private/"matplotlib-cache")
    calls = dict(fixed_whole_paths=0, tail_emissions=0, local_checks=0, raw_checks=0,
        wrapper_results=0, new_compile=0, reused_compile=0, cache_hits=0, postcheck=0,
        new_H4=0, sampling=0, model_fit_N_RPE=0, other_job_operations=0)
    hashes = {}; prior_payloads = {}; prior_indices = {}; success = False
    try:
        plan = json.loads((ROOT/DOC/"PLAN.json").read_text())
        head = git("rev-parse", "HEAD").decode().strip()
        for p in plan["source_files"]:
            data = (ROOT/p).read_bytes()
            if data != git("show", head+":"+p):
                raise ValueError("UNFIXED_SOURCE:"+p)
            hashes[p] = sha(data)
        spec = json.loads((ROOT/plan["fixture_path"]).read_text())
        if sha((ROOT/plan["fixture_path"]).read_bytes()) != plan["fixture_sha256"]:
            raise ValueError("FIXED_INPUT")
        path = ROOT/"scripts/native_df_qiskit_validation_support.py"
        loader = importlib.util.spec_from_file_location("c1_head_tail_boot", path)
        boot = importlib.util.module_from_spec(loader); loader.loader.exec_module(boot)
        boot.package("trotterlib", ROOT/"src/trotterlib")
        boot.package("trotterlib.df_trotter", ROOT/"src/trotterlib/df_trotter")
        sys.path.insert(0, str(ROOT/"src"))
        from trottertracks import track_d_head_tail_artificial as m
        from trottertracks.pr_dictionary_validation.fixture import CompilerProfile
        from trottertracks.pr_dictionary_validation.gates import compile_cache_key
        profile = CompilerProfile(); profile.validate()
        if importlib.metadata.version("qiskit") != "1.3.0":
            raise ValueError("P0_VERSION")
        write(out/"SOURCE.json", dict(execution_source=head, base=plan["base"], files=hashes,
            fixture_sha256=plan["fixture_sha256"], compiler=asdict(profile), phase_bits=160,
            scope="fixed artificial q2_mixed/cut1 only", molecular_input_loaded=False))
        write(private/"resources.json", dict(affinity=len(os.sched_getaffinity(0)),
            load=os.getloadavg(), meminfo=Path("/proc/meminfo").read_text()))
        prior_source=json.loads((ROOT/PRIOR/"SOURCE.json").read_text())
        prior_calls=json.loads((ROOT/PRIOR/"CALL_COUNTS.json").read_text())
        prior_failure=json.loads((ROOT/PRIOR/"FAILURE.json").read_text())
        completed_local=json.loads((ROOT/PRIOR/"LOCAL_CHECKS.json").read_text())
        if (prior_source["execution_source"] != "da975efa9c11fcc41602d6acb56e8fd85f2ff514"
            or prior_source["fixture_sha256"] != plan["fixture_sha256"]
            or prior_calls["new_compile"] != 0 or prior_calls["raw_checks"] != 0
            or prior_failure["type"] != "AttributeError" or "to_dict" not in prior_failure["error"]
            or completed_local["status"] != "PASS"):
            raise ValueError("TECHNICAL_CONTINUATION_BINDING")
        write(out/"CONTINUATION.json",dict(prior_execution=prior_source["execution_source"],
            prior_output=str(PRIOR),preserved_prior_STOP=True,local_checks_reused=1,
            prior_local_sha256=sha((ROOT/PRIOR/"LOCAL_CHECKS.json").read_bytes()),
            reason="SimpleNamespace event metadata serialization; no prior raw checks or P0",
            raw_reconstruction_required=True,fixture_precision_scope_changed=False))
        calls["fixed_whole_paths"] = 1
        context, steps, baseline, fused, joins, basis, local = m.fixture(spec, boot.exterior_fock_lift,completed_local=completed_local)
        calls.update(tail_emissions=2, local_checks=0, local_checks_reused=1)
        write(out/"LOCAL_CHECKS.json", local)
        write(out/"INTERFACES.json", joins)
        write(out/"FIXTURE_IDENTITY.json", dict(id="q2_mixed", cut=1, q=2, modes=2,
            frame_angles=spec["frame_angles"], scalar=spec["scalar"], extracted_identity=spec["identity"],
            h="1/10", tau=spec["tau"], K=0,
            ordered_events=[[dict(rotation_component_id=e.rotation_component_id,
                rotation_angle=e.rotation_angle,signed_rotation_angle=e.unsigned_rotation_angle,
                event_normalization=e.event_normalization,phase=dict(real=e.phase.real,imag=e.phase.imag),
                applications=[dict(vars(app)) for app in e.application_sequence]) for e in s["events"]] for s in steps],
            phase_witness_in_whole_path=False, phase_chi=baseline.phase))
        raw_checks = m.raw_checks(baseline, fused, steps); calls["raw_checks"] = 2
        write(out/"RAW_CHECKS.json", dict(basis=basis, **raw_checks))
        raws = {(cfg, axis): b.wrappers[arm, axis]
            for cfg, b, arm in (("S0", baseline, "symmetric_directional"), ("S1", fused, "symmetric_directional"),
                               ("R0", baseline, "reference"), ("R1", fused, "reference")) for axis in ("X", "Y")}
        for (cfg, axis), raw in raws.items():
            m.c3.base.save_qpy(private/f"{cfg}_{axis}_raw.qpy", raw)
        for label, b in (("baseline", baseline), ("fused", fused)):
            for name in ("earlier", "later", "later_adjoint"):
                m.c3.base.save_qpy(private/f"{label}_{name}.qpy", getattr(b, name))
        key_context = dict(input=plan["fixture_sha256"], case="q2_mixed", cut=1, phase="MPFR160")
        keys = {k: compile_cache_key(v, profile, key_context) for k, v in raws.items()}
        compiled = {}; reuse = []; prior_indices = {}
        for name in ("SOURCE.json", "STOP.json", "payload_index.json", "postchecks.json"):
            saved = args.saved_artificial/name
            if saved.read_bytes() != (ROOT/OLD/name).read_bytes():
                raise ValueError("SAVED_PUBLIC_EVIDENCE_CHANGED:"+name)
            prior_indices[name] = sha(saved.read_bytes())
        old_source = json.loads((args.saved_artificial/"SOURCE.json").read_text())
        old_post = json.loads((args.saved_artificial/"postchecks.json").read_text())
        old_stop = json.loads((args.saved_artificial/"STOP.json").read_text())
        if old_post["status"] != "PASS" or not old_stop["mandatory_stop"]:
            raise ValueError("SAVED_POSTCHECK_OR_STOP")
        labels = {r["label"]: r for r in old_post["checks"]}
        for arm in ("symmetric_directional", "reference"):
            for axis in ("X", "Y"):
                if labels["q2_mixed"+arm+axis]["status"] != "PASS":
                    raise ValueError("SAVED_WRAPPER_POSTCHECK")
        index = {x["path"]:x for x in json.loads((args.saved_artificial/"payload_index.json").read_text())["members"]}
        source_compatible = (old_source["compiler"] == asdict(profile) | {"basis_gates":list(profile.basis_gates)})
        source_compatible &= all(sha((ROOT/p).read_bytes()) == h for p, h in
            {**old_source["files"], **old_source["origin_files"]}.items()
            if p.startswith("src/") and (ROOT/p).exists())
        for cfg in ("S0", "R0"):
            arm = "symmetric_directional" if cfg == "S0" else "reference"
            for axis in ("X", "Y"):
                names = [f"q2_mixed_{arm}_{axis}_{kind}.qpy" for kind in ("raw", "compiled")]
                pair = []
                for name in names:
                    p = args.saved_artificial/"local"/name
                    if not p.exists(): break
                    if sha(p.read_bytes()) != index[name]["sha256"] or p.stat().st_size != index[name]["bytes"]:
                        raise ValueError("SAVED_ARTIFICIAL_QPY_HASH:"+name)
                    prior_payloads[name] = index[name]["sha256"]
                    pair.append(m.c3.base.load_qpy(p))
                compatible = source_compatible and len(pair) == 2 and compile_cache_key(pair[0], profile, key_context) == keys[cfg,axis]
                if compatible:
                    # The immutable postchecks record is preserved and the
                    # newly loaded compiled unitary is rechecked below.
                    compiled[keys[cfg,axis]] = pair[1]; calls["reused_compile"] += 1
                reuse.append(dict(config=cfg, axis=axis, reused=compatible, source_compatible=bool(source_compatible),
                    canonical_raw_IR_match=bool(len(pair)==2 and compile_cache_key(pair[0],profile,key_context)==keys[cfg,axis]),
                    saved_raw_sha256=index[names[0]]["sha256"],saved_compiled_sha256=index[names[1]]["sha256"]))
        write(out/"REUSE.json", dict(records=reuse, source=old_source["source_commit"], public_indices=prior_indices))
        jobs = {keys[k]: v for k,v in raws.items() if keys[k] not in compiled}
        workers = min(len(jobs), len(os.sched_getaffinity(0)))
        write(out/"COMPILE_PLAN.json", dict(rows=8, unique_IR=len(set(keys.values())), new=len(jobs),
            reused=calls["reused_compile"], keys={"/".join(k):v for k,v in keys.items()}))
        if len(jobs) > 8: raise ValueError("COMPILE_CAP")
        calls["new_compile"] = len(jobs)
        if jobs:
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("fork")) as pool:
                futures = {pool.submit(m.checks.compile_one, qc): key for key,qc in jobs.items()}
                for future in as_completed(futures):
                    key = futures[future]; qc, record = future.result(); compiled[key] = qc
                    write(private/(key+"_compile.json"), record)
        calls["cache_hits"] = len(raws)-len(jobs)-calls["reused_compile"]
        rows=[]; post={}
        for cfg in ("S0","S1","R0","R1"):
            pairs={axis: compiled[keys[cfg,axis]] for axis in ("X","Y")}
            post[cfg]=m.checks.post_checks(None,pairs,baseline.reference,cfg.startswith("R"))
            for axis in ("X","Y"):
                qc=pairs[axis];raw=raws[cfg,axis]
                phase=m.toy.check(m.Operator(qc.remove_final_measurements(inplace=False)).data,
                    m.Operator(raw.remove_final_measurements(inplace=False)).data,cfg+axis+":compiled_full_unitary_phase")
                calls["postcheck"]+=1
                m.c3.base.save_qpy(private/f"{cfg}_{axis}_compiled.qpy",qc)
                rows.append(dict(configuration=cfg,axis=axis,raw=m.toy.structure(raw),compiled=m.toy.structure(qc),phase_check=phase))
        write(out/"POST_CHECKS.json",dict(status="PASS",configurations=post))
        write(out/"RESOURCES.json",rows);calls["wrapper_results"]=len(rows)
        comparisons=[]
        for axis in ("X","Y"):
            table={r["configuration"]:r["compiled"]["metrics"] for r in rows if r["axis"]==axis}
            diff=lambda a,b:{g:table[a][g]-table[b][g] for g in m.toy.METRICS}
            d0,d1=diff("R0","S0"),diff("R1","S1")
            comparisons.append(dict(axis=axis,metrics=table,strong_change=diff("S1","S0"),reference_change=diff("R1","R0"),
                reference_minus_strong_before=d0,reference_minus_strong_after=d1,relative_change={g:d1[g]-d0[g] for g in d0}))
        write(out/"COMPARISONS.json",comparisons)
        success=True
    except BaseException as error:
        write(out/"FAILURE.json",dict(type=type(error).__name__,error=str(error).replace(str(ROOT),"<worktree>").replace(str(args.saved_artificial),"<saved-artificial>")))
        raise
    finally:
        preserved=all(sha((ROOT/p).read_bytes())==h for p,h in hashes.items())
        saved_preserved=(all(sha((args.saved_artificial/p).read_bytes())==h for p,h in prior_indices.items())
            and all(sha((args.saved_artificial/"local"/p).read_bytes())==h for p,h in prior_payloads.items()))
        write(out/"CALL_COUNTS.json",calls)
        write(out/"PRESERVATION.json",dict(source_unchanged=preserved,saved_artificial_unchanged=saved_preserved,
            q64_operations=0,other_worktree_edits=0,shared_environment_edits=0))
        write(out/"PAYLOAD_INDEX.json",dict(public_payload=False,members=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p.read_bytes())) for p in sorted(private.glob("*.qpy"))]))
        write(out/"STOP.json",dict(status="TECHNICAL_PASS_MANDATORY_STOP" if success and preserved and saved_preserved else "FAIL_MANDATORY_STOP",
            source_preserved=preserved,mandatory_stop=True,next_actor="user/GPT",next_scientific_stage_authorized=False))


if __name__ == "__main__":
    main()
