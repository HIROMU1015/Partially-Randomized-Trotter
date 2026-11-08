#!/usr/bin/env python3
"""One saved H4/q4 C1 transfer; raw gate, <=4 P0/postchecks, mandatory STOP."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
DOC = Path("docs/research/track_d_o5_c1_head_tail_h4_q4_20261108")
OUT = Path("artifacts/track_d_head_tail_h4/2026-11-08/run01")
OLD = Path("artifacts/track_d_c6a_paired_price/2026-11-06/run01")
C6A = "6b6667026ec8b670f610e41141d195883a9eb513"
CONTRACT = Path("artifacts/research_c_fixed_word_pilot/2026-11-04/run01/contract.json")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def bootstrap():
    sys.dont_write_bytecode = True
    boot = load("head_tail_h4_boot", ROOT/"scripts/native_df_qiskit_validation_support.py")
    boot.package("trotterlib", ROOT/"src/trotterlib")
    boot.package("trotterlib.df_trotter", ROOT/"src/trotterlib/df_trotter")
    sys.path.insert(0, str(ROOT/"src"))
    from trottertracks import track_d_head_tail_h4 as m
    return m, boot


def numerical_adapter(threads):
    from trottertracks import track_d_c7_3_acceleration as acceleration
    from trottertracks import track_d_c8_1_nofusion as nofusion
    return nofusion.install(threads, existing_install=acceleration.install)


def wrapper_binding(qc, width, *, raw=False, axis=None):
    if qc.parameters or qc.num_qubits != width or qc.num_clbits != 1:
        raise ValueError("FULL_WRAPPER_REGISTER")
    last = qc.data[-1]
    if (last.operation.name != "measure" or qc.find_bit(last.qubits[0]).index != width-1
            or qc.find_bit(last.clbits[0]).index != 0):
        raise ValueError("ANCILLA_BIT_MAPPING")
    if raw:
        names = [x.operation.name for x in qc.data[-3:]]
        if names[-2:] != ["h", "measure"] or (axis == "Y" and names != ["sdg", "h", "measure"]):
            raise ValueError("XY_READOUT")


def qualify_saved(m, payload, record):
    from trottertracks import track_d_cut_h4_reuse as reuse
    from trottertracks.pr_dictionary_validation.fixture import CompilerProfile
    base = m.c3.base
    directory = ROOT/OLD/"paths/q4_sample00"
    rows = [r for r in read(ROOT/OLD/"prices.json") if r["sample_id"] == "q4_sample00"]
    reuse.check_request(record, {"selected": 2}, rows)
    raw = read(directory/"precompile_checks.json")
    if raw["status"] != "PASS" or raw["input_basis_columns"] != 256 or raw["full_reference_Operator_constructed"]:
        raise ValueError("OLD_RAW_SCOPE")
    reuse.checked_errors(raw["checks"])
    reuse.checked_errors(list(raw["paired_observables"].values()))
    reuse.checked_errors(list(read(directory/"paired_observables.json").values()))
    if any(not raw["omitted_chi_negative_control"][a]["omitted_phase_detected"] for a in base.AXES):
        raise ValueError("OLD_CHI")
    profile = asdict(CompilerProfile()); profile["basis_gates"] = list(profile["basis_gates"])
    audit = []
    for row in rows:
        name = row["method"]+"_"+row["axis"]
        comp, check = read(directory/(name+"_compile.json")), read(directory/(name+"_check.json"))
        if comp["status"] != "PASS" or comp["compile_calls"] != 1 or comp["compiler"] != profile or row["compiler"] != profile:
            raise ValueError("OLD_P0_PROFILE")
        if check["status"] != "PASS" or check["global_phase_fit"] or (check["arm"], check["axis"]) != (row["method"], row["axis"]):
            raise ValueError("OLD_POSTCHECK")
        reuse.checked_errors(check["checks"])
        width = 17 if row["method"] == "reference" else 9
        for kind in ("raw", "compiled"):
            qc = base.load_qpy(payload/"circuits"/(name+"_"+kind+".qpy"))
            wrapper_binding(qc, width, raw=kind == "raw", axis=row["axis"])
            if kind == "compiled":
                metrics = base.artificial.metrics(qc)
                if metrics != check["metrics"] or any(metrics[k] != v for k, v in dict(row["metrics"], qubit_count=row["peak_qubits"]).items()):
                    raise ValueError("OLD_QPY_PRICE")
        audit.append(dict(configuration="R0" if width == 17 else "S0", method=row["method"],
                          axis=row["axis"], metrics=metrics, status="QUALIFIED_REUSE"))
    return dict(status="PASS", wrappers=audit, old_raw_PASS=1, old_compiled_PASS=4,
                canonical_raw_IR="pending reconstruction before ON emission", phase_fit=False)


def compiled_task(method, axis, threads, reuse_old):
    m, _ = bootstrap(); base = m.c3.base
    numerical_adapter(threads)
    import numpy as np
    from threadpoolctl import threadpool_limits
    from trottertracks.c1_h4_native_fair import compile_checked
    from trottertracks.pr_dictionary_validation.fixture import CompilerProfile
    out = ROOT/OUT; private = out/"local"; name = method+"_"+axis
    profile = CompilerProfile(); raw = base.load_qpy(private/"circuits"/(name+"_raw.qpy"))
    if reuse_old:
        compiled = base.load_qpy(private/"saved/circuits"/(name+"_compiled.qpy"))
        write(out/(name+"_compile.json"), dict(status="REUSED_IDENTICAL_OLD_IR", compile_calls=0, compiler=asdict(profile)))
    else:
        write(out/(name+"_compile_started.json"), dict(compile_calls=1, method=method, axis=axis))
        with threadpool_limits(limits=1):
            compiled = compile_checked(raw, profile)
        write(out/(name+"_compile.json"), dict(status="PASS", compile_calls=1, compiler=asdict(profile)))
    base.save_qpy(private/"circuits"/(name+"_compiled.qpy"), compiled)
    wrapper_binding(compiled, 17 if method == "reference" else 9)
    checks = load("head_tail_post_checks", ROOT/"tests/track_d_h4_reference_checks.py")
    with np.load(private/"references.npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    write(out/(name+"_post_started.json"), dict(new_postcheck=1))
    with threadpool_limits(limits=1, user_api="blas"):
        obs, report = checks.check_compiled(compiled, method, axis, arrays, base, threads)
    np.savez_compressed(private/(name+"_observable.npz"), observable=obs)
    report.update(method=method, axis=axis, metrics=base.artificial.metrics(compiled))
    write(out/(name+"_post.json"), report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--origin", type=Path, required=True)
    args = parser.parse_args()
    out = ROOT/OUT; out.mkdir(parents=True, exist_ok=False)
    private = out/"local"; private.mkdir(mode=0o700)
    (out/".gitignore").write_text("/local/\n")
    os.environ["MPLCONFIGDIR"] = str(private/"matplotlib-cache")
    os.environ["QISKIT_PARALLEL"] = "false"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "RAYON_NUM_THREADS"):
        os.environ[key] = "1"
    calls = dict(saved_paths=0, event_decode=0, new_sampling=0, fused_S1_core=0, fused_R1_core=0,
                 new_full_wrappers=0, raw_checks=0, new_P0_compile=0, new_compiled_postcheck=0,
                 reused_baseline_wrappers=0, new_ground_DF_fit=0, model_fit_N_13round=0, science_retry=0)
    success = False; hashes = {}; preserved = {}
    try:
        m, boot = bootstrap(); base = m.c3.base
        from trottertracks import track_d_cut_h4_saved as saved
        from trottertracks.pr_dictionary_validation.fixture import CompilerProfile
        import numpy as np
        from threadpoolctl import threadpool_limits
        plan = read(ROOT/DOC/"PLAN.json")
        head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"]).decode().strip()
        for name in plan["source_files"]:
            data = (ROOT/name).read_bytes()
            if data != subprocess.check_output(["git", "-C", str(ROOT), "show", head+":"+name]):
                raise ValueError("UNFIXED_SOURCE:"+name)
            hashes[name] = sha(data)
        for name in plan["unchanged_C6a_files"]:
            if (ROOT/name).read_bytes() != subprocess.check_output(["git", "-C", str(ROOT), "show", C6A+":"+name]):
                raise ValueError("C6A_SOURCE_CHANGED:"+name)
        contract = read(ROOT/CONTRACT)
        for name in (contract[k] for k in ("input", "candidate", "sector", "atoms")):
            data = (args.origin/name).read_bytes()
            if sha(data) != contract["saved_inputs"][name]:
                raise ValueError("FIXED_INPUT:"+name)
            preserved[str(args.origin/name)] = sha(data)
        law = "artifacts/research_c_near_practical_r32/2026-11-03/run01/PR/A/law.json"
        if sha((args.origin/law).read_bytes()) != plan["law_sha256"]:
            raise ValueError("FIXED_LAW")
        preserved[str(args.origin/law)] = plan["law_sha256"]
        preserved[str(args.archive)] = sha(args.archive.read_bytes())
        manifest = read(ROOT/OLD/"paths/q4_sample00/payload_archive.json")
        binding = next(r for r in read(ROOT/OLD/"event_bindings.json") if r["sample_id"] == "q4_sample00")
        if manifest["sha256"] != plan["archive_sha256"] or binding["record_sha256"] != plan["events_sha256"]:
            raise ValueError("FIXED_PUBLIC_BINDING")
        versions = {k: importlib.metadata.version(k) for k in plan["versions"]}
        if versions != plan["versions"]:
            raise ValueError("FIXED_ENVIRONMENT")
        import threadpoolctl
        if sha(Path(threadpoolctl.__file__).read_bytes()) != plan["threadpoolctl_sha256"]:
            raise ValueError("FIXED_PRIVATE_DEPENDENCY")
        write(out/"SOURCE.json", dict(execution_source=head, files=hashes, plan=plan,
              compiler=asdict(CompilerProfile()), versions=versions, source_fixed_before_event_decode=True))
        payload = saved.checked_payload(args.archive, manifest)
        destination = private/"saved"; destination.mkdir()
        for name, content in payload.items():
            path = destination/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(content)
        record = saved.checked_record(payload["events.json"], binding, C6A, m.c3.paired.ENDPOINT,
                                       plan["input_packet_sha256"], plan["law_sha256"])
        calls.update(saved_paths=1, event_decode=1)
        write(out/"INPUT_CHECK.json", dict(status="PASS", archive_sha256=manifest["sha256"], members_verified=len(payload),
              events_sha256=binding["record_sha256"], endpoint=record["endpoint"], input_packet_sha256=record["input_packet_sha256"],
              law_sha256=record["law_sha256"], sample_id="q4_sample00", q=4, cut=2, events_per_step=32))
        reuse = qualify_saved(m, destination, record)
        calls["reused_baseline_wrappers"] = 4
        write(out/"BASELINE_REUSE.json", reuse)
        resource_module = load("head_tail_resources", ROOT/"scripts/run_track_d_b1_c1.py")
        resources = resource_module.cpu_observation()
        budget = resources["numerical_threads"]
        write(private/"resources.json", resources)
        engine = numerical_adapter(budget)
        with threadpool_limits(limits=1, user_api="blas"):
            context = m.c3.paired.prepare_common(args.origin, contract)
            print("Saved input/reuse PASS; constructing single S1/R1", flush=True)
            def count(kind):
                calls[kind] += 1
                if calls[kind] > (4 if kind == "new_full_wrappers" else 1):
                    raise ValueError("SCIENTIFIC_CALL_CAP:"+kind)
            baseline, fused, interfaces, indices = m.prepare(context, record, destination, on_call=count)
            write(out/"CANONICAL_IR.json", dict(status="PASS", cores=2, halves_and_actual_inverse=3,
                  saved_OFF_wrappers_reused=4, newly_generated_OFF_wrappers=0, declared_phase_included=True))
            local = []
            for index in sorted(set(indices)):
                gh, gt = m.certified_basis(context, 2), m.certified_basis(context, index)
                bridge = context.builder.transition(context.arm.model, context.arm.model.blocks[2], context.arm.model.blocks[index])
                actual = engine(bridge)
                local.append(m.old.toy.check(actual, engine(gt).conj().T @ engine(gh), f"H4_bridge_{index}:full_phase"))
                rotation = np.asarray(context.arm.model.blocks[index].rotation).conj().T @ np.asarray(context.arm.model.blocks[2].rotation)
                local.append(m.old.toy.check(actual, np.exp(1j*m.vacuum_phase(bridge))*boot.exterior_fock_lift(rotation),
                                             f"H4_bridge_{index}:independent_exterior_lift"))
            write(out/"LOCAL_CHECKS.json", dict(status="PASS", checks=local, phase_fit=False))
            write(private/"interfaces.json", interfaces)
            public_interfaces = [{k: v for k, v in r.items() if k not in ("head_rotation", "tail_rotation")} | {
                k+"_sha256": sha(json.dumps(r[k]).encode()) for k in ("head_rotation", "tail_rotation")} for r in interfaces]
            write(out/"INTERFACES.json", dict(fused=public_interfaces, refused=[], scope="head exit -> first tail full-frame entry only"))
            private.joinpath("circuits").mkdir()
            checks = load("head_tail_raw_checks", ROOT/"tests/track_d_h4_reference_checks.py")
            calls["raw_checks"] = 1
            print("Local bridges and canonical OFF IR PASS; all256 raw START", flush=True)
            # save_and_check creates circuits; use a dedicated fresh raw directory.
            rawdir = private/"raw"; rawdir.mkdir()
            report = m.c3.save_and_check(fused, rawdir, checks, budget)
            write(out/"RAW_CHECKS.json", report)
            for path in (rawdir/"circuits").glob("*.qpy"):
                (private/"circuits"/path.name).write_bytes(path.read_bytes())
            (private/"references.npz").write_bytes((rawdir/"references.npz").read_bytes())
            from trottertracks.c1_stage3_molecular import component_check
            equality = []
            with np.load(rawdir/"references.npz", allow_pickle=False) as z, np.load(destination/"references.npz", allow_pickle=False) as off:
                for key in ("A", "B", "A_adjoint", "raw_core_symmetric_directional"):
                    equality.append(component_check(z[key], off[key], 1e-10, label="S1_R1_vs_saved_OFF:"+key))
                s = z["raw_core_symmetric_directional"]; W0, W1 = s[:256, :256], s[256:, 256:]
                equality.append(component_check(W0.conj().T @ W1, z["V"], 1e-10, label="Strong_W0_adjoint_W1_equals_V"))
            write(out/"RAW_EQUALITY.json", dict(status="PASS", checks=equality, rtol=0, phase_fit=False))
        print("All256 raw PASS; at most four new P0 wrappers START", flush=True)
        threads = max(1, budget//4)
        results = []
        with ProcessPoolExecutor(max_workers=min(4, budget), mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = []
            for method in ("symmetric_directional", "reference"):
                for axis in base.AXES:
                    identical = m.old.same_ir(fused.wrappers[method, axis], baseline.wrappers[method, axis])
                    futures.append(pool.submit(compiled_task, method, axis, threads, identical))
            for future in as_completed(futures):
                row = future.result(); results.append(row)
                print(row["method"], row["axis"], "postcheck PASS", flush=True)
        write(out/"POST_CHECKS.json", dict(status="PASS", wrappers=results))
        with threadpool_limits(limits=1):
            observables = {}
            for row in results:
                name = row["method"]+"_"+row["axis"]
                with np.load(private/(name+"_observable.npz"), allow_pickle=False) as z:
                    observables[row["method"], row["axis"]] = z["observable"].copy()
            paired = []
            for axis in base.AXES:
                row = component_check(observables["reference", axis], observables["symmetric_directional", axis], 1e-10, label="compiled_paired_"+axis)
                if row["frobenius"] > 2e-10:
                    raise ValueError("COMPILED_PAIRED_SIGNAL")
                paired.append(row)
            for method in ("symmetric_directional", "reference"):
                coherence = observables[method, "X"]+1j*observables[method, "Y"]
                row = component_check(coherence, fused.reference, 1e-10, label=method+":compiled_full_complex_coherence")
                if row["frobenius"] > 2e-10:
                    raise ValueError("COMPILED_COHERENCE_SIGNAL")
                paired.append(row)
                paired.append(component_check(np.array([coherence[0, 0]]), np.array([np.exp(1j*fused.phase)]),
                                              1e-10, label=method+":analytic_chi"))
        write(out/"PAIRED_AND_CHI.json", dict(status="PASS", checks=paired))
        rows = reuse["wrappers"]+[
            dict(configuration="R1" if r["method"] == "reference" else "S1", method=r["method"], axis=r["axis"], metrics=r["metrics"], status="PASS") for r in results]
        write(out/"RESOURCES.json", rows)
        comparisons = []
        for axis in base.AXES:
            table = {r["configuration"]: r["metrics"] for r in rows if r["axis"] == axis}
            comparisons.append(dict(axis=axis, **table,
                delta_S={k: table["S1"][k]-table["S0"][k] for k in table["S0"]},
                delta_R={k: table["R1"][k]-table["R0"][k] for k in table["S0"]},
                relative_before={k: table["R0"][k]-table["S0"][k] for k in table["S0"]},
                relative_after={k: table["R1"][k]-table["S1"][k] for k in table["S0"]},
                difference_in_differences={k: table["R1"][k]-table["S1"][k]-table["R0"][k]+table["S0"][k] for k in table["S0"]}))
        write(out/"COMPARISONS.json", comparisons)
        if any(sha((ROOT/p).read_bytes()) != h for p, h in hashes.items()) or any(sha(Path(p).read_bytes()) != h for p, h in preserved.items()):
            raise ValueError("SOURCE_INPUT_PRESERVATION")
        write(out/"PRESERVATION.json", dict(status="PASS", source_files=len(hashes), input_files=len(preserved),
              private_payload_public=False, old_SOURCE_results_STOP_modified=False, other_jobs_modified=False))
        success = True
    except BaseException as error:
        write(private/"failure_private.json", dict(type=type(error).__name__, error=str(error), traceback=traceback.format_exc()))
        message = str(error).replace(str(ROOT), "<task>").replace(str(args.origin), "<private-origin>").replace(str(args.archive), "<private-archive>")
        write(out/"FAILURE.json", dict(type=type(error).__name__, error=message))
        print(type(error).__name__, message, flush=True)
    calls["new_P0_compile"] = len(list(out.glob("*_compile_started.json")))
    calls["new_compiled_postcheck"] = len(list(out.glob("*_post_started.json")))
    write(out/"CALL_COUNTS.json", calls)
    write(out/"PAYLOAD_INDEX.json", [dict(path=str(p.relative_to(private)), bytes=p.stat().st_size, sha256=sha(p.read_bytes()))
          for p in sorted(private.rglob("*")) if p.is_file() and p.suffix in (".qpy", ".npz", ".json") and "matplotlib-cache" not in p.parts])
    write(out/"STOP.json", dict(status="PASS_MANDATORY_STOP" if success else "FAIL_MANDATORY_STOP", mandatory_stop=True,
          new_compile=calls["new_P0_compile"], sample_id="q4_sample00", next_stage_authorized=False, retry=0))
    print("PASS_MANDATORY_STOP" if success else "FAIL_MANDATORY_STOP", flush=True)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
