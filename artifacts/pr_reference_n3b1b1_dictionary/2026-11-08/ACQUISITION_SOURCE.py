"""N3-B1b-1 preparation and one acquisition per fixed key.

Only the existing acquire/guard/word functions are executed. Each child has
its own process group; no other user, old cache or old job is controlled.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import platform
import runpy
import signal
import subprocess
import sys
import tarfile
import time
import traceback
from fractions import Fraction as F

ROOT = Path(__file__).resolve().parents[1]
BASE = "fbe7ac8b53acd5c795e94939c5c5e400ced5679a"
OUT = ROOT / "artifacts/pr_reference_n3b1b1_dictionary/2026-11-08"
B0 = "artifacts/pr_reference_n3b0_numeric/2026-11-08/"
C0 = "artifacts/pr_reference_ct4/2026-11-07/"
OLD_SOURCES = {
    "src/trottertracks/track_c_h4_synthesis.py": ["exact_word_matrix", "strict_guard", "acquire"],
    "src/trottertracks/track_c_h4.py": ["save_json"],
    "src/trottertracks/pr_reference_ct0.py": ["positive", "validate_record"],
}
POOLS = [C0 + "strict_records.tar.xz",
         "artifacts/track_c2_b_pilot/2026-11-06/raw_synthesis.tar.xz",
         "artifacts/pr_reference_ct1/2026-11-07/strict_records.tar.xz"]


def blob(path):
    return subprocess.check_output(["git", "show", BASE + ":" + path], cwd=ROOT)


def read_base(path):
    return json.loads(blob(path))


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def provider_module():
    path = ROOT / "src/trottertracks/pr_reference_dyadic_dictionary.py"
    spec = importlib.util.spec_from_file_location("n3b1b1_provider", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def old_functions():
    import fcntl
    from functools import lru_cache
    import mpmath as mp
    ns = dict(Path=Path, json=json, hashlib=hashlib, math=math, time=time,
              Fraction=F, F=F, mp=mp, OUT=OUT, fcntl=fcntl, lru_cache=lru_cache)
    for path, names in OLD_SOURCES.items():
        raw = (ROOT / path).read_bytes()
        assert raw == blob(path)
        funcs = [x for x in ast.parse(raw).body
                 if isinstance(x, ast.FunctionDef) and x.name in names]
        assert {x.name for x in funcs} == set(names)
        exec(compile(ast.Module(body=funcs, type_ignores=[]), str(ROOT/path), "exec"), ns)
    return ns


def archive_records(path):
    raw = blob(path)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:xz") as archive:
        records = [(m.name, json.load(archive.extractfile(m)))
                   for m in archive if m.isfile() and m.name.endswith(".json")]
    return raw, records


def retained_cache(validate):
    _, rows = archive_records(POOLS[0])
    adopted = read_base(C0 + "KEY_REUSE_AND_SYNTHESIS.json")
    keys = {x["key"] for x in adopted["records"]}
    result = {}
    for member, r in rows:
        validate(r)
        assert r["key"] in keys and r["precision_exponent"] == -11
        assert r.get("guard_method") == "exact cyclotomic word matrix + outward interval strict Frobenius"
        assert r["theta_hex"] not in result
        result[r["theta_hex"]] = r
    assert len(result) == len(keys) == 299
    return result


def prepare():
    if (OUT / "ACQUISITION_STARTED.json").exists():
        raise RuntimeError("Acquisition already started; do not prepare/retry it")
    OUT.mkdir(parents=True, exist_ok=True)
    p = read_base(B0 + "FIXED_PARAMETERS.json")
    budget = read_base(B0 + "ERROR_BUDGET.json")
    coverage = read_base(B0 + "BASIC_KEY_COVERAGE.json")
    assert p["b"] == 36 and p["M_b"] == 43 and p["internal_p"] == 13
    assert [x["j"] for x in p["basic_keys"]] == list(range(-36, 7))
    assert coverage["covered_basic_keys"] == 0
    assert [(x["j"], x["positive_binary64_key"]) for x in coverage["rows"]] == [
        (x["j"], x["positive_binary64_key"]) for x in p["basic_keys"]]
    keys = {x["positive_binary64_key"] for x in p["basic_keys"]}
    ns = old_functions()
    pool_checks = []
    previous_max_wall = 0.0
    for path in POOLS:
        raw, records = archive_records(path)
        matches = []
        for member, r in records:
            ns["validate_record"](r)
            previous_max_wall = max(previous_max_wall, float(r.get("wall_seconds", 0)))
            if r["theta_hex"] in keys:
                matches.append(dict(member=member, key=r["theta_hex"]))
        assert not matches
        pool_checks.append(dict(path=path, sha256=sha(raw), records=len(records),
                                target_key_matches=matches))
    cache = retained_cache(ns["validate_record"])
    mod = provider_module()
    c0 = [dict(key=key, record_key=r["key"], sequence_sha256=r["sequence_sha256"],
        strict_error_upper=r["strict_error_upper"],
        emitter_upper=str(mod.CACHE_EMITTER_UPPER),
        total_upper=str(F(r["strict_error_upper"]) + mod.CACHE_EMITTER_UPPER),
        passed=F(r["strict_error_upper"]) + mod.CACHE_EMITTER_UPPER < F(budget["epsilon"]),
        in_vacuum_domain=abs(float.fromhex(key)) <= float.fromhex(p["L_hex"]))
        for key, r in sorted(cache.items())]
    assert all(x["passed"] and x["in_vacuum_domain"] for x in c0)
    # Reuse only the runtime probe, not its old main/runner or output writer.
    runtime = runpy.run_path(str(ROOT / "scripts/verify_pr_reference_n3b1b0_runtime.py"),
                            run_name="n3b1b1_runtime_import")
    env = runtime["runtime_checks"]()
    oldenv = read_base("artifacts/pr_reference_n3b1b0_runtime/2026-11-08/ENVIRONMENT.json")
    assert env["acceleration_binary"]["sha256"] == oldenv["acceleration_binary"]["sha256"]
    assert env["qiskit_version"] == "1.3.0"
    affinity = len(os.sched_getaffinity(0))
    idle_slots = max(1, math.floor(affinity - os.getloadavg()[0]))
    workers = min(43, affinity, idle_slots)
    # Operational safeguard, not a scientific stopping-time theorem.
    # A 1h floor and 1000x historical latency avoid a low arbitrary timeout.
    deadline = max(3600.0, 1000.0 * previous_max_wall)
    sources = [*OLD_SOURCES, "scripts/verify_pr_reference_n3b1b0_runtime.py",
               "src/trottertracks/pr_reference_dyadic_dictionary.py",
               str(Path(__file__).relative_to(ROOT))]
    source_hashes = {s: sha((ROOT/s).read_bytes()) for s in sources}
    scope = dict(base_commit=BASE, scientific_conditions=p["scientific_conditions"],
        b=36, exponent_order=list(range(-36, 7)), order="ascending j in written product",
        basic_precision="1/10000000000000", final_precision=budget["epsilon"],
        backend="pygridsynth 2.0.0", seed=17, up_to_phase=False,
        acquired_word_length_limit=20000, no_retry_or_alternate_seed=True,
        join="inplace=True; whole positive word inverted once for negative; no QPY",
        workers=workers, resource_reason="min(43 independent jobs, available affinity, currently idle CPU estimate)",
        monitor=dict(poll_seconds=2, per_job_deadline_seconds=deadline,
            historical_max_acquisition_wall_seconds=previous_max_wall,
            rule="one-hour minimum and 1000 times observed old-pool maximum; fail/no-retry on expiry, worker error, corrupt output or severe memory scarcity",
            memory_rule="stop own jobs if MemAvailable falls below their combined resident bytes",
            transient_PSI="record only; no automatic PSI-only kill",
            process_scope="only newly launched child process groups"),
        source_hashes=source_hashes,
        technical_cases=dict(rounding="zero, +/-ties, adjacent grid floats, both domain endpoints, out-of-domain rejection",
            actual_word_diagnostics="all 43 basics; fixed valid composites and artificial 43-chunk bound fixture",
            max43_fixture="all basic words once; NOT a physical target/domain extension"),
        prohibited=["new keys/precision/seed/backend", "N2", "SK", "native PF build/compile",
                    "sampling/physics simulation", "vacuum price distribution", "18round total T",
                    "199-shot product", "T-depth"])
    manifest = []
    for row in p["basic_keys"]:
        angle = row["positive_binary64_key"]
        assert float.fromhex(angle).hex() == float(2.0**row["j"]).hex()
        spec = f"{angle}:10^-13:pygridsynth2.0.0:strict:seed17"
        key = sha(spec.encode())
        manifest.append(dict(j=row["j"], theta_hex=angle, spec=spec, key=key,
            precision_exponent=-13, seed=17, up_to_phase=False, status="PENDING",
            record="synthesis/"+key+".json", acquisition="acquisition/"+key+".json"))
        assert not (OUT / "synthesis" / (key+".json")).exists()
    save(OUT/"SCOPE.json", scope)
    save(OUT/"BASIC_WORDS_MANIFEST.json", dict(status="PREPARED", rows=manifest))
    save(OUT/"INPUTS.json", dict(base_commit=BASE, fixed_parameters=p,
        error_budget=budget, record_inventory=pool_checks,
        N1_exact=read_base("artifacts/pr_reference_n1_numeric/2026-11-08/RESULTS.json")["exact"]))
    save(OUT/"C0_STRICT_HEADROOM.json", dict(status="PASS", rows=c0,
        old_words_unchanged=True, new_guard_calls=0, operator_diagnostics_not_used=True))
    env.update(current_loadavg=os.getloadavg(), selected_workers=workers,
        MemAvailable=next(x for x in Path("/proc/meminfo").read_text().splitlines()
                          if x.startswith("MemAvailable:")),
        cpu_pressure=Path("/proc/pressure/cpu").read_text(),
        memory_pressure=Path("/proc/pressure/memory").read_text(),
        io_pressure=Path("/proc/pressure/io").read_text())
    save(OUT/"ENVIRONMENT.json", env)
    print(json.dumps(dict(stage="PREPARED", keys=43, workers=workers,
        C0_headroom_passed=299, deadline_seconds=deadline,
        previous_max_wall=previous_max_wall)), flush=True)


def worker(index):
    scope = json.loads((OUT/"SCOPE.json").read_text())
    for path, expected in scope["source_hashes"].items():
        assert sha((ROOT/path).read_bytes()) == expected
    m = json.loads((OUT/"BASIC_WORDS_MANIFEST.json").read_text())["rows"][index]
    path = OUT/m["acquisition"]
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents a second attempt for this key.
    with path.open("x") as f:
        json.dump(dict(status="RUNNING", key=m["key"], pid=os.getpid()), f)
    try:
        ns = old_functions()
        record = ns["acquire"](float.fromhex(m["theta_hex"]), -13)
        ns["validate_record"](record)
        assert record["key"] == m["key"]
        assert record["guard_method"] == "exact cyclotomic word matrix + outward interval strict Frobenius"
        save(path, dict(status="PASS", key=m["key"], pid=os.getpid(),
                        record=m["record"], record_sha256=sha((OUT/m["record"]).read_bytes())))
    except BaseException as exc:
        save(path, dict(status="FAILED", key=m["key"], pid=os.getpid(),
            error=repr(exc), traceback=traceback.format_exc()))
        raise


def own_rss(pid):
    try:
        return 1024 * int(next(x.split()[1] for x in Path(f"/proc/{pid}/status").read_text().splitlines()
                              if x.startswith("VmRSS:")))
    except (FileNotFoundError, StopIteration, ProcessLookupError):
        return 0


def acquire_all():
    scope = json.loads((OUT/"SCOPE.json").read_text())
    manifest = json.loads((OUT/"BASIC_WORDS_MANIFEST.json").read_text())
    assert manifest["status"] == "PREPARED"
    for path, expected in scope["source_hashes"].items():
        assert sha((ROOT/path).read_bytes()) == expected
    with (OUT/"ACQUISITION_STARTED.json").open("x") as f:
        json.dump(dict(base_commit=BASE, pid=os.getpid(), keys=43,
                       started_unix=time.time()), f)
    (OUT/"ACQUISITION_SOURCE.py").write_bytes(Path(__file__).read_bytes())
    pending = list(enumerate(manifest["rows"]))
    active = {}
    abort = None
    began = time.monotonic()
    try:
        while pending or active:
            while pending and len(active) < scope["workers"] and abort is None:
                idx, row = pending.pop(0)
                logpath = OUT/"acquisition"/(row["key"]+".log")
                logpath.parent.mkdir(parents=True, exist_ok=True)
                log = logpath.open("xb")
                proc = subprocess.Popen([sys.executable, str(Path(__file__)), "worker", str(idx)],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                    env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
                    start_new_session=True)
                active[proc.pid] = (proc, row, time.monotonic(), log)
                row["status"] = "RUNNING"
            for pid, (proc, row, start, log) in list(active.items()):
                code = proc.poll()
                if code is not None:
                    log.close()
                    del active[pid]
                    result = json.loads((OUT/row["acquisition"]).read_text()) if (
                        OUT/row["acquisition"]).exists() else dict(status="FAILED", error="No result record")
                    row["status"] = result["status"]
                    row["exit_code"] = code
                    if code != 0 or result["status"] != "PASS":
                        abort = dict(key=row["key"], reason=result.get("error", "worker failure"))
                    else:
                        raw = (OUT/row["record"]).read_bytes()
                        assert sha(raw) == result["record_sha256"]
                        row["record_sha256"] = sha(raw)
                elif time.monotonic() - start > scope["monitor"]["per_job_deadline_seconds"]:
                    abort = dict(key=row["key"], reason="Operational per-key deadline; no retry")
            mem = 1024 * int(next(x.split()[1] for x in Path("/proc/meminfo").read_text().splitlines()
                                  if x.startswith("MemAvailable:")))
            rss = sum(own_rss(pid) for pid in active)
            if active and mem < rss:
                abort = dict(reason="Severe memory scarcity: MemAvailable < own combined resident bytes")
            manifest["status"] = "FAILED_STOP" if abort else "RUNNING"
            save(OUT/"BASIC_WORDS_MANIFEST.json", manifest)
            print(json.dumps(dict(finished=sum(x["status"]=="PASS" for x in manifest["rows"]),
                active=len(active), pending=len(pending), own_RSS=rss,
                MemAvailable=mem, wall_seconds=time.monotonic()-began, abort=abort)), flush=True)
            if abort:
                break
            if active:
                time.sleep(scope["monitor"]["poll_seconds"])
    except BaseException as exc:
        abort = dict(reason=repr(exc), traceback=traceback.format_exc())
    finally:
        if abort:
            for proc, row, _, log in active.values():
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGTERM)
            for proc, row, _, log in active.values():
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                log.close()
                row["status"] = "STOPPED_AFTER_FAILURE"
            for _, row in pending:
                row["status"] = "NOT_STARTED_AFTER_FAILURE"
        manifest["status"] = "ACQUIRED" if all(x["status"]=="PASS" for x in manifest["rows"]) else "FAILED_STOP"
        save(OUT/"BASIC_WORDS_MANIFEST.json", manifest)
        save(OUT/"ACQUISITION_SUMMARY.json", dict(status=manifest["status"], abort=abort,
            succeeded=sum(x["status"]=="PASS" for x in manifest["rows"]),
            failed_or_missing=[x["theta_hex"] for x in manifest["rows"] if x["status"]!="PASS"],
            wall_seconds=time.monotonic()-began, attempts_per_key_at_most=1,
            retries=0))
    if abort:
        raise RuntimeError("Acquisition STOP; do not retry: "+str(abort))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "acquire", "worker"])
    parser.add_argument("index", nargs="?", type=int)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
    elif args.stage == "acquire":
        acquire_all()
    else:
        worker(args.index)
