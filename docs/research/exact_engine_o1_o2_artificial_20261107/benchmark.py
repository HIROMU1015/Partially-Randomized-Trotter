"""Reproducible artificial-only benchmark; no molecular input option exists."""
import argparse
from dataclasses import asdict
from fractions import Fraction as F
import gc
import hashlib
from itertools import permutations
import json
import os
from pathlib import Path
import platform
import resource
import socket
import statistics
import subprocess
import sys
import time
import tracemalloc
import unittest

import fixtures as fx
from oracle import old, FIXED_COMMIT, SOURCE_PATH, SOURCE_SHA256
import prototype as p
import test_o1_o2

METHODS = ("dense", "O1", "O1+O2")
CATEGORIES = ("warm_contraction", "warm_end_to_end", "cold_end_to_end")
REPEATS = 6  # All six method permutations, once each, for every fixture/category.
HERE = Path(__file__).resolve().parent


def json_text(value):
    return json.dumps(value, default=lambda x: str(x) if isinstance(x, F) else vars(x),
                      ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"


def write(path, value):
    with Path(path).open("x") as stream:
        stream.write(json_text(value))


def memory_status():
    return {k: Path(v).read_text() for k, v in (
        ("meminfo", "/proc/meminfo"), ("memory_pressure", "/proc/pressure/memory"))}


def environment():
    model = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                  if line.startswith("model name")), "UNKNOWN")
    return dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                host=socket.gethostname(), processor=model, affinity=sorted(os.sched_getaffinity(0)),
                load=os.getloadavg(), resources=memory_status(), dependencies="Python standard library only",
                fractions_source_sha256=hashlib.sha256(Path(sys.modules["fractions"].__file__).read_bytes()).hexdigest(),
                scientific_modules_loaded=[name for name in sys.modules
                                           if name.split(".")[0] in ("numpy", "qiskit", "mpmath", "trotterlib", "trottertracks")])


def context(fixture, method, warm=False):
    rows = tuple(map(p.prepare_matrix, fixture.matrices)) if method != "dense" else None
    cache = p.IndexCache() if method == "O1+O2" else None
    if warm and cache is not None:
        for mi, qs in fixture.gates:
            if rows[mi].nnz != len(rows[mi].matrix)**2:
                cache.get(len(fixture.initial), qs)
    return rows, cache


def evaluate(fixture, method, ctx, with_born):
    rows, cache = ctx
    v = fixture.initial
    for mi, qs in fixture.gates:
        if method == "dense":
            v = old.apply_exact(v, fixture.matrices[mi], qs)
        elif method == "O1":
            v = p.apply_o1(v, rows[mi], qs)
        else:
            v = p.apply_o1_o2(v, rows[mi], qs, cache)
    threshold = old.threshold_count(old.born_probability(v, fixture.ancilla)) if with_born else None
    return v, threshold


def run_once(fixture, method, category, warm_ctx):
    ctx = context(fixture, method) if category == "cold_end_to_end" else warm_ctx
    return evaluate(fixture, method, ctx, category != "warm_contraction"), ctx


def timed(call, iterations=1):
    wall = time.perf_counter_ns()
    cpu = time.process_time_ns()
    for _ in range(iterations):
        result = call()
    cpu = time.process_time_ns() - cpu
    wall = time.perf_counter_ns() - wall
    return result, dict(wall_ns=wall, cpu_ns=cpu, iterations=iterations)


def append(path, row):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def fixture_metadata(fixture):
    values = [x for z in fixture.initial for x in z]
    return dict(name=fixture.name, dimension=len(fixture.initial), gates=len(fixture.gates),
                supports=[list(qs) for _, qs in fixture.gates],
                matrix_nnz=[sum(z != fx.Z for row in m for z in row) for m in fixture.matrices],
                matrix_dimensions=[len(m) for m in fixture.matrices],
                initial_max_numerator_bits=max(abs(x.numerator).bit_length() for x in values),
                initial_max_denominator_bits=max(x.denominator.bit_length() for x in values))


def memory_sample(fixture, method):
    gc.collect()
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    tracemalloc.start()
    result, ctx = run_once(fixture, method, "cold_end_to_end", None)
    retained, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    return dict(fixture=fixture.name, method=method, category="cold_end_to_end",
                python_retained_bytes=retained, python_peak_bytes=peak,
                RSS_high_water_before_bytes=before, RSS_high_water_after_bytes=after,
                cache=ctx[1].info() if ctx[1] is not None else None,
                threshold=result[1], pid=os.getpid(),
                note="fresh subprocess; fixture/import memory excluded from tracemalloc; RSS is process high-water")


def summarize(out):
    records = [json.loads(line) for line in (out / "timings.jsonl").read_text().splitlines()]
    summary = []
    for fixture in fx.benchmark_fixtures():
        for category in CATEGORIES:
            medians = {}
            groups = {}
            for method in METHODS:
                rows = [r for r in records if (r["fixture"], r["category"], r["method"]) == (fixture.name, category, method)]
                assert len(rows) == REPEATS
                wall = [r["wall_ns"] / r["iterations"] for r in rows]
                cpu = [r["cpu_ns"] / r["iterations"] for r in rows]
                medians[method] = statistics.median(wall)
                groups[method] = dict(wall_median_ns=medians[method], wall_min_ns=min(wall), wall_max_ns=max(wall),
                                      cpu_median_ns=statistics.median(cpu), samples=len(rows))
            summary.append(dict(fixture=fixture.name, category=category, methods=groups,
                                dense_over_O1=medians["dense"] / medians["O1"],
                                dense_over_O1O2=medians["dense"] / medians["O1+O2"],
                                O1_over_O1O2=medians["O1"] / medians["O1+O2"]))
    return summary


def run(out):
    out.mkdir(exist_ok=False)
    source_files = ("oracle.py", "prototype.py", "fixtures.py", "test_o1_o2.py", "benchmark.py")
    source = {f: hashlib.sha256((HERE / f).read_bytes()).hexdigest() for f in source_files}
    write(out / "SOURCE.json", dict(design_commit="fb28b6a287803ce63a51a1513a57828302003090",
                                   oracle_commit=FIXED_COMMIT, oracle_path=SOURCE_PATH,
                                   oracle_sha256=SOURCE_SHA256, files_sha256=source))
    write(out / "ENVIRONMENT.json", environment())
    if environment()["scientific_modules_loaded"]:
        raise RuntimeError("unexpected scientific dependency import")
    fixtures = fx.benchmark_fixtures()
    write(out / "FIXTURES.json", [asdict(f) for f in fixtures])
    write(out / "PLAN.json", dict(status="FROZEN_BEFORE_MEASUREMENT", fixtures=[fixture_metadata(f) for f in fixtures],
                                 methods=METHODS, categories=CATEGORIES, repeats=REPEATS,
                                 method_orders=list(permutations(METHODS)), gates_per_sequence=12,
                                 warmup_per_fixture_category_method=1, preprocessing_timer_iterations=50,
                                 parallelism="single process for controlled kernel comparison; no scientific worker cap",
                                 correctness_outside_timing=True, memory_in_fresh_subprocess=True,
                                 scientific_inputs=False, RNG=False, build_compile=False))
    with (out / "TESTS.log").open("x") as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_o1_o2))
    write(out / "CORRECTNESS.json", dict(tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                                        success=result.wasSuccessful(), comparisons_per_gate=True, artificial_only=True))
    if not result.wasSuccessful():
        raise RuntimeError("correctness failed before benchmark")
    for fixture in fixtures:
        prepared = tuple(map(p.prepare_matrix, fixture.matrices))
        for repeat in range(REPEATS):
            _, sample = timed(lambda: tuple(map(p.prepare_matrix, fixture.matrices)), 50)
            append(out / "preprocessing.jsonl", dict(fixture=fixture.name, repeat=repeat, category="sparse_rows", **sample))
            keys = tuple(dict.fromkeys(qs for mi, qs in fixture.gates if prepared[mi].nnz != len(prepared[mi].matrix)**2))
            if keys:
                _, sample = timed(lambda: tuple(p.build_index_plan(len(fixture.initial), qs) for qs in keys), 50)
                append(out / "preprocessing.jsonl", dict(fixture=fixture.name, repeat=repeat, category="index_build", **sample))
                cache = p.IndexCache()
                for qs in keys:
                    cache.get(len(fixture.initial), qs)
                _, sample = timed(lambda: tuple(cache.get(len(fixture.initial), qs) for qs in keys), 50)
                append(out / "preprocessing.jsonl", dict(fixture=fixture.name, repeat=repeat, category="index_warm_lookup", **sample))
        for category in CATEGORIES:
            contexts = {method: context(fixture, method, warm=True) for method in METHODS}
            for method in METHODS:
                run_once(fixture, method, category, contexts[method])  # explicit unmeasured warmup
            for repeat, order in enumerate(permutations(METHODS)):
                for position, method in enumerate(order):
                    gc.collect()  # outside measured interval; GC remains enabled during kernels
                    (result, ctx), sample = timed(lambda: run_once(fixture, method, category, contexts[method]))
                    append(out / "timings.jsonl", dict(fixture=fixture.name, method=method, category=category,
                                                        repeat=repeat, position=position, method_order=order,
                                                        cache=ctx[1].info() if ctx[1] is not None else None, **sample))
        print(fixture.name + ": time measurements complete", flush=True)
    for fixture in fixtures:
        for method in METHODS:
            child = subprocess.check_output([sys.executable, "-B", str(Path(__file__).resolve()),
                                             "--memory", fixture.name, method], text=True)
            append(out / "memory.jsonl", json.loads(child))
    write(out / "SUMMARY.json", summarize(out))
    write(out / "RESOURCE_END.json", memory_status())
    current = {f: hashlib.sha256((HERE / f).read_bytes()).hexdigest() for f in source_files}
    if current != source:
        raise RuntimeError("benchmark SOURCE changed during run")
    write(out / "STOP.json", dict(status="ARTIFICIAL_BENCHMARK_COMPLETE", scientific_authorization=False,
                                 no_scientific_execution=True, old_engine_modified=False,
                                 mandatory_STOP_after_publication=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", type=Path)
    group.add_argument("--memory", nargs=2, metavar=("FIXTURE", "METHOD"))
    args = parser.parse_args()
    if args.memory:
        name, method = args.memory
        if method not in METHODS:
            parser.error("unknown method")
        fixture = next(f for f in fx.benchmark_fixtures() if f.name == name)
        print(json.dumps(memory_sample(fixture, method), sort_keys=True))
    else:
        run(args.run)
