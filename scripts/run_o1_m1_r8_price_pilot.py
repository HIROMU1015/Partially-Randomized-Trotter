"""Fixed r8 H4 pricing pilot: 8 IID trajectories per arm, paired full X/Y.

No signal, shot computation, baseline compilation, proxy, retry or optimizer.
Uses the unchanged matched resource bridge and parallel price provider.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import run_o1_m1_c0_pilot as c0
from trottertracks.o1_m1_parallel_price import ParallelMatchedNoPrepCostProvider

BASE = "2263000e203f8b9b075f7296a9e33f34e8ea24db"
SAVED = "c7e9587fb7398ebdab156d83abc869d82e5f42c9"
SAVED_PATH = "artifacts/track_a_o1_m1_cost_feasibility_science_20261106/run01/result.json"
SEEDS = {"P": 2026110612, "DF": 2026110613}
Q, R, K, DELTA, SAMPLES = 64, 8, 2, .05, 8
CORE = (
    "src/trotterlib/o1_m1_matched_split.py",
    "src/trotterlib/o1_m1_resource_bridge.py",
    "src/trotterlib/o1_m1_circuit_signal_validation.py",
    "src/trotterlib/pauli_signal_adapter.py",
    "src/trotterlib/rte.py",
    "src/trotterlib/df_rte_tail.py",
    "src/trotterlib/df_rte_circuit.py",
    "src/trotterlib/df_rte_qiskit.py",
    "src/trotterlib/df_trotter/ops.py",
    "src/trotterlib/rpe_hadamard_interrogation.py",
    "src/trotterlib/rte_compiled_cost.py",
    "src/trottertracks/o1_m1_parallel_price.py",
    "scripts/run_o1_m1_c0_pilot.py",
)


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def append_log(path, value):
    # One append syscall per small progress record; never edits old records.
    data = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        require(os.write(fd, data) == len(data), "Short progress-log write")
    finally:
        os.close(fd)


def record_compile(circuit, compiler, **kwargs):
    from trottertracks.o1_m1_parallel_price import _bridge
    started = time.perf_counter()
    price = _bridge().transpile_and_measure_cost(circuit, compiler, **kwargs)
    append_log(Path(os.environ["O1_M1_R8_PRICE_LOG"]), dict(
        pid=os.getpid(), utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.perf_counter()-started,
        fingerprint=kwargs["actual_circuit_fingerprint"],
        cx=price.cx_count))
    return price


class RecordingProvider(ParallelMatchedNoPrepCostProvider):
    """Record the existing provider's ordered raw results, without resampling."""
    def __init__(self, compiler, *, arm, output):
        super().__init__(compiler, compile_function=record_compile)
        self.arm, self.output = arm, output
        self.raw_pairs = None

    def _map(self, function, tasks, **kwargs):
        require(len(tasks) == 2*SAMPLES, "Fixed paired-axis tasks required")
        draws = []
        for i in range(SAMPLES):
            x, y = tasks[2*i:2*i+2]
            require(x[-1] == "cosine" and y[-1] == "sine" and x[4] == y[4],
                    "X/Y must receive exactly the same canonical events")
            require(x[3] == Q and len(x[4]) == Q*R, "Trajectory length differs")
            draws.append(dict(trajectory=i,
                event_selections=[dict(order=e.taylor_order,
                    rotation=e.rotation_component_id, products=e.product_component_ids)
                    for e in x[4]]))
        c0.dump(self.output / (self.arm+"_draws.json"), dict(
            dictionary=self.arm, seed=SEEDS[self.arm], draws=draws))
        pairs = super()._map(function, tasks, **kwargs)
        bridge = importlib.import_module("trotterlib.o1_m1_resource_bridge")
        rows = []
        for i in range(SAMPLES):
            x, y = pairs[2*i:2*i+2]
            rows.append(dict(trajectory=i,
                cosine=dict(zip(bridge.RPE_COST_METRICS, x["cosine"], strict=True)),
                sine=dict(zip(bridge.RPE_COST_METRICS, y["sine"], strict=True))))
        self.raw_pairs = rows
        c0.dump(self.output / (self.arm+"_raw_prices.json"), dict(
            dictionary=self.arm, seed=SEEDS[self.arm], rows=rows,
            execution=self.last_execution))
        return pairs


def paired_statistics(rows, shots):
    require(len(rows) == SAMPLES, "Exactly eight paired raw prices required")
    nx, ny = shots["cosine"], shots["sine"]
    require(type(nx) is int and type(ny) is int and nx > 0 and ny > 0,
            "Positive saved integer shots required")
    weighted = []
    for row in rows:
        x, y = row["cosine"]["cx_count"], row["sine"]["cx_count"]
        require(x >= 0 and y >= 0 and int(x) == x and int(y) == y, "Invalid raw CX")
        weighted.append(Fraction(nx*int(x)+ny*int(y), nx+ny))
    exact_mean = sum(weighted, Fraction())/SAMPLES
    return dict(shot_weighted_mean=float(exact_mean),
        shot_weighted_mean_fraction=dict(numerator=exact_mean.numerator,
                                        denominator=exact_mean.denominator),
        shot_weighted_sample_SE=statistics.stdev(map(float, weighted))/math.sqrt(SAMPLES),
        uncertainty="paired sample SE only; no confidence interval")


def run(input_path, output):
    require(not output.exists(), "Fresh output required; no retry/resume")
    output.mkdir(parents=True)
    tick = time.perf_counter()
    result = dict(status="RUNNING", mandatory_stop=False, arms={},
        fixed=dict(q=Q, r=R, K=K, delta=DELTA, samples_per_arm=SAMPLES, seeds=SEEDS,
            state_field="state", state_preparation_included=False,
            identity_policy="faithful_identity_in_tail", metric="cx_count",
            scope="single_hadamard_interrogation_without_state_preparation"),
        source=dict(base=BASE, head=git("rev-parse", "HEAD").decode().strip()),
        started_utc=datetime.now(timezone.utc).isoformat(),
        guarantee_status="not_certified", price_confidence_interval=None,
        mathematical_exclusion=None, full_rpe_total_cost=None, full_rpe_improvement=None)
    os.environ["O1_M1_R8_PRICE_LOG"] = str(output/"compile_progress.jsonl")
    try:
        hashes = {}
        for name in CORE:
            content = (ROOT/name).read_bytes()
            require(content == git("show", BASE+":"+name), "Fixed SOURCE differs: "+name)
            hashes[name] = hashlib.sha256(content).hexdigest()
        hashes["scripts/run_o1_m1_r8_price_pilot.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        require(hashlib.sha256(input_path.read_bytes()).hexdigest() == c0.INPUT_SHA,
                "Fixed input differs")
        saved_bytes = git("show", SAVED+":"+SAVED_PATH)
        saved = json.loads(saved_bytes)
        require(saved["original_snapshot_binding"] == dict(
            commit=c0.INPUT_COMMIT, path=c0.INPUT_PATH, sha256=c0.INPUT_SHA),
            "Saved shot input binding differs")
        runtime = {p:importlib.metadata.version(p) for p in
                   ("numpy", "scipy", "openfermion", "qiskit", "qiskit-nature")}
        require(runtime["qiskit"] == "1.3.0", "Actual Qiskit 1.3.0 required")
        result["source"].update(files_sha256=hashes,
            saved_shots=dict(commit=SAVED,path=SAVED_PATH,
                            sha256=hashlib.sha256(saved_bytes).hexdigest()),
            signal_commit="185104f64a4594355a3813d04d9e50bd4abdb6b5",
            baseline_commit="50ee55bc30f30838e1ac0704bc94f720914640de")
        result["input"] = dict(commit=c0.INPUT_COMMIT, path=c0.INPUT_PATH,
                              sha256=c0.INPUT_SHA, local_path=str(input_path))
        result["runtime"] = dict(packages=runtime, python=sys.version,
            host=platform.node(), available_cpus=len(os.sched_getaffinity(0)))
        from trottertracks.o1_m1_parallel_price import _bridge
        bridge = _bridge()
        adapter = importlib.import_module("trotterlib.o1_m1_matched_split")
        from trotterlib.rte import CompilerSettings
        compiler = CompilerSettings(("rz","sx","x","cx"),None,None,1,None,None,7,"1.3.0")
        result["compiler"] = asdict(compiler)
        require(bridge.compiler_settings_hash(compiler)
                == saved["baseline"]["context"]["compiler_hash"], "Compiler cohort differs")
        require(saved["baseline"]["total_cost"] == 9827008128, "Saved baseline differs")
        require(saved["baseline"]["context"]["cost_scope"]
                == result["fixed"]["scope"], "Full no-prep scope differs")
        c0.dump(output/"result.json", result)
        print("Preparing fixed saved H4 resources (no signal or shot recomputation)", flush=True)
        inputs = adapter.load_h4_a_snapshot(input_path)
        state_hash = hashlib.sha256(inputs.state.tobytes()).hexdigest()
        require(state_hash == "d4432409eb6b1d4f9f9f86a3b57664d9e3c84a4fe3fe9de400d64b7ba60270af",
                "Original saved state differs")
        split = adapter.prepare_matched_a(inputs)
        preparations = {name:bridge.prepare_matched_resources(split,dictionary=name) for name in SEEDS}
        laws = {name:bridge.configure_law(p,delta_time=DELTA,r=R,K=K) for name,p in preparations.items()}
        conditions = {}
        for name, p in preparations.items():
            config, distribution = laws[name]
            bridge._validate_law(p, config, distribution)
            old = next(row for row in saved["rows"] if row["dictionary"] == name and row["r"] == R)
            account = old["account"]
            require(account["feasible"] and account["q"] == Q and account["K"] == K
                    and account["delta_time"] == DELTA, "Saved signal conditions differ")
            require((p.context_hash,p.preparation_hash) == (account["context_hash"],account["arm_hash"]),
                    "Canonical pricing preparation differs from saved physical cohort")
            # Normalization parameters bind the saved signal; its tail hash is not a pricing law.
            require(config.lambda_r == old["restored_config"]["lambda_r"]
                    and config.distribution_normalization == old["restored_config"]["distribution_normalization"],
                    "Canonical law normalization differs from saved signal")
            conditions[name] = old
            result["arms"][name] = dict(status="READY", config=asdict(config),
                distribution=asdict(distribution), context_hash=p.context_hash,
                preparation_hash=p.preparation_hash, law_hash=bridge._law_hash(config),
                component_count=len(p.tail.components), saved_shots=account["shots"],
                saved_required_price=old["necessary_price_condition"]["strict_unit_price_ceiling"],
                required_price_display=old["price_ceiling_display"])
        result["original_state_sha256"] = state_hash
        c0.dump(output/"result.json", result)

        def price_arm(name):
            print("Pricing arm="+name+" seed="+str(SEEDS[name])+" draws=8 paired axes", flush=True)
            provider = RecordingProvider(compiler, arm=name, output=output)
            cost = provider.monte_carlo(preparations[name], *laws[name],
                                       q=Q, sample_count=SAMPLES, seed=SEEDS[name])
            require(cost.classical_sample_count == SAMPLES, "Wrong sample count")
            require(cost.circuit_cost_scope == result["fixed"]["scope"], "Price scope differs")
            stats = paired_statistics(provider.raw_pairs, conditions[name]["account"]["shots"])
            for axis in ("cosine","sine"):
                values = [row[axis]["cx_count"] for row in provider.raw_pairs]
                mean = statistics.fmean(values)
                se = statistics.stdev(values)/math.sqrt(SAMPLES)
                require(math.isclose(mean,getattr(cost,axis+"_expected_cost").cx_count,
                                     rel_tol=1e-12,abs_tol=1e-12), "Raw mean differs from provider")
                require(math.isclose(se,getattr(cost,axis+"_standard_error").cx_count,
                                     rel_tol=1e-10,abs_tol=1e-9), "Raw SE differs from provider")
            ceiling = conditions[name]["necessary_price_condition"]["strict_unit_price_ceiling"]
            fraction = stats["shot_weighted_mean_fraction"]
            stats["sample_mean_below_necessary_ceiling"] = (
                Fraction(fraction["numerator"],fraction["denominator"])
                < Fraction(ceiling["numerator"],ceiling["denominator"]))
            return dict(status="PILOT_COMPLETE", price=asdict(cost), raw_prices=provider.raw_pairs,
                        statistics=stats, execution=provider.last_execution)

        # Independent arms; each provider dynamically uses the CPUs and observed memory.
        with ThreadPoolExecutor(max_workers=len(SEEDS)) as pool:
            futures = {name:pool.submit(price_arm,name) for name in SEEDS}
            for name, future in futures.items():
                result["arms"][name].update(future.result())
                c0.dump(output/"result.json", result)
                print("Completed arm="+name, flush=True)
        require(sum(len(row["raw_prices"]) for row in result["arms"].values()) == 16,
                "Fixed trajectory total differs")
        result.update(status="FIXED_R8_PRICE_PILOT_COMPLETE", mandatory_stop=True,
            mathematical_exclusion="UNDETERMINED; sample estimates are not lower bounds",
            elapsed_seconds=time.perf_counter()-tick,
            finished_utc=datetime.now(timezone.utc).isoformat())
        c0.dump(output/"result.json", result)
        print("MANDATORY STOP: 16 trajectories / 32 X/Y prices complete", flush=True)
        return result
    except BaseException as exc:
        result.update(status="ERROR_STOP", mandatory_stop=True, error_type=type(exc).__name__,
                      error=str(exc), elapsed_seconds=time.perf_counter()-tick)
        c0.dump(output/"result.json", result)
        print("MANDATORY STOP: "+type(exc).__name__+": "+str(exc), flush=True)
        raise


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    run(args.input.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
