#!/usr/bin/env python3
"""Prepare plan without synthesis; run only after separate source-bound approval."""
from __future__ import annotations

import argparse
from fractions import Fraction
import importlib.metadata
import json
import multiprocessing as multiprocessing
import os
from pathlib import Path
import platform
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from trottertracks.algorithm_codesign.synthesis_placement.economics import (
    configure, economics, key, package_tree_sha, pai, scaled, synthesize,
)
from trottertracks.algorithm_codesign.synthesis_placement.launch import (
    consume_marker, sha, verify_launch,
)

PREPARATION = ROOT / "artifacts/track_b_sp05_economics_preparation/2026-10-06"
CONTRACT = PREPARATION / "contract_v1.json"


def requests(contract):
    unique = {}
    for target in contract["targets"]:
        for a in (target["angle"], scaled(target["angle"], Fraction(1, 2)),
                  scaled(target["angle"], Fraction(-1, 2))):
            unique[key(a)] = a
    for a in contract["catalogue"]["angles"]:
        unique[key(a)] = a
    if len(unique) > contract["caps"]["synthesis_keys"]:
        raise ValueError("result-prior key cap exceeded")
    return unique


def verify_environment(identity):
    if platform.python_version() != identity["python"]:
        raise RuntimeError("Python version changed")
    versions = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    if versions != identity["packages"]:
        raise RuntimeError("runtime lock mismatch")
    for name, digest in identity["source_py_tree_sha256"].items():
        if package_tree_sha(name) != digest:
            raise RuntimeError(f"runtime source mismatch: {name}")


def verify_preparation(manifest):
    if manifest.get("focused_tests_passed") is not True:
        raise PermissionError("focused semantics / launch tests must pass before execution")
    for relative, digest in manifest["source_sha256"].items():
        if sha(ROOT / relative) != digest:
            raise PermissionError(f"reviewed preparation identity mismatch: {relative}")


def _worker(connection, a, contract):
    caps = contract["caps"]
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (caps["per_key_cpu_seconds"],)*2)
        memory = caps["child_virtual_address_MiB"] * 1024**2
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
        row = synthesize(a, contract)
        if len(row["sequence"]) > caps["sequence_characters"]:
            raise RuntimeError("sequence output cap exceeded")
        usage = resource.getrusage(resource.RUSAGE_SELF)
        row["cpu_seconds"] = usage.ru_utime + usage.ru_stime
        row["peak_RSS_KiB"] = usage.ru_maxrss
        connection.send({"row": row})
    except Exception as e:
        connection.send({"error": f"{type(e).__name__}: {e}"[:1000]})
    finally:
        connection.close()


def _rss_kib(pid):
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except FileNotFoundError:
        return 0
    return 0


def bounded_key(a, contract, run_start, cpu_used):
    caps = contract["caps"]
    receiving, sending = multiprocessing.get_context("fork").Pipe(duplex=False)
    process = multiprocessing.get_context("fork").Process(target=_worker, args=(sending, a, contract))
    key_start = time.monotonic()
    process.start()
    sending.close()
    try:
        while not receiving.poll(0.025):
            now = time.monotonic()
            if now-key_start > caps["per_key_wall_seconds"] or now-run_start > caps["total_wall_seconds"]:
                raise TimeoutError("wall cap hit; no retry")
            if _rss_kib(os.getpid()) + _rss_kib(process.pid) > caps["combined_RSS_MiB"]*1024:
                raise MemoryError("combined parent/child RSS cap hit")
            if not process.is_alive():
                raise RuntimeError(f"synthesis worker exited {process.exitcode}")
        response = receiving.recv()
        if "error" in response:
            raise RuntimeError(response["error"])
        row = response["row"]
        process.join(timeout=1)
        if row["peak_RSS_KiB"] + resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > caps["combined_RSS_MiB"]*1024:
            raise MemoryError("recorded combined peak RSS cap hit")
        parent = resource.getrusage(resource.RUSAGE_SELF)
        if cpu_used + row["cpu_seconds"] + parent.ru_utime + parent.ru_stime > caps["total_cpu_seconds"]:
            raise TimeoutError("CPU cap hit")
        if time.monotonic()-run_start > caps["total_wall_seconds"]:
            raise TimeoutError("total wall cap hit")
        return row
    finally:
        if process.is_alive():
            process.terminate()
        process.join(timeout=1)
        receiving.close()


def scores(contract, cache):
    rows = []
    for target in contract["targets"]:
        for kind, natives in (
            ("ordinary_Rz", [target["angle"]]),
            ("controlled_pair", [scaled(target["angle"], Fraction(1,2)),
                                 scaled(target["angle"], Fraction(-1,2))])):
            interpolations = [pai(a) for a in natives]
            costs = [[cache[key(contract["catalogue"]["angles"][k])]["T_count"]
                      for k in interpolation["notch_indices"]] for interpolation in interpolations]
            det = sum(cache[key(a)]["T_count"] for a in natives)
            row = economics(interpolations, costs, det)
            row.update(target_id=target["id"], primitive=kind, native_angles=natives,
                       interpolations=interpolations, notch_T_counts=costs, deterministic_T_count=det)
            rows.append(row)
    return rows


def run():
    # Approval checked before importing/calling the actual synthesizer or consuming marker.
    contract, auth, head = verify_launch(ROOT, CONTRACT)
    configure()
    verify_preparation(json.loads((PREPARATION / "preparation_manifest_v1.json").read_text()))
    identity = json.loads((PREPARATION / "tool_identity_v1.json").read_text())
    verify_environment(identity)
    keys = requests(contract)
    directory = ROOT / contract["result_directory"]
    receipt = {"source_commit": auth["source_commit"], "authorization_commit": head,
               "contract_sha256": sha(CONTRACT), "mandatory_STOP": True, "retry": False}
    consume_marker(directory, receipt)
    start = time.monotonic()
    result = {**receipt, "status": "INCONCLUSIVE", "synthesis_rows": [], "economics_rows": [],
              "wrapper_pilot_authorized": False}
    cache, cpu = {}, 0.0
    try:
        for k, a in keys.items():
            row = bounded_key(a, contract, start, cpu)
            cache[k] = row
            result["synthesis_rows"].append(row)
            cpu += row["cpu_seconds"]
            if not row["error_pass"]:
                raise ArithmeticError("synthesis error guard failed; no retry")
        result["economics_rows"] = scores(contract, cache)
        if time.monotonic()-start > contract["caps"]["total_wall_seconds"]:
            raise TimeoutError("total wall cap hit during final accounting")
        labels = [r["classification"] for r in result["economics_rows"]]
        # A strict existence witness survives other uncertain J rows only after
        # every registered synthesis/error check has completed successfully.
        result["status"] = ("PRIMITIVE_TRADEOFF_EXISTS" if "STRICT_TRADEOFF" in labels else
                            "INCONCLUSIVE" if "NUMERIC_INCONCLUSIVE" in labels else
                            "NO_PRIMITIVE_TRADEOFF_IN_REGISTERED_SET")
    except Exception as e:
        result["failure"] = f"{type(e).__name__}: {e}"[:1000]
    result["wall_seconds"] = time.monotonic()-start
    result["worker_cpu_seconds"] = cpu
    parent_usage = resource.getrusage(resource.RUSAGE_SELF)
    result["parent_cpu_seconds"] = parent_usage.ru_utime + parent_usage.ru_stime
    if cpu + result["parent_cpu_seconds"] > contract["caps"]["total_cpu_seconds"]:
        result.update(status="INCONCLUSIVE",failure="total CPU cap hit during final accounting")
    payload = json.dumps(result, indent=2) + "\n"
    if len(payload.encode()) > contract["caps"]["output_bytes"]:
        result = {**receipt, "status": "INCONCLUSIVE", "failure": "output cap hit",
                  "partial_rows_not_saved": True, "wrapper_pilot_authorized": False}
        payload = json.dumps(result, indent=2) + "\n"
    with (directory / "result.json").open("x") as f:
        f.write(payload)
    print(json.dumps({"status": result["status"], "mandatory_STOP": True,
                      "result": str(directory / "result.json")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "run"))
    args = parser.parse_args()
    if args.mode == "plan":
        contract = json.loads(CONTRACT.read_text())
        print(json.dumps({"contract_sha256": sha(CONTRACT), "synthesis_keys": len(requests(contract)),
                          "targets": len(contract["targets"]), "catalogues": 1,
                          "science_execution_authorized": False, "synthesis_calls": 0}))
    else:
        try:
            run()
        except Exception as e:
            parser.exit(2, f"launch rejected: {type(e).__name__}: {e}\n")


if __name__ == "__main__":
    main()
