"""Run fixed H6 Q3 requests for the selected priced methods."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import run_q3
from h6_q3_adapter import METHODS


def requests(registry_path, signals_path, output, method=None):
    signal_sha = hashlib.sha256(signals_path.read_bytes()).hexdigest()
    price_sha = hashlib.sha256(registry_path.read_bytes()).hexdigest()
    registry = run_q3.check_registered_prices(
        registry_path, signal_sha, run_q3.read(signals_path)["compiler"],
        None if method is None else METHODS[method], expected_sha=price_sha,
    )
    allocation = [11 + 4 * (12 - m) for m in range(13)]
    selected = METHODS if method is None else {method: METHODS[method]}
    for method, label in selected.items():
        entry = registry["methods"][label]
        cid = entry["configuration_id"]
        units = [run_q3.exact_price(row["unit_price_RZ"]) for row in entry["axes"]]
        cost = sum(n * unit for n, unit in zip(
            (n for n in allocation for _ in range(2)), units, strict=True,
        ))
        path = output / method
        route = dict(kind="h6_df19_fixed_q3", view_commit=run_q3.Q3_SOURCE,
                     registered_prices=str(registry_path), registered_prices_sha256=price_sha,
                     configurations={cid: dict(
                         configuration=entry["configuration"], configuration_sha256=entry["configuration_sha256"],
                         signal_packet=str(signals_path), signal_packet_sha256=signal_sha)})
        yield path, dict(request_id=method, method=method, configuration_id=cid, N_X=allocation, N_Y=allocation,
                         cost=dict(numerator=cost.numerator, denominator=cost.denominator),
                         origin=dict(kind="RPE_SCALED_SHOTS_V1", M=12, s=1),
                         route=route, expected=dict(configuration_id=cid, M=12))


def run(output, tasks, *, dry_check):
    for path, request in tasks:
        path.mkdir(mode=0o700, exist_ok=True)
        manifest = path / "REQUEST.json"
        if manifest.exists():
            previous = run_q3.read(manifest)
            # Older Q3 requests recorded the price path without its file hash.
            if "registered_prices_sha256" not in previous["route"]:
                previous["route"]["registered_prices_sha256"] = request["route"]["registered_prices_sha256"]
            if previous != request:
                raise ValueError(f"Existing H6 request differs for {path.name}")
        else:
            run_q3.write(manifest, request)

    cpus = int(os.environ["H6_ALLOCATED_CPUS"])
    adapter = Path(__file__).with_name("h6_q3_adapter.py")
    environment = dict(os.environ, H6_ALLOCATED_CPUS=str(max(1, cpus // len(tasks))))

    def call(path, mode):
        with (path / f"{mode}.log").open("x" if mode == "execute" else "a") as log:
            return subprocess.run(
                [sys.executable, "-B", "-u", str(adapter), "--request", str(path / "REQUEST.json"),
                 "--output", str(path), "--mode", mode],
                env=environment, stdout=log, stderr=subprocess.STDOUT,
            ).returncode

    def evaluate(task):
        path, _ = task
        outcome = path / "OUTCOME.json"
        if not dry_check:
            if outcome.is_file():
                return 2 if run_q3.read(outcome)["status"] == "UNKNOWN" else 0
            if (path / "execute.log").exists() or (path / "H6_STARTED.json").exists():
                print(f"H6_Q3_EXECUTION_HOLD {path.name}: prior attempt has no committed outcome", flush=True)
                return 3
        if not (path / "DRY_CHECK.json").is_file() and call(path, "dry-check"):
            return 4
        if dry_check:
            return 0
        code = call(path, "execute")
        return code or (0 if outcome.is_file() else 3)

    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        status = dict(zip((p.name for p, _ in tasks), pool.map(evaluate, tasks), strict=True))
    if not dry_check:
        summary = output / "PROCESS_RETURN.json"
        previous = run_q3.read(summary) if summary.exists() else {}
        run_q3.write(summary, previous | status, replace=True)
    print(json.dumps(status), flush=True)
    return 0 if all(code == 0 for code in status.values()) else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("registered-prices", "signals", "output"):
        parser.add_argument(f"--{arg}", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--dry-check", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    tasks = list(requests(args.registered_prices.resolve(), args.signals.resolve(), output, args.method))
    output.mkdir(mode=0o700, exist_ok=True)
    with (output / ".dispatch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run(output, tasks, dry_check=args.dry_check)


if __name__ == "__main__":
    raise SystemExit(main())
