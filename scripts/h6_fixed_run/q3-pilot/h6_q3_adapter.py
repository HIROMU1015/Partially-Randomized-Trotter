"""H6 request/receipt adapter for the existing RPE optimizer executor."""

import argparse
import hashlib
import math
import os
from pathlib import Path
import sys

import run_q3 as q3


METHODS = {"FULL_PF": "Full PF", "TRUNCATED_PF": "B0I", "PR_DF": "PR-DF"}


def request_inputs(request):
    route = request["route"]
    name = METHODS[request["method"]]
    cid = request["configuration_id"]
    scoped = route["kind"] == "h6_df19_scoped_q3"
    if route["kind"] not in ("h6_df19_fixed_q3", "h6_df19_scoped_q3") or route["view_commit"] != q3.Q3_SOURCE:
        raise ValueError("Wrong route for the H6 fixed Q3 adapter")
    entry = route["configurations"][cid]
    config_path = Path(entry["configuration"])
    if hashlib.sha256(config_path.read_bytes()).hexdigest() != entry["configuration_sha256"]:
        raise ValueError("Canonical configuration bytes differ from the queued entry")
    config = q3.read(config_path)
    signals_file = Path(entry["signal_packet"])
    signal_sha = hashlib.sha256(signals_file.read_bytes()).hexdigest()
    signals = q3.read(signals_file)
    target = signals["target"]
    M = signals["M"]
    if scoped:
        if (type(M) is not int or M < 0 or not 0 < signals["h"] < math.inf
                or route["scope_owner"] != signals.get("scope_owner")):
            raise ValueError("Invalid scoped H6 horizon or owner")
    elif (M, signals["h"]) != (12, .1):
        raise ValueError("Fixed H6 route requires M=12 and h=0.1")
    if (config["id"] != cid or config["spec"]["binding"]["system"] != "H6"
            or config["spec"]["binding"]["rank"] != 19 or target != q3.TARGET
            or signals["status"] != "SIGNAL_PACKETS_READY"
            or config["spec"]["M"] != M or signal_sha != entry["signal_packet_sha256"]):
        raise ValueError("H6 canonical configuration, H/state binding or saved raw packet differs")
    method_rows = signals["methods"][name]
    if [row["m"] for row in method_rows] != list(range(M + 1)):
        raise ValueError("H6 input must cover all requested rounds")
    for config_round, row in zip(config["rounds"], method_rows, strict=True):
        candidate = row.get("candidate")
        mean = row["signal"]["raw"]
        if any(not math.isfinite(v) or abs(v) > 1 for v in mean.values()) or set(mean) != {"real", "imag"}:
            raise ValueError("Invalid binary64 raw measurement mean")
        if (row["m"] != config_round["m"] or candidate is not None
                and (candidate["q"] != config_round["q_PF"] or candidate["R_outer"] != config_round["R_outer"]
                     or candidate["formula"] != config["spec"]["formula"]
                     or candidate["prefix"] != config["spec"]["L_D"])):
            raise ValueError("H6 signal schedule disagrees with the canonical candidate")
    nx, ny = request["N_X"], request["N_Y"]
    if len(nx) != M + 1 or nx != ny or any(type(n) is not int or n <= 0 for n in nx):
        raise ValueError("The fixed H6 route requires a positive symmetric allocation")
    if request.get("origin", {}).get("M", M) != M:
        raise ValueError("H6 shot allocation horizon differs from the signal")
    if request["expected"]["configuration_id"] != cid:
        raise ValueError("Expected Q3 request identity differs")
    registry = q3.check_registered_prices(Path(route["registered_prices"]), signal_sha, signals["compiler"],
                                         name, expected_sha=route.get("registered_prices_sha256"), M=M)
    price = registry["methods"][name]
    if scoped:
        for row in price["axes"]:
            kind = row["kind"]
            if kind == "CONDITIONAL_STEP_TRANSFER_PRICE":
                raise ValueError("H6 Q3 requires a model calibrated at the physical step")
            if kind == "CONDITIONAL_COMPILED_PRICE_MODEL":
                step = config["rounds"][row["m"]]["actual_delta_pf"]
                evidence = row["price_evidence"]
                if evidence.get("calibration_step") != step or evidence.get("target_step") != step:
                    raise ValueError("H6 Q3 price calibration step differs from the physical step")
    if (price["configuration_id"] != cid or price["configuration_sha256"] != entry["configuration_sha256"]
            or Path(price["configuration"]).resolve() != config_path.resolve()):
        raise ValueError("Registered price configuration differs from the H6 request")
    cost = sum((nx if row["axis"] == "X" else ny)[row["m"]] * q3.exact_price(row["unit_price_RZ"])
               for row in price["axes"])
    if cost != q3.exact_price(request["cost"]):
        raise ValueError("Registered prices do not reproduce the H6 request cost")
    source = dict(view_commit=q3.Q3_SOURCE, source_job=signals["run_id"],
        signal_packet_sha256=signal_sha, configuration_id=cid, evaluator="rpe_total_risk", route=route["kind"])
    identity = dict(configuration_id=cid, M=M, contract_matches=True,
        allocation=[dict(m=m, N_X=n, N_Y=n) for m, n in enumerate(nx)])
    return name, signals, method_rows, nx, source, identity


def main():
    if hasattr(sys, "set_int_max_str_digits"):
        sys.set_int_max_str_digits(0)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("execute", "collect", "dry-check"), required=True)
    args = parser.parse_args()
    out = args.output
    if args.mode == "collect":
        # A collection invocation can read already completed evidence only.
        return 0 if (out / "OUTCOME.json").is_file() else 3
    request = q3.read(args.request)
    risk_source = Path(request["route"].get("risk_source", q3.ROOT / "risk-source/src"))
    if not (risk_source / "trottertracks/rpe_total_risk.py").is_file():
        raise ValueError("H6 Q3 risk source is unavailable")
    sys.path.insert(0, str(risk_source))
    name, signal, rows, shots, source, identity = request_inputs(request)
    if args.mode == "dry-check":
        from trottertracks.rpe_total_dp_parallel import Accelerator
        Accelerator(workers=1)
        q3.write(out / "DRY_CHECK.json", dict(status="DRY_CHECK_PASS", source=source, identity=identity))
        return 0
    affinity = len(os.sched_getaffinity(0))
    allocated = int(os.environ.get("H6_ALLOCATED_CPUS", request["route"].get("allocated_cpus", affinity)))
    workers = max(1, min(affinity, allocated) - 1)
    q3.write(out / "H6_STARTED.json", dict(source=source, identity=identity, request_id=request["request_id"]))
    result = q3.compute((name, rows, shots, source, workers, signal["h"], signal["target"]))
    q3.write(out / "H6_RESULT.json", result)
    q3.write(out / "OUTCOME.json", dict(request_id=request["request_id"], route=request["route"]["kind"],
        status=result["status"], MSE_lower=result["MSE_lower"], MSE_upper=result["MSE_upper"],
        scientific_evaluation_calls=1, identity=identity))
    return 0 if result["status"] != "UNKNOWN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
