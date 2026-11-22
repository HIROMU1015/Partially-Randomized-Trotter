"""Run one exact-affine, conditional 13-round H6 Q3 diagnostic per method."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from fractions import Fraction
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent
Q3_SOURCE = "cbc471c913072de45c25a0ed22d51dff8a9b7034"
METHODS = {"Full PF": "deterministic", "B0I": "deterministic", "PR-DF": "ideal_finite_RTE_law"}
TARGET = -3.2360662799087647
EPSILON = "0.00159360010199040"


def read(path):
    return json.loads(path.read_bytes())


def write(path, record, *, replace=False):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        pending = Path(stream.name)
        try:
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.close()
            if replace:
                pending.replace(path)
            else:
                os.link(pending, path)
        finally:
            pending.unlink(missing_ok=True)


def check_registered_prices(path, signal_sha, compiler, method=None, *, expected_sha=None, M=12):
    payload = path.read_bytes()
    if expected_sha is not None and hashlib.sha256(payload).hexdigest() != expected_sha:
        raise ValueError("Registered prices changed after the Q3 request was prepared")
    registry = json.loads(payload)
    labels = set(registry["methods"])
    expected_labels = labels == set(METHODS) if method is None else method in labels and labels <= set(METHODS)
    if (registry["signal_packet_sha256"] != signal_sha or registry["compiler"] != compiler
            or not expected_labels):
        raise ValueError("Registered prices are for a different signal packet or compiler")
    expected = [(m, axis) for m in range(M + 1) for axis in ("X", "Y")]
    names = METHODS if method is None else (method,)
    for name in names:
        entry = registry["methods"][name]
        axes = entry["axes"]
        if [(row["m"], row["axis"]) for row in axes] != expected:
            raise ValueError(f"{name} requires all {2 * (M + 1)} registered physical-axis prices in X/Y round order")
        for row in axes:
            price = exact_price(row["unit_price_RZ"])
            if price <= 0 or not row.get("price_evidence") or row.get("status") != "ADMITTED":
                raise ValueError(f"{name} has prices without accepted evidence")
        if not entry.get("configuration_id"):
            raise ValueError(f"{name} price evidence lacks a canonical configuration")
    return registry


def exact_price(value):
    if isinstance(value, dict):
        return Fraction(value["numerator"], value["denominator"])
    return Fraction(value)


@contextmanager
def reuse_round_tables(risk):
    """Share exact count tables for the lifetime of one Q3 allocation."""
    build_table = risk.round_distribution
    tables = {}
    counts = {"built": 0, "reused": 0}

    def round_distribution(row, *, h, target_energy, radius, probability_rule, maximum_outcomes):
        raw = row["raw"]
        key = (row["m"], float(raw["real"]).hex(), float(raw["imag"]).hex(), row["N_X"], row["N_Y"],
               float(h).hex(), float(target_energy).hex(), float(radius).hex(), probability_rule, maximum_outcomes)
        if key not in tables:
            tables[key] = build_table(
                row, h=h, target_energy=target_energy, radius=radius, probability_rule=probability_rule,
                maximum_outcomes=maximum_outcomes,
            )
            counts["built"] += 1
        else:
            counts["reused"] += 1
        return tables[key]

    with patch.object(risk, "round_distribution", round_distribution):
        yield counts


def compute(job):
    name, rows, shots, source, workers, h, target = job
    from trottertracks import rpe_deterministic_risk as D, rpe_total_risk as R
    from trottertracks.rpe_total_dp_parallel import installed

    start = time.monotonic()
    law = METHODS[name]
    h_exact = Fraction(h)
    threshold = Fraction(EPSILON) ** 2
    loss_cap, _ = R.loss_range(h_exact, target)
    entry = dict(
        prepared=dict(M=len(rows) - 1, h=D.a.exact_value(h_exact), decoder=D.S.DECODER,
                      input_complete=True, physical_qualification_complete=False,
                      rows=[dict(m=r["m"], mean=dict(raw=r["signal"]["raw"]),
                                 shots=dict(N_X=n, N_Y=n), law_binding=dict(definition=dict(model=law)))
                            for r, n in zip(rows, shots, strict=True)]),
        statistical_contract=dict(allow_conditional_unqualified=True,
            simulation_policy=dict(kind=D.S.IID, source=source, target_law=law),
            summary_options=dict(unresolved_loss=dict(kind="BOUNDED_INTERVAL", lower=0,
                                                       upper=D.a.exact_value(loss_cap), source=source))),
        evaluation_target=dict(energy_ha=target),
    )
    cells = [(n + 1) ** 2 for n in shots]
    frontier = 8192
    options = dict(estimator=R.T.ESTIMATOR, probability_rule=D.EXACT_AFFINE,
        phase_radii=[1.] * len(shots), maximum_round_outcomes=max(cells),
        maximum_verification_pairs=sum(a * b for a, b in zip(cells, cells[1:])),
        model_loss_contract=dict(adopted=True, source=source, scope="H6_FIXED_ALLOCATION_DIAGNOSTIC"),
        total_state_refinement=dict(source=source, maximum_frontier_states=frontier,
                                    maximum_transition_pairs=cells[0] + frontier * sum(cells[1:])),
        q3_limit=dict(MSE_upper=D.a.exact_value(threshold), scope="MODEL_MSE_FILTER_ONLY", source=source),
    )
    print(f"Q3_START {name}: {workers} children available", flush=True)
    with reuse_round_tables(D) as tables:
        partial = D._bound(entry, options)
        with installed(workers=workers) as engine:
            total = R._total(entry, partial, options)
            timing = engine.observations
    lower = total.get("MSE_lower")
    upper = total.get("MSE_upper")
    if lower is None or upper is None:
        status = "UNKNOWN"
    elif D.a.exact_scalar(upper) <= threshold:
        status = "MODEL_BOUND_MEETS_ADOPTED_THRESHOLD"
    elif D.a.exact_scalar(lower) > threshold:
        status = "MODEL_BOUND_EXCEEDS_ADOPTED_THRESHOLD"
    else:
        status = "INCONCLUSIVE"
    elapsed = time.monotonic() - start
    print(f"Q3_END {name}: {status} in {elapsed:.1f}s", flush=True)
    return dict(method=name, law=law, status=status, MSE_lower=lower, MSE_upper=upper,
                result=total, partial=partial, engine_observations=timing, round_tables=tables, wall_seconds=elapsed)


def main():
    if hasattr(sys, "set_int_max_str_digits"):
        sys.set_int_max_str_digits(0)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signals", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--registered-prices", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / "risk-source/src"))
    from trottertracks import rpe_deterministic_risk as D, rpe_total_risk as R
    from trottertracks.rpe_total_dp_parallel import Accelerator

    raw = read(args.signals)
    if (raw["status"] != "SIGNAL_PACKETS_READY" or raw["target"] != TARGET
            or raw["M"] != 12 or raw["h"] != .1 or set(raw["methods"]) != set(METHODS)):
        raise ValueError("Unexpected source signal packets")
    for rows in raw["methods"].values():
        if [r["m"] for r in rows] != list(range(13)):
            raise ValueError("Nonconsecutive method signal packet")
        for row in rows:
            signal = row["signal"]["raw"]
            if set(signal) != {"real", "imag"} or any(not math.isfinite(v) or abs(v) > 1 for v in signal.values()):
                raise ValueError("Physical raw means are required")
    if args.output is None:
        Accelerator(workers=1)
        print("Three complete H6 packets and Q3 engine PASS; scientific calls 0", flush=True)
        return
    if args.registered_prices is None:
        parser.error("--output requires --registered-prices for all three methods")
    signal_sha = hashlib.sha256(args.signals.read_bytes()).hexdigest()
    registry = check_registered_prices(args.registered_prices, signal_sha, raw["compiler"])
    shots = [11 + 4 * (12 - m) for m in range(13)]  # Existing s=1 allocation.
    source = dict(signal_job=raw["run_id"], signal_packet=str(args.signals),
                  signal_packet_sha256=signal_sha, registered_prices=str(args.registered_prices),
                  evaluator_commit=Q3_SOURCE, evaluator=R.__name__,
                  exact_affine_probability_rule=D.EXACT_AFFINE)
    allocated = min(len(os.sched_getaffinity(0)), int(os.environ["H6_ALLOCATED_CPUS"]))
    workers = max(1, allocated // len(METHODS) - 1)
    args.output.mkdir(mode=0o700, exist_ok=False)
    write(args.output / "INPUT.json", dict(source=source, N_X=shots, N_Y=shots, epsilon=EPSILON,
            target=TARGET, M=12, h=.1, pbs_job=os.environ.get("PBS_JOBID"), workers_per_method=workers,
            configurations={name: r["configuration_id"] for name, r in registry["methods"].items()}))
    with ProcessPoolExecutor(max_workers=min(len(METHODS), allocated), mp_context=mp.get_context("fork")) as pool:
        pending = {name: pool.submit(compute, (name, raw["methods"][name], shots, source, workers, raw["h"], raw["target"]))
                   for name in METHODS}
        results = {}
        for index, (name, future) in enumerate(pending.items(), 1):
            results[name] = future.result()
            write(args.output / f"method-{index}.json", results[name])
    write(args.output / "RESULTS.json", {name: {"status": r["status"], "MSE_lower": r["MSE_lower"],
          "MSE_upper": r["MSE_upper"], "wall_seconds": r["wall_seconds"]} for name, r in results.items()})


if __name__ == "__main__":
    main()
