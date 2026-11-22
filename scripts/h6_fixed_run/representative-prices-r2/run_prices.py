"""Acquire first-price evidence for the fixed H6 B0I and PR-DF representatives."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

from preparation_codec import decode_preparations, file_hash


ROOT = Path(__file__).resolve().parent
COHORT = "H6_DF19_INDEPENDENT_PBS_EARLY_PR_20261122"
CONTEXT = None


def acquire(job):
    kind, m, candidate, index, output = job
    full, q4, helpers, fingerprint_context, preps, tail_scalar, pieces = CONTEXT
    prep = preps[str(candidate["prefix"])]
    leaf = output / f"{kind}-m{m}-{index:02}"
    leaf.mkdir(exist_ok=False)
    if kind == "PR":
        requests, seeds, laws = helpers["sample_requests"](candidate, prep, COHORT, index)
        events = [[e.to_dict() for e in request.rte_occurrence.events] for request in requests]
        if sum(map(len, events)) != candidate["q"] * candidate["R_outer"]:
            raise ValueError("PR sampled event count differs from fixed physical schedule")
        full.write(leaf / "PATH.json", dict(candidate=candidate, seeds=seeds, laws=laws, events=events))
        scalar = prep.constant_coefficient + prep.extracted_identity_coefficient
    else:
        requests = []
        scalar = prep.constant_coefficient + tail_scalar
    print(f"BUILD_START {kind} m={m} path={index}", flush=True)
    start = time.monotonic()
    try:
        with fingerprint_context() as session:
            if pieces is None:
                fingerprints = session
                evolution = helpers["compose"](candidate, prep, requests, scalar, session.fingerprint)
            else:
                from h6_price_acceleration import PriceFingerprints

                fingerprints = PriceFingerprints(session)
                evolution = pieces[candidate["prefix"]].assemble(
                    candidate, helpers["core"], requests, scalar, fingerprints.prepare,
                )
            full.write(leaf / "NATIVE.json", dict(candidate=candidate, scalar=scalar,
                num_instructions=len(evolution.circuit.data), fingerprint=evolution.circuit_fingerprint,
                preparation_hash=prep.preparation_hash))
            axes = q4.parallel_axes(leaf, evolution, fingerprints)
        if len({a["fingerprint"] for a in axes.values()}) != 2:
            raise ValueError("X/Y compiled receipt has identical input fingerprints")
        row = dict(method=kind, m=m, path_index=index, candidate=candidate, axes=axes,
                   preparation_hash=prep.preparation_hash, wall_seconds=time.monotonic() - start)
        full.write(leaf / "RESULT.json", row)
        print(f"PRICE_DONE {kind} m={m} path={index}", flush=True)
        return row
    except BaseException as exc:
        full.write(leaf / "STOP.json", dict(error=str(exc)))
        raise


def main():
    global CONTEXT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signals", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--accelerated", action="store_true")
    parser.add_argument("--method", choices=("B0I", "PR-DF"))
    parser.add_argument("--first-round", type=int, default=0)
    parser.add_argument("--last-round", type=int, default=3)
    args = parser.parse_args()
    if not 0 <= args.first_round <= args.last_round <= 12:
        parser.error("Require a nonempty round interval within m=0..12")
    sys.path.insert(0, str(ROOT.parent / "payload-bound"))
    import run_full_m7 as full

    q4, helpers, fingerprint_context, fixed, _, runtime = full.check()
    archive = full.read(ROOT / "PREPARATIONS.json")
    if (archive["scientific_binding"] != fixed["scientific_binding"] or archive["compiler"] != full.COMPILER_HASH
            or archive["provider"] != fixed["provider_source"] or archive["carrier"] != q4.p.CARRIER):
        raise ValueError("Archived preparation source/physics/price identity differs")
    preps = decode_preparations(archive["preparations"])
    if (set(preps) != {"7", "15"} or archive["earlier_pr7_preparation_hash"] != fixed["preparation_hash"]["7"]
            or archive["independent_pr_pricing_cohort"] is not True):
        raise ValueError("Expected an independent PR7 price cohort with the prior binding recorded")
    for key, prep in preps.items():
        if (prep.preparation_hash != archive["preparation_hashes"][key]
                or prep.hamiltonian_hash != fixed["scientific_binding"]["hamiltonian"]
                or prep.deterministic_fragment_indices != tuple(range(int(key)))):
            raise ValueError("Exact H6 prepared block identity differs")
    signal_packet = full.read(args.signals)
    if (signal_packet["status"] != "SIGNAL_PACKETS_READY" or signal_packet["compiler"] != full.COMPILER_HASH
            or signal_packet["h"] != .1 or signal_packet["M"] != 12):
        raise ValueError("Verified H6 signal source required")
    tail_scalar = signal_packet["B0I_tail_scalar"]
    if not math.isclose(tail_scalar, preps["15"].extracted_identity_coefficient, rel_tol=3e-14):
        raise ValueError("B0I signal/circuit tail identity differs")
    menu = []
    for label, method, reps, formula, prefix, width, events, order in (
        ("B0I", "B0I", 1, "2nd", 15, .2, None, None),
        ("PR-DF", "PR", 8, "4th(m5_best)", 7, 1.6, 16, 2),
    ):
        if args.method is not None and args.method != label:
            continue
        for m in range(args.first_round, args.last_round + 1):
            row = signal_packet["methods"][label][m]
            c = row["candidate"]
            total = math.ldexp(.1, m)
            q = math.ceil(total / width)
            expected = dict(system="H6", method=method, T=total, q=q, delta=total / q,
                            formula=formula, prefix=prefix, R_outer=events, K=order)
            identity = [expected[k] for k in ("system", "method", "formula", "prefix", "T", "q", "R_outer", "K")]
            expected["id"] = hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:20]
            if row["m"] != m or c != expected:
                raise ValueError(f"{label} physical schedule differs at m={m}")
            menu.extend((method, m, c, index, args.output) for index in range(reps))
    pieces = None
    q4.price_axis = full.acquire_axis
    if args.accelerated:
        sys.path.insert(0, str(ROOT.parent))
        from h6_price_acceleration import CircuitPieces, acquire_axis, fingerprint_context as load_fast_fingerprint

        fingerprint_context = load_fast_fingerprint()
        q4.price_axis = acquire_axis
        if args.output is not None:
            pieces = {int(prefix): CircuitPieces(prep) for prefix, prep in preps.items()}
            # Warm the common PR circuits once before the path workers fork.
            for kind, _, candidate, index, _ in menu:
                if kind == "PR" and index == 0:
                    pieces[candidate["prefix"]].prepare_pr_flanks(candidate, helpers["core"])
    if args.output is None:
        print(f"H6 bound B0I/PR-DF native preparations PASS; {len(menu)} paired-axis cases; compile calls 0", flush=True)
        return
    # PBS_NODEFILE lists MPI ranks, which can be 1 even for a wide CPU request.
    allocated = min(runtime["affinity_cpus"], int(os.environ["H6_ALLOCATED_CPUS"]))
    if allocated < 2:
        raise ValueError("Two allocated PBS CPUs required for the paired-axis compiler")
    CONTEXT = full, q4, helpers, fingerprint_context, preps, tail_scalar, pieces
    args.output.mkdir(mode=0o700, exist_ok=False)
    full.write(args.output / "INPUT.json", dict(signal_packet_sha256=file_hash(args.signals),
        preparation_archive_sha256=file_hash(ROOT / "PREPARATIONS.json"),
        run_id=os.environ.get("PBS_JOBID"), compiler=full.COMPILER_HASH, cohort=COHORT,
        paired_paths=len(menu), requested_cores=allocated,
        preparation_hashes=archive["preparation_hashes"], prior_pr_binding=archive["earlier_pr7_preparation_hash"],
        scope="B0I_DIRECT_PR_INDEPENDENT_FINITE_COHORT",
        method=args.method, rounds=list(range(args.first_round, args.last_round + 1)),
        build="EXACT_NATIVE_PIECE_REUSE" if args.accelerated else "ORIGINAL",
        fingerprint_backend="H46_B3_SHARED_WRAPPER_PREFIX" if args.accelerated else "ORIGINAL"))
    results, errors = [], []
    with ProcessPoolExecutor(max_workers=min(len(menu), allocated // 2), mp_context=mp.get_context("fork")) as pool:
        pending = {pool.submit(acquire, job): job for job in menu}
        for task in as_completed(pending):
            try:
                results.append(task.result())
            except Exception as exc:
                method, m, _, path, _ = pending[task]
                errors.append(dict(method=method, m=m, path=path, error=str(exc)))
                print(f"PRICE_ERROR {method} m={m} path={path}: {exc}", flush=True)
    results.sort(key=lambda row: (row["method"], row["m"], row["path_index"]))
    full.write(args.output / "RESULTS.json", dict(rows=results, errors=errors,
        independent_pr_path_count_per_round=8, pr_population_expectation_qualified=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
