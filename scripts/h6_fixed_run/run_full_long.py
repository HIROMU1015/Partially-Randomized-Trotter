"""Acquire H6 Full PF prices with exact native circuit piece reuse."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parent
EXPECTED_IDS = {
    0: "dec32b474f3fc33ddfab", 1: "7791db0dc2fab94ce734", 2: "e7200ae9e16ba879a78d",
    3: "76601611270ed9f0677d", 4: "a4f5ffc0f320644dadfa",
    5: "3f813a46e17461bfe484", 6: "201a8e365ab6f4af6ebc", 7: "5cc99f967c6a176c43fb",
    8: "26608f8bf53993c7095d", 9: "408142c6d1cd1b104b72", 10: "20a160dafc0fa7d6bce5",
    11: "2d56c445820c4accd676", 12: "fb0cb7b53bbfc8080224",
}


def identity(candidate):
    values = [candidate[k] for k in ("system", "method", "formula", "prefix", "T", "q", "R_outer", "K")]
    return hashlib.sha256(json.dumps(values).encode()).hexdigest()[:20]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, choices=EXPECTED_IDS, required=True)
    parser.add_argument("--axis", choices=("X", "Y", "both"), default="both")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / "payload-bound"))
    import run_full_m7 as fixed

    q4, helpers, _, inputs, prep, runtime = fixed.check()
    from h6_price_acceleration import CircuitPieces, PriceFingerprints, acquire_axis, fingerprint_context

    use_fast_fingerprint = fingerprint_context()
    plan = fixed.read(ROOT / "representatives/INPUTS.json")
    row = plan["rounds"][str(args.m)]
    q = 1 << max(0, args.m - 4)
    candidate = dict(fixed.CANDIDATE, T=row["T"], q=q, delta=row["T"] / q)
    candidate["id"] = identity(candidate)
    if (candidate["id"] != EXPECTED_IDS[args.m] or candidate["T"] != math.ldexp(plan["h"], args.m)
            or q != math.ceil(candidate["T"] / fixed.CANDIDATE["delta"])):
        raise ValueError("Saved Full PF candidate identity differs")
    if args.output is None:
        print(json.dumps({"candidate": candidate, "axis": args.axis, "science_calls": 0}))
        return

    started = time.monotonic()
    output = args.output
    output.mkdir(mode=0o700, exist_ok=False)
    axes = {"X": ("cosine",), "Y": ("sine",), "both": q4.p.AXES}[args.axis]
    fixed.write(output / "STARTED.json", dict(candidate=candidate, round=args.m, axes=axes, runtime=runtime,
        scientific_binding=inputs["scientific_binding"], pbs_job_id=os.environ.get("PBS_JOBID"),
        preparation_hash=prep.preparation_hash, build="EXACT_NATIVE_PIECE_REUSE",
        fingerprint_backend="H46_B3_SHARED_WRAPPER_PREFIX"))
    try:
        print(f"BUILD_START Full PF m={args.m} qPF={candidate['q']} {args.axis}", flush=True)
        q4.price_axis = acquire_axis
        with use_fast_fingerprint() as session:
            fingerprints = PriceFingerprints(session, axes)
            pieces = CircuitPieces(prep)
            evolution = pieces.assemble(candidate, helpers["core"], [], prep.scalar, fingerprints.prepare)
            fixed.write(output / "native.json", dict(instructions=len(evolution.circuit.data),
                instruction_bound=evolution.instruction_upper_bound, fingerprint=evolution.circuit_fingerprint))
            if len(axes) == 2:
                result = q4.parallel_axes(output, evolution, fingerprints)
            else:
                result = {axes[0]: acquire_axis(axes[0], output, evolution, fingerprints)}
        fixed.write(output / "result.json", dict(status="PASS_PRICE_ACQUIRED", candidate=candidate,
            round=args.m, axes=result, wall_seconds=time.monotonic() - started, signals_generated=0, q3_calls=0))
        print(f"PASS_PRICE_ACQUIRED Full PF m={args.m} {args.axis}", flush=True)
    except BaseException as exc:
        fixed.write(output / "STOP.json", {"error": str(exc)})
        raise


if __name__ == "__main__":
    main()
