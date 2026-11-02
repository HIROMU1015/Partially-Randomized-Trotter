#!/usr/bin/env python3
"""RC-DF-type one-fit proposal. Default is passive byte/environment preflight."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fit-python", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--fit-worker", action="store_true")
    args = parser.parse_args()
    from trottertracks import c_rcdf_comparator_pilot as pilot
    c = pilot.preflight(ROOT, args.contract, fit=args.fit_worker)
    if not (args.execute or args.fit_worker):
        print(json.dumps(dict(status="BYTE_PREFLIGHT_PASS",
                              scientific_calls=0, maximum_calls=pilot.MAX_CALLS)))
        return
    expected = ROOT/c["fresh_output"]
    if args.fit_worker:
        expected = expected/"fit"
    if args.output is None or args.output.resolve() != expected.resolve() or expected.exists():
        parser.error("Fixed fresh output required")
    os.environ["JAX_ENABLE_X64"] = "1"
    for name, leaf in (("XDG_CACHE_HOME", ""), ("MPLCONFIGDIR", "matplotlib"),
                       ("JAX_COMPILATION_CACHE_DIR", "jax"), ("TMPDIR", "tmp"),
                       ("NUMBA_CACHE_DIR", "numba")):
        target = ROOT/".cache/regularization-comparator"/leaf
        target.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(target)
    if args.fit_worker:
        pilot.fit_worker(ROOT, args.output, c)
    else:
        if args.fit_python is None:
            parser.error("Separate fit-python required")
        pilot.run(ROOT, args.contract, args.output, args.fit_python)


if __name__ == "__main__":
    main()
