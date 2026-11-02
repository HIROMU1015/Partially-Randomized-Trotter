#!/usr/bin/env python3
"""Saved first-order diagnostic. Default is passive byte/environment preflight."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract",type=Path,required=True)
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--output",type=Path)
    args = parser.parse_args()
    from trottertracks import c_rcdf_first_order as m
    m.preflight(ROOT,args.contract)
    if not args.execute:
        print(json.dumps(dict(status="BYTE_PREFLIGHT_PASS",scientific_calls=0,
                              maximum_calls=m.MAX_CALLS)))
        return
    if args.output is None:
        parser.error("Fixed fresh output required")
    os.environ["JAX_ENABLE_X64"] = "1"
    for name, leaf in (("XDG_CACHE_HOME",""),("MPLCONFIGDIR","matplotlib"),
                       ("JAX_COMPILATION_CACHE_DIR","jax"),("TMPDIR","tmp")):
        target = ROOT/".cache/regularization-first-order"/leaf
        target.mkdir(parents=True,exist_ok=True)
        os.environ[name] = str(target)
    m.run(ROOT,args.contract,args.output)


if __name__ == "__main__":
    main()
