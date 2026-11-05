#!/usr/bin/env python3
"""Check the unapproved integrated proposal without scientific execution."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--contract", type=Path, default=ROOT / "docs/research/research_c_integrated_contract_unapproved_20261105.json")
    a = p.parse_args()
    from trottertracks.c_integrated_plan import preflight
    print(json.dumps(preflight(ROOT, a.contract), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
