#!/usr/bin/env python3
"""Write metadata-only preparation proposal; scientific execution unsupported."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from trottertracks.resource_applicability.ax2b_h6_contract import preparation_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Actual rank remains unknown. No input read/generation or science imports.
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(preparation_plan(), stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print("H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION")


if __name__ == "__main__":
    main()
