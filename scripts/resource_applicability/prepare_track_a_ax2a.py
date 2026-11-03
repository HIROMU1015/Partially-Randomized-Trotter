#!/usr/bin/env python3
"""Metadata-only AX-2A preparation writer. No science execution option."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from trottertracks.resource_applicability.ax2a_preparation import build_preparation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Exclusive create: never replace a preparation or existing result file.
    payload = build_preparation(ROOT, args.review)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(payload["status"])


if __name__ == "__main__":
    main()
