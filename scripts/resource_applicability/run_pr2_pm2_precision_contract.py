#!/usr/bin/env python3
"""Print PM-2 preparation only. No analysis command or filesystem output option."""
import argparse
import json
from pathlib import Path

from trottertracks.resource_applicability.pm2_precision_contract import build_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_bundle(args.project_root), ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
