#!/usr/bin/env python3
"""Emit a PM-0 file-name/content bundle to stdout; never writes project files."""
import argparse
import json
from pathlib import Path

from trottertracks.resource_applicability.pm0_evidence_attribution import build_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_bundle(args.project_root),ensure_ascii=False))


if __name__ == "__main__":
    main()
