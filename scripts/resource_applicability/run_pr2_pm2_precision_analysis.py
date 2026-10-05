#!/usr/bin/env python3
"""Future saved-value PM-2 launch, never a science runner."""
import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--execute-saved-analysis", action="store_true",
                        help="operational launch barrier; requires a separate explicit user instruction")
    args = parser.parse_args()
    from trottertracks.resource_applicability.pm2_precision_analysis import install_boundary, run
    counter = install_boundary()
    result = run(args.project_root, args.source_commit, execute_saved_analysis=args.execute_saved_analysis)
    print(result["status"])
    print("Protected-access audit:", counter)
    return 0 if result["failure_reason"] is None else 1


if __name__ == "__main__":
    sys.exit(main())
