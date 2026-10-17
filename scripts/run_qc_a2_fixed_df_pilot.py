"""Closed-by-default A2 pilot launcher; does not issue adoption/authorization."""

import argparse
import json
from pathlib import Path
import signal
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trottercert.a2_pilot.runner import run_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repository", "contract", "authorization", "contract-sha256", "authorization-sha256", "source-commit", "input-manifest-id"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--execute", action="store_true", help="Still requires adopted contract and separately pinned user authorization")
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise InterruptedError("Owned pilot launcher interrupted by signal " + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    pins = {name: getattr(args, name) for name in ("contract_sha256", "authorization_sha256", "source_commit", "input_manifest_id")}
    result = run_files(args.repository, args.contract, args.authorization, pins, execute=args.execute)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "COMPLETED_FIXED_DF_CERTIFICATE_COMPARISON" else 2


if __name__ == "__main__":
    raise SystemExit(main())
