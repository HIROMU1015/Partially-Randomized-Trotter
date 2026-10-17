"""Stage S runner: default/candidate invocation rejects before scientific input."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))
from trottertracks.algorithm_codesign.m1s_stage_s_v1 import control, spec


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--contract", default=spec.ART + "/execution_contract_candidate.json")
    p.add_argument("--contract-commit")
    p.add_argument("--authorization")
    p.add_argument("--authorization-commit")
    args = p.parse_args()
    try:
        permit = control.authorize(ROOT, args.contract, args.contract_commit,
                                   args.authorization, args.authorization_commit)
        status = control.run(ROOT, permit)
    except control.ExecutionClosed as exc:
        print(json.dumps({"status": "EXECUTION_CLOSED", "reason": str(exc)[:4000],
                          "science_inputs_decoded": False, "mandatory_STOP": True}))
        return 2
    print(json.dumps({"status": status, "mandatory_STOP": True}))
    return 0 if status == "STAGE_S_COMPLETE_AWAITING_USER_JUDGMENT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
