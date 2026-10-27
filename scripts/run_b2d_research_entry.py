"""Non-scientific B2-D entry; old scientific runners are never called."""
import argparse
import json
from trottertracks.b2d_evaluator.research_entry import STAGES, plan, run_mock, artificial_request, write_record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--mode", choices=("plan", "mock"), default="plan")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    record = (plan(args.stage) if args.mode == "plan" else run_mock(artificial_request(args.stage)))
    if args.output:
        write_record(args.output, record)
    else:
        print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
