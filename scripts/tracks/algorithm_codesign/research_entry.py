"""Select one Track B reference or artificial descriptor; no scientific runner."""
import argparse
import json
from trottertracks.algorithm_codesign.research_entry import describe, artificial_context, run_mock, write_record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, choices=("G6", "M1-S-gamma", "M1-S-pi3"))
    parser.add_argument("--stage", choices=("reference", "descriptor", "mock"), default="reference")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    record = (run_mock(artificial_context(args.study)) if args.stage == "mock"
              else describe(args.study, stage=args.stage))
    if args.output:
        write_record(args.output, record)
    else:
        print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
