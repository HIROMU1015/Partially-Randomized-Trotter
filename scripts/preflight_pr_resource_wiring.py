"""Read-only wiring preflight for an explicitly selected A/B/C/D SOURCE tree."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--series', choices=('A','B','C','D'), required=True)
    parser.add_argument('--source', required=True, help='Full commit for the selected source tree')
    parser.add_argument('--request', type=Path, help='Optional D price request to validate')
    parser.add_argument('--plan', type=Path, help='Optional D Backend execution plan to validate')
    args = parser.parse_args(); root = args.root.resolve()
    sys.path[:0] = [str(root/'src'), str(root/'scripts')]
    # The preflight implementation can live outside the selected branch. Only
    # the selected root supplies production imports; never fall back to ours.
    path = Path(__file__).resolve().parents[1]/'src/trottertracks/pr_resource_wiring.py'
    spec = importlib.util.spec_from_file_location('trottertracks._resource_wiring_check', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    try:
        result = module.preflight(root, args.series, args.source,
            request=None if args.request is None else json.loads(args.request.read_bytes()),
            plan=None if args.plan is None else json.loads(args.plan.read_bytes()))
    except Exception as error:
        print(json.dumps(dict(status='WIRING_STOP_BEFORE_SCIENCE', type=type(error).__name__,
                              message=str(error), science_calls=0, mandatory_stop=True)))
        return 2
    print(json.dumps(result, ensure_ascii=False)); return 0

if __name__ == '__main__': sys.exit(main())
