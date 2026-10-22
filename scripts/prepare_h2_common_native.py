"""Validate only the H2 technical draft; --execute always stops before file IO."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', required=True)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args(argv)
    # No user-supplied path, SOURCE or scientific module is opened on this route.
    if args.execute:
        p.exit(2, 'STOP: SCIENCE_DISPATCH_UNBOUND; no science grant is issued\n')
    path = Path(__file__).resolve().parents[1] / 'src/trotterlib/h2_common_native_policy.py'
    spec = importlib.util.spec_from_file_location('h2_common_technical_only', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    try:
        result = module.validate_draft(json.loads(Path(args.plan).read_text()))
    except (ValueError, TypeError, KeyError) as exc:
        p.exit(2, 'STOP: ' + str(exc) + '\n')
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
