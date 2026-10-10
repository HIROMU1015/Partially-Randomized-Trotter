#!/usr/bin/env python3
"""Saved H4-P verification only. No execute, worker or approval generation."""
import argparse
import builtins
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
original_import = builtins.__import__
def saved_only_import(name, *args, **kwargs):
    if name.split('.')[0] in {'numpy', 'scipy', 'mpmath', 'qiskit', 'openfermion'}:
        raise RuntimeError('NUMERICAL_IMPORT_FORBIDDEN_IN_SAVED_REAUDIT:'+name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = saved_only_import
sys.path.insert(0, str(ROOT/'src'))
from trottertracks.resource_applicability.ax2b_h4_saved_receipt_gate_v2 import audit_saved_run
from trottertracks.resource_applicability.ax2b_limits import exclusive_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-source-commit', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    allowed = ROOT/'artifacts/resource_applicability/track_a_ax2b_h4_native_receipt_reaudit/2026-10-10'
    if args.output.resolve().parent != allowed.resolve() or args.output.exists():
        parser.error('Exclusive output in the dedicated v2 re-audit directory required.')
    report = audit_saved_run(ROOT, args.audit_source_commit)
    exclusive_json(args.output, report)
    print(report['status']+' / H4_LIMITED_NOT_AUTHORIZED / H6_NOT_AUTHORIZED')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
