#!/usr/bin/env python3
"""Zero-compute design preflight: JSON/hash metadata only, no library imports.

No execute flag, quantum/source evaluator, sampler, worker, or marker creation.
Even a future adopted design cannot be executed by this entry point.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys


DECISIONS = ('comparison_methods', 'allocation_candidates', 'proposal_candidates',
             'regularization_policy', 'accuracy_confidence_budget',
             'primary_endpoint', 'scientific_interpretation_rules')


def inspect_metadata(plan, manifest, root):
    root = Path(root).resolve()
    if plan.get('science_execution_authorized') is not False:
        raise ValueError('preparation metadata must explicitly deny scientific execution')
    if plan.get('authorization_commit') is not None or manifest.get('science_execution_authorized') is not False:
        raise ValueError('preparation cannot contain an execution authorization')
    pending = [key for key in DECISIONS if plan.get('scientific_choices', {}).get(key) is None]
    verified = []
    for item in manifest.get('files', []):
        relative = Path(item['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('manifest paths must stay inside the source root')
        path = (root/relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError('manifest symlink leaves source root')
        raw = path.read_bytes()
        if len(raw) != item['bytes'] or hashlib.sha256(raw).hexdigest() != item['sha256']:
            raise ValueError('source/input identity mismatch: '+item['path'])
        verified.append(item['path'])
    if not verified:
        raise ValueError('source manifest must contain identities')
    return {'kind': 'ZERO_COMPUTE_DESIGN_PREFLIGHT', 'identity_check': 'PASS',
            'verified_paths': verified, 'pending_scientific_decisions': pending,
            'science_execution_allowed': False, 'scientific_calls': 0,
            'marker_or_authorization_created': False,
            'registered_inputs_instantiated': False,
            'not_an_execution_launch_gate': True, 'mandatory_STOP': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--root', default='.')
    args = parser.parse_args()
    try:
        result = inspect_metadata(json.loads(Path(args.plan).read_text()),
                                  json.loads(Path(args.manifest).read_text()), args.root)
    except (ValueError, KeyError, OSError, TypeError) as error:
        print(json.dumps({'kind': 'ZERO_COMPUTE_DESIGN_PREFLIGHT', 'identity_check': 'FAIL',
                          'reason': str(error), 'science_execution_allowed': False,
                          'scientific_calls': 0, 'mandatory_STOP': True}))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
