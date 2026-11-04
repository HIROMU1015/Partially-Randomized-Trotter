"""Describe/check fixed bytes. Scientific execution remains blocked by a missing kernel bound."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
INCLUSION_COMMIT = '34840ce9bf81d985b32764d9bc9d77645ed52e59'
INCLUSION = 'artifacts/rpe_m17_rigorous_mean_enclosure/2026-11-03/run01/'
INPUTS = [
    (INCLUSION_COMMIT, INCLUSION+'RESULT.json', 'b58caa9dd8a0267eb20c049636e6d8c0b56d67125912e6b5a6a872334a1a2938'),
    (INCLUSION_COMMIT, INCLUSION+'source_replay.json', 'ad7fd3617a524207fbc86a77de471ed6222b010046822a11bd9675edcace1f08'),
    (INCLUSION_COMMIT, INCLUSION+'categorical_laws.json', 'a75324a1ca014bedf56fbd4e8eacbe6c9829f4bfbc3f8ccf248b5db6cc897e91'),
    (INCLUSION_COMMIT, INCLUSION+'enclosure_operators.zip', '708095a92f7d3080f595388b80be6b6b9a26b4e4536f416edd8d7434731a5d56'),
    ('e22d7718c47358ff650dbc04e8e72f6ab1c8bf41', 'artifacts/rpe_fixed_request_measurement_b/2026-11-03/run01/physical_state.npz',
     '793300c86417f52569289ce86980c5f9a87292e130199355a0f4f7056225c9c4'),
    ('cd749066904d3f29b28b03c19ab69370372b157f', 'artifacts/rpe_m17_mean_validation/2026-11-03/run01/operators.npz',
     'e1a65238e9d9b61a6d0aa542ca7f509827b0a761fc751615f1f2018a0e7020df'),
]
SOURCES = [
    ('18fffce48e33136b91eb6ac5a6f9a5f77a08d36f', 'docs/research/rpe_m17_rigorous_mean_enclosure_preparation_20261103/'+name)
    for name in ['ball_adapter.py', 'mean_adapter.py']
] + [
    ('d0b1da1b04c266a1f7abf8e43966ca6e289a8695', 'docs/research/rpe_fresh_iid_measurement_a_preparation_20261103/'+name)
    for name in ['fresh_iid_adapter.py', 'run_fresh_iid_a_v2.py']
]


def fixed_bytes(commit, path):
    return subprocess.check_output(['git', '-C', str(ROOT), 'show', commit+':'+path])


def metadata():
    inventory = []
    for commit, path, expected in INPUTS:
        raw = (ROOT/path).read_bytes()
        if raw != fixed_bytes(commit, path) or hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('Fixed input byte mismatch: '+path)
        inventory.append({'commit': commit, 'path': path, 'sha256': expected, 'bytes': len(raw)})
    for commit, path in SOURCES:
        if (ROOT/path).read_bytes() != fixed_bytes(commit, path):
            raise ValueError('Fixed helper changed: '+path)
    return inventory


def describe():
    from virtual_bridge_adapter import installed_kernel_contract
    return {'status': 'TECHNICAL_ADAPTER_PREPARED_KERNEL_UNRESOLVED',
            'scientific_execution_ready': False, 'kernel': installed_kernel_contract(),
            'inputs': metadata(), 'new_real_numerical_calls': 0,
            'U_exec_upper': None, 'RPE_tolerance_certified': False,
            'scientific_status': 'UNDETERMINED', 'original_numeric_diagnostic_status': 'INCONSISTENT',
            'old_u_num_bound': None, 'mandatory_STOP': True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--describe', action='store_true')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args(argv)
    description = describe()
    if args.run:
        # This is a numerical-method boundary, not proof/authentication of permission.
        description['status'] = 'BLOCKED_KERNEL_CONTRACT_MANDATORY_STOP'
        print(json.dumps(description, indent=2))
        return 2
    print(json.dumps(description, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
