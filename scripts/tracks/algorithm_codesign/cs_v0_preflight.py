#!/usr/bin/env python3
"""Static V0 byte/runtime preflight ONLY; no scientific activation option."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'src'))


def inspect(root=ROOT, *, source_sha256=None):
    from trottertracks.algorithm_codesign.parent_allocation_native import bound_json
    from trottertracks.algorithm_codesign.synthesis_placement.wrapper_launch import verify_runtime
    directory = Path(root)/'docs/research/cs_v0_technical_preparation_20261108'
    lock = json.loads((directory/'SOURCE_AND_CHECKS.json').read_text())
    for row in lock['fixed_inputs']:
        bound_json((Path(root)/row['path']).read_bytes(), expected_bytes=row['bytes'],
                   expected_sha256=row['sha256'])
    expected_source = lock['execution_source_sha256'] if source_sha256 is None else source_sha256
    if set(expected_source) != set(lock['execution_source_sha256']):
        raise ValueError('complete explicit execution SOURCE lock required')
    for name, expected in expected_source.items():
        if hashlib.sha256((Path(root)/name).read_bytes()).hexdigest() != expected:
            raise ValueError('execution SOURCE mismatch: '+name)
    profile = json.loads((directory/'NATIVE_PROFILE.json').read_text())
    if hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest() != profile['tool']['runtime_executable_sha256']:
        raise ValueError('runtime executable mismatch')
    verify_runtime(Path(root), dict(tool_identity=dict(
        path='docs/research/cs_v0_technical_preparation_20261108/TOOL_IDENTITY.json',
        sha256=profile['tool']['tool_identity_sha256'])))
    return dict(status='STATIC_PREFLIGHT_PASS', science_calls=0,
                slot_binding=0, coarse_or_residual_evaluation=0,
                synthesis=0, operator_evaluation=0,
                next_action='STOP; separate V0 science permission required')


if __name__ == '__main__':
    print(json.dumps(inspect(), indent=2))
