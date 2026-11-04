"""Default: bytes and static constants only. --run needs separate science permission.

Diagnoses a lower bound on a proposed proof's upper bound. Never certifies a
signal, a guard, a mean, or fixed-PCG64 IID. Does not implement norm propagation.
"""
from fractions import Fraction as Q
from io import BytesIO
import argparse
import hashlib
import json
from math import isqrt
from pathlib import Path
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT/'docs/research/rpe_m17_openblas_kernel_bound_preflight_20261104'))
sys.path.insert(0, str(ROOT/'docs/research/rpe_m17_virtual_bridge_scientific_runner_preparation_20261104'))
import kernel_bounds as kernel
import run_virtual_bridge as prior
from runtime_checks import validate_array
from runtime_policy import write_json

# Exact RNE binary64 values used by the unchanged measurement SOURCE.
COORDINATE_LIMIT = Q(562949953421875, 562949953421312)  # 1.+1e-12
MAGNITUDE_LIMIT = Q(562949953477607, 562949953421312)   # 1.+1e-10
INPUT_COMMIT = '81fa6813194e4e14189092626f66dd10793225e1'
INPUT_DIR = 'artifacts/rpe_m17_virtual_measurement_bridge/2026-11-04/run01/'
INPUTS = [
    (INPUT_COMMIT, INPUT_DIR+'numeric_context.npz', 'f8c1f3185423f479d82ba0bc1eb94bfb8fd9d0cafa487c618964f3bd22bddbae'),
    (INPUT_COMMIT, INPUT_DIR+'RESULT.json', 'd2989174012c41e6908d21ccc65d894820e763749e922835f9c5e191eb3c8735'),
]
HELPERS = [
    ('e2fd35f1dcc73b697ac85661ef91f21082fd2136', 'docs/research/rpe_m17_openblas_kernel_bound_preflight_20261104/kernel_bounds.py'),
] + [('c3a5cc387b8962b63782dc71ab175147f2d87e33',
      'docs/research/rpe_m17_virtual_bridge_scientific_runner_preparation_20261104/'+name)
     for name in ['runtime_checks.py', 'runtime_policy.py']
] + [('728c90a2d4d4335ef5b7237808c15b182e694bb9',
      'docs/research/rpe_m17_virtual_measurement_bridge_preparation_20261104/'+name)
     for name in ['virtual_bridge_adapter.py', 'run_virtual_bridge.py']
] + [('18fffce48e33136b91eb6ac5a6f9a5f77a08d36f',
      'docs/research/rpe_m17_rigorous_mean_enclosure_preparation_20261103/'+name)
     for name in ['ball_adapter.py', 'mean_adapter.py']]
STATUS = dict(U_virtual_IID_upper=None, U_exec_upper=None, RPE_tolerance_certified=False,
              scientific_status='UNDETERMINED', original_numeric_diagnostic_status='INCONSISTENT',
              old_u_num_bound=None, fixed_PCG64_IID_certified=False, mandatory_STOP=True,
              finite_intermediates_proved=False, all_support_completion_proved=False)


def proof_floor(rows=70, q=131072):
    """Rational lower bound on M/n² for the nonnegative-defect norm method.

    No context, powers of huge rational integers, floating norms or H4 calls.
    See PROOF_LIMIT.md for the all-matrix argument and scope of the obstruction.
    """
    if type(q) is not int or q < 1:
        raise ValueError('Positive integer q required')
    bounds = kernel.path_bounds(rows, rows)
    c = isqrt(rows)  # c <= sqrt(rows), not an estimate of a saved matrix norm.
    if (1+c*bounds.mv_rel)**2 > 2:
        raise ValueError('Universal matrix-floor lemma not applicable')
    return (1+bounds.dot_rel)*(1+2*q*c*bounds.mv_rel)


def assess_norm_squared(norm_squared, *, rows=70, q=131072):
    if not (isinstance(norm_squared, Q) or type(norm_squared) in (int, str)):
        raise ValueError('Exact rational norm squared required')
    n2 = Q(norm_squared)
    if n2 <= 0:
        raise ValueError('Positive exact norm squared required')
    factor = proof_floor(rows, q)
    floor = n2*factor
    coordinate, magnitude = floor > COORDINATE_LIMIT, floor > MAGNITUDE_LIMIT
    return dict(method_status='NORM_PROOF_OBSTRUCTION_CONFIRMED' if coordinate or magnitude else 'FLOOR_INCONCLUSIVE',
                raw_norm_squared=str(n2), lower_on_candidate_upper=str(floor), floor_factor=str(factor),
                critical_norm_squared=str(COORDINATE_LIMIT/factor),
                coordinate_bound_obstructed=coordinate, magnitude_bound_obstructed=magnitude,
                **STATUS)


def exact_norm_squared(psi):
    if not isinstance(psi, np.ndarray) or psi.ndim != 1 or not len(psi):
        raise ValueError('Nonempty complex128 vector required')
    validate_array(psi, (len(psi),))
    return sum((Q.from_float(float(z.real))**2+Q.from_float(float(z.imag))**2 for z in psi), Q(0))


def load_state(raw):
    with np.load(BytesIO(raw), allow_pickle=False) as arrays:
        expected = {f'P{i:03d}' for i in range(218)} | {'forward', 'reverse', 'psi'}
        if len(arrays.files) != len(expected) or set(arrays.files) != expected:
            raise ValueError('Fixed context inventory mismatch')
        psi = arrays['psi']  # No other array is decompressed or evaluated.
    validate_array(psi, (70,))
    return psi


def source_metadata(frozen):
    own = HERE/'run_norm_proof_diagnostic.py'
    path = own.relative_to(ROOT).as_posix()
    if frozen and own.read_bytes() != prior.fixed_bytes('HEAD', path):
        raise ValueError('Freeze diagnostic SOURCE before --run')
    hashes = {path: hashlib.sha256(own.read_bytes()).hexdigest()}
    for commit, path in HELPERS:
        raw = (ROOT/path).read_bytes()
        if raw != prior.fixed_bytes(commit, path):
            raise ValueError('Fixed helper changed: '+path)
        hashes[path] = hashlib.sha256(raw).hexdigest()
    return dict(commit=subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
                code_sha256=hashes, frozen_required=frozen, numpy=np.__version__, python=sys.version.split()[0],
                kernel_profile_as_theorem_input=kernel.PROFILE, current_runtime_certified=False)


def checked_inputs():
    payload, inventory = {}, []
    for commit, path, expected in INPUTS:
        raw = (ROOT/path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected or raw != prior.fixed_bytes(commit, path):
            raise ValueError('Fixed input byte mismatch: '+path)
        payload[Path(path).name] = raw
        inventory.append(dict(commit=commit, path=path, sha256=expected, bytes=len(raw)))
    return payload, inventory


def describe():
    source = source_metadata(False)
    _, inputs = checked_inputs()
    factor = proof_floor()
    return dict(mode='STATIC_AND_BYTES_ONLY', SOURCE=source, inputs=inputs,
                diagnostic_source_prepared=True, virtual_bridge_execution_ready=False,
                floor_factor=str(factor), critical_norm_squared=str(COORDINATE_LIMIT/factor),
                static_unit_norm_obstruction=factor > COORDINATE_LIMIT, new_real_numerical_calls=0, **STATUS)


def execute(output):
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT/'artifacts/rpe_m17_norm_proof_feasibility'):
        raise ValueError('Use a fresh diagnostic output directory in its artifact namespace')
    output.mkdir(parents=True, exist_ok=False)
    record = dict(execution_status='STARTED', mode='SAVED_STATE_PROOF_FEASIBILITY_ONLY', **STATUS,
                  scientific_calls=dict(saved_context_load_started=0, norm_pass_started=0, norm_pass_completed=0),
                  sampling_RNG_matrix_action_defect_mean_budget_compile_calls=0)
    def save(): write_json(output/'RESULT.json', record)
    try:
        save()
        record['SOURCE'] = source_metadata(True)
        payload, record['inputs'] = checked_inputs()
        previous = json.loads(payload['RESULT.json'])
        literals = previous['fixed_literals']
        if (previous['SOURCE']['commit'] != 'c3a5cc387b8962b63782dc71ab175147f2d87e33'
                or previous['execution_status'] != 'COMPLETED_MANDATORY_STOP'
                or any(literals[name] != value for name, value in
                       dict(ld='3', m='17', q='131072', r='32', K='2', delta='0.02').items())
                or any(literals['axes'][axis]['N'] != '411' for axis in ['cosine', 'sine'])):
            raise ValueError('Fixed m17 provenance/conditions mismatch')
        record['fixed_literals'] = literals  # Preserve alpha/beta/epsilon; never re-evaluate them.
        record['scientific_calls']['saved_context_load_started'] = 1
        save()
        psi = load_state(payload['numeric_context.npz'])
        record['scientific_calls']['norm_pass_started'] = 1
        save()
        record['diagnostic'] = assess_norm_squared(exact_norm_squared(psi))
        record['scientific_calls']['norm_pass_completed'] = 1
        record['execution_status'] = 'COMPLETED_MANDATORY_STOP'
        return record
    except BaseException as exc:
        record['execution_status'] = 'INTERRUPTED_OR_INVALID_MANDATORY_STOP'
        record['error'] = dict(type=type(exc).__name__, message=str(exc).replace(str(ROOT), '<repository>'))
        raise
    finally:
        save()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--describe', action='store_true')
    mode.add_argument('--run', action='store_true')
    parser.add_argument('--output-dir')
    args = parser.parse_args(argv)
    if args.run and not args.output_dir:
        parser.error('--run requires a fresh --output-dir')
    if args.output_dir and not args.run:
        parser.error('--output-dir requires --run')
    try:
        print(json.dumps(execute(args.output_dir) if args.run else describe(), indent=2))
        return 0
    except Exception as exc:
        print(json.dumps(dict(error=str(exc), execution_status='INVALID_MANDATORY_STOP', **STATUS), indent=2))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
