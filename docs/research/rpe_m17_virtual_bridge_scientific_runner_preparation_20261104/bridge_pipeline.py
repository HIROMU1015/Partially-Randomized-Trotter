"""One saved-input Model-II bound pass. No default real-data execution."""
from fractions import Fraction as Q
import json
import math
from pathlib import Path
import sys
import zipfile

import numpy as np
from flint import ctx

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'docs/research/rpe_m17_openblas_kernel_bound_preflight_20261104'))
import kernel_bounds as kernel
import virtual_bridge_adapter as bridge
import run_virtual_bridge as prior
from runtime_checks import validate_array

BITS = 96


def load_fixed_data():
    folder = ROOT/prior.INCLUSION
    saved = json.loads((folder/'RESULT.json').read_text())
    export = json.loads((folder/'source_replay.json').read_text())
    laws = json.loads((folder/'categorical_laws.json').read_text())
    with zipfile.ZipFile(folder/'enclosure_operators.zip') as archive:
        if archive.namelist() != ['96.json']:
            raise ValueError('Unexpected saved operator precision/inventory')
        halves = json.loads(archive.read('96.json'))
    with np.load(ROOT/prior.INPUTS[4][1], allow_pickle=False) as arrays:
        psi = arrays['state']
    with np.load(ROOT/prior.INPUTS[5][1], allow_pickle=False) as arrays:
        forward, reverse = arrays['forward'], arrays['reverse']
    for value, shape in [(psi, (70,)), (forward, (70, 70)), (reverse, (70, 70))]:
        validate_array(value, shape)
    expected = np.asarray([complex(float.fromhex(re), float.fromhex(im)) for re, im in export['state']], dtype=complex)
    if psi.tobytes() != expected.tobytes():
        raise ValueError('Raw state/export byte identity mismatch')
    literals = saved['original_fixed_literals']
    fixed = dict(ld='3', m='17', q='131072', r='32', K='2', delta='0.02')
    if (any(literals[name] != value for name, value in fixed.items())
            or any(literals['axes'][axis]['N'] != '411' for axis in ('cosine', 'sine'))
            or export['n_qubits'] != 8 or export['n_electrons'] != 4
            or export['identity_policy'] != 'extract_identity_phase' or export['coefficient_atol'] != 1e-12
            or len(export['components']) != 218 or len(export['bases']) != 11
            or len(export['deterministic_blocks']) != 4):
        raise ValueError('Fixed m17 context mismatch')
    if ([float(item['probability64']).hex() for item in export['components']]
            != laws['component']['input_probability64_hex']):
        raise ValueError('Fixed component/CDF identity mismatch')
    if any(laws[name]['grid_bits'] != 53 for name in ('component', 'order')):
        raise ValueError('Expected saved uniform53 law')
    enclosure, = saved['enclosures']
    angles = [float.fromhex(value) for value in saved['fixed_theta64_hex']]
    if len(angles) != 2 or not all(math.isfinite(x) for x in angles):
        raise ValueError('Invalid saved theta64')
    return dict(export=export, laws=laws, halves=halves, psi=psi, forward=forward, reverse=reverse,
                literals=literals, saved_enclosure=enclosure, angles=angles)


def construct_context(data, output):
    numeric = bridge.numeric_components(data['export'])
    for value in numeric.values():
        validate_array(value, data['forward'].shape)
    trig = [(math.cos(angle), math.sin(angle)) for angle in data['angles']]
    if any(not math.isfinite(x) for pair in trig for x in pair):
        raise ValueError('Nonfinite trig output')
    arrays = {f'P{i:03d}': numeric[item['id']] for i, item in enumerate(data['export']['components'])}
    arrays.update(psi=data['psi'], forward=data['forward'], reverse=data['reverse'])
    temporary = Path(output)/'numeric_context.part.npz'
    np.savez_compressed(temporary, **arrays)
    temporary.replace(Path(output)/'numeric_context.npz')
    return numeric, trig


def evaluate(data, numeric, trig):
    """Also usable with artificial dimension/state/law fixtures, not real loaders."""
    size = len(data['psi'])
    for value, shape in [(data['psi'], (size,)), (data['forward'], (size, size)), (data['reverse'], (size, size))]:
        validate_array(value, shape)
    bounds = kernel.path_bounds(size, size)
    saved = data['saved_enclosure']
    state_norm = saved['state']['saved_vector_norm_squared']
    out = bridge.bound_context(data['export'], numeric, data['forward'], data['reverse'], data['halves'],
        data['angles'], trig, data['laws']['order']['grid_masses'], [0, 2],
        r=int(data['literals']['r']), q=int(data['literals']['q']), norm_squared=state_norm, kernel=bounds)
    b = bridge.b
    psi_norm = b.upper(bridge.interval(state_norm).sqrt())
    growth = out['uniform_growth']
    boundary = b.upper(bridge.nonnegative(growth.gain)*psi_norm+bridge.nonnegative(growth.additive))
    frobenius = max([item['frobenius'] for item in out['component_bounds'].values()]
                   + [b.frobenius(bridge.exact_matrix(data[name])) for name in ('forward', 'reverse')])
    scalar = max([b.real(1)]+[b.upper(abs(b.real(value))) for pair in trig for value in pair])
    finite = kernel.finite_intermediate_envelope(matrix_frobenius=b.dyadic(frobenius),
        initial_norm=b.dyadic(psi_norm), boundary_norm=b.dyadic(boundary), scalar_modulus=b.dyadic(scalar))
    magnitude = b.dyadic(out['uniform_magnitude'])
    abs_bound = kernel.complex_abs_error(magnitude) if finite and magnitude <= 1 << 400 else None
    complete = bridge.guard_completion(out['uniform_magnitude'], abs_evaluation_error=abs_bound,
                                       finite_intermediates=finite)
    literals = data['literals']
    axes = bridge.classify_axes(saved['logical_signed_mean_interval'],
        {coordinate: float(literals['saved_mean'][coordinate]) for coordinate in ('real', 'imag')},
        {axis: saved['axes'][axis]['shot_half_width'] for axis in ('cosine', 'sine')},
        {axis: literals['axes'][axis]['epsilon'] for axis in ('cosine', 'sine')},
        out['average_error'], completion=complete, arithmetic_resolved=finite)
    decision = ('CONDITIONAL_VIRTUAL_IID_SUFFICIENT_CONDITION_CONFIRMED'
                if all(axis['decision'] == 'SUFFICIENT_CONDITION_CONFIRMED' for axis in axes.values())
                else 'ARITHMETIC_BOUND_OBTAINED_BUT_COMPLETION_UNRESOLVED' if finite and not complete
                else 'UNDETERMINED_FINITE_ENVELOPE' if not finite else 'NOT_CERTIFIED_BY_THIS_BOUND')
    return {'bits': BITS, 'local_model_II_status': decision, 'finite_intermediates_proved': finite,
            'all_support_completion_proved': complete, 'complex_abs_error_upper': str(abs_bound) if abs_bound is not None else None,
            'conditional_average_error_upper': str(b.dyadic(out['average_error'])),
            'conditional_uniform_error_upper': str(b.dyadic(out['uniform_error'])),
            'uniform_signal_magnitude_upper': str(magnitude), 'boundary_norm_upper': str(b.dyadic(boundary)),
            'matrix_frobenius_upper': str(b.dyadic(frobenius)), 'axes': axes,
            'component_bounds': {name: {field: str(b.dyadic(value)) for field, value in item.items()}
                                 for name, item in out['component_bounds'].items()},
            'kernel_bounds': {name: str(getattr(bounds, name)) for name in bounds.__dataclass_fields__},
            'U_exec_upper': None, 'fixed_PCG64_IID_certified': False}
