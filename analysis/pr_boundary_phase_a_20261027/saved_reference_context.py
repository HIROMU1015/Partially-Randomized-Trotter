"""Signal-context adapter for the two proposed boundaries; not a launcher.

No science runs on import. Call only within separately authorized signal work.
Reuse the frozen restore/algebra kernels while excluding reference generation.
"""
import hashlib
import json
import math
from types import SimpleNamespace

SCOPES = {
    'H4': dict(id='3a55c1dac3947ae23c76', prefix=3, R_outer=128, K=2,
               rank=10, reference_id='f85eb7b56ba5131d1eb1'),
    'H6': dict(id='0e0e9fb6da33cb6ab2ab', prefix=7, R_outer=11, K=6,
               rank=19, reference_id='efbb07c848b303a6abab'),
}


def saved_reference(row, record):
    """Validate fixed scope before any scientific import or input restoration."""
    scope = SCOPES.get(row.get('system'))
    if scope is None:
        raise ValueError('FIXED_BOUNDARY_SYSTEM')
    expected = dict(system=row['system'], method='PR', formula='4th(m5_best)',
                    prefix=scope['prefix'], T=2.0, q=1,
                    R_outer=scope['R_outer'], K=scope['K'])
    if any(type(row.get(k)) is not type(v) or row[k] != v for k, v in expected.items()):
        raise ValueError('FIXED_BOUNDARY_SIGNAL_ONLY')
    fields = tuple(row[k] for k in
                   ('system', 'method', 'formula', 'prefix', 'T', 'q', 'R_outer', 'K'))
    identity = hashlib.sha256(json.dumps(fields).encode()).hexdigest()[:20]
    if row.get('id') != scope['id'] or identity != scope['id']:
        raise ValueError('BOUNDARY_CANDIDATE_ID')
    origin = record['candidate']
    if (origin['id'] != scope['reference_id'] or origin['system'] != row['system']
            or origin['T'] != row['T']):
        raise ValueError('SAVED_T2_REFERENCE_SELECTOR')
    value = record['signal']['reference']
    if not all(math.isfinite(value[a]) for a in ('real', 'imag')):
        raise ValueError('SAVED_REFERENCE_FINITE')
    return complex(value['real'], value['imag'])


def signal_context(row, config, record, *, operations=None):
    """Restore saved H/state; reuse one T2 scalar reference, never solve full H.

    config is the per-system saved input_info mapping, not the old launch config.
    operations injection supports artificial tests without scientific imports.
    """
    reference = saved_reference(row, record)
    scope = SCOPES[row['system']]
    if config['binding']['rank'] != scope['rank']:
        raise ValueError('SAVED_RANK_BINDING')
    if operations is None:
        import numpy as np
        from h46_m5_tol_native import restore
        from h46_new2_core import occupation_one_body, spectrum, matrix_only_tail_coefficients
        operations = SimpleNamespace(np=np, restore=restore,
            occupation_one_body=occupation_one_body, spectrum=spectrum,
            matrix_only_tail_coefficients=matrix_only_tail_coefficients)
    op = operations
    ham, sector, full, state, metadata = op.restore(dict(config, system=row['system']))
    binding = config['binding']
    if (metadata['hamiltonian_hash'] != binding['hamiltonian']
            or metadata['state_hash'] != binding['state']
            or ham.n_blocks != scope['rank']):
        raise ValueError('SAVED_INPUT_IDENTITY')
    matrices = [op.occupation_one_body(ham.one_body, sector)]
    for weight, fragment in zip(ham.lambdas, ham.g_matrices):
        value = op.occupation_one_body(fragment, sector)
        matrices.append(weight * (value @ value))
    prefix = scope['prefix']
    scalar, lam = op.matrix_only_tail_coefficients(ham, prefix)
    tail = sum(matrices[prefix + 1:], op.np.zeros_like(matrices[0]))
    tail = tail - scalar * op.np.eye(len(state))
    # Only spectra needed by evolve_batch: deterministic prefix and this tail.
    # No full-H spectrum, expm_multiply, state evolution or reference file solve.
    specs = tuple(op.spectrum(matrix) for matrix in matrices[:prefix + 1])
    tails = {prefix: dict(scalar=scalar, lam=lam, spec=op.spectrum(tail))}
    return SimpleNamespace(system=row['system'], ham=ham, sector=sector, full=full,
        state=state, specs=specs, tails=tails, refs={row['T']: reference}, matrices=matrices)
