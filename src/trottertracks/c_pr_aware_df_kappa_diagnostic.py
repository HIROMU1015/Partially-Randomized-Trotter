"""Opt-in saved-input diagnostics only. No CAR target or input is changed.

Import/preflight reads no molecular arrays. Fractions provide an independent
exact oracle for the *stored* binary64 values, not the unknown exact integrals.
Scientific execution requires separate chat permission; no approval proof.
"""
from fractions import Fraction as F
import ctypes
import importlib.metadata
import math
from pathlib import Path
import resource
import time

from . import c1_stage1_execution as io

VERSION = 'kappa_binary64_exact_dyadic_diagnostic_v1'
SCOPE = 'C_PR_AWARE_DF_KAPPA_HERMITICITY_DIAGNOSTIC_V1'
BASE = 'acf61e9f1a4d5fd8948b517b978c65994b797bc5'
FAMILY = 'artifacts/research_c_pr_aware_df_kappa_diagnostic/2026-10-27'
FRESH = FAMILY + '/diagnostic-run01'
LIMITS = dict(input_load=1, input_identity=1, kappa_generation=1,
              binary64_diagnostics=1, exact_dyadic_diagnostics=1, classification=1)
POLICY = dict(norb=4, dtype='float64', convention='spatial_chemist_pqrs',
              contraction='prrq->pq', correction='h - 0.5*T',
              allowance='binary64_operation_scale_guard8_v1', rank_context=8,
              stage1_comparison='kappa_only_structural_Frobenius',
              exact_oracle='Fraction_of_stored_binary64_real_and_imaginary_parts',
              norms=['max_abs_entry', 'Frobenius', 'spectral2_numerical_pilot'],
              target_adoption=False, retry=0, resume=0, mandatory_stop=True)


def checked(root, name, digest, label):
    root = Path(root).resolve()
    p = (root / name).resolve()
    if not p.is_relative_to(root) or io.sha(p) != digest:
        raise ValueError(label + ':' + name)
    return p


def preflight(root, path):
    """Bytes/JSON/versions only, with no NPZ decode or scientific arithmetic."""
    root = Path(root).resolve()
    c = io.read(path)
    if (c['scope'] != SCOPE or c['implementation'] != VERSION
            or c['base_commit'] != BASE or c['status'] != 'UNAPPROVED'
            or c['policy'] != POLICY or c['call_limits'] != LIMITS
            or c['maximum_logical_units'] != sum(LIMITS.values())
            or c['fresh_output'] != FRESH or c['executable'] is not True):
        raise ValueError('DIAGNOSTIC_SCOPE')
    manifest = checked(root, c['source_manifest'], c['source_manifest_sha256'], 'SOURCE_MANIFEST')
    for group, label in ((io.read(manifest)['files'], 'SOURCE'),
                         (c['inputs'], 'INPUT'), (c['evidence'], 'EVIDENCE')):
        for name, digest in group.items():
            checked(root, name, digest, label)
    if c['input'] not in c['inputs'] or c['original_input'] not in c['inputs']:
        raise ValueError('INPUT_BINDING')
    for name, version in c['packages'].items():
        if importlib.metadata.version(name) != version:
            raise ValueError('ENVIRONMENT:' + name)
    import sys
    if list(sys.version_info[:3]) != c['python']:
        raise ValueError('PYTHON_VERSION')
    rounding = ctypes.CDLL(None).fegetround()
    if rounding != 0:
        raise ValueError('FE_TONEAREST_REQUIRED')
    from . import c1_stage1 as stage
    if stage.ALLOWANCE_VERSION != POLICY['allowance']:
        raise ValueError('STAGE1_ALLOWANCE_VERSION')
    if (root / FRESH).exists():
        raise ValueError('FRESH_OUTPUT_REQUIRED_NO_RETRY')
    return dict(status='PASSIVE_IDENTITIES_PASS_NO_SCIENCE', scientific_calls=0,
                source_manifest_sha256=c['source_manifest_sha256'],
                maximum_logical_units=sum(LIMITS.values()), executable=True,
                rounding_mode='FE_TONEAREST',
                permission='separate explicit chat science permission required')


def validate(h, g, constant):
    import numpy as np
    h, g = np.asarray(h), np.asarray(g)
    n = len(h) if h.ndim == 2 else 0
    if (not n or h.shape != (n, n) or g.shape != (n,) * 4
            or h.dtype not in (np.dtype('float64'), np.dtype('complex128'))
            or g.dtype != h.dtype or np.ndim(constant) != 0
            or not np.isrealobj(constant)):
        raise ValueError('ARRAY_SCHEMA')
    from .c1_stage1 import finite_arrays
    finite_arrays(h, g, constant)
    return h, g


def load_inputs(root, c):
    """Whitelist h/g/scalar only, including original-source identity check."""
    import numpy as np
    def load(name, keys):
        with np.load(checked(root, name, c['inputs'][name], 'INPUT'), allow_pickle=False) as p:
            return tuple(p[k] for k in keys)
    return (load(c['input'], ('h', 'g', 'constant')),
            load(c['original_input'], ('one_mo', 'eri_chemist', 'nuclear_repulsion')))


def input_identity(pair):
    import numpy as np
    current, original = pair
    for arrays in pair:
        h, g = validate(*arrays)
        if h.shape != (4, 4) or g.dtype != np.dtype('float64'):
            raise ValueError('FIXED_H4_SCHEMA')
    if any(a.dtype != b.dtype or a.shape != b.shape or not np.array_equal(a, b)
           for a, b in zip(current, original, strict=True)):
        raise ValueError('ORIGINAL_INPUT_ARRAY_MISMATCH')
    return dict(status='ORIGINAL_INPUT_ARRAY_IDENTITY_PASS', convention=POLICY['convention'])


def generate(h, g, constant):
    import numpy as np
    from .c1_stage1 import corrected_kappa, finite_arrays
    h, g = validate(h, g, constant)
    # Preserve the existing default einsum path; no optimize/reordering option.
    T = np.einsum('prrq->pq', g)
    kappa = corrected_kappa(h, g)
    finite_arrays(T, kappa)
    if not np.array_equal(kappa, h - 0.5 * T):
        raise ValueError('CORRECTION_PATH_MISMATCH')
    return T, kappa


def metrics(a):
    """Norms are numerical observations, not rigorous enclosures."""
    import numpy as np
    a = np.asarray(a)
    fro = math.hypot(*(float(x) for z in a.flat for x in (z.real, z.imag)))
    maximum = max((math.hypot(float(z.real), float(z.imag)) for z in a.flat), default=0.)
    spectral = float(np.linalg.norm(a, 2))
    if not all(math.isfinite(v) for v in (fro, maximum, spectral)):
        raise ValueError('NONFINITE_DIAGNOSTIC_NORM')
    return dict(max_abs_entry=maximum, Frobenius=fro,
                spectral2_numerical_pilot=spectral, rigorous_spectral_bound=None)


def binary64_diagnostics(h, g, constant, T, kappa):
    import numpy as np
    from . import c1_stage1 as stage
    rows = {}
    for name, a in (('h', h), ('T', T), ('kappa', kappa)):
        delta = a - a.conj().T
        stage.finite_arrays(delta)
        mismatch = np.argwhere(a != a.conj().T).tolist()
        rows[name] = dict(dtype=a.dtype.str, shape=list(a.shape), finite=True,
                          exact_Hermitian=not mismatch, unequal_indices=mismatch,
                          scale=metrics(a), residual=metrics(delta))
    n = len(h)
    delta = kappa - kappa.conj().T
    base = abs(float(constant)) + 2 * float(np.sum(np.abs(kappa)))
    operations = stage.operation_lengths(n, 1 << (2 * n), 8)['base']
    limit = math.sqrt(n) * stage.allowance(base, base, operations, operations, n, delta)
    stage.finite_arrays(base, limit)
    error = rows['kappa']['residual']['Frobenius']
    # Reproduce the registered check's actual norm implementation as well.
    stage_error = float(np.linalg.norm(delta.ravel()))
    rows['stage1_comparison'] = dict(version=stage.ALLOWANCE_VERSION,
        error=stage_error, stable_Frobenius_observation=error,
        allowance=limit, passed=bool(stage_error <= limit),
        base_scale=base, operations=operations, absolute=True, rtol=None,
        applicability='kappa only; not reused for h/T or an operator certificate')
    scale = rows['kappa']['scale']['Frobenius']
    rows['relative_antiHermitian_Frobenius'] = error / (2 * scale) if scale else None
    rows['relative_status'] = 'DEFINED' if scale else 'UNRESOLVED_ZERO_SCALE'
    rows['norm_qualification'] = 'BINARY64_NUMERICAL_DIAGNOSTIC_ONLY'
    return rows


# Exact complex scalars are pairs of rational real/imaginary components. This
# small contraction oracle avoids implementing any CAR or numerical library.
def q(z):
    return F(float(z.real)), F(float(z.imag))


def add(a, b):
    return a[0] + b[0], a[1] + b[1]


def mul(a, s):
    return a[0] * s, a[1] * s


def sub(a, b):
    return add(a, mul(b, -1))


def exact_record(a):
    return dict(entries=[[[str(x), str(y)] for x, y in row] for row in a],
                nonzero_indices=[[i, j] for i, row in enumerate(a)
                                 for j, z in enumerate(row) if z != (0, 0)],
                Frobenius_squared_exact=str(sum(x*x + y*y for row in a for x, y in row)),
                spectral_upper_entry_l1_exact=str(sum(abs(x) + abs(y) for row in a for x, y in row)))


def dyadic_diagnostics(h, g, T, kappa):
    """Separate inherited asymmetry from contraction/subtraction roundoff.

    No Arb approximation is needed: these are only finite dyadic sums/differences.
    This oracle does not establish how PySCF originally rounded its integrals.
    """
    n = len(h)
    H = [[q(x) for x in row] for row in h]
    t = [[tuple(sum((q(g[p, r, r, s])[k] for r in range(n)), F(0))
                for k in (0, 1)) for s in range(n)] for p in range(n)]
    tt = [[q(x) for x in row] for row in T]
    kk = [[q(x) for x in row] for row in kappa]
    def zipmat(a, b, op):
        return [[op(a[i][j], b[i][j]) for j in range(n)] for i in range(n)]
    def adj(a):
        return [[(a[j][i][0], -a[j][i][1]) for j in range(n)] for i in range(n)]
    def scale(a, c):
        return [[mul(z, c) for z in row] for row in a]
    def delta(a):
        return zipmat(a, adj(a), sub)
    ideal = zipmat(H, scale(t, F(1, 2)), sub)
    e_t = zipmat(tt, t, sub)
    e_sub = zipmat(kk, zipmat(H, scale(tt, F(1, 2)), sub), sub)
    propagated = zipmat(delta(ideal), scale(delta(e_t), F(-1, 2)), add)
    propagated = zipmat(propagated, delta(e_sub), add)
    if propagated != delta(kk):
        raise ValueError('EXACT_DECOMPOSITION_IDENTITY_FAIL')
    herm = scale(zipmat(kk, adj(kk), add), F(1, 2))
    anti = scale(delta(kk), F(1, 2))
    matrices = dict(delta_h=delta(H), delta_T_exact=delta(t),
        delta_kappa_exact_formula=delta(ideal), delta_kappa_returned=delta(kk),
        T_rounding=e_t, kappa_half_scale_and_subtraction_rounding=e_sub,
        delta_T_rounding=delta(e_t), delta_subtraction_rounding=delta(e_sub),
        kappa_literal=kk, kappa_H_exact_diagnostic_only=herm,
        kappa_A_exact_diagnostic_only=anti)
    return dict(status='EXACT_STORED_DYADIC_DECOMPOSITION_PASS',
                matrices={k: exact_record(v) for k, v in matrices.items()},
                identity='delta_K=delta_h-delta_T_exact/2-delta_E_T/2+delta_E_sub',
                original_integral_rounding_cause=None, changed_input=False)


def classify(binary, exact):
    if binary is None or exact is None:
        return dict(status='UNRESOLVED_MISSING_DIAGNOSTICS', cause=None,
                    CAR_target=None, raw_CAR_bound=None, F2_retry=False)
    records = exact['matrices']
    nonzero = lambda name: bool(records[name]['nonzero_indices'])
    inherited = nonzero('delta_kappa_exact_formula')
    literal = nonzero('delta_kappa_returned')
    return dict(status='DIAGNOSTIC_COMPLETE', observations=dict(
        A_stored_h_asymmetry=nonzero('delta_h'),
        B_exact_contraction_of_stored_g_asymmetry=nonzero('delta_T_exact'),
        C_arithmetic_alone_breaks_exact_formula_symmetry=literal and not inherited,
        contraction_antiHermitian_rounding=nonzero('delta_T_rounding'),
        subtraction_antiHermitian_rounding=nonzero('delta_subtraction_rounding'),
        D_SOURCE_input_convention='NO_MISMATCH_OBSERVED_BY_PINNED_PROVENANCE_CHECKS'),
        E_original_generation_or_reduction_tree_cause='UNRESOLVED',
        literal_kappa_exact_Hermitian=not literal,
        stage1_numerical_allowance_PASS=binary['stage1_comparison']['passed'],
        CAR_target=None, raw_CAR_bound=None, model_bridge=None,
        whole_law_bound=None, CA_achieved=None, shots=None, total_resources=None,
        target_adopted=False, F2_retry=False, mandatory_stop=True)


def execute(root, c, out, call, *, loader=load_inputs):
    pair = call('input_load', 'stageA and original h/g/c only', lambda: loader(root, c))
    identity = call('input_identity', 'source array equality', lambda: input_identity(pair))
    io.save(out / 'input_identity.json', identity)
    h, g, constant = pair[0]
    T, kappa = call('kappa_generation', 'unchanged corrected_kappa', lambda: generate(h, g, constant))
    binary = call('binary64_diagnostics', 'h/T/kappa residuals and Stage1 allowance',
                  lambda: binary64_diagnostics(h, g, constant, T, kappa))
    io.save(out / 'binary64.json', binary)
    exact = call('exact_dyadic_diagnostics', 'stored dyadic contraction and error decomposition',
                 lambda: dyadic_diagnostics(h, g, T, kappa))
    io.save(out / 'exact_dyadic.json', exact)
    result = call('classification', 'observations only; no target adoption', lambda: classify(binary, exact))
    io.save(out / 'result.json', result)
    return result


def run(root, path):
    """Dormant until separately permitted; historical UNAPPROVED token stays."""
    from .c1_stage3_integrated import Calls
    root = Path(root).resolve()
    preflight(root, path)
    c = io.read(path)
    out = root / FRESH
    out.mkdir(parents=True, exist_ok=False)
    call = Calls(out, LIMITS, sum(LIMITS.values()))
    start = time.monotonic()
    try:
        io.save(out / 'SOURCE.json', io.read(root / c['source_manifest']))
        io.save(out / 'contract.json', c)
        return execute(root, c, out, call)
    except BaseException as error:
        call.stop.set()
        io.save(out / 'failure.json', dict(error=str(error), first_failure=call.first_failure))
        raise
    finally:
        io.save(out / 'STOP.json', dict(mandatory_stop=True, scientific_units=sum(call.counts.values()),
            counts=dict(call.counts), first_failure=call.first_failure, retry=0, resume=0,
            new_fit=0, CAR_reference=0, circuit_build_compile=0, input_modified=False))
        io.save(out / 'resources.json', dict(wall_seconds=time.monotonic() - start,
            peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
