"""Independent saved-H/eigh difference diagnosis. Default entry is bytes-only.

The explicit execution flag is for a caller with separate user permission.
No ground, spectral-gap, circuit, signal, or research-target operation exists.
"""
from fractions import Fraction as F
from io import BytesIO
from pathlib import Path
import hashlib
import importlib.metadata
import resource
import subprocess
import sys
import time

from . import c1_stage1_execution as io

BASE = '56d2f85e471d2ab3927c9c9568f31034df143b26'
SCOPE = 'C_PR_SAVED_H_EIGH_L_DIFFERENCE_ONLY_V1'
FAMILY = 'artifacts/research_c_saved_eigh_difference/2026-11-05'
CONTRACT = 'docs/research/research_c_saved_eigh_difference_contract_unapproved_20261105.json'
SOURCE = FAMILY+'/technical/SOURCE.json'
ENTRY = 'scripts/run_research_c_saved_eigh_difference.py'
LIMITS = dict(metadata_load=1, arrays_load_exact=1, eigh_semantics=1, difference_norm=1)
MAXIMUM = sum(LIMITS.values())
H_FILE = 'artifacts/research_c_near_practical/2026-11-03/run01/physical/PR_update016.npz'
H_SHA256 = 'ebdebe8f2c3f156c8202834181e143298d667490b3f33fc611dc0fab97b976cd'
BASIS_FILE = 'artifacts/research_c_near_practical/2026-11-03/run01/physical_reference.npz'
CANDIDATE = 'a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a'
ORIGIN = 'artifacts/research_c_near_practical_r32/2026-11-03/run01/contract.json'
LINEAGE = 'artifacts/research_c_hermitian_oracle_bridge/2026-11-05/technical/producer_provenance.json'
OLD_STOP = 'artifacts/research_c_direct_m0_oracle_binding/2026-11-05/science-run01/STOP.json'


def basis_indices():
    # Producer's sorted interleaved occupation labels; no physical H is built.
    return [s for s in range(256)
            if sum((s >> q) & 1 for q in (0, 2, 4, 6)) == 2
            and sum((s >> q) & 1 for q in (1, 3, 5, 7)) == 2]


def fixed_spec():
    return dict(scope=SCOPE, base_commit=BASE, candidate_sha256=CANDIDATE,
        roles=dict(H=H_FILE, basis=BASIS_FILE), array='H', shape=[36,36], dtype='<c16',
        basis_array='sector_basis_indices', basis_dtype='<i8', expected_basis_indices=basis_indices(),
        numpy='1.26.4', UPLO='L', exact_components='finite binary64 real/imag -> exact rational',
        precision_bits=192, difference='H_saved-H_eigh',
        norm='exact squared Frobenius; Arb192 outward sqrt; operator norm <= Frobenius upper',
        call_limits=LIMITS, maximum_logical_units=MAXIMUM,
        entry=ENTRY, source_manifest=SOURCE, fresh_output=FAMILY+'/science-run01',
        packages={'numpy':'1.26.4','python-flint':'0.8.0'}, python=[3,11,1], FLINT='3.3.1',
        H_target=None, target_adopted=False, ground_eigensolve=0, ground_certificate=0,
        spectral_gap=0, XY_signal=0, sampling=0, build_compile=0,
        Stage_C=False, actual_CA=None, retry=0, resume=0, mandatory_stop=True)


def _path(root, name):
    p = (root/name).resolve()
    if not p.is_relative_to(root):
        raise ValueError('PATH_OUTSIDE_THIS_WORKTREE')
    return p


def metadata_binding(root, c):
    origin = io.read(root/ORIGIN)
    lineage = io.read(root/LINEAGE)
    old = io.read(root/OLD_STOP)
    if (origin['endpoints']['PR']['reference'] != c['roles']['H']
            or origin['candidate_sha256']['PR'] != c['candidate_sha256']
            or origin['physical_reference'] != c['roles']['basis']
            or origin['policy']['oracle'] != 'own_DF_ground_state_oracle'
            or origin['saved_inputs'][c['roles']['H']] != c['saved_inputs'][c['roles']['H']]
            or lineage['H']['path'] != c['roles']['H']
            or lineage['H']['sha256'] != c['saved_inputs'][c['roles']['H']]
            or lineage['numpy']['version'] != c['numpy']
            or lineage['numpy']['UPLO'] != 'default L'):
        raise ValueError('SAVED_H_PRODUCER_BINDING')
    if (old['scientific_units'] != 3 or old['retry'] != 0 or old['resume'] != 0
            or old['failed']['error'] != 'EXACT_SYMMETRY_REQUIRED_NO_HERMITIANIZATION'
            or old['mandatory_stop'] is not True):
        raise ValueError('OLD_DIRECT_M0_STOP_PRESERVATION')
    return dict(candidate=CANDIDATE, H_file=c['roles']['H'], H_array='H',
        H_sha256=c['saved_inputs'][c['roles']['H']], basis_file=c['roles']['basis'],
        basis_sha256=c['saved_inputs'][c['roles']['basis']], UPLO='L',
        producer='prepare_arm -> ground(sector H) -> loader.ground -> MolecularBackend._ground -> np.linalg.eigh(matrix)',
        lineage=LINEAGE, research_target_adopted=False, old_STOP_preserved=True,
        unused_old_12_calls_transferable=0)


def preflight(root, path=None):
    """JSON/byte/hash/Git/package-metadata only. No numerical imports or output."""
    root = Path(root).resolve()
    c = io.read(path or root/CONTRACT)
    if any(c.get(k) != v for k, v in fixed_spec().items()):
        raise ValueError('FIXED_DIFFERENCE_ONLY_SCOPE')
    source_path = _path(root, c['source_manifest'])
    if io.sha(source_path) != c['source_manifest_sha256']:
        raise ValueError('SOURCE_MANIFEST_HASH')
    source = io.read(source_path)
    if source['scope'] != SCOPE or source['base_commit'] != BASE:
        raise ValueError('INDEPENDENT_SOURCE_SCOPE')
    if source['saved_inputs'] != c['saved_inputs']:
        raise ValueError('FROZEN_INPUT_INVENTORY')
    for group in (source['files'], source['producer_files'], c['saved_inputs']):
        for name, digest in group.items():
            if io.sha(_path(root, name)) != digest:
                raise ValueError('SOURCE_OR_INPUT_BYTES:'+name)
    if set(c['roles'].values()) - set(c['saved_inputs']):
        raise ValueError('UNPINNED_ARRAY_ROLE')
    if c['saved_inputs'][H_FILE] != H_SHA256:
        raise ValueError('FIXED_SAVED_H_SHA256')
    binding = metadata_binding(root, c)
    if list(sys.version_info[:3]) != c['python']:
        raise ValueError('PINNED_PYTHON')
    if any(importlib.metadata.version(k) != v for k,v in c['packages'].items()):
        raise ValueError('PINNED_PACKAGE_METADATA')
    if _path(root, c['fresh_output']).exists():
        raise ValueError('FRESH_NO_RETRY_RESUME')
    missing = []
    sha = c.get('scientific_SOURCE')
    if c.get('executable') is not True or c.get('status') != 'UNAPPROVED_EXECUTABLE_AFTER_SEPARATE_PERMISSION':
        missing.append('contract_not_frozen_executable_proposal')
    if not isinstance(sha, str) or len(sha) != 40 or any(x not in '0123456789abcdef' for x in sha):
        missing.append('scientific_SOURCE_commit_not_frozen')
    else:
        for name, digest in {c['source_manifest']:c['source_manifest_sha256'],
                             **source['files'], **source['producer_files']}.items():
            frozen = subprocess.run(['git','show',sha+':'+name], cwd=root, capture_output=True)
            if frozen.returncode or hashlib.sha256(frozen.stdout).hexdigest() != digest:
                raise ValueError('FROZEN_SCIENTIFIC_SOURCE_COMMIT:'+name)
    return dict(status='NOT_READY' if missing else 'UNAPPROVED_EXECUTABLE_AFTER_SEPARATE_PERMISSION',
        execution_path_ready=not missing, missing=missing, scientific_execution_permitted=False,
        scientific_calls=0, arrays_decoded=0, maximum_logical_units=MAXIMUM,
        fresh_output=c['fresh_output'], scientific_SOURCE=sha, binding=binding,
        mandatory_stop=True)


def _matrix_record(a):
    from . import c_hermitian_oracle_bridge as bridge
    return dict(matrix_id=bridge.matrix_id(a),
        components=[[[str(r),str(i)] for r,i in row] for row in a])


def load_exact(root, c):
    """Read only H and occupation labels, from the SAME bytes just hashed."""
    import numpy as np
    from . import c_hermitian_oracle_bridge as bridge
    arrays = {}
    for role, key in [('H', c['array']), ('basis', c['basis_array'])]:
        raw = (root/c['roles'][role]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != c['saved_inputs'][c['roles'][role]]:
            raise ValueError('DECODED_INPUT_BYTES:'+role)
        with np.load(BytesIO(raw), allow_pickle=False) as archive:
            arrays[role] = archive[key]
    h, basis = arrays['H'], arrays['basis']
    if list(h.shape) != c['shape'] or h.dtype.str != c['dtype']:
        raise ValueError('H_SHAPE_DTYPE')
    if not np.isfinite(h).all():
        raise ValueError('NONFINITE_SAVED_H')
    if (basis.dtype.str != c['basis_dtype'] or basis.ndim != 1
            or basis.tolist() != c['expected_basis_indices'] or len(basis) != len(h)):
        raise ValueError('SAVED_SECTOR_BASIS_IDENTITY')
    exact = bridge.matrix([[complex(x) for x in row] for row in h])
    return exact, dict(H=io.array_record(h), basis=io.array_record(basis),
        saved_matrix_id=bridge.matrix_id(exact), basis_reordered=False,
        binary64_components_preserved=True)


def norm_result(a, problem):
    from flint import ctx, __FLINT_VERSION__
    from . import c_hermitian_oracle_bridge as bridge
    from . import c_pr_aware_df_zsxx_enclosure as z
    if __FLINT_VERSION__ != '3.3.1':
        raise ValueError('PINNED_FLINT')
    b = problem.values
    bridge.hermitian(b)
    if problem.uplo != 'L' or problem.saved_matrix_id != bridge.matrix_id(a):
        raise ValueError('EIGH_SEMANTICS_IDENTITY')
    bound = bridge.difference_bound(a, b)
    s, upper = F(bound['frobenius_squared_exact']), F(bound['operator_norm_upper_exact'])
    with z._BALL_LOCK, ctx.workprec(192):
        lower = max(F(0), z._fraction(z._real(s).sqrt().lower()))
    if not 0 <= lower <= upper or lower**2 > s or upper**2 < s:
        raise ValueError('ARBITRARY_PRECISION_ENCLOSURE_NOT_VALIDATED')
    delta = tuple(tuple(tuple(a[i][j][k]-b[i][j][k] for k in range(2))
        for j in range(len(a))) for i in range(len(a)))
    return dict(status='VALIDATED_SAVED_EIGH_DIFFERENCE_ONLY', **bound,
        scope=SCOPE, H_saved_id=bridge.matrix_id(a), H_eigh_id=bridge.matrix_id(b),
        difference_id=bridge.matrix_id(delta), difference_definition='H_saved-H_eigh',
        frobenius_norm_interval_exact=[str(lower),str(upper)],
        operator_norm_is_exact_value=False, difference_exactly_zero=s == 0,
        rounding_enclosed=True, magnitude_threshold=None, H_target=None,
        research_target_adopted=False, ground_certificate=None, spectral_gap=None,
        XY_signal=None), delta


def execute(root, c, out, call):
    """Shared scientific/artificial pipeline; only run() loads the fixed contract."""
    from . import c_hermitian_oracle_bridge as bridge
    binding = call('metadata_load', 'saved producer, input and old STOP provenance',
                   lambda: metadata_binding(root, c))
    io.save(out/'producer_binding.json', binding)
    a, record = call('arrays_load_exact', 'H and saved sector labels; exact binary64 components',
                     lambda: load_exact(root, c))
    io.save(out/'arrays.json', record)
    io.save(out/'H_saved_exact.json', _matrix_record(a))
    p = call('eigh_semantics', 'fixed L; real diagonal and conjugate upper triangle',
        lambda: bridge.eigh_problem(a, uplo=c['UPLO'], dtype=c['dtype'],
            source_sha256=c['source_manifest_sha256'],
            input_sha256=c['saved_inputs'][c['roles']['H']]))
    io.save(out/'H_eigh_exact.json', _matrix_record(p.values))
    result, delta = call('difference_norm', 'exact Frobenius square; Arb192 outward enclosure',
                         lambda: norm_result(a, p))
    io.save(out/'difference_exact.json', _matrix_record(delta))
    io.save(out/'result.json', result)
    return result


def _attempt(root, c, out):
    from .c1_stage3_integrated import Calls
    call = Calls(out, LIMITS, MAXIMUM)
    wall, cpu = time.monotonic(), time.process_time()
    try:
        io.save(out/'contract.json', c)
        io.save(out/'SOURCE.json', io.read(root/c['source_manifest']))
        result = execute(root, c, out, call)
        if dict(call.counts) != LIMITS:
            raise ValueError('FIXED_FOUR_CALL_COVERAGE')
        return result
    except BaseException as exc:
        call.stop.set()
        if call.first_failure is None:
            call.first_failure = dict(error_type=type(exc).__name__, error=str(exc))
        io.save(out/'partial_failure.json', dict(error_type=type(exc).__name__, error=str(exc)))
        raise
    finally:
        io.save(out/'STOP.json', dict(scope=SCOPE, call_counts=dict(call.counts),
            scientific_units=sum(call.counts.values()), maximum=MAXIMUM,
            failed=call.first_failure, retry=0, resume=0, mandatory_stop=True,
            research_target_adopted=False, Stage_C_executed=False, actual_CA_achieved=None))
        io.save(out/'resources.json', dict(wall_seconds=time.monotonic()-wall,
            CPU_seconds=time.process_time()-cpu,
            process_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


def run(root, path=None, *, execute_science=False):
    if not execute_science:
        raise PermissionError('SEPARATE_EXPLICIT_FOUR_CALL_DIFFERENCE_PERMISSION_REQUIRED')
    root = Path(root).resolve()
    ready = preflight(root, path)
    if not ready['execution_path_ready']:
        raise ValueError('SCIENTIFIC_SOURCE_NOT_READY')
    c = io.read(path or root/CONTRACT)
    out = _path(root, c['fresh_output'])
    out.mkdir(parents=True, exist_ok=False)
    return _attempt(root, c, out)
