"""Saved H4 prefix DF-budget evaluation; import/preflight perform no science.

The old H4 SOURCE is immutable. The small sector adapter below mirrors
saved_xdf_sector, but journals the actual 36 matvecs individually. All DF,
sector, bit-order, SVD and allowance mathematics use existing implementations.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
import hashlib
import importlib.metadata
from pathlib import Path
import resource
import subprocess
import time
from types import SimpleNamespace

from . import c1_stage1 as stage1
from . import c1_stage1_execution as io
from . import c1_stage3_integrated as integrated

SCOPE = 'C1_H4_SAVED_XDF_PREFIX_DF_BUDGET_V1'
FAMILY = 'artifacts/research_c_c1_h4_xdf_prefix_budget'
RANKS = tuple(range(1, 10))
LIMITS = dict(saved_bundle_load=1, sector_context=1, input_kappa_scalar_identity=1,
              physical_reference_attachment=1, xdf_prefix_model=9, xdf_matvec=324,
              sector_error_norm=9, numerical_allowance_budget=9, prefix_selection_report=1)
MAX_CALLS = sum(LIMITS.values())
CONDITIONS = dict(molecule='linear H4', distance_angstrom=1.0, basis='STO-3G',
    Nalpha=2, Nbeta=2, sector_dimension=36, modes=8, number_representation=True,
    decomposition='saved OpenFermion1.6.1 eigh', order='original saved lambda/G prefix',
    ranks=list(RANKS), saved_anchor_rank=10, CA_Ha=.00159, DF_budget_Ha=.00019875,
    allowance='binary64_operation_scale_guard8_v1', component_atol=1e-10, rtol=0,
    sector_order='ascending Qiskit little-endian interleaved spin occupations',
    kappa_policy='retain saved original-g corrected one-body for every prefix')
FORBIDDEN = ('new_fit', 'new_decomposition', 'physical_regeneration', 'rank10_norm',
             'PF', 'native_build_compile', 'synthesis', 'shots', 'total_cost', 'retry', 'resume')


def preflight(root, path, input_path):
    """Only JSON, file bytes and package versions. Never decode any NPZ/QPY."""
    root = Path(root).resolve()
    c = io.read(path)
    if (c['scope'] != SCOPE or c['status'] != 'UNAPPROVED_SCIENCE_CONTRACT'
            or c['conditions'] != CONDITIONS or c['call_limits'] != LIMITS
            or c['maximum_logical_units'] != MAX_CALLS or any(c[k] for k in FORBIDDEN)):
        raise ValueError('PREFIX_CONTRACT_SCOPE')
    source = root / c['source_manifest']
    if io.sha(source) != c['source_manifest_sha256']:
        raise ValueError('SOURCE_MANIFEST')
    for group, label in ((io.read(source)['files'], 'SOURCE'), (c['saved_inputs'], 'SAVED_INPUT')):
        for f, want in group.items():
            p = (root / f).resolve()
            if not p.is_relative_to(root) or io.sha(p) != want:
                raise ValueError(label + ': ' + f)
    if (c['original_input']['sha256'] != io.P_H4_SHA256
            or io.sha(input_path) != c['original_input']['sha256']):
        raise ValueError('ORIGINAL_INPUT')
    if set(c['records']) != {'physical', 'stage1', 'provenance', 'stage3'} or not set(c['records'].values()) <= c['saved_inputs'].keys():
        raise ValueError('UNPINNED_RECORD')
    for name, expected in c['evaluation_packages'].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError('ENVIRONMENT: ' + name)
    for record in c.get('historical_sources', []):
        content = subprocess.check_output(['git', 'show', record['commit'] + ':' + record['path']], cwd=root)
        if hashlib.sha256(content).hexdigest() != record['sha256']:
            raise ValueError('HISTORICAL_SOURCE: ' + record['path'])
    if c['fresh_output'] != FAMILY + '/<execution-date>/run01':
        raise ValueError('OUTPUT_SCOPE')
    return c


def output_path(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    relative = output.relative_to(root / FAMILY)
    if (len(relative.parts) != 2 or relative.parts[1] != 'run01'
            or date.fromisoformat(relative.parts[0]).isoformat() != relative.parts[0]
            or output.exists()):
        raise ValueError('FRESH_OUTPUT_REQUIRED_NO_RETRY_RESUME')
    return output


def prefix_rank(rank):
    if type(rank) is not int or rank not in RANKS:
        raise ValueError('REGISTERED_PREFIX_1_TO_9_ONLY')


def context():
    import numpy as np
    df = stage1.legacy_df_module()
    sector = df.PhysicalSector.spin_sector(n_qubits=8, nelec_alpha=2, nelec_beta=2)
    indices = stage1.sector_indices(4, 2, 2)
    reverse = stage1.bit_reverse_indices(8)
    positions = {int(reverse[s]): i for i, s in enumerate(sector.basis_indices)}
    perm = np.array([positions[int(s)] for s in indices])
    if sector.dimension != 36 or len(set(perm)) != 36:
        raise ValueError('SECTOR_CONTEXT')
    return SimpleNamespace(df=df, sector=sector, indices=indices, perm=perm)


def load_bundle(root, c, input_path):
    import numpy as np
    root = Path(root)
    packet = io.load_input(input_path, evaluation=True)  # original H4 whitelist/schema
    records = {k: io.read(root / v) for k, v in c['records'].items() if k != 'physical'}
    with np.load(root / c['records']['physical'], allow_pickle=False) as a:
        if a.files != ['matrix']:
            raise ValueError('PHYSICAL_MEMBER_SCHEMA')
        physical = a['matrix']
    if (physical.shape != (36, 36) or physical.dtype != np.complex128
            or records['stage1']['status'] != 'STAGE1_REPRESENTATION_PASS'
            or records['provenance']['input_sha256'] != c['original_input']['sha256']):
        raise ValueError('SAVED_PHYSICAL_PROVENANCE')
    anchor = {}
    for arm in ('XDF', 'CDF'):
        row = next(r for r in records['stage1']['records'] if r['id'] == 'sector_DF_' + arm)
        anchor[arm] = dict(rank=10 if arm == 'XDF' else 8, sector_error_Ha=row['error'],
            allowance_Ha=row['allowance'], norm_plus_allowance_Ha=records['stage3']['df_error_bounds_ha'][arm],
            status='PASS', origin='saved Stage1 norm/allowance; saved Stage3 DF reserve', recomputed=False)
        stage1.finite_arrays(*[anchor[arm][k] for k in ('sector_error_Ha', 'allowance_Ha', 'norm_plus_allowance_Ha')])
        if any(anchor[arm][k] < 0 for k in ('sector_error_Ha', 'allowance_Ha', 'norm_plus_allowance_Ha')) or anchor[arm]['norm_plus_allowance_Ha'] > CONDITIONS['DF_budget_Ha']:
            raise ValueError('SAVED_ANCHOR_BUDGET')
    for a in (*packet.values(), physical):
        a.setflags(write=False)
    return dict(packet=packet, physical=physical, anchors=anchor,
                basis_order=c['conditions']['sector_order'])


def hermitian(matrix, label):
    import numpy as np
    stage1.finite_arrays(matrix)
    error = float(np.max(abs(matrix - matrix.conj().T)))
    if error > CONDITIONS['component_atol']:
        raise ValueError('HERMITICITY: ' + label)
    return error


def identity_gate(bundle, ctx):
    import numpy as np
    p = bundle['packet']
    if (bundle['basis_order'] != CONDITIONS['sector_order']
            or not np.array_equal(ctx.indices, stage1.sector_indices(4, 2, 2))):
        raise ValueError('BASIS_ORDER')
    reverse = stage1.bit_reverse_indices(8)
    positions = {int(reverse[s]): i for i, s in enumerate(ctx.sector.basis_indices)}
    if not np.array_equal(ctx.perm, [positions[int(s)] for s in ctx.indices]):
        raise ValueError('BASIS_PERMUTATION')
    stage1.finite_arrays(*p.values(), bundle['physical'])
    # Original-g correction is checked ONCE; each prefix uses saved df_one_body.
    expected = np.kron(stage1.corrected_kappa(p['one_mo'], p['eri_chemist']), np.eye(2))
    error = float(np.max(abs(p['df_one_body'] - expected)))
    if error > CONDITIONS['component_atol']:
        raise ValueError('ORIGINAL_G_KAPPA')
    hermitian(p['df_one_body'], 'saved one-body')
    for G in p['df_g_matrices']:
        hermitian(G, 'saved G')
    return dict(status='PASS', kappa_max_entry=error, scalar=float(p['nuclear_repulsion']),
                saved_scalar_used_once=True, basis_order=bundle['basis_order'])


def attach_reference(bundle):
    import numpy as np
    p = bundle['packet']
    residual = hermitian(bundle['physical'], 'saved physical')
    one, two = stage1.spin_lift_unthresholded(p['one_mo'], p['eri_chemist'])
    scale = abs(float(p['nuclear_repulsion'])) + float(np.sum(abs(one))) + float(np.sum(abs(two / 2)))
    stage1.finite_arrays(scale)
    return dict(scale=scale, hermiticity_max_entry=residual, physical_regenerated=False)


def prefix_model(bundle, ctx, rank):
    """Same model/backend as saved_xdf_sector; no rehermitization or reordering."""
    prefix_rank(rank)
    p = bundle['packet']
    model = ctx.df.DFHamiltonian(float(p['nuclear_repulsion']), p['df_one_body'],
        p['df_lambdas'][:rank], tuple(p['df_g_matrices'][:rank]), {})
    op, counter = ctx.df.df_linear_operator(model, ctx.sector, backend='python')
    return op, counter


def prefix_matrix(model, ctx, rank, call):
    import numpy as np
    from scipy.sparse.linalg import LinearOperator
    op, counter = model
    column = 0
    def action(vector):
        nonlocal column
        index = column
        column += 1
        def evaluate():
            value = op @ vector
            stage1.finite_arrays(value)
            return value
        return call('xdf_matvec', f'r{rank}:basis{index}', evaluate)
    wrapped = LinearOperator(op.shape, matvec=action, dtype=np.complex128)
    matrix = ctx.df._linear_operator_to_dense(wrapped)
    if column != 36 or counter['count'] != 36:
        raise ValueError('XDF_MATVEC_INVENTORY')
    return matrix[np.ix_(ctx.perm, ctx.perm)]


def error_norm(delta):
    from scipy.linalg import svd
    stage1.finite_arrays(delta)
    value = float(svd(delta, compute_uv=False, lapack_driver='gesdd')[0])
    stage1.finite_arrays(value)
    return value


def assess_prefix(bundle, reference, rank, matrix, error):
    import numpy as np
    prefix_rank(rank)
    h = hermitian(matrix, f'prefix{rank}')
    p = bundle['packet']
    scale = abs(float(p['nuclear_repulsion'])) + float(np.sum(abs(p['df_one_body'])))
    scale += sum(abs(float(l)) * float(np.sum(abs(G))) ** 2
                 for l, G in zip(p['df_lambdas'][:rank], p['df_g_matrices'][:rank], strict=True))
    lengths = stage1.operation_lengths(4, 36, rank)
    guard = stage1.allowance(reference['scale'], scale, lengths['physical'], lengths['xdf'],
                             36, bundle['physical'] - matrix)
    stage1.finite_arrays(error, guard)
    if error < 0 or guard < 0:
        raise ValueError('NEGATIVE_ERROR_OR_ALLOWANCE')
    upper = error + guard
    stage1.finite_arrays(upper)
    return dict(rank=rank, sector_error_Ha=error, allowance_Ha=guard, norm_plus_allowance_Ha=upper,
        DF_budget_Ha=CONDITIONS['DF_budget_Ha'], status='PASS' if upper <= CONDITIONS['DF_budget_Ha'] else 'NOT_MET',
        hermiticity_max_entry=h, xdf_scale=scale, operation_lengths=lengths,
        allowance_version=stage1.ALLOWANCE_VERSION, rigorous_enclosure=None)


def select_prefix(rows, anchors):
    if [r['rank'] for r in rows] != list(RANKS):
        raise ValueError('INCOMPLETE_REGISTERED_PREFIXES')
    passing = [r['rank'] for r in rows if r['status'] == 'PASS']
    return dict(status='H4_XDF_PREFIX_BUDGET_COMPLETE', prefixes=rows, saved_anchors=anchors,
        first_passing_saved_prefix=passing[0] if passing else 10,
        interpretation='first eligible in this saved order only; not cost-optimal or a minimum over all XDF decompositions',
        rank10_recomputed=False, CA_achieved=None, rigorous_enclosure=None, mandatory_stop=True)


def run(root, contract_path, input_path, output, *, backend=None, workers=None):
    """Future permitted science. Tests inject artificial data; CLI has no override."""
    from threadpoolctl import threadpool_limits
    root = Path(root).resolve()
    c = preflight(root, contract_path, input_path)
    output = output_path(root, output)
    output.mkdir(parents=True, exist_ok=False)
    call = integrated.Calls(output, LIMITS, MAX_CALLS)
    start, cpu = time.monotonic(), time.process_time()
    rows, worker_errors = {}, []
    plan = integrated.resource_plan(task_count=9, full_dimension=36)
    if workers is not None:  # synthetic tests only, never exposed by CLI
        plan['workers'] = workers
    b = backend or SimpleNamespace(load=load_bundle, context=context, identity=identity_gate,
        attach=attach_reference, model=prefix_model, matrix=prefix_matrix, norm=error_norm, assess=assess_prefix)
    try:
        io.save(output / 'contract.json', c)
        source = io.read(root / c['source_manifest'])
        io.save(output / 'SOURCE.json', dict(source, execution_commit=subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()))
        io.save(output / 'resources_plan.json', plan)
        bundle = call('saved_bundle_load', 'original packet/physical/anchors', lambda: b.load(root, c, input_path))
        ctx = call('sector_context', 'sector36 Qiskit/OF permutation', b.context)
        identity = call('input_kappa_scalar_identity', 'same input/kappa/scalar/order', lambda: b.identity(bundle, ctx))
        reference = call('physical_reference_attachment', 'saved matrix/formation scale', lambda: b.attach(bundle))
        io.save(output / 'identity.json', identity)
        io.save(output / 'reference.json', reference)
        def job(rank):
            model = call('xdf_prefix_model', f'r{rank}', lambda: b.model(bundle, ctx, rank))
            matrix = b.matrix(model, ctx, rank, call)
            error = call('sector_error_norm', f'r{rank}', lambda: b.norm(bundle['physical'] - matrix))
            row = call('numerical_allowance_budget', f'r{rank}', lambda: b.assess(bundle, reference, rank, matrix, error))
            io.save(output / f'prefix_{rank}.json', row)
            return row
        # All nine are registered. NOT_MET is a normal result, never early exit.
        with threadpool_limits(limits=1 if plan['workers'] > 1 else None):
            with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
                futures = {}
                for rank in RANKS:
                    if call.stop.is_set():
                        break
                    futures[pool.submit(job, rank)] = rank
                for f in as_completed(futures):
                    try:
                        row = f.result()
                        rows[row['rank']] = row
                    except BaseException as error:
                        worker_errors.append(dict(rank=futures[f], error_type=type(error).__name__, error=str(error)))
                        call.stop.set()  # drain started work; no new dependent calls
        if call.stop.is_set():
            raise ValueError('PREFIX_CORRECTNESS_STOP: ' + str(call.first_failure))
        result = call('prefix_selection_report', 'all9 plus saved rank10',
            lambda: select_prefix([rows[r] for r in RANKS], bundle['anchors']))
        io.save(output / 'result.json', dict(result, call_counts=dict(call.counts), scientific_units=sum(call.counts.values())))
        return result
    except BaseException as error:
        call.stop.set()
        io.save(output / 'failure.json', dict(error=str(error), first_failure=call.first_failure,
                                             worker_errors=worker_errors))
        raise
    finally:
        io.save(output / 'partial_prefixes.json', [rows[r] for r in sorted(rows)])
        io.save(output / 'STOP.json', dict(mandatory_stop=True, failed=call.stop.is_set(), retry=0, resume=0,
            call_counts=dict(call.counts), scientific_units=sum(call.counts.values()),
            new_fit=0, new_decomposition=0, rank10_norm=0, physical_regeneration=0))
        io.save(output / 'resources.json', dict(wall_seconds=time.monotonic()-start,
            process_CPU_seconds=time.process_time()-cpu, peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
