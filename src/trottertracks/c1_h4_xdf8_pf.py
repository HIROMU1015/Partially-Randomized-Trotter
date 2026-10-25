"""XDF8 sector representation/ground/PF only; passive until separate permission.

No fit, DF-error reevaluation, CDF/XDF10 numerical evaluation or circuit work.
Complete spin-preserving factors are restricted; individual gates/densities
with spin mixing are never projected into the sector.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.metadata
import math
from pathlib import Path
import resource
import subprocess
import time
from types import SimpleNamespace

from . import c1_h4_xdf_prefix_budget as prefix
from . import c1_molecule_extension as molecule
from . import c1_general_k_adapter as general
from .c1_stage3_molecular import MolecularBackend, component_check

io, stage1, integrated = prefix.io, prefix.stage1, prefix.integrated
SCOPE = 'C1_H4_SAVED_XDF8_REPRESENTATION_GROUND_PF_V1'
FAMILY = 'artifacts/research_c_c1_h4_xdf8_pf'
CONDITIONS = dict(atoms=4, distance_angstrom=1., basis='STO-3G', Nalpha=2, Nbeta=2,
    sector_dimension=36, rank=8, source_order='saved OpenFermion1.6.1 eigh prefix8',
    number_representation=True, original_g_kappa=True, scalar='original nuclear once',
    product_formula='all original frames ordered symmetric S2', substeps=[1, 4],
    base_time='1/10', rounds=list(range(13)), CA_Ha=.00159, DF_budget_Ha=.00019875,
    target='own XDF8 sector ground-state oracle', state_preparation='EXCLUDED', PR_split=None)
ALLOWANCES = dict(component=1e-10, ground_residual=1e-10, normalization=1e-12, rtol=0,
    alias_energy_window_Ha=[-10., 10.], beta_total=math.pi/6, beta_stat=math.pi/12,
    PF_operator_phase_room=math.sin(math.pi/12), spin_lift='exact; no dropped off-spin terms',
    qualification='binary64 numerical pilot; no rounding enclosure')
LIMITS = dict(saved_input_load=1, saved_DF_gate=1, sector_context=1, input_identity=1,
    xdf_prefix_model=1, xdf_matvec=36, raw_factor=9, K_model=1, representation_gate=9,
    sector_sum_gate=1, ground_eigenpair=1, ground_alias=1, sector_kernel_reference=2,
    PF_round_assessment=26, candidate_summary=2)
MAX_CALLS = sum(LIMITS.values())
FORBIDDEN = ('new_fit', 'new_decomposition', 'DF_norm', 'CDF_evaluation', 'XDF10_evaluation',
    'physical_regeneration', 'native_build_compile', 'synthesis', 'CA_evaluation', 'shots',
    'total_cost', 'retry', 'resume')


def preflight(root, path, input_path):
    root = Path(root).resolve(); c = io.read(path)
    if (c['scope'] != SCOPE or c['status'] != 'UNAPPROVED_SCIENCE_CONTRACT'
            or c['conditions'] != CONDITIONS or c['allowances'] != ALLOWANCES
            or c['call_limits'] != LIMITS or c['maximum_logical_units'] != MAX_CALLS
            or any(c[k] for k in FORBIDDEN)):
        raise ValueError('XDF8_PF_SCOPE')
    if io.sha(root/c['source_manifest']) != c['source_manifest_sha256']:
        raise ValueError('SOURCE_MANIFEST')
    for group, label in [(io.read(root/c['source_manifest'])['files'], 'SOURCE'), (c['saved_inputs'], 'SAVED_INPUT')]:
        for f, expected in group.items():
            p = (root/f).resolve()
            if not p.is_relative_to(root) or io.sha(p) != expected:
                raise ValueError(label + ': ' + f)
    if io.sha(input_path) != io.P_H4_SHA256 or c['original_input']['sha256'] != io.P_H4_SHA256:
        raise ValueError('ORIGINAL_INPUT')
    if set(c['records']) != {'DF_row', 'DF_STOP', 'DF_SOURCE'} or not set(c['records'].values()) <= c['saved_inputs'].keys():
        raise ValueError('UNPINNED_DF_RECORD')
    if c['fresh_output'] != FAMILY+'/<execution-date>/run01':
        raise ValueError('OUTPUT_SCOPE')
    for name, expected in c['evaluation_packages'].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError('ENVIRONMENT: '+name)
    return c


def output_path(root, output):
    # Same exclusive/date/family semantics as the consumed prefix pilot.
    from datetime import date
    root, output = Path(root).resolve(), Path(output).resolve()
    parts = output.relative_to(root/FAMILY).parts
    if len(parts) != 2 or parts[1] != 'run01' or date.fromisoformat(parts[0]).isoformat() != parts[0] or output.exists():
        raise ValueError('FRESH_OUTPUT_REQUIRED_NO_RETRY_RESUME')
    return output


def load_saved(root, c, input_path):
    packet = io.load_input(input_path, evaluation=True)
    records = {k: io.read(Path(root)/v) for k, v in c['records'].items()}
    for value in packet.values(): value.setflags(write=False)
    return dict(packet=packet, records=records, input_identity=c['original_input']['sha256'])


def saved_df_gate(bundle, c):
    r = bundle['records']['DF_row']; stop = bundle['records']['DF_STOP']; source = bundle['records']['DF_SOURCE']
    if (r['rank'] != 8 or r['status'] != 'PASS' or r['DF_budget_Ha'] != CONDITIONS['DF_budget_Ha']
            or r['allowance_version'] != stage1.ALLOWANCE_VERSION
            or stop['failed'] or not stop['mandatory_stop'] or stop['scientific_units'] != 356
            or source['execution_commit'] != c['lineage']['DF_evaluation_SOURCE']):
        raise ValueError('SAVED_XDF8_DF_IDENTITY')
    values = [r[k] for k in ('sector_error_Ha', 'allowance_Ha', 'norm_plus_allowance_Ha')]
    stage1.finite_arrays(*values)
    if any(x < 0 for x in values) or values[2] > CONDITIONS['DF_budget_Ha']:
        raise ValueError('SAVED_DF_BUDGET')
    return dict(status='PASS', rank=8, saved_norm_Ha=values[0], saved_allowance_Ha=values[1],
        saved_norm_plus_allowance_Ha=values[2], DF_budget_Ha=CONDITIONS['DF_budget_Ha'], recomputed=False)


def contexts():
    return SimpleNamespace(df=prefix.context(), car=molecule.SectorCAR(4))


def spatial_spin_lift(matrix):
    import numpy as np
    a = np.asarray(matrix); stage1.finite_arrays(a)
    if a.shape != (8, 8) or not np.array_equal(a, np.kron(a[::2, ::2], np.eye(2))):
        raise ValueError('EXACT_SPIN_LIFT_REQUIRED_NO_PROJECTION')
    prefix.hermitian(a, 'saved spin lift')
    return a[::2, ::2]


def identity(bundle, ctx):
    import numpy as np
    p = bundle['packet']; stage1.finite_arrays(*p.values())
    if (not np.array_equal(ctx.df.indices, stage1.sector_indices(4, 2, 2)) or ctx.car.d != 36
            or not np.array_equal(ctx.car.sector.basis_indices, ctx.df.sector.basis_indices)):
        raise ValueError('SECTOR_BASIS_ORDER')
    reverse = stage1.bit_reverse_indices(8)
    positions = {int(reverse[s]): i for i, s in enumerate(ctx.df.sector.basis_indices)}
    if not np.array_equal(ctx.df.perm, [positions[int(s)] for s in ctx.df.indices]):
        raise ValueError('SECTOR_PERMUTATION')
    kappa = spatial_spin_lift(p['df_one_body'])
    for G in p['df_g_matrices'][:8]: spatial_spin_lift(G)
    check = component_check(kappa, stage1.corrected_kappa(p['one_mo'], p['eri_chemist']),
        ALLOWANCES['component'], label='saved original-g kappa')
    return dict(status='PASS', kappa=check, scalar=float(p['nuclear_repulsion']),
        rank=8, frame_order=list(range(8)), exact_spin_preserving_complete_factors=True,
        sector_projection_of_intermediate_gates=False)


def raw_factor(bundle, ctx, index):
    import numpy as np
    p = bundle['packet']
    if index == 0:
        value = ctx.car.one(spatial_spin_lift(p['df_one_body'])) + float(p['nuclear_repulsion'])*np.eye(36)
    else:
        A = ctx.car.one(spatial_spin_lift(p['df_g_matrices'][index-1]))
        value = float(p['df_lambdas'][index-1]) * (A @ A)
    prefix.hermitian(value, f'raw factor{index}')
    # SectorCAR and saved matvec both use OF order; export all factors in the
    # existing H4 ascending Qiskit basis, with the same explicit permutation.
    return value[np.ix_(ctx.df.perm, ctx.df.perm)]


def k_model(bundle):
    import numpy as np
    from trotterlib.df_trotter.decompose import diag_hermitian
    p = bundle['packet']; Js, Us = [], []
    for lam, G in zip(p['df_lambdas'][:8], p['df_g_matrices'][:8], strict=True):
        U, eta = diag_hermitian(spatial_spin_lift(G), assume_hermitian=True)
        component_check(U.conj().T@U, np.eye(4), ALLOWANCES['component'], label='orbital orthogonality')
        # Existing number adapter uses 1/2 J_ij N_i N_j, hence J=2 lambda eta eta^T.
        Js.append(2*float(lam)*np.outer(eta, eta)); Us.append(U)
    tag = bundle['input_identity']+':saved-XDF-prefix8'
    return general.from_ffsim_number_arrays(one_body_tensor=spatial_spin_lift(p['df_one_body']),
        diag_coulomb_mats=Js, orbital_rotations=Us, constant=float(p['nuclear_repulsion']),
        z_representation=False, frame_ids=[f'F{i}' for i in range(8)], group_ids=[f'F{i}' for i in range(8)],
        source_identity=tag, input_identity=bundle['input_identity'],
        one_body_role='ffsim_corrected_kappa', enable_general_k=True)


def representation(ctx, model, raw, index):
    import numpy as np
    if [f.original_index for f in model.frames] != list(range(8)) or [f.frame_id for f in model.frames] != [f'F{i}' for i in range(8)]:
        raise ValueError('ORIGINAL_FRAME_ORDER')
    view = SimpleNamespace(spatial_one_body_tensor=model.spatial_one_body_tensor if index == 0 else np.zeros((4, 4), dtype=complex),
        constant=model.constant if index == 0 else 0., frames=() if index == 0 else (model.frames[index-1],))
    candidate = ctx.car.candidate(view)  # independent I/Z/ZZ coefficient expansion
    candidate = candidate[np.ix_(ctx.df.perm, ctx.df.perm)]
    return component_check(candidate, raw, ALLOWANCES['component'], label=f'general-K/raw factor{index}'), candidate


def ground_solve(H, output):
    # Reuse the H4 eigensolver/export entry; never call its CDF/XDF10 prepare().
    import numpy as np
    helper = SimpleNamespace(output=output, tol={'operator_component_abs': ALLOWANCES['component']})
    pair = MolecularBackend._ground(helper, H, 'XDF8')
    with (output/'spectrum_XDF8.npz').open('xb') as f:
        np.savez_compressed(f, energies=pair[0], eigenvectors=pair[1])
    return pair


def ground_gate(H, pair, indices):
    import numpy as np
    vals, vecs = pair; stage1.finite_arrays(vals, vecs)
    if vals.shape != (36,) or vecs.shape != (36, 36) or not np.array_equal(indices, stage1.sector_indices(4, 2, 2)):
        raise ValueError('GROUND_SECTOR_SCHEMA')
    psi = vecs[:, 0]; residual = float(np.linalg.norm(H@psi-vals[0]*psi))
    norm_error = float(abs(np.vdot(psi, psi)-1.))
    stage1.finite_arrays(residual, norm_error)
    if residual > ALLOWANCES['ground_residual'] or norm_error > ALLOWANCES['normalization']:
        raise ValueError('GROUND_RESIDUAL_OR_NORMALIZATION')
    lo, hi = ALLOWANCES['alias_energy_window_Ha']
    margin = math.pi-(max(abs(lo), abs(hi))*.1+ALLOWANCES['beta_total'])
    status = 'PASS' if lo <= vals[0] <= hi and margin > 0 else 'UNRESOLVED_ALIAS_WINDOW'
    return dict(status=status, energy=float(vals[0]), residual=residual, normalization_error=norm_error,
        sector_dimension=36, base_alias_margin=margin, alias_energy_window_Ha=[lo, hi],
        rule='H4 registered [-10,10] energy window and max(abs(window))*0.1+pi/6 < pi',
        state_model=CONDITIONS['target'], physical_ground_overlap=None)


def summarize(rows):
    if [r['m'] for r in rows] != list(range(13)): raise ValueError('PF_ROUND_COVERAGE')
    unknown = [r['m'] for r in rows if r.get('pf_operator_error') is None]
    bad = [r['m'] for r in rows if r.get('pf_operator_error') is not None and r['pf_operator_error'] > ALLOWANCES['PF_operator_phase_room']]
    return dict(status='UNRESOLVED' if unknown else 'INFEASIBLE_REGISTERED_PHASE_ROOM' if bad else 'POTENTIALLY_FEASIBLE',
        unresolved_rounds=unknown, insufficient_rounds=bad, native_candidate_eligible=not unknown and not bad,
        interpretation='DF reserve and registered PF sufficient condition only; not final CA or physical impossibility',
        synthesis_error=None, CA_achieved=None, shots=None, total_cost=None)


def run(root, path, input_path, output, *, workers=None):
    import numpy as np
    from threadpoolctl import threadpool_limits
    root = Path(root).resolve(); c = preflight(root, path, input_path); output = output_path(root, output)
    output.mkdir(parents=True, exist_ok=False); (output/'cells').mkdir(); (output/'kernels').mkdir()
    call = integrated.Calls(output, LIMITS, MAX_CALLS); start, cpu = time.monotonic(), time.process_time()
    plan = integrated.resource_plan(task_count=2, full_dimension=36)
    if workers is not None: plan['workers'] = workers  # artificial tests only, no CLI option
    candidates, errors = {}, []
    try:
        io.save(output/'contract.json', c)
        io.save(output/'SOURCE.json', dict(io.read(root/c['source_manifest']), execution_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=root, text=True).strip()))
        io.save(output/'resource_plan.json', plan)
        bundle = call('saved_input_load', 'original packet and saved XDF8 DF proof', lambda: load_saved(root, c, input_path))
        df = call('saved_DF_gate', 'XDF8 only', lambda: saved_df_gate(bundle, c)); io.save(output/'saved_DF.json', df)
        ctx = call('sector_context', 'H4 sector36 contexts', contexts)
        io.save(output/'identity.json', call('input_identity', 'kappa/scalar/order/spin', lambda: identity(bundle, ctx)))
        with threadpool_limits(limits=1 if plan['workers'] > 1 else None):
            op = call('xdf_prefix_model', 'saved prefix8', lambda: prefix.prefix_model(bundle, ctx.df, 8))
            saved = prefix.prefix_matrix(op, ctx.df, 8, call)
            raw = [call('raw_factor', str(i), lambda i=i: raw_factor(bundle, ctx, i)) for i in range(9)]
            model = call('K_model', 'saved square to general-K', lambda: k_model(bundle))
            ks, gates = [], []
            for i, factor in enumerate(raw):
                gate, K = call('representation_gate', str(i), lambda i=i,factor=factor: representation(ctx, model, factor, i))
                ks.append(K); gates.append(gate)
            H = sum(raw)
            def sum_gate():
                prefix.hermitian(saved, 'saved XDF8 sector')
                return dict(raw=component_check(H, saved, ALLOWANCES['component'], label='raw sum/saved-XDF operator'),
                    general_K=component_check(sum(ks), saved, ALLOWANCES['component'], label='K sum/saved-XDF operator'))
            io.save(output/'representation.json', dict(factors=gates, total=call('sector_sum_gate', 'XDF8 target only', sum_gate)))
            with (output/'generators.npz').open('xb') as f: np.savez_compressed(f, generators=np.asarray(raw), sector_indices=ctx.df.indices)
            pair = call('ground_eigenpair', 'XDF8 own ground', lambda: ground_solve(H, output))
            ground = call('ground_alias', 'H4 registered window', lambda: ground_gate(H, pair, ctx.df.indices))
            io.save(output/'ground_alias.json', ground)
            if ground['status'] == 'PASS':
                def job(s):
                    def cell_call(kind, label, function):
                        def evaluate():
                            value = function()
                            if kind == 'sector_kernel_reference':
                                with (output/'kernels'/f'XDF8_s{s}.npz').open('xb') as f: np.savez_compressed(f, pf=value[0]['pf'])
                            if kind == 'PF_round_assessment':
                                if value['pf_operator_error'] is not None:
                                    stage1.finite_arrays(value['pf_operator_error'])
                                    if value['pf_operator_error'] < 0: raise ValueError('NEGATIVE_PF_ERROR')
                                value.update(target='XDF8', s=s, occurrences=s*(1<<value['m']),
                                    time=.1*(1<<value['m']), kernel_time=.1/s,
                                    sufficient_condition='UNRESOLVED' if value['pf_operator_error'] is None else
                                        'PASS' if value['pf_operator_error'] <= ALLOWANCES['PF_operator_phase_room'] else 'NOT_MET')
                                io.save(output/'cells'/f'XDF8_s{s}_m{value["m"]}.json', value)
                            return value
                        return call(kind, label, evaluate)
                    # Original repeated-squaring/eigenpair PF implementation, no new PF engine.
                    rows = molecule.PF_rounds(raw, s, cell_call, f'XDF8:s{s}', eigenpair=pair)
                    summary = call('candidate_summary', f'XDF8:s{s}', lambda: summarize(rows))
                    io.save(output/f'XDF8_s{s}.json', dict(rounds=rows, summary=summary))
                    return s, dict(rounds=rows, summary=summary)
                with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
                    futures = {pool.submit(job, s): s for s in (1, 4)}
                    for future in as_completed(futures):
                        try:
                            s, value = future.result(); candidates[str(s)] = value
                        except BaseException as e:
                            errors.append(dict(s=futures[future], error=str(e))); call.stop.set()
                if call.stop.is_set(): raise ValueError('XDF8_PF_CORRECTNESS_STOP')
        result = dict(status='XDF8_STAGE1_COMPLETE' if ground['status'] == 'PASS' else 'UNRESOLVED_GROUND_ALIAS',
            representation='PASS', saved_DF=df, ground=ground, candidates=candidates,
            CA_achieved=None, rigorous_enclosure=None, mandatory_stop=True)
        io.save(output/'result.json', result); return result
    except BaseException as e:
        call.stop.set(); io.save(output/'failure.json', dict(error=str(e), first_failure=call.first_failure, workers=errors)); raise
    finally:
        io.save(output/'partial_candidates.json', candidates)
        io.save(output/'STOP.json', dict(mandatory_stop=True, failed=call.stop.is_set(), retry=0, resume=0,
            scientific_units=sum(call.counts.values()), call_counts=dict(call.counts)))
        io.save(output/'resources.json', dict(wall_seconds=time.monotonic()-start, process_CPU_seconds=time.process_time()-cpu,
            peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
