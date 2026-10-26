"""Saved H4 XDF9-only representation/ground/PF proposal; no molecular work on import.

Reuses the consumed XDF8 route's input, CAR, ground/alias and PF functions.
Only rank-dependent assembly and the isolated orchestration are new. No old
SOURCE, constants or scientific output is modified.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.metadata
import resource
import subprocess
import time
from types import SimpleNamespace
from pathlib import Path

from . import c1_h4_xdf8_pf as previous

io, stage1, integrated = previous.io, previous.stage1, previous.integrated
prefix, molecule, general = previous.prefix, previous.molecule, previous.general
component_check, MolecularBackend = previous.component_check, previous.MolecularBackend
SCOPE = 'C1_H4_SAVED_XDF9_REPRESENTATION_GROUND_PF_V1'
FAMILY = 'artifacts/research_c_c1_h4_xdf9_pf'
CONDITIONS = dict(previous.CONDITIONS, rank=9,
    source_order='saved OpenFermion1.6.1 eigh prefix9',
    target='own XDF9 sector ground-state oracle')
ALLOWANCES = dict(previous.ALLOWANCES)
LIMITS = dict(previous.LIMITS, raw_factor=10, representation_gate=10)
MAX_CALLS = sum(LIMITS.values())
FORBIDDEN = previous.FORBIDDEN + ('XDF8_evaluation',)
load_saved, contexts = previous.load_saved, previous.contexts
spatial_spin_lift, raw_factor = previous.spatial_spin_lift, previous.raw_factor
summarize = previous.summarize

def preflight(root, path, input_path):
    root = Path(root).resolve(); c = io.read(path)
    if (c['scope'] != SCOPE or c['status'] != 'UNAPPROVED_SCIENCE_CONTRACT'
            or c['conditions'] != CONDITIONS or c['allowances'] != ALLOWANCES
            or c['call_limits'] != LIMITS or c['maximum_logical_units'] != MAX_CALLS
            or any(c[k] for k in FORBIDDEN)):
        raise ValueError('XDF9_PF_SCOPE')
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


def saved_df_gate(bundle, c):
    r = bundle['records']['DF_row']; stop = bundle['records']['DF_STOP']; source = bundle['records']['DF_SOURCE']
    if (r['rank'] != 9 or r['status'] != 'PASS' or r['DF_budget_Ha'] != CONDITIONS['DF_budget_Ha']
            or r['allowance_version'] != stage1.ALLOWANCE_VERSION
            or stop['failed'] or not stop['mandatory_stop'] or stop['scientific_units'] != 356
            or source['execution_commit'] != c['lineage']['DF_evaluation_SOURCE']):
        raise ValueError('SAVED_XDF9_DF_IDENTITY')
    values = [r[k] for k in ('sector_error_Ha', 'allowance_Ha', 'norm_plus_allowance_Ha')]
    stage1.finite_arrays(*values)
    if any(x < 0 for x in values) or values[2] > CONDITIONS['DF_budget_Ha']:
        raise ValueError('SAVED_DF_BUDGET')
    return dict(status='PASS', rank=9, saved_norm_Ha=values[0], saved_allowance_Ha=values[1],
        saved_norm_plus_allowance_Ha=values[2], DF_budget_Ha=CONDITIONS['DF_budget_Ha'], recomputed=False)


def identity(bundle, ctx):
    result = previous.identity(bundle, ctx)
    # The previous identity covers base, sector/permutation and first 8 factors;
    # explicitly require the ninth complete factor to preserve spin as well.
    spatial_spin_lift(bundle['packet']['df_g_matrices'][8])
    return dict(result, rank=9, frame_order=list(range(9)))


def k_model(bundle):
    import numpy as np
    from trotterlib.df_trotter.decompose import diag_hermitian
    p = bundle['packet']; Js, Us = [], []
    for lam, G in zip(p['df_lambdas'][:9], p['df_g_matrices'][:9], strict=True):
        U, eta = diag_hermitian(spatial_spin_lift(G), assume_hermitian=True)
        component_check(U.conj().T@U, np.eye(4), ALLOWANCES['component'], label='orbital orthogonality')
        # Existing number adapter uses 1/2 J_ij N_i N_j, hence J=2 lambda eta eta^T.
        Js.append(2*float(lam)*np.outer(eta, eta)); Us.append(U)
    tag = bundle['input_identity']+':saved-XDF-prefix9'
    return general.from_ffsim_number_arrays(one_body_tensor=spatial_spin_lift(p['df_one_body']),
        diag_coulomb_mats=Js, orbital_rotations=Us, constant=float(p['nuclear_repulsion']),
        z_representation=False, frame_ids=[f'F{i}' for i in range(9)], group_ids=[f'F{i}' for i in range(9)],
        source_identity=tag, input_identity=bundle['input_identity'],
        one_body_role='ffsim_corrected_kappa', enable_general_k=True)


def representation(ctx, model, raw, index):
    import numpy as np
    if [f.original_index for f in model.frames] != list(range(9)) or [f.frame_id for f in model.frames] != [f'F{i}' for i in range(9)]:
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
    pair = MolecularBackend._ground(helper, H, 'XDF9')
    with (output/'spectrum_XDF9.npz').open('xb') as f:
        np.savez_compressed(f, energies=pair[0], eigenvectors=pair[1])
    return pair


def ground_gate(H, pair, indices):
    return dict(previous.ground_gate(H, pair, indices), state_model=CONDITIONS['target'])


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
        bundle = call('saved_input_load', 'original packet and saved XDF9 DF proof', lambda: load_saved(root, c, input_path))
        df = call('saved_DF_gate', 'XDF9 only', lambda: saved_df_gate(bundle, c)); io.save(output/'saved_DF.json', df)
        ctx = call('sector_context', 'H4 sector36 contexts', contexts)
        io.save(output/'identity.json', call('input_identity', 'kappa/scalar/order/spin', lambda: identity(bundle, ctx)))
        with threadpool_limits(limits=1 if plan['workers'] > 1 else None):
            op = call('xdf_prefix_model', 'saved prefix9', lambda: prefix.prefix_model(bundle, ctx.df, 9))
            saved = prefix.prefix_matrix(op, ctx.df, 9, call)
            raw = [call('raw_factor', str(i), lambda i=i: raw_factor(bundle, ctx, i)) for i in range(10)]
            model = call('K_model', 'saved square to general-K', lambda: k_model(bundle))
            ks, gates = [], []
            for i, factor in enumerate(raw):
                gate, K = call('representation_gate', str(i), lambda i=i,factor=factor: representation(ctx, model, factor, i))
                ks.append(K); gates.append(gate)
            H = sum(raw)
            def sum_gate():
                prefix.hermitian(saved, 'saved XDF9 sector')
                return dict(raw=component_check(H, saved, ALLOWANCES['component'], label='raw sum/saved-XDF operator'),
                    general_K=component_check(sum(ks), saved, ALLOWANCES['component'], label='K sum/saved-XDF operator'))
            io.save(output/'representation.json', dict(factors=gates, total=call('sector_sum_gate', 'XDF9 target only', sum_gate)))
            with (output/'generators.npz').open('xb') as f: np.savez_compressed(f, generators=np.asarray(raw), sector_indices=ctx.df.indices)
            pair = call('ground_eigenpair', 'XDF9 own ground', lambda: ground_solve(H, output))
            ground = call('ground_alias', 'H4 registered window', lambda: ground_gate(H, pair, ctx.df.indices))
            io.save(output/'ground_alias.json', ground)
            if ground['status'] == 'PASS':
                def job(s):
                    def cell_call(kind, label, function):
                        def evaluate():
                            value = function()
                            if kind == 'sector_kernel_reference':
                                with (output/'kernels'/f'XDF9_s{s}.npz').open('xb') as f: np.savez_compressed(f, pf=value[0]['pf'])
                            if kind == 'PF_round_assessment':
                                if value['pf_operator_error'] is not None:
                                    stage1.finite_arrays(value['pf_operator_error'])
                                    if value['pf_operator_error'] < 0: raise ValueError('NEGATIVE_PF_ERROR')
                                value.update(target='XDF9', s=s, occurrences=s*(1<<value['m']),
                                    time=.1*(1<<value['m']), kernel_time=.1/s,
                                    sufficient_condition='UNRESOLVED' if value['pf_operator_error'] is None else
                                        'PASS' if value['pf_operator_error'] <= ALLOWANCES['PF_operator_phase_room'] else 'NOT_MET')
                                io.save(output/'cells'/f'XDF9_s{s}_m{value["m"]}.json', value)
                            return value
                        return call(kind, label, evaluate)
                    # Original repeated-squaring/eigenpair PF implementation, no new PF engine.
                    rows = molecule.PF_rounds(raw, s, cell_call, f'XDF9:s{s}', eigenpair=pair)
                    summary = call('candidate_summary', f'XDF9:s{s}', lambda: summarize(rows))
                    io.save(output/f'XDF9_s{s}.json', dict(rounds=rows, summary=summary))
                    return s, dict(rounds=rows, summary=summary)
                with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
                    futures = {pool.submit(job, s): s for s in (1, 4)}
                    for future in as_completed(futures):
                        try:
                            s, value = future.result(); candidates[str(s)] = value
                        except BaseException as e:
                            errors.append(dict(s=futures[future], error=str(e))); call.stop.set()
                if call.stop.is_set(): raise ValueError('XDF9_PF_CORRECTNESS_STOP')
        result = dict(status='XDF9_STAGE1_COMPLETE' if ground['status'] == 'PASS' else 'UNRESOLVED_GROUND_ALIAS',
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
