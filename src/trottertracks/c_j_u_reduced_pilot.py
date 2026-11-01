"""Reduced-objective H4 proposal. Passive preflight never decodes arrays.

Reuse the previous parent/child snapshot, physical CAR/DF, native same-point,
resource planner, JSON, call and STOP pipeline. Only the optimization worker,
its named call budget and trial evidence differ; frozen source is untouched.
"""
from dataclasses import asdict
import importlib.metadata
from pathlib import Path
import subprocess
import time
from . import c_j_u_reduced as algorithm
from . import c_separated_j_u_consistent_pilot as prior
from .c_common_convergence_export_repair import save_json as save
common, io, previous, existing = prior.common, prior.io, prior.previous, prior.existing
molecule, ROLES = prior.molecule, prior.ROLES
BASE = 'e5a58753fa64124233522b92182e68f4a00a0461'
SCOPE = 'C_J_U_COUPLED_REDUCED_OBJECTIVE_PILOT_V1'
FAMILY = 'artifacts/research_c_j_u_reduced/2026-11-02'
FRESH = FAMILY+'/run01'
CONDITIONS = dict(prior.CONDITIONS, algorithm=algorithm.VERSION,
    initialization='same_original_qualified_J_U_x0_independent_arms')
PARENT = dict(prior.PARENT)
WORKER = dict(fit_input_load=1, objective_setup=1, initial_identity_gate=1,
    saved_history_load=1, outer_driver=1,
    inner_J_solve=2*algorithm.DEFAULT.max_points,
    reduced_state=algorithm.DEFAULT.max_points,
    envelope_gradient=algorithm.DEFAULT.max_points,
    terminal_independent=1, final_tensor=1, final_export=1)
LIMITS = {**PARENT, **{k:2*v for k,v in WORKER.items()}}
MAX_CALLS = sum(LIMITS.values())


def _record(output,kind,label,data,arrays):
    # Include all solver states and trial gradients, before any qualification stop.
    import json
    if arrays is not None:
        # Consistency metadata is emitted twice by Newton-CG but arrays once.
        existing.arrays(output/(kind+'_'+label+'.npz'),**arrays)
    with (output/'optimization_trace.jsonl').open('a') as f:
        f.write(json.dumps(prior.json_ready(dict(kind=kind,label=label,**data)),allow_nan=False)+'\n')


def preflight(root, path, *, fit=False):
    """Only bytes/metadata/version; actual numerical arrays remain closed."""
    root, path = Path(root).resolve(), Path(path)
    c = io.read(path)
    for name, wanted in dict(scope=SCOPE, base_commit=BASE, conditions=CONDITIONS,
            entry='scripts/run_research_c_j_u_reduced.py',
            algorithm_policy=asdict(algorithm.DEFAULT), call_limits=LIMITS,
            parent_limits=PARENT, per_arm_limits=WORKER,
            maximum_logical_units=MAX_CALLS, fresh_output=FRESH,
            retry=0, resume=0, mandatory_stop=True).items():
        if c[name] != wanted:
            raise ValueError('FIXED_REDUCED_PILOT:'+name)
    if io.sha(root/c['source_manifest']) != c['source_manifest_sha256']:
        raise ValueError('SOURCE_MANIFEST')
    for group in (io.read(root/c['source_manifest'])['files'], c['saved_inputs']):
        for name, digest in group.items():
            target = (root/name).resolve()
            if not target.is_relative_to(root) or io.sha(target) != digest:
                raise ValueError('SOURCE_OR_INPUT:'+name)
    required = {c[k] for k in ('input_snapshot', 'initial_snapshot',
                               'initial_scales', 'initial_sector')}
    if c['entry'] not in io.read(root/c['source_manifest'])['files']:
        raise ValueError('UNPINNED_ENTRY')
    required.update(v['path'] for v in c.get('dependency_locks',{}).values())
    for role in ROLES:
        required.update(c['saved_histories'][role].values())
    if not required <= c['saved_inputs'].keys():
        raise ValueError('UNPINNED_SEPARATED_INPUT')
    for name, version in c['fit_packages' if fit else 'evaluation_packages'].items():
        if importlib.metadata.version(name) != version:
            raise ValueError('ENVIRONMENT:'+name)
    if fit:
        dist = importlib.metadata.distribution('ffsim')
        for name, digest in c['ffsim_sources'].items():
            if io.sha(dist.locate_file('ffsim/'+name)) != digest:
                raise ValueError('FFSIM_SOURCE:'+name)
    return c

def fit_worker(root, output, c, role):
    import numpy as np
    root, output = Path(root), Path(output)
    output.mkdir()
    calls = common.old.Calls(output, WORKER, sum(WORKER.values()))
    started = time.monotonic()
    try:
        if io.read(output.parent/'initial_gate.json')['status'] != 'PASS':
            raise ValueError('SAVED_INITIAL_QUALIFICATION_REQUIRED')
        packet, a, info = calls('fit_input_load', role, lambda: previous.load(root, c))
        def setup():
            module, spec = common.setup(role, info)
            if not np.isclose(algorithm.rc2.full_l2(a['J'], np),
                    CONDITIONS['l2_scale'], atol=0., rtol=1e-12):
                raise ValueError('SAVED_INITIAL_P0_CHANGED')
            save(output/'objective.json', dict(role=role, spec=asdict(spec),
                policy=asdict(algorithm.DEFAULT), original_source=c['initialization_SOURCE'],
                tail_tensor_constraint=False, frame_reordering=False))
            return module, spec
        module, spec = calls('objective_setup', role, setup)
        def initial():
            common.saved.validate_point(dict(J=a['J'], U=a['U'], x=a['x0']), 8, 4,
                                        rotation_atol=1e-10)
            value = module.components(packet['g'], a['x0'], spec)
            if abs(value['normalized_tensor_objective']-1.) > 1e-8:
                raise ValueError('SAVED_INITIAL_LOSS_SCALE_CHANGED')
            if abs(value['normalized_penalty']-1.) > 1e-8:
                raise ValueError('SAVED_INITIAL_PENALTY_SCALE_CHANGED')
            return dict(status='PASS', source_snapshot=c['initial_snapshot'],
                sha256=c['saved_inputs'][c['initial_snapshot']], components=value,
                same_original_J_U_x0=True, legacy_endpoint_warm_start=False)
        save(output/'initial_identity.json', calls('initial_identity_gate', role, initial))
        def history():
            rows = {name:io.read(root/path) for name,path in c['saved_histories'][role].items()}
            save(output/'old_same_metric.json', rows['old_joint_metric'])
            return dict(status='SAVED_SCALARS_REUSED_NO_ARRAY_DECODE_OR_NEW_METRIC',
                        paths=c['saved_histories'][role])
        save(output/'saved_history.json',calls('saved_history_load',role,history))
        J, U, result = algorithm.optimize(packet['g'], a['J'], a['U'], spec,
            call=calls, record=lambda kind, label, data, arrays:
                _record(output, kind, label, data, arrays))
        save(output/'optimization.json', result)
        def independent():
            x, gradient, diagnostic = prior.native_diagnostic(packet['g'], J, U,
                module, spec, result['terminal'], algorithm.DEFAULT,
                record=lambda data:save(output/'terminal_consistency.json',data))
            existing.arrays(output/'terminal_native_parameters.npz', x=x, jac=gradient)
            save(output/'final_diagnostic.json', diagnostic)
            return x, gradient
        x, gradient = calls('terminal_independent', role, independent)
        save(output/'tensor.json', calls('final_tensor', role,
            lambda: common.old.tensor_metrics(packet['g'], J, U)))
        def export():
            existing.arrays(output/'final.npz', J=J, U=U, x=x, jac=gradient)
            return dict(status='EXPORTED_FOR_INDEPENDENT_DF_ASSESSMENT',
                sha256=io.sha(output/'final.npz'), role=role,
                first_order_qualified=result['first_order_qualified'],
                native_x_jac='diagnostic codec/native AD only; block raw iterates stored separately')
        save(output/'candidate.json', calls('final_export', role, export))
    except BaseException as error:
        calls.stop.set()
        save(output/'failure.json', dict(error=str(error), first_failure=calls.first_failure))
        raise
    finally:
        existing.finish(output, calls, started)

def decide(rows,reports,references):
    result=prior.decide(rows,reports,references)
    result['reduced_gradient_qualification']={k:reports[k]['optimization']['qualification'] for k in ROLES}
    result['scientific_guarantee']='NUMERICAL_PILOT_ONLY'
    result['repaired_separated_comparison']={}
    for role in ROLES:
        if 'repaired_optimization' not in references.get(role,{}):continue
        old=references[role]['repaired_optimization']
        new=reports[role]['optimization']
        result['repaired_separated_comparison'][role]=dict(
            saved_old_status=old['status'],new_status=new['status'],
            old_terminal=old['terminal'],new_terminal=new['terminal'],
            objective_difference_new_minus_saved=new['terminal']['components']['objective']-old['terminal']['components']['objective'],
            old_arrays_redecoded=False,matched_classical_effort=False)
    return result


def run(root, path, output, fit_python):
    root, output = Path(root).resolve(), Path(output).resolve()
    c = preflight(root, path)
    if output != root/FRESH or output.exists():
        raise ValueError('FRESH_REDUCED_RUN_NO_RETRY_RESUME')
    output.mkdir(parents=True)
    calls = common.old.Calls(output, PARENT, sum(PARENT.values()))
    started = time.monotonic()
    try:
        save(output/'contract.json', c)
        save(output/'SOURCE.json', dict(io.read(root/c['source_manifest']),
            execution_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                                       cwd=root, text=True).strip()))
        plan = common.old.resource_plan(task_count=2, full_dimension=36)
        # Reuse the previous observed AD-worker peak estimate, not a quota.
        estimate = 2*387852*1024
        if plan['available_memory_bytes'] is not None:
            plan['workers'] = min(plan['workers'], max(1, plan['available_memory_bytes']//estimate))
        plan['estimated_fit_bytes_per_worker'] = estimate
        save(output/'resource_plan.json', plan)
        packet, _, _ = calls('saved_snapshots_load', 'same original x0',
                              lambda: previous.load(root, c))
        def initial():
            row = io.read(root/c['initial_sector'])
            if row['sector_error']+row['numerical_guard'] > CONDITIONS['DF_budget_Ha']:
                raise ValueError('SAVED_INITIAL_DF_BUDGET')
            return dict(status='PASS', sector=row, reuse='same saved qualified snapshot bytes')
        save(output/'initial_gate.json', calls('initial_qualification_gate', 'saved PASS', initial))
        ctx = calls('sector_context', 'sector36', lambda: molecule.SectorCAR(4))
        physical = calls('physical_sector', 'original g/kappa/scalar',
            lambda: ctx.physical(packet['h'], packet['g'], packet['constant']))
        common.launch_workers(root, path, output, c, fit_python, plan['workers'], calls)
        rows, reports = {}, {}
        for role in ROLES:
            def load(k=role):
                data = existing.old_input(output/k/'final.npz')
                report = {name:io.read(output/k/file) for name, file in dict(
                    optimization='optimization.json', diagnostic='final_diagnostic.json',
                    old_metric='old_same_metric.json', tensor='tensor.json', candidate='candidate.json').items()}
                return data, report
            final, reports[role] = calls('final_snapshot_load', role, load)
            row, G = calls('sector_assessment', role, lambda: molecule.rank_assessment(
                ctx, packet, final['J'], final['U'], physical, 8))
            rep = calls('representation_assessment', role, lambda: molecule.representation_gate(
                ctx, packet, final['J'], final['U'], G))
            rows[role] = dict(row, representation=rep)
            save(output/(role+'_sector.json'), rows[role])
        def compare():
            references = {k:{name:io.read(root/p) for name, p in paths.items()}
                          for k, paths in c['saved_histories'].items()}
            result=decide(rows,reports,references)
            result['classifications']={k:reports[k]['optimization']['classification'] for k in ROLES}
            result['within_same_objective_old_separated']={k:dict(
                old=references[k]['old_separated_optimization']['terminal'],
                new=reports[k]['optimization']['terminal'],
                old_status=references[k]['old_separated_optimization']['status'],
                new_status=reports[k]['optimization']['status']) for k in ROLES}
            return result
        result = calls('comparison', 'same-metric convergence, separate DF and lambda', compare)
        save(output/'result.json', result)
        return result
    except BaseException as error:
        calls.stop.set()
        save(output/'failure.json', dict(error=str(error), first_failure=calls.first_failure))
        raise
    finally:
        existing.finish(output, calls, started, ROLES)
