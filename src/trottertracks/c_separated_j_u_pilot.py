"""Dormant paired separated-J/U pilot; two task-local numerical environments.

Reuse original qualified snapshots, scalar scales, process launcher, call
journal, sector physics and independent representation gates. No science at
import or passive preflight; no legacy fitter/run/STOP is restarted.
"""
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time
from . import c_separated_j_u as algorithm
from . import c_common_convergence_pilot as common
from .c_common_convergence_export_repair import save_json as save, json_ready

io, previous, existing = common.io, common.previous, common.existing
molecule, ROLES = common.molecule, common.ROLES
BASE = '77b021802e0edbc330fae0387aa525637557f224'
SCOPE = 'C_SEPARATED_J_U_PR_RC2_H4_PILOT_V1'
FAMILY = 'artifacts/research_c_separated_J_U/2026-11-01'
FRESH = FAMILY+'/run01'
CONDITIONS = dict(common.CONDITIONS, algorithm=algorithm.VERSION,
    initialization='same_original_qualified_J_U_x0_independent_arms',
    coefficient_variables='all frames; signed symmetric; no tail preservation equality',
    ff_sim_fitter_invocations=0)
PARENT = dict(common.PARENT)
WORKER = dict(fit_input_load=1, objective_setup=1, initial_identity_gate=1,
    old_endpoint_load=1, old_metric_diagnostic=1,
    J_solve=algorithm.DEFAULT.max_cycles+1, U_solve=algorithm.DEFAULT.max_cycles,
    state_diagnostic=algorithm.DEFAULT.max_cycles+2,
    terminal_independent=1, final_tensor=1, final_export=1)
LIMITS = {**PARENT, **{k:2*v for k, v in WORKER.items()}}
MAX_CALLS = sum(LIMITS.values())


def preflight(root, path, *, fit=False):
    """Only bytes/metadata/version; actual numerical arrays remain closed."""
    root, path = Path(root).resolve(), Path(path)
    c = io.read(path)
    for name, wanted in dict(scope=SCOPE, base_commit=BASE, conditions=CONDITIONS,
            algorithm_policy=asdict(algorithm.DEFAULT), call_limits=LIMITS,
            parent_limits=PARENT, per_arm_limits=WORKER,
            maximum_logical_units=MAX_CALLS, fresh_output=FRESH,
            retry=0, resume=0, mandatory_stop=True).items():
        if c[name] != wanted:
            raise ValueError('FIXED_SEPARATED_PILOT:'+name)
    if io.sha(root/c['source_manifest']) != c['source_manifest_sha256']:
        raise ValueError('SOURCE_MANIFEST')
    for group in (io.read(root/c['source_manifest'])['files'], c['saved_inputs']):
        for name, digest in group.items():
            target = (root/name).resolve()
            if not target.is_relative_to(root) or io.sha(target) != digest:
                raise ValueError('SOURCE_OR_INPUT:'+name)
    required = {c[k] for k in ('input_snapshot', 'initial_snapshot',
                               'initial_scales', 'initial_sector')}
    for role in ROLES:
        required.add(c['old_endpoints'][role])
        required.update(c['comparators'][role].values())
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


def _record(output, kind, label, data, arrays):
    body = json.dumps(json_ready(dict(kind=kind, label=label, **data)), allow_nan=False)
    if kind in ('J_raw', 'U_raw', 'state'):
        existing.arrays(output/(kind+'_'+label+'.npz'), **arrays)
    with (output/'optimization_trace.jsonl').open('a') as f:
        f.write(body+'\n')


def native_diagnostic(g, J, U, module, spec, state, policy):
    """Existing codec and full ffsim objective; never replace the actual U."""
    import numpy as np
    _, _, x, codec, _ = algorithm.pr.corrected.codec.roundtrip(J, U)
    value, gradient = module.make_value_and_grad(spec)(x, g)
    value, gradient = float(value), np.asarray(gradient)
    p = algorithm.Coefficients(g, U, spec)
    v, gj = p.vg(p.pack(J), independent=True)
    split = spec.rank*spec.norb*(spec.norb-1)//2
    native_j = gradient[split:]/p.w
    # Small tensor reconstruction differences are amplified by fixed f*.
    # Separate this consistency allowance from first-order qualification.
    if (not np.isfinite(value) or not np.isfinite(gradient).all()
            or abs(value-v) > policy.value_atol+policy.value_rtol*abs(v)
            or not np.allclose(native_j, gj, atol=1e-6, rtol=1e-7)):
        raise ValueError('INDEPENDENT_FFSIM_OBJECTIVE_J_GRADIENT_MISMATCH')
    return x, gradient, dict(status='PASS', objective=value, codec=codec,
        J_gradient_agreement_maxabs=float(np.max(abs(native_j-gj))),
        J_gradient_agreement_atol=1e-6, J_gradient_agreement_rtol=1e-7,
        native_parameter_gradient_infinity=float(np.max(abs(gradient))),
        physical_first_order=state, native_x_is_diagnostic_not_one_global_optimizer=True,
        rigorous_stationarity=False)


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
        old = calls('old_endpoint_load', role,
            lambda: existing.old_input(root/c['old_endpoints'][role]))
        def old_metric():
            reference = io.read(root/c['comparators'][role]['diagnostic'])
            assessment, _, _ = algorithm.state(packet['g'], old['J'], old['U'], spec)
            before = reference['components']['combined_objective']
            after = assessment['components']['objective']
            if abs(before-after) > algorithm.DEFAULT.value_atol+algorithm.DEFAULT.value_rtol*abs(before):
                raise ValueError('OLD_SAME_OBJECTIVE_IDENTITY_MISMATCH')
            return dict(assessment, source='saved joint endpoint; fresh metric evaluation only',
                        old_endpoint_optimized_again=False)
        save(output/'old_same_metric.json', calls('old_metric_diagnostic', role, old_metric))
        J, U, result = algorithm.optimize(packet['g'], a['J'], a['U'], spec,
            call=calls, record=lambda kind, label, data, arrays:
                _record(output, kind, label, data, arrays))
        save(output/'optimization.json', result)
        def independent():
            x, gradient, diagnostic = native_diagnostic(packet['g'], J, U,
                module, spec, result['terminal'], algorithm.DEFAULT)
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


def decide(rows, reports, references):
    budget = {k:bool(rows[k]['sector_error']+rows[k]['numerical_guard'] <=
                    CONDITIONS['DF_budget_Ha']) for k in ROLES}
    qualified = {k:bool(reports[k]['optimization']['first_order_qualified']) for k in ROLES}
    status = ('DF_BUDGET_NOT_MET' if not all(budget.values()) else
              'ONE_OR_BOTH_FIRST_ORDER_UNQUALIFIED' if not all(qualified.values()) else
              'BOTH_FIRST_ORDER_AND_DF_MET')
    masses = {k:reports[k]['optimization']['terminal']['components']['exact_tail_mass_Ha']
              for k in ROLES}
    delta = masses['pr']-masses['rc2']
    relation = None
    if status == 'BOTH_FIRST_ORDER_AND_DF_MET':
        relation = ('UNRESOLVED_WITHIN_ENGINEERING_ALLOWANCE' if abs(delta) <=
                    CONDITIONS['mass_comparison_allowance_Ha'] else
                    'PR_LOWER_NUMERICAL_TAIL' if delta < 0 else 'RC2_LOWER_NUMERICAL_TAIL')
    within_objective = {}
    for k in ROLES:
        old = reports[k]['old_metric']
        new = reports[k]['optimization']['terminal']
        within_objective[k] = dict(old_first_order_met=old['first_order_met'],
            new_first_order_met=new['first_order_met'],
            objective_difference_new_minus_old=new['components']['objective']-old['components']['objective'],
            old=old, new=new)
    return dict(status=status, DF_budget=budget, first_order_qualified=qualified,
        exact_tail_mass_Ha=masses, descriptive_PR_minus_RC2_Ha=delta,
        qualified_tail_comparison=relation, within_same_objective=within_objective,
        cross_objective_raw_F_ranking=None, reports=reports, sectors=rows,
        saved_scalar_references=references, initial_policy='same original qualified x0; no old endpoint warm start',
        rigorous_stationarity=False, global_optimality=None, optimal_strength=None,
        PF_error=None, RTE_error=None, attenuation=None, shots=None, native_cost=None,
        total_cost=None, CA_achieved=None, mandatory_stop=True)


def run(root, path, output, fit_python):
    root, output = Path(root).resolve(), Path(output).resolve()
    c = preflight(root, path)
    if output != root/FRESH or output.exists():
        raise ValueError('FRESH_SEPARATED_RUN_NO_RETRY_RESUME')
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
                          for k, paths in c['comparators'].items()}
            return decide(rows, reports, references)
        result = calls('comparison', 'same-metric convergence, separate DF and lambda', compare)
        save(output/'result.json', result)
        return result
    except BaseException as error:
        calls.stop.set()
        save(output/'failure.json', dict(error=str(error), first_failure=calls.first_failure))
        raise
    finally:
        existing.finish(output, calls, started, ROLES)
