"""Saved, finite-budget RC2/PR endpoints: Stage B only, no fit or circuits.

Import/preflight are passive. Real saved arrays require separate chat permission.
Old SOURCE, runner, failed STOP and repaired result are never edited/restarted.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from fractions import Fraction
import importlib.metadata
import math
from pathlib import Path
import resource
import subprocess
import time

from . import c_pr_aware_df_stage_b as b
from . import c_pr_aware_df_stage_b_runner as old
from . import c1_stage3_feasibility as feasibility
from .c_common_convergence_export_repair import save_json as save

io = old.io
BASE = 'f6bb336b41cddface5647c6b1acf23c330f32637'
FIT_SOURCE = 'e2f7dfd44357510d54ab9c6c9c5032e0072ab376'
VERSION = 'saved_common_budget_rc2_pr_stage_b_v1'
SCOPE = 'C_RC2_PR_SAVED_STAGE_B_M4_POLICY_V1'
FAMILY = 'artifacts/research_c_rc2_pr_stage_b/2026-11-01'
FRESH = FAMILY + '/run01'
ROLES = ('rc2', 'pr')
API_ROLES = dict(rc2='baseline', pr='proposed')
POLICY = dict(old.POLICY, s=1, r=128, h='1/10')
ALLOWANCES = dict(old.ALLOWANCES)
LIMITS = dict(input_load=1, saved_candidate_gate=1, sector_context=1,
    physical_operator=1, fit_load=2, model_prepare=2, raw_factor=18,
    factor_representation=18, arm_sum=2, tail_law=2, tail_gate=2,
    mass_gate=2, law_distribution_gate=2, ground_eigenpair=3, ground_alias=3,
    kernel_reference=2, exact_evolution=26, round_reference=26,
    round_classification=26, arm_summary=2, comparison=1)
MAX_CALLS = sum(LIMITS.values())


def saved_candidates(root, c):
    """Validate existing JSON evidence; convergence is NOT an admission gate."""
    root = Path(root)
    result = io.read(root/c['records']['result'])
    recovery = io.read(root/c['records']['recovery'])
    stopped = io.read(root/c['records']['fit_STOP'])
    fit = io.read(root/c['records']['fit_contract'])
    post = io.read(root/c['records']['fit_postflight'])
    repaired_stop = io.read(root/c['records']['repair_STOP'])
    if (result['status'] != 'ONE_OR_BOTH_UNQUALIFIED'
            or result['qualified'] != dict(rc2=False, pr=False)
            or result['DF_budget'] != dict(rc2=True, pr=True)
            or recovery['status'] != 'JSON_EXPORT_RECOVERED'
            or recovery['new_fits'] != 0 or recovery['NPZ_decodes'] != 0
            or not stopped['failed'] or stopped['scientific_units'] != 25
            or stopped['retry'] or stopped['resume'] or not stopped['mandatory_stop']
            or post['scientific_SOURCE'] != FIT_SOURCE or post['COMPLETE'] != 25
            or post['call_FAIL'] != 0 or post['new_fits'] != 2
            or repaired_stop['failed'] or repaired_stop['scientific_units'] != 1
            or fit['conditions']['rank'] != 8 or fit['conditions']['ld'] != 2
            or fit['input_snapshot'] != c['input']):
        raise ValueError('SAVED_COMMON_CONVERGENCE_PROVENANCE')
    for role in ROLES:
        if c['exports'][role] != str(Path(fit['fresh_output'])/role/'final.npz'):
            raise ValueError('ROLE_EXPORT_PATH_BINDING')
        if post['raw_sha256'][c['exports'][role]] != c['saved_inputs'][c['exports'][role]]:
            raise ValueError('EXPORT_FIT_SOURCE_BINDING')
        row = result['sectors'][role]
        report = result['reports'][role]
        if (row['actual_rank'] != 8 or row['guard_policy'] != old.molecule.GUARD_POLICY
                or report['tensor']['actual_rank'] != 8
                or report['optimizer']['success'] is not False
                or report['diagnostic']['assessment']['convergence_criteria_met'] is not False):
            raise ValueError('SAVED_ROLE_RANK_OR_QUALIFICATION')
        # Same saved norm and engineering guard, not a new DF norm evaluation.
        bound = row['sector_error'] + row['numerical_guard']
        if not math.isfinite(bound) or not 0 <= bound <= POLICY['DF_reserve_Ha']:
            raise ValueError('SAVED_DF_BUDGET')
    return result


def preflight(root, contract_path):
    """Hashes, JSON, NPY headers, package metadata only; no array payloads."""
    root = Path(root).resolve()
    c = io.read(contract_path)
    expected = dict(scope=SCOPE, implementation=VERSION, base_commit=BASE,
        fit_source=FIT_SOURCE, policy=POLICY, allowances=ALLOWANCES,
        call_limits=LIMITS, maximum_logical_units=MAX_CALLS,
        fresh_output=FRESH, roles=list(ROLES), api_roles=API_ROLES, retry=0, resume=0)
    for key, want in expected.items():
        if c[key] != want: raise ValueError('FIXED_STAGE_B_CONDITIONS:'+key)
    if io.sha(root/c['source_manifest']) != c['source_manifest_sha256']:
        raise ValueError('SOURCE_MANIFEST')
    for group in (io.read(root/c['source_manifest'])['files'], c['saved_inputs']):
        for name, digest in group.items():
            p = (root/name).resolve()
            if not p.is_relative_to(root) or io.sha(p) != digest:
                raise ValueError('SOURCE_OR_INPUT:'+name)
    required = {c['input'], *c['exports'].values(), *c['records'].values()}
    if (set(c['exports']) != set(ROLES) or len(set(c['exports'].values())) != 2
            or not required <= c['saved_inputs'].keys()):
        raise ValueError('UNPINNED_OR_DUPLICATE_CANDIDATE')
    for role in ROLES:
        identity = c['identities'][role]
        if identity != dict(role=role, export=c['exports'][role],
                sha256=c['saved_inputs'][c['exports'][role]], fit_source=FIT_SOURCE,
                physical_input=c['saved_inputs'][c['input']]):
            raise ValueError('SCIENTIFIC_ARM_IDENTITY')
        old.export_headers(root/c['exports'][role])
    saved_candidates(root, c)
    for name, want in c['packages'].items():
        if importlib.metadata.version(name) != want: raise ValueError('ENVIRONMENT:'+name)
    if (root/FRESH).exists(): raise ValueError('FRESH_OUTPUT_NO_RETRY_RESUME')
    return c


def prepare_candidate(packet, data, *, identity):
    """Legacy role is only an API dispatch label; outputs carry new identity."""
    import numpy as np
    role = identity['role']
    if role not in ROLES: raise ValueError('SCIENTIFIC_ROLE')
    J, U = data['J'], data['U']
    if (J.shape != (8,4,4) or U.shape != J.shape
            or J.dtype != np.float64 or U.dtype != np.float64):
        raise ValueError('SAVED_CDF_RANK_SHAPE_DTYPE')
    b.stage1.finite_arrays(J, U)
    arm = b.prepare_arm(**packet, J=J, U=U, role=API_ROLES[role],
        source_identity=identity['sha256'], input_identity=identity['physical_input'], ld=2)
    model = replace(arm.model, label='SAVED_COMMON_BUDGET_'+role)
    return replace(arm, role=role, model=model, plan=replace(arm.plan, model=model))


def distribution_gate(tail, config, distribution, *, expected_steps=128):
    """Audit the regenerated K0 law, not an implemented PRNG/IID certificate."""
    if config is None or distribution is None: raise ValueError('REGISTERED_NONZERO_TAIL')
    tol = ALLOWANCES['normalization']
    if (config.rte_steps != expected_steps or config.evolution_time != .1
            or config.finite_taylor_order != 0 or distribution.orders != (0,)
            or distribution.order_probabilities != (1.,)
            or not math.isclose(config.lambda_r, tail.lambda_r, rel_tol=1e-12, abs_tol=1e-14)
            or not math.isclose(math.fsum(x.probability for x in tail.components), 1., abs_tol=tol, rel_tol=0)
            or not math.isclose(config.distribution_normalization,
                distribution.exact_finite_distribution, rel_tol=1e-12, abs_tol=1e-14)):
        raise ValueError('FINITE_RTE_LAW_NORMALIZATION')
    return dict(status='PASS',lambda_R=tail.lambda_r,tau=config.dimensionless_step_time,
        normalization=config.distribution_normalization,orders=distribution.orders,
        event_probabilities=[dict(id=x.component_id, probability=x.probability,
            sign=x.coefficient_sign) for x in tail.components],
        law='ideal_finite_RTE_IID',implemented_law_error=None)


def mass_gate(arm, tail, saved):
    mass = saved['reports'][arm.role]['diagnostic']['components']['exact_tail_mass_Ha']
    if not math.isclose(tail.lambda_r, mass, rel_tol=1e-12, abs_tol=1e-14):
        raise ValueError('SAVED_EXACT_MASS_LAW_MISMATCH')
    return dict(status='PASS',saved_exact_mass_Ha=mass,law_lambda_R=tail.lambda_r)


def round_reference(kernel, H, psi, m, exact):
    """Keep old PF/RTE norm definitions; never equate underflow with zero error."""
    import numpy as np
    elapsed = Fraction(1,10)*(1 << m)
    row = b.reference.reference_round(kernel, (H,elapsed), repetitions=1 << m,
        state=psi, exact_operator=exact)
    if row['attenuation'] == 0:
        # Existing reference intentionally omits unstable corrected powers at A=0.
        # PF is still measurable independently, not inferred from lambda.
        row['pf_operator_error'] = float(np.linalg.norm(
            np.linalg.matrix_power(kernel['pf'],1 << m)-exact,2))
    signal = row.pop('raw_signal',None)
    row['raw_signal_parts'] = None if signal is None else [float(signal.real),float(signal.imag)]
    row['evidence_kind'] = 'binary64_sector_operator_numerical_pilot'
    return row


def summarize_arm(role, rows, saved, target, identity):
    sector = saved['sectors'][role]
    bound = sector['sector_error'] + sector['numerical_guard']
    return dict(arm=role,identity=identity,rounds=rows,
        classification=feasibility.summarize([x['feasibility'] for x in rows],bound,target),
        saved_DF_bound_Ha=bound,DF_budget='PASS',candidate_validity='FIXED_FINITE_ENDPOINT',
        optimizer_qualification='UNQUALIFIED',optimizer=saved['reports'][role]['optimizer'],
        synthesis_bound=None,implemented_law_bound=None,shots=None,total_cost=None,
        CA_achieved=None,guaranteed_advantage=None)


def run(root, contract_path, output):
    import numpy as np
    from scipy.linalg import expm
    from threadpoolctl import threadpool_limits
    from .c1_stage3_accuracy import AccuracyTarget
    root = Path(root).resolve()
    c = preflight(root,contract_path)
    output = Path(output).resolve()
    if output != root/FRESH or output.exists(): raise ValueError('FRESH_OUTPUT_NO_RETRY_RESUME')
    output.mkdir(parents=True)
    call = old.integrated.Calls(output,LIMITS,MAX_CALLS)
    start = time.monotonic()
    try:
        save(output/'contract.json',c)
        save(output/'SOURCE.json',dict(io.read(root/c['source_manifest']),
            execution_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()))
        p = call('input_load','same saved h/g/constant',lambda:old.load(root/c['input'],('h','g','constant')))
        if p['h'].shape != (4,4) or p['g'].shape != (4,)*4 or p['constant'].shape != ():
            raise ValueError('SAVED_PHYSICAL_INPUT_SHAPE')
        b.stage1.finite_arrays(*p.values())
        saved = call('saved_candidate_gate','DF budget vs optimizer qualification',lambda:saved_candidates(root,c))
        ctx = call('sector_context','Nalpha=Nbeta=2/sector36',lambda:old.molecule.SectorCAR(4))
        physical = call('physical_operator','saved original Hamiltonian, no new DF norm',
            lambda:ctx.physical(p['h'],p['g'],float(p['constant'])))
        pair = call('ground_eigenpair','physical',lambda:old.ground(physical,output,'physical'))
        save(output/'physical_ground.json',call('ground_alias','physical',lambda:old.ground_gate(physical,pair)))
        target = AccuracyTarget(POLICY['CA_Ha'],1/8,.1,POLICY['beta_total'],
            POLICY['beta_stat'],POLICY['alpha_axis'],POLICY['alpha_total'])
        plan = old.integrated.resource_plan(task_count=26,full_dimension=36)
        save(output/'resource_plan.json',plan)
        prepared = {}
        # Complete both arms' correctness gates before launching any round tasks.
        for role in ROLES:
            data = call('fit_load',role,lambda role=role:old.load(root/c['exports'][role],('J','U')))
            arm = call('model_prepare',role,lambda:prepare_candidate(p,data,identity=c['identities'][role]))
            G, C = [], []
            for i in range(9):
                G.append(call('raw_factor',f'{role}:{i}',lambda i=i:
                    ctx.one(b.stage1.corrected_kappa(p['h'],p['g']))+float(p['constant'])*np.eye(ctx.d)
                    if i == 0 else ctx.raw_frame(data['J'][i-1],data['U'][i-1])))
                def check(i=i):
                    candidate = b.candidate_factor(ctx,arm,i)
                    b.component_check(G[i],G[i].conj().T,ALLOWANCES['component'],label='Hermitian factor')
                    report = b.component_check(candidate,G[i],ALLOWANCES['component'],label='general-K/raw CAR')
                    return candidate,report
                matrix, report = call('factor_representation',f'{role}:{i}',check)
                C.append(matrix)
                save(output/f'{role}_factor_{i}.json',report)
            def total():
                H = sum(G)
                return H,b.component_check(sum(C),H,ALLOWANCES['component'],label='arm sum general-K/raw')
            H, report = call('arm_sum',role,total)
            save(output/f'{role}_sum.json',report)
            tail, config, distribution = call('tail_law',role,lambda:b.configure(arm,Fraction(1,10),128,0))
            save(output/f'{role}_tail.json',call('tail_gate',role,lambda:b.tail_gate(ctx,arm,G,tail)))
            save(output/f'{role}_mass.json',call('mass_gate',role,lambda:mass_gate(arm,tail,saved)))
            save(output/f'{role}_law.json',call('law_distribution_gate',role,
                lambda:distribution_gate(tail,config,distribution)))
            pair = call('ground_eigenpair',role,lambda:old.ground(H,output,role))
            save(output/f'{role}_ground.json',call('ground_alias',role,lambda:old.ground_gate(H,pair)))
            kernel = call('kernel_reference',role,lambda:b.kernels(arm,G,Fraction(1,10),config))
            old.save_arrays(output/f'{role}_reference.npz',generators=np.array(G),H=H,
                pf=kernel['pf'],corrected=kernel['corrected'])
            prepared[role] = (H,pair[1][:,0],kernel)
        def task(role,m):
            H, psi, kernel = prepared[role]
            elapsed = Fraction(1,10)*(1 << m)
            exact = call('exact_evolution',f'{role}:m{m}',lambda:expm(-1j*float(elapsed)*H))
            row = call('round_reference',f'{role}:m{m}',lambda:round_reference(kernel,H,psi,m,exact))
            decision = call('round_classification',f'{role}:m{m}',lambda:feasibility.classify_round(row,target))
            value = dict(arm=role,m=m,time=str(elapsed),physical_time_Ha_inverse=float(elapsed),
                coherent_repetitions=1 << m,reference=row,feasibility=decision,
                phase_room_sine=math.sin(target.beta_total-target.beta_stat),
                synthesis_error=None,implemented_law_error=None,shots=None,total_cost=None,CA_achieved=None)
            save(output/f'{role}_m{m:02d}.json',value)
            return role,m,value
        rounds = {role:{} for role in ROLES}
        with threadpool_limits(limits=plan['BLAS_threads_per_worker']):
            with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
                futures = [pool.submit(task,role,m) for role in ROLES for m in POLICY['rounds']]
                try:
                    for future in as_completed(futures):
                        role,m,row = future.result()
                        rounds[role][m] = row
                except BaseException:
                    call.stop.set()
                    for future in futures: future.cancel()
                    raise  # Executor drains started work before failure/STOP export.
        results = {}
        for role in ROLES:
            results[role] = call('arm_summary',role,lambda role=role:summarize_arm(role,
                [rounds[role][m] for m in POLICY['rounds']],saved,target,c['identities'][role]))
            save(output/f'{role}_result.json',results[role])
        result = call('comparison','fixed endpoint comparison; no resource ranking',lambda:dict(
            status='SAVED_ENDPOINT_STAGE_B_COMPLETE',arms=results,
            matched_rounds=[dict(m=m,rc2=results['rc2']['rounds'][m],
                pr=results['pr']['rounds'][m]) for m in POLICY['rounds']],
            comparison_scope='same-budget unqualified endpoints, fixed M4 policy',
            synthesis_error=None,implemented_law_error=None,shots=None,total_resources=None,
            CA_achieved=None,guaranteed_advantage=None,new_fits=0,native_build=0,compile=0,mandatory_stop=True))
        save(output/'result.json',result)
        return result
    except BaseException as error:
        call.stop.set()
        save(output/'failure.json',dict(error=str(error),first_failure=call.first_failure))
        raise
    finally:
        save(output/'STOP.json',dict(failed=call.stop.is_set(),scientific_units=sum(call.counts.values()),
            call_counts=dict(call.counts),retry=0,resume=0,mandatory_stop=True))
        save(output/'resources.json',dict(wall_seconds=time.monotonic()-start,
            CPU_seconds=time.process_time(),peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
