"""H2 FP/D/A metadata, preregistration and fail-closed one-shot gates.

Importing this module never opens molecular arrays or imports a science package.
"""
import ast
from copy import deepcopy
from fractions import Fraction
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import h2_pr_resource_contract as f
import h4_dp_contract as stats
from h4_native_contract import cell, Writer

require, read, save, sha, digest = f.require, f.read, f.save, f.sha, f.digest
ROOT, PRIVATE, PUBLIC, owned = f.ROOT, f.PRIVATE, f.PUBLIC, f.owned
SCOPE = 'H2_FP_D_A_V1_20261019'
STORE = PRIVATE / 'fp_d_a_preparation01'
EVIDENCE = PUBLIC / 'fp_d_a_preparation01'
EPS = [.05, .01, .005, .001, .0005]
F_COMMIT = '7ea6df2dbe34e64a34f468c343bb12922d3e2829'
A_SOURCE = '31ef62a12407010e9c1a07f9b37facedb8dff5fe'
BINDING = dict(system='linear_H2_1Angstrom', basis='STO-3G', rank=3,
    rank_policy='tol_only_1e-8', n_qubits=4, sector_dimension=4, nelec_alpha=1, nelec_beta=1,
    T=.8, P=0, control='symmetric_directional', scalar='original_plus_tail_once', reference_rotated=False,
    hamiltonian='d0bc260db6fc0a485fe518050075c18ccc0dff532253dcc719a52a61f41afa80',
    state='080b7ad2a46f38ed51e2989eb399ceda4cae95fca688722959ee65463cf174bf',
    snapshot='62c6481bd1f8b05ec3b399b3a5293873796737866409049d808e8fe432dcacaa',
    compiler='1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b')
COMPILER = dict(qiskit='1.3.0', basis_gates=['rz', 'sx', 'x', 'cx'], optimization_level=1,
                seed_transpiler=17, backend=None, coupling_map=None, layout=None, routing=None)
ENTRY = ('scripts/h2_panels_contract.py', 'scripts/h2_panels_science.py',
         'scripts/run_h2_panels.py', 'scripts/prepare_h2_panels.py',
         'scripts/h6_dba_science.py', 'scripts/h4_p_seed_inventory.py')


def registrations():
    d = [cell(m, p, q, form) for p in (0, 1, 2) for q in (1, 2, 4, 8, 16)
         for form in ('2nd', '4th') for m in ('B0', 'B0I')]
    d += [cell('B1', 3, q, form) for q in (1, 2, 4, 8, 16) for form in ('2nd', '4th')]
    a = [cell('B2', p, q, r=4, K=k) for p in (0, 1, 2)
         for q in (1, 2, 4, 8, 16) for k in (2, 4)]
    a += [cell('B2', p, q, r=r, K=k) for p in (0, 1, 2)
          for q in (1, 4, 16) for r in (2, 8) for k in (2, 4)]
    fp = [cell(m, p, 1, form) for m, p in [('B0', 1), ('B0I', 1), ('B1', 3)]
          for form in ('2nd', '4th')]
    fp += [cell('B2', 1, 1, r=2, K=2), cell('B2', 2, 2, r=4, K=4)]
    for stage, cells, size in [('FP', fp, 8), ('D', d, 70), ('A', a, 66)]:
        require(len(cells) == len({x['id'] for x in cells}) == size, 'UNIQUE_GRID:'+stage)
    return dict(FP=fp, D=d, A=a)


def budget(stage, primitive_probes):
    cells = registrations()[stage]
    fp = registrations()['FP']; reused = {x['id'] for x in fp}
    raw = [x for x in cells if x['method'] != 'B0I' and (stage == 'FP' or x['id'] not in reused)]
    pr = [x for x in cells if x['R']]
    trajectories = len(pr)*(8 if stage == 'A' else 1)
    det = sum(not x['R'] and (stage == 'FP' or x['id'] not in reused) for x in cells)
    native = trajectories + det
    return dict(SCF=0, integral_build=0, DF=0, saved_state_solver=0, quantum_shots=0,
        full_reference_pair=1 if stage == 'FP' else 0,
        full_reference_primary_columns=4 if stage == 'FP' else 0,
        full_reference_independent_columns=4 if stage == 'FP' else 0,
        truncated_reference_pair={'FP': 1, 'D': 2, 'A': 0}[stage],
        preparation={'FP': 4, 'D': 4, 'A': 3}[stage],
        primitive_probe=primitive_probes, primitive_spectrum=4,
        independent_tail_matrix={'FP': 2, 'D': 0, 'A': 3}[stage],
        base_signal=len(raw), independent_forward_signal=len(raw),
        B0I_overlay=sum(x['method'] == 'B0I' and (stage == 'FP' or x['id'] not in reused) for x in cells),
        new_signed_shot_endpoints=5*(len(raw)+sum(x['method']=='B0I' and
            (stage=='FP' or x['id'] not in reused) for x in cells)),
        trajectory=trajectories, occurrence=(8 if stage=='A' else 1)*sum(x['q'] for x in pr),
        RTE_event=(8 if stage=='A' else 1)*sum(x['R'] for x in pr),
        native_evolution=native, wrapper_compile=2*native,
        control_probe=4*(len(pr)+det), independent_event_oracle=len(pr),
        endpoint_accounting_pass=0 if stage=='FP' else 1)


def plan(stage):
    require(stage in ('FP', 'D', 'A'), 'FP_D_A_ONLY')
    return dict(scope=SCOPE, stage=stage, binding=BINDING, cells=registrations()[stage],
        epsilons=EPS, bias_mode='USUAL_SIGNED', allowance_factor=1., compiler_options=COMPILER,
        cohort='FP_NATIVE_DIAGNOSTIC_N1' if stage=='FP' else ('EXPLORATION_N8' if stage=='A' else 'DETERMINISTIC'),
        n=8 if stage=='A' else 1,
        acquisition='FP all deterministic plus nondegenerate PR; D/A any eligible signed factor1 endpoint',
        grid_rule='all proper prefixes incl0; full PF once; A r4 all q plus r2/8 at q1/4/16',
        full_rank_aliases='B0/B0I prefix3 alias B1; B2 prefix3 empty tail excluded',
        equivalences='prefix0 S2/S4 ideal unitary may coincide; retain distinct native decompositions',
        empty_tail='DEGENERATE_EMPTY_TAIL; explicitly ineligible; no replacement or sampling',
        unresolved='STOP; never treat unknown as ineligible',
        boundaries='q16/r2/r8 and p0/p2 are registered boundaries; no automatic expansion',
        seed_namespace=SCOPE+'_'+stage, cross_candidate_CRN=False,
        seed_hierarchy='direct_master_to_step_seeds_v1',
        shared_cosine_sine_trajectory=True, FP_PR_cost_pooled_with_A=False,
        instruction_guard=1000000, global_optimum_certified=False, population_mean_certified=False,
        resources=dict(agent_imposed_CPU_RAM_worker_wall_caps=None,
            worker_threads=1, reason='outer process parallelism; avoid nested BLAS pools',
            CPUs='fresh affinity, free resources and known occupied CPUs before activation',
            PSI='observation only', unrelated_worktree_changes_are_gates=False),
        automatic_retry=False, next_stage_authorized=False, mandatory_STOP=True,
        S_STATS_P_and_detailed_review_in_scope=False)


def source_closure():
    pending=list(ENTRY); found={}
    while pending:
        rel=pending.pop()
        if rel in found: continue
        path=ROOT/rel; found[rel]=sha(path)
        for node in ast.walk(ast.parse(path.read_text())):
            mods=([x.name for x in node.names] if isinstance(node, ast.Import) else
                  [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            for mod in mods:
                candidate='scripts/'+mod.split('.')[0]+'.py'
                if (ROOT/candidate).is_file(): pending.append(candidate)
    return found


def seed_tasks(stage, source):
    p=plan(stage); tasks=[]
    for x in p['cells']:
        if not x['R']: continue
        for rep in range(p['n']):
            master=int(digest([p['seed_namespace'],source,digest(p),BINDING,x['id'],rep])[:16],16)
            hierarchy=stats.seed_hierarchy(master,x['q'])
            tasks.append(dict(id=x['id']+'_'+str(rep),cell=x,replica=rep,seed=master,
                step_seeds=hierarchy if x['q']==1 else hierarchy[1:]))
    return tasks


def seed_values(tasks):
    values=[]
    for t in tasks: values.extend(stats.seed_hierarchy(t['seed'],t['cell']['q']))
    require(len(values)==len(set(values)), 'NEW_MASTER_OR_DERIVED_COLLISION')
    return set(values)


def invalid_grant(stage):
    return dict(scope=SCOPE,stage=stage,approved_by_user=False,runtime_authorization=False,
                one_shot=True,automatic_retry=False,next_stage_authorized=False)


def validate_metadata(spec):
    stage=spec['stage']
    require(spec['scope']==SCOPE and spec['plan']==plan(stage), 'FIXED_H2_PLAN')
    require(spec['source_hashes']==source_closure(), 'COMPLETE_SOURCE_CLOSURE')
    for rel,h in spec['source_hashes'].items():
        require(sha(ROOT/rel)==h, 'SOURCE_BYTES_CHANGED:'+rel)
        blob=subprocess.check_output(['git','--no-optional-locks','show',spec['SOURCE']+':'+rel],cwd=ROOT)
        require(hashlib.sha256(blob).hexdigest()==h, 'GIT_SOURCE_CHANGED:'+rel)
    for group in ('inputs','provider_files','kernel_files'):
        for row in spec[group]:
            path=owned(row['path']); require(sha(path)==row['sha256'], group+'_CHANGED:'+row['role'])
    require({str(p) for root in spec['provider_roots'] for p in owned(root).rglob('*')
             if p.is_file()}=={r['path'] for r in spec['provider_files']}, 'EXACT_PROVIDER_FILE_SET')
    require({str(p) for p in (STORE/'A_kernel').rglob('*') if p.is_file()}==
            {r['path'] for r in spec['kernel_files']}, 'EXACT_A_KERNEL_FILE_SET')
    require(sha(spec['runtime_manifest'])==spec['runtime_manifest_sha256'], 'RUNTIME_MANIFEST_CHANGED')
    require(sha(spec['python'])==spec['python_sha256'], 'FIXED_INTERPRETER_BYTES')
    require(spec['calls']==budget(stage,spec['primitive_probes']), 'FIXED_CALL_BUDGET')
    require(owned(spec['attempt'])==PRIVATE/('attempts/'+stage+'_run01'), 'STAGE_ATTEMPT_SCOPE')
    require(spec['native_config']['binding']==BINDING and spec['native_config']['compiler_options']==COMPILER,
            'FIXED_NATIVE_BINDING')
    config=spec['native_config']; paths={r['path']:r['sha256'] for r in spec['inputs']}
    require(config['provider_root'] in spec['provider_roots'] and spec['A_kernel_SOURCE']==A_SOURCE
        and spec['A_kernel_scripts']==str(STORE/'A_kernel/scripts'), 'ACTUAL_PROVIDER_AND_KERNEL_ROOTS')
    require(paths.get(config['mapping']['tol_snapshot'])==BINDING['snapshot']
            and config['mapping']['tol_receipt'] in paths, 'ACTUALLY_USED_F_INPUTS')
    if stage in ('FP','A'):
        registry=read(spec['seed_registry'])
        require(registry['tasks']==seed_tasks(stage,spec['SOURCE']), 'FIXED_SEED_TASKS')
        require(spec['seed_registry'] in paths, 'SEED_BYTES_BOUND')
        receipt=read(spec['seed_check']); values=seed_values(registry['tasks'])
        require(spec['seed_check'] in paths and receipt['status']=='COLLISION_ZERO'
            and receipt['new_values_digest']==digest(sorted(values))
            and all(not r['collisions'] for r in receipt['geometry_checks']), 'SEED_INVENTORY_CHECK')
        old=read(STORE/'old_seed_values.json')['values']
        require(str(STORE/'old_seed_values.json') in paths and
            receipt['old_explicit_exclusions_sha256']==sha(STORE/'old_seed_values.json') and
            not values&set(old), 'OLD_EXPLICIT_SEED_CHECK')


def authorize(spec,grant,*,runtime=True,interpreter=True):
    # This check intentionally precedes all provider/array/output access.
    require(grant.get('approved_by_user') is True and grant.get('runtime_authorization') is True,
            'SEPARATE_EXPLICIT_STAGE_APPROVAL_REQUIRED')
    require(grant.get('scope')==SCOPE and grant.get('stage')==spec.get('stage'), 'EXACT_STAGE_APPROVAL')
    require(all(grant.get(k) is v for k,v in [('one_shot',True),('automatic_retry',False),
            ('next_stage_authorized',False)]), 'ONE_SHOT_MANDATORY_STOP')
    require(spec.get('status')=='SEALED_FOR_EXECUTION' and spec.get('pending')==[], 'PENDING_CONTRACT')
    require(grant.get('SOURCE')==spec.get('SOURCE') and grant.get('spec_digest')==digest(spec), 'GRANT_BINDING')
    receipt=owned(grant['approval_receipt'])
    require(sha(receipt)==grant['approval_receipt_sha256'], 'APPROVAL_RECEIPT_BYTES')
    user=read(receipt)
    require(user.get('role')=='user' and user.get('stage')==spec['stage']
        and user.get('SOURCE')==spec['SOURCE'] and user.get('plan_digest')==digest(spec['plan'])
        and bool(user.get('verbatim_user_message','').strip()), 'EXPLICIT_STAGE_USER_RECEIPT')
    validate_metadata(spec)
    pending=owned(spec['pending_spec'])
    require(sha(pending)==spec['pending_spec_sha256'], 'ORIGINAL_PENDING_SEAL_PRESERVED')
    original=read(pending)
    for key in ('scope','stage','SOURCE','plan','calls','source_hashes','inputs','native_config',
                'provider_roots','provider_files','kernel_files','runtime_manifest','runtime_manifest_sha256',
                'python','python_sha256','attempt','primitive_probes','A_kernel_SOURCE','A_kernel_scripts'):
        require(spec[key]==original[key], 'ACTIVATION_MAY_NOT_CHANGE_SEAL:'+key)
    if spec['stage'] in ('FP','A'):
        for key in ('seed_registry','seed_check'): require(spec[key]==original[key],'SEALED_SEED_PATHS')
        refresh=owned(spec['seed_refresh'])
        require(sha(refresh)==spec['seed_refresh_sha256'],'ACTIVATION_SEED_REFRESH_BYTES')
        r=read(refresh); values=seed_values(read(spec['seed_registry'])['tasks'])
        require(r['status']=='COLLISION_ZERO' and r['new_values_digest']==digest(sorted(values))
            and r['known_campaigns_checked'] and r['new_RTE_sampling']==0 and r['checked_at_UTC'],
            'FRESH_SEED_METADATA_CHECK_REQUIRED')
    if runtime:
        for path,h in read(spec['runtime_manifest']).items(): require(sha(path)==h,'RUNTIME_CHANGED:'+path)
    if interpreter:
        require(Path(sys.executable).absolute()==Path(spec['python']).absolute(), 'FIXED_PYTHON_REQUIRED')
        require(sys.pycache_prefix==str(PRIVATE/'panels_runtime_pycache'), 'PRIVATE_PYCACHE_REQUIRED')
    required=['FP'] if spec['stage']=='D' else ['FP','D'] if spec['stage']=='A' else []
    require(set(spec.get('dependencies',{}))==set(required), 'EXACT_ACCEPTED_DEPENDENCIES')
    for name in required:
        dep=spec['dependencies'][name]; result=owned(dep['result']); accepted=owned(dep['acceptance'])
        require(sha(result)==dep['sha256'] and sha(accepted)==dep['acceptance_sha256'], 'DEPENDENCY_BYTES')
        r,a=read(result),read(accepted)
        require(r['status']=='COMPLETE_MANDATORY_STOP' and r['stage']==name and r['binding']==BINDING
            and a.get('accepted_by_user') is True and a.get('result_sha256')==dep['sha256']
            and bool(a.get('verbatim_user_message','').strip()), 'ACCEPTED_SAME_INPUT_DEPENDENCY')
        completion=owned(dep['completion_audit'])
        require(sha(completion)==dep['completion_audit_sha256'] and
            read(completion)['result_sha256']==dep['sha256'] and
            read(completion)['status']=='COMPLETE_MANDATORY_STOP', 'DEPENDENCY_NATIVE_COMPLETION')
    cpus=spec['worker_cpus']; require(cpus and len(cpus)==len(set(cpus)) and
        set(cpus)<=os.sched_getaffinity(0), 'FRESH_ALLOWED_WORKER_CPUS')
    resource=owned(spec['resource_receipt'])
    require(sha(resource)==spec['resource_receipt_sha256'] and read(resource)['admitted'] is True,
            'FRESH_NONINTERFERENCE_RESOURCE_ADMISSION')


def selected(records,cells):
    require(len(records)==len(cells) and {r['cell']['id'] for r in records}=={x['id'] for x in cells},
            'COMPLETE_SIGNAL_GRID')
    wanted={x['id']:x for x in cells}; out=[]
    for record in records:
        require(record['cell']==wanted[record['cell']['id']], 'EXACT_CELL')
        flags=[]
        for e in EPS:
            ep=record['precision_rows'][str(e)]['1.0']; flag=ep['cost_acquisition_eligible']
            require(type(flag) is bool,'RESOLVED_ELIGIBILITY')
            if flag:
                require(all(type(ep['axes'][a]['shots']) is int and ep['axes'][a]['shots']>0
                    for a in ('real','imag')), 'POSITIVE_SHOTS')
            else: require(ep['eligibility_under_declared_allowance']=='INELIGIBLE', 'UNKNOWN_IS_NOT_EXCLUSION')
            flags.append(flag)
        if any(flags): out.append(record['cell'])
    return out


def check_pair(pair):
    require(set(pair['axes'])=={'cosine','sine'}, 'PAIRED_AXES')
    for a in pair['axes'].values():
        require(type(a['RZ']) is int and a['RZ']>=0 and a['compiler_hash']==BINDING['compiler']
                and len(a['fingerprint'])==64, 'COMPILED_METRICS_AND_IDENTITY')
        if pair.get('seed') is not None:
            require(a['seed']==pair['seed'] and a['event_digest']==pair['event_digest'], 'PAIRED_SAME_TRAJECTORY')


def account(records,pairs,stage,d_result=None):
    require(stage in ('D','A'), 'FP_DIAGNOSTIC_HAS_NO_N1_STATISTICAL_SELECTION')
    selected(records,registrations()[stage]); rows=[]; covariances={}
    for r in records:
        cid=r['cell']['id']; values=pairs.get(cid,[]); weights={}
        for e in EPS:
            ep=r['precision_rows'][str(e)]['1.0']
            row=dict(cell_id=cid,epsilon=e,eligible=ep['cost_acquisition_eligible'],
                G_RZ_exact=None,G_RZ_SE=None,n=None,cohort=plan(stage)['cohort'])
            if row['eligible']:
                require(len(values)==(8 if stage=='A' else 1),'COMPLETE_ELIGIBLE_COST')
                for v in values: check_pair(v)
                w={a:ep['axes'][b]['shots'] for a,b in [('cosine','real'),('sine','imag')]}
                if stage=='A':
                    require({v['replica'] for v in values}==set(range(8)) and
                        all(v['cohort']=='EXPLORATION_N8' for v in values), 'FRESH_N8_NO_FP_POOLING')
                    s=stats.evaluate_pairs(values,w,1,1)
                    row.update(G_RZ_exact=s['G_exact'],G_RZ_SE=s['paired_SE'],sample_SD=s['sample_SD'],
                        sample_variance_exact=s['sample_variance_exact'],n=8)
                    weights[str(e)]=w
                else: row.update(G_RZ_exact=str(sum(w[a]*values[0]['axes'][a]['RZ'] for a in w)),n=1)
                row['shots']=w
            rows.append(row)
        if weights: covariances[cid]=dict(endpoints=list(weights),
            sample_covariance=stats.within_candidate_covariance(values,list(weights.values())))
    winners={}
    for e in EPS:
        candidates=[r for r in rows if r['epsilon']==e and r['eligible']]
        winners[str(e)]=min(candidates,key=lambda r:(Fraction(r['G_RZ_exact']),r['cell_id'])) if candidates else None
    comparisons=[]
    if stage=='A':
        require(d_result and d_result['stage']=='D' and d_result['binding']==BINDING,'ACCEPTED_D_COMPARATOR')
        for e in EPS:
            pf=d_result['accounting']['winners'][str(e)]; pr=winners[str(e)]
            ratio=None if not pf or not pr or Fraction(pf['G_RZ_exact'])==0 else float(Fraction(pr['G_RZ_exact'])/Fraction(pf['G_RZ_exact']))
            comparisons.append(dict(epsilon=e,PF=pf,PR=pr,PR_to_PF_ratio=ratio,
                status='NO_ELIGIBLE_PR' if not pr else 'NO_ELIGIBLE_PF' if not pf else
                       'ZERO_PF_DENOMINATOR' if ratio is None else 'FINITE_SAMPLE_POINT_COMPARISON'))
    return dict(rows=rows,winners=winners,covariances=covariances,comparisons=comparisons,
        registered_grid_only=True,population_mean_certified=False)


class Calls:
    """Append-only nonrefundable reservations; claim a wave before starting it."""
    def __init__(self,output,limits):
        self.output=Path(output); self.limits=limits; self.used={k:0 for k in limits}; self.serial=0
    def take(self,name,count=1):
        require(type(count) is int and count>=0 and name in self.limits and
                self.used[name]+count<=self.limits[name], 'SCIENTIFIC_CALL_BUDGET:'+name)
        if not count: return
        save(self.output/('reservation_%04d.json'%self.serial),dict(category=name,count=count,
             nonrefundable=True,returned_success_not_implied=True))
        self.used[name]+=count; self.serial+=1
