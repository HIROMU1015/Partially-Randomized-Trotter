"""H2 independent P: immutable metadata and approval gates, no science on import."""
import ast
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import h2_panels_contract as base

require,read,save,sha,digest,owned=base.require,base.read,base.save,base.sha,base.digest,base.owned
ROOT,PRIVATE,PUBLIC=base.ROOT,base.PRIVATE,base.PUBLIC
STORE=PRIVATE/'P_seal02'
EVIDENCE=PUBLIC/'P_seal02'
SCOPE='H2_P_INDEPENDENT_CONFIRMATION_V1_20261019'
COHORT='INDEPENDENT_H2_CONFIRMATION'
STATS_COMMIT='829d697ae7905b3ea90a61da9d2d4da15d8b757a'
DEPENDENCIES={
 'F':('7ea6df2dbe34e64a34f468c343bb12922d3e2829','F_completed_run01','37b1cf72990885962d8449d8bd5ec7e303329350'),
 'FP':('af724928689be3592b391fab3a14538bf4752b76','FP_completed_run01','de72e5d4e48177734c0ddd79dffac14222410658'),
 'D':('7b202a1a569c6839c4c7a5dfb18b6ff51ec3f3eb','D_completed_run02','cf99e4452df96291d7e9d37d76d2e91be5cbc413'),
 'A':('1d2f67ec285956ef4090ea26520d64b11e925f9c','A_completed_run01','de72e5d4e48177734c0ddd79dffac14222410658'),
 'S_STATS':(STATS_COMMIT,'S_STATS_completed_run01','e0b959e8a617f7934d05c3e249832ccc1c43909f')}
TARGETS=((.05,'B2_p1_q1_2nd_r2_K4','B1_p3_q1_2nd',80),
         (.0005,'B2_p1_q4_2nd_r2_K4','B1_p3_q4_2nd',12))
N_FIXED={cid:n for _,cid,_,n in TARGETS}
CALLS=dict(saved_input_load=1,preparation=1,SCF=0,integral_build=0,DF=0,saved_state_solver=0,
 quantum_shots=0,full_reference_pair=0,truncated_reference_pair=0,primitive_probe=0,primitive_spectrum=0,
 base_signal=0,independent_forward_signal=0,B0I_overlay=0,new_signed_shot_endpoints=0,
 trajectory=92,native_evolution=92,wrapper_compile=184,occurrence=128,RTE_event=256,
 control_probe=8,independent_event_oracle=2,endpoint_accounting_pass=1,
 PF_compile=0,bootstrap=0,shot_recalculation=0,candidate_reselection=0,additional_sampling=0)
PENDING=['SEPARATE_EXPLICIT_P_SCIENCE_APPROVAL','FRESH_NONINTERFERENCE_RESOURCE_ADMISSION',
         'FRESH_KNOWN_SEED_INVENTORY_CHECK']
ENTRY=('scripts/h2_p_contract.py','scripts/h2_p_science.py','scripts/prepare_h2_p_seal.py',
       'scripts/run_h2_p.py','tests/test_h2_p_seal.py')


def cells():
    registered={x['id']:x for x in base.registrations()['A']}
    return [deepcopy(registered[cid]) for _,cid,_,_ in TARGETS]


def plan(stage='P'):
    require(stage=='P','H2_P_ONLY')
    return dict(scope=SCOPE,stage='P',binding=deepcopy(base.BINDING),cells=cells(),n_fixed=dict(N_FIXED),
        targets=[dict(epsilon=e,cell_id=cid,PF_cell_id=pf,n=n) for e,cid,pf,n in TARGETS],
        compiler_options=deepcopy(base.COMPILER),cohort=COHORT,seed_namespace=SCOPE,
        bias_mode='USUAL_SIGNED',allowance_factor=1.,saved_shots_only=True,PF_saved_only=True,
        original_finite_RTE_law=True,shared_cosine_sine_trajectory=True,cross_candidate_CRN=False,
        seed_hierarchy='direct_master_to_step_seeds_v1',calls=dict(CALLS),
        one_shot=True,automatic_retry=False,additional_sampling=False,candidate_reselection=False,
        optional_stopping=False,first_replica_controls_before_remaining=True,
        law_probabilities=None,unseen_event_cost=None,unseen_event_variance=None,
        population_mean_certified=False,actual_power_certified=False,selection_adjusted_inference=False,
        resources=dict(agent_imposed_CPU_RAM_worker_wall_caps=None,worker_threads=1,
            worker_count='fresh allowed CPUs, available resources and task count',PSI_diagnostic_only=True,
            shared_settings_modified=False,unrelated_HEAD_status_are_gates=False),
        next_stage_authorized=False,mandatory_STOP=True)


def source_closure():
    pending=list(ENTRY);rows={}
    while pending:
        rel=pending.pop()
        if rel in rows:continue
        path=ROOT/rel;rows[rel]=sha(path)
        for node in ast.walk(ast.parse(path.read_text())):
            mods=[x.name for x in node.names] if isinstance(node,ast.Import) else (
                [node.module] if isinstance(node,ast.ImportFrom) and node.module else [])
            for mod in mods:
                target='scripts/'+mod.split('.')[0]+'.py'
                if (ROOT/target).is_file():pending.append(target)
    return rows


def seed_plan(source,endpoints,dependencies,exclusion_sha):
    return dict(namespace=SCOPE,SOURCE=source,plan=plan(),endpoints=deepcopy(endpoints),
        dependencies=deepcopy(dependencies),old_exclusions_sha256=exclusion_sha,
        cohort=COHORT,redraw_on_collision=False,actual_RTE_event_sampling=False)


def tasks(source,seed_plan_digest):
    out=[]
    for cell in cells():
        for replica in range(N_FIXED[cell['id']]):
            master=int(digest([SCOPE,source,seed_plan_digest,cell['id'],replica])[:16],16)
            hierarchy=base.stats.seed_hierarchy(master,cell['q'])
            out.append(dict(id=cell['id']+'_'+str(replica),cell=cell,replica=replica,seed=master,
                step_seeds=hierarchy if cell['q']==1 else hierarchy[1:],cohort=COHORT))
    return out


def check_tasks(registry,source,plan_digest,excluded):
    require(registry['SOURCE']==source and registry['seed_plan_digest']==plan_digest
        and registry['tasks']==tasks(source,plan_digest),'EXACT_92_SEALED_P_TASKS')
    values=base.seed_values(registry['tasks'])
    require(len(registry['tasks'])==92 and len(values)==140 and not values&set(excluded),
        'OLD_OR_NEW_P_SEED_COLLISION')
    return values


def selected_endpoints(a,d,stats):
    """Copy adopted saved rows; never recompute n/SD/shots or select candidates."""
    require(a['binding']==d['binding']==stats['plan']['binding']==base.BINDING,'SAME_H2_RANK3_SECTOR4')
    require(stats['recommended']['n_fixed']==N_FIXED,'USER_ADOPTED_N80_12')
    selected=stats['recommended']['proposed_P_calls']
    for key in ('trajectory','native_evolution','wrapper_compile','occurrence','RTE_event','endpoint_accounting_pass'):
        require(selected[key]==CALLS[key],'S_STATS_SAVED_P_BUDGET')
    require(selected['seed_slots']==140 and selected['control_probe_maximum']==8
        and selected['independent_event_oracle_maximum']==2 and not stats['P_executed'],'S_STATS_P_UNEXECUTED')
    out=[]
    for epsilon,cid,pf,n in TARGETS:
        pr=a['accounting']['winners'][str(epsilon)];control=d['accounting']['winners'][str(epsilon)]
        direction=next(r for r in stats['reference_directions'] if r['epsilon']==epsilon)
        require(pr['cell_id']==cid and control['cell_id']==pf and pr['eligible'] and control['eligible']
            and pr['cohort']=='EXPLORATION_N8' and pr['n']==8 and control['cohort']=='DETERMINISTIC'
            and direction['cell_id']==cid and direction['PR_mean_exact']==pr['G_RZ_exact']
            and direction['PF_cell_id']==pf and direction['PF_G_exact']==control['G_RZ_exact']
            and direction['expected_direction']==1,'SAVED_FIXED_P_ENDPOINT')
        signal=next(r for r in a['signals'] if r['cell']['id']==cid)
        require(signal['precision_rows'][str(epsilon)]['1.0']['cost_acquisition_eligible'], 'SAVED_ELIGIBLE_SIGNAL')
        require(pr['shots']=={axis:signal['precision_rows'][str(epsilon)]['1.0']['axes'][old]['shots']
            for axis,old in [('cosine','real'),('sine','imag')]},'EXACT_SAVED_P_SHOTS')
        out.append(dict(epsilon=epsilon,cell_id=cid,n=n,saved_shots=deepcopy(pr['shots']),
            PF=deepcopy(control),expected_direction=direction['expected_direction'],
            signal=deepcopy(signal),preparation_hash=a['preparation_identities'][cid]))
    return out


def verify_rows(rows):
    require(len({r['role'] for r in rows})==len(rows),'UNIQUE_P_INPUT_ROLES')
    for r in rows:
        p=owned(r['path']);require(p.is_file() and sha(p)==r['sha256'],'P_BYTES_CHANGED:'+r['role'])
        if r.get('commit'):
            blob=subprocess.check_output(['git','--no-optional-locks','show',r['commit']+':'+r['repo_path']],cwd=ROOT)
            require(hashlib.sha256(blob).hexdigest()==r['sha256'],'P_INPUT_GIT_IDENTITY')


def public_manifest_rows(manifest):
    """Read the existing F dict, FP file-key list and later path-key lists."""
    files=manifest['files']
    if isinstance(files,dict):rows=[dict(path=p,sha256=h) for p,h in files.items()]
    else:
        require(isinstance(files,list),'KNOWN_PUBLIC_MANIFEST_FORMAT')
        rows=[dict(path=r.get('path',r.get('file')),sha256=r['sha256']) for r in files]
    require(rows and len({r['path'] for r in rows})==len(rows),'UNIQUE_PUBLIC_MANIFEST_PATHS')
    for r in rows:
        require(isinstance(r['path'],str) and Path(r['path']).name==r['path']
            and r['path'] not in ('.','..') and isinstance(r['sha256'],str) and len(r['sha256'])==64,
            'PUBLIC_MANIFEST_FILE_AND_SHA')
    return rows


def verify(spec,*,runtime=True):
    require(spec['scope']==SCOPE and spec['stage']=='P' and spec['plan']==plan()
        and spec['calls']==CALLS,'EXACT_H2_P_SCOPE_N_BUDGET')
    require(spec['source_hashes']==source_closure(),'EXACT_P_SOURCE_CLOSURE')
    for rel,h in spec['source_hashes'].items():
        blob=subprocess.check_output(['git','--no-optional-locks','show',spec['SOURCE']+':'+rel],cwd=ROOT)
        require(sha(ROOT/rel)==hashlib.sha256(blob).hexdigest()==h,'P_SOURCE_BYTES:'+rel)
    for group in ('inputs','provider_files','kernel_files'):verify_rows(spec[group])
    bound={r['path']:r['sha256'] for r in spec['inputs']}
    for key in ('seed_registry','seed_check','seed_exclusions','seed_origins','prior_manifest','adoption_receipt'):
        require(bound.get(spec[key])==sha(owned(spec[key])),'ACTUALLY_USED_P_INPUT:'+key)
    require(set(spec['dependencies'])==set(DEPENDENCIES),'F_FP_D_A_STATS_DEPENDENCIES')
    results={}
    for stage,dep in spec['dependencies'].items():
        require((dep['commit'],dep['folder'],dep['SOURCE'])==DEPENDENCIES[stage],'FIXED_COMPLETED_DEPENDENCY')
        for key in ('result','audit','native','binding'):
            require(bound.get(dep[key])==sha(dep[key]),'ACTUAL_DEPENDENCY_PATH')
        result,audit,native,binding=[read(dep[k]) for k in ('result','audit','native','binding')]
        require(result['SOURCE']==audit['SOURCE']==binding['SOURCE']==dep['SOURCE']
            and result['status']==audit['status']=='COMPLETE_MANDATORY_STOP'
            and audit['result_sha256']==sha(dep['result']) and native['exit_code']==audit['native_exit_code']==0
            and native['same_parent_reaped'] and native['wait4_complete'] and audit['same_parent_reaped']
            and (stage=='F' or audit['wait4_complete']),
            'ACCEPTED_NATIVE_DEPENDENCY')
        for rel,h in binding['source_hashes'].items():
            blob=subprocess.check_output(['git','--no-optional-locks','show',dep['SOURCE']+':'+rel],cwd=ROOT)
            require(sha(ROOT/rel)==hashlib.sha256(blob).hexdigest()==h,'UPSTREAM_SOURCE_PRESERVED')
        results[stage]=result
    require(spec['endpoints']==selected_endpoints(results['A'],results['D'],results['S_STATS']),
        'VERBATIM_SAVED_P_ENDPOINTS')
    f=results['F'];require(f['rank']==3 and f['sector']['dimension']==4
        and f['hamiltonian_hash']==base.BINDING['hamiltonian'] and f['state_hash']==base.BINDING['state'], 'SAVED_F_IDENTITY')
    require(results['FP']['binding']==base.BINDING,'FP_SAME_H2_REFERENCE')
    config=spec['native_config']
    require(config['binding']==base.BINDING and config['compiler_options']==base.COMPILER
        and bound.get(config['mapping']['tol_snapshot'])==base.BINDING['snapshot']
        and config['mapping']['tol_receipt'] in bound,'ACTUAL_SAVED_F_AND_COMPILER')
    require(config['provider_root'] in spec['provider_roots'] and spec['A_kernel_SOURCE']==base.A_SOURCE
        and spec['A_kernel_scripts']==str(base.STORE/'A_kernel/scripts'),'FIXED_P_PROVIDER_KERNEL_ROOTS')
    require({str(p) for d in spec['provider_roots'] for p in owned(d).rglob('*') if p.is_file()}==
        {r['path'] for r in spec['provider_files']},'EXACT_P_PROVIDER_FILE_SET')
    require({str(p) for p in (base.STORE/'A_kernel').rglob('*') if p.is_file()}==
        {r['path'] for r in spec['kernel_files']},'EXACT_P_KERNEL_FILE_SET')
    for r in read(spec['prior_manifest']):require(sha(r['path'])==r['sha256'],'PRIOR_H2_EVIDENCE_CHANGED')
    old=read(spec['seed_exclusions'])['values'];registry=read(spec['seed_registry'])
    require(spec['seed_plan']==seed_plan(spec['SOURCE'],spec['endpoints'],spec['dependencies'],sha(spec['seed_exclusions']))
        and spec['seed_plan_digest']==digest(spec['seed_plan']),'SEALED_P_SEED_PLAN')
    values=check_tasks(registry,spec['SOURCE'],spec['seed_plan_digest'],old);receipt=read(spec['seed_check'])
    require(receipt['status']=='COLLISION_ZERO' and receipt['new_values_digest']==digest(sorted(values))
        and receipt['seed_registry_sha256']==sha(spec['seed_registry']) and not receipt['seed_redrawn']
        and receipt['new_RTE_sampling']==0 and sum(r['old_seed_slots_checked'] for r in receipt['geometry_checks'])==7737216
        and all(not r['collisions'] for r in receipt['geometry_checks']),'P_GEOMETRY_AND_SEED_PROOF')
    for origin in read(spec['seed_origins']):require(sha(origin['snapshot'])==origin['sha256'],'FROZEN_P_SEED_ORIGIN')
    adoption=read(spec['adoption_receipt'])
    require(adoption['role']=='user' and adoption['accepted_S_STATS_commit']==STATS_COMMIT
        and adoption['adopted_n']==N_FIXED and not adoption['P_scientific_execution_authorized'], 'USER_N_ADOPTION_SEAL_ONLY')
    require(sha(spec['runtime_manifest'])==spec['runtime_manifest_sha256']
        and sha(spec['python'])==spec['python_sha256'],'P_RUNTIME_MANIFEST_PYTHON')
    if runtime:
        for path,h in read(spec['runtime_manifest']).items():require(sha(path)==h,'P_RUNTIME_CHANGED:'+path)
    require(owned(spec['attempt'])==PRIVATE/'attempts/P_run01','OWNED_P_ATTEMPT')
    return dict(status='PASS',SOURCE=spec['SOURCE'],trajectory=92,compile_axes=184,RTE_event=256)


def invalid_grant():
    return dict(scope=SCOPE,stage='P',approved_by_user=False,runtime_authorization=False,
        one_shot=True,automatic_retry=False,next_stage_authorized=False)


def authorize(spec,grant):
    # Must precede scientific imports, arrays, output, marker and resource probes.
    require(grant.get('approved_by_user') is True and grant.get('runtime_authorization') is True,
        'SEPARATE_EXPLICIT_P_SCIENCE_APPROVAL_REQUIRED')
    require(grant.get('scope')==spec.get('scope')==SCOPE and grant.get('stage')==spec.get('stage')=='P','H2_P_ONLY')
    require(grant.get('one_shot') is True and grant.get('automatic_retry') is False
        and grant.get('next_stage_authorized') is False,'P_ONE_SHOT_STOP_REQUIRED')
    require(spec.get('status')=='SEALED_FOR_EXECUTION' and spec.get('pending')==[],'P_STILL_PENDING')
    require(grant['SOURCE']==spec['SOURCE'] and grant['spec_digest']==digest(spec),'P_GRANT_BINDING')
    pending=owned(spec['pending_spec']);require(sha(pending)==spec['pending_spec_sha256'],'P_PENDING_SEAL_BYTES')
    allowed={'status','pending','pending_spec','pending_spec_sha256','worker_cpus','resource_receipt',
        'resource_receipt_sha256','seed_refresh','seed_refresh_sha256'}
    require({k:v for k,v in spec.items() if k not in allowed}==
        {k:v for k,v in read(pending).items() if k not in allowed},'P_ACTIVATION_MAY_NOT_CHANGE_SEAL')
    approval=owned(grant['approval_receipt']);require(sha(approval)==grant['approval_receipt_sha256'],'P_APPROVAL_BYTES')
    user=read(approval)
    require(user['role']=='user' and user['stage']=='P' and user['SOURCE']==spec['SOURCE']
        and user['plan_digest']==digest(plan()) and user['P_scientific_execution_authorized'] is True
        and user['verbatim_user_message'].strip(),'SEPARATE_P_SCIENCE_RECEIPT')
    verify(spec)
    cpus=spec['worker_cpus'];require(cpus and len(cpus)==len(set(cpus))
        and set(cpus)<=os.sched_getaffinity(0),'FRESH_P_WORKER_CPUS')
    resource=owned(spec['resource_receipt']);require(sha(resource)==spec['resource_receipt_sha256']
        and read(resource)['admitted'] is True,'FRESH_P_NONINTERFERENCE_RESOURCES')
    refresh=owned(spec['seed_refresh']);require(sha(refresh)==spec['seed_refresh_sha256'],'FRESH_P_SEED_BYTES')
    r=read(refresh)
    require(r['status']=='COLLISION_ZERO' and r['checked_at_UTC'] and r['new_RTE_sampling']==0
        and r['new_values_digest']==read(spec['seed_check'])['new_values_digest']
        and r['seed_registry_sha256']==sha(spec['seed_registry']) and r['known_campaigns_checked']
        and r['exclusion_values_checked']>=len(read(spec['seed_exclusions'])['values']), 'FRESH_P_SEED_CHECK')
    exclusions=owned(r['exclusions_path']);origins=owned(r['origins_path'])
    require(sha(exclusions)==r['exclusions_sha256'] and sha(origins)==r['origins_sha256'],'FRESH_P_SEED_INVENTORY_BYTES')
    old=set(read(exclusions)['values']);new=base.seed_values(read(spec['seed_registry'])['tasks'])
    require(set(read(spec['seed_exclusions'])['values'])<=old and not old&new
        and r['exclusion_values_checked']==len(old),'FRESH_P_SEED_COLLISION_OR_LOST_EXCLUSIONS')
    for origin in read(origins):require(sha(owned(origin['snapshot']))==origin['sha256'],'FRESH_P_SEED_ORIGIN_BYTES')
    require(Path(sys.executable).absolute()==Path(spec['python']).absolute(),'PINNED_P_PYTHON')


def check_measurements(measured,registry,endpoints):
    expected={(t['cell']['id'],t['replica']):t for t in registry['tasks']}
    require(len(measured)==len(expected)==92 and {(r['cell_id'],r['replica']) for r in measured}==set(expected),
        'ALL_92_FIXED_P_MEASUREMENTS_REQUIRED')
    hashes={e['cell_id']:e['preparation_hash'] for e in endpoints}
    for row in measured:
        task=expected[row['cell_id'],row['replica']];base.check_pair(row)
        require(row['cohort']==COHORT and row['seed']==task['seed']
            and row['preparation_hash']==hashes[row['cell_id']],'INDEPENDENT_P_ONLY_NO_A_POOLING')


def account(measured,registry,endpoints):
    """Future separately authorized P pass, or artificial fixtures only."""
    check_measurements(measured,registry,endpoints);out=[]
    for ep in endpoints:
        pairs=[r for r in measured if r['cell_id']==ep['cell_id']]
        value=base.stats.evaluate_pairs(pairs,ep['saved_shots'],ep['PF']['G_RZ_exact'],ep['expected_direction'])
        out.append(dict(cell_id=ep['cell_id'],epsilon=ep['epsilon'],n=ep['n'],PF=deepcopy(ep['PF']),
            shots=deepcopy(ep['saved_shots']),**value))
    return dict(comparisons=out,candidate_reselected=False,old_cohorts_pooled=False,
        law_probabilities=None,unseen_event_cost=None,unseen_event_variance=None,
        population_mean_certified=False,actual_power_certified=False)
