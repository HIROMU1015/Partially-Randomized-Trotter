"""Prepare P metadata and INVALID grants. No scientific execution entry point."""
import argparse
import csv
import hashlib
import importlib.util
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import h4_dp_contract as c
import h4_p_seed_inventory as seeds

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT/'.h4_p_seal_private'
PUBLIC = ROOT/'artifacts/resource_applicability/track_a_h4_p_seal/2026-10-19'
OLD_PHASE = ROOT.parent/'h4-dp-phase0-20261019/.h4_dp_phase0_private'
OLD_STATS = ROOT.parent/'h4-s-stats-20261019/.h4_s_stats_private/run01_20261019'
BATCH = Path('/home/AbeHiromu/projects/track-a-pr-cost-batch-seal/20261018/private/sealed03')
FOLLOWUP = Path('/home/AbeHiromu/projects/h4-followup-validation/20261018/private')
GEOMETRY = Path('/home/AbeHiromu/projects/h4-handoff-evidence/20261011/h4-production-resume02-20261011/h4-newhost-signal-compile-20261011-resume02')
PHASE_ART = ROOT/'artifacts/resource_applicability/track_a_h4_dp_phase0/2026-10-19'
STATS_ART = ROOT/'artifacts/resource_applicability/track_a_h4_s_stats/2026-10-19'
GEOM_ART = ROOT/'artifacts/resource_applicability/track_a_h4_geometry_results/2026-10-13'
SOURCE_FILES = ['scripts/h4_dp_contract.py', 'scripts/h4_dp_science.py',
    'scripts/run_h4_dp_verification.py', 'scripts/h4_dp_witness.py',
    'scripts/h4_p_seed_inventory.py', 'scripts/prepare_h4_p_seal.py']


class Store:
    def __init__(self): self.rows=[]
    def file(self, role, path):
        path=Path(path);h=c.sha(path);copy=PRIVATE/'store'/h;copy.parent.mkdir(parents=True,exist_ok=True)
        if not copy.exists():copy.write_bytes(path.read_bytes())
        c.require(c.sha(copy)==h,'INPUT_COPY_BYTES')
        self.rows.append(dict(role=role, origin=str(path), path=str(copy), sha256=h))
        return copy


def preserve(rows):
    for item in rows:
        c.require(c.sha(item['path'])==item['sha256'] and c.sha(item['copy'])==item['sha256'],
                  'PRIOR_D_OR_STATS_RECORD_CHANGED')


def collect_inputs():
    PRIVATE.mkdir(exist_ok=True);PUBLIC.mkdir(parents=True,exist_ok=True)
    c.require(not (PRIVATE/'inventory_inputs_PRIVATE.json').exists(),'NEW_P_PREPARATION_SCOPE')
    store=Store()
    previous=c.read(OLD_STATS/'previous_preservation_manifest_PRIVATE.json')+c.read(OLD_STATS/'preservation_manifest_PRIVATE.json')
    preserve(previous);c.save(PRIVATE/'prior_preservation_PRIVATE.json',previous)
    draft=c.read(store.file('old_P_DRAFT',OLD_PHASE/'P_contract_DRAFT_PRIVATE.json'))
    stats_result=store.file('S_STATS_result',STATS_ART/'result.json')
    stats_audit=store.file('S_STATS_completion',STATS_ART/'completion_audit.json')
    c.require(c.read(stats_audit)['result_sha256']==c.sha(stats_result),'ACCEPTED_S_STATS_BYTES')
    store.file('S_STATS_user_approval',OLD_STATS/'user_approval_PRIVATE.json')
    store.file('S_STATS_native_exit',OLD_STATS/'native_exit.json')
    store.file('S_STATS_proposal',STATS_ART/'P_contract_after_S_STATS_DRAFT.json')
    for row in draft['inputs']:store.file(row['role'],row['path'])
    # Snapshot the exact original A outer module closure, omitted by the older
    # import_paths. Its provider/stage snapshots remain unchanged and separate.
    a_spec=c.read(store.file('A_original_technical_seal',BATCH/'contracts/H4_A_COST/technical_seal.json'))
    kernel=PRIVATE/'frozen_A_kernel';kernel_hashes={}
    for name,h in a_spec['source_hashes'].items():
        data=subprocess.check_output(['git','show',a_spec['SOURCE']+':'+name],cwd=ROOT)
        c.require(hashlib.sha256(data).hexdigest()==h,'ORIGINAL_A_KERNEL_GIT_SHA')
        target=kernel/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        kernel_hashes[name]=h
    for directory,files in draft['frozen_stage_closure'].items():
        for name,h in files.items():c.require(c.sha(Path(directory)/name)==h,'FROZEN_STAGE_OR_PROVIDER_CHANGED')
    staged_roles={x['role']:x['path'] for x in store.rows}
    adapter=c.read(draft['A_native_adapter']);adapter=dict(adapter,
        native_config=staged_roles['P_native_config'],
        import_paths=[str(kernel/'scripts')]+adapter['import_paths'])
    c.save(PRIVATE/'A_adapter_for_P.json',adapter)
    # Resolve and import only the lazy outer module; never call load_context.
    check='import sys;sys.path[:0]='+repr(adapter['import_paths'])+';import pr_cost_native;assert not any(n in sys.modules for n in ("numpy","qiskit","scipy"));print("LAZY_FROZEN_A_IMPORT_PASS")'
    completed=subprocess.run([draft['python'],'-B','-c',check],capture_output=True,text=True,
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1'))
    c.require(completed.returncode==0,'LAZY_A_MODULE_IMPORT:'+completed.stderr)
    # Explicit legacy seed values plus independently documented cohorts.
    a_registry=c.read(store.file('exploration_A_registry',OLD_PHASE/'existing_A_seed_registry.json'))
    values=set(c.read(store.file('old_explicit_exclusions',a_registry['old_seed_exclusions']))['values'])
    coverage=[dict(scope='LEGACY_EXPLICIT_EXCLUSION',status='COVERED',seed_values=len(values))]
    values.update(seeds.task_seed_values(a_registry['tasks']))
    coverage.append(dict(scope='EXPLORATION_A_N8',status='COVERED',trajectories=len(a_registry['tasks'])))
    native=c.read(staged_roles['P_native_config'])
    metrics=store.file('legacy_wrapper_seed_metadata',native['mapping']['old_metrics'])
    masters=set()
    with metrics.open() as f:
        for row in csv.DictReader(f):
            if str(row.get('seed','')).isdigit():
                master=int(row['seed']);q=int(re.search(r'_q(\d+)_',row['cell_id'])[1])
                masters.add(master);values.update(c.seed_hierarchy(master,q))
    values.update(native.get('old_seed_values',[]))
    coverage.append(dict(scope='LEGACY_FIXED_N32_AND_RECORDED_EXPLORATION',status='COVERED',unique_recorded_masters=len(masters)))
    for scope,relative,conditional in [
        ('CONDITIONAL_B','B_run01/output/seed_registry.json',True),
        ('T_0.4','T_run01/output/T_0.4/seed_registry.json',False),
        ('T_1.6','T_run01/output/T_1.6/seed_registry.json',False),
        ('DF_POLICY_FP','FP_run01/output/seed_registry.json',False)]:
        registry=c.read(store.file(scope+'_registry',FOLLOWUP/relative));values.update(seeds.task_seed_values(registry['tasks'],conditional))
        coverage.append(dict(scope=scope,status='COVERED',trajectories=sum(x.get('seed') is not None for x in registry['tasks'])))
    plan_path=ROOT/'artifacts/resource_applicability/track_a_h4_production_resume02/2026-10-11/plan_authorized_v1.json'
    plan=c.read(store.file('geometry_frozen_plan',plan_path))
    projection=c.read(store.file('geometry_fixed_identity_projection',GEOM_ART/'saved_candidates_v1.json'))
    cases=seeds.geometry_cases(plan,projection)
    by_geom={x['identity']['geometry'] for x in cases};anchors=[]
    for name in sorted(by_geom):
        case=next(x for x in cases if x['identity']['geometry']==name and x['template_id']=='B2-rank3-q1-r4-K2')
        identity=case['identity'];master=seeds.geometry_master(identity,0)
        key=seeds.geometry_fingerprint('h4-wrapper-v1',dict(identity,axis='cosine',trajectory_seed=master,index=0))
        path=store.file('geometry_master_anchor_'+name,GEOMETRY/('record-'+key+'.json'))
        item=c.read(path)
        c.require(item['trajectory_seed']==master and item['trajectory_index']==0
                  and item['wrapper_key']==key and item['status']=='COMPLETE','SAVED_GEOMETRY_MASTER_ANCHOR')
        anchors.append(dict(geometry=name,record_sha256=c.sha(path),saved_master_matches=True))
    for case in cases:
        values.update(seeds.geometry_master(case['identity'],replica) for replica in range(32))
    coverage.append(dict(scope='GEOMETRY_SIX_DISTANCES',status='COVERED_BY_FIXED_RULE_SUPERSET',
        registered_candidates=1308,stochastic_candidates=len(cases),trajectories=32*len(cases),
        derived_seed_policy='ALL_ORDER_AND_COMPONENT_SLOTS_UP_TO_K; NO_RTE_EVENT_SAMPLING',anchors=anchors))
    c.save(PRIVATE/'old_seed_exclusions_PRIVATE.json',dict(values=sorted(values),geometry_derived_streams='CHECKED_IN_SEPARATE_EXACT_RECEIPT'))
    c.save(PRIVATE/'geometry_seed_cases_PRIVATE.json',cases)
    c.save(PRIVATE/'inventory_inputs_PRIVATE.json',dict(inputs=store.rows,coverage=coverage,
        frozen_A_kernel=str(kernel),frozen_A_kernel_hashes=kernel_hashes,lazy_A_import_passed=True))
    print('P_INPUT_METADATA_COLLECTED; SCIENCE=0; OLD_RECORDS='+str(len(previous)))


def seal(source):
    metadata=c.read(PRIVATE/'inventory_inputs_PRIVATE.json');preserve(c.read(PRIVATE/'prior_preservation_PRIVATE.json'))
    policy_rows=[];policy_public=[]
    geometry_source='d0726e946f89f8c30a08d383fb8ff48b363629ec'
    for name in ['src/trottertracks/resource_applicability/h4_geometry/identity.py',
                 'src/trottertracks/resource_applicability/h4_geometry/signal.py']:
        data=subprocess.check_output(['git','show',geometry_source+':'+name],cwd=ROOT)
        h=hashlib.sha256(data).hexdigest()
        c.require(c.sha(ROOT/name)==h,'FIXED_GEOMETRY_INTEGER_POLICY_SOURCE')
        target=PRIVATE/'geometry_policy'/Path(name).name;target.parent.mkdir(exist_ok=True);target.write_bytes(data)
        policy_rows.append(dict(role='fixed_geometry_policy:'+Path(name).name,path=str(target),sha256=h))
        policy_public.append(dict(SOURCE=geometry_source,repo_path=name,sha256=h))
    source_hashes={name:c.sha(ROOT/name) for name in SOURCE_FILES}
    for name,h in source_hashes.items():
        data=subprocess.check_output(['git','show',source+':'+name],cwd=ROOT)
        c.require(hashlib.sha256(data).hexdigest()==h,'P_SOURCE_GIT_BINDING')
    by_role={x['role']:x['path'] for x in metadata['inputs']}
    spec=c.read(by_role['old_P_DRAFT']);result=c.read(by_role['S_STATS_result'])
    proposal=c.read(by_role['S_STATS_proposal'])
    d_path=ROOT/'artifacts/resource_applicability/track_a_h4_d_completed/2026-10-19/deterministic_result.json'
    c.require(c.sha(d_path)==result['D_result_sha256'],'FIXED_D_RESULT')
    endpoints=[dict(cell_id=x['cell_id'],epsilon=x['epsilon'],role=x['role'],saved_shots=x['PR_saved_shots'],
        PF_id=x['PF_cell_id'],PF_G_exact=x['PF_G_exact'],expected_direction=x['expected_direction'])
        for x in result['reference_directions']]
    spec.update(SOURCE=source,source_hashes=source_hashes,P_final_contract_version=2,
        status='SEALED_TECHNICAL_PENDING_USER_APPROVAL',approved_by_user=False,runtime_authorization=False,
        native_config=by_role['P_native_config'],A_native_adapter=str(PRIVATE/'A_adapter_for_P.json'),
        n_fixed=result['recommended']['n_fixed'],
        D_result_path=str(d_path),D_result_sha256=c.sha(d_path),evaluation_endpoints=endpoints,
        registration=by_role['P_registration'],signal_bindings=by_role['P_signal_bindings'],
        saved_preparation_identities=by_role['P_preparation_identities'],
        stats_result_path=by_role['S_STATS_result'],stats_result_sha256=c.sha(by_role['S_STATS_result']),
        stats_approval_receipt_sha256=c.sha(by_role['S_STATS_user_approval']),
        seed_exclusions=str(PRIVATE/'old_seed_exclusions_PRIVATE.json'),workers=None,
        output=str(PRIVATE/'P_run01_20261019/output'),lock=str(PRIVATE/'P_run01_20261019/ONE_SHOT.json'),
        pending=['SEPARATE_P_EXECUTION_APPROVAL','FRESH_P_RESOURCE_ADMISSION'])
    budget=result['recommended']['proposed_P_calls']
    spec['calls'].update({key:budget[key] for key in ['trajectory','native_evolution','occurrence','RTE_event','wrapper_compile']})
    spec['calls'].update(control_probe=budget['control_probe_maximum'],independent_event_oracle=budget['independent_event_oracle_maximum'])
    spec['frozen_stage_closure']=dict(spec['frozen_stage_closure'],**{metadata['frozen_A_kernel']:metadata['frozen_A_kernel_hashes']})
    spec['provider_hashes']={str(Path(root)/name):h for root,files in spec['frozen_stage_closure'].items() for name,h in files.items()}
    spec['seed_plan']=dict(schema='H4_P_SEED_PLAN_V1',SOURCE=source,binding=c.read(spec['A_native_adapter'])['binding'],
        cells=spec['cells'],n_fixed=spec['n_fixed'],evaluation_endpoints=endpoints,
        D_result_sha256=spec['D_result_sha256'],stats_result_sha256=spec['stats_result_sha256'],
        old_seed_exclusions_sha256=c.sha(spec['seed_exclusions']),cohort='INDEPENDENT_H4_CONFIRMATION')
    spec['seed_plan_digest']=c.digest(spec['seed_plan'])
    signals=c.read(spec['signal_bindings']);preps=c.read(spec['saved_preparation_identities'])['prefixes']
    spec['tasks'],new_values=seeds.confirmation_tasks(spec,signals,preps)
    c.require(len(spec['tasks'])==4185 and len(new_values)==11559,'FIXED_P_TRAJECTORY_AND_SEED_SLOTS')
    start=time.monotonic();checks,workers=seeds.check_geometry(c.read(PRIVATE/'geometry_seed_cases_PRIVATE.json'),new_values)
    c.require(all(not x['collisions'] for x in checks),'OLD_GEOMETRY_SEED_COLLISION')
    receipt=dict(status='COMPLETE_COLLISION_ZERO',all_registered_scopes_covered=True,
        inventory_cutoff_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),geometry_policy_sources=policy_public,
        geometry_distances=6,geometry_registered_candidates=1308,coverage=metadata['coverage'],geometry_checks=checks,
        old_explicit_exclusions_sha256=c.sha(spec['seed_exclusions']),new_seed_values_digest=c.digest(sorted(new_values)),
        new_seed_slots=len(new_values),new_master_count=len(spec['tasks']),internal_collision_count=0,
        comparison_is_mathematical_independence_proof=False,integer_metadata_workers=workers,
        geometry_check_wall_seconds=time.monotonic()-start,scientific_RTE_sampling=0,source=source)
    c.save(PRIVATE/'seed_inventory_receipt_PRIVATE.json',receipt)
    spec.update(seed_inventory_receipt=str(PRIVATE/'seed_inventory_receipt_PRIVATE.json'),
        seed_inventory_receipt_sha256=c.sha(PRIVATE/'seed_inventory_receipt_PRIVATE.json'),seed_collision_check_complete=True)
    input_roles=[dict(role=x['role'],path=x['path'],sha256=x['sha256']) for x in metadata['inputs'] if x['role']!='old_P_DRAFT']
    for role,path in [('A_adapter_final',spec['A_native_adapter']),('D_result_final',d_path),
        ('old_seed_exclusions_final',spec['seed_exclusions']),('old_geometry_seed_rules',PRIVATE/'geometry_seed_cases_PRIVATE.json'),
        ('seed_collision_receipt',spec['seed_inventory_receipt'])]:
        input_roles.append(dict(role=role,path=str(path),sha256=c.sha(path)))
    native=c.read(spec['native_config'])
    for role in ('snapshot','snapshot_receipt','df_receipt','saved_reference'):
        path=native['mapping'][role]
        input_roles.append(dict(role='actual_scientific_input:'+role,path=path,sha256=c.sha(path)))
    spec['inputs']=input_roles+policy_rows
    Path(spec['output']).parent.mkdir(exist_ok=True)
    c.require(not Path(spec['output']).exists() and not Path(spec['lock']).exists(),'P_NOT_STARTED')
    for path,h in spec['runtime_hashes'].items():c.require(c.sha(path)==h,'PINNED_P_RUNTIME_CHANGED')
    for path,h in spec['provider_hashes'].items():c.require(c.sha(path)==h,'PINNED_P_KERNEL_CHANGED')
    c.validate_tasks(spec)
    c.save(PRIVATE/'P_contract_TECHNICALLY_SEALED_PRIVATE.json',spec)
    c.save(PRIVATE/'new_confirmation_seeds_PRIVATE.json',dict(seed_plan=spec['seed_plan'],tasks=spec['tasks'],
        all_master_and_step_values=sorted(new_values)))
    grant=dict(stage='P',SOURCE=source,spec_digest=c.digest(spec),approved_by_user=False,runtime_authorization=False,
        approval_receipt=None,approval_receipt_sha256=None,one_shot=True,retry=False,next_stage_authorized=False)
    c.save(PRIVATE/'P_grant_INVALID.json',grant)
    public_receipt={k:v for k,v in receipt.items() if k not in ['coverage','geometry_checks']}
    public_receipt.update(coverage=metadata['coverage'],geometry_checks=[{k:v for k,v in x.items() if k!='collisions'} for x in checks],
        geometry_collision_count=0,seed_values_published=False)
    c.save(PUBLIC/'seed_inventory_projection.json',public_receipt)
    c.save(PUBLIC/'P_contract_TECHNICALLY_SEALED.json',dict(stage='P',SOURCE=source,status=spec['status'],
        technical_sealed=True,approved_by_user=False,runtime_authorization=False,
        scientific_binding=spec['seed_plan']['binding'],cells=spec['cells'],n_fixed=spec['n_fixed'],
        evaluation_endpoints=endpoints,calls=spec['calls'],pending=spec['pending'],
        seed_plan_digest=spec['seed_plan_digest'],private_seed_manifest_sha256=c.sha(PRIVATE/'new_confirmation_seeds_PRIVATE.json'),
        seed_collision_receipt_sha256=spec['seed_inventory_receipt_sha256'],private_spec_digest=c.digest(spec),
        source_hashes=source_hashes,input_hashes=[{k:v for k,v in x.items() if k!='path'} for x in spec['inputs']],
        provider_closure_files=len(spec['provider_hashes']),runtime_files=len(spec['runtime_hashes']),
        preparation_prefixes=[2,3],preparation_reconstruction_maximum_per_worker=2,
        new_scientific_sampling=0,new_compile=0,new_statistics=0,one_shot=True,retry=False,next_stage_authorized=False))
    c.save(PUBLIC/'P_grant_INVALID.json',grant)
    c.save(PUBLIC/'metadata_input_provenance.json',[{k:v for k,v in x.items() if k not in ['path','origin']} for x in metadata['inputs']])
    print('P_TECHNICALLY_SEALED; 4185 MASTERS / 11559 SEED_SLOTS; COLLISIONS=0; NO_SCIENCE; INVALID_GRANT')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--collect-inputs',action='store_true');p.add_argument('--source')
    args=p.parse_args()
    if args.collect_inputs:collect_inputs()
    elif args.source:seal(args.source)
    else:p.error('choose --collect-inputs or --source')


if __name__=='__main__':main()
