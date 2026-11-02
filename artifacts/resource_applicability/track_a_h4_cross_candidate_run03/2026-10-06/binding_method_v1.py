"""Metadata/source-only fixation for user-authorized cross-candidate run03."""
import ast,copy,hashlib,json,os,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT=Path('/home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/worktrees/track-a-h4-cross-candidate-run03-20261006')
BASE=Path('/home/AbeHiromu/projects/partially-randomized-trotter')
CONTROL=BASE/'.server-preparation/executions/h4-signal-compile-run03-revision'
BUNDLE=ROOT/'artifacts/resource_applicability/track_a_h4_cross_candidate_run03/2026-10-06'
OLD=BASE/'.server-preparation/worktrees/track-a-h4-worker-bootstrap-run02-20261006'
PREP=BASE/'.server-preparation/worktrees/track-a-h4-signal-compile-plan-review-20261006/artifacts/resource_applicability/track_a_h4_signal_compile_plan_review/2026-10-06'
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.resource_applicability.h4_geometry import gates
from trottertracks.resource_applicability.h4_geometry.identity import sha,fingerprint,require

def save(root,name,value):
    with (root/name).open('x') as f:
        json.dump(value,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write('\n')

def main():
    source=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    require(source=='79a552d517ee4d391629b1929ab90d2ba1dae451','frozen source')
    stopped=json.loads((CONTROL/'old_owned_run_stopped_v1.json').read_bytes())
    require(stopped['driver_stopped'] and not stopped['remaining_owned_workers'],'predecessor stopped')
    startup=json.loads((BASE/'.server-preparation/executions/h4-signal-compile-run02-review/startup_observation_v1.json').read_bytes())
    for child in startup['owned_workers']:
        p=Path('/proc/%d/stat'%child['pid'])
        if p.exists():
            fields=p.read_text().rsplit(')',1)[1].split()
            require(fields[19]!=child['starttime_ticks'] or fields[0]=='Z','old owned worker still alive')
    oldaudit_path=OLD/'artifacts/resource_applicability/track_a_h4_worker_bootstrap_run02/2026-10-06/source_freeze_v1.json'
    oldaudit=json.loads(oldaudit_path.read_bytes())
    hashes={p:sha((ROOT/p).read_bytes()) for p in oldaudit['new_source_hashes']}
    parents={p:sha((ROOT/p).read_bytes()) for p in oldaudit['namespace_parent_hashes']}
    for p,h in {**hashes,**parents}.items():
        require(sha(gates.git_blob(ROOT,source,p))==h,'SOURCE blob mismatch')
        ast.parse((ROOT/p).read_bytes(),filename=p)
    contract=gates.verify_contract(ROOT)
    env=gates.environment_matches(oldaudit['dependency_observations'])
    options=gates.compiler_matches(contract['compiler_environment_reference']['compiler'])
    for p,h in oldaudit['installed_source_hashes'].items():require(sha(Path(p).read_bytes())==h,'installed source bytes changed')
    tests=ET.parse(BUNDLE/'test_attempt_3.xml').getroot().find('testsuite').attrib
    require(int(tests['tests'])==49 and all(int(tests[k])==0 for k in ('errors','failures','skipped')),'final local tests')
    audit={**oldaudit,'schema_version':'h4-cross-candidate-run03-source-audit-v1','status':'SOURCE_BLOBS_VERIFIED',
      'source_commit':source,'source_checkout_root':str(ROOT),'source_audit_reference':gates.SOURCE_AUDIT,
      'old_source_commit':oldaudit['source_commit'],'old_source_audit_sha256':sha(oldaudit_path.read_bytes()),
      'new_source_hashes':hashes,'namespace_parent_hashes':parents,'dependency_observations':env,
      'compiler_options':options,'observed_utc':datetime.now(timezone.utc).isoformat(),
      'changed_source19_paths':sorted(p for p,h in hashes.items() if h!=oldaudit['new_source_hashes'][p]),
      'unchanged_source19_paths':sorted(p for p,h in {**hashes,**parents}.items() if h=={**oldaudit['new_source_hashes'],**oldaudit['namespace_parent_hashes']}[p]),
      'only_changes':'bounded cross-candidate scheduling, job lifetime cleanup/first-error reporting, new explicit run03 input reuse and cumulative byte/wall/invocation bindings',
      'new_regression_tests':{'tests':49,'failures':0,'errors':0,'skipped':0,'production_science_jobs':0,'actual_transpile_invocations':0},
      'science_inputs_execution_signal_compiler_parallel_resources_unchanged':False,
      'scientific_math_circuits_inputs_identity_signal_compiler_caps_unchanged':True,
      'runtime_success_or_scientific_results_claimed':False}
    save(BUNDLE,'source_freeze_v1.json',audit)
    old_root=Path(gates.REUSE_ROOT);freeze_bytes=(old_root/'generation-freeze.json').read_bytes();freeze=json.loads(freeze_bytes)
    require(sha(freeze_bytes)==stopped['generation_freeze_sha256'],'original freeze bytes')
    journal=(old_root/'byte-budget.journal').read_bytes();require(sha(journal)==stopped['prior_journal_sha256'],'stopped journal unchanged')
    charged=sum(int(journal[i:i+128].strip()) for i in range(0,len(journal),128));require(charged==stopped['prior_cumulative_charge_bytes'],'prior byte charge')
    entries={};reservations={};chain=None
    paths=sorted(old_root.glob('ledger-*.json'))
    for index,p in enumerate(paths):
        payload=json.loads(p.read_bytes());require(payload['version']==index and payload['previous_digest']==chain,'old ledger chain')
        chain=fingerprint('h4-ledger-delta-v1',payload);entries.update(payload['entries']);reservations.update(payload['reservations'])
    require(len(reservations)==4 and len(entries)==2,'old invocation accounting')
    reuse={'schema_version':'h4-input-reuse-and-prior-budget-v1','new_run_id':gates.RUN_ID,'input_root':gates.REUSE_ROOT,
      'generation_run_id':freeze['run_id'],'generation_source_commit':freeze['source_commit'],
      'generation_freeze_file_sha256':sha(freeze_bytes),'generation_freeze_fingerprint':fingerprint('h4-generation-freeze-v1',freeze),
      'old_run_stopped':True,'old_run_status':'OWNED_POOL_FAILED_STOP_BEFORE_REVISION_INSPECTION',
      'authorization_authority':'USER_EXPLICIT_MESSAGE','user_instruction':'その修正を入れて、ワーカー数を増やして再実行して',
      'prior_cumulative_charge_bytes':charged,'prior_journal_sha256':sha(journal),'prior_journal_bytes':len(journal),
      'prior_actual_invocations':len(reservations),'prior_completed_compile_records':len(entries),
      'prior_wall_seconds':stopped['conservative_prior_wall_seconds_for_reexecution'],
      'wall_basis':'conservative upper bound through revision inspection plus60s; includes any downtime, never refunds predecessor wall',
      'old_ledger_head_digest':chain,'old_ledger_files':[{'name':p.name,'sha256':sha(p.read_bytes())} for p in paths],
      'old_stop_record_sha256':sha((CONTROL/'old_owned_run_stopped_v1.json').read_bytes()),
      'old_inputs_or_partial_compile_metrics_copied_relabelled_or_deleted':False,
      'old_partial_compile_metrics_reused':False,'new_one_shot_no_retry_or_resume':True,
      'output_cap_bytes':10*2**30,'wall_cap_seconds':72*3600,'actual_invocation_cap':74784,
      'remaining_actual_invocation_cap':74784-len(reservations)}
    save(BUNDLE,'input_reuse_and_prior_budget_v1.json',reuse)
    save(BUNDLE,'user_authority_v1.json',json.loads((CONTROL/'user_revision_authority_v1.json').read_bytes()))
    plan=json.loads((PREP/'signal_compile_plan_v1.json').read_bytes())
    plan.update(run_id=gates.RUN_ID,source_commit=source,source_hashes={**hashes,**parents},source_root=str(ROOT),
      output_root=gates.OUTPUT,requested_workers=12,source_audit_sha256=sha((BUNDLE/'source_freeze_v1.json').read_bytes()),
      reexecution={'manifest_path':gates.REUSE_MANIFEST,'manifest_sha256':sha((BUNDLE/'input_reuse_and_prior_budget_v1.json').read_bytes())})
    allowed=[3,5,6,7,8,9,10,11,12,13,14,15]
    auth={'schema_version':'h4-native-authorization-v1','stage':'signal_compile','run_id':gates.RUN_ID,
          'permission':'signal_compile','one_shot':True,'allowed_cpus':allowed,'result_prior':True,
          'plan_fingerprint':fingerprint('h4-execution-plan-v1',plan)}
    review={'schema_version':'h4-native-stage-review-v1','stage':'signal_compile','run_id':gates.RUN_ID,'approved':True,
            'plan_fingerprint':auth['plan_fingerprint'],'authorization_digest':fingerprint('h4-authorization-v1',auth)}
    save(BUNDLE,'signal_compile_plan_v1.json',plan);save(BUNDLE,'authorization_user_approved_v1.json',auth)
    save(BUNDLE,'stage_review_user_approved_v1.json',review)
    permit=gates.authorize('signal_compile',plan,auth,review,explicit_launch=True);gates.checkout_gate(permit)
    estimate=copy.deepcopy(json.loads((PREP/'stage_storage_estimate_v1.json').read_bytes()))
    estimate.update(schema_version='h4-run03-stage-storage-estimate-v1',source_commit=source,
      prior_cumulative_charge_bytes=charged,prior_wall_seconds=reuse['prior_wall_seconds'],
      max_concurrent_temporary_publishers=14,prior_actual_invocations=len(reservations),
      required_available_bytes=3758096384,required_available_inodes=560000,production_stage_executed=False)
    extra=6*512*1024
    estimate['physical_planning_bound_bytes']+=extra
    estimate['other_physical_components_bytes']['concurrent_temporary_files_extra_allowance']+=extra
    estimate['combined_cumulative_charge_bound']=charged+128+estimate['stage_cumulative_charge_bound']
    estimate['remaining_cumulative_wall_seconds']=72*3600-reuse['prior_wall_seconds']
    estimate['source_hashes_used']={p:h for p,h in hashes.items() if p.startswith('src/')}
    save(BUNDLE,'stage_storage_estimate_v1.json',estimate)
    check={'schema_version':'h4-run03-binding-check-v1','status':'PASS','source_commit':source,
      'source19_blobs_verified':True,'fixed_old_input_source_commit':freeze['source_commit'],
      'plan_sha256':sha((BUNDLE/'signal_compile_plan_v1.json').read_bytes()),'plan_fingerprint':auth['plan_fingerprint'],
      'authorization_digest':review['authorization_digest'],'review_digest':permit.review_digest,
      'CPUs':allowed,'taskset_mask':hex(sum(1<<cpu for cpu in allowed)),'requested_workers':12,
      'old_run_attempts_preserved':True,'input_regeneration':False,'production_launch_before_fresh_gate':False,
      'synthetic_tests':49,'test_attempt_1':{'passed':47,'failed':1,'reason':'pre-existing source-audit assertion pinned earlier parallel source path; updated for run03'},
      'test_attempt_2':{'passed':48,'failed':0},'test_attempt_3':{'passed':49,'failed':0,'errors':0,'skipped':0},
      'new_synthetic_transpiles':0,'source_series_synthetic_transpiles':28,
      'registered_master_seed_and_seed_derivation_unchanged':True,
      'numerical_seeds_are_bound_to_new_SOURCE_COMMIT_as_required_by_existing_identity_algorithm':True}
    save(BUNDLE,'binding_checks_v1.json',check)
    with (BUNDLE/'binding_method_v1.py').open('xb') as f:f.write(Path(__file__).read_bytes())
    save(CONTROL,'fixed_binding_location_v1.json',{'source_commit':source,'bundle':str(BUNDLE),'plan_sha256':check['plan_sha256'],
         'plan_fingerprint':auth['plan_fingerprint'],'review_digest':permit.review_digest,'CPU_list':allowed,'taskset_mask':check['taskset_mask']})
    print(json.dumps({'source_commit':source,'bundle':str(BUNDLE),'tests':49,'CPUs':allowed,'mask':check['taskset_mask'],
        'prior_invocations':len(reservations),'prior_charge':charged,'prior_wall':reuse['prior_wall_seconds'],'plan_sha256':check['plan_sha256'],'binding':'PASS'},sort_keys=True))

if __name__=='__main__':main()
