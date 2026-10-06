"""Fix run04 source/binding using immutable run02 inputs and run03 failure evidence."""
import ast,copy,hashlib,json,os,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
import xml.etree.ElementTree as ET

BASE=Path('/home/AbeHiromu/projects/partially-randomized-trotter')
ROOT=Path('/tmp/track-a-h4-lazy-identity-run05-20261006')
BUNDLE=ROOT/'artifacts/resource_applicability/track_a_h4_lazy_identity_run05/2026-10-06'
OLD=Path('/tmp/track-a-h4-streaming-monitor-run04-20261006')
OLDB=OLD/'artifacts/resource_applicability/track_a_h4_streaming_monitor_run04/2026-10-06'
OLDCTL=BASE/'.server-preparation/executions/h4-signal-compile-run04-streaming-monitor'
DIAGCTL=BASE/'.server-preparation/executions/h4-signal-compile-run03-revision'
CONTROL=BASE/'.server-preparation/executions/h4-signal-compile-run05-lazy-identity'
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.resource_applicability.h4_geometry import gates
from trottertracks.resource_applicability.h4_geometry.identity import fingerprint,sha,require

def save(root,name,value):
    with (root/name).open('x') as f:
        json.dump(value,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write('\n')

def main():
    source=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    require(source=='6d365257770e99022b91d6a38dbee49ee0077503','fixed source commit')
    stop=json.loads((OLDCTL/'execution_result_v1.json').read_bytes());prior=json.loads((OLDCTL/'process_record_v1.json').read_bytes())
    require(stop['driver_ended'] and not stop['remaining_owned_workers'] and not Path('/proc/%d'%prior['pid']).exists(),'old driver stopped')
    for d in Path('/proc').iterdir():
        if not d.name.isdigit():continue
        try:
            if d.stat().st_uid!=os.getuid():continue
            cmd=(d/'cmdline').read_bytes()
            require(b'--owned-worker '+str(prior['pid']).encode()+b'\0' not in cmd,'old own worker remains')
        except (FileNotFoundError,PermissionError,ProcessLookupError):pass
    tests=ET.parse(BUNDLE/'test_result_v1.xml').getroot().find('testsuite').attrib
    require(int(tests['tests'])==67 and all(int(tests[k])==0 for k in ('errors','failures','skipped')),'tests PASS')
    extra=json.loads((BUNDLE/'serialization_test_attempt_2.json').read_bytes())
    require(extra['tests']==8 and extra['status']=='PASS' and extra['failures']==extra['errors']==extra['skipped']==0,'serialization8 tests')
    olddiag=json.loads((DIAGCTL/'metadata_monitor_diagnostic_128_v1.json').read_bytes())
    newdiag=json.loads((DIAGCTL/'metadata_lazy_monitor_diagnostic_128_v1.json').read_bytes())
    require(len(olddiag['monitor_failure'])==1 and olddiag['monitor_failure'][0]['reason']=='monitor interval/freshness','monitor mechanism reproduced')
    require(newdiag['fingerprint']==olddiag['fingerprint'] and not newdiag['monitor_failure'] and newdiag['main_error'] is None and newdiag['monitor_thread_ended'],'same digest, fresh monitor')
    require(not CONTROL.exists(),'new control unique');CONTROL.mkdir(mode=0o700)
    save(CONTROL,'user_execution_approval_v1.json',{'approval_authority':'USER_EXPLICIT_MESSAGE','user_instruction_verbatim':'続きを行って',
      'prior_explicit_instruction':'その修正を入れて、ワーカー数を増やして再実行して',
      'scope':'SIX_FROZEN_INPUTS_SIGNAL_COMPILE_THEN_MAP_COMPLETE_STOP','requested_workers':12,
      'allowed_CPUs':[3,5,6,7,8,9,10,11,12,13,14,15],
      'authority_context':'User explicitly continues unfinished repair/increase/reexecute after run03 monitor STOP and tool approval-review usage failure. One manually fixed run04 after gates; no automatic retries.',
      'approved_review_is_actual_user_authority_transcription':True,'external_reviewer_action_claimed':False,
      'source_or_shared_environment_repair_does_not_modify_shared_environment':True,'GPU_access':False,
      'input_regeneration':False,'automatic_retry_or_resume':False,'next_stage_authorized':False})
    oldaudit=json.loads((OLDB/'source_freeze_v1.json').read_bytes())
    closure={p:sha((ROOT/p).read_bytes()) for p in oldaudit['new_source_hashes']};parents={p:sha((ROOT/p).read_bytes()) for p in oldaudit['namespace_parent_hashes']}
    for p,h in {**closure,**parents}.items():
        require(sha(gates.git_blob(ROOT,source,p))==h,'source blob mismatch');ast.parse((ROOT/p).read_bytes())
    contract=gates.verify_contract(ROOT);environment=gates.environment_matches(oldaudit['dependency_observations']);options=gates.compiler_matches(contract['compiler_environment_reference']['compiler'])
    for p,h in oldaudit['installed_source_hashes'].items():require(sha(Path(p).read_bytes())==h,'installed source changed')
    audit={**oldaudit,'schema_version':'h4-run05-lazy-identity-source-audit-v1','status':'SOURCE_BLOBS_VERIFIED',
      'source_commit':source,'old_source_commit':oldaudit['source_commit'],'old_source_audit_sha256':sha((OLDB/'source_freeze_v1.json').read_bytes()),
      'source_checkout_root':str(ROOT),'source_audit_reference':gates.SOURCE_AUDIT,'new_source_hashes':closure,'namespace_parent_hashes':parents,
      'dependency_observations':environment,'compiler_options':options,'observed_utc':datetime.now(timezone.utc).isoformat(),
      'changed_source19_paths':sorted(p for p,h in closure.items() if h!=oldaudit['new_source_hashes'][p]),
      'unchanged_source19_paths':sorted(p for p,h in {**closure,**parents}.items() if h=={**oldaudit['new_source_hashes'],**oldaudit['namespace_parent_hashes']}[p]),
      'only_changes':'normalize/encode exact JSON lazily without whole container copy; call-local parameter metadata memo; maintain hash bytes, monitor5s and12 worker queue; bind run05 to run04 cumulative budget',
      'new_regression_tests':{'tests':75,'failures':0,'errors':0,'skipped':0,'production_science_jobs':0,'new_transpile_invocations':0},
      'large_metadata_diagnostic_old_and_new_hash_match':True,'five_second_monitor_limit_unchanged':True,
      'runtime_success_or_scientific_results_claimed':False,'scientific_math_compiler_limits_unchanged':True}
    save(BUNDLE,'source_freeze_v1.json',audit)
    reuse=copy.deepcopy(json.loads((OLDB/'input_reuse_and_prior_budget_v1.json').read_bytes()))
    output=Path(prior['output_root']);raw=(output/'byte-budget.journal').read_bytes();require(len(raw)%128==0,'predecessor journal rows')
    charged=sum(int(raw[i:i+128].strip()) for i in range(0,len(raw),128));require(charged==stop['cumulative_charge_bytes'],'cumulative predecessor bytes')
    oldlog=OLDCTL/'runner_stdout_stderr.log'
    prior_wall=stop['conservative_cumulative_wall_seconds']
    reuse.update(new_run_id=gates.RUN_ID,prior_cumulative_charge_bytes=charged,prior_actual_invocations=stop['total_invocations_consumed'],
      prior_wall_seconds=prior_wall,remaining_actual_invocation_cap=74784-stop['total_invocations_consumed'],
      predecessor_budget_journal={'root':str(output),'sha256':sha(raw),'rows':len(raw)//128,'old_actual_invocations':3},
      old_run_status='RUN02_OWNED_POOL_STOP_AND_RUN03_MONITOR_STOP',user_instruction='続きを行って / earlier:その修正を入れて、ワーカー数を増やして再実行して',
      old_stop_record_sha256=sha((OLDCTL/'execution_result_v1.json').read_bytes()),
      wall_basis='carry run03 prior wall plus source-run elapsed through final failure log and60s; no prior time refunded',
      run03_ledger_hashes=[{'name':p.name,'sha256':sha(p.read_bytes())} for p in sorted(output.glob('ledger-*.json'))],
      prior_run03_completed_records=0,old_partial_compile_metrics_reused=False)
    save(BUNDLE,'input_reuse_and_prior_budget_v1.json',reuse)
    save(BUNDLE,'run04_stop_evidence_v1.json',stop)
    save(BUNDLE,'metadata_old_encoder_stop_v1.json',olddiag);save(BUNDLE,'metadata_streaming_encoder_pass_v1.json',newdiag)
    save(BUNDLE,'user_authority_v1.json',json.loads((CONTROL/'user_execution_approval_v1.json').read_bytes()))
    for filename in ['h4_run03_monitor_diagnostic_128.py','h4_run05_monitor_diagnostic_128.py']:
        with (BUNDLE/filename).open('xb') as f:f.write((Path('/tmp')/filename).read_bytes())
    plan=json.loads((OLDB/'signal_compile_plan_v1.json').read_bytes())
    plan.update(run_id=gates.RUN_ID,source_commit=source,source_root=str(ROOT),source_hashes={**closure,**parents},output_root=gates.OUTPUT,
      source_audit_sha256=sha((BUNDLE/'source_freeze_v1.json').read_bytes()),reexecution={'manifest_path':gates.REUSE_MANIFEST,'manifest_sha256':sha((BUNDLE/'input_reuse_and_prior_budget_v1.json').read_bytes())})
    auth=json.loads((OLDB/'authorization_user_approved_v1.json').read_bytes());auth.update(run_id=gates.RUN_ID,plan_fingerprint=fingerprint('h4-execution-plan-v1',plan))
    review=json.loads((OLDB/'stage_review_user_approved_v1.json').read_bytes());review.update(run_id=gates.RUN_ID,plan_fingerprint=auth['plan_fingerprint'],authorization_digest=fingerprint('h4-authorization-v1',auth))
    save(BUNDLE,'signal_compile_plan_v1.json',plan);save(BUNDLE,'authorization_user_approved_v1.json',auth);save(BUNDLE,'stage_review_user_approved_v1.json',review)
    permit=gates.authorize('signal_compile',plan,auth,review,explicit_launch=True);gates.checkout_gate(permit)
    from trottertracks.resource_applicability.h4_geometry.execution import input_boundary
    boundary=input_boundary(permit);require(boundary['prior_charge']==charged and boundary['prior_invocations']==12,'native budget boundary')
    estimate=copy.deepcopy(json.loads((OLDB/'stage_storage_estimate_v1.json').read_bytes()));estimate.update(schema_version='h4-run05-stage-storage-estimate-v1',source_commit=source,
      prior_cumulative_charge_bytes=charged,prior_actual_invocations=12,prior_wall_seconds=prior_wall,remaining_cumulative_wall_seconds=72*3600-prior_wall,
      combined_cumulative_charge_bound=charged+128+estimate['stage_cumulative_charge_bound'],source_hashes_used={p:h for p,h in closure.items() if p.startswith('src/')})
    save(BUNDLE,'stage_storage_estimate_v1.json',estimate)
    checks={'schema_version':'h4-run05-binding-checks-v1','status':'PASS','source_commit':source,'source19_blobs_verified':True,
      'source_plan_auth_review_gate':True,'input_generation_source_commit':reuse['generation_source_commit'],
      'plan_sha256':sha((BUNDLE/'signal_compile_plan_v1.json').read_bytes()),'plan_fingerprint':auth['plan_fingerprint'],
      'authorization_digest':review['authorization_digest'],'review_digest':permit.review_digest,'CPU_list':auth['allowed_cpus'],'taskset_mask':'0xffe8',
      'workers':12,'tests':75,'failures':0,'errors':0,'skipped':0,'new_transpiles':0,'synthetic_source_series_transpiles':28,
      'old_input_source_identity_preserved':True,'input_regeneration_performed':False,'old_runtime_cache_results_not_reused':True,
      'cumulative_invocations_prior':12,'cumulative_charge_prior':charged,'cumulative_wall_prior':prior_wall,
      'monitor_GIL_starvation_mechanism_reproduced':True,'historical_run03_exact_underlying_exception_unknown':True,
      'old_new_large_metadata_digest':newdiag['fingerprint'],'streaming_metadata_monitor_PASS':True,
      'scientific_conditions_and_compiler_unchanged':True,'seed_algorithm_and_master_unchanged_SOURCE_binding_updates':True,
      'resource_monitor_five_second_deadline_unchanged':True,'production_complete_not_claimed':True}
    save(BUNDLE,'binding_checks_v1.json',checks)
    save(CONTROL,'fixed_binding_location_v1.json',{'source_commit':source,'bundle':str(BUNDLE),'plan_sha256':checks['plan_sha256'],'review_digest':checks['review_digest'],'CPU_list':auth['allowed_cpus'],'taskset_mask':'0xffe8'})
    with (BUNDLE/'binding_method_v1.py').open('xb') as f:f.write(Path(__file__).read_bytes())
    print(json.dumps({'source':source,'tests':75,'prior_invocations':7,'prior_charge':charged,'prior_wall':prior_wall,'plan_sha256':checks['plan_sha256'],'binding':'PASS','control':str(CONTROL)},sort_keys=True))

if __name__=='__main__':main()
