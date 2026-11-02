"""Create SOURCE-bound newhost review bundle from read-only byte/metadata audits."""
import argparse
import ast
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).absolute().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.resource_applicability.h4_geometry import gates,launch_binding as bind,prelaunch_audit as audit
from trottertracks.resource_applicability.h4_geometry.identity import require,fingerprint

BASE='3e0494e1e8a0c5ea72ef0bdc1cfc9b08a542edcc'
PREVIOUS='artifacts/resource_applicability/track_a_h4_new_server_monitor_fix_a/2026-10-07'
EXTRA=['src/trottertracks/resource_applicability/h4_geometry/launch_binding.py',
       'src/trottertracks/resource_applicability/h4_geometry/prelaunch_audit.py',
       'scripts/resource_applicability/run_h4_prelaunch_tests.py',
       'scripts/resource_applicability/audit_h4_prelaunch_preparation.py',
       'tests/tracks/resource_applicability/test_h4_prelaunch.py',
       'tests/tracks/resource_applicability/h4_prelaunch_minimal_process.py',
       'schemas/h4_newhost_launch_v2.json']


def sha(data):return hashlib.sha256(data).hexdigest()
def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True,ensure_ascii=False);f.write('\n')
def git(*a):return subprocess.check_output(['git','-c','maintenance.auto=false','-c','gc.auto=0','-C',str(ROOT),*a])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source-commit','bundle','evidence','test-attempt'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args();evidence=audit.private_path(args.evidence);bundle=audit.private_path(ROOT/args.bundle)
    require(git('rev-parse',args.source_commit).decode().strip()==args.source_commit and len(args.source_commit)==40,'actual SOURCE SHA')
    bundle.mkdir(parents=True,exist_ok=False)
    previous=json.loads((ROOT/PREVIOUS/'source_freeze_v1.json').read_text())
    paths=sorted(set(previous['closure'])|set(EXTRA));closure={};comparisons={}
    for path in paths:
        data=(ROOT/path).read_bytes();require(data==git('show',args.source_commit+':'+path),'SOURCE byte mismatch')
        if path.endswith('.py'):ast.parse(data,filename=path)
        else:json.loads(data)
        closure[path]=sha(data)
        if path in previous['closure']:
            before=git('show',previous['source_commit']+':'+path)
            comparisons[path]=dict(old_sha256=sha(before),new_sha256=sha(data),
               old_blob=git('rev-parse',previous['source_commit']+':'+path).decode().strip(),
               new_blob=git('rev-parse',args.source_commit+':'+path).decode().strip(),changed=data!=before)
    source=dict(source_commit=args.source_commit,source_hashes=closure,source_root=str(ROOT),source_count=len(paths),
        previous_source_commit=previous['source_commit'],old_new_blobs=comparisons,new_paths=EXTRA,
        change_reason='Newhost approval/profile/source/input/output/observer/budget/one-shot bindings, future own role CPU placement, bounded formats/control/temp log and cached journal; scientific algorithms/templates/options unchanged.',
        old_19_science_source_unchanged_fields='SCF/DF/state/physics/signals/circuit builder/compiler math unchanged; runtime infrastructure only.',
        seed_rebinding_required=True,old_random_partial_results_reused=False)
    write(bundle/'source_freeze_v2.json',source)
    env=json.loads((evidence/'environment_candidate_v2.json').read_text())
    comp=json.loads((evidence/'compiler_candidate_v2.json').read_text())
    require(audit.environment_profile(env['dependencies'],env['installed_sources'])==env,'candidate environment changed during preparation')
    require(audit.compiler_profile(comp['explicit_options'],comp['inherited_defaults'])==comp,'candidate compiler changed')
    write(bundle/'environment_profile_v2.json',env);write(bundle/'compiler_profile_v2.json',comp)
    old=json.loads((ROOT/'artifacts/resource_applicability/track_a_h4_lazy_identity_run05/2026-10-06/source_freeze_v1.json').read_text())
    delta=[]
    for name,before in old['dependency_observations'].items():
        now=env['dependencies'][name];delta.append(dict(name=name,old=before,new=now,
            version_differs=before['version']!=now['version'],normalized_RECORD_differs=before['installed_record_sha256']!=now['normalized_RECORD_sha256'],
            historical_reference_vs_raw_differs=before['installed_record_sha256']!=now['raw_RECORD_sha256']))
    write(bundle/'environment_evaluation_v2.json',dict(candidate_selected=True,production_adoption_approved=False,
        selection_reason='Existing private venv remains untouched; core qiskit/numpy/scipy/openfermion/pyscf versions and installed11 source files match. Freeze its own compiler/resource layer rather than requiring old version equivalence.',
        version_differences=sum(d['version_differs'] for d in delta),normalized_RECORD_differences=sum(d['normalized_RECORD_differs'] for d in delta),
        historical_reference_vs_raw_RECORD_differences=sum(d['historical_reference_vs_raw_differs'] for d in delta),differences=delta,
        limits='Raw old RECORD bytes unavailable; RECORD/source checks are not whole binary/output compiler equivalence. Rustworkx change may affect compile results. No additional transpile/benchmark.',
        old_compiler_equivalence_established=False,install_upgrade_or_configuration_changes=False))
    obs=audit.host_readonly(ROOT,sample_seconds=3);proposal=audit.cpu_proposal(obs)
    write(bundle/'host_resource_readonly_v2.json',obs);write(bundle/'cpu_proposal_v2.json',proposal)
    contract=gates.verify_contract(ROOT);counts=audit.static_invocations(contract['templates']);storage=audit.storage_projection(obs['filesystem']['block_bytes'])
    write(bundle/'static_budget_v2.json',counts);write(bundle/'storage_projection_v2.json',storage)
    oldbinding=json.loads((ROOT/PREVIOUS/'binding_draft_v1.json').read_text());inputs=oldbinding['expected_inputs']['expected_npz']
    incoming='/home/AbeHiromu/projects/h4-input-receipts/20261007/run02'
    expected={v['file']:v['bytes_sha256'] for v in inputs.values()};expected['generation-freeze.json']=bind.EXPECTED_FREEZE
    receipt=audit.receipt_inventory(incoming,expected)
    write(bundle/'input_receipt_v2.json',receipt)
    stop_root=evidence/'received-run05';stop_root.mkdir(mode=0o700,exist_ok=True)
    stop=audit.receipt_inventory(stop_root,{'byte-budget.journal':bind.EXPECTED_JOURNAL,'runner.log':bind.EXPECTED_LOG})
    write(bundle/'stop_receipt_v2.json',dict(**stop,control_complete=False,control_files=[],all_old_owned_processes_ended=False,
       handoff_state_says_ended=True,native_control_receipt_verified=False,
       reason='Native runtime/control proof not received. Do not treat handoff metadata as independently verified stop proof.',
       transfer_outcome='EXISTING_AUTH_FAILED; no credentials/connection information published'))
    tests=json.loads((evidence/args.test_attempt/'test_result_v2.json').read_text())
    monitor=json.loads((evidence/'monitor-regression/attempt-01/test_result_v1.json').read_text())
    require(tests['status']==monitor['status']=='PASS','final artificial suites')
    write(bundle/'tests_v2.json',dict(new_binding_cleanup=tests,monitor_serialization_regression=monitor,
       total=tests['tests']+monitor['tests'],production_equivalence=False,artificial_only=True))
    plan=dict(schema_version=bind.VERSIONS['plan'],stage='signal_compile',run_id=bind.RUN_ID,source_commit=args.source_commit,
       source_root=str(ROOT),source_hashes=closure,
       source_audit=dict(path=str((bundle/'source_freeze_v2.json').relative_to(ROOT)),sha256=sha((bundle/'source_freeze_v2.json').read_bytes())),
       environment_profile=dict(path=str((bundle/'environment_profile_v2.json').relative_to(ROOT)),sha256=sha((bundle/'environment_profile_v2.json').read_bytes())),
       compiler_profile=dict(path=str((bundle/'compiler_profile_v2.json').relative_to(ROOT)),sha256=sha((bundle/'compiler_profile_v2.json').read_bytes())),
       library_cache_profile={},  # unsealed historical preparation; fixed cache required for runtime
       stop_evidence_receipt=dict(path=str((bundle/'stop_receipt_v2.json').relative_to(ROOT)),sha256=sha((bundle/'stop_receipt_v2.json').read_bytes())),
       input_root=incoming,stop_evidence_root=str(stop_root),output_root=str(evidence/bind.RUN_ID),
       control_root=str(evidence/'control'/bind.RUN_ID),inputs=inputs,generation_freeze_digest=oldbinding['expected_inputs']['expected_generation_freeze_fingerprint'],
       templates=contract['templates'],contract_plan_fingerprint=gates.PLAN_FP,compiler_fingerprint=comp['fingerprint'],environment_fingerprint=env['fingerprint'],
       requested_workers=len(proposal['workers']),cpu_proposal={k:proposal[k] for k in ('driver','workers','observer')},carry=bind.CARRY,
       caps=dict(actual_invocations=74784,wall_seconds=259200,output_bytes=10*2**30,driver_AS_RSS=8*2**30,worker_AS_RSS=8*2**30,
          headroom=16*2**30,monitor_seconds=5,observer_AS=256*2**20,observer_RSS=64*2**20),storage=storage,sealed=False)
    p=fingerprint('h4-newhost-plan-v2',plan)
    auth=dict(schema_version=bind.VERSIONS['authorization'],stage='signal_compile',run_id=bind.RUN_ID,source_commit=args.source_commit,
        plan_fingerprint=p,approved=False,runtime_authorization=False,allowed_cpus=[],one_shot=True,permission='signal_compile',result_prior=True,
        environment_accepted=False,observer_role=dict(approved=False,runtime_authorization=False,AS_bytes=256*2**20,RSS_bytes=64*2**20),
        budget_amendment=dict(approved=False,**{'from':74784,'to':74804},authority_reference=''))
    review=dict(schema_version=bind.VERSIONS['review'],stage='signal_compile',run_id=bind.RUN_ID,source_commit=args.source_commit,
       plan_fingerprint=p,authorization_digest=fingerprint('h4-newhost-authorization-v2',auth),approved=False,runtime_authorization=False,reviewer='',mandatory_stop=True)
    bind.structural(plan,auth,review)
    write(bundle/'plan_draft_v2.json',plan);write(bundle/'authorization_draft_v2.json',auth);write(bundle/'review_draft_v2.json',review)
    command='env '+' '.join(k+'='+v for k,v in gates.THREAD_ENV.items())+' '+env['python']+' -P -B '+str(ROOT/'scripts/resource_applicability/run_h4_geometry_signal_compile.py')+' '+' '.join('--'+k+' '+str(bundle/(name+'_draft_v2.json')) for k,name in [('plan','plan'),('authorization','authorization'),('review','review')])+' --explicit-launch-signal-compile'
    write(bundle/'final_approval_proposal_v2.json',dict(source_commit=args.source_commit,approved=False,runtime_authorization=False,allowed_cpus=[],sealed=False,
        proposed_scope='H4 signal/compile map once; stop after complete or fail-closed STOP; no automatic retry, input regeneration or further campaign.',
        candidate_python=env['python'],candidate_environment=env['fingerprint'],candidate_compiler=comp['fingerprint'],CPU_proposal=plan['cpu_proposal'],
        workers=plan['requested_workers'],observer_AS=256*2**20,observer_RSS=64*2**20,admission_bytes=(8+8*plan['requested_workers']+16)*2**30+256*2**20,
        current_actual_cap=74784,requested_minimal_actual_cap=74804,requested_budget_change='Only +20 actual cumulative invocations; retain carry20 and full74784 logical map; all other caps fixed.',
        storage=storage,input_receipt_complete=receipt['complete'],stop_control_receipt_complete=False,
        approval_requires=['candidate environment/compiler adoption and new source/profile seed binding','CPU roles and worker/observer scope',
          'explicit +20 cumulative actual invocation amendment','input6/freeze and native stop/control proof byte receipt',
          'reseal plan and recompute auth/review fingerprints after receipt/cap/permission changes','independent final review','user explicit launch'],
        absolute_launch_command=command,command_executed=False,command_currently_denied_by_false_flags=True,
        fresh_conditions='After full byte/source/profile checks, 3sec passive CPU sample<=20%, distinct approved physical cores/online/scheduler mask, memory and quota/free inode/bytes<=5sec freshness, OOM/pressure stable, new control/output, exclusive one-shot marker, bounded logs and role enforcement.',
        scientific_started=False,observed_JST=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat()))
    records=[]
    for name in (args.test_attempt,'monitor-regression/attempt-01'):
        for path in sorted((evidence/name).rglob('*')):
            if path.is_file():records.append(dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path.read_bytes())))
    write(bundle/'local_artificial_evidence_manifest_v2.json',dict(files=records,raw_logs_not_committed=True,
          prior_failed_attempt_preserved=True,transfer_connection_information_not_committed=True))
    print(json.dumps(dict(source=args.source_commit,closure=len(closure),tests=tests['tests']+monitor['tests'],input_files_received=len(receipt['files']),
        quota=obs['quota']['status'],workers=len(proposal['workers']),physical_GiB=storage['required_bytes']/2**30,
        charged_GiB=storage['cumulative_charge_bound']/2**30,proposed_cumulative_invocations=74804)))


if __name__=='__main__':main()
