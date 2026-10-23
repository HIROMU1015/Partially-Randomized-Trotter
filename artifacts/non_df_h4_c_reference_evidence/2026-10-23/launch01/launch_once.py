"""Prepare an approved fixed40 contract, then request exactly one new launch.

Scientific implementation, runtime and old evidence remain at their sealed paths.
This artifact orchestrates the unchanged production grant and native witnesses.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

META=Path(__file__).resolve().parent;ROOT=META.parents[3]
TECH=ROOT.parent/'non-df-h4-c-reference-fix-20261023'
PACKAGE=TECH/'artifacts/non_df_h4_c_reference_fix/2026-10-23'
SEAL=PACKAGE/'SOURCE.json'
NS=META.parent/'run01'
PYTHON=TECH/'.c-reference-fix-venv/bin/python'
RUNNER=TECH/'scripts/run_h4_numerical_evidence.py'
SOURCE='e80d7510880f8db993b777c627bb8e95b05a070d'

def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def baseline(ctl):
    before=ctl.read(PACKAGE/'preservation_before.json')
    paths=dict(before['files'])
    inventory=ctl.read(PACKAGE/'result_inventory.json')
    paths.update({str(TECH/name):digest for name,digest in inventory['files'].items()})
    paths[str(PACKAGE/'result_inventory.json')]=ctl.sha(PACKAGE/'result_inventory.json')
    for name,digest in paths.items():assert ctl.sha(name)==digest,name
    return dict(files=paths,files_verified=len(paths),scope='Original H4/B/investigation/old C failed run and new corrected technical package at original absolute paths')

def prepare(ctl):
    assert Path(sys.executable).resolve()==PYTHON.resolve()
    seal=ctl.read(SEAL);assert seal['SOURCE']==SOURCE
    ctl.verify_seal(seal)
    assert not NS.exists()
    inactive=ctl.read(PACKAGE/'C_execution_contract.json');ctl.validate_contract(inactive)
    assert not any(inactive[k] for k in ('execution_authorized','grant_valid','dispatch_enabled','consumed'))
    tests=ctl.read(META/'preflight_artificial_reference/tests.json')
    assert tests['tests']==14 and tests['failures']==tests['errors']==tests['skips']==0
    preserved=baseline(ctl);ctl.save(META/'preservation_before.json',preserved)
    approval=dict(schema='H4_C_reference_fixed_candidate_execution_approval_record',
        user_messages=[dict(role='user',content='次の作業に進んで大丈夫')],
        preceding_assistant_proposal='ユーザーが修正版での固定40軸実行を承認し、Codexが承認後に新grantで実行する',
        interpretation='Approval of the immediately proposed next stage: one new one-shot fixed40 numerical evidence candidate run with corrected SOURCE',
        assistant_interpretation_distinct_from_verbatim_user_message=True,
        approved_scope=dict(SOURCE=SOURCE,cases=inactive['cases'],epsilon_sig=inactive['epsilon_sig'],axes=inactive['axes'],precision_schedule_bits=inactive['precision_schedule_bits'],call_budget=inactive['total_call_budget_max']),
        excluded=['N/G accounting','formal scientific certificate acceptance','new chemistry or optimized state','sampling or compile','old scientific run retry/resume or old grant reuse','push','detailed scientific review','final evidence freeze'],
        recorded_UTC=ctl.utc())
    ctl.save(META/'approval.json',approval)
    active=dict(inactive,execution_authorized=True,grant_valid=True,dispatch_enabled=True,
        approval_file=str(META/'approval.json'),approval_sha256=ctl.sha(META/'approval.json'),
        output_namespace=str(NS),status='APPROVED_CORRECTED_SOURCE_FIXED40_NEW_ONE_SHOT')
    ctl.validate_contract(active,active=True);ctl.validate_routing(active,seal)
    assert ctl.sha(SEAL)==active['technical_seal_sha256']
    changes={k:dict(before=inactive.get(k),after=active.get(k)) for k in set(inactive)|set(active) if inactive.get(k)!=active.get(k)}
    assert set(changes)=={'execution_authorized','grant_valid','dispatch_enabled','approval_file','approval_sha256','output_namespace','status'}
    ctl.save(META/'active_contract.json',active)
    ctl.save(META/'contract_delta.json',dict(status='PASS_APPROVAL_AND_NAMESPACE_ONLY_NO_SCIENCE_CONDITION_CHANGE',changes=changes))
    stat=os.statvfs(ROOT)
    resources=dict(cpu_affinity=sorted(os.sched_getaffinity(0)),available_bytes=stat.f_bavail*stat.f_frsize,
        memory=[line for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemTotal:','MemFree:','MemAvailable:'))],
        loadavg=Path('/proc/loadavg').read_text().strip(),cpu_pressure=Path('/proc/pressure/cpu').read_text().strip(),memory_pressure=Path('/proc/pressure/memory').read_text().strip())
    result=dict(status='PASS_FIXED_SOURCE_INPUT_RUNTIME_PRESERVATION',SOURCE=SOURCE,
        technical_seal_sha256=ctl.sha(SEAL),UTC=ctl.utc(),new_namespace_unused=True,
        grant_issued=False,scientific_dispatch=0,preserved_files=len(preserved['files']),
        manifest_counts={k:len(seal[k]) for k in ('code_files','runtime_files','legacy_source_files','legacy_runtime_files','input_files')},
        artificial_reference_tests=14,resources=resources,old_original_native_verified=False,
        original_supervisor_cause='UNKNOWN',automatic_retry=False)
    ctl.save(META/'preflight.json',result);return result

def launch(ctl):
    assert Path(sys.executable).resolve()==PYTHON.resolve()
    assert ctl.read(META/'preflight.json')['SOURCE']==SOURCE
    assert baseline(ctl)==ctl.read(META/'preservation_before.json')
    ctl.verify_seal(ctl.read(SEAL))
    assert not NS.exists()
    # Exclusive fuse prevents repeating even this orchestration request.
    ctl.save(META/'DISPATCH_REQUEST.json',dict(UTC=ctl.utc(),SOURCE=SOURCE,one_shot=True,automatic_retry=False))
    grant=ctl.issue_grant(META/'active_contract.json',SEAL,NS)
    command=[str(PYTHON),'-B',str(RUNNER),'--execute',str(NS)]
    witness=[str(PYTHON),'-B',str(RUNNER),'--witness',str(META/'supervisor_native'),'--command-json',json.dumps(command)]
    with (META/'dispatcher_stdout.log').open('xb') as out,(META/'dispatcher_stderr.log').open('xb') as err:
        proc=subprocess.Popen(witness,start_new_session=True,stdin=subprocess.DEVNULL,stdout=out,stderr=err)
    result=dict(UTC=ctl.utc(),grant_id=grant['grant_id'],witness_pid=proc.pid,
        identity=ctl._proc_identity(proc.pid),witness_command=witness,
        launch_request_count=1,scientific_dispatch_proven_by='run01/CONSUMED.json and LAUNCH_CLAIM.json after authorization',
        scope='NEW_CORRECTED_SOURCE_H4_C_CANDIDATE_ONLY',automatic_retry=False,
        orchestration_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    ctl.save(META/'dispatcher.json',result);return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['prepare','launch']);args=parser.parse_args()
    control=module('reference_candidate_control',TECH/'src/trotterlib/h4_certificate_control.py')
    print(json.dumps(prepare(control) if args.mode=='prepare' else launch(control),ensure_ascii=False))
