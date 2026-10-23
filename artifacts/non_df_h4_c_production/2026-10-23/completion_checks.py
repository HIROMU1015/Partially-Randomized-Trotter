#!/usr/bin/env python3
"""C technical completion only: hashes, structure and inactive CLI rejection.

No real-H4 numerical call or scientific grant issuance. Small native process
examples below run only literal artificial Python programs with no input paths.
"""
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]

def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
c=load('sealed_completion_control',ROOT/'src/trotterlib/h4_certificate_control.py')
a=load('sealed_preservation_checker',HERE/'technical_audit.py')

def checks():
    source=c.read(HERE/'SOURCE.json');contract=c.read(HERE/'C_execution_contract.json')
    c.verify_seal(source);c.validate_contract(contract);c.validate_routing(contract,source)
    assert all(contract[x] is False for x in ['execution_authorized','grant_valid','dispatch_enabled'])
    assert contract['grant_id'] is None and contract['C_production_runner_implemented'] is True
    assert contract['technical_seal_sha256']==c.sha(HERE/'SOURCE.json')
    assert contract['C_production_implementation_SOURCE']==source['SOURCE']
    assert not Path(contract['output_namespace']).exists()
    assert a.preservation()==c.read(HERE/'preservation_before.json')==c.read(HERE/'preservation_after.json')
    tests=c.read(HERE/'tests_release/tests.json');assert tests['tests']==110 and tests['failures']==tests['errors']==tests['skips']==0 and tests['status']=='PASS'
    assert c.read(HERE/'backend13_regression.json')['check_count']==13
    bridge=c.read(HERE/'artificial_bridge_final/verification.json');assert bridge['actual_qubits']==2 and bridge['real_H4_C_calls']==0 and all(x['status']=='TARGET_MET' for x in bridge['cases'])
    assert c.read(HERE/'legacy_original_runtime_verification.json')==dict(checked=3428,mismatches=[],status='PASS')
    current=c.read(ROOT/'artifacts/validation_manifest.json');entry=current.pop('non_df_h4_c_production_20261023')
    original=json.loads(subprocess.check_output(['git','show',source['SOURCE']+':artifacts/validation_manifest.json'],cwd=ROOT));assert current==original
    assert entry['SOURCE']==source['SOURCE'] and entry['scientific_grants_issued']==entry['real_H4_C_calls']==0
    expected={(case,eps,axis) for case in contract['cases'] for eps in contract['epsilon_sig'] for axis in contract['axes']}
    mapping=c.read(contract['method_map_path'])['rows']
    assert len(mapping)==40 and {(r['case'],r['epsilon_sig'],r['axis']) for r in mapping}==expected
    assert all(r['current_N'] is None and r['current_G'] is None and r['current_status']=='UNDETERMINED' for r in mapping)
    links=[]
    for path in [ROOT/'docs/non_df_h4_c_production_20261023.md',HERE/'README.md',HERE/'CONTINUATION.md',ROOT/'docs/research/研究ノート/2026-10-23_H4_C_production.md']:
        for url in re.findall(r'\]\(([^)]+)\)',path.read_text()):
            if url.startswith(('https://','http://','#')):continue
            linked=(path.parent/url.split('#')[0]).resolve()
            assert linked.exists() or linked in [HERE/'completion_audit.json',HERE/'STOP.json'],(path,url)
            links.append(str(linked.relative_to(ROOT)))
    return dict(status='PASS_C_TECHNICAL_SOURCE_CONTRACT_AND_PRESERVATION',UTC=datetime.now(timezone.utc).isoformat(),
        SOURCE=source['SOURCE'],branch='codex/non-df-h4-c-production-20261023',code_files=len(source['code_files']),runtime_files=len(source['runtime_files']),legacy_runtime_files=len(source['legacy_runtime_files']),input_files=len(source['input_files']),
        old_SOURCE_files=738,old_result_claim_files=1147,old_investigation_files=17,BC_inventory_files=99,old_runtime_hashes_verified=3428,
        unit_and_regression_tests=110,failures=0,errors=0,skips=0,backend_checks=13,artificial_bridge_axes=40,artificial_bridge_qubits=2,
        real_H4_C_calls=0,real_H4_B_repeated=0,scientific_grants_issued=0,new_N_G_accounting=0,
        old_N_G='NULL_UNDETERMINED',original_H4_native_verified=False,original_supervisor_cause='UNKNOWN',
        execution_authorized=False,grant_valid=False,dispatch_enabled=False,preservation_unchanged=True,
        literal_compiled_wrapper_transfer_certified=False,formal_certificate_accepted=False,checked_local_links=links,
        push_performed=False,final_scientific_evidence_freeze=False,mandatory_STOP=True)

def record():
    cli=ROOT/'scripts/run_h4_numerical_evidence.py';contract=c.read(HERE/'C_execution_contract.json')
    fixture=HERE/'negative_authority_fixture';fixture.mkdir();c.save(fixture/'contract.json',contract)
    tests=[['--issue-grant',str(HERE/'C_execution_contract.json'),'--seal',str(HERE/'SOURCE.json'),'--namespace',contract['output_namespace']],
           ['--execute',str(fixture)],['--worker',str(fixture),'--representation','pauli_jw'],['--bridge',str(fixture),'--representation','df_native']]
    records=[]
    for args in tests:
        p=subprocess.run([sys.executable,'-B',str(cli),*args],capture_output=True,text=True)
        assert p.returncode!=0 and 'C_EXECUTION_NOT_AUTHORIZED' in p.stderr
        records.append(dict(arguments=args,exit_code=p.returncode,stderr=p.stderr,expected='C_EXECUTION_NOT_AUTHORIZED'))
    assert not Path(contract['output_namespace']).exists() and not (fixture/'grant.json').exists() and not (fixture/'CONSUMED.json').exists()
    c.save(HERE/'launch_rejection.json',dict(status='PASS_DISABLED_ACTUAL_CLI',attempts=records,real_H4_C_calls=0,science_grants_issued=0))
    native=[]
    for name,program,expected in [('native_artificial_pass','pass',0),('native_artificial_failure','raise SystemExit(7)',7)]:
        directory=HERE/name
        p=subprocess.run([sys.executable,'-B',str(cli),'--witness',str(directory),'--command-json',json.dumps([sys.executable,'-B','-c',program])],capture_output=True,text=True)
        native_exit=c.read(directory/'worker_native_exit.json');stop=c.read(directory/'STOP.json')
        assert native_exit['exit_code']==expected and native_exit['reaped'] and stop['all_owned_reaped']
        c.save(directory/'ARTIFICIAL_SCOPE.json',dict(scope='Artificial literal Python program only; no scientific input or grant',program=program,real_H4_C_calls=0))
        native.append(dict(example=name,witness_exit=p.returncode,worker_native_exit=expected,reaped=True))
    c.save(HERE/'native_artificial_verification.json',dict(status='PASS_ARTIFICIAL_NATIVE_RECORDS',examples=native,real_H4_C_calls=0,original_H4_native_verified=False))
    audit=checks();c.save(HERE/'completion_audit.json',audit)
    c.save(HERE/'STOP.json',dict(status='C_PRODUCTION_TECHNICAL_COMPLETE_MANDATORY_STOP',UTC=c.utc(),SOURCE=audit['SOURCE'],scope='NEW_C_IMPLEMENTATION_AND_ARTIFICIAL_TESTS_ONLY',execution_authorized=False,grant_valid=False,dispatch_enabled=False,scientific_grants_issued=0,real_H4_C_calls=0,new_N_G_accounting=0,original_H4_native_verified=False,original_supervisor_cause='UNKNOWN',automatic_retry=False,next_science_requires_explicit_user_approval=True,push_performed=False))
    return audit

if __name__=='__main__':
    result=record() if '--record' in sys.argv else checks()
    print(json.dumps({k:result[k] for k in ['status','SOURCE','unit_and_regression_tests','backend_checks','real_H4_C_calls','scientific_grants_issued','preservation_unchanged','mandatory_STOP']},sort_keys=True))
