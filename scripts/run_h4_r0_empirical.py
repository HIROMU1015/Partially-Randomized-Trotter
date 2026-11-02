#!/usr/bin/env python3
"""R0 one-shot: structural inspection, source seal, five reserved state actions."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types

ROOT=Path(__file__).resolve().parents[1]
pkg=types.ModuleType('r0_private');pkg.__path__=[str(ROOT/'src/trotterlib')];sys.modules['r0_private']=pkg
from r0_private import h4_r0_empirical as A
C=A.C;BASE=A.BASE
PYTHON='/home/AbeHiromu/projects/h6-environments/20261012/native-dependency-v2/venv/bin/python'
CODE=['src/trotterlib/'+n+'.py' for n in ('h4_r0_empirical','h4_central_rho_plan','h4_conditional_cng',
      'h4_d2_pauli_control','h4_certificate_control','h4_ball_evidence','h4_enclosure_prototype',
      'h4_a_axes','fixed_signal_evaluation','rte_compiled_cost','h2_non_df_pilot','rte')]+[
      'artifacts/non_df_h4_d1_reproducibility_design/2026-10-24/inspect_saved_evidence.py',
      'artifacts/non_df_h4_central_r0_empirical/2026-10-24/prepare_metadata.py',
      'scripts/run_h4_r0_empirical.py','tests/test_h4_r0_empirical.py','tests/test_h4_central_rho_plan.py']
SCHEMA=dict(version='R0_BINARY64_EMPIRICAL_V1',keys=list(A.KEYS),logical_diagnostics=2,initial_maximum_state_actions=5,
    axis='cosine',required=['m_C','m_input','m_output','delta','eta','d','rho_budget_exact','d_over_budget','budget_minus_d'],
    status='EMPIRICAL_ONLY_NO_RHO_UPPER',rho_upper=None,numerical_error_upper=None,PR_law_rho=None,formal_certificate_accepted=False)
FORBIDDEN=dict(new_compile=0,new_circuit_build=0,sampling=0,RNG_replay=0,Hamiltonian_state_generation=0,
               C_L_full_runs=0,N_G_reanalysis=0,Arb_interval=0,precision_escalation=0,sine=0,R1=0,R2=0,certificate_acceptance=0)


def verify(seal):
    C.require(seal['kind']=='R0_BINARY64_SOURCE_INPUT_RUNTIME_SEAL_V1','SEAL_KIND')
    C.require(C.sha(sys.executable)==seal['python_sha256'],'PYTHON')
    for p,h in seal['code_files'].items():
        C.require(C.sha(ROOT/p)==h,'CODE:'+p)
        data=subprocess.check_output(['git','show',seal['SOURCE']+':'+p],cwd=ROOT)
        C.require(hashlib.sha256(data).hexdigest()==h,'COMMITTED_SOURCE:'+p)
    for p,h in seal['bound_manifests'].items():C.require(C.sha(BASE/p)==h,'MANIFEST:'+p)
    C.hashes(C.read(BASE/'runtime.json')['files'])
    C.hashes({str(ROOT/k):v for k,v in C.read(BASE/'inputs.json')['files'].items()})
    C.hashes({str(ROOT/k):v for k,v in C.read(BASE/'preservation_before.json').items()})
    C.require(C.read(BASE/'output_schema.json')==SCHEMA,'OUTPUT_SCHEMA')


def prepare():
    C.require(not (BASE/'seal.json').exists() and not (BASE/'run01').exists(),'ALREADY_SEALED')
    C.require(Path(sys.executable)==Path(PYTHON),'FIXED_POINT_ENVIRONMENT')
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    code={p:C.sha(ROOT/p) for p in CODE}
    for p in code:C.require(subprocess.check_output(['git','show',source+':'+p],cwd=ROOT)==(ROOT/p).read_bytes(),'UNCOMMITTED_CODE')
    prior=C.read(ROOT/'artifacts/non_df_h4_a_axes/2026-10-24/seal.json')
    runtime_path=ROOT/prior['runtime_manifest']
    C.require(C.sha(runtime_path)==prior['runtime_manifest_sha256'],'PRIOR_RUNTIME_MANIFEST')
    runtime=C.read(runtime_path)
    C.hashes(runtime['files']);C.save(BASE/'runtime.json',runtime)
    C.save(BASE/'output_schema.json',SCHEMA)
    bound=['approval.txt','github_verification.json','inputs.json','preservation_before.json','runtime.json','output_schema.json']
    seal=dict(kind='R0_BINARY64_SOURCE_INPUT_RUNTIME_SEAL_V1',SOURCE=source,code_files=code,python_sha256=C.sha(sys.executable),
              bound_manifests={p:C.sha(BASE/p) for p in bound},real_state_actions_before_seal=0,UTC=C.utc())
    C.save(BASE/'seal.json',seal)
    contract=dict(kind='R0_BINARY64_EMPIRICAL_ACTIVE_V1',SOURCE=source,base_commit='2734b97c2dd5f017e93cd81619ff52a6a0c6f16b',
        execution_authorized=True,dispatch_enabled=True,grant_valid=True,keys=list(A.KEYS),axis='cosine',
        limits=A.LIMITS,state_action_limits=A.ACTION_LIMITS,max_initial_state_actions=5,logical_diagnostics=2,
        fixed='H4 linear1A STO-3G n8+ancilla8 Pauli184/prefix27 S2 T.8 q8 r4 K4; D2 Qiskit1.3 rz/sx/x/cx opt1 seed17 no routing/backend/layout',
        scientific_SOURCE=dict(original='0c329b30a81accb35c212be8defa66fee0dd0f12',C='e80d7510880f8db993b777c627bb8e95b05a070d',
            D2='74515d8eb4f073c40fc783681a3775dbca89e9b1',A_AXES='a37232f85812688c3b9ab957c5073c5ce143b17e',CNG='d96dfe5ca0a6c950a5f1791ff61659e6dc8f1c34'),
        forbidden_calls=FORBIDDEN,schema=SCHEMA,seal_sha256=C.sha(BASE/'seal.json'),approval_sha256=C.sha(BASE/'approval.txt'),
        numerical_policy='binary64 empirical state actions only; saved PF interval scalar conversion and exact residual/budget accounting, no new state/error enclosure',
        mandatory_STOP=True,automatic_retry=False,automatic_next_stage=False,formal_certificate_accepted=False,
        technical_recovery='diagnosed technical only; reuse successes, preserve failed calls/grant/STOP; code changes need new SOURCE/seal/unused grant/namespace',
        resources='five short 256/512-amplitude program walks; one worker avoids repeated parsing; no CPU/RAM/thread/affinity ceiling or shared changes')
    C.save(BASE/'active_contract.json',contract);verify(seal)
    ns=BASE/'run01';ns.mkdir()
    grant=dict(kind='R0_BINARY64_ONE_SHOT',namespace=str(ns),contract_sha256=C.sha(BASE/'active_contract.json'),
               seal_sha256=C.sha(BASE/'seal.json'),approval_sha256=C.sha(BASE/'approval.txt'),one_shot=True)
    grant['grant_id']=C.digest(grant);C.save(ns/'grant.json',grant)
    print(json.dumps(dict(status='SEALED_NOT_EXECUTED',SOURCE=source,grant_id=grant['grant_id'])))


def check(consume=False):
    ns=BASE/'run01';c=C.read(BASE/'active_contract.json');s=C.read(BASE/'seal.json');g=C.read(ns/'grant.json')
    C.require(c['kind']=='R0_BINARY64_EMPIRICAL_ACTIVE_V1' and all(c[k] is True for k in ('execution_authorized','dispatch_enabled','grant_valid')),'AUTHORITY')
    C.require(c['SOURCE']==s['SOURCE'] and c['keys']==list(A.KEYS) and c['axis']=='cosine','SOURCE_TARGET_SCOPE')
    C.require(c['limits']==A.LIMITS and c['state_action_limits']==A.ACTION_LIMITS and c['schema']==SCHEMA and c['forbidden_calls']==FORBIDDEN,'FINITE_SCOPE')
    C.require(c['max_initial_state_actions']==5 and c['logical_diagnostics']==2 and c['mandatory_STOP'] and not c['automatic_retry'] and not c['automatic_next_stage'] and not c['formal_certificate_accepted'],'STOP_NO_CERTIFICATE')
    C.require(g['kind']=='R0_BINARY64_ONE_SHOT' and g['namespace']==str(ns) and g['one_shot'],'GRANT_SCOPE')
    C.require(g['grant_id']==C.digest({k:v for k,v in g.items() if k!='grant_id'}),'GRANT_DIGEST')
    C.require(g['contract_sha256']==C.sha(BASE/'active_contract.json') and g['seal_sha256']==C.sha(BASE/'seal.json')==c['seal_sha256'],'GRANT_BINDING')
    C.require(g['approval_sha256']==c['approval_sha256']==s['bound_manifests']['approval.txt'],'APPROVAL_BINDING')
    verify(s);C.require(not (ns/'STOP.json').exists(),'ALREADY_STOPPED')
    if consume:C.save(ns/'CONSUMED.json',dict(grant_id=g['grant_id'],UTC=C.utc(),nonrefundable=True))
    else:C.require(C.read(ns/'CONSUMED.json')['grant_id']==g['grant_id'],'NOT_CONSUMED')
    return c,s


def worker():
    c,s=check();ns=BASE/'run01';C.require((ns/'LAUNCH_CLAIM.json').is_file(),'NO_LAUNCH')
    C.save(ns/'WORKER_CLAIM.json',dict(pid=os.getpid(),UTC=C.utc()))
    A.run(ns,c,C.CallBudget(ns,A.LIMITS));verify(s)


def launch():
    c,s=check(consume=True);ns=BASE/'run01';C.save(ns/'LAUNCH_CLAIM.json',dict(pid=os.getpid(),UTC=C.utc()))
    tree=ast.parse((ROOT/'src/trotterlib/h4_certificate_control.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='witness')
    for n in ast.walk(node):
        if isinstance(n,ast.Constant) and n.value=='NEW_C_WORKER_ONLY':n.value='NEW_R0_BINARY64_EMPIRICAL_ONLY'
    env=dict(vars(C));exec(compile(ast.Module(body=[node],type_ignores=[]),'<reused R0 native witness>','exec'),env)
    status=env['witness'](ns/'native',[sys.executable,'-B',str(Path(__file__).resolve()),'--worker'])
    error=None
    try:
        if status==0:
            verify(s);C.require(C.read(ns/'call_ledger.json')['used']==A.LIMITS,'INCOMPLETE_CALLS')
            C.require(C.read(ns/'result_summary.json')['logical_diagnostics']==2,'INCOMPLETE_DIAGNOSTICS')
    except Exception as e:status=1;error=str(e)
    C.save(ns/'STOP.json',dict(status='COMPLETE_MANDATORY_STOP' if status==0 else 'FAILED_NO_AUTOMATIC_RETRY',
        native_exit=C.read(ns/'native/worker_native_exit.json'),all_owned_reaped=C.read(ns/'native/STOP.json')['all_owned_reaped'],
        validation_error=error,original_H4_native_verified=False,automatic_next_stage=False,UTC=C.utc()))
    return status


if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    for name in ('inspect','prepare','launch','worker'):g.add_argument('--'+name,action='store_true')
    args=p.parse_args()
    if args.inspect:
        *_,result=A.preflight();C.save(BASE/'structural_preflight.json',result);print(json.dumps(dict(status='PASS',state_actions=0)))
    elif args.prepare:prepare()
    elif args.launch:sys.exit(launch())
    else:worker()
