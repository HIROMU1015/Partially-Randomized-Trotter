#!/usr/bin/env python3
"""PR0 cosine one-shot controller derived from the existing PF controller; reuse C audit and call reservations."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types
import runpy

ROOT=Path(__file__).resolve().parents[1]
pkg=types.ModuleType('arb_pr0_run_private');pkg.__path__=[str(ROOT/'src/trotterlib')];sys.modules['arb_pr0_run_private']=pkg
from arb_pr0_run_private import h4_arb_pr0_cosine as A
C,B,BASE=A.C,A.B,A.BASE
PYTHON='/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/non-df-h4-c-reference-fix-20261023/.c-reference-fix-venv/bin/python'
CODE=['src/trotterlib/'+n+'.py' for n in ('h4_arb_bridge','h4_arb_pr0_cosine','h4_arb_pr0_preparation','h4_arb_pf_cosine',
    'h4_ball_evidence','h4_certificate_control','h4_r0_empirical','h4_central_rho_plan','h4_conditional_cng',
    'h4_d2_pauli_control','h4_a_axes','rte_compiled_cost','h2_non_df_pilot','rte','rte_event_audit')]+[
    'scripts/run_h4_arb_pr0_cosine.py','scripts/inspect_h4_arb_pr0.py',
    'tests/test_h4_arb_pr0_cosine.py','tests/test_h4_arb_pr0_preparation.py','tests/test_h4_arb_bridge.py','tests/test_h4_arb_qpy.py',
    'artifacts/non_df_h4_d1_reproducibility_design/2026-10-24/inspect_saved_evidence.py',
    'artifacts/non_df_h4_central_arb_pr0_cosine/2026-10-25/prepare_metadata.py']


def resources():
    names=['/proc/meminfo','/proc/loadavg','/proc/self/cgroup','/proc/pressure/cpu','/proc/pressure/memory',
           '/proc/self/status','/sys/fs/cgroup/memory/memory.limit_in_bytes','/sys/fs/cgroup/memory/memory.usage_in_bytes',
           '/sys/fs/cgroup/memory/memory.failcnt']
    return dict(UTC=C.utc(),files={p:Path(p).read_text() for p in names if Path(p).is_file()},
        allowed_CPUs=sorted(os.sched_getaffinity(0)),observation_only=True,resource_ceiling_imposed=False)


def runtime():
    import ctypes
    import flint
    old=C.read(ROOT/'artifacts/non_df_h4_c_reference_fix/2026-10-23/runtime_manifest.json')
    paths={Path(p).resolve() for p in old['files'] if '/flint/' in p or '/python_flint.libs/' in p}
    paths.add(Path(sys.executable).resolve())
    for mod in list(sys.modules.values()):
        name=getattr(mod,'__file__',None)
        if name and Path(name).is_file() and not Path(name).resolve().is_relative_to(ROOT):paths.add(Path(name).resolve())
    for line in Path('/proc/self/maps').read_text().splitlines():
        parts=line.split(maxsplit=5)
        if len(parts)==6 and parts[5].startswith('/') and Path(parts[5]).is_file():paths.add(Path(parts[5]).resolve())
    files={str(p):C.sha(p) for p in sorted(paths)}
    common={p for p in files if p in old['files']}
    differences={p:dict(old_sha256=old['files'][p],current_sha256=files[p]) for p in sorted(common) if files[p]!=old['files'][p]}
    numeric={p for p in common if '/flint/' in p or '/python_flint.libs/' in p}
    C.require(all(files[p]==old['files'][p] for p in numeric),'REUSED_NUMERICAL_BACKEND_CHANGED')
    C.require(flint.__version__=='0.9.0','FLINT_BINDING_VERSION')
    return dict(files=files,python=sys.version,python_flint=flint.__version__,FLINT=old['FLINT'],
        executable=PYTHON,reused_manifest='artifacts/non_df_h4_c_reference_fix/2026-10-23/runtime_manifest.json',
        reused_unchanged_files=len(common)-len(differences),reused_numeric_files=len(numeric),prior_runtime_differences=differences,
        difference_policy='observe inherited system differences; new runtime sealed before any grant; numeric FLINT package must match; no shared files changed',
        scope='loaded stdlib/shared libraries and existing complete flint binary package; unused numpy excluded')


def verify(seal):
    C.require(seal['kind']=='H4_PR0_COSINE_ARB_SOURCE_INPUT_RUNTIME_SEAL_V1','SEAL_KIND')
    C.require(C.sha(sys.executable)==seal['python_sha256'],'PYTHON_EXECUTABLE')
    for p,h in seal['code_files'].items():
        C.require(C.sha(ROOT/p)==h,'CODE:'+p)
        data=subprocess.check_output(['git','show',seal['SOURCE']+':'+p],cwd=ROOT)
        C.require(hashlib.sha256(data).hexdigest()==h,'COMMITTED_SOURCE:'+p)
    for p,h in seal['bound_manifests'].items():C.require(C.sha(BASE/p)==h,'MANIFEST:'+p)
    C.hashes(C.read(BASE/'runtime.json')['files'])
    C.hashes({str(ROOT/p):h for p,h in C.read(BASE/'inputs.json')['files'].items()})


def inspect():
    # Reuse the entire static PR0 source/control/event/4-QPY inspector, new output only.
    env=runpy.run_path(str(ROOT/'scripts/inspect_h4_arb_pr0.py'))
    env['inspect'].__globals__['BASE']=BASE
    env['inspect']()
    for name in ('PR0_selected_C_plan_NO_EXECUTION.json','PR0_compiled_program_NO_EXECUTION.json','state_identity_QUOTED.json'):
        C.require(C.sha(BASE/name)==C.sha(A.PREP/name),'SAVED_DESCRIPTOR_OR_PLAN_IDENTITY:'+name)
    A.load_inputs()
    C.save(BASE/'structural_preflight.json',dict(status='PASS_PR0_ACTUAL_QPY_PLAN_DESCRIPTOR_IDENTITY',QPY_decodes=4,
        state_actions=0,new_compile=0,compiled_sha256=A.QPY_SHA,fingerprint=A.QPY_FP,
        C_target_unscaled=True,B_saved_exact=A.SCALE,trajectory_id=A.P.TID,
        exact_plan_sha256=C.sha(BASE/'PR0_selected_C_plan_NO_EXECUTION.json'),
        descriptor_sha256=C.sha(BASE/'PR0_compiled_program_NO_EXECUTION.json')))


def prepare():
    C.require(Path(sys.executable)==Path(PYTHON),'FIXED_PRIVATE_ARB_ENVIRONMENT')
    C.require(not (BASE/'seal.json').exists() and not (BASE/'run01').exists(),'ALREADY_SEALED')
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    code={p:C.sha(ROOT/p) for p in CODE}
    for p in code:C.require(subprocess.check_output(['git','show',source+':'+p],cwd=ROOT)==(ROOT/p).read_bytes(),'UNCOMMITTED_CODE')
    C.require(C.read(BASE/'structural_preflight.json')['status']=='PASS_PR0_ACTUAL_QPY_PLAN_DESCRIPTOR_IDENTITY','NO_STRUCTURAL_PREFLIGHT')
    C.save(BASE/'runtime.json',runtime());C.save(BASE/'resource_before.json',resources())
    bound=['approval.md','inputs.json','runtime.json','structural_preflight.json']
    seal=dict(kind='H4_PR0_COSINE_ARB_SOURCE_INPUT_RUNTIME_SEAL_V1',SOURCE=source,code_files=code,
        python_sha256=C.sha(sys.executable),bound_manifests={p:C.sha(BASE/p) for p in bound},UTC=C.utc(),state_actions_before_seal=0)
    C.save(BASE/'seal.json',seal)
    contract=dict(kind='H4_PR0_COSINE_ARB_ACTIVE_V1',base_commit=A.BASE_COMMIT,SOURCE=source,
        execution_authorized=True,dispatch_enabled=True,grant_valid=True,target_key='PR0_cosine',cases=['pauli_jw_PR'],axes=['cosine'],trajectory_id=A.P.TID,
        precision_schedule_bits=list(A.PRECISIONS),limits=A.LIMITS,max_state_actions=6,compiled_sha256=A.QPY_SHA,
        fingerprint=A.QPY_FP,state_hash=A.STATE_HASH,rho_budget_exact=A.GUARD,B_saved_exact=A.SCALE,forbidden_calls=A.FORBIDDEN,
        mandatory_STOP=True,automatic_next_stage=False,formal_certificate_accepted=False,automatic_retry=False,
        seal_sha256=C.sha(BASE/'seal.json'),approval_sha256=C.sha(BASE/'approval.md'),
        precision_rule='stop on first decisive interval-budget relation; only straddle or nonfinite advances; width1e-8 advisory',
        recovery='technical only: preserve failure calls/SOURCE/grant/STOP; new SOURCE and unused grant; reuse successes',
        resource_policy='no arbitrary CPU/RAM/worker ceiling; two independent process-local Arb actions per precision, then sequential precision decision; phase cache/permutations; no shared changes')
    A.validate_contract(contract);C.save(BASE/'active_contract.json',contract);verify(seal)
    ns=BASE/'run01';ns.mkdir()
    grant=dict(kind='H4_PR0_COSINE_ARB_ONE_SHOT',namespace=str(ns),SOURCE=source,contract_sha256=C.sha(BASE/'active_contract.json'),
        seal_sha256=C.sha(BASE/'seal.json'),approval_sha256=C.sha(BASE/'approval.md'),one_shot=True,UTC=C.utc())
    grant['grant_id']=C.digest(grant);C.save(ns/'grant.json',grant)
    print(json.dumps(dict(status='SEALED_NOT_EXECUTED',SOURCE=source,grant_id=grant['grant_id'])))


def check(*,consume=False):
    ns=BASE/'run01';c=A.validate_contract(C.read(BASE/'active_contract.json'));s=C.read(BASE/'seal.json');g=C.read(ns/'grant.json')
    C.require(g['kind']=='H4_PR0_COSINE_ARB_ONE_SHOT' and g['one_shot'] and g['namespace']==str(ns),'GRANT_NAMESPACE')
    C.require(g['grant_id']==C.digest({k:v for k,v in g.items() if k!='grant_id'}),'GRANT_DIGEST')
    C.require(g['SOURCE']==s['SOURCE']==c['SOURCE'],'SOURCE_BINDING')
    C.require(g['contract_sha256']==C.sha(BASE/'active_contract.json') and g['seal_sha256']==C.sha(BASE/'seal.json')==c['seal_sha256'],'GRANT_SEAL_CONTRACT_BINDING')
    C.require(g['approval_sha256']==c['approval_sha256']==s['bound_manifests']['approval.md'],'APPROVAL_BINDING')
    verify(s);C.require(not (ns/'STOP.json').exists(),'ALREADY_STOPPED')
    if consume:C.save(ns/'CONSUMED.json',dict(grant_id=g['grant_id'],UTC=C.utc(),nonrefundable=True))
    else:C.require(C.read(ns/'CONSUMED.json')['grant_id']==g['grant_id'],'NO_CONSUMPTION')
    return c,s


def worker():
    c,s=check();ns=BASE/'run01';C.require((ns/'LAUNCH_CLAIM.json').is_file(),'NO_LAUNCH')
    C.save(ns/'WORKER_CLAIM.json',dict(pid=os.getpid(),UTC=C.utc()))
    try:
        A.run(ns,c,C.CallBudget(ns,A.LIMITS),lambda:verify(s));verify(s)
        C.save(ns/'resource_after.json',resources())
    except BaseException as e:
        C.save(ns/'worker_failure.json',dict(error=repr(e),UTC=C.utc(),reserved_calls_preserved=True))
        raise


def native_witness(directory,command):
    # Same native wait/subreaper body as C; change only the scope label.
    tree=ast.parse((ROOT/'src/trotterlib/h4_certificate_control.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='witness')
    for n in ast.walk(node):
        if isinstance(n,ast.Constant) and n.value=='NEW_C_WORKER_ONLY':n.value='NEW_H4_PR0_COSINE_ARB_WORKER_ONLY'
    env=dict(vars(C));exec(compile(ast.Module(body=[node],type_ignores=[]),'<reused C native witness, PR0 scope>','exec'),env)
    return env['witness'](directory,command)


def launch():
    c,s=check(consume=True);ns=BASE/'run01';C.save(ns/'LAUNCH_CLAIM.json',dict(pid=os.getpid(),UTC=C.utc()))
    status=native_witness(ns/'native',[sys.executable,'-B',str(Path(__file__).resolve()),'--worker'])
    error=None
    try:
        if status==0:
            verify(s);result=C.read(ns/'result_summary.json');used=C.read(ns/'call_ledger.json')['used']
            C.require(2<=result['state_action_attempts']<=6 and result['state_action_attempts']==2*len(result['tiers']),'STATE_ACTION_COUNT')
            C.require(sum(v for k,v in used.items() if k.startswith(('target_','compiled_')))==result['state_action_attempts'],'LEDGER_COUNT')
            C.require(all(C.read(p)['exit_code']==0 and C.read(p)['reaped'] for p in ns.glob('*_native_exit.json')),'ACTION_NATIVE_EXITS')
            C.require([t['precision_bits'] for t in result['tiers']]==list(A.PRECISIONS)[:len(result['tiers'])],'PRECISION_ORDER')
            C.require(all(t['relation'] in ('UNDETERMINED','NONFINITE') for t in result['tiers'][:-1]),'UNNECESSARY_PRECISION_ESCALATION')
    except Exception as e:status=1;error=str(e)
    C.save(ns/'STOP.json',dict(status='COMPLETE_MANDATORY_STOP' if status==0 else 'FAILED_NO_AUTOMATIC_RETRY',
        native_exit=C.read(ns/'native/worker_native_exit.json'),all_owned_reaped=C.read(ns/'native/STOP.json')['all_owned_reaped'],
        validation_error=error,original_H4_native_verified=False,automatic_next_stage=False,formal_certificate_accepted=False,UTC=C.utc()))
    return status


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    for name in ('inspect','prepare','launch','worker'):g.add_argument('--'+name,action='store_true')
    args=p.parse_args()
    if args.inspect:inspect()
    elif args.prepare:prepare()
    elif args.launch:sys.exit(launch())
    else:worker()
