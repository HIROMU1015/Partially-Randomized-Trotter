#!/usr/bin/env python3
"""Two-case sine one-shot with one root grant, reused finite walks and audit."""
import argparse
import ast
import multiprocessing.popen_fork
from pathlib import Path
import runpy
import subprocess
import sys
import types

ROOT=Path(__file__).resolve().parents[1]
pkg=types.ModuleType('arb_pair_sine_private');pkg.__path__=[str(ROOT/'src/trotterlib')];sys.modules['arb_pair_sine_private']=pkg
from arb_pair_sine_private import h4_arb_pf_pr0_sine as A
C,B,BASE=A.C,A.B,A.BASE


class RootScope(ast.NodeTransformer):
    def visit_Constant(self,node):
        labels={'H4_PR0_COSINE_ARB_SOURCE_INPUT_RUNTIME_SEAL_V1':'H4_PF_PR0_SINE_ARB_SOURCE_INPUT_RUNTIME_SEAL_V1',
            'H4_PR0_COSINE_ARB_ONE_SHOT':'H4_PF_PR0_SINE_ARB_ONE_SHOT',
            'NEW_H4_PR0_COSINE_ARB_WORKER_ONLY':'NEW_H4_PF_PR0_SINE_ARB_WORKER_ONLY'}
        if isinstance(node.value,str):node.value=labels.get(node.value,node.value)
        return node


def controller():
    prior=runpy.run_path(str(ROOT/'scripts/run_h4_arb_pr1_sine.py'))['controller']()
    env={**prior,**globals()}
    env['CODE']=prior['CODE']+['src/trotterlib/h4_arb_pf_pr0_sine.py','scripts/run_h4_arb_pf_pr0_sine.py',
        'tests/test_h4_arb_pf_pr0_sine.py','artifacts/non_df_h4_central_arb_pf_pr0_sine/2026-10-25/prepare_metadata.py']
    A.functions('scripts/run_h4_arb_pr0_cosine.py',('resources','runtime','verify','check','native_witness'),env,RootScope())
    return env


def inspect():
    """Scope the existing sine preflight to PF/PR0; omit empirical cache use."""
    C.hashes({str(ROOT/p):h for p,h in C.read(BASE/'inputs.json')['files'].items()})
    tree=ast.parse((ROOT/'src/trotterlib/h4_r1_sine.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='preflight')
    body=[]
    for n in node.body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Tuple) and any(isinstance(v,ast.Name) and v.id=='state0' for v in t.elts) for t in n.targets):break
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Tuple) and any(isinstance(v,ast.Name) and v.id=='targets' for v in t.elts) for t in n.targets):
            n=ast.parse("targets, cosines, files = inpmanifest['targets'][:2], inpmanifest['cosine_targets'][:2], inpmanifest['files']").body[0]
        body.append(n)
    C.require(len(body)<len(node.body),'NO_EMPIRICAL_CACHE_STAGE')
    body.append(ast.parse("return targets, model, circuits, dict(status='PASS_PF_PR0_SINE_STATIC',QPY_decodes=10,state_actions=0,records=records,empirical_cache_used=False)").body[0])
    node.body=body;env=dict(vars(A.S),KEYS=('PF0_sine','PR0_sine'),CASES=('PF','PR0'))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'<existing sine preflight, two targets, no point caches>','exec'),env)
    targets,model,circuits,binding=env['preflight']()
    state=C.read(A.S.OLD0/'run01/state_identity.json');nu=A.Q(state['nu_exact'])
    C.require(model['state_hash']==state['state_hash']==A.STATE_HASH and nu>0,'COMMON_RAW_RAY')
    guards=C.read(A.S.P.BASE/'guard_inputs_quoted.json')['axes'];files=C.read(BASE/'inputs.json')
    for key,t in zip(A.KEYS,targets):
        a=A.CASES[key];folder=BASE/key;folder.mkdir()
        program=B.descriptor_from_circuit(circuits['PF' if key=='PF_sine' else 'PR0']['output'],axis='sine')
        program['origin']=dict(qpy_sha256=C.sha(ROOT/t['compiled_qpy']),fingerprint=t['output_fingerprint'],decode_only=True,state_actions=0,compile=0)
        C.require(program['origin']['qpy_sha256']==a.QPY_SHA and program['origin']['fingerprint']==a.QPY_FP,'FIXED_SINE_QPY:'+key)
        guard=next(g for g in guards if (g['case'],g['epsilon_sig'],g['axis'])==('pauli_jw_'+t['method'],'0.0005','sine'))
        C.require(guard['rho_guard_exact']==a.GUARD,'SAVED_EXACT_SINE_BUDGET')
        C.save(folder/'state_identity_QUOTED.json',state);C.save(folder/'guard_QUOTED.json',guard)
        if key=='PF_sine':
            result=C.read(ROOT/t['C_encoded_mean_pointer'][0])
            C.save(folder/'PF_sine_saved_target.json',t);C.save(folder/'PF_sine_program_NO_EXECUTION.json',program)
            C.save(folder/'PF_C_target_interval_QUOTED.json',A.quote_pf_interval(result,nu))
        else:
            plan=A.PR_PLAN.build_plan(model,C.read(ROOT/t['price_join']))
            C.save(folder/'PR0_target_QUOTED.json',t);C.save(folder/'PR0_guard_QUOTED.json',guard)
            C.save(folder/'PR0_selected_C_plan_NO_EXECUTION.json',plan);C.save(folder/'PR0_compiled_program_NO_EXECUTION.json',program)
        a.load_inputs()
        for p in folder.glob('*.json'):files['files'][str(p.relative_to(ROOT))]=C.sha(p)
    C.save(BASE/'inputs.json',files,exclusive=False)
    C.save(BASE/'structural_preflight.json',binding)
    print(A.S.R.json.dumps(dict(status=binding['status'],QPY_decodes=10,state_actions=0)))


def prepare(env):
    C.require(Path(sys.executable)==Path(env['PYTHON']),'FIXED_PRIVATE_ARB_ENVIRONMENT')
    C.require(not (BASE/'seal.json').exists() and not (BASE/'run01').exists(),'ALREADY_SEALED')
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    code={p:C.sha(ROOT/p) for p in env['CODE']}
    for p in code:C.require(subprocess.check_output(['git','show',source+':'+p],cwd=ROOT)==(ROOT/p).read_bytes(),'UNCOMMITTED_CODE')
    C.require(C.read(BASE/'structural_preflight.json')['status']=='PASS_PF_PR0_SINE_STATIC','PREFLIGHT_REQUIRED')
    C.save(BASE/'runtime.json',env['runtime']());C.save(BASE/'resource_before.json',env['resources']())
    seal=dict(kind='H4_PF_PR0_SINE_ARB_SOURCE_INPUT_RUNTIME_SEAL_V1',SOURCE=source,code_files=code,
        python_sha256=C.sha(sys.executable),bound_manifests={p:C.sha(BASE/p) for p in ('approval.md','inputs.json','runtime.json','structural_preflight.json')},
        UTC=C.utc(),state_actions_before_seal=0)
    C.save(BASE/'seal.json',seal)
    contract=dict(kind='H4_PF_PR0_SINE_ARB_ACTIVE_V1',base_commit=A.BASE_COMMIT,SOURCE=source,
        execution_authorized=True,dispatch_enabled=True,grant_valid=True,cases=list(A.KEYS),axes=['sine'],max_state_actions=9,
        precision_schedule_bits=list(A.PRECISIONS),targets={key:A.case_contract(key,source) for key in A.KEYS},
        forbidden_calls=A.FORBIDDEN,mandatory_STOP=True,automatic_next_stage=False,formal_certificate_accepted=False,automatic_retry=False,
        seal_sha256=C.sha(BASE/'seal.json'),approval_sha256=C.sha(BASE/'approval.md'),
        precision_rule='independent target schedules; stop each at first decisive interval; straddle/nonfinite only advances; width1e-8 advisory',
        authority='one root one-shot; case markers are delegated authority, no child grant/consumption',
        resource_policy='parallel independent cases and PR pair; no CPU/RAM/affinity ceiling or PSI kill; no shared changes',
        recovery='technical only; preserve failure calls/SOURCE/grant/STOP; new SOURCE and unused grant; never repeat successful actions unnecessarily')
    A.validate_contract(contract);C.save(BASE/'active_contract.json',contract);env['verify'](seal)
    ns=BASE/'run01';ns.mkdir()
    grant=dict(kind='H4_PF_PR0_SINE_ARB_ONE_SHOT',namespace=str(ns),SOURCE=source,contract_sha256=C.sha(BASE/'active_contract.json'),
        seal_sha256=C.sha(BASE/'seal.json'),approval_sha256=C.sha(BASE/'approval.md'),one_shot=True,UTC=C.utc())
    grant['grant_id']=C.digest(grant);C.save(ns/'grant.json',grant)
    print(A.S.R.json.dumps(dict(status='SEALED_NOT_EXECUTED',SOURCE=source,grant_id=grant['grant_id'])))


def worker(env):
    c,s=env['check']();ns=BASE/'run01';C.require((ns/'LAUNCH_CLAIM.json').is_file(),'NO_LAUNCH')
    C.save(ns/'WORKER_CLAIM.json',dict(pid=__import__('os').getpid(),UTC=C.utc()))
    try:
        A.run(ns,c,lambda:env['verify'](s));env['verify'](s);C.save(ns/'resource_after.json',env['resources']())
    except BaseException as exc:
        C.save(ns/'worker_failure.json',dict(error=repr(exc),reserved_calls_preserved=True,UTC=C.utc()));raise


def launch(env):
    c,s=env['check'](consume=True);ns=BASE/'run01';C.save(ns/'LAUNCH_CLAIM.json',dict(pid=__import__('os').getpid(),UTC=C.utc()))
    status=env['native_witness'](ns/'native',[sys.executable,'-B',str(Path(__file__).resolve()),'--worker']);error=None
    try:
        if status==0:
            env['verify'](s);r=C.read(ns/'result_summary.json')
            C.require(r['logical_diagnostics']==2 and 3<=r['state_action_attempts']<=9,'TOTAL_STATE_ACTIONS')
            for key in A.KEYS:
                result=r['targets'][key];factor=1 if key=='PF_sine' else 2
                C.require(result['state_action_attempts']==factor*len(result['tiers']),'TARGET_STATE_ACTIONS')
                C.require([t['precision_bits'] for t in result['tiers']]==list(A.PRECISIONS)[:len(result['tiers'])],'TARGET_PRECISION_ORDER')
                C.require(all(t['relation'] in ('UNDETERMINED','NONFINITE') for t in result['tiers'][:-1]),'NO_UNNECESSARY_ESCALATION')
    except Exception as exc:status=1;error=str(exc)
    C.save(ns/'STOP.json',dict(status='COMPLETE_MANDATORY_STOP' if status==0 else 'FAILED_NO_AUTOMATIC_RETRY',
        native_exit=C.read(ns/'native/worker_native_exit.json'),all_owned_reaped=C.read(ns/'native/STOP.json')['all_owned_reaped'],
        validation_error=error,original_H4_native_verified=False,formal_certificate_accepted=False,automatic_next_stage=False,UTC=C.utc()))
    return status


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    for mode in ('inspect','prepare','launch','worker'):g.add_argument('--'+mode,action='store_true')
    args=p.parse_args();env=controller()
    if args.inspect:inspect()
    elif args.prepare:prepare(env)
    elif args.launch:sys.exit(launch(env))
    else:worker(env)
