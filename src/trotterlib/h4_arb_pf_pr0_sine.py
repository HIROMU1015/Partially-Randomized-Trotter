"""Two authorized sine targets, sharing one grant and unchanged Arb kernels."""
import ast
from fractions import Fraction as Q
import multiprocessing
import os
from pathlib import Path
from types import SimpleNamespace

from . import h4_arb_pf_cosine as PF_OLD
from . import h4_arb_pr0_cosine as PR_OLD
from . import h4_arb_pr0_preparation as P0
from . import h4_r1_sine as S

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'artifacts/non_df_h4_central_arb_pf_pr0_sine/2026-10-25'
BASE_COMMIT='06ca7c838808cbde9e1c1b5030c46f260d929c32'
B,C=P0.B,P0.C
KEYS=('PF_sine','PR0_sine')
PRECISIONS=PF_OLD.PRECISIONS
STATE_HASH=PR_OLD.STATE_HASH
FORBIDDEN=dict(new_compile=0,sampling=0,Hamiltonian_state_generation=0,additional_trajectory=0,
    cosine=0,PR1=0,DF=0,PF_C_target_state_actions=0,R2=0,N_G_reanalysis=0,formal_certificate_acceptance=0)
IDENTITIES={
    'PF_sine':dict(sha='a478034cc3bcafa4e3d85e69739ad92a997d5073b5db3e452d2d0030542549b1',
        fp='56ede2d2f721f8c5916e57865c4f47a4745832776d8e972fc7d3a0c91cd376fb',guard='25651956654414547227/7631344236317346665332736',scale='1'),
    'PR0_sine':dict(sha='5228f426d532ffd04f973b19fb65162c10eceb064f555b46c06af337f0106c0f',
        fp='a29f63a4fdbcb687d67de8bb98114529a7106b23b101a0dbdc2f2c67a6088619',guard='102330934580088554157/30525376945269386661330944',scale=PR_OLD.SCALE)}


class SineProjection(ast.NodeTransformer):
    """Keep PR0 index/order scope; select sine only and delegated authority marker."""
    def visit_Constant(self,node):
        if isinstance(node.value,str):node.value=(node.value.replace('SAVED_PR0_NO_EXECUTION','SAVED_PR0_SINE_NO_EXECUTION')
            .replace('COSINE','SINE').replace('cosine','sine').replace('CONSUMED.json','PARENT_AUTHORIZATION.json'))
        return node

    def visit_Attribute(self,node):
        node=self.generic_visit(node)
        if isinstance(node.value,ast.Name) and node.value.id=='dot' and node.attr=='real':node.attr='imag'
        return node

    def visit_Subscript(self,node):
        node=self.generic_visit(node)
        # PF whole saved C imag interval, not an empirical representative.
        if (isinstance(node.slice,ast.Constant) and node.slice.value=='real' and isinstance(node.value,ast.Subscript)
                and isinstance(node.value.slice,ast.Constant) and node.value.slice.value=='corrected_encoded_mean'):node.slice.value='imag'
        return node


def functions(path,names,env,transform=None):
    tree=ast.parse((ROOT/path).read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    C.require(len(nodes)==len(names),'REUSED_FUNCTION_SET:'+path)
    module=ast.Module(body=nodes,type_ignores=[])
    if transform:module=transform.visit(module)
    exec(compile(ast.fix_missing_locations(module),'<reused '+path+'>','exec'),env)
    return env


_plan_env=dict(vars(P0))
functions('src/trotterlib/h4_arb_pr0_preparation.py',('build_plan','validate_plan','_selected_unitary_enclosure'),_plan_env,SineProjection())
PR_PLAN=SimpleNamespace(**{**vars(P0),**{k:_plan_env[k] for k in ('build_plan','validate_plan','_selected_unitary_enclosure')}})


def make_case(key):
    C.require(key in KEYS,'TWO_TARGET_SCOPE');pf=key=='PF_sine';old=PF_OLD if pf else PR_OLD;identity=IDENTITIES[key]
    env={**vars(old),'BASE':BASE/key,'PREP':BASE/key,'BASE_COMMIT':BASE_COMMIT,
        'QPY_SHA':identity['sha'],'QPY_FP':identity['fp'],'GUARD':identity['guard'],'FORBIDDEN':FORBIDDEN}
    if not pf:env.update(P=PR_PLAN,SCALE=identity['scale'])
    names=('validate_contract','load_inputs','run') if pf else ('validate_contract','load_inputs','_action','action_pair','reduce_pair','run')
    functions('src/trotterlib/h4_arb_'+('pf' if pf else 'pr0')+'_cosine.py',names,env,SineProjection())
    return SimpleNamespace(**env)


CASES={key:make_case(key) for key in KEYS}


def quote_pf_interval(result,nu):
    """Exact metadata conversion of the ENTIRE saved imaginary interval."""
    nu=Q(nu);interval=result['implemented_law']['PF']['corrected_encoded_mean']['imag'];norm=result['state_norm']
    C.require(nu>0 and Q(norm['lower'])<=nu<=Q(norm['upper']),'SAVED_RAW_NORM')
    C.require(Q(interval['lower'])<=Q(interval['upper']),'SAVED_IMAG_INTERVAL')
    return dict(axis='sine',C_SOURCE=P0.C_SOURCE,saved_raw_interval=interval,saved_norm_interval=norm,
        normalized_interval_exact=[str(Q(interval[side])/nu) for side in ('lower','upper')],
        operation='whole saved C imag interval / same exact raw nu',new_C_target_state_actions=0)


def case_contract(key,source):
    a=CASES[key];pf=key=='PF_sine'
    c=dict(kind='H4_'+('PF' if pf else 'PR0')+'_SINE_ARB_ACTIVE_V1',base_commit=BASE_COMMIT,SOURCE=source,
        execution_authorized=True,dispatch_enabled=True,grant_valid=True,target_key='PF0_sine' if pf else key,
        cases=['pauli_jw_PF' if pf else 'pauli_jw_PR'],axes=['sine'],precision_schedule_bits=list(PRECISIONS),
        limits=a.LIMITS,max_state_actions=3 if pf else 6,compiled_sha256=a.QPY_SHA,fingerprint=a.QPY_FP,
        state_hash=STATE_HASH,rho_budget_exact=a.GUARD,B_saved_exact=IDENTITIES[key]['scale'],forbidden_calls=FORBIDDEN,
        mandatory_STOP=True,automatic_next_stage=False,formal_certificate_accepted=False,
        authority='one parent grant; no separate child grant or consumption')
    if not pf:c['trajectory_id']=P0.TID
    a.validate_contract(c);return c


def validate_contract(c):
    C.require(c['kind']=='H4_PF_PR0_SINE_ARB_ACTIVE_V1' and c['base_commit']==BASE_COMMIT,'TWO_CASE_CONTRACT')
    C.require(all(c[k] is True for k in ('execution_authorized','dispatch_enabled','grant_valid')),'EXPLICIT_AUTHORITY')
    C.require(c['cases']==list(KEYS) and c['axes']==['sine'] and c['max_state_actions']==9,'TWO_SINE_NINE_ACTIONS_ONLY')
    C.require(c['precision_schedule_bits']==list(PRECISIONS) and c['forbidden_calls']==FORBIDDEN,'FIXED_SCHEDULE_FORBIDDEN')
    C.require(set(c['targets'])==set(KEYS),'EXACT_TARGET_SET')
    for key in KEYS:C.require(c['targets'][key]==case_contract(key,c['SOURCE']),'BOUND_TARGET_CONTRACT:'+key)
    C.require(c['mandatory_STOP'] and not c['automatic_next_stage'] and not c['formal_certificate_accepted'],'MANDATORY_STOP')
    return c


def authorize_case(ns,key,contract):
    """Read the consumed ROOT grant. Child authority is not a new receipt/grant."""
    ns=Path(ns);C.require(key in KEYS and ns==BASE/'run01','FIXED_PARENT_NAMESPACE')
    root=validate_contract(C.read(BASE/'active_contract.json'));grant=C.read(ns/'grant.json')
    C.require(grant['kind']=='H4_PF_PR0_SINE_ARB_ONE_SHOT' and grant['one_shot'],'PARENT_GRANT_KIND')
    C.require(grant['namespace']==str(ns) and grant['SOURCE']==root['SOURCE']==contract['SOURCE'],'PARENT_SOURCE_NAMESPACE')
    C.require(grant['grant_id']==C.digest({k:v for k,v in grant.items() if k!='grant_id'}),'PARENT_GRANT_DIGEST')
    C.require(C.sha(BASE/'active_contract.json')==grant['contract_sha256'] and root['targets'][key]==contract,'PARENT_CONTRACT_BINDING')
    C.require(C.read(ns/'CONSUMED.json')['grant_id']==grant['grant_id'] and (ns/'WORKER_CLAIM.json').is_file(),'CONSUMED_PARENT_REQUIRED')
    C.require(not (ns/'STOP.json').exists() and not (ns/key/'STOP.json').exists(),'PARENT_OR_TARGET_STOPPED')
    return dict(parent_namespace=str(ns),grant_id=grant['grant_id'],SOURCE=root['SOURCE'],
        contract_sha256=grant['contract_sha256'],target=key,delegation_only=True,new_grant=False,new_consumption=False,UTC=C.utc())


def case_worker(ns,key,contract,verify):
    ns=Path(ns);directory=ns/key;authority=authorize_case(ns,key,contract);verify()
    C.save(directory/'WORKER_CLAIM.json',dict(pid=os.getpid(),UTC=C.utc(),target=key))
    C.save(directory/'PARENT_AUTHORIZATION.json',authority)
    try:
        CASES[key].run(directory,contract,C.CallBudget(directory,CASES[key].LIMITS),verify);verify()
    except BaseException as exc:
        C.save(directory/'worker_failure.json',dict(error=repr(exc),UTC=C.utc(),reserved_calls_preserved=True));raise


def run(ns,contract,verify):
    validate_contract(contract);children=[]
    try:
        for key in KEYS:
            (ns/key).mkdir();child=multiprocessing.get_context('fork').Process(target=case_worker,args=(str(ns),key,contract['targets'][key],verify))
            child.start();children.append((key,child))
        for key,child in children:
            child.join();receipt=dict(pid=child.pid,exit_code=child.exitcode,native_wait=True,reaped=True,UTC=C.utc())
            C.save(ns/(key+'_native_exit.json'),receipt)
            C.save(ns/key/'STOP.json',dict(status='COMPLETE_MANDATORY_STOP' if child.exitcode==0 else 'FAILED_NO_AUTOMATIC_RETRY',
                target_native_exit=receipt,automatic_next_stage=False,formal_certificate_accepted=False,original_H4_native_verified=False,UTC=C.utc()))
        C.require(all(child.exitcode==0 for _,child in children),'SINE_TARGET_FAILURE')
        results={key:C.read(ns/key/'result_summary.json') for key in KEYS};verify()
        C.save(ns/'result_summary.json',dict(status='TWO_SINE_DIAGNOSTICS_COMPLETE_MANDATORY_STOP',SOURCE=contract['SOURCE'],
            logical_diagnostics=2,targets=results,state_action_attempts=sum(r['state_action_attempts'] for r in results.values()),
            successful_state_actions=sum(r['successful_state_actions'] for r in results.values()),
            residual_reductions=sum(r['residual_reductions'] for r in results.values()),new_PF_C_target_state_actions=0,
            formal_certificate_accepted=False,LOCAL_BUDGET_MET=False,PR_law_rho=None,old_N_G_changed=False,
            forbidden_calls=FORBIDDEN,mandatory_STOP=True,automatic_next_stage=False))
    finally:
        for _,child in children:
            if child.is_alive():child.join()
