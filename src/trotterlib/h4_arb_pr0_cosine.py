"""Bound PR0 cosine pair: selected exact C-target and saved native QPY."""
from fractions import Fraction as Q
import multiprocessing
import os
from pathlib import Path
import time

from . import h4_arb_pr0_preparation as P
from .h4_arb_pf_cosine import finite_schedule

B,C=P.B,P.C
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'artifacts/non_df_h4_central_arb_pr0_cosine/2026-10-25'
PREP=ROOT/'artifacts/non_df_h4_central_arb_pr0_preparation/2026-10-25'
BASE_COMMIT='f817c492c527d4e436b81abda9e9341defa1c49d'
PRECISIONS=(64,96,128)
LIMITS={role+'_'+str(bits):1 for role in ('target','compiled','residual') for bits in PRECISIONS}
QPY_SHA='f68965bde6a001bc3a9dd6f0d55a5a76682b868738360cef620e926c0a99fa2c'
QPY_FP='f7e855e7560ccd9c64c188075c86a7a4c1b2f023272c9bf7f6fa9cf3147c169a'
STATE_HASH='8339b1a8b0bec35ae07688e667fde7ba602d56c1182523d58e0889ea114e8b80'
GUARD='19792741170690161541/7631344236317346665332736'
SCALE='1449450801590761/1125899906842624'
FORBIDDEN=dict(new_compile=0,sampling=0,Hamiltonian_state_generation=0,input_QPY_state_actions=0,
    PR1=0,sine=0,DF=0,R2=0,N_G_reanalysis=0,formal_certificate_acceptance=0)
NONFINITE=('NONFINITE_GATE_ENCLOSURE','FINITE_SIGNAL_ENCLOSURE','NONFINITE_ENCLOSURE','FINITE_SELECTED_SIGNAL')


def validate_contract(c):
    C.require(c['kind']=='H4_PR0_COSINE_ARB_ACTIVE_V1' and c['base_commit']==BASE_COMMIT,'ACTIVE_CONTRACT_KIND_BASE')
    C.require(all(c[k] is True for k in ('execution_authorized','dispatch_enabled','grant_valid')),'EXPLICIT_EXECUTION_AUTHORITY')
    C.require(c['precision_schedule_bits']==list(PRECISIONS) and c['limits']==LIMITS and c['max_state_actions']==6,'FINITE_CALL_SCOPE')
    C.require(c['target_key']=='PR0_cosine' and c['axes']==['cosine'] and c['cases']==['pauli_jw_PR'] and c['trajectory_id']==P.TID,'ONE_PR0_COSINE_ONLY')
    C.require(c['compiled_sha256']==QPY_SHA and c['fingerprint']==QPY_FP and c['state_hash']==STATE_HASH,'FIXED_SCIENTIFIC_INPUTS')
    C.require(c['rho_budget_exact']==GUARD and c['B_saved_exact']==SCALE,'UNCHANGED_BUDGET_SCALE')
    C.require(c['forbidden_calls']==FORBIDDEN,'FORBIDDEN_SCOPE')
    C.require(c['mandatory_STOP'] and not c['automatic_next_stage'] and not c['formal_certificate_accepted'],'NO_ACCEPTANCE_OR_CONTINUATION')
    return c


def load_inputs():
    """Saved metadata only; compare the cached exact plan to original operands."""
    target=C.read(PREP/'PR0_target_QUOTED.json')
    model=C.read(ROOT/target['state_model']);join=C.read(ROOT/target['price_join'])
    p=C.read(PREP/'PR0_selected_C_plan_NO_EXECUTION.json')
    native=C.read(PREP/'PR0_compiled_program_NO_EXECUTION.json');state=C.read(PREP/'state_identity_QUOTED.json')
    C.require(C.digest({k:v for k,v in model.items() if k!='binding_hash'})==model['binding_hash'],'MODEL_BINDING')
    C.require(P.build_plan(model,join)==p,'SAVED_PLAN_ORIGINAL_C_EVENT_OPERANDS')
    C.require(p['B_saved_binary64_exact']==SCALE and len(p['retained_blocks'])==27,'FIXED_PR0_SCALE_SPLIT')
    B.validate_program(native)
    C.require((native['domain'],native['axis'],native['n_system'],native['ancilla'])==('SAVED_QPY_NO_EXECUTION','cosine',8,8),'PR0_LAYOUT_AXIS')
    C.require(native['origin']['qpy_sha256']==QPY_SHA==C.sha(ROOT/target['compiled_qpy']) and native['origin']['fingerprint']==QPY_FP,'QPY_ORIGIN')
    C.require(model['state_hash']==state['state_hash']==target['state_hash']==STATE_HASH,'RAW_RAY_IDENTITY')
    C.require(len(model['state'])==256 and Q(state['nu_exact'])>0,'RAW_STATE_NORM_SHAPE')
    guard=C.read(PREP/'PR0_guard_QUOTED.json')
    C.require(guard['rho_guard_exact']==GUARD,'SAVED_RHO_GUARD')
    return p,native,model['state'],state


def _action(ns,role,spec,raw,bits,source):
    """One child, one reserved action; numeric routines never retry here."""
    ns=Path(ns);key=role+'_'+str(bits);start=time.monotonic();last=[0.]
    C.save(ns/(key+'_worker_claim.json'),dict(pid=os.getpid(),UTC=C.utc(),role=role,precision_bits=bits))
    def progress(done,total):
        now=time.monotonic()
        if now-last[0]>=5 or done==total:
            C.save(ns/(role+'_progress.json'),dict(precision_bits=bits,completed=done,total=total,
                elapsed_seconds=now-start,UTC=C.utc()),exclusive=False);last[0]=now
    try:
        kernel=P._selected_unitary_enclosure if role=='target' else B._native_unitary_enclosure
        _,record=kernel(spec,raw,bits,progress=progress)
        record.update(SOURCE=source,target_key='PR0_cosine',wall_seconds=time.monotonic()-start)
        result=dict(status='FINITE',record=record)
    except ValueError as exc:
        if str(exc) not in NONFINITE:raise
        result=dict(status='NONFINITE',error=str(exc))
    C.save(ns/(key+'.json'),result)


def action_pair(ns,plan,program,raw,bits,source,budget):
    """Two independent walks overlap; process-local Arb contexts, own children only."""
    children=[]
    try:
        for role,spec in (('target',plan),('compiled',program)):
            key=role+'_'+str(bits);budget.take(key)
            C.save(ns/(key+'_claim.json'),dict(precision_bits=bits,UTC=C.utc(),nonrefundable=True))
            child=multiprocessing.get_context('fork').Process(target=_action,args=(str(ns),role,spec,raw,bits,source))
            child.start();children.append((key,child))
        for key,child in children:
            child.join()
            C.save(ns/(key+'_native_exit.json'),dict(pid=child.pid,exit_code=child.exitcode,native_wait=True,reaped=True,UTC=C.utc()))
        C.require(all(child.exitcode==0 for _,child in children),'ACTION_WORKER_FAILURE')
        return {key.split('_')[0]:C.read(ns/(key+'.json')) for key,_ in children}
    finally:
        # Never leave a child behind; ordinary pressure never triggers cancellation.
        for key,child in children:
            if child.is_alive():child.join()


def reduce_pair(target,compiled,*,bits,budget_exact=GUARD,scale_exact=SCALE):
    """Whole interval difference; a second outward union is safe, no midpoint."""
    backend=B._backend()
    with backend.ctx.workprec(bits):
        ep=compiled['interval']
        signal=backend.real(Q(ep['lower'])).union(backend.real(Q(ep['upper'])))
        _,rho=B._residual_interval(signal,target['interval'],B_saved_exact=scale_exact,budget_exact=budget_exact,bits=bits)
    lo,hi=map(Q,(rho['interval']['lower'],rho['interval']['upper']))
    rho.update(precision_bits=bits,interval_width_exact=str(hi-lo),
        nonnegative_intersection=dict(lower=str(max(Q(0),lo)),upper=str(hi)),raw_ball_preserved=True,
        numerical_local_budget_satisfied=rho['budget_relation']=='UPPER_WITHIN_BUDGET',
        preferred_width_target_met=hi-lo<=Q(1,10**8),rho_budget_exact=budget_exact,
        definition='B_saved*abs(compiled cosine whole enclosure - selected C-target whole enclosure)',
        B_applied_exactly_once=True,PR_law_mean_used=False)
    return rho


def run(ns,contract,budget,verify):
    validate_contract(contract)
    C.require((ns/'WORKER_CLAIM.json').is_file() and (ns/'CONSUMED.json').is_file(),'NO_DISPATCHED_WORKER')
    plan,program,raw,state=load_inputs()
    C.require(B._backend().array_hash(raw,vector=True)==STATE_HASH,'ACTUAL_RAW_STATE_HASH')
    C.save(ns/'state_target_binding.json',dict(state_hash=STATE_HASH,raw_norm_exact=state['nu_exact'],
        trajectory_id=P.TID,C_SOURCE=P.C_SOURCE,selected_events=32,state_renormalized=False,
        C_target_unscaled=True,B_saved_exact=SCALE,PR_law_mean_used=False))
    def evaluate(bits):
        verify();pair=action_pair(ns,plan,program,raw,bits,contract['SOURCE'],budget)
        if all(r['status']=='FINITE' for r in pair.values()):
            t,c=pair['target']['record'],pair['compiled']['record']
            C.require(t['raw_norm_exact']==c['raw_norm_exact']==state['nu_exact'],'COMMON_EXACT_NU')
            budget.take('residual_'+str(bits));rho=reduce_pair(t,c,bits=bits)
            C.save(ns/('rho_'+str(bits)+'.json'),rho)
            result=dict(precision_bits=bits,relation=rho['budget_relation'],target=t,compiled=c,rho=rho)
        else:result=dict(precision_bits=bits,relation='NONFINITE',actions=pair)
        C.save(ns/('tier_'+str(bits)+'.json'),result);verify();return result
    summary=finite_schedule(evaluate);used=C.read(ns/'call_ledger.json')['used']
    summary.update(SOURCE=contract['SOURCE'],key='PR0_cosine',precision_bits=summary['last']['precision_bits'],
        state_action_attempts=sum(v for k,v in used.items() if k.startswith(('target_','compiled_'))),
        successful_state_actions=sum(C.read(p)['status']=='FINITE' for p in ns.glob('target_[0-9]*.json') if p.stem.split('_')[-1].isdigit())+
                                 sum(C.read(p)['status']=='FINITE' for p in ns.glob('compiled_[0-9]*.json') if p.stem.split('_')[-1].isdigit()),
        residual_reductions=sum(v for k,v in used.items() if k.startswith('residual_')),
        formal_certificate_accepted=False,LOCAL_BUDGET_MET=False,PR_law_rho=None,old_N_G_changed=False,
        forbidden_calls=FORBIDDEN,mandatory_STOP=True,automatic_next_stage=False)
    C.save(ns/'result_summary.json',summary);return summary
