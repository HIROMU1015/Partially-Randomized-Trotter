"""One bound PF cosine local enclosure; authority comes from the new controller."""
from fractions import Fraction as Q
from pathlib import Path
import time

from . import h4_arb_bridge as B
from . import h4_certificate_control as C

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'artifacts/non_df_h4_central_arb_pf_cosine/2026-10-25'
PREP = ROOT/'artifacts/non_df_h4_central_arb_preparation/2026-10-24'
BASE_COMMIT = '5c69e1339955c77273b356df30e94ec3c57458b1'
PRECISIONS = (64,96,128)
LIMITS = {**{'compiled_'+str(p):1 for p in PRECISIONS}, **{'residual_'+str(p):1 for p in PRECISIONS}}
QPY_SHA = '8537e01152d7b9a169bcea2dc20f4ce107d560b3af39b47b326e37c3c309a77d'
QPY_FP = 'ac65c4de7d57b254b5fced776a039dd36a76b35fed0a33c79eb5478cdc44a173'
STATE_HASH = '8339b1a8b0bec35ae07688e667fde7ba602d56c1182523d58e0889ea114e8b80'
GUARD = '10107730069649565297/3815672118158673332666368'
FORBIDDEN = dict(new_compile=0,sampling=0,Hamiltonian_state_generation=0,new_C_target_state_actions=0,
    input_QPY_state_actions=0,PR=0,sine=0,DF=0,R2=0,N_G_reanalysis=0,formal_certificate_acceptance=0)


def validate_contract(c):
    C.require(c['kind']=='H4_PF_COSINE_ARB_ACTIVE_V1' and c['base_commit']==BASE_COMMIT,'ACTIVE_CONTRACT_KIND_BASE')
    C.require(all(c[k] is True for k in ('execution_authorized','dispatch_enabled','grant_valid')),'EXPLICIT_EXECUTION_AUTHORITY')
    C.require(c['precision_schedule_bits']==list(PRECISIONS) and c['limits']==LIMITS and c['max_state_actions']==3,'FINITE_CALL_SCOPE')
    C.require(c['target_key']=='PF0_cosine' and c['axes']==['cosine'] and c['cases']==['pauli_jw_PF'],'ONE_PF_COSINE_ONLY')
    C.require(c['compiled_sha256']==QPY_SHA and c['fingerprint']==QPY_FP and c['state_hash']==STATE_HASH,'FIXED_SCIENTIFIC_INPUTS')
    C.require(c['rho_budget_exact']==GUARD and c['B_saved_exact']=='1','UNCHANGED_BUDGET_SCALE')
    C.require(c['forbidden_calls']==FORBIDDEN,'FORBIDDEN_SCOPE')
    C.require(c['mandatory_STOP'] and not c['automatic_next_stage'] and not c['formal_certificate_accepted'],'NO_ACCEPTANCE_OR_CONTINUATION')
    return c


def load_inputs():
    """Saved JSON and exact/hash validation only, no state action or new target."""
    p = C.read(PREP/'PF_cosine_program_NO_EXECUTION.json')
    t = C.read(PREP/'PF_cosine_saved_target.json')
    state_id = C.read(PREP/'state_identity_QUOTED.json')
    target = C.read(PREP/'PF_C_target_interval_QUOTED.json')
    model = C.read(ROOT/t['state_model'])
    cresult = C.read(ROOT/t['C_encoded_mean_pointer'][0])
    B.validate_program(p)
    C.require((p['domain'],p['axis'],p['n_system'],p['ancilla'])==('SAVED_QPY_NO_EXECUTION','cosine',8,8),'PF_LAYOUT_AXIS')
    C.require(p['origin']['qpy_sha256']==QPY_SHA==C.sha(ROOT/t['compiled_qpy']) and p['origin']['fingerprint']==QPY_FP,'QPY_ORIGIN')
    C.require(model['state_hash']==state_id['state_hash']==t['state_hash']==STATE_HASH,'RAW_RAY_IDENTITY')
    C.require(C.digest({k:v for k,v in model.items() if k!='binding_hash'})==model['binding_hash'],'MODEL_BINDING')
    C.require(model['representation']=='pauli_jw' and model['qubit_order']=='qiskit_little_endian' and model['n_qubits']==8,'MODEL_LAYOUT')
    C.require(model['cases']['PF']['config']==dict(method='PF',T=.8,q=8,r=4,K=4),'FIXED_MODEL')
    C.require(len(model['state'])==256,'RAW_STATE_SHAPE')
    nu = Q(state_id['nu_exact'])
    C.require(cresult['implemented_law']['PF']['corrected_encoded_mean']['real']==target['saved_raw_interval']
              and cresult['state_norm']==target['saved_norm_interval'],'SAVED_C_SOURCE_INTERVAL_IDENTITY')
    C.require(nu>0 and Q(target['saved_norm_interval']['lower'])<=nu<=Q(target['saved_norm_interval']['upper']),'SAVED_RAW_NORM')
    lo,hi = map(Q,target['normalized_interval_exact'])
    C.require([lo,hi]==[Q(target['saved_raw_interval'][s])/nu for s in ('lower','upper')],'EXACT_SAVED_TARGET_NO_MIDPOINT')
    return p,model['state'],state_id,dict(lower=str(lo),upper=str(hi))


def finite_schedule(evaluate):
    """evaluate reserves each walk; only unresolved/nonfinite advances."""
    tiers=[]
    for bits in PRECISIONS:
        result=evaluate(bits);tiers.append(result)
        C.require(result['relation'] in ('UPPER_WITHIN_BUDGET','LOWER_EXCEEDS_BUDGET','UNDETERMINED','NONFINITE'),'TIER_RELATION')
        if result['relation'] in ('UPPER_WITHIN_BUDGET','LOWER_EXCEEDS_BUDGET'):
            return dict(status='LOCAL_NUMERICAL_RELATION_DETERMINED',relation=result['relation'],tiers=tiers,last=result)
    return dict(status='INCONCLUSIVE_PRECISION_EXHAUSTED',relation='INCONCLUSIVE',tiers=tiers,last=tiers[-1])


def run(ns,contract,budget,verify):
    validate_contract(contract)
    C.require((ns/'WORKER_CLAIM.json').is_file() and (ns/'CONSUMED.json').is_file(),'NO_DISPATCHED_WORKER')
    program,raw,state_id,target=load_inputs()
    # Existing C array hash, not a numerical state action.
    C.require(B._backend().array_hash(raw,vector=True)==STATE_HASH,'ACTUAL_RAW_STATE_HASH')
    C.save(ns/'state_target_binding.json',dict(state_hash=STATE_HASH,raw_norm_exact=state_id['nu_exact'],
        C_target_interval_exact=target,state_renormalized=False,new_C_target_state_actions=0))
    def evaluate(bits):
        verify();key='compiled_'+str(bits);budget.take(key)
        C.save(ns/(key+'_claim.json'),dict(precision_bits=bits,pid=__import__('os').getpid(),UTC=C.utc(),nonrefundable=True))
        start=time.monotonic();last=[0.]
        def progress(done,total):
            now=time.monotonic()
            if now-last[0]>=5 or done==total:
                C.save(ns/'progress.json',dict(precision_bits=bits,completed_gates=done,total_gates=total,
                    elapsed_seconds=now-start,UTC=C.utc()),exclusive=False);last[0]=now
        try:
            signal,record=B._native_unitary_enclosure(program,raw,bits,progress=progress)
            C.require(record['raw_norm_exact']==state_id['nu_exact'],'SAME_EXACT_NU')
            record.update(wall_seconds=time.monotonic()-start,SOURCE=contract['SOURCE'],target_key='PF0_cosine')
            C.save(ns/('signal_'+str(bits)+'.json'),record)
            budget.take('residual_'+str(bits))
            _,rho=B._residual_interval(signal,target,B_saved_exact='1',budget_exact=GUARD,bits=bits)
            rho.update(precision_bits=bits,interval_width_exact=str(Q(rho['interval']['upper'])-Q(rho['interval']['lower'])),
                numerical_local_budget_satisfied=rho['budget_relation']=='UPPER_WITHIN_BUDGET',
                preferred_width_target_met=Q(rho['interval']['upper'])-Q(rho['interval']['lower'])<=Q(1,10**8),
                definition='B_saved*abs(compiled_signal - whole saved C-target interval)',rho_budget_exact=GUARD)
            C.save(ns/('rho_'+str(bits)+'.json'),rho)
            result=dict(precision_bits=bits,relation=rho['budget_relation'],signal=record,rho=rho)
        except ValueError as error:
            if str(error) not in ('NONFINITE_GATE_ENCLOSURE','FINITE_SIGNAL_ENCLOSURE','NONFINITE_ENCLOSURE'):
                raise
            result=dict(precision_bits=bits,relation='NONFINITE',error=str(error))
        C.save(ns/('tier_'+str(bits)+'.json'),result)
        verify()
        return result
    summary=finite_schedule(evaluate)
    ledger=C.read(ns/'call_ledger.json')
    summary.update(SOURCE=contract['SOURCE'],key='PF0_cosine',precision_bits=summary['last']['precision_bits'],
        state_action_attempts=sum(v for k,v in ledger['used'].items() if k.startswith('compiled_')),
        residual_reductions=sum(v for k,v in ledger['used'].items() if k.startswith('residual_')),
        successful_state_actions=sum('signal' in r for r in summary['tiers']),new_C_target_state_actions=0,
        formal_certificate_accepted=False,LOCAL_BUDGET_MET=False,PR_law_rho=None,old_N_G_changed=False,
        forbidden_calls=FORBIDDEN,mandatory_STOP=True,automatic_next_stage=False)
    C.save(ns/'result_summary.json',summary)
    return summary
