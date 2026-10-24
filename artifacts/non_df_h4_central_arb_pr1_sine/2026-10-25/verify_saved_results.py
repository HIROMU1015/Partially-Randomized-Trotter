"""Hashes, saved rational intervals and native receipts only; no Arb import."""
from fractions import Fraction as Q
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[3]
BASE=Path(__file__).resolve().parent
KEY='h4_central_arb_pr1_sine_20261025'


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(o):return hashlib.sha256(json.dumps(o,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def verify():
    checks=[]
    def check(value,name):
        if not value:raise AssertionError(name)
        checks.append(name)
    ns=BASE/'run01';s=read(BASE/'seal.json');c=read(BASE/'active_contract.json');g=read(ns/'grant.json')
    check(s['SOURCE']==c['SOURCE']==g['SOURCE'],'SCIENTIFIC_SOURCE_BINDING')
    for path,h in s['code_files'].items():
        check(sha(ROOT/path)==h,'CODE_CURRENT:'+path)
        check(hashlib.sha256(subprocess.check_output(['git','show',s['SOURCE']+':'+path],cwd=ROOT)).hexdigest()==h,'CODE_COMMITTED:'+path)
    for name,h in s['bound_manifests'].items():check(sha(BASE/name)==h,'BOUND:'+name)
    runtime=read(BASE/'runtime.json')
    check(all(sha(Path(p))==h for p,h in runtime['files'].items()),'RUNTIME_UNCHANGED')
    inputs=read(BASE/'inputs.json')['files'];old=read(BASE/'preservation_before.json')
    check(all(sha(ROOT/p)==h for p,h in inputs.items()),'SCIENTIFIC_INPUTS_UNCHANGED')
    check(all(sha(ROOT/p)==h for p,h in old.items()),'OLD_EVIDENCE_UNCHANGED')
    prior=read(BASE/'prior_manifest_rows.json');manifest=read(ROOT/'artifacts/validation_manifest.json')
    check(set(manifest)==set(prior)|{KEY} and all(digest(manifest[k])==h for k,h in prior.items()),'OLD_MANIFEST_ROWS_UNCHANGED')
    check(g['grant_id']==digest({k:v for k,v in g.items() if k!='grant_id'}),'NEW_GRANT_DIGEST')
    check(g['namespace']==str(ns) and g['one_shot'],'NEW_GRANT_NAMESPACE')
    check(g['contract_sha256']==sha(BASE/'active_contract.json') and g['seal_sha256']==sha(BASE/'seal.json')==c['seal_sha256'],'GRANT_SEAL_CONTRACT')
    check(g['approval_sha256']==c['approval_sha256']==sha(BASE/'approval.md'),'GRANT_APPROVAL')
    check(read(ns/'CONSUMED.json')['grant_id']==g['grant_id'],'ONE_SHOT_CONSUMED')
    result=read(ns/'result_summary.json');ledger=read(ns/'call_ledger.json');state=read(ns/'state_target_binding.json')
    bits=[t['precision_bits'] for t in result['tiers']]
    check(bits==[64,96,128][:len(bits)] and 1<=len(bits)<=3,'FINITE_PRECISION_SCHEDULE')
    check(all(t['relation'] in ('UNDETERMINED','NONFINITE') for t in result['tiers'][:-1]),'NO_ESCALATION_AFTER_DETERMINATION')
    check(sum(v for k,v in ledger['used'].items() if k.startswith(('target_','compiled_')))==result['state_action_attempts']==2*len(bits)<=6,'NONREFUNDABLE_STATE_ACTION_COUNT')
    check(all(v<=ledger['limits'][k]==1 for k,v in ledger['used'].items()),'PER_PRECISION_LIMITS')
    check(all(e['reserved_before_call'] for e in ledger['events']),'RESERVATIONS_PRECEDE_CALLS')
    check(not (ns/'worker_failure.json').exists(),'NO_SCIENTIFIC_WORKER_FAILURE')
    count=0
    for tier in result['tiers']:
        p=str(tier['precision_bits'])
        for role in ('target','compiled'):
            saved=read(ns/(role+'_'+p+'.json'));claim=read(ns/(role+'_'+p+'_worker_claim.json'));receipt=read(ns/(role+'_'+p+'_native_exit.json'))
            check(receipt['pid']==claim['pid'] and receipt['exit_code']==0 and receipt['native_wait'] and receipt['reaped'],'ACTION_NATIVE_WAIT_REAP:'+role+'_'+p)
            event=next(e for e in ledger['events'] if e['call']==role+'_'+p)
            check(event['UTC']<=claim['UTC']<=receipt['UTC'],'RESERVATION_ACTION_EXIT_ORDER:'+role+'_'+p)
            if saved['status']=='FINITE':
                count+=1;record=saved['record']
                check(record==tier[role] and record['SOURCE']==s['SOURCE'] and record['real_H4_state_actions']==1,'SAVED_ACTION_SOURCE:'+role+'_'+p)
                check(Q(record['raw_norm_exact'])==Q(state['raw_norm_exact']),'COMMON_EXACT_RAW_NU:'+role+'_'+p)
        if 'rho' not in tier:continue
        t,compiled,rho=tier['target'],tier['compiled'],tier['rho']
        cl,cu=map(Q,[t['interval']['lower'],t['interval']['upper']]);sl,su=map(Q,[compiled['interval']['lower'],compiled['interval']['upper']])
        check(cl<=cu and sl<=su,'FINITE_ORDERED_SIGNAL_INTERVALS_'+p)
        dl,du=sl-cu,su-cl;scale=Q(c['B_saved_exact'])
        lower=(Q(0) if dl<=0<=du else min(abs(dl),abs(du)))*scale;upper=max(abs(dl),abs(du))*scale
        rl,ru=Q(rho['interval']['lower']),Q(rho['interval']['upper']);guard=Q(c['rho_budget_exact'])
        check(rl<=lower<=upper<=ru,'DIRECT_WHOLE_INTERVAL_DIFFERENCE_B_ONCE_'+p)
        check(read(ns/('rho_'+p+'.json'))==rho and rho['B_applied_exactly_once'] and not t['normalization_B_applied_to_target'] and not t['PR_law_mean_used'],'UNSCALED_SELECTED_TARGET_NO_LAW_MEAN_'+p)
        check(rho['nonnegative_intersection']==dict(lower=str(max(Q(0),rl)),upper=str(ru)) and rho['raw_ball_preserved'],'RAW_RHO_BALL_AND_NONNEGATIVE_INTERSECTION_'+p)
        relation='UPPER_WITHIN_BUDGET' if ru<=guard else 'LOWER_EXCEEDS_BUDGET' if rl>guard else 'UNDETERMINED'
        check(relation==tier['relation']==rho['budget_relation'],'EXACT_RATIONAL_BUDGET_COMPARISON_'+p)
        check(not rho['LOCAL_BUDGET_MET'] and not rho['formal_certificate_accepted'],'LOCAL_NUMERICAL_ONLY_'+p)
    check(count==result['successful_state_actions'],'SUCCESSFUL_STATE_ACTION_COUNT')
    check(result['formal_certificate_accepted'] is False and result['LOCAL_BUDGET_MET'] is False and result['PR_law_rho'] is None and not result['old_N_G_changed'],'NO_CERTIFICATE_OR_GLOBAL_PROMOTION')
    check(all(v==0 for v in result['forbidden_calls'].values()),'FORBIDDEN_CALLS_ZERO')
    native=read(ns/'native/worker_native_exit.json');stop=read(ns/'native/STOP.json');final=read(ns/'STOP.json')
    check(native['pid']==read(ns/'WORKER_CLAIM.json')['pid']==read(ns/'native/worker_identity.json')['pid'],'WORKER_PID_BINDING')
    check(native['exit_code']==0 and native['reaped'] and stop['all_owned_reaped'] and not stop['descendants_terminated'],'NATIVE_EXIT_REAP')
    check(final['status']=='COMPLETE_MANDATORY_STOP' and not final['automatic_next_stage'] and not final['original_H4_native_verified'],'CURRENT_STOP_NOT_ORIGINAL_NATIVE_REPLACEMENT')
    prep=ROOT/'artifacts/non_df_h4_central_arb_pr0_preparation/2026-10-25'
    disabled=read(prep/'pilot_contract_DRAFT_DISABLED.json')
    check(all(disabled[k] is False for k in ('execution_authorized','dispatch_enabled','grant_valid')),'OLD_PREPARATION_CONTRACT_STAYS_DISABLED')
    plan=read(BASE/'PR1_selected_C_plan_NO_EXECUTION.json');preflight=read(BASE/'structural_preflight.json')
    original=next(t for t in read(ROOT/'artifacts/non_df_h4_central_r1_sine/2026-10-24/inputs.json')['targets'] if t['key']=='PR1_sine')
    check(read(BASE/'PR1_target_QUOTED.json')==original,'ORIGINAL_R1_SINE_TARGET_QUOTE')
    check(plan['trajectory_id']==original['trajectory_id']==c['trajectory_id'] and plan['event_record_hash']==original['event_record_hash']
        and plan['trajectory_record_hash']==original['trajectory_record_hash'],'FIXED_PR1_TRAJECTORY_HASHES')
    check([e['taylor_order'] for e in plan['events']]==[0]*30+[2,0],'FIXED_ORDER2_PROFILE')
    event=plan['events'][30];components=plan['components']
    check(event['product_component_ids']==['43','4'] and event['rotation_component_id']=='72'
        and [components[i]['sign'] for i in ('43','4','72')]==[1,-1,-1]
        and [components[i]['label'] for i in ('43','4','72')]==['IYXXYIII','IIIZIIII','YYIIIIXX']
        and Q(event['order_phase_exact'])==-1 and Q(event['rotation_angle_exact'])==Q(.029672025283184623),'FIXED_ORDER2_SIGNS_ORDER_PHASE_ANGLE')
    check(preflight['QPY_decodes']==6 and preflight['state_actions']==0 and preflight['new_compile']==0,'STATIC_ONLY_PREFLIGHT')
    check(preflight['exact_plan_sha256']==sha(BASE/'PR1_selected_C_plan_NO_EXECUTION.json')
        and preflight['descriptor_sha256']==sha(BASE/'PR1_compiled_program_NO_EXECUTION.json'),'EXACT_PLAN_DESCRIPTOR_HASHES')
    binding=read(BASE/'structural_inspection.json');sine=binding['sine_binding']
    check(sine['aggregate_phase']==original['aggregate_source_phase']==.26518225073344787
        and sine['compiled_qpy']['global_phase']==2.4887856155591095,'SAVED_PR1_SINE_RELATIVE_AND_GLOBAL_PHASES')
    check(binding['order2']['signs']==[1,-1,-1] and binding['order2']['order_phase']==dict(real=-1.,imag=0.),'R1_SOURCE_ORDER2_BINDING')
    check(plan['axis']=='sine' and plan['domain']=='SAVED_PR1_SINE_NO_EXECUTION' and sine['axis']=='sine'
        and sine['convention']=='+Im using Sdg before final H' and binding['cosine_state_actions']==0,'SINE_AXIS_SDG_PLUS_IM_ONLY')
    descriptor=read(BASE/'PR1_compiled_program_NO_EXECUTION.json')
    check(descriptor['axis']=='sine' and descriptor['origin']['qpy_sha256']==sine['compiled_qpy']['sha256']==original['output_sha256']
        and descriptor['origin']['fingerprint']==sine['compiled_qpy']['fingerprint']==original['output_fingerprint'],'ACTUAL_SAVED_SINE_COMPILED_IDENTITY')
    quoted=read(BASE/'PR1_guard_QUOTED.json')
    original_guard=next(g for g in read(ROOT/'artifacts/non_df_h4_central_rho_preparation/2026-10-24/guard_inputs_quoted.json')['axes']
        if (g['case'],g['epsilon_sig'],g['axis'])==('pauli_jw_PR','0.0005','sine'))
    check(quoted==original_guard and quoted['rho_guard_exact']==c['rho_budget_exact']=='102330934580088554157/30525376945269386661330944',
        'ORIGINAL_SINE_BUDGET_NO_COSINE_SUBSTITUTION')
    check(c['target_key']==result['key']=='PR1_sine' and c['axes']==['sine']
        and g['kind']=='H4_PR1_SINE_ARB_ONE_SHOT' and c['kind']=='H4_PR1_SINE_ARB_ACTIVE_V1','SINE_AUTHORITY_KIND_SCOPE')
    check(all(t['target']['axis']=='sine' and t['target']['cosine_evaluated'] is False
        and t['compiled']['target_key']=='PR1_sine' for t in result['tiers'] if 'rho' in t),'NO_COSINE_SCIENTIFIC_ACTION')
    reused=read(BASE/'reuse.json')['files']
    check(all(sha(ROOT/p)==h and hashlib.sha256(subprocess.check_output(['git','show',c['base_commit']+':'+p],cwd=ROOT)).hexdigest()==h
        for p,h in reused.items()),'OLD_SOURCE_KERNELS_AND_R1_UNCHANGED')
    art=read(BASE/'artificial_verification.json')
    check(art['total']==43 and all(sha(BASE/p)==h for p,h in art['log_sha256'].items()),'ARTIFICIAL_43_LOG_BINDING')
    check(not read(BASE/'policy_observation.json')['section_10_3_present'],'POLICY_10_3_VERSION_MISMATCH_RECORDED')
    return dict(status='PASS_SAVED_ENCLOSURE_BINDING_AND_EXIT_AUDIT',checks=checks,check_count=len(checks),SOURCE=s['SOURCE'],
        code_files=len(s['code_files']),preserved_files=len(old),saved_inputs=len(inputs),runtime_files=len(runtime['files']),prior_manifest_rows=len(prior),
        precision_bits=bits,state_action_attempts=result['state_action_attempts'],successful_state_actions=count,residual_reductions=result['residual_reductions'],
        scientific_grants_issued=1,scientific_grants_consumed=1,scientific_worker_failures=0,
        local_rho_upper=result['last'].get('rho',{}).get('interval',{}).get('upper'),relation=result['relation'],
        formal_certificate_accepted=False,original_native_status='UNKNOWN_UNCHANGED',mandatory_STOP=True)


if __name__=='__main__':print(json.dumps(verify(),indent=2,sort_keys=True))
