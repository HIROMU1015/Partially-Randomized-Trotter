"""Saved hashes, exact rational interval arithmetic and receipts; no scientific kernels."""
from fractions import Fraction as Q
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[3]
BASE=Path(__file__).resolve().parent
KEY='h4_central_arb_pf_pr0_sine_20261025'

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(o):return hashlib.sha256(json.dumps(o,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def verify():
    checks=[]
    def check(v,n):
        if not v:raise AssertionError(n)
        checks.append(n)
    ns=BASE/'run01';s=read(BASE/'seal.json');c=read(BASE/'active_contract.json');g=read(ns/'grant.json')
    check(s['SOURCE']==c['SOURCE']==g['SOURCE'],'SOURCE_BINDING')
    for p,h in s['code_files'].items():
        check(sha(ROOT/p)==h and hashlib.sha256(subprocess.check_output(['git','show',s['SOURCE']+':'+p],cwd=ROOT)).hexdigest()==h,'CURRENT_AND_COMMITTED_CODE:'+p)
    for p,h in s['bound_manifests'].items():check(sha(BASE/p)==h,'BOUND:'+p)
    runtime=read(BASE/'runtime.json');inputs=read(BASE/'inputs.json')['files'];old=read(BASE/'preservation_before.json')
    check(all(sha(Path(p))==h for p,h in runtime['files'].items()),'RUNTIME_FILES')
    check(all(sha(ROOT/p)==h for p,h in inputs.items()),'INPUT_FILES')
    check(all(sha(ROOT/p)==h for p,h in old.items()),'OLD_EVIDENCE_UNCHANGED')
    prior=read(BASE/'prior_manifest_rows.json');manifest=read(ROOT/'artifacts/validation_manifest.json')
    check(set(manifest)==set(prior)|{KEY} and all(digest(manifest[k])==h for k,h in prior.items()),'OLD_MANIFEST_ROWS_UNCHANGED')
    quoted=read(BASE/'prior_four_bindings.json')
    check(all(sha(ROOT/p)==h for p,h in quoted['files'].items()) and not quoted['worker_cancellation_gate'],'PRIOR_FOUR_QUOTED_UNCHANGED')
    check(g['grant_id']==digest({k:v for k,v in g.items() if k!='grant_id'}),'GRANT_DIGEST')
    check(g['namespace']==str(ns) and g['one_shot'] and g['kind']=='H4_PF_PR0_SINE_ARB_ONE_SHOT','ROOT_ONE_SHOT_SCOPE')
    check(g['contract_sha256']==sha(BASE/'active_contract.json') and g['seal_sha256']==sha(BASE/'seal.json')==c['seal_sha256'],'GRANT_CONTRACT_SEAL')
    check(g['approval_sha256']==c['approval_sha256']==sha(BASE/'approval.md'),'DIRECT_APPROVAL_BINDING')
    check(read(ns/'CONSUMED.json')['grant_id']==g['grant_id'],'SINGLE_ROOT_CONSUMPTION')
    check(c['cases']==['PF_sine','PR0_sine'] and c['axes']==['sine'] and c['max_state_actions']==9,'TWO_SINE_CASES_ONLY')
    result=read(ns/'result_summary.json');summary={};total=0;reduced=0;common_nu=None
    for key,factor,guard in [('PF_sine',1,'25651956654414547227/7631344236317346665332736'),('PR0_sine',2,'102330934580088554157/30525376945269386661330944')]:
        folder=ns/key;target=c['targets'][key];r=read(folder/'result_summary.json');state=read(folder/'state_target_binding.json');ledger=read(folder/'call_ledger.json')
        check(result['targets'][key]==r and r['SOURCE']==s['SOURCE'],'SUMMARY_BINDING:'+key)
        check(target['rho_budget_exact']==guard and target['max_state_actions']==factor*3,'EXACT_GUARD_AND_CASE_CAP:'+key)
        bits=[t['precision_bits'] for t in r['tiers']]
        check(1<=len(bits)<=3 and bits==[64,96,128][:len(bits)],'FINITE_PRECISION_ORDER:'+key)
        check(all(t['relation'] in ('UNDETERMINED','NONFINITE') for t in r['tiers'][:-1]),'NO_ESCALATION_AFTER_DECISION:'+key)
        used=sum(v for k,v in ledger['used'].items() if k.startswith(('target_','compiled_')))
        check(used==r['state_action_attempts']==factor*len(bits)<=factor*3,'NONREFUNDABLE_ACTION_COUNT:'+key)
        check(all(v<=ledger['limits'][k]==1 for k,v in ledger['used'].items()) and ledger['limits']==target['limits'],'PER_PRECISION_LIMITS:'+key)
        check(all(e['reserved_before_call'] for e in ledger['events']),'RESERVATIONS_BEFORE_CALLS:'+key)
        check(sum(v for k,v in ledger['used'].items() if k.startswith('residual_'))==r['residual_reductions']<=3,'RESIDUAL_COUNT:'+key)
        delegation=read(folder/'PARENT_AUTHORIZATION.json')
        check(delegation['grant_id']==g['grant_id'] and not delegation['new_grant'] and not delegation['new_consumption'],'DELEGATED_ROOT_AUTHORITY:'+key)
        check(not (folder/'grant.json').exists() and not (folder/'CONSUMED.json').exists(),'NO_CHILD_GRANT_OR_CONSUMPTION:'+key)
        exit=read(ns/(key+'_native_exit.json'));claim=read(folder/'WORKER_CLAIM.json')
        check(exit['pid']==claim['pid'] and exit['exit_code']==0 and exit['native_wait'] and exit['reaped'],'CASE_NATIVE_JOIN_REAP:'+key)
        check(read(folder/'STOP.json')['status']=='COMPLETE_MANDATORY_STOP' and not (folder/'worker_failure.json').exists(),'CASE_STOP_NO_FAILURE:'+key)
        nu=Q(state['raw_norm_exact']);common_nu=nu if common_nu is None else common_nu
        check(nu==common_nu and nu>0 and state['state_renormalized'] is False,'SAME_EXACT_RAW_NORM:'+key)
        successes=0
        for tier in r['tiers']:
            p=str(tier['precision_bits'])
            if key=='PF_sine':
                signal=read(folder/('signal_'+p+'.json'))
                check(signal==tier['signal'] and signal['SOURCE']==s['SOURCE'] and Q(signal['raw_norm_exact'])==nu,'PF_COMPILED_SOURCE_NORM:'+p)
                check(not any(k.startswith('target_') for k in ledger['limits']) and state['new_C_target_state_actions']==0,'PF_ZERO_TARGET_ACTIONS')
                ci=state['C_target_interval_exact'];successes+=1
            else:
                for role in ('target','compiled'):
                    saved=read(folder/(role+'_'+p+'.json'));action_claim=read(folder/(role+'_'+p+'_worker_claim.json'));receipt=read(folder/(role+'_'+p+'_native_exit.json'))
                    check(receipt['pid']==action_claim['pid'] and receipt['exit_code']==0 and receipt['native_wait'] and receipt['reaped'],'ACTION_NATIVE_JOIN_REAP:'+role+p)
                    event=next(e for e in ledger['events'] if e['call']==role+'_'+p)
                    check(event['UTC']<=action_claim['UTC']<=receipt['UTC'],'ACTION_RESERVATION_ORDER:'+role+p)
                    if saved['status']=='FINITE':
                        record=saved['record'];successes+=1
                        check(record==tier[role] and record['SOURCE']==s['SOURCE'] and Q(record['raw_norm_exact'])==nu,'ACTION_SOURCE_NORM:'+role+p)
                check(not tier['target']['PR_law_mean_used'] and not tier['target']['normalization_B_applied_to_target'],'INDIVIDUAL_UNSCALED_C_TARGET')
                ci=tier['target']['interval'];signal=tier['compiled']
            check(signal['real_H4_state_actions']==1 and signal['global_phase_preserved'] and not signal['clipping'],'COMPILED_CONVENTIONS:'+key+p)
            cl,cu=Q(ci['lower']),Q(ci['upper']);sl,su=Q(signal['interval']['lower']),Q(signal['interval']['upper']);rho=tier['rho']
            check(cl<=cu and sl<=su,'ORDERED_WHOLE_INTERVALS:'+key+p)
            dl,du=sl-cu,su-cl;scale=Q(target['B_saved_exact'])
            lower=(Q(0) if dl<=0<=du else min(abs(dl),abs(du)))*scale;upper=max(abs(dl),abs(du))*scale
            rl,ru=Q(rho['interval']['lower']),Q(rho['interval']['upper'])
            if key=='PF_sine':
                # PF reduces the live Arb signal, BEFORE endpoints() outward
                # rounds it for JSON. Its rho need not contain every point of
                # that larger serialized Cartesian product. Check compatibility
                # and the conservative saved-endpoint bound separately; do not
                # invent a tolerance, change the rho ball or rerun the walk.
                check(rl<=upper and lower<=ru,'PF_LIVE_BALL_SERIALIZED_HULL_COMPATIBILITY:'+p)
                check(upper<=Q(guard),'PF_SERIALIZED_ENDPOINT_BOUND_ALSO_WITHIN_SAME_GUARD:'+p)
                summary['PF_saved_endpoint_check']=dict(upper_exact=str(upper),
                    operation='pure rational audit of outward serialized endpoints; not a new Arb reduction',
                    live_ball_rho_upper_exact=str(ru),live_ball_and_serialized_hull_distinguished=True)
            else:
                check(rl<=lower<=upper<=ru,'DIRECT_INTERVAL_DIFFERENCE_B_ONCE:'+key+p)
            check(read(folder/('rho_'+p+'.json'))==rho and rho['B_saved_exact']==target['B_saved_exact']
                and (key=='PF_sine' or rho['B_applied_exactly_once']),'RHO_SAVED_B_ONCE:'+key+p)
            check(Q(rho['interval_width_exact'])==ru-rl and Q(signal['interval_width_exact'])==su-sl,'SAVED_WIDTHS:'+key+p)
            rel='UPPER_WITHIN_BUDGET' if ru<=Q(guard) else 'LOWER_EXCEEDS_BUDGET' if rl>Q(guard) else 'UNDETERMINED'
            check(rel==tier['relation']==rho['budget_relation'],'EXACT_GUARD_COMPARISON:'+key+p)
            check(not rho['formal_certificate_accepted'] and not rho['LOCAL_BUDGET_MET'],'NUMERICAL_LOCAL_ONLY:'+key+p)
        check(successes==r['successful_state_actions'],'SUCCESS_COUNT:'+key)
        check(not r['formal_certificate_accepted'] and not r['LOCAL_BUDGET_MET'] and r['PR_law_rho'] is None and not r['old_N_G_changed'],'NO_FORMAL_GLOBAL_NG_PROMOTION:'+key)
        check(all(v==0 for v in r['forbidden_calls'].values()),'FORBIDDEN_CALLS_ZERO:'+key)
        total+=used;reduced+=r['residual_reductions'];summary[key]=dict(precision_bits=bits,state_action_attempts=used,relation=r['relation'],rho_upper_exact=r['last']['rho']['interval']['upper'])
    pf=read(BASE/'PF_sine/PF_C_target_interval_QUOTED.json');state=read(ns/'PF_sine/state_target_binding.json')
    original=read(ROOT/'artifacts/non_df_h4_c_reference_evidence/2026-10-23/run01/pauli_jw_result.json')
    raw=original['implemented_law']['PF']['corrected_encoded_mean']['imag']
    check(pf['saved_raw_interval']==raw and pf['normalized_interval_exact']==[state['C_target_interval_exact'][side] for side in ('lower','upper')],'PF_WHOLE_SAVED_IMAG_QUOTE')
    check(all(Q(state['C_target_interval_exact'][side])==Q(raw[side])/common_nu for side in ('lower','upper')),'PF_FULL_INTERVAL_EXACT_NORM_TRANSFER')
    check(result['state_action_attempts']==total<=9 and result['residual_reductions']==reduced<=6,'TOTAL_FINITE_BUDGETS')
    check(not result['formal_certificate_accepted'] and not result['LOCAL_BUDGET_MET'] and result['PR_law_rho'] is None and not result['old_N_G_changed'],'ROOT_NO_PROMOTION')
    native=read(ns/'native/worker_native_exit.json');native_stop=read(ns/'native/STOP.json');final=read(ns/'STOP.json')
    check(native['pid']==read(ns/'WORKER_CLAIM.json')['pid']==read(ns/'native/worker_identity.json')['pid'],'ROOT_WORKER_PID_BINDING')
    check(native['exit_code']==0 and native['reaped'] and native_stop['all_owned_reaped'] and not native_stop['descendants_terminated'],'ROOT_NATIVE_EXIT_ALL_REAPED')
    check(final['status']=='COMPLETE_MANDATORY_STOP' and not final['automatic_next_stage'] and not final['original_H4_native_verified'],'MANDATORY_STOP_OLD_NATIVE_UNKNOWN')
    plan=read(BASE/'PR0_sine/PR0_selected_C_plan_NO_EXECUTION.json');preflight=read(BASE/'structural_preflight.json')
    check(len(plan['events'])==32 and all(e['taylor_order']==0 for e in plan['events']) and plan['axis']=='sine','PR0_FIXED_ALLORDER0_IMAG_PLAN')
    check(preflight['QPY_decodes']==10 and preflight['state_actions']==0 and not preflight['empirical_cache_used'],'STATIC_ONLY_PREFLIGHT')
    for key,record in zip(c['cases'],preflight['records']):
        check(record['axis']=='sine' and record['convention']=='+Im using Sdg before final H' and record['output']['sha256']==c['targets'][key]['compiled_sha256'],'ACTUAL_SAVED_SINE_QPY:'+key)
    reused=read(BASE/'reuse.json')['files']
    check(all(sha(ROOT/p)==h for p,h in reused.items()),'REUSED_CODE_UNCHANGED')
    art=read(BASE/'artificial_verification.json')
    check(art['total']==37 and all(sha(BASE/p)==h for p,h in art['log_sha256'].items()),'ARTIFICIAL_37_PASS_LOGS')
    check(not read(BASE/'policy_observation.json')['section_10_3_present'],'POLICY_VERSION_DIFFERENCE_RECORDED')
    return dict(status='PASS_SAVED_ENCLOSURES_AND_NATIVE_EXIT',SOURCE=s['SOURCE'],checks=checks,check_count=len(checks),
        code_files=len(s['code_files']),runtime_files=len(runtime['files']),saved_inputs=len(inputs),preserved_files=len(old),prior_manifest_rows=len(prior),
        targets=summary,state_action_attempts=total,residual_reductions=reduced,scientific_grants_issued=1,scientific_grants_consumed=1,
        scientific_failures=0,formal_certificate_accepted=False,PR_law_rho=None,original_native_status='UNKNOWN_UNCHANGED',mandatory_STOP=True)

if __name__=='__main__':print(json.dumps(verify(),indent=2,sort_keys=True))
