"""Saved rational endpoints, hashes and native records only; no Arb/state action."""
from fractions import Fraction as Q
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[3]
BASE=Path(__file__).resolve().parent
KEY='h4_central_arb_pf_cosine_20261025'


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
    inputs=read(BASE/'inputs.json')['files']
    check(all(sha(ROOT/p)==h for p,h in inputs.items()),'SCIENTIFIC_INPUTS_UNCHANGED')
    old=read(BASE/'preservation_before.json')
    check(all(sha(ROOT/p)==h for p,h in old.items()),'OLD_EVIDENCE_UNCHANGED')
    prior=read(BASE/'prior_manifest_rows.json');manifest=read(ROOT/'artifacts/validation_manifest.json')
    check(set(manifest)==set(prior)|{KEY} and all(digest(manifest[k])==h for k,h in prior.items()),'OLD_MANIFEST_ROWS_UNCHANGED')
    check(g['grant_id']==digest({k:v for k,v in g.items() if k!='grant_id'}),'NEW_GRANT_DIGEST')
    check(g['contract_sha256']==sha(BASE/'active_contract.json') and g['seal_sha256']==sha(BASE/'seal.json')==c['seal_sha256'],'GRANT_SEAL_CONTRACT')
    check(read(ns/'CONSUMED.json')['grant_id']==g['grant_id'],'ONE_SHOT_CONSUMED')
    result=read(ns/'result_summary.json');ledger=read(ns/'call_ledger.json')
    bits=[t['precision_bits'] for t in result['tiers']]
    check(bits==[64,96,128][:len(bits)] and 1<=len(bits)<=3,'FINITE_PRECISION_SCHEDULE')
    check(all(t['relation'] in ('UNDETERMINED','NONFINITE') for t in result['tiers'][:-1]),'NO_ESCALATION_AFTER_DETERMINATION')
    check(sum(v for k,v in ledger['used'].items() if k.startswith('compiled_'))==result['state_action_attempts']==len(bits),'NONREFUNDABLE_STATE_ACTION_COUNT')
    check(all(v<=ledger['limits'][k]==1 for k,v in ledger['used'].items()),'PER_PRECISION_LIMITS')
    check(not (ns/'worker_failure.json').exists(),'NO_SCIENTIFIC_WORKER_FAILURE')
    target=read(ns/'state_target_binding.json')['C_target_interval_exact']
    for tier in result['tiers']:
        if 'rho' not in tier:continue
        signal=tier['signal'];rho=tier['rho'];sl,su=map(Q,[signal['interval']['lower'],signal['interval']['upper']])
        cl,cu=Q(target['lower']),Q(target['upper']);dl,du=sl-cu,su-cl
        lower=Q(0) if dl<=0<=du else min(abs(dl),abs(du));upper=max(abs(dl),abs(du))
        rl,ru=Q(rho['interval']['lower']),Q(rho['interval']['upper']);budget=Q(c['rho_budget_exact'])
        check(rl<=lower<=upper<=ru,'DIRECT_WHOLE_TARGET_INTERVAL_DIFFERENCE_'+str(tier['precision_bits']))
        relation='UPPER_WITHIN_BUDGET' if ru<=budget else 'LOWER_EXCEEDS_BUDGET' if rl>budget else 'UNDETERMINED'
        check(relation==tier['relation']==rho['budget_relation'],'EXACT_RATIONAL_BUDGET_COMPARISON_'+str(tier['precision_bits']))
        check(Q(signal['raw_norm_exact'])==Q(read(ns/'state_target_binding.json')['raw_norm_exact']),'SAME_RAW_NU_'+str(tier['precision_bits']))
    check(result['formal_certificate_accepted'] is False and result['LOCAL_BUDGET_MET'] is False and result['PR_law_rho'] is None and not result['old_N_G_changed'],'NO_CERTIFICATE_OR_GLOBAL_PROMOTION')
    native=read(ns/'native/worker_native_exit.json');stop=read(ns/'native/STOP.json');final=read(ns/'STOP.json')
    check(native['pid']==read(ns/'WORKER_CLAIM.json')['pid']==read(ns/'native/worker_identity.json')['pid'],'WORKER_PID_BINDING')
    check(native['exit_code']==0 and native['reaped'] and stop['all_owned_reaped'] and not stop['descendants_terminated'],'NATIVE_EXIT_REAP')
    check(final['status']=='COMPLETE_MANDATORY_STOP' and not final['automatic_next_stage'] and not final['original_H4_native_verified'],'CURRENT_STOP_NOT_ORIGINAL_NATIVE_REPLACEMENT')
    setup=read(BASE/'pre_grant_STOP.json')
    check(setup['science_state_actions']==setup['science_grants_issued']==setup['science_grants_consumed']==0,'SETUP_RECOVERY_NO_SCIENTIFIC_CALL')
    for name in ('pre_grant_STOP.json','pre_grant_runtime_diagnosis.json'):
        path=str((BASE/name).relative_to(ROOT));b=subprocess.check_output(['git','show',s['SOURCE']+':'+path],cwd=ROOT)
        check(hashlib.sha256(b).hexdigest()==sha(BASE/name),'PRE_GRANT_FAILURE_PRESERVED:'+name)
    return dict(status='PASS_SAVED_ENCLOSURE_BINDING_AND_EXIT_AUDIT',checks=checks,check_count=len(checks),SOURCE=s['SOURCE'],
        preserved_files=len(old),saved_inputs=len(inputs),runtime_files=len(runtime['files']),prior_manifest_rows=len(prior),
        precision_bits=bits,state_action_attempts=result['state_action_attempts'],successful_state_actions=result['successful_state_actions'],
        scientific_grants_issued=1,scientific_grants_consumed=1,pre_grant_setup_failures=1,scientific_worker_failures=0,
        local_rho_upper=result['last'].get('rho',{}).get('interval',{}).get('upper'),relation=result['relation'],
        formal_certificate_accepted=False,original_native_status='UNKNOWN_UNCHANGED',mandatory_STOP=True)


if __name__=='__main__':print(json.dumps(verify(),indent=2,sort_keys=True))
