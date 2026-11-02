"""Saved G8 metadata/arithmetic audit; no generator, synthesis or quantum calls."""
from fractions import Fraction as F
from math import factorial,isqrt
from pathlib import Path
import hashlib,json,time

ROOT=Path(__file__).resolve().parents[3]
PREP=ROOT/'artifacts/track_b_g8_on_demand_preparation/2026-10-10'
OUT=ROOT/'artifacts/track_b_g8_on_demand_result/2026-10-10/v1'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ceil(v):return (v.numerator+v.denominator-1)//v.denominator


def independent_log_bounds(q):
    k=0
    while q>=2:q/=2;k+=1
    def series(z):
        lo=2*sum(z**(2*i+1)/(2*i+1) for i in range(90))
        hi=lo+2*z**181/(181*(1-z*z))
        return lo,hi
    a,b=series(F(1,3)),series((q-1)/(q+1))
    return k*a[0]+b[0],k*a[1]+b[1]


def main():
    wall=time.monotonic();c=json.loads((PREP/'contract_v1.json').read_text())
    r=json.loads((OUT/'result_v1.json').read_text());m=json.loads((PREP/'source_manifest_v1.json').read_text())
    source_ok=all(sha(ROOT/p)==h for p,h in m['sha256'].items())
    ledger=json.loads((PREP/'prior_protected_hashes.json').read_text());violations=[]
    for p,record in ledger.items():
        if p.lower().endswith('.npz'):raise PermissionError('NPZ forbidden before access')
        data=(ROOT/p).read_bytes();data=data[:record['bytes']] if p in c['append_only_paths'] else data
        if hashlib.sha256(data).hexdigest()!=record['sha256']:violations.append(p)
    keys=r['on_demand_acquired_sequences'];checks={
        'source_hashes':source_ok,'old_protected_hashes':not violations,
        'marker_contract':sha(OUT/'one_shot_consumed.json')==r['marker_sha256'] and sha(PREP/'contract_v1.json')==r['contract_sha256'],
        'one_shot_complete':len(r['rows'])==8 and r['runs']==1 and r['retries']==0 and r['mandatory_STOP'] and not r['next_science_authorized'],
        'bounded_unique_live_acquisitions':r['synthesis_calls']==len(keys)<=32 and r['acquisition_cache']['bytes']<=1048576,
        'sequence_identity':all(sha_val['sequence_sha256']==hashlib.sha256(sha_val['sequence'].encode()).hexdigest()
            and sha_val['T_count']==sha_val['sequence'].count('T')+sha_val['sequence'].count('t')
            and sha_val['Tdagger_count']==sha_val['sequence'].count('t')
            and sha_val['one_qubit_count']==len(sha_val['sequence'])-sha_val['sequence'].count('W')
            and sha_val['error_pass'] and F(sha_val['strict_operator_error_upper'])<=F(c['primitive_error']) for sha_val in keys.values()),
        'budgets_with_independent_log_arithmetic':True,'accepted_tail_caps':True,
        'cold_warm_semantic_and_cache_bounds':True,'failure_allocation':True,
        'observed_sequences_match_saved_G7':all(r['reference_diagnostic']['observed_sequences_vs_G7_identity'].values()),
        'oracle_coefficient_and_lower_bound_flags':True,
        'resource_caps':r['resource']['wall_seconds']<1200 and r['resource']['cpu_seconds']<900 and r['resource']['peak_RSS_KiB']<=512*1024,
        'scope':r['DF_molecule_NPZ_GPU_LP_quantum_measurement_trajectory_actual_provider_or_circuit_compile']==0}
    exp9=sum(F(9)**i/factorial(i) for i in range(41));assert exp9>8000
    for row in r['rows']:
        b=row['budget'];s=F(b['remaining']);factor=2*F(b['m2_upper'])/s**2+4*F(b['range_upper'])/(3*s)
        lo,hi=independent_log_bounds(2/F(b['alpha_axis']))
        checks['budgets_with_independent_log_arithmetic'] &= ceil(lo*factor)==b['N_per_axis']==ceil(hi*factor)
        checks['failure_allocation'] &= 16*F(b['alpha_axis'])+8*F(b['resource_failure_per_row'])==F(1,20)
        M=b['hard_attempt_cap_two_axes'];v=M*F(b['acceptance_upper_non_enumerative']);cap=b['accepted_call_cap_two_axes']
        checks['accepted_tail_caps'] &= (cap==M or (cap-v)**2>=18*(v+(cap-v)/3)) and cap<=M
        a,t=row['cold_counters'],row['total_counters']
        checks['cold_warm_semantic_and_cache_bounds'] &= (a['attempts']==128 and t['attempts']==256
            and a['accepted']+a['zeros']==128 and t['accepted']==2*a['accepted']
            and row['cold_pass']['digest']==row['warm_pass']['digest']
            and t['row_cache_entries']<=8 and t['row_peak_cache_bytes']<=131072 and row['global_table_or_key_plan_inputs']==0)
    comparisons=[]
    new=r['reference_diagnostic']['finite_provider_rebudget_saved_G7_costs']
    for o in r['reference_diagnostic']['oracle_IS_Rz_only']:
        d=o['diagnostic'];checks['oracle_coefficient_and_lower_bound_flags'] &= (d['all_coefficients_preserved_exactly']
            and not d['production_law_changed'] and not d['G7_primary_classification_changed']
            and F(d['T_Rz_two_axes'])>=F(d['fixed_Bernstein_policy_T_Rz_lower_for_all_positive_proposals']))
        if o['arm']=='full_return':continue
        full=next(v for v in new if v['input']==o['input'] and v['arm']=='full_return')
        value=F(full['T_Rz_two_axes_saved_per_trial_cost']);lower=F(d['fixed_Bernstein_policy_T_Rz_lower_for_all_positive_proposals'])
        comparisons.append({'input':o['input'],'baseline':o['arm'],'local_full_minus_saved_oracle_policy_lower':str(value-lower),
            'local_full_below_saved_policy_lower':value<lower,
            'scope':'Rz-only saved prices/coefficients, common hypothetical delta, fixed Bernstein policy; no physical-shot or method optimality claim'})
    report={'kind':'G8_SAVED_ONLY_PROVENANCE_AND_RATIONAL_ARITHMETIC','passed':all(checks.values()),'checks':checks,
        'source_commit':r['source_commit'],'source_hashes_checked':len(m['sha256']),'protected_paths':len(ledger),
        'protected_violations':violations,'raw_result_sha256':sha(OUT/'result_v1.json'),'marker_sha256':r['marker_sha256'],
        'saved_policy_lower_comparisons':comparisons,'full_oracle_event_certificate_reconstructed':False,
        'oracle_evidence_binding':'frozen source/formal tests and saved scalar output; no oracle regeneration after STOP',
        'additional_generator_synthesis_matrix_quantum_solver_calls':0,'wall_seconds':time.monotonic()-wall,'mandatory_STOP':True}
    with (OUT/'saved_output_audit.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({'passed':report['passed'],'checks':len(checks),'protected_paths':len(ledger)}))
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
