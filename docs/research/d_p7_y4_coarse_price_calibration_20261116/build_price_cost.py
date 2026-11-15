"""Join accepted direct receipts, the frozen width-specific model and saved Q3."""
from decimal import Decimal,localcontext
from fractions import Fraction
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts'))
import run_p7_y4_coarse_prices as A
HERE=Path(__file__).resolve().parent


def exact(x):return Fraction(x['numerator'],x['denominator']) if isinstance(x,dict) else Fraction(x)
def enc(x):return dict(numerator=x.numerator,denominator=x.denominator)
def percent(x):
    with localcontext() as c:
        c.prec=28;return format(100*Decimal(x.numerator)/Decimal(x.denominator),'.6f')


def build(calibration,holdout,frozen_commit):
    ref=lambda c,p,*s:dict(commit=c,path=A.STUDY+'/'+p,selector=list(s))
    fixed=A.read(ref(calibration,'INPUTS.json'));frozen=A.read(ref(frozen_commit,'FROZEN_MODEL.json'))
    check=A.check;check(frozen['calibration_publication']==calibration,'Wrong frozen calibration publication')
    q3=A.read(A.Q3_SOURCE);V=A.M.T.c_model()
    check(q3['configuration_id']==A.CONFIG and q3['model_status']=='MODEL_BOUND_MEETS_ADOPTED_THRESHOLD','Wrong Q3 acceptance')
    check(q3['allocation']==fixed['allocation'],'Wrong fixed allocation')
    direct={(r['m'],r['axis']):dict(price=Fraction(r['compiled_RZ']),source=r['receipt'],gate=r['gate'],acquisition='REUSED_EARLY_DIRECT',original_trial=r['original_trial'],original_SOURCE=r['original_SOURCE'],original_cohort=r['original_cohort'],native_source=r['native_source'],proof_of_current_admission=r['proof_of_current_admission']) for r in fixed['early_direct_axes']}
    phase_info={};compiles=0
    for phase,pub,run in [('calibration',calibration,'run02'),('holdout',holdout,'run01')]:
        stem=phase+'/'+run+'/'
        result=A.read(ref(pub,stem+'RESULT.json'));guard=A.read(ref(pub,stem+'GUARD_CHECK.json'))
        prices=A.read(ref(pub,stem+'PRICES.json'))
        check(result['status']=='ACQUIRED_PENDING_C_GUARD' and result['child_exit_code']==0 and not result['error'],'Failed native acquisition')
        check(len(guard['guard']['checked'])==len(prices),'Missing C guard receipts')
        for r in result['results']:
            check(r['ok'] and r['worker_exit_code']==0 and r['trial']['cache_hits']==0,'Invalid/reused scientific trial')
            check(len(r['trial']['checks'])>=7 and len(r['wrappers'])==2,'Missing action/wrapper gate')
            check(r['wrappers']['cosine']['fingerprint']!=r['wrappers']['sine']['fingerprint'],'X/Y wrapper conflation')
            compiles+=len(r['wrappers'])
        if phase=='holdout':
            job=A.read(ref(pub,stem+'JOB_METADATA.json'));check(job['frozen_model_source']==frozen_commit,'Holdout did not use the fixed model')
        phase_info[phase]=dict(result=ref(pub,stem+'RESULT.json'),guard=ref(pub,stem+'GUARD_CHECK.json'),wall_seconds=result['wall_seconds'],resources=result['resources'],worker_resources=result['worker_resources'],context_peak_RSS_bytes=result['context_peak_RSS_bytes'],runtime=result['runtime'])
        for i,p in enumerate(prices):
            check((p['m'],p['axis']) not in direct,'Duplicate accepted direct axis')
            direct[p['m'],p['axis']]=dict(price=Fraction(p['mean']['rz_count']),source=ref(pub,stem+'PRICES.json',i),gate='C_IMPLEMENTATION_GUARD_PASS',acquisition='NEW_DIRECT_'+phase.upper(),original_SOURCE=p['source'],cohort=p['cohort'],compiled_wrapper=p['diagnostic']['compiled_wrapper'])
    check(compiles==6 and len(direct)==16,'Finite compile/direct count differs')
    comparisons=[]
    for axis in ('X','Y'):
        predicted=exact(frozen['frozen_m9_predictions'][axis]);observed=direct[9,axis]['price'];delta=observed-predicted
        check(observed>0,'Undefined relative difference')
        comparisons.append(dict(m=9,q=80,axis=axis,frozen_prediction=enc(predicted),observed_RZ=enc(observed),signed_observed_minus_predicted=enc(delta),signed_relative_to_observed=enc(delta/observed),absolute_relative_to_observed=enc(abs(delta)/observed),status='EXACT_MATCH_AT_Q80' if delta==0 else 'DEVIATION_AT_Q80',tolerance=None,model_refit=False,receipt=direct[9,axis]['source']))
    axes=[];direct_part=Fraction(0);model_part=Fraction(0);long_part=Fraction(0)
    q3rows={(r['m'],r['axis']):r for r in q3['rows']}
    for r in fixed['rows']:
        m,axis=r['m'],r['axis'];c=r['physical_candidate'];N=A.VECTOR[m]
        check(q3rows[m,axis]['physical_id']==r['physical_id'] and q3rows[m,axis]['N']==N,'Q3/price axis binding differs')
        observed=direct.get((m,axis))
        if observed is not None:
            price=observed['price'];part=price*N;direct_part+=part
            evidence={k:v for k,v in observed.items() if k!='price'};kind='DIRECT_COMPILED_DETERMINISTIC';region='DIRECT_OBSERVED_PHYSICAL_POINT'
        else:
            check(c['delta'].hex()==(.64).hex() and m in (7,8,10,11,12),'Model applied to early/foreign width')
            price=V.predict(frozen['models'][axis],dict(one=1,q=c['q']));part=price*N;model_part+=part
            if c['q']>80:long_part+=part
            evidence=dict(source=ref(frozen_commit,'FROZEN_MODEL.json','models',axis),features=dict(one=1,q=c['q']),gate='CONDITIONAL_COMPARISON_REFERENCE_ONLY',acquisition='FROZEN_COARSE_WIDTH_MODEL')
            kind='CONDITIONAL_AFFINE_PRICE_REFERENCE'
            region='UNVALIDATED_EXTRAPOLATION_ABOVE_Q80' if c['q']>80 else 'REFERENCE_WITHIN_OBSERVED_Q_RANGE_NOT_A_DIRECT_OBSERVATION'
        check(price>0,'Nonpositive price reference')
        axes.append(dict(configuration_id=A.CONFIG,physical_id=r['physical_id'],original_candidate_id=q3rows[m,axis]['original_candidate_id'],total_candidate_id=q3rows[m,axis]['total_candidate_id'],m=m,axis=axis,T=c['T'],T_hex=c['T'].hex(),q=c['q'],actual_delta=c['delta'],actual_delta_hex=c['delta'].hex(),N=N,unit_price_RZ=enc(price),N_times_price_RZ=enc(part),kind=kind,region=region,price_evidence=evidence,model_error_bound_RZ=None if observed is None else dict(numerator=0,denominator=1)))
    total=direct_part+model_part
    check(len(axes)==len({(r['m'],r['axis']) for r in axes})==26 and total==sum(exact(r['N_times_price_RZ']) for r in axes),'Cost/axis partition differs')
    baselines=[('SAME_A_SHAPE_CURRENT_Y4',444807010),('CURRENT_P7_Y4_LOWEST_ACCEPTED',435641568),('USER_FIXED_CURRENT_P7_LOWEST_ACCEPTED',395962678)]
    cost=dict(conditional_total_RZ=enc(total),direct_part_RZ=enc(direct_part),model_part_RZ=enc(model_part),model_fraction=enc(model_part/total),model_percent_display=percent(model_part/total),unvalidated_q_above_80_part_RZ=enc(long_part),unvalidated_q_above_80_percent_display=percent(long_part/total),comparisons=[dict(reference=name,reference_RZ=value,difference_coarse_minus_reference_RZ=enc(total-value),source='USER_FIXED_REFERENCE_FOR_THIS_COMPARISON') for name,value in baselines],price_error_bound_RZ=None,CI=None,G_qualified=None,winner=None)
    output=dict(status='CONDITIONAL_WIDTH_SPECIFIC_A_SHAPE_COST',configuration_id=A.CONFIG,allocation_id=q3['allocation']['allocation_id'],allocation=q3['allocation'],Q3_acceptance=A.Q3_SOURCE,Q3_model_status=q3['model_status'],quantity='FULL_MEASURED_CONTROLLED_HADAMARD_WRAPPER_COMPILED_RZ_PER_SHOT_P0_EXCLUDED',model_source=ref(frozen_commit,'FROZEN_MODEL.json'),model_formula=frozen['model_formula'],coefficients={a:m['coefficients'] for a,m in frozen['models'].items()},axes=axes,holdout=comparisons,conditional_cost=cost,phase_sources=phase_info,reused_direct_axes=10,new_compiles=compiles,accepted_direct_axes=16,model_axes=10,model_fits=2,calibration_rounds=[5,6],holdout_round=9,early_fit_excluded=True,m9_refit=False,long_q640_error_bound=None,new_signal=0,new_Q3=0,other_allocations=0,scientific_circuit_retries=0,calibration_initialization_attempts=2,native_context_restores=2,initialization_failed_before_H_restore=1,initialization_failure_source=ref('7f7622a944a6c6b446b959cec54f0cb48ecc9f08','calibration/run01/RESULT.json'),mandatory_stop=True)
    A.write(HERE/'PRICE_AND_COST.json',output)
    print('COEFFICIENTS',output['coefficients']);print('M9',[(r['axis'],r['status'],r['signed_observed_minus_predicted']) for r in comparisons])
    print('COST',cost)


if __name__=='__main__':build(*sys.argv[1:])
