"""Fixed Full-H F1 scalar diagnostic; reuse D7 exact predict, never fit/compile."""
import argparse
from copy import deepcopy
import csv
from datetime import datetime, timezone
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import time

from trottertracks import price_implementation_guard as G
from admit_pr_df_ld1_q3_prices import equal, require
import freeze_full_h_d7_models as D7
import admit_full_h_d7_holdout as A

ROOT=Path(__file__).resolve().parents[1]
BASE='docs/research/full_h_f1_model_diagnostic_20261115'
ADMISSION=dict(commit='61f249e6e18d5afab9ba8797bef5e8f9f14e29de',path=A.BASE+'/ADMISSION.json')
PLAN=dict(commit=ADMISSION['commit'],path=A.BASE+'/NEXT_SCIENCE_PLAN.md')
PF=dict(commit='1002f1dfea34de439ca1b704a0192bca473c5387',path='docs/research/pf6_allround_cost_20261115/RESULT.json')
BENCHMARKS={'B0I_P7_S2_16h_over_7':593555352,'FULL_PF_S2_16h_over_7':838693580}
MODULE='scripts/run_full_h_f1_diagnostic.py'


def exact(v):
    v=Fraction(v)
    return dict(numerator=v.numerator,denominator=v.denominator)


def rational(v):
    require(isinstance(v,dict) and set(v)=={'numerator','denominator'} and
            type(v['numerator']) is int and type(v['denominator']) is int and v['denominator']>0,'Exact rational required')
    return Fraction(v['numerator'],v['denominator'])


def evaluate_axis(row,model,predictions,features,predict):
    """One source-bound scenario quote. A negative value is kept, never clamped."""
    m,axis=row['m'],row['axis'];p=row['physical_candidate']
    equal((model['model'],model['feature_map'],model['features']),
          ('PRICE_LINEAR_FEATURE_MODEL_V1','FULL_H_GROUP_V1',['one','event_count','outer_boundary_count']), 'D7 formula/unit')
    equal(model['domain']['route'],'FULL_H_RTE','Full-H model route')
    equal(model['domain']['axis'],axis,'Axis-specific model')
    equal((p['family'],p['L_D'],p['K']),('FULL_H_PURE_RTE',0,2),'Full-H physical domain')
    equal((row['q_grouping'],row['R_outer'],row['event_positions']),
          (p['q'],p['R_outer'],p['s']),'Saved grouping/event units')
    f=features(dict(q=p['q'],R=p['R_outer']),'FULL_H_GROUP_V1')
    equal(f['event_count'],p['s'],'Original FULL_H_GROUP_V1 event count')
    if m in (5,6):
        found=[(i,v) for i,v in enumerate(predictions) if (v['m'],v['axis'])==(m,axis)]
        equal(len(found),1,'One saved prediction per holdout axis')
        i,saved=found[0]
        equal(saved['physical'],p,'Saved prediction physical descriptor')
        equal(saved['features'],{k:exact(v) for k,v in f.items()},'Saved prediction feature units')
        value=rational(saved['predicted_RZ']);origin='REUSED_PREOBSERVATION_FROZEN_PREDICTION'
        locator=A.at(A.FROZEN,'predictions',i,'predicted_RZ')
    else:
        value=predict(model,f);origin='NEW_EXACT_FIXED_COEFFICIENT_EVALUATION';locator=None
    return dict(m=m,axis=axis,physical_id=p['id'],original_configuration_id=row['original_configuration_id'],
        total_configuration_id=row['total_configuration_id'],original_candidate_id=row['original_candidate_id'],
        total_candidate_id=row['total_candidate_id'],allocation_id=row['allocation_id'],N=row['N'],
        physical_candidate=deepcopy(p),T_exact=deepcopy(row['T_exact']),q_grouping=p['q'],q_PF=None,R_outer=p['R_outer'],
        features={k:exact(v) for k,v in f.items()},model_source=deepcopy(row['frozen_model']),
        admission_source=deepcopy(row['admission_source']),B_join=deepcopy(row['B_join']),
        C_ref=exact(value),origin=origin,reused_prediction=locator,
        gates=dict(SOURCE='PASS',physical_domain='PASS',feature_units='PASS',
                   nonnegative='PASS' if value>=0 else 'FAIL_NEGATIVE_REFERENCE'),
        status='MODEL_SCENARIO_ONLY' if value>=0 else 'OUT_OF_DOMAIN_NEGATIVE_MODEL_REFERENCE',
        reason=None if value>=0 else 'Unchanged frozen model gives a negative compiled-RZ reference; no clamp or replacement',
        scientific_model_qualification=False,qualified_for_candidate=False,E_C=None,CI=None)


def summarize(rows,benchmarks):
    equal(len(rows),26,'Exactly 26 axes before any total')
    equal({(r['m'],r['axis']) for r in rows},{(m,a) for m in range(13) for a in ('X','Y')},'Unique full schedule')
    invalid=[dict(m=r['m'],axis=r['axis'],physical_id=r['physical_id'],status=r['status'],reason=r['reason'])
             for r in rows if r['status']!='MODEL_SCENARIO_ONLY']
    total=None
    if not invalid:
        total=sum((r['N']*rational(r['C_ref']) for r in rows),Fraction(0))
    full=[]
    for b in benchmarks:
        full.append(dict(benchmark=deepcopy(b),T_ref=exact(total) if total is not None else None,
            D_PF=exact(Fraction(b['RZ'])-total) if total is not None else None,
            Delta_lower_cost_condition='Delta < D_PF' if total is not None else None,
            Delta_equal_cost_condition='Delta = D_PF' if total is not None else None,
            Delta_higher_cost_condition='Delta > D_PF' if total is not None else None,
            status='MODEL_SCENARIO_ONLY' if total is not None else 'NOT_EVALUATED_INVALID_AXIS'))
    m6=[r for r in rows if r['m']==6]
    equal({r['axis'] for r in m6},{'X','Y'},'m6 X/Y required')
    require(all(r['N']==57 for r in m6),'Fixed m6 N=57')
    partial=None;comparisons=[]
    if all(r['status']=='MODEL_SCENARIO_ONLY' for r in m6):
        partial=57*sum((rational(r['C_ref']) for r in m6),Fraction(0))
        for b in benchmarks:
            difference=Fraction(b['RZ'])-partial
            comparisons.append(dict(benchmark=deepcopy(b),PF_minus_m6_reference=exact(difference),
                m6_reference_relation_to_PF='ABOVE' if difference<0 else 'BELOW' if difference>0 else 'EQUAL',
                mother_mean_lower_bound_condition='57*(L_6X+L_6Y) > T_PF implies T_I > T_PF',
                required_axis_mean_sum_strictly_greater_than=exact(Fraction(b['RZ'],57)),
                current_m6_is_mother_mean_lower_bound=False,rank_assessed=False))
    return dict(status='MODEL_SCENARIO_ONLY' if total is not None else 'TOTAL_UNAVAILABLE_INVALID_AXIS',
        valid_axes=26-len(invalid),invalid_axes=invalid,T_ref=exact(total) if total is not None else None,
        full_total_comparisons=full,m6=dict(status='MODEL_SCENARIO_ONLY' if partial is not None else 'UNAVAILABLE',
            model_partial_RZ=exact(partial) if partial is not None else None,N_X=57,N_Y=57,
            prediction_sources=[r['reused_prediction'] for r in m6],comparisons=comparisons,
            all_other_rounds_used=False,is_mother_mean_lower_bound=False),
        ideal_mean_compiled_price=None,CI=None,G=None,winner=None)


def load_inputs(root):
    data=G.evidence(root,ADMISSION);frozen=G.evidence(root,A.FROZEN)
    bj=G.evidence(root,A.B_JOIN)['candidates'][0];bi=G.evidence(root,A.B_INPUT)['cases'][0]
    pf=G.evidence(root,PF);benchmarks=[]
    equal(data['sources']['frozen'],A.FROZEN,'Frozen coefficients SOURCE')
    equal(data['sources']['B_join'],A.B_JOIN,'Fixed B join SOURCE')
    equal(data['status'],'DESCRIPTIVE_COMPLETE','Saved C admission')
    equal(bj['model_status'],'MODEL_BOUND_MEETS_ADOPTED_THRESHOLD','Fixed allocation conditional Q3 result')
    equal((data['B_law'],data['B_scalar'],data['B_physical_binding']),
          (bi['law'],bi['scalar'],bi['physical_binding']),'Full-H law/scalar/H/state/snapshot SOURCE')
    equal((bi['finite_K'],bi['kappa']['value']),(2,dict(numerator=2,denominator=1)),'Finite K2/kappa2')
    for label,value in BENCHMARKS.items():
        found=[(i,r) for i,r in enumerate(pf['rows']) if r['label']==label]
        equal(len(found),1,'One PF benchmark')
        i,r=found[0];equal(r['total'],value,'User-specified saved PF total')
        benchmarks.append(dict(label=label,RZ=value,source=A.at(PF,'rows',i),
            qualification='FIXED_Q3_ALLOCATION_AND_ADOPTED_COMPILER_CLOSURE_CONDITIONAL_REFERENCE'))
    equal(len(data['rows']),26,'26 fixed C axes')
    for i,row in enumerate(data['rows']):
        b=G.evidence(root,row['B_join']);p=G.evidence(root,row['B_physical_source'])
        equal(row['frozen_model'],A.at(A.FROZEN,'models',row['axis']),'Axis model SOURCE')
        equal((b['m'],b['axis'],b['physical_id'],b['N']),
              (row['m'],row['axis'],row['physical_id'],row['N']),'B_join m/axis/physical/N')
        equal(row['N'],A.SHOTS[row['m']],'Fixed 1.5x vector')
        equal(row['physical_candidate'],p['physical_candidate'],'B physical schedule')
        equal(row['T_exact'],p['T'],'B exact time')
        model=frozen['models'][row['axis']];domain=model['domain']
        equal((domain['law']['model'],domain['law']['K']),(bi['law']['model'],bi['finite_K']),'Ideal target law')
        equal(domain['implementation']['lowering_id'],'CONTROLLED_ZZ_CX_CRZ_CX_V1','Fixed new ZZ')
        equal(domain['implementation']['scalar_application'],'ONCE_AT_TOTAL_T','Scalar application')
        equal(domain['profile'],frozen['models']['X']['domain']['profile'],'Same fixed compiler/control/full wrapper/P0')
        equal(domain['profile'],dict(compiler=bi['physical_binding']['compiler'],qiskit='1.3.0',
              basis=['rz','sx','x','cx'],optimization_level=1,seed_transpiler=17,control='symmetric_directional',
              wrapper='full_Hadamard_with_measurement',P=0),'Compiler/wrapper conditions')
        equal((model['status'],model['qualified_for_candidate']),('CALIBRATED_UNVALIDATED',False),'Preserve D7 status')
        row['admission_source']=A.at(ADMISSION,'rows',i)
    return data,frozen,benchmarks


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    if not args.execute:parser.error('Choose --execute only for the permitted fixed F1')
    out=ROOT/BASE/'run01';require(not out.exists(),'Reuse fixed F1 results; do not rerun')
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    equal((ROOT/MODULE).read_bytes(),G.C.blob(ROOT,source,MODULE),'Fixed execution SOURCE')
    out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    with (out/'PROGRESS.jsonl').open('x') as log:
        def mark(stage,**details):
            msg=dict(stage=stage,utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-start,**details)
            log.write(json.dumps(msg)+'\n');log.flush();print(json.dumps(msg),flush=True)
        try:
            mark('BEGIN_METADATA_SOURCE_GATES')
            data,frozen,benchmarks=load_inputs(ROOT);D,_=D7.metadata_api(ROOT);M=D.c_model(ROOT)
            mark('BEGIN_22_EXACT_EVALUATIONS_4_REUSES')
            rows=[]
            for row in data['rows']:
                value=evaluate_axis(row,frozen['models'][row['axis']],frozen['predictions'],D.features,M.predict)
                rows.append(value)
                mark('AXIS_COMPLETE',m=value['m'],axis=value['axis'],status=value['status'],origin=value['origin'])
            mark('BEGIN_TOTAL_GATE_AND_M6_DIAGNOSTIC')
            result=summarize(rows,benchmarks)
            result.update(schema='FULL_H_F1_FIXED_SHOT_MODEL_DIAGNOSTIC_V1',execution_source=source,
                sources=dict(admission=ADMISSION,plan=PLAN,frozen=A.FROZEN,B_join=A.B_JOIN,PF=PF),rows=rows,
                formula='a + b*qR + c*(q-1)',target_law=deepcopy(data['B_law']),
                calibration_law='PCG64_numpy_float64_CDF_saved_cohort',model_domains={a:frozen['models'][a]['domain'] for a in ('X','Y')},
                model_original_status={a:frozen['models'][a]['status'] for a in ('X','Y')},
                new_exact_predictions=sum(r['origin']=='NEW_EXACT_FIXED_COEFFICIENT_EVALUATION' for r in rows),
                reused_frozen_predictions=sum(r['origin']=='REUSED_PREOBSERVATION_FROZEN_PREDICTION' for r in rows),
                numerical_encoding='exact_rational',model_qualified_for_candidate=False,
                old_holdout_tolerance=None,old_holdout_verdict=A.VERDICT,
                new_paths=0,builds=0,compiles=0,fits=0,Q3_calls=0,law_distance=None,
                finite_cohort_uncertainty=None,shared_covariance=None,mandatory_stop=True,wall_seconds=time.perf_counter()-start)
            (out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
            with (out/'AXES.csv').open('w') as handle:
                fields=['m','axis','physical_id','q_grouping','R_outer','N','C_ref_numerator','C_ref_denominator','status','origin']
                writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
                for r in rows:writer.writerow({**{k:r[k] for k in fields if not k.startswith('C_ref_')},
                    'C_ref_numerator':r['C_ref']['numerator'],'C_ref_denominator':r['C_ref']['denominator']})
            mark('COMPLETE_MANDATORY_STOP',valid_axes=result['valid_axes'],invalid_axes=len(result['invalid_axes']),
                 new_exact_predictions=result['new_exact_predictions'],reused_frozen_predictions=result['reused_frozen_predictions'])
        except Exception as error:
            (out/'STOP.json').write_text(json.dumps(dict(status='STOP_NO_AUTOMATIC_RETRY',source=source,error=repr(error),mandatory_stop=True),indent=2)+'\n')
            mark('STOP',error=repr(error));raise


if __name__=='__main__':
    main()
