"""Fixed small r16 price pilot. No molecular work/sampling on import.

Canonical law, C3 assembly and C6a descriptive helpers are reused. Historical
C6a/C6b cohort-size guards remain untouched; this adapter binds the new plan.
"""
from dataclasses import asdict
import hashlib
import json

from . import track_d_r16_validation as fixed

BASE = 'e6c6edac084943843b66c87c7899afee327c51c1'
FAMILY = 'artifacts/track_d_r16_price_calibration/2026-11-10/run01'
MASTER = 2026111017
DOMAIN = 'track_d_r16_q2_q4_q8_price_pilot_v1'
QS = (2, 4, 8)
SAMPLES = 4
METHODS = ('symmetric_directional', 'reference')
AXES = ('X', 'Y')
METRICS = ('rz_count','cx_count','rz_depth','cx_depth','total_depth','circuit_size','qubit_count')
OLD = 'artifacts/track_d_r16_limited_validation/2026-11-10/run01'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',',':')).encode()).hexdigest()


def sampling_plan():
    from .c1_stage3_connection import trajectory_seed
    rows=[]
    for q in QS:
        for sample in range(SAMPLES):
            reused=q==4 and sample==0
            seeds=fixed.seed_plan()['seeds'] if reused else [trajectory_seed((MASTER,DOMAIN,BASE),'proposed',fixed.ENDPOINT,
                q.bit_length()-1,'R16_PAIRED_PRICE_CALIBRATION_ONLY',sample,i) for i in range(q)]
            rows.append(dict(q=q,sample=sample,sample_id=f'r16_cal_q{q}_sample{sample:02d}' if not reused
                else fixed.SAMPLE_ID,reused=reused,seeds=seeds,cut_after_complete_step=q//2))
    return dict(base=BASE,master=MASTER,domain=DOMAIN,rows=rows,
        independent_path_units_per_q=SAMPLES,new_paths=11,new_occurrence_calls=52,
        new_event_count=832,new_compile_limit=44,saved_wrappers_reused=4,
        sampler='canonical sample_occurrence / numpy PCG64',
        actual_IID_guaranteed=False,XY_independent_samples=False,
        interval_or_precision_target=None,expectation_qualification='exploratory finite sample only')


def load_arm(root):
    """Reconstruct only the unchanged r16 law; never evaluate PF/RTE errors/N."""
    from . import c_pr_aware_df_stage_b as b
    records={}
    packet=fixed.fixed(root,fixed.INPUT,records,fixed.PACKET,arrays=True)
    endpoint=fixed.fixed(root,fixed.CDF,records,fixed.ENDPOINT,arrays=True)
    arm=b.prepare_arm(**{k:packet[k] for k in ('h','g','constant')},
        **{k:endpoint[k] for k in ('J','U')},role='proposed',source_identity=fixed.ENDPOINT,
        input_identity=fixed.PACKET,ld=2)
    tail,config,dist=b.configure(arm,fixed.H,16,0)
    saved=json.loads((root/OLD/'r16_law.json').read_text())
    if (asdict(config)!=saved['law']['config'] or tail.lambda_r!=saved['law']['lambda_R']
            or arm.plan.extracted_identity!=saved['law']['extracted_identity']):
        raise ValueError('SAVED_R16_LAW_IDENTITY')
    return arm,tail,config,dist,packet,records,saved['sha256']


def sample_path(row,arm,tail,config,dist,source,law_hash,*,before_occurrence):
    from .c1_stage3_connection import sample_occurrence
    if row not in sampling_plan()['rows'] or row['reused']:
        raise ValueError('ONLY_FIXED_NEW_PATH')
    steps=[]
    for i,seed in enumerate(row['seeds']):
        before_occurrence(row['sample_id'],i,seed)
        events=sample_occurrence(tail,dist,16,seed)
        if len(events)!=16 or any(e.taylor_order or e.product_component_ids or e.phase!=1 for e in events):
            raise ValueError('R16_K0_ONLY')
        steps.append([e.to_dict() for e in events])
    return dict(**row,source=source,law_sha256=law_hash,packet=fixed.PACKET,
                endpoint=fixed.ENDPOINT,ordered_events=steps)


def build_path(context,record,car):
    from . import track_d_c3_whole_path as c3
    from . import c_pr_aware_df_stage_b as b
    from . import c_pr_aware_df_stage_b2 as b2
    from .track_d_h4_reference import stored_events
    row=next((r for r in sampling_plan()['rows'] if r['sample_id']==record['sample_id']),None)
    if row is None or row['reused'] or any(record[k]!=v for k,v in row.items()):
        raise ValueError('FIXED_PATH_BINDING')
    if record['packet']!=fixed.PACKET or record['endpoint']!=fixed.ENDPOINT:
        raise ValueError('PATH_INPUT')
    if context.config.rte_steps!=16 or len(record['ordered_events'])!=row['q']:
        raise ValueError('R16_PATH_STEPS')
    context.builder.fusion_records.clear()
    steps=[]
    for i,values in enumerate(record['ordered_events']):
        events=stored_events(values)
        if len(events)!=16 or any(e.taylor_order or e.product_component_ids or e.phase!=1 for e in events):
            raise ValueError('R16_K0_ONLY')
        steps.append(dict(sample=f"{row['sample_id']}_step{i:02d}",events=events,
            reference=b2.trajectory_reference(context.arm,car,fixed.H,events),
            raw=b.build_kernel(context.arm,fixed.H,events,context.config,builder=context.builder)))
    bundle=c3.prepare_steps(context,steps,row['cut_after_complete_step'],
        ordered_samples=tuple(r['sample'] for r in steps),cohort=DOMAIN,
        fixture_selection='pre-fixed q2/q4/q8 four paths each; no outcome-dependent replacement')
    bundle.identity.update(r=16,sample_id=row['sample_id'],law_sha256=record['law_sha256'],
        execution_source=record['source'],events_sha256=digest(record['ordered_events']),C1_extra_fusion=False)
    return bundle


def paired_statistics(records):
    """C6a/C6b describe/covariance arithmetic, bound to this 12-path plan."""
    import numpy as np
    from .track_d_c6a_pilot import describe
    expected={(r['q'],r['sample'],m,a) for r in sampling_plan()['rows'] for m in METHODS for a in AXES}
    keys=[(r['q'],r['sample'],r['method'],r['axis']) for r in records]
    if len(set(keys))!=len(keys) or set(keys)!=expected:
        raise ValueError('COMPLETE_FIXED_CAMPAIGN_REQUIRED_NO_COMPLETE_CASE')
    table=dict(zip(keys,records));out={}
    labels=[(m,a,g) for m in METHODS for a in AXES for g in METRICS]
    plabels=[(a,g) for a in AXES for g in METRICS]
    def correlations(matrix):
        # Constant resource columns (e.g. width, sometimes paired CX) have
        # undefined correlation, not evidence of independence or zero error.
        size=matrix.shape[1];result=[[None]*size for _ in range(size)]
        active=np.flatnonzero(np.var(matrix,axis=0,ddof=1)>0)
        if len(active)==1:result[active[0]][active[0]]=1.
        elif len(active)>1:
            corr=np.corrcoef(matrix[:,active],rowvar=False)
            for i,a in enumerate(active):
                for j,b in enumerate(active):result[a][b]=float(corr[i,j])
        return result
    for q in QS:
        def value(i,m,a,g):return table[q,i,m,a]['metrics'][g]
        matrix=np.array([[value(i,*label) for label in labels] for i in range(SAMPLES)],float)
        paired=np.array([[value(i,'reference',a,g)-value(i,'symmetric_directional',a,g)
                          for a,g in plabels] for i in range(SAMPLES)],float)
        rows=[]
        for a,g in plabels:
            s=[value(i,'symmetric_directional',a,g) for i in range(SAMPLES)]
            r=[value(i,'reference',a,g) for i in range(SAMPLES)]
            delta=[y-x for x,y in zip(s,r)]
            rows.append(dict(axis=a,metric=g,strong=describe(s),reference=describe(r),
                paired_delta=describe(delta),method_sample_covariance=float(np.cov(s,r,ddof=1)[0,1]),
                paired_deltas_by_sample=delta))
        out[str(q)]=dict(n_path_units=SAMPLES,rows=rows,
            sample_covariance_labels=labels,sample_covariance=np.cov(matrix,rowvar=False,ddof=1).tolist(),
            sample_correlation=correlations(matrix),
            paired_covariance_labels=plabels,paired_sample_covariance=np.cov(paired,rowvar=False,ddof=1).tolist(),
            paired_sample_correlation=correlations(paired),
            covariance_rank_limit=SAMPLES-1,covariance_ddof=1,
            old_path_included=q==4,XY_are_not_independent_samples=True)
    return dict(status='COMPLETE_EXPLORATORY_FINITE_PILOT',by_q=out,
        confidence_interval=None,population_confidence_guaranteed=False,ideal_joint_IID_guaranteed=False,
        shared_source_and_input=True,cross_q_joint_covariance_qualified=False,
        support='observed finite extrema only; no all-law support',
        q4_reuse='one historical fixed path mandated for reuse, not prospective holdout',
        long_q_model_registered=False,total_RPE_price_evaluated=False)
