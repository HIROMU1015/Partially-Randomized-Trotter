"""Saved-metadata residual analysis. No science imports; NumPy only at fit time."""
from collections import Counter
from fractions import Fraction
import hashlib
import json
import math

SCOPE='CONDITIONAL_SAVED_UNIT_RZ_RESIDUALS_UNIFORM_A_B_ONLY'
THEORY_SOURCE='7e41670565d9284da9aded94063be591fdcaa3dd'
COMPILER='1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b'
CAPS=dict(theory_rows=832,connection_rows=10480,index_entries=262,unit_model_rows=1048,strata=64,fits=128)
LIMITS=dict(processes=2,max_workers=0,threads=1,AS_bytes=4*2**30,RSS_bytes=2**30,
    observer_AS_bytes=512*2**20,observer_RSS_bytes=512*2**20,aggregate_RSS_bytes=1536*2**20,
    CPU_seconds=600,aggregate_CPU_seconds=600,wall_seconds=1800,output_bytes=64*2**20,stage_storage_bytes=128*2**20)
CRITICAL=('input','hamiltonian','state','df','fragment_order','compiler_profile','basis_variant',
          'control','scalar_policy','native_policy','lineage_compatibility','sample_scope')
PROOF=('origin_sources','wrapper_identity','dataset_seed_source','preparation_binding','law_binding')


def canonical(value):return json.dumps(value,ensure_ascii=True,sort_keys=True,separators=(',',':'),allow_nan=False).encode('ascii')
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()


def policy():
    return dict(schema='track_a_v2_residual_policy_v1',scope=SCOPE,theory_source=THEORY_SOURCE,
        caps=CAPS,limits=LIMITS,metrics=['RZ'],threshold='1/20',boundary_tolerance='1/1000000000000',
        holdout='sha256(canonical [system,prefix,q,formula]) big-endian modulo5 ==0',
        critical=list(CRITICAL),proof=list(PROOF),interval='lower_then_upper_no_midpoint',
        rcond=1e-12,maximum_condition_number=1e10,one_shot=True,retry=False,resume=False,
        new_compile=0,new_signal=0,new_sampling=0,new_law=0,new_DP=0,new_primitive=0,
        new_shots=0,new_total_cost=0,new_ranking=0,mixed_theory=0,next_stage_authorized=False)


def rational(value):
    if isinstance(value,dict):value=Fraction(int(value['numerator']),int(value['denominator']))
    elif isinstance(value,bool) or value is None:raise ValueError('INVALID_COST')
    else:value=Fraction(value)
    if value<=0:raise ValueError('NONPOSITIVE_COST')
    return value


def holdout(cell,system):
    key=[system,cell['prefix'],cell['q'],cell['formula']]
    return int.from_bytes(hashlib.sha256(canonical(key)).digest(),'big')%5==0


def classify(entry,expected):
    """Only externally pinned evidence can promote provenance to resolved."""
    if expected is None:return ['NO_INDEPENDENT_EXPECTED_REGISTRY']
    identity=(entry['system'],entry['scope'],entry['cell_id'])
    if identity!=tuple(expected['key']):raise ValueError('REGISTRY_IDENTITY_MISMATCH')
    for name,value in expected['observed'].items():
        if entry.get(name)!=value:raise ValueError('PROVENANCE_MISMATCH:'+name)
    reasons=[]
    for name in (*CRITICAL,*PROOF):
        value=expected['values'].get(name)
        if value is None or value==[] or value=='':reasons.append('MISSING:'+name)
    if not expected.get('evidence_refs'):reasons.append('NO_INDEPENDENT_EVIDENCE_REFERENCES')
    if expected['values'].get('compiler_hash')!=COMPILER:reasons.append('COMPILER_UNRESOLVED')
    # Full numeric envelope is deliberately not claimed even for resolved metadata.
    if expected.get('metadata_resolved') is not True:reasons.append('UNRESOLVED_PROVENANCE')
    return sorted(set(reasons))


def metadata_audit(index,registry):
    expected={tuple(r['key']):r for r in registry}
    if len(expected)!=len(registry):raise ValueError('DUPLICATE_REGISTRY_KEY')
    rows=[];keys=set()
    for system,entries in index.items():
        if system not in ('H6','H8'):raise ValueError('SYSTEM_SCOPE')
        for original in entries:
            entry=dict(original,system=system);key=(system,entry['scope'],entry['cell_id'])
            if key in keys:raise ValueError('DUPLICATE_INDEX_KEY')
            keys.add(key);reasons=classify(entry,expected.get(key))
            rows.append(dict(key=list(key),status='UNRESOLVED_PROVENANCE' if reasons else 'METADATA_RESOLVED_CONDITIONAL_ONLY',reasons=reasons))
    if set(expected)!=keys:raise ValueError('REGISTRY_INDEX_COVERAGE')
    if len(rows)>CAPS['index_entries']:raise ValueError('INDEX_CAP')
    return dict(rows=rows,status_counts=dict(Counter(x['status'] for x in rows)),
                numerical_analysis_performed=False,full_numeric_envelope_certified=False)


def prepare_observations(theory,connections,index,registry):
    """Called only inside an approved run; no fit or expectation evaluation here."""
    if len(theory)>CAPS['theory_rows'] or len(connections)>CAPS['connection_rows']:raise ValueError('INPUT_ROW_CAP')
    audit=metadata_audit(index,registry)
    records={(system,x['scope'],x['cell_id']):dict(x,system=system)
             for system,entries in index.items() for x in entries}
    reg={tuple(x['key']):x for x in registry}
    known={(x['system'],x['candidate_id'],x['model'],x['axis']):x for x in theory}
    if len(known)!=len(theory):raise ValueError('DUPLICATE_THEORY_KEY')
    for row in theory:
        if row['system'] not in ('H6','H8') or row['model'] not in ('A','B') or row['axis'] not in ('cosine','sine'):
            raise ValueError('THEORY_SCOPE')
        if row['theory_profile']!='explicit_rz_sx_x_cx_v1' or row['control']!='symmetric_directional':raise ValueError('THEORY_PROFILE')
    unresolved={tuple(x['key']) for x in audit['rows'] if x['reasons']}
    repeats={};skipped=[]
    for row in connections:
        if row['metric']!='RZ':continue
        key=(row['system'],row['scope'],row['candidate_id'])
        if key not in records:raise ValueError('CONNECTION_WITHOUT_INDEX')
        entry=records[key];expected=reg[key]
        if row['model'] not in ('A','B'):raise ValueError('MODEL_SCOPE')
        means=row['saved_axis_means']
        if means is None:
            skipped.append(dict(key=list(key),reason='MISSING_COMPILED_UNIT_COST'));continue
        if set(means)!=set(('cosine','sine')):raise ValueError('AXIS_PAIR_REQUIRED')
        origins=entry.get('execution_sources',[])
        origin=row.get('compiled_origin_SOURCE')
        if origin and origin not in origins:raise ValueError('COMPILED_ORIGIN_MISMATCH')
        if type(row.get('sample_count')) is not int or row['sample_count']!=entry.get('paired_groups'):
            raise ValueError('SAMPLE_COUNT_MISMATCH')
        if row.get('sample_phase') is not None and row['sample_phase']!=entry.get('wrapper_phase'):
            raise ValueError('SAMPLE_PHASE_MISMATCH')
        for axis in ('cosine','sine'):
            try:cost=rational(means[axis])
            except (ValueError,TypeError,ZeroDivisionError,OverflowError):
                skipped.append(dict(key=list(key),axis=axis,reason='INVALID_OR_ZERO_COMPILED_COST'));continue
            duplicate=key+(axis,)
            observation=(str(cost),row.get('sample_count'),row.get('sample_phase'),origin)
            if duplicate in repeats and repeats[duplicate]!=observation:raise ValueError('CONFLICTING_REPEATED_UNIT_COST')
            repeats[duplicate]=observation
    result=[];reuse={}
    for key_axis,observation in sorted(repeats.items()):
        key=key_axis[:3];axis=key_axis[3]
        if key in unresolved:continue
        entry=records[key];p=reg[key]['values'];cell=entry['cell']
        family='PR' if entry['cell_id'].startswith('B2_') else 'PF'
        # PF cross-scope reuse requires an independent reuse-cohort proof.
        cohort=p.get('reuse_cohort') or entry['scope']
        if family=='PF' and not p.get('reuse_cohort'):raise ValueError('PF_REUSE_COHORT_UNRESOLVED')
        obs_key=(key[0],cohort,key[2],axis,digest(p['origin_sources']),p['wrapper_identity'])
        sample_scope=p['sample_scope']
        if family=='PF':sample_scope=p.get('calibration_scope') or sample_scope
        stratum_values={k:p[k] for k in CRITICAL};stratum_values['sample_scope']=sample_scope
        signature=(observation[0],digest(stratum_values))
        if obs_key in reuse:
            if reuse[obs_key]!=signature:raise ValueError('CONFLICTING_REUSED_PF_COST_OR_STRATUM')
            continue
        reuse[obs_key]=signature
        for model in ('A','B'):
            tkey=(key[0],key[2],model,axis)
            if tkey not in known:raise ValueError('MISSING_REGISTERED_THEORY_AXIS')
            t=known[tkey]
            if any(t['cell'].get(f)!=cell.get(f) for f in ('prefix','q','formula')):raise ValueError('THEORY_CELL_MISMATCH')
            lo,hi=rational(t['lower']['rz']),rational(t['upper']['rz'])
            if lo>hi:raise ValueError('THEORY_INTERVAL_ORDER')
            result.append(dict(id=digest(list(obs_key)),system=key[0],model=model,axis=axis,
                stratum=digest(stratum_values),family=family,holdout=holdout(cell,key[0]),
                lower=str(lo),upper=str(hi),compiled=observation[0],candidate=key[2]))
    if len(result)>CAPS['unit_model_rows']:raise ValueError('OBSERVATION_CAP')
    return result,dict(provenance=audit,skipped=skipped,observations=len(result),full_numeric_envelope_certified=False)


def threshold_status(difference):
    value=abs(Fraction(difference));threshold=Fraction(1,20);tolerance=Fraction(1,10**12)
    if value>=threshold:return 'STOP_RESIDUAL_GAP_GE_005'
    if threshold-value<=tolerance:return 'BOUNDARY_UNRESOLVED'
    return 'PASS_POINT_DIAGNOSTIC'


def fit_endpoint(rows,endpoint,*,np=None):
    train=[r for r in rows if not r['holdout']];test=[r for r in rows if r['holdout']]
    for group in (train,test):
        if any(sum(r['family']==f for r in group)<2 for f in ('PR','PF')):
            return dict(status='INCONCLUSIVE_INSUFFICIENT_PR_PF')
    if len({r[endpoint] for r in train})<2:return dict(status='INCONCLUSIVE_DISTINCT_X')
    if np is None:import numpy as np
    x=np.asarray([float(Fraction(r[endpoint])) for r in train]);y=np.asarray([float(Fraction(r['compiled'])) for r in train])
    if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('NONFINITE_FIT_INPUT')
    mean=float(x.mean());scale=float(np.sqrt(np.mean((x-mean)**2)))
    if not math.isfinite(scale) or scale<=0:raise ValueError('FIT_SCALE')
    design=np.column_stack((np.ones(len(x)),(x-mean)/scale))
    condition=float(np.linalg.cond(design))
    if not math.isfinite(condition) or condition>1e10:raise ValueError('DESIGN_CONDITION')
    beta,_,rank,_=np.linalg.lstsq(design,y,rcond=1e-12)
    if rank!=2:raise ValueError('FIT_RANK')
    a=float(beta[0]-beta[1]*mean/scale);b=float(beta[1]/scale)
    if not math.isfinite(a+b) or b<0:raise ValueError('INVALID_FIT_COEFFICIENT')
    residuals=[]
    for row in rows:
        c=float(Fraction(row['compiled']));prediction=a+b*float(Fraction(row[endpoint]))
        if not math.isfinite(prediction) or prediction<=0:raise ValueError('NONPOSITIVE_PREDICTION')
        if row['holdout']:
            residual=(c-prediction)/c
            if not math.isfinite(residual):raise ValueError('NONFINITE_RESIDUAL')
            residuals.append(dict(id=row['id'],family=row['family'],residual=residual))
    means={f:math.fsum(r['residual'] for r in residuals if r['family']==f)/sum(r['family']==f for r in residuals) for f in ('PR','PF')}
    difference=abs(means['PR']-means['PF'])
    return dict(status=threshold_status(difference),intercept=a,slope=b,condition=condition,
                residuals=residuals,mean_residuals=means,difference=difference,endpoint=endpoint)


def evaluate(rows,*,reserve,emit,fit=fit_endpoint):
    strata={}
    for row in rows:
        key=(row['system'],row['model'],row['axis'],row['stratum']);strata.setdefault(key,[]).append(row)
    if len(strata)>CAPS['strata']:raise ValueError('STRATUM_CAP')
    calls=0;results=[]
    for key,group in sorted(strata.items(),key=lambda p:(p[0][0],p[0][1],0 if p[0][2]=='cosine' else 1,p[0][3])):
        for endpoint in ('lower','upper'):
            if calls>=CAPS['fits']:raise ValueError('FIT_CAP')
            reserve(key,endpoint);calls+=1
            try:result=fit(group,endpoint)
            except BaseException as exc:
                emit(dict(stratum=list(key),endpoint=endpoint,status='STOP_FIT_ERROR',error=str(exc)))
                raise
            item=dict(result,stratum=list(key),endpoint=endpoint);emit(item);results.append(item)
            if result['status'].startswith('STOP') or result['status']=='BOUNDARY_UNRESOLVED':
                return dict(status='STOP',first_stop=result['status'],fits=calls,mandatory_stop=True)
    return dict(status='COMPLETE',fits=calls,all_strata_passed=bool(results) and all(x['status']=='PASS_POINT_DIAGNOSTIC' for x in results),
        scientific_status='NO_ASSESSABLE_STRATA' if not results else 'CONDITIONAL_POINT_DIAGNOSTICS',
        mandatory_stop=True,full_numeric_envelope_certified=False)
