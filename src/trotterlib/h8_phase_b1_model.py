"""Fixed H8 B1 model and independent-axis checks; standard library only.

The caller must validate immutable receipt bytes before supplying rows. This
module has no file loader, compiler, molecular imports, grant issuer or retry.
"""
from fractions import Fraction as F
from hashlib import sha256
import json

from h8_phase_b1_contract import PROFILE, COMPILER, REQUIRED

SCOPE = 'H8_B1_NEW_B0_8_B0I16_FIT_VALIDATION_V1'
STRATA = (('B0', 'cosine'), ('B0', 'sine'), ('B0I', 'cosine'), ('B0I', 'sine'))
TRAIN = ((14, 1), (21, 1), (14, 8))
HOLDOUT = (21, 8)
B0I_SOURCE = '54ed98c5fa5b7190f77255271f4e3bec8db15630'
PREPARATIONS = {
    14: '993f3b7cd9390daa64e2fc9b6584bc1797a783e27001e427abed097eddf803a6',
    21: 'b3baf8d6fe14d3e0e637e7b0b0152a4f2b2c0ecddc4702e98e2dbfd02651ab47',
}
SCALARS = dict(B0='ORIGINAL_CONSTANT_ONCE_NO_TAIL_IDENTITY',
               B0I='ORIGINAL_PLUS_FULL_FOCK_TAIL_IDENTITY_ONCE')
CAPS = dict(cost_fit=4, B0_training_checks=6, B0_holdout_checks=2,
            B0I_anchor_checks=16, compile=0, signal=0, reaccount=0,
            sampling=0, refit=0, additional_measurement=0)
RESOURCES = dict(driver_AS_bytes=512*2**20, driver_RSS_bytes=256*2**20,
    observer_AS_bytes=512*2**20, observer_RSS_bytes=256*2**20,
    AS_bytes=512*2**20, RSS_bytes=256*2**20, CPU_seconds=240,
    aggregate_CPU_seconds=360, observer_CPU_seconds=120, wall_seconds=600,
    max_workers=0, aggregate_RSS_bytes=512*2**20, admission_reserve_bytes=512*2**20,
    output_bytes=64*2**20, log_bytes=16*2**20, stage_storage_bytes=128*2**20,
    minimum_free_inodes=40000, output_files_max=1000)


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                             allow_nan=False).encode()).hexdigest()


def hex_digest(value, length=64):
    return type(value) is str and len(value) == length and all(c in '0123456789abcdef' for c in value)


def plan():
    return dict(schema='H8_PHASE_B1_MODEL_PLAN_V1', scope=SCOPE,
        S2='a+b*q+d*prefix*q', S4='a+3*b*q+3*d*prefix*q',
        stratum_order=[m+'/'+a for m,a in STRATA],
        B0_TRAIN=[list(x) for x in TRAIN], B0_HOLDOUT=[list(HOLDOUT)],
        B0I_fit='ALL_8_REGISTERED_S2_ANCHORS_PER_AXIS_EXACT_RATIONAL_OLS',
        B0I_validation='ALL_8_TRAINING_RESIDUALS_PER_AXIS_NOT_INDEPENDENT_HOLDOUT',
        validation='EVERY_AXIS_abs(predicted-measured)/measured < 1/50',
        missing_or_mismatch='STOP_BEFORE_ANY_FIT', independent_axes=True,
        all_four_strata_required=True, old_B0_allowed=False, H6_transfer=False,
        holdout_in_fit=False, refit=False, additional_measurement=False,
        one_shot=True, retry=False, B2_authorized=False, C_authorized=False,
        next_stage_authorized=False, mandatory_stop=True,
        resources=dict(RESOURCES), call_caps=dict(CAPS))


def validate_rows(rows, acquisition_source, input_binding):
    require(hex_digest(acquisition_source,40) and acquisition_source not in
        ('cb8a56c6b2a772827275f9200b02638030f33f72',
         'c31ef424175c02379eafebe8fc6c7548c1d9aabe'), 'NEW_B0_SOURCE_REQUIRED')
    expected={(m,p,q,a) for m,a in STRATA for p in (14,21)
              for q in ((1,8) if m=='B0' else (1,2,4,8))}
    require(type(rows) is list and len(rows)==24, 'EXACT_24_AXES_REQUIRED')
    seen=set()
    for row in rows:
        method,p,q,axis=(row[k] for k in ('method','prefix','q','axis'))
        require(type(p) is int and type(q) is int, 'INTEGER_DESIGN_COORDINATES')
        key=method,p,q,axis
        require(key in expected and key not in seen, 'MISSING_DUPLICATE_OR_EXTRA_AXIS')
        seen.add(key)
        require(row['formula']=='2nd' and row['control']=='symmetric_directional', 'FORMULA_CONTROL')
        require(type(row['saved_RZ']) is int and row['saved_RZ']>0, 'POSITIVE_INTEGER_RZ')
        require(row['runtime_profile_sha256']==PROFILE and row['compiler_hash']==COMPILER,
                'PROFILE_COMPILER_MISMATCH')
        require(row['origin_SOURCE']==(acquisition_source if method=='B0' else B0I_SOURCE), 'ANCHOR_SOURCE')
        require(row['cost_preparation_hash']==PREPARATIONS[p], 'PREPARATION_MISMATCH')
        require(row['scalar_policy']==SCALARS[method], 'SCALAR_POLICY')
        require(row['input_binding_sha256']==input_binding, 'INPUT_BINDING')
        require(hex_digest(row['fingerprint']) and all(row['evidence'].get(k) is True for k in REQUIRED),
                'INCOMPLETE_ANCHOR_PROVENANCE')
        role='HOLDOUT' if method=='B0' and (p,q)==HOLDOUT else 'TRAIN'
        require(row['role']==role, 'TRAIN_HOLDOUT_REASSIGNMENT')
    require(seen==expected, 'MISSING_AXIS')


def solve(training):
    """Exact fixed three-column OLS; singular designs STOP without fallback."""
    require(len(training)>=3, 'THREE_TRAINING_ANCHORS_REQUIRED')
    X=[[F(1),F(r['q']),F(r['prefix']*r['q'])] for r in training]
    matrix=[[sum(x[i]*x[j] for x in X) for j in range(3)]+
            [sum(x[i]*r['saved_RZ'] for x,r in zip(X,training))] for i in range(3)]
    for i in range(3):
        pivot=next((j for j in range(i,3) if matrix[j][i]),None)
        require(pivot is not None,'SINGULAR_DESIGN_STOP_NO_FALLBACK')
        matrix[i],matrix[pivot]=matrix[pivot],matrix[i]
        value=matrix[i][i];matrix[i]=[x/value for x in matrix[i]]
        for j in range(3):
            if i!=j:
                value=matrix[j][i];matrix[j]=[x-value*y for x,y in zip(matrix[j],matrix[i])]
    return tuple(matrix[i][3] for i in range(3))


def predict(coefficients,prefix,q):
    a,b,d=coefficients;result=a+b*q+d*prefix*q
    require(result>0,'NONPOSITIVE_PREDICTION')
    return result


def diagnostic(predicted,measured):
    require(type(measured) is int and measured>0,'MEASURED_DENOMINATOR')
    require(F(predicted)>0,'NONPOSITIVE_PREDICTION')
    error=abs(F(predicted)-measured)/measured
    return dict(predicted_RZ=str(predicted),measured_RZ=measured,
                relative_error=str(error),PASS=error<F(1,50))


def evaluate(rows, *, acquisition_source, input_binding, take, emit, guard=lambda:None):
    validate_rows(rows,acquisition_source,input_binding)
    completed=[]
    for method,axis in STRATA:
        guard();selected=sorted((r for r in rows if (r['method'],r['axis'])==(method,axis)),
                              key=lambda r:(r['prefix'],r['q']))
        training=[r for r in selected if r['role']=='TRAIN']
        require(len(training)==(3 if method=='B0' else 8),'FIXED_TRAINING_COUNT')
        take('cost_fit');coefficients=solve(training)
        current=dict(method=method,axis=axis,coefficients=[str(x) for x in coefficients],
                     training=[dict(prefix=r['prefix'],q=r['q']) for r in training],
                     checks=[],status='PENDING_VALIDATION')
        emit('partial',dict(current))
        for row in training+[r for r in selected if r['role']=='HOLDOUT']:
            guard();take('B0I_anchor_checks' if method=='B0I' else
                         'B0_holdout_checks' if row['role']=='HOLDOUT' else 'B0_training_checks')
            check=dict(prefix=row['prefix'],q=row['q'],role=row['role'],
                       **diagnostic(predict(coefficients,row['prefix'],row['q']),row['saved_RZ']))
            current['checks'].append(check);emit('partial',dict(current))
            require(check['PASS'],'AXIS_ERROR_GE_2_PERCENT:'+method+'/'+axis)
        current['status']='PASS';completed.append(current);emit('stratum_PASS',dict(current))
    frozen=dict(schema='H8_B1_FROZEN_MODEL_V1',models=completed,
        input_binding_sha256=input_binding,acquisition_SOURCE=acquisition_source,
        profile_sha256=PROFILE,compiler_hash=COMPILER,model=plan()['S2'],
        S4_unvalidated_hypothesis=plan()['S4'],S4_validation=False,
        full_grid_guaranteed=False,B0I_independent_holdout=False,
        B2_authorized=False,C_authorized=False,next_stage_authorized=False)
    return dict(status='COMPLETE_MANDATORY_STOP',frozen_model=frozen,
                model_sha256=digest(frozen),all_four_strata_PASS=True)
