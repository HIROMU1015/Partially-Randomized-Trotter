"""Fixed p7/q1 S2 two-axis validation contract; stdlib, no scientific I/O."""
from fractions import Fraction
import hashlib
import json

BASE='25bbdd33fe758a0e730138cb91ddac8c41b0a298'
PROVIDER='87b6b9f46b7e8ba9abe132ee21478c114b6f46f7'
MODEL_SHA='406c6f36f93dac317f9709e68b042c4563edf5ec4b06219e1b60de660b695211'
COMPILER='1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b'
CELL_ID='B0I_p7_q1_2nd'
AXES=('cosine','sine')
PREDICTION='72621/5'
OPTIONS=dict(basis_gates=['rz','sx','x','cx'],optimization_level=1,seed_transpiler=17)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def cell():
    return dict(id=CELL_ID,method='B0I',prefix=7,q=1,formula='2nd',r=None,K=None,R=None)


def plan():
    return dict(schema='track_a_v1_p7_compile_plan_v1',base_commit=BASE,scope='p7_compile_validation',
        target=dict(system='linear_H6',geometry_angstrom=1.,basis='STO-3G',df_rank=19,
                    modes=12,sector_dimension=400,T=.8,delta=.8),
        cell=cell(),axes=list(AXES),axis_order='cosine then sine, one compile in flight',
        predicted_RZ_per_axis=PREDICTION,frozen_model_sha256=MODEL_SHA,
        compiler_hash=COMPILER,compiler_options=OPTIONS,control='symmetric_directional',
        scalar='saved full preparation constant + saved Phase A full-Fock tail C; one final ancilla phase',
        full_H_reference='unchanged Phase A full-H reference, never rotated or recomputed',
        allowance='saved empirical policy unchanged; not a numerical certificate',
        discrepancy=dict(relative_error='abs(frozen prediction - measured RZ)/measured RZ, per axis',
            stop_when='>= 1/50 exactly; no axis averaging',strict_before_next_axis=True,
            missing_or_changed_identity='STOP',nonpositive_or_noninteger_measurement='STOP'),
        call_caps=dict(snapshot_load=1,frozen_preparation_decode=1,native_evolution_max=1,
            measured_wrappers_max=2,control_probes_max=4,native_compile_max=2,
            new_signal=0,new_reference=0,new_scalar=0,new_primitive=0,new_SCF=0,new_DF=0,new_state=0,
            new_sampling=0,new_fit=0,cost_reaccounting=0,additional_candidate_search=0,V2=0),
        native_instruction_cap=1000000,transpiled_instruction_cap=5000000,
        correctness_guard=dict(absolute_gate=1e-9,relative_gate=1e-10,certified=False),
        resources=dict(processes=2,max_workers=0,threads=1,AS_bytes=32*2**30,RSS_bytes=9*2**30,
            observer_AS_bytes=512*2**20,observer_RSS_bytes=512*2**20,aggregate_RSS_bytes=9728*2**20,
            CPU_seconds=3600,aggregate_CPU_seconds=3600,wall_seconds=7200,
            output_bytes=128*2**20,stage_storage_bytes=512*2**20),
        approved=False,runtime_authorization=False,allowed_cpus=[],absolute_launch_command=None,
        one_shot=True,retry=False,resume=False,refit=False,mandatory_stop=True,next_stage_authorized=False)


def predictions(frozen):
    if frozen.get('fitted_after_validation') is not False:raise ValueError('REFITTED_MODEL_FORBIDDEN')
    rows=[]
    for axis in AXES:
        matches=[x for x in frozen['models'] if x['stratum']['method']=='B0I' and x['stratum']['formula']=='2nd'
                 and x['stratum']['axis']==axis]
        if len(matches)!=1:raise ValueError('UNIQUE_FROZEN_STRATUM_REQUIRED')
        model=matches[0];s=model['stratum']
        if (s['control']!='symmetric_directional' or s['compiler_hash']!=COMPILER
                or s['phase_policy']!='single_scalar_on_ancilla'):raise ValueError('MODEL_STRATUM_CHANGED')
        a,b,d=map(Fraction,model['coefficients']);value=a+b+7*d
        if value!=Fraction(PREDICTION):raise ValueError('FROZEN_P7_PREDICTION_CHANGED')
        rows.append(dict(cell_id=CELL_ID,axis=axis,exact_prediction=str(value),predicted=float(value),
                         parent_stratum=s,kind='FROZEN_MODEL_ESTIMATE_NOT_MEASURED'))
    return rows


def validate_axis(row):
    if (row.get('cell_id')!=CELL_ID or row.get('axis') not in AXES or row.get('model_sha256')!=MODEL_SHA
            or row.get('status')!='MEASURED_COMPLETE' or row.get('identity_verified') is not True
            or row.get('compiler_hash')!=COMPILER or row.get('exact_prediction')!=PREDICTION):
        raise ValueError('P7_MEASUREMENT_IDENTITY')
    measured=row['measured']
    if type(measured) is not int or measured<=0:raise ValueError('POSITIVE_INTEGER_RZ_REQUIRED')
    error=abs(Fraction(PREDICTION)-measured)/measured
    if error>=Fraction(1,50):raise ValueError('COMPILE_MODEL_DISAGREEMENT_GE_2_PERCENT')
    return str(error)


def validate_pair(rows):
    if len(rows)!=2 or [x.get('axis') for x in rows]!=list(AXES):raise ValueError('EXACT_REGISTERED_AXIS_PAIR_REQUIRED')
    for row in rows:validate_axis(row)
    return dict(status='VALIDATED_ON_REGISTERED_P7_AXES',full_grid_error_bound=False,
        refit=False,cost_reaccounting_authorized=False,next_stage_authorized=False,mandatory_stop=True)
