"""Exact scalar accumulation and a fixed principal determinant-phase ledger.

No statevector, gate matrix, transpile, phase correction or acceptance override.
"""
from fractions import Fraction
import math
from pathlib import Path
import time


def require(ok,reason):
    if not ok:raise ValueError(reason)


def exact_float(value):
    require(type(value) is float and math.isfinite(value),'FINITE_BINARY64_FLOAT_REQUIRED')
    return Fraction.from_float(value)


def rational_payload(value):
    return dict(numerator=str(value.numerator),denominator=str(value.denominator))


def restore_rational(payload):
    n,d=int(payload['numerator']),int(payload['denominator'])
    require(d>0,'POSITIVE_RATIONAL_DENOMINATOR')
    f=Fraction(n,d)
    require(rational_payload(f)==payload,'CANONICAL_EXACT_RATIONAL')
    return f


def extract(qc,kind,nqubits=13):
    import inspect_h6_m5_phase_structure as structure
    metadata=structure.summarize(qc,kind,nqubits)
    require(nqubits==13,'FIXED_THIRTEEN_QUBITS')
    angle_sum=Fraction(0);count=0
    for item in qc.data:
        if item.operation.name=='p':
            require(kind=='WRAPPER','NO_COMPILED_P_IN_FIXED_BASIS')
            angle_sum+=exact_float(item.operation.params[0]);count+=1
    return dict(kind=kind,metadata=metadata,P_count=count,P_angle_sum_exact=rational_payload(angle_sum),
        global_phase_exact=rational_payload(exact_float(float(qc.global_phase))),
        gate_matrices_evaluated=0,statevector_replays=0,new_compile=0,acceptance_override=False)


def extract_worker(cfg,kind):
    import h6_m5_phase_ledger_contract as c
    from qiskit import qpy
    started=time.monotonic();row=cfg['inputs'][kind.lower()+'_qpy']
    c.consumed(cfg)
    c.require(c.read(Path(cfg['ledger'])/('EXTRACT_'+kind+'.json'))['stage']=='EXTRACT_'+kind,'RESERVED_EXTRACTION')
    c.require(c.sha(row['path'])==row['sha256'],'INPUT_QPY_SHA')
    with Path(row['path']).open('rb') as f:
        circuits=qpy.load(f);c.require(f.read()==b'','NO_QPY_TRAILING_BYTES')
    c.require(len(circuits)==1,'EXACT_ONE_CIRCUIT')
    result=extract(circuits[0],kind)
    expected=c.read(cfg['inputs']['static_metadata']['path'])['circuits'][kind]
    for key in ('ordered_metadata_sha256','gate_counts','global_phase_binary64_hex','operation_count'):
        c.require(result['metadata'][key]==expected[key],'STATIC_METADATA_IDENTITY:'+key)
    c.require(c.sha(row['path'])==row['sha256'],'QPY_UNCHANGED')
    result.update(SOURCE=cfg['SOURCE'],plan_sha256=cfg['plan_sha256'],input_sha256=row['sha256'],wall_seconds=time.monotonic()-started)
    c.save(Path(cfg['output'])/('extract_'+kind+'.json'),result)
    return result


def one_precision(base,precision,nqubits=13):
    from mpmath import mp
    ctx=mp.clone();ctx.dps=precision
    b=ctx.mpf(base.numerator)/base.denominator
    period=2*ctx.pi/(1<<nqubits)
    branch=int(ctx.floor(b/period+ctx.mpf(1)/2))
    delta=b-branch*period
    require(ctx.isfinite(delta) and -period/2<=delta<period/2,'FINITE_FIXED_PRINCIPAL_BRANCH')
    return dict(decimal_digits=precision,branch_integer=str(branch),
        principal_delta_rad=ctx.nstr(delta,precision),period_rad=ctx.nstr(period,precision),
        distance_to_branch_boundary_rad=ctx.nstr(period/2-abs(delta),precision))


def evaluate(wrapper,compiled,saved_phases):
    from mpmath import mp
    require(wrapper['kind']=='WRAPPER' and compiled['kind']=='COMPILED','EXACT_SAVED_PAIR')
    require(compiled['P_count']==0 and restore_rational(compiled['P_angle_sum_exact'])==0,'NO_COMPILED_P')
    base=restore_rational(compiled['global_phase_exact'])-restore_rational(wrapper['global_phase_exact'])-restore_rational(wrapper['P_angle_sum_exact'])/2
    evaluations=[one_precision(base,p) for p in (80,120)]
    ctx=mp.clone();ctx.dps=120
    a,b=(ctx.mpf(v['principal_delta_rad']) for v in evaluations)
    difference=abs(a-b);tolerance=ctx.mpf('1e-60')
    require(evaluations[0]['branch_integer']==evaluations[1]['branch_integer'] and
        difference<=tolerance and all(ctx.mpf(v['distance_to_branch_boundary_rad'])>tolerance for v in evaluations),
        'INTERNAL_PRECISION_OR_BRANCH_NOT_STABLE')
    comparisons={}
    require(set(saved_phases)=={'ordinary','extended'},'EXACT_SAVED_PHASE_REFERENCES')
    for name,phase in saved_phases.items():
        f=exact_float(phase);reference=ctx.mpf(f.numerator)/f.denominator
        comparisons[name]=dict(saved_common_phase_binary64_hex=phase.hex(),
            saved_common_phase_rad=ctx.nstr(reference,120),
            ledger_minus_saved_common_phase_rad=ctx.nstr(b-reference,120))
    return dict(status='SCALAR_LEDGER_COMPLETE_NO_ACCEPTANCE_OVERRIDE',
        B_exact=rational_payload(base),wrapper_P_angle_sum_exact=wrapper['P_angle_sum_exact'],
        evaluations=evaluations,internal_precision_difference_rad=ctx.nstr(difference,120),
        internal_precision_check='PASS_AT_MOST_1e-60_RAD_NOT_A_PHYSICAL_ERROR_CERTIFICATE',
        comparisons=comparisons,physical_match_threshold=None,causal_classification=None,
        branch_selected_to_fit_saved_phase=False,interpretive_limit='DETERMINANT_ALIAS_NOT_ALL_STATE_UNITARY_OR_CAUSAL_CERTIFICATE',
        nqubits=13,dimension=8192,new_compile=0,new_statevector_replays=0,new_sampling=0,
        phase_correction_applied=False,acceptance_override=False,old_STOP_remains_STOP=True,
        original_raw_threshold_changed=False,accepted_new_cost_axes=0,mandatory_STOP=True,next_phase_started=False)
