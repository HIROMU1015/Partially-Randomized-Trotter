"""C7-3 q32/n8 domain binding of the unchanged C7-1 scientific helpers.

The historical module and its request guards remain untouched. A private
module instance binds only domain constants, the seed function and request
type; algorithms/emitters/statistics are the original functions, not copies.
Imports perform no sampling, circuit build, compile or numerical evaluation.
"""
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

_path=Path(__file__).with_name('track_d_c7_1_q16.py')
_spec=importlib.util.spec_from_file_location(__name__+'_shared',_path)
_shared=importlib.util.module_from_spec(_spec)
sys.modules[_spec.name]=_shared
_spec.loader.exec_module(_shared)
api,calibration=_shared.api,_shared.calibration
digest=calibration.digest
BASE='709d6ed9c234bf6d5a8848d3a30be69d88fd6476'
CALIBRATION,CALDIR=_shared.CALIBRATION,_shared.CALDIR
Q,SAMPLES,CUT,CX_CANDIDATE=32,8,16,-6960
COHORT='track_d_c7_3_q32_exploratory_paired_price_v1'
MASTER=2026110703
DOMAIN=COHORT+':independent_holdout:q32'
METRICS=calibration.METRICS


def binding(sample):
    from .c1_stage3_connection import trajectory_seed
    if type(sample) is not int or not 0<=sample<SAMPLES:
        raise ValueError('C7_3_FIXED_SAMPLE_DOMAIN')
    seeds=[trajectory_seed((MASTER,DOMAIN,BASE),'proposed',api.c3.paired.ENDPOINT,4,
                           'C7_3_PAIRED_PRICE_ONLY',sample,i) for i in range(Q)]
    return dict(q=Q,sample=sample,sample_id=f'q32_sample{sample:02d}',
                seed_domain=DOMAIN,seeds=seeds,cut_after_complete_step=CUT)


def seed_plan():
    root=Path(__file__).resolve().parents[2]
    previous=[CALDIR+'/seed_plan.json',
        'artifacts/track_d_c6b_q8/2026-11-07/technical/seed_plan.json',
        'artifacts/track_d_c7_1_q16/2026-11-07/technical/seed_plan.json']
    packets={p:json.loads((root/p).read_text()) for p in previous}
    prior={seed for packet in packets.values() for row in packet['generation_order'] for seed in row['seeds']}
    rows=[binding(i) for i in range(SAMPLES)];seeds=[s for row in rows for s in row['seeds']]
    if len(set(seeds))!=256 or set(seeds)&prior:raise ValueError('SEED_DOMAIN_COLLISION')
    return dict(cohort=COHORT,q=Q,samples=SAMPLES,generation_order=rows,master=MASTER,
        master_tuple=[MASTER,DOMAIN,BASE],
        seed_key_fields=['master_tuple','proposed',api.c3.paired.ENDPOINT,4,'C7_3_PAIRED_PRICE_ONLY','sample','occurrence'],
        seed_function='existing trajectory_seed SHA256(repr(tuple)) first8 little endian',PRNG='numpy1.26.4 PCG64',
        sampler='existing configure_rte / sample_occurrence / iter_sample_rte_events',
        previous_seed_plan_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in previous},
        previous_distinct_seeds=len(prior),C6a_C6b_C7_1_seed_overlap=0,new_domain=True,
        paired_methods=True,same_path_XY=True,ideal_IID_guaranteed=False,implemented_law_distance=None)


def sampling_context(origin,contract):
    """Law-only prefix of prepare_common; emit no head before all events seal.

    The existing helper unconditionally emits head circuits. Its numerical arm
    and configure functions are reused here; only this ingestion prefix is
    separated to enforce event-hash-before-circuit ordering.
    """
    from types import SimpleNamespace
    import math
    import numpy as np
    base=api.c3.base;origin=Path(origin)
    with np.load(origin/contract['input'],allow_pickle=False) as z:
        packet={k:z[k] for k in ('h','g','constant')}
    with np.load(origin/contract['candidate'],allow_pickle=False) as z:
        endpoint={k:z[k] for k in ('J','U')}
    with np.load(origin/contract['sector'],allow_pickle=False) as z:
        if not all(np.array_equal(endpoint[k],z[k]) for k in ('J','U')):raise ValueError('SAVED_JU_REFERENCE_BINDING')
    if any(endpoint[k].shape!=(8,4,4) for k in ('J','U')):raise ValueError('FIXED_CDF8_INPUT')
    arm=base.original.prepare_arm(**packet,**endpoint,role='proposed',
        source_identity=contract['saved_inputs'][contract['candidate']],
        input_identity=contract['saved_inputs'][contract['input']],ld=2)
    if arm.model.modes!=base.N:raise ValueError('FIXED_EIGHT_MODES')
    rows=json.loads((origin/contract['atoms']).read_text())['atoms'];actual={a.atom_id:a for a in arm.plan.atoms}
    if len(actual)!=len(rows):raise ValueError('SAVED_ATOM_COUNT')
    for row in rows:
        atom=actual[row['atom_id']]
        if (atom.frame_id!=row['frame_id'] or list(atom.support)!=row['support']
                or base.Fraction(atom.coefficient)!=base.Fraction(row['coefficient'])):raise ValueError('SAVED_SIGNED_ATOM')
    tail,config,_=base.original.configure(arm,base.H,32,0)
    law=json.loads((origin/'artifacts/research_c_near_practical_r32/2026-11-03/run01/PR/A/law.json').read_text())
    if arm.plan.extracted_identity!=law['identity'] or not math.isclose(tail.lambda_r,law['lambda_R'],rel_tol=1e-12,abs_tol=1e-14):
        raise ValueError('SAVED_IDENTITY_OR_LAMBDA')
    return SimpleNamespace(arm=arm,tail=tail,config=config)


@dataclass(frozen=True)
class Q32Request:
    sample:int
    seeds:tuple[int,...]
    events_sha256:str
    execution_source:str
    q:int=Q
    cut_after_complete_step:int=CUT
    science_source:str=api.SCIENCE_SOURCE
    endpoint:str=api.c3.paired.ENDPOINT
    input_packet_sha256:str=api.INPUT_PACKET
    law_sha256:str=api.LAW
    cohort:str=COHORT
    def __post_init__(self):
        row=binding(self.sample);object.__setattr__(self,'seeds',tuple(self.seeds))
        if self.q!=Q or self.cut_after_complete_step!=CUT or self.seeds!=tuple(row['seeds']):
            raise ValueError('FIXED_Q32_CUT16_SEEDS')
        if (self.science_source,self.endpoint,self.input_packet_sha256,self.law_sha256,self.cohort)!=(
                api.SCIENCE_SOURCE,api.c3.paired.ENDPOINT,api.INPUT_PACKET,api.LAW,COHORT):
            raise ValueError('C7_3_SOURCE_INPUT_LAW')
        for value,n in [(self.execution_source,40),(self.events_sha256,64)]:
            if len(value)!=n or any(c not in '0123456789abcdef' for c in value):raise ValueError('SOURCE_OR_EVENT_DIGEST')
    @property
    def ordered_samples(self):return tuple(f'q32_sample{self.sample:02d}_step{i:02d}' for i in range(Q))
    @property
    def fingerprint(self):return digest(asdict(self))


# Explicit, isolated domain binding. No mutation of the normal q16 namespace.
for _key in ('BASE','Q','SAMPLES','CUT','CX_CANDIDATE','COHORT','MASTER','DOMAIN'):
    setattr(_shared,_key,globals()[_key])
_shared.binding=binding
_shared.Q16Request=Q32Request
sample_record=_shared.sample_record
request_from_record=_shared.request_from_record


def prepare_fresh(context,record,reference_context):
    bundle,request=_shared.prepare_fresh(context,record,reference_context)
    bundle.identity['fixture_selection']='pre-fixed independent q32 domain; eight fresh whole-paths; pseudo-independent design only'
    return bundle,request


def price_record(request,**kwargs):
    row=_shared.price_record(request,**kwargs)
    row['estimate_kind']='exploratory_independent_q32_holdout_compiled'
    return row


def paired_statistics(records):
    value=_shared.paired_statistics(records)
    value['status']='COMPLETE_FINITE_Q32_HOLDOUT_DESCRIPTIVE'
    return value


def frozen_model(root):
    value=_shared.frozen_model(root)
    value.update(status='FROZEN_BEFORE_Q32_HOLDOUT_EXPLORATORY_ONLY',
        primary_rule='abs(observed paired mean - frozen prediction)+z*s8/sqrt(8)<=tau',
        secondary_rule='population-line diagnostic: abs(error)+z*sqrt(s8^2/8+saved_vcal)<=tau',
        C7_1_q16_used_for_fit=False,
        cross_q_covariance_qualification='zero ONLY conditional on independently designed C6a q2/q4 calibration sets; ideal independence unproved; unconditional coverage unqualified')
    return value


def diagnose(summary,model):
    value=_shared.diagnose(summary,model)
    value['limitations']=[x.replace('q16 only, no q>=32','q32 only, no q>=64') for x in value['limitations']]
    return value


def cx_candidate_residuals(summary):
    value=_shared.cx_candidate_residuals(summary)
    for row in value['rows']:row['residual_definition']='observed DeltaCX - (-6960)'
    value['global_formula_status']='CX_GLOBAL_FORMULA_UNQUALIFIED'
    return value


def wrapper_features(circuit,method):
    """Read existing circuits only; no additional build, compile or simulator.

    Reuse C7-2 max-plus recurrence on condition-free, directive-free wrappers.
    Filtered wire endpoints and an unweighted DAG longest path are distinct.
    """
    from collections import Counter
    from qiskit.converters import circuit_to_dag
    from .track_d_c7_2_toy import weighted_depth
    anc=circuit.num_qubits-1
    n=anc//2 if method=='reference' else anc
    def role(index):return 'ancilla' if index==anc else 'system' if index<n else 'reference'
    depths={key:weighted_depth(circuit,gate) for key,gate in
            [('total_depth',None),('rz_depth','rz'),('cx_depth','cx')]}
    known=api.c3.base.artificial.metrics(circuit)
    if any(depths[key]['depth']!=known[key] for key in depths):raise ValueError('C7_2_DEPTH_FEATURE_MISMATCH')
    roles=Counter()
    for item in circuit.data:
        if item.operation.name in ('cx','cswap'):
            roles[item.operation.name+':'+','.join(role(circuit.find_bit(q).index) for q in item.qubits)]+=1
    path=[node for node in circuit_to_dag(circuit).longest_path() if hasattr(node,'op')]
    return dict(method=method,qubits=circuit.num_qubits,metrics=known,wire_roles=[role(i) for i in range(circuit.num_qubits)],
        weighted_wire_finish=depths,total_DAG_longest_path_counts=dict(Counter(n.op.name for n in path)),
        total_DAG_longest_path_operations=len(path),coupling_roles=dict(roles),
        filtered_semantics='excluded operations synchronize wires with zero weight; no classical conditions/directives',
        filtered_critical_path_features='weighted wire finish times and maximum; total-DAG gate counts are a different path',
        prediction_without_compile=False,extra_local_compile_calls=0)


def raw_features(bundle,record):
    from collections import Counter
    supports=Counter();signs=Counter();runs=0;transitions=0;zero=0
    for step in record['ordered_events']:
        previous=None
        for event in step:
            for app in event['application_sequence']:
                basis=app['basis_id'];support=app['diagonal_pauli_support']
                supports[','.join(map(str,support))]+=1
                signs[str(app['coefficient_sign'])]+=1
                if basis!=previous:
                    runs+=1;transitions+=int(previous is not None)
                previous=basis
            zero+=int(event['rotation_angle']==0)
    return dict(status='RECORDED_FROM_EXISTING_RAW',q=Q,cut=CUT,basis_runs_within_tails=runs,
        different_basis_transitions_within_tails=transitions,events=sum(map(len,record['ordered_events'])),
        support_counts=dict(supports),coefficient_sign_counts=dict(signs),zero_signed_angles=zero,
        aggregate_only_no_ordered_events_published=True,
        scalar=bundle.identity['scalar'],extracted_identity=bundle.identity['extracted_identity'],
        phase_chi=bundle.phase,aggregated_scalar_phase=bundle.identity['aggregated_scalar_phase'],
        strong=bundle.identity['strong_symmetric'],reference=bundle.identity['reference_optimization'],
        head_tail_cross_fusion=False,global_cut_boundary_losses=1,
        logical_CSWAP=int(bundle.cores['reference'].count_ops().get('cswap',0)),
        lane_raw_metrics={name:api.c3.base.artificial.metrics(circuit) for name,circuit in
                          [('A_actual_inverse',bundle.later_adjoint),('B',bundle.earlier)]},
        wrappers={m+'_'+a:wrapper_features(c,m) for (m,a),c in bundle.wrappers.items()})
