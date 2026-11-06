"""C6a opt-in q2/q4 pilot: canonical law binding and descriptive paired prices.

No sampler, circuits, compiler or molecular data enter on import. Existing C3/C5
builders, B2 CAR reference primitives and canonical finite-RTE sampler are reused.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import statistics

from . import track_d_whole_path_price as api

BASE = "9a6920b8355643701ab7436a4fd3b34862e9e3bd"
COHORT = "track_d_c6a_q2_q4_paired_price_pilot_v1"
MASTER = 2026110606
DOMAINS = {2: COHORT+":calibration:q2", 4: COHORT+":calibration:q4"}
RESERVED_HOLDOUT_DOMAIN = COHORT+":independent_holdout:q8:unused"
SAMPLES = 16
METRICS = (*api.METRICS, "peak_qubits")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def binding(q, sample):
    from .c1_stage3_connection import trajectory_seed
    if type(q) is not int or q not in DOMAINS or type(sample) is not int or not 0 <= sample < SAMPLES:
        raise ValueError("C6A_Q_SAMPLE_DOMAIN")
    master = (MASTER, DOMAINS[q], BASE)
    seeds = tuple(trajectory_seed(master, "proposed", api.c3.paired.ENDPOINT,
        q.bit_length()-1, "C6A_PAIRED_PRICE_ONLY", sample, i) for i in range(q))
    return dict(q=q, sample=sample, sample_id=f"q{q}_sample{sample:02d}", seed_domain=DOMAINS[q],
                seeds=list(seeds), cut_after_complete_step=q//2)


def seed_plan():
    rows = [binding(q, sample) for q in DOMAINS for sample in range(SAMPLES)]
    seeds = [seed for row in rows for seed in row['seeds']]
    if len(set(seeds)) != 96:
        raise ValueError("SEED_COLLISION")
    return dict(cohort=COHORT, master=MASTER, generation_order=rows,
        seed_function="existing trajectory_seed SHA256(repr(tuple)) first8 little endian",
        sampler="existing configure_rte / sample_occurrence / iter_sample_rte_events",
        PRNG="numpy.random.PCG64", paired_methods=True, same_path_XY=True,
        holdout_domain_reserved=RESERVED_HOLDOUT_DOMAIN, holdout_generated=False,
        ideal_IID_guaranteed=False, implemented_law_distance=None)


def sample_record(context, q, sample, execution_source, *, before_occurrence=None):
    """Exactly q canonical calls, each r32; caller persists before circuit use."""
    from . import c_pr_aware_df_stage_b as original
    row = binding(q, sample)
    tail, config, distribution = original.configure(context.arm, api.c3.base.H, 32, 0)
    if config != context.config or tail != context.tail:
        raise ValueError("CANONICAL_LAW_CONTEXT_CHANGED")
    steps = []
    for occurrence, seed in enumerate(row['seeds']):
        if before_occurrence is not None:before_occurrence(row['sample_id'], occurrence, seed)
        events = original.connection.sample_occurrence(tail, distribution, config.rte_steps, seed)
        if len(events) != 32 or any(e.taylor_order or e.product_component_ids or e.phase != 1 for e in events):
            raise ValueError("C6A_R32_K0_EVENTS")
        steps.append([event.to_dict() for event in events])
    return dict(**row, cohort=COHORT, execution_source=execution_source,
        endpoint=api.c3.paired.ENDPOINT, law_sha256=api.LAW,
        input_packet_sha256=api.INPUT_PACKET, ordered_events=steps,
        sampler="numpy1.26.4 PCG64 / canonical iter_sample_rte_events", ideal_IID_guaranteed=False)


@dataclass(frozen=True)
class FreshRequest:
    q: int
    sample: int
    seeds: tuple[int, ...]
    events_sha256: str
    execution_source: str
    cut_after_complete_step: int
    science_source: str = api.SCIENCE_SOURCE
    endpoint: str = api.c3.paired.ENDPOINT
    input_packet_sha256: str = api.INPUT_PACKET
    law_sha256: str = api.LAW
    cohort: str = COHORT

    def __post_init__(self):
        row = binding(self.q, self.sample)
        object.__setattr__(self, "seeds", tuple(self.seeds))
        if self.seeds != tuple(row['seeds']) or self.cut_after_complete_step != row['cut_after_complete_step']:
            raise ValueError("FIXED_SEEDS_HALF_STEP_CUT")
        if (self.science_source, self.endpoint, self.input_packet_sha256, self.law_sha256, self.cohort) != (
                api.SCIENCE_SOURCE, api.c3.paired.ENDPOINT, api.INPUT_PACKET, api.LAW, COHORT):
            raise ValueError("FRESH_SOURCE_INPUT_LAW")
        for value, length in [(self.execution_source, 40), (self.events_sha256, 64)]:
            if len(value) != length or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("SOURCE_OR_EVENT_DIGEST")

    @property
    def ordered_samples(self):
        return tuple(f"q{self.q}_sample{self.sample:02d}_step{i:02d}" for i in range(self.q))

    @property
    def fingerprint(self):
        return digest(asdict(self))


def request_from_record(record):
    row = binding(record['q'], record['sample'])
    if any(record[k] != v for k,v in row.items()) or record['cohort'] != COHORT:
        raise ValueError("SAMPLE_SEED_BINDING")
    if (record['endpoint'], record['law_sha256'], record['input_packet_sha256']) != (
            api.c3.paired.ENDPOINT, api.LAW, api.INPUT_PACKET):
        raise ValueError("EVENT_INPUT_LAW_BINDING")
    if len(record['ordered_events']) != record['q']:
        raise ValueError("WHOLE_PATH_STEP_COUNT")
    return FreshRequest(record['q'], record['sample'], tuple(record['seeds']),
        digest(record['ordered_events']), record['execution_source'], record['cut_after_complete_step'])


def car_context(context, packet):
    """Existing raw full CAR factors/atoms/exponentials; no ground solve or fit."""
    b = api.c3.base.original
    from . import c_pr_aware_df_stage_b2 as b2
    car = b.stage1.car_bilinears(api.c3.base.N)
    generators = [b.stage1.raw_base(b.stage1.corrected_kappa(packet['h'], packet['g']),
                                  packet['constant'], car)]
    generators += [b.stage1.raw_frame(J, U, car)[0] for J,U in zip(context.arm.J, context.arm.U)]
    return b2.reference_context(context.arm, generators, car, api.c3.base.H)


def prepare_fresh(context, record, reference_context):
    """Only the event/target ingestion differs; circuit assembly is shared C3."""
    request = request_from_record(record)
    if (context.arm.model.source_identity, context.arm.model.input_identity, context.arm.model.modes) != (
            request.endpoint, request.input_packet_sha256, api.c3.base.N):
        raise ValueError("CONTEXT_SOURCE_ENDPOINT_INPUT")
    b = api.c3.base.original
    from . import c_pr_aware_df_stage_b2 as b2
    context.builder.fusion_records.clear()
    steps=[]
    for sample_id, rows in zip(request.ordered_samples, record['ordered_events']):
        events = api.c3.base.stored_events(rows)
        if len(events) != 32 or any(e.taylor_order or e.product_component_ids or e.phase != 1 for e in events):
            raise ValueError("FIXED_FRESH_R32_K0")
        target = b2.trajectory_reference(context.arm, reference_context, api.c3.base.H, events)
        raw = b.build_kernel(context.arm, api.c3.base.H, events, context.config, builder=context.builder)
        steps.append(dict(sample=sample_id, events=events, reference=target, raw=raw))
    bundle = api.c3.prepare_steps(context, steps, request.cut_after_complete_step,
        ordered_samples=request.ordered_samples, cohort=COHORT,
        fixture_selection="pre-fixed fresh PCG64 domains; 16 paths/q; not certified ideal IID")
    bundle.identity.update(request=asdict(request), request_fingerprint=request.fingerprint,
        optimization_policy=api.OPTIMIZATION, independent_reference="existing raw full256 CAR / B2 cached head and signed atoms")
    return bundle, request


def price_record(request, **kwargs):
    if not isinstance(request, FreshRequest):
        raise TypeError("EXPLICIT_FRESH_REQUEST")
    if kwargs['execution_source'] != request.execution_source:
        raise ValueError("PRICE_EXECUTION_SOURCE")
    row=api._checked_price(request, **kwargs)
    row.update(estimate_kind="paired_calibration_pilot_compiled", sample=request.sample,
               sample_id=binding(request.q,request.sample)['sample_id'])
    return row


def describe(values):
    return dict(n=len(values), mean=statistics.mean(values), sample_SD=statistics.stdev(values),
                min=min(values), max=max(values))


def paired_statistics(records):
    """Complete planned pilot only; no favourable complete-case mean or CI."""
    import numpy as np
    expected={(q,i,m,a) for q in DOMAINS for i in range(SAMPLES) for m in api.METHODS for a in api.AXES}
    keys=[(r['q'],r['sample'],r['method'],r['axis']) for r in records]
    if len(set(keys)) != len(keys) or set(keys) != expected:
        raise ValueError("COMPLETE_PLANNED_32_REQUIRED_FOR_PILOT_SUMMARY")
    table=dict(zip(keys,records));out={}
    def value(q,i,m,a,g):
        row=table[q,i,m,a]
        return row[g] if g=='peak_qubits' else row['metrics'][g]
    labels=[(m,a,g) for m in api.METHODS for a in api.AXES for g in METRICS]
    for q in DOMAINS:
        matrix=np.array([[value(q,i,*label) for label in labels] for i in range(SAMPLES)],float)
        covariance=np.cov(matrix,rowvar=False,ddof=1)
        rows=[]
        for axis in api.AXES:
            for g in METRICS:
                s=[value(q,i,api.METHODS[0],axis,g) for i in range(SAMPLES)]
                r=[value(q,i,api.METHODS[1],axis,g) for i in range(SAMPLES)]
                delta=[y-x for x,y in zip(s,r)]
                rows.append(dict(axis=axis,metric=g,symmetric=describe(s),reference=describe(r),
                    paired_delta=describe(delta),method_sample_covariance=float(np.cov(s,r,ddof=1)[0,1]),
                    paired_deltas_by_sample=delta))
        out[str(q)]=dict(n_independent_path_units=SAMPLES,rows=rows,
            sample_covariance_labels=labels,sample_covariance_ddof=1,sample_covariance=covariance.tolist(),
            covariance_rank_limit=SAMPLES-1,XY_are_not_independent_samples=True)
    return dict(status='COMPLETE_FINITE_PILOT_DESCRIPTIVE',by_q=out,confidence_interval=None,
        SE_reported=False,ideal_IID_guaranteed=False,implemented_law_distance=None,
        independence='disjoint q/sample/occurrence PCG64 seeds; pseudo-independent conditional on fixed input/code; no mathematical joint-IID proof',
        shared_calibration='future model parameters shared across extrapolated rounds; retain method/axis/metric covariance',
        support='observed extrema only; no all-law support or uniform bound')


def exploratory_affine(summary):
    """Reuse the existing generic affine fit, without legacy dataset identity."""
    from dataclasses import asdict
    from trotterlib.rpe_hadamard_compiled_cost_proxy import RPEHadamardProxyReferencePoint, _fit_affine_metric_model
    models={}
    for method,field in zip(api.METHODS,['symmetric','reference']):
        points=[]
        for q in DOMAINS:
            for axis in api.AXES:
                rows={r['metric']:r[field] for r in summary['by_q'][str(q)]['rows'] if r['axis']==axis}
                points.append(RPEHadamardProxyReferencePoint('calibration',q.bit_length()-1,q,{'X':'cosine','Y':'sine'}[axis],
                    tuple((g,rows[g]['mean']) for g in api.METRICS),
                    tuple((g,None) for g in api.METRICS), digest(dict(q=q,method=method,axis=axis,rows=rows))))
        models[method]=[asdict(_fit_affine_metric_model(tuple(points),axis={'X':'cosine','Y':'sine'}[a],metric=g,weighting='uniform'))
                         for a in api.AXES for g in api.METRICS]
    return dict(status='EXPLORATORY_TWO_Q_INTERPOLATION_ONLY',formula='slope*q+intercept',models=models,
        axis_convention={'cosine':'X','sine':'Y'},residual_degrees_of_freedom=0,long_q_qualified=False,holdout_generated=False,
        confidence_interval=None,prediction_error_bound=None,
        other_candidates='native boundary counts and Reference lane/critical path models need independent features/holdout; no identifiable separate effects from two q means')
