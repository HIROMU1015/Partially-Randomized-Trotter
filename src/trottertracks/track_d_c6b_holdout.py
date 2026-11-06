"""Opt-in C6b q8/n32 holdout; frozen C6a guards and models stay unchanged.

Imports do not sample, build or compile. Canonical sampling, CAR primitives,
C3 assembly, C5 price checks and C6a descriptive helpers are reused explicitly.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from statistics import NormalDist

from . import track_d_c6a_pilot as calibration

api = calibration.api
BASE = "c93f3c2373b158f8eff3800a62fe1a94cce5c3b0"
CALIBRATION = "e9d4e2f602d5f9e014507ca7e391719bab017b88"
CALDIR = "artifacts/track_d_c6a_paired_price/2026-11-06/run01"
COHORT = "track_d_c6b_q8_exploratory_paired_price_v1"
DOMAIN = calibration.RESERVED_HOLDOUT_DOMAIN
Q, SAMPLES, CUT = 8, 32, 4
METRICS = calibration.METRICS
digest = calibration.digest


def binding(sample):
    from .c1_stage3_connection import trajectory_seed
    if type(sample) is not int or not 0 <= sample < SAMPLES:
        raise ValueError("C6B_FIXED_SAMPLE_DOMAIN")
    master = (calibration.MASTER, DOMAIN, calibration.BASE)
    seeds = [trajectory_seed(master, "proposed", api.c3.paired.ENDPOINT, 3,
                            "C6B_PAIRED_PRICE_ONLY", sample, i) for i in range(Q)]
    return dict(q=Q, sample=sample, sample_id=f"q8_sample{sample:02d}",
                seed_domain=DOMAIN, seeds=seeds, cut_after_complete_step=CUT)


def seed_plan():
    rows = [binding(i) for i in range(SAMPLES)]
    seeds = [s for row in rows for s in row['seeds']]
    old = calibration.seed_plan()
    old_seeds = {s for row in old['generation_order'] for s in row['seeds']}
    if len(set(seeds)) != 256 or set(seeds) & old_seeds:
        raise ValueError("HOLDOUT_CALIBRATION_SEED_COLLISION")
    return dict(cohort=COHORT, q=Q, samples=SAMPLES, generation_order=rows,
        master=calibration.MASTER, master_tuple=[calibration.MASTER, DOMAIN, calibration.BASE],
        seed_key_fields=['master_tuple', 'proposed', api.c3.paired.ENDPOINT, 3,
                         'C6B_PAIRED_PRICE_ONLY', 'sample', 'occurrence'],
        seed_function=old['seed_function'], PRNG='numpy1.26.4 PCG64',
        sampler=old['sampler'], paired_methods=True, same_path_XY=True,
        reserved_domain_previously_unused=True, calibration_seed_overlap=0,
        ideal_IID_guaranteed=False, implemented_law_distance=None)


def sample_record(context, sample, execution_source, *, before_occurrence=None):
    # C6a.sample_record intentionally only accepts q2/q4. Keep it sealed and
    # adapt its short ingestion layer; the actual sampler is the same function.
    b = api.c3.base.original
    row = binding(sample)
    tail, config, distribution = b.configure(context.arm, api.c3.base.H, 32, 0)
    if config != context.config or tail != context.tail:
        raise ValueError("CANONICAL_LAW_CONTEXT_CHANGED")
    steps = []
    for i, seed in enumerate(row['seeds']):
        if before_occurrence is not None:
            before_occurrence(row['sample_id'], i, seed)
        events = b.connection.sample_occurrence(tail, distribution, config.rte_steps, seed)
        if len(events) != 32 or any(e.taylor_order or e.product_component_ids or e.phase != 1 for e in events):
            raise ValueError("C6B_R32_K0_EVENTS")
        steps.append([event.to_dict() for event in events])
    return dict(**row, cohort=COHORT, execution_source=execution_source,
        endpoint=api.c3.paired.ENDPOINT, law_sha256=api.LAW,
        input_packet_sha256=api.INPUT_PACKET, ordered_events=steps,
        sampler='numpy1.26.4 PCG64 / canonical iter_sample_rte_events', ideal_IID_guaranteed=False)


@dataclass(frozen=True)
class HoldoutRequest:
    sample: int
    seeds: tuple[int, ...]
    events_sha256: str
    execution_source: str
    q: int = Q
    cut_after_complete_step: int = CUT
    science_source: str = api.SCIENCE_SOURCE
    endpoint: str = api.c3.paired.ENDPOINT
    input_packet_sha256: str = api.INPUT_PACKET
    law_sha256: str = api.LAW
    cohort: str = COHORT

    def __post_init__(self):
        row = binding(self.sample)
        object.__setattr__(self, 'seeds', tuple(self.seeds))
        if self.q != Q or self.cut_after_complete_step != CUT or self.seeds != tuple(row['seeds']):
            raise ValueError('FIXED_Q8_CUT4_SEEDS')
        if (self.science_source, self.endpoint, self.input_packet_sha256, self.law_sha256, self.cohort) != (
                api.SCIENCE_SOURCE, api.c3.paired.ENDPOINT, api.INPUT_PACKET, api.LAW, COHORT):
            raise ValueError('HOLDOUT_SOURCE_INPUT_LAW')
        for value, n in [(self.execution_source, 40), (self.events_sha256, 64)]:
            if len(value) != n or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('SOURCE_OR_EVENT_DIGEST')

    @property
    def ordered_samples(self):
        return tuple(f'q8_sample{self.sample:02d}_step{i:02d}' for i in range(Q))

    @property
    def fingerprint(self):
        return digest(asdict(self))


def request_from_record(record):
    if any(record[k] != v for k, v in binding(record['sample']).items()) or record['cohort'] != COHORT:
        raise ValueError('HOLDOUT_SEED_BINDING')
    if (record['endpoint'], record['law_sha256'], record['input_packet_sha256']) != (
            api.c3.paired.ENDPOINT, api.LAW, api.INPUT_PACKET):
        raise ValueError('HOLDOUT_EVENT_INPUT_LAW')
    if len(record['ordered_events']) != Q or any(len(step) != 32 for step in record['ordered_events']):
        raise ValueError('WHOLE_PATH_EVENT_COUNT')
    return HoldoutRequest(record['sample'], tuple(record['seeds']), digest(record['ordered_events']), record['execution_source'])


def prepare_fresh(context, record, reference_context):
    request = request_from_record(record)
    if (context.arm.model.source_identity, context.arm.model.input_identity, context.arm.model.modes) != (
            request.endpoint, request.input_packet_sha256, api.c3.base.N):
        raise ValueError('CONTEXT_SOURCE_ENDPOINT_INPUT')
    from . import c_pr_aware_df_stage_b2 as b2
    b = api.c3.base.original
    context.builder.fusion_records.clear()
    steps = []
    for sample_id, rows in zip(request.ordered_samples, record['ordered_events']):
        events = api.c3.base.stored_events(rows)
        if len(events) != 32 or any(e.taylor_order or e.product_component_ids or e.phase != 1 for e in events):
            raise ValueError('FIXED_FRESH_R32_K0')
        steps.append(dict(sample=sample_id, events=events,
            reference=b2.trajectory_reference(context.arm, reference_context, api.c3.base.H, events),
            raw=b.build_kernel(context.arm, api.c3.base.H, events, context.config, builder=context.builder)))
    bundle = api.c3.prepare_steps(context, steps, CUT, ordered_samples=request.ordered_samples,
        cohort=COHORT, fixture_selection='pre-fixed independent holdout domain; 32 q8 paths; pseudo-independent design only')
    bundle.identity.update(request=asdict(request), request_fingerprint=request.fingerprint,
        optimization_policy=api.OPTIMIZATION, independent_reference='existing raw full256 CAR / B2 cached head and signed atoms')
    return bundle, request


def price_record(request, **kwargs):
    if not isinstance(request, HoldoutRequest) or kwargs['execution_source'] != request.execution_source:
        raise ValueError('EXPLICIT_HOLDOUT_REQUEST_SOURCE')
    row = api._checked_price(request, **kwargs)
    row.update(estimate_kind='exploratory_independent_q8_holdout_compiled', sample=request.sample,
               sample_id=binding(request.sample)['sample_id'])
    return row


def paired_statistics(records):
    import numpy as np
    expected = {(i, m, a) for i in range(SAMPLES) for m in api.METHODS for a in api.AXES}
    keys = [(r['sample'], r['method'], r['axis']) for r in records]
    if len(set(keys)) != len(keys) or set(keys) != expected or any(r['q'] != Q for r in records):
        raise ValueError('COMPLETE_PLANNED_32_REQUIRED_NO_COMPLETE_CASE')
    table = dict(zip(keys, records))
    labels = [(m, a, g) for m in api.METHODS for a in api.AXES for g in METRICS]
    def value(i, m, a, g):
        row = table[i, m, a]
        return row[g] if g == 'peak_qubits' else row['metrics'][g]
    matrix = np.array([[value(i, *label) for label in labels] for i in range(SAMPLES)], float)
    paired_labels = [(a, g) for a in api.AXES for g in api.METRICS]
    paired_matrix = np.array([[value(i, 'reference', a, g)-value(i, 'symmetric_directional', a, g)
                               for a, g in paired_labels] for i in range(SAMPLES)], float)
    rows = []
    for a in api.AXES:
        for g in METRICS:
            s = [value(i, 'symmetric_directional', a, g) for i in range(SAMPLES)]
            r = [value(i, 'reference', a, g) for i in range(SAMPLES)]
            delta = [y-x for x, y in zip(s, r)]
            rows.append(dict(axis=a, metric=g, symmetric=calibration.describe(s), reference=calibration.describe(r),
                paired_delta=calibration.describe(delta), method_sample_covariance=float(np.cov(s, r, ddof=1)[0, 1]),
                paired_deltas_by_sample=delta, sample_ids=[binding(i)['sample_id'] for i in range(SAMPLES)]))
    return dict(status='COMPLETE_FINITE_HOLDOUT_DESCRIPTIVE', q=Q, n_whole_paths=SAMPLES, rows=rows,
        sample_covariance_labels=labels, sample_covariance=np.cov(matrix, rowvar=False, ddof=1).tolist(),
        paired_covariance_labels=paired_labels, paired_sample_covariance=np.cov(paired_matrix, rowvar=False, ddof=1).tolist(),
        covariance_ddof=1, covariance_rank_limit=SAMPLES-1, XY_are_not_independent_samples=True,
        ideal_IID_guaranteed=False, implemented_law_distance=None, population_confidence_guaranteed=False,
        support='observed extrema only; no whole-law support or tail bound')


def frozen_model(root):
    """Read existing coefficients/covariance; do not fit, update or use holdout."""
    folder = root / CALDIR
    names = ['exploratory_affine.json', 'paired_statistics.json', 'C6b_planning_proposal.json']
    packets = {name: json.loads((folder/name).read_text()) for name in names}
    fit = packets['exploratory_affine.json']; proposal = packets['C6b_planning_proposal.json']
    primary = []
    for row in proposal['per_quantity']:
        if row['quantity'] != 'paired_delta':
            continue
        coefficients = {m: next(r for r in rows if r['metric'] == row['metric'] and
            r['axis'] == {'X':'cosine', 'Y':'sine'}[row['axis']]) for m, rows in fit['models'].items()}
        predictions = {m: Q*r['slope']+r['intercept'] for m, r in coefficients.items()}
        if predictions['reference']-predictions['symmetric_directional'] != row['hypothetical_affine_q8_value_for_planning_only']:
            raise ValueError('SAVED_MODEL_PREDICTION_IDENTITY')
        scale = abs(predictions['symmetric_directional'])
        if row['pre_holdout_scale'] != scale or row['tau'] != 1+.05*scale:
            raise ValueError('SAVED_PRE_HOLDOUT_WIDTH')
        primary.append(dict(axis=row['axis'], metric=row['metric'], method_predictions=predictions,
            predicted_paired_delta=row['hypothetical_affine_q8_value_for_planning_only'],
            symmetric_pre_holdout_scale=scale, tau=row['tau'],
            calibration_variance_conditional=row['calibration_variance_conditional']))
    if len(primary) != 12:
        raise ValueError('FIXED_PRIMARY_FAMILY_12')
    return dict(status='FROZEN_BEFORE_HOLDOUT_EXPLORATORY_ONLY', calibration_commit=CALIBRATION,
        calibration_source='6b6667026ec8b670f610e41141d195883a9eb513', q=Q, n=SAMPLES,
        input_file_sha256={n:hashlib.sha256((folder/n).read_bytes()).hexdigest() for n in names},
        saved_models=fit['models'], saved_coefficient_covariance=proposal['conditional_parameter_covariance'],
        primary_family_size=12, alpha_classical_nominal=.05,
        nominal_two_sided_normal_z=NormalDist().inv_cdf(1-.05/(2*12)), primary=primary,
        primary_rule='abs(observed paired mean - frozen prediction) + z*s8/sqrt(32) <= tau',
        conditioning='saved fitted prediction held fixed; calibration variance NOT re-added to primary',
        secondary_rule='population-line diagnostic: abs(error)+z*sqrt(s8^2/32+saved_vcal)<=tau',
        secondary_not_an_additional_confirmatory_family=True, absolute_prices_precision_PASS=False,
        assumptions='nominal normal approximation with sample SD; disjoint pseudo-stream design, not proven ideal joint IID; no support/tail/finite-bit-law/bias coverage',
        model_refit=False, confirmatory_PASS=False, qualified_95_percent_coverage=False,
        long_q_qualified=False, scientific_GO=False)


def diagnose(summary, model):
    if summary['n_whole_paths'] != SAMPLES or summary['q'] != Q:
        raise ValueError('COMPLETE_FIXED_Q8_FOR_DIAGNOSTICS')
    table = {(r['axis'], r['metric']):r for r in summary['rows']}
    z = model['nominal_two_sided_normal_z']; rows = []
    for frozen in model['primary']:
        stats = table[frozen['axis'], frozen['metric']]['paired_delta']
        error = stats['mean']-frozen['predicted_paired_delta']
        se = stats['sample_SD']/math.sqrt(SAMPLES)
        primary_margin = abs(error)+z*se
        combined = math.sqrt(se*se+frozen['calibration_variance_conditional'])
        rows.append(dict(**frozen, observed_paired_mean=stats['mean'], prediction_error=error,
            holdout_mean_nominal_SE=se, frozen_conditional_nominal_halfwidth=z*se,
            nominal_error_interval=[error-z*se, error+z*se], primary_margin=primary_margin,
            primary_status='EXPLORATORY_WITHIN_WIDTH' if primary_margin <= frozen['tau'] else 'EXPLORATORY_OUTSIDE_WIDTH',
            population_line_nominal_halfwidth=z*combined, population_line_margin=abs(error)+z*combined,
            population_line_diagnostic='EXPLORATORY_WITHIN_WIDTH' if abs(error)+z*combined <= frozen['tau'] else 'EXPLORATORY_OUTSIDE_WIDTH'))
    return dict(status='EXPLORATORY_WITHIN_WIDTH' if all(r['primary_status']=='EXPLORATORY_WITHIN_WIDTH' for r in rows)
                else 'EXPLORATORY_OUTSIDE_WIDTH', rows=rows, statistical_qualification='STATISTICALLY_UNQUALIFIED',
        n_whole_paths=SAMPLES, primary_family_size=12, alpha_classical_nominal=.05,
        nominal_two_sided_normal_z=z, model_fingerprint=digest(model), confirmatory_PASS=False,
        long_q_qualified=False, qualified_95_percent_coverage=False,
        outside_width_semantics='nominal conservative-margin rule not met; not a confidence-qualified proof that true error exceeds tau',
        zero_SD_semantics='finite sample SD=0 is not a whole-law zero variance or support theorem',
        limitations=['normal/tail/IID coverage unproved', 'ideal versus finite-bit/PRNG joint-law unqualified',
                     'affine model bias/domain not bounded', 'q8 only, no q>8 or RPE total',
                     'calibration uncertainty separate from frozen-conditional primary', 'no confirmed resource-advantage family'])
