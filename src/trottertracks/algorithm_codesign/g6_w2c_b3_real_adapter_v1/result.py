"""Pure document -> result/evidence conversion. No callable resolution.

Structural checks establish exact representation and binding, never scientific
law, sampled frequency, IID, or mathematical angle correctness.
"""
from collections import Counter
from copy import deepcopy
from fractions import Fraction as F
from . import codec as C

IDS = ('input_id', 'policy_id', 'source_id', 'preflight_id')
EVENT_FIELDS = {'word','raw_word','child','ratio','phase_i_power','coefficient','proposal',
                'weight','provider_calls','helper','event_unitary','circuit_time_word'}


class AdapterFailure(RuntimeError):
    def __init__(self, category, reason, stage, *, input_id=None, details=None):
        super().__init__(reason)
        self.record = C.Document.create(dict(schema='g6_f3_failure_v1', category=category,
            reason=reason, stage=stage, input_id=input_id, details=details or {},
            mandatory_STOP=True, run_complete=False, is_random_zero=False))

    def to_dict(self):
        return dict(schema='b3_adapter_terminal_failure_v1', original=self.record.transport(),
                    category=self.record.read()['category'], reason=str(self),
                    stage=self.record.read()['stage'], mandatory_STOP=True, is_random_zero=False)


def reduced(word):
    result = []
    for i in word:
        if result and result[-1] == i:
            result.pop()
        else:
            result.append(i)
    return tuple(result)


def word(value):
    C.need(type(value) is tuple and all(type(i) is int and 0 <= i < 3 for i in value), 'WORD_TYPE_LABEL')
    return value


def validate_event(event, *, full):
    C.need(type(event) is dict and set(event) == EVENT_FIELDS, 'EVENT_FIELDS')
    raw, w = word(event['raw_word']), word(event['word'])
    C.need(reduced(raw) == w and (not full or raw == w), 'RAW_WORD_BINDING')
    C.need(len(raw) <= 4 and len(raw)%2 == 0, 'WORD_DEGREE')
    child = event['child']
    C.need(type(child) is int and 0 <= child < 3 and (not full or not w or child != w[0]), 'CHILD_BINDING')
    C.need(type(event['phase_i_power']) is int and event['phase_i_power'] == len(raw)%4,
           'EVENT_PHASE')
    C.need(type(event['helper']) is int and event['helper'] == 1 and
           type(event['event_unitary']) is str and bool(event['event_unitary']), 'EVENT_HELPER')
    C.need(type(event['circuit_time_word']) is tuple and event['circuit_time_word'] == tuple(reversed(w)),
           'CIRCUIT_WORD')
    C.need(all(type(event[k]) is F and event[k] > 0 for k in ('ratio','coefficient','proposal','weight')),
           'EXACT_EVENT_RATIONAL')
    C.need(event['proposal']*event['weight'] == event['coefficient'], 'COEFFICIENT_WEIGHT')
    calls = Counter(w); calls[child] += 2
    C.need(type(event['provider_calls']) is dict and all(type(k) is int and type(v) is int
           for k,v in event['provider_calls'].items()) and event['provider_calls'] == dict(calls),
           'PROVIDER_CALLS')
    return C.fingerprint(event)


def validate_f3_trace(trace, *, kind, reason=None, event=None):
    fields = {'draws','raw_word','degree','bits_consumed','draw_count',
              'completed_bit_map_calls_by_stage','extra_random_draws_for_audit','fresh_IID_verified'}
    C.need(type(trace) is dict and set(trace) == fields, 'F3_TRACE_FIELDS')
    raw = word(trace['raw_word']); degree = trace['degree']
    C.need(type(degree) is int and degree == len(raw) and degree in (0,2,4), 'F3_TRACE_DEGREE')
    suffix = [] if reason == 'RAW_NONREDUCED' else ['acceptance'] if kind == 'ZERO' else ['acceptance','child']
    stages = ['order'] + ['label']*degree + suffix
    C.need(type(trace['draws']) is list and len(trace['draws']) == len(stages), 'F3_DRAW_COUNT')
    for row, stage in zip(trace['draws'], stages):
        C.need(type(row) is dict and set(row) == {'stage','value','width'} and row['stage'] == stage,
               'F3_DRAW_STAGE')
        C.need(type(row['width']) is int and row['width'] == 160 and type(row['value']) is int and
               0 <= row['value'] < (1 << 160), 'F3_DRAW_VALUE')
    C.need(type(trace['draw_count']) is int and trace['draw_count'] == len(stages) and
           type(trace['bits_consumed']) is int and trace['bits_consumed'] == 160*len(stages), 'F3_TRACE_TOTAL')
    C.need(trace['extra_random_draws_for_audit'] == 0 and type(trace['extra_random_draws_for_audit']) is int
           and trace['fresh_IID_verified'] is False, 'F3_TRACE_AUDIT_IID')
    counts = trace['completed_bit_map_calls_by_stage']
    C.need(type(counts) is dict and set(counts) == {'order','label','acceptance','child'} and
           all(type(v) is int for v in counts.values()) and
           counts == {k:stages.count(k) for k in counts}, 'F3_TRACE_STAGE_COUNTS')
    C.need((reduced(raw) != raw) == (reason == 'RAW_NONREDUCED'), 'ZERO_TRACE_REASON')
    if event is not None:
        C.need(event['raw_word'] == raw, 'TRACE_EVENT_RAW')


def bind_context(data, expected):
    for k in (*IDS, 'trial_id', 'sigma'):
        C.need(k in data and k in expected and type(data[k]) is type(expected[k]) and data[k] == expected[k],
               'CONTEXT_MISMATCH:'+k)
    C.need(all(type(data[k]) is str and bool(data[k]) for k in (*IDS,'trial_id')) and
           type(data['sigma']) is int and data['sigma'] in (-1,1), 'CONTEXT_TYPES')


def failure(record):
    C.need(type(record) is C.Document, 'EXACT_FAILURE_DOCUMENT')
    data = record.read()
    C.need(type(data) is dict and set(data) == {'schema','category','reason','stage','input_id',
           'details','mandatory_STOP','run_complete','is_random_zero'}, 'FAILURE_FIELDS')
    C.need(data['schema'] == 'g6_f3_failure_v1' and all(type(data[k]) is str and bool(data[k])
           for k in ('category','reason','stage')) and (data['input_id'] is None or type(data['input_id']) is str)
           and type(data['details']) is dict and data['mandatory_STOP'] is True and
           data['run_complete'] is False and data['is_random_zero'] is False, 'TERMINAL_FAILURE')
    exc = AdapterFailure(data['category'],data['reason'],data['stage'],
                         input_id=data['input_id'],details=data['details'])
    C.need(exc.record == record, 'FAILURE_LOSSLESS')
    return exc


def adapt_f3(record, *, expected):
    """F3-shaped exact document; synthetic origin must be recorded by caller."""
    C.need(type(record) is C.Document, 'DOCUMENT_ONLY_NO_SCIENCE_OBJECTS')
    data = record.read()
    C.need(type(data) is dict, 'RESULT_DOCUMENT_MAPPING')
    if data.get('schema') == 'g6_f3_failure_v1':
        raise failure(record)
    base = {'schema','kind','trial_id',*IDS,'sigma','run_completion_or_confidence_certified'}
    C.need(data.get('schema') == 'g6_f3_trial_v1', 'F3_TRIAL_SCHEMA')
    bind_context(data, expected)
    C.need(data['run_completion_or_confidence_certified'] is False, 'NO_RUN_CERTIFICATE')
    kind = data['kind']
    C.need(kind in ('EVENT','ZERO','IDENTITY'), 'UNKNOWN_RESULT_VARIANT')
    event = None
    query = kind == 'EVENT' and 'selection_scope' in data
    if kind == 'EVENT':
        C.need(set(data) == base | {'event','trace'} | ({'selection_scope'} if query else set()), 'EVENT_TRIAL_FIELDS')
        event = data['event']; fp = validate_event(event, full=True)
        if query:
            C.need(data['trace'] is None and data['selection_scope'] ==
                   'deterministic formal event query; not a sampled trial', 'DETERMINISTIC_QUERY_SCOPE')
        else:
            validate_f3_trace(data['trace'],kind=kind,event=event)
        driver_kind, classification = (None,'EVENT_QUERY') if query else ('EVENT','ACCEPTED')
    elif kind == 'ZERO':
        C.need(set(data) == base | {'reason','trace','is_valid_zero_trial'} and
               data['is_valid_zero_trial'] is True, 'ZERO_TRIAL_FIELDS')
        reason = data['reason']
        C.need(reason in ('RAW_NONREDUCED','COIN_REJECT'), 'UNKNOWN_ZERO_REASON')
        validate_f3_trace(data['trace'],kind=kind,reason=reason)
        driver_kind, classification, fp = ('ZERO_R' if reason == 'RAW_NONREDUCED' else 'ZERO_C'),reason,None
    else:
        C.need(set(data) == base | {'deterministic_operator','bits_consumed'} and
               data['deterministic_operator'] == 'I' and type(data['bits_consumed']) is int and
               data['bits_consumed'] == 0, 'IDENTITY_FIELDS')
        driver_kind, classification, fp = None,'IDENTITY',None
    return dict(schema='b3_adapted_result_v1', original_kind=kind, classification=classification,
                driver_kind=driver_kind,event=deepcopy(event),event_fingerprint=fp,
                event_identity=record.digest, source_identity=data['source_id'],trial_id=data['trial_id'],
                codec=C.convert(record,source_identity=data['source_id']),
                is_random_zero=kind == 'ZERO', mandatory_STOP=False)


def adapt_g7(event, *, observation, expected, arm):
    """Static candidate for G7 dict/None + captured observation, never G7 call.

    Canonical group index/raw/child and Full packet/trace are required provenance.
    None alone cannot identify either stochastic zero branch.
    """
    C.need(type(observation) is C.Document, 'EXACT_G7_OBSERVATION_DOCUMENT')
    data = observation.read()
    fields = {'schema',*IDS,'trial_id','sigma','arm','branch','trace','packet_calls','group_index','event'}
    C.need(type(data) is dict and set(data) == fields and data['schema'] == 'g7_observation_candidate_v1',
           'G7_OBSERVATION_FIELDS')
    bind_context(data, expected)
    C.need(data['arm'] == arm, 'G7_ARM_BINDING')
    C.need(C.canonical(data['event']) == C.canonical(event), 'G7_EVENT_OBSERVATION')
    C.need(type(data['packet_calls']) is int and data['packet_calls'] >= 0, 'PACKET_PROVENANCE')
    if arm == 'full_return':
        base = {k:data[k] for k in (*IDS,'trial_id','sigma')}
        base.update(schema='g6_f3_trial_v1',run_completion_or_confidence_certified=False)
        if event is None:
            C.need(data['branch'] in ('RAW_NONREDUCED','COIN_REJECT'), 'NONE_WITHOUT_BRANCH_PROVENANCE')
            C.need(data['packet_calls'] == (0 if data['branch'] == 'RAW_NONREDUCED' else 1), 'PACKET_BRANCH')
            base.update(kind='ZERO', reason=data['branch'],trace=data['trace'],is_valid_zero_trial=True)
        else:
            C.need(data['branch'] == 'ACCEPTED' and data['packet_calls'] == 1, 'ACCEPTED_PACKET')
            base.update(kind='EVENT',event=event,trace=data['trace'])
        result = adapt_f3(C.Document.create(base), expected=expected)
        result['g7_observation'] = C.convert(observation,source_identity=data['source_id'])
        return result
    C.need(arm in ('ordinary','partial_return_tail','closed_P3_tail'), 'CANONICAL_ARM')
    C.need(event is not None and data['branch'] == 'ACCEPTED', 'CANONICAL_NONE_NOT_ZERO')
    C.need(type(data['group_index']) is int and data['group_index'] >= 0 and data['packet_calls'] == 0,
           'CANONICAL_GROUP_PROVENANCE')
    C.need(type(data['trace']) is dict and type(data['trace'].get('draws')) is list and
           bool(data['trace']['draws']), 'CANONICAL_TRACE_REQUIRED')
    fp = validate_event(event,full=False)
    return dict(schema='b3_adapted_result_v1',original_kind='EVENT',classification='ACCEPTED',
                driver_kind='EVENT',event=deepcopy(event),event_fingerprint=fp,event_identity=observation.digest,
                source_identity=data['source_id'],trial_id=data['trial_id'],
                codec=C.convert(observation,source_identity=data['source_id']),
                is_random_zero=False, mandatory_STOP=False)


def angle_binding(adapted, *, requery_input, requery_event, angle_record, synthetic):
    C.need((adapted['classification'],adapted['driver_kind']) in
           (('ACCEPTED','EVENT'),('EVENT_QUERY',None)), 'ANGLE_ACCEPTED_EVENT')
    event = adapted['event']
    expected_query = dict(word=event['word'],child=event['child'])
    source = C.restore(adapted['codec'],source_identity=adapted['source_identity']).read()
    if source['schema'] == 'g7_observation_candidate_v1' and source['arm'] != 'full_return':
        expected_query['group_index'] = source['group_index']
        expected_query['raw_word'] = event['raw_word']
    C.need(C.canonical(requery_input) == C.canonical(expected_query), 'REQUERY_INPUT')
    C.need(C.fingerprint(requery_event) == adapted['event_fingerprint'], 'REQUERY_EVENT_FINGERPRINT')
    C.need(type(angle_record) is C.Document, 'EXACT_ANGLE_DOCUMENT')
    data = angle_record.read()
    fields = {'schema','ratio','sigma','root_bits','terms','tau','context','sqrt_interval',
              'half_argument_interval','endpoint_series_evidence','phi_interval','representative',
              'representative_error_upper','signed_phi_interval','signed_representative','precision_check',
              'native_rotation_acquired','physical_provider_certified','basis'}
    C.need(type(data) is dict and set(data) == fields and data['schema'] == 'g6_f3_angle_enclosure_v1',
           'ANGLE_FIELDS')
    C.need(type(data['ratio']) is F and data['ratio'] == event['ratio'] and
           type(data['sigma']) is int and data['sigma'] == source['sigma'], 'ANGLE_RATIO_SIGMA')
    context = data['context']
    C.need(type(context) is dict and C.canonical(context) == C.canonical({**{k:source[k] for k in (*IDS,'sigma')},
           'event_id':adapted['event_identity'],'parent':event['word'],'child':event['child']}), 'ANGLE_CONTEXT')
    C.need(all(type(data[k]) is int and data[k] > 0 for k in ('root_bits','terms')) and
           type(data['tau']) is F and data['tau'] > 0, 'ANGLE_PRECISION_TYPES')
    for field in ('phi_interval','signed_phi_interval'):
        interval = data[field]
        C.need(type(interval) is tuple and len(interval) == 2 and all(type(v) is F for v in interval)
               and interval[0] <= interval[1], 'ENCLOSURE_TYPE')
    lo, hi = data['phi_interval']; mid, radius = (lo+hi)/2, (hi-lo)/2
    C.need(type(data['representative']) is F and data['representative'] == mid and
           type(data['representative_error_upper']) is F and data['representative_error_upper'] == radius,
           'ENCLOSURE_REPRESENTATIVE')
    sign = data['sigma']
    C.need(data['signed_phi_interval'] == ((lo,hi) if sign == 1 else (-hi,-lo)) and
           type(data['signed_representative']) is F and data['signed_representative'] == sign*mid,
           'SIGNED_ENCLOSURE')
    check = data['precision_check']
    C.need(type(check) is dict and C.canonical(check) == C.canonical(dict(
        name='representative_radius',left=radius,operator='<=',right=data['tau'],passed=radius<=data['tau']))
        and check['passed'] is True, 'ANGLE_DECLARED_PRECISION_CHECK')
    C.need(data['native_rotation_acquired'] is False and data['physical_provider_certified'] is False,
           'NO_NATIVE_CERTIFICATE')
    C.need(type(synthetic) is bool, 'ANGLE_ORIGIN_REQUIRED')
    return dict(schema='b3_angle_requery_binding_v1', event_identity=adapted['event_identity'],
                event_fingerprint=adapted['event_fingerprint'],requery_input=requery_input,
                requery_output=C.Document.create(requery_event).transport(),
                angle=C.convert(angle_record,source_identity=adapted['source_identity']),
                synthetic=synthetic,mathematical_precision_verified=False,
                terminal_STOP=False)
