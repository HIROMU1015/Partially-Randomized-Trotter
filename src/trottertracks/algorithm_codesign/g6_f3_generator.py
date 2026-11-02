"""F3 terminal fail-closed facade over frozen FullReturnGenerator.

Only caller-supplied bits and exact rational formal inputs are accepted. There
is no backend, measurement, estimator, scientific runner, selector or retry.
"""
from fractions import Fraction as F
from pathlib import Path

from . import g7_generator as legacy
from .return_aggregation import Interval, dyadic_distribution, dyadic_index, reduce_word
from .g6_f3_contract import (Evidence, F3Failure, fingerprint, invalid,
                             make_request, source_identity, validate_word)
from .g6_f3_precision import assess_precision, bind_law, enclose_atan


def _source():
    try:
        expected_directory = Path(__file__).resolve().parent
        if Path(legacy.__file__).resolve() != expected_directory / 'g7_generator.py':
            raise invalid('IMPORTED_SOURCE_LOCATION_MISMATCH', 'source_identity')
        return source_identity()
    except F3Failure:
        raise
    except Exception as exc:
        raise F3Failure('UNEXPECTED_TECHNICAL_FAILURE', 'SOURCE_IDENTITY_UNAVAILABLE',
                        'source_identity', details={'exception_type': type(exc).__name__}) from exc


def _state(generator):
    k = generator.kernel
    return {'parameters': {'p': generator.p, 'x': generator.x, 'm': generator.m,
                           'arm': generator.arm, 'eta': generator.eta, 'rho': generator.rho,
                           'root_bits': generator.K, 'probability_bits': generator.H},
            'kernel': {'p': k.p, 'x': k.x, 'm': k.m, 'chi': k.chi,
                       'first_passage': k.first_passage, 'returns': k.returns},
            'order': generator.order, 'labels': generator.labels,
            'B': (generator.B.lo, generator.B.hi), 'U': (generator.U.lo, generator.U.hi)}


class _TraceBits:
    """Audit deterministic bit mapping without making any additional draw."""
    def __init__(self, provider, generator):
        if not callable(getattr(provider, 'bits', None)):
            raise invalid('BITS_PROTOCOL_REQUIRED', 'bits.entrance')
        self.provider, self.generator = provider, generator
        self.draws, self.raw = [], []
        self.degree = None

    def bits(self, n):
        if n != self.generator.H:
            raise F3Failure('CONTRACT_INCONSISTENCY', 'BIT_WIDTH_CHANGED', 'bits.width')
        index = len(self.draws)
        stage = ('order' if index == 0 else 'label' if index <= self.degree
                 else 'acceptance' if index == self.degree+1 else 'child')
        if self.degree is not None and index > self.degree+2:
            raise F3Failure('CONTRACT_INCONSISTENCY', 'UNEXPECTED_EXTRA_DRAW', 'bits.' + stage)
        try:
            value = self.provider.bits(n)
        except F3Failure:
            raise
        except Exception as exc:
            raise F3Failure('UNEXPECTED_TECHNICAL_FAILURE', 'BIT_SOURCE_FAILED', 'bits.' + stage,
                            details={'exception_type': type(exc).__name__, 'draw_index': index}) from exc
        if type(value) is not int or not 0 <= value < (1 << n):
            raise invalid('INVALID_H_BIT_INTEGER', 'bits.' + stage, draw_index=index,
                          received_type=type(value).__name__)
        self.draws.append({'stage': stage, 'value': value, 'width': n})
        if index == 0:
            self.degree = 2*dyadic_index(self.generator.order, n, value)
        elif index <= self.degree:
            self.raw.append(dyadic_index(self.generator.labels, n, value))
        return value

    def evidence(self):
        return {'draws': self.draws, 'raw_word': tuple(self.raw), 'degree': self.degree,
                'bits_consumed': sum(d['width'] for d in self.draws),
                'draw_count': len(self.draws),
                'completed_bit_map_calls_by_stage': {
                    stage: sum(d['stage'] == stage for d in self.draws)
                    for stage in ('order', 'label', 'acceptance', 'child')},
                'extra_random_draws_for_audit': 0, 'fresh_IID_verified': False}


class GeneratorSession:
    """Prepare via prepare(); any failure leaves the session terminally stopped."""
    def __init__(self, request, source, certificate, generator=None):
        self._request, self._source, self._certificate = request, source, certificate
        self._generator = generator
        self._request_id = fingerprint(request.payload())
        self._source_id, self._preflight_id = source.digest, certificate.digest
        self._law_id = fingerprint(_state(generator)) if generator is not None else None
        self._failure = None

    @property
    def request(self):
        return self._request

    @property
    def preflight(self):
        return self._certificate

    @property
    def last_failure(self):
        return self._failure

    def status(self):
        return {'state': 'STOPPED' if self._failure else 'IDENTITY' if self._generator is None else 'READY',
                'input_id': self._request.input_id, 'source_id': self._source_id,
                'preflight_id': self._preflight_id, 'run_complete': False,
                'confidence_certified': False,
                'last_failure': self._failure.to_dict() if self._failure is not None else None}

    def _guard(self):
        if self._failure is not None:
            raise F3Failure('INVALID_INPUT', 'SESSION_ALREADY_STOPPED', 'session.guard',
                            input_id=self._request.input_id,
                            details={'original_failure_id': self._failure.digest})
        if fingerprint(self._request.payload()) != self._request_id:
            raise invalid('INPUT_OR_POLICY_IDENTITY_CHANGED', 'session.guard')
        if _source().digest != self._source_id:
            raise invalid('SOURCE_IDENTITY_CHANGED', 'session.guard')
        self._source.read()
        data = self._certificate.read()
        if self._certificate.digest != self._preflight_id or data['input_id'] != self._request.input_id:
            raise invalid('PREFLIGHT_IDENTITY_CHANGED', 'session.guard')
        if self._generator is not None:
            if not data['ready'] or fingerprint(_state(self._generator)) != self._law_id:
                raise invalid('ACTUAL_LAW_STATE_CHANGED', 'session.guard')

    def _stop(self, exc, stage, trial_id=None, trace=None, operation_context=None):
        if self._failure is not None:
            return exc
        original = exc.evidence.read() if isinstance(exc, F3Failure) else {
            'exception_type': type(exc).__name__, 'exception_message': str(exc)}
        if isinstance(exc, F3Failure):
            category, reason, where = exc.category, exc.reason, exc.stage
        elif isinstance(exc, ValueError):
            category, reason, where = 'CONTRACT_INCONSISTENCY', 'LEGACY_GUARD_AFTER_PREFLIGHT', stage
        else:
            category, reason, where = 'UNEXPECTED_TECHNICAL_FAILURE', 'UNEXPECTED_FAILURE', stage
        locations = []
        tb = exc.__traceback__
        while tb is not None:
            locations.append({'module_file': Path(tb.tb_frame.f_code.co_filename).name,
                              'function': tb.tb_frame.f_code.co_name, 'line': tb.tb_lineno})
            tb = tb.tb_next
        failure = F3Failure(category, reason, where, input_id=self._request.input_id, details={
            'trial_id': trial_id if type(trial_id) is str else None,
            'received_trial_id_type': type(trial_id).__name__, 'source_id': self._source_id,
            'policy_id': self._request.policy_id, 'preflight_id': self._preflight_id,
            'original': original, 'source_locations': locations,
            'trace': trace.evidence() if trace is not None else None,
            'operation_context': operation_context,
            'prefix_only_no_run_completion': True})
        self._failure = failure.evidence
        return failure

    def _result(self, kind, trial_id, **payload):
        return Evidence.create({'schema': 'g6_f3_trial_v1', 'kind': kind, 'trial_id': trial_id,
                                'input_id': self._request.input_id, 'policy_id': self._request.policy_id,
                                'source_id': self._source_id, 'preflight_id': self._preflight_id,
                                'sigma': self._request.sigma,
                                'run_completion_or_confidence_certified': False, **payload})

    @staticmethod
    def _trial_id(trial_id):
        if type(trial_id) is not str or not trial_id:
            raise invalid('NONEMPTY_TRIAL_ID_REQUIRED', 'trial.id')

    def _event_checks(self, event, word=None):
        req = self._request
        u = validate_word(event['word'], req)
        child = event['child']
        if type(child) is not int or not 0 <= child < len(req.p) or (u and child == u[0]):
            raise invalid('INVALID_CHILD', 'event.child')
        if (word is not None and u != word) or event['raw_word'] != u:
            raise F3Failure('CONTRACT_INCONSISTENCY', 'RAW_WORD_REMAPPED', 'event.word')
        if event['phase_i_power'] != len(u) % 4 or event['circuit_time_word'] != tuple(reversed(u)):
            raise F3Failure('CONTRACT_INCONSISTENCY', 'PARENT_PHASE_OR_ORDER_CHANGED', 'event.phase')
        for key in ('ratio', 'coefficient', 'proposal', 'weight'):
            if type(event[key]) is not F or event[key] <= 0:
                raise F3Failure('CONTRACT_INCONSISTENCY', 'INVALID_EXACT_EVENT_VALUE', 'event.' + key)
        if event['ratio'] > 2*req.x/(len(u)+1):
            raise F3Failure('CONTRACT_INCONSISTENCY', 'TANGENT_BOUND_VIOLATION', 'event.ratio')
        if event['proposal']*event['weight'] != event['coefficient']:
            raise F3Failure('CONTRACT_INCONSISTENCY', 'PI_WEIGHT_NOT_COEFFICIENT', 'event.weight')

    def sample(self, bits=None, *, trial_id):
        trace = None
        try:
            self._guard()
            self._trial_id(trial_id)
            if self._generator is None:
                return self._result('IDENTITY', trial_id, deterministic_operator='I', bits_consumed=0)
            trace = _TraceBits(bits, self._generator)
            event = self._generator.sample(trace)
            raw = tuple(trace.raw)
            if event is None:
                raw_zero = reduce_word(raw) != raw
                expected_draws = trace.degree+1 if raw_zero else trace.degree+2
                if len(trace.draws) != expected_draws:
                    raise F3Failure('CONTRACT_INCONSISTENCY', 'ZERO_PATH_TRACE_MISMATCH', 'generator.sample')
                return self._result('ZERO', trial_id, reason='RAW_NONREDUCED' if raw_zero else 'COIN_REJECT',
                                    trace=trace.evidence(), is_valid_zero_trial=True)
            if len(trace.draws) != trace.degree+3:
                raise F3Failure('CONTRACT_INCONSISTENCY', 'EVENT_PATH_TRACE_MISMATCH', 'generator.sample')
            self._event_checks(event, raw)
            return self._result('EVENT', trial_id, event=event, trace=trace.evidence())
        except Exception as exc:
            failure = self._stop(exc, 'generator.sample', trial_id, trace)
            if failure is exc:
                raise
            raise failure from exc

    def event(self, word, child, *, trial_id):
        try:
            self._guard()
            self._trial_id(trial_id)
            if self._generator is None:
                raise invalid('IDENTITY_HAS_NO_STOCHASTIC_EVENT', 'event.identity')
            word = validate_word(word, self._request)
            if type(child) is not int or not 0 <= child < len(self._request.p) or (word and child == word[0]):
                raise invalid('INVALID_CHILD', 'helper.child')
            event = self._generator.event(word, child)
            self._event_checks(event, word)
            return self._result('EVENT', trial_id, event=event, trace=None,
                                selection_scope='deterministic formal event query; not a sampled trial')
        except Exception as exc:
            context = {'word_type': type(word).__name__, 'child_type': type(child).__name__}
            if type(word) is tuple and all(type(i) is int for i in word):
                context['parent'] = word
            if type(child) is int:
                context['child'] = child
            failure = self._stop(exc, 'generator.event', trial_id, operation_context=context)
            if failure is exc:
                raise
            raise failure from exc

    def parent_packet(self, word):
        try:
            self._guard()
            if self._generator is None:
                raise invalid('IDENTITY_HAS_NO_PARENT_QUERY', 'parent.identity')
            word = validate_word(word, self._request)
            packet = self._generator.packet(word)
            par = packet['parent']
            return Evidence.create({'input_id': self._request.input_id, 'source_id': self._source_id,
                'policy_id': self._request.policy_id, 'preflight_id': self._preflight_id,
                'word': par.word, 'a': par.a, 's': par.s, 'children': par.children,
                'masses': par.masses, 'd_interval': (packet['d_interval'].lo, packet['d_interval'].hi),
                **{k: v for k, v in packet.items() if k not in ('parent', 'd_interval')}})
        except Exception as exc:
            context = {'word_type': type(word).__name__}
            if type(word) is tuple and all(type(i) is int for i in word):
                context['parent'] = word
            failure = self._stop(exc, 'generator.packet', operation_context=context)
            if failure is exc:
                raise
            raise failure from exc

    def angle_for_event(self, record):
        try:
            self._guard()
            if type(record) is not Evidence:
                raise invalid('EVENT_EVIDENCE_REQUIRED', 'angle.binding')
            data = record.read()
            if data.get('kind') != 'EVENT' or self._generator is None:
                raise invalid('ACCEPTED_EVENT_REQUIRED', 'angle.binding')
            expected_ids = {'input_id': self._request.input_id, 'policy_id': self._request.policy_id,
                            'source_id': self._source_id, 'preflight_id': self._preflight_id,
                            'sigma': self._request.sigma}
            if any(data.get(k) != v for k, v in expected_ids.items()):
                raise invalid('EVENT_CONTEXT_IDENTITY_MISMATCH', 'angle.binding')
            event = data.get('event')
            if type(event) is not dict or 'word' not in event or 'child' not in event:
                raise invalid('EVENT_MAPPING_REQUIRED', 'angle.binding')
            word = validate_word(event['word'], self._request)
            child = event['child']
            if type(child) is not int or not 0 <= child < len(self._request.p) or (word and child == word[0]):
                raise invalid('INVALID_CHILD', 'angle.binding')
            expected = self._generator.event(word, child)
            if fingerprint(event) != fingerprint(expected):
                raise invalid('EVENT_CONTENT_IDENTITY_MISMATCH', 'angle.binding')
            self._event_checks(expected, word)
            return enclose_atan(event['ratio'], tau=self._request.angle_tau,
                                root_bits=self._request.angle_root_bits, terms=self._request.angle_terms,
                                sigma=self._request.sigma,
                                context={**expected_ids, 'event_id': record.digest,
                                         'parent': event['word'], 'child': event['child']})
        except Exception as exc:
            failure = self._stop(exc, 'angle.enclosure', operation_context={
                'event_id': record.digest if type(record) is Evidence and type(record.digest) is str else None})
            if failure is exc:
                raise
            raise failure from exc


def prepare(p, x, m, **precision_and_context):
    """Entrance -> identity OR supplied-precision preflight -> sealed legacy law."""
    try:
        req = make_request(p, x, m, **precision_and_context)
    except TypeError as exc:
        raise invalid('REQUEST_ARGUMENTS_MISSING_OR_UNKNOWN', 'prepare.entrance',
                      exception_type=type(exc).__name__) from exc
    source = _source()
    if req.x == 0:
        identity = Evidence.create({'schema': 'g6_f3_identity_v1', 'input_id': req.input_id,
                                    'policy_id': req.policy_id, 'source_id': source.digest,
                                    'mode': 'IDENTITY', 'positive_preflight': 'NOT_APPLICABLE',
                                    'deterministic_operator': 'I', 'run_complete': False})
        return GeneratorSession(req, source, identity)
    if req.arm != 'full_return':
        raise invalid('CANONICAL_PREFLIGHT_NOT_IMPLEMENTED', 'prepare.arm')
    try:
        assessment = assess_precision(req, source)
    except F3Failure:
        raise
    except Exception as exc:
        raise F3Failure('UNEXPECTED_TECHNICAL_FAILURE', 'PREFLIGHT_EVALUATION_FAILED',
                        'prepare.preflight', input_id=req.input_id,
                        details={'exception_type': type(exc).__name__,
                                 'exception_message': str(exc), 'source_id': source.digest,
                                 'policy_id': req.policy_id}) from exc
    data = assessment.read()
    if not data['precision_conditions_satisfied']:
        raise F3Failure('TECHNICAL_PRECISION_FAILURE', 'PREFLIGHT_CONDITIONS_FAILED',
                        'prepare.preflight', input_id=req.input_id, details=data)
    try:
        g = legacy.FullReturnGenerator(req.p, req.x, req.m, root_bits=req.root_bits,
                                       probability_bits=req.probability_bits, eta=req.eta, rho=req.rho)
        expected_laws = {
            'order': dyadic_distribution(g.kernel.order_intervals(req.root_bits),
                                         req.probability_bits, req.eta),
            'labels': dyadic_distribution(tuple(Interval(v, v) for v in req.p),
                                          req.probability_bits, req.eta),
        }
        certificate = bind_law(assessment, _state(g), expected_laws, source)
        return GeneratorSession(req, source, certificate, g)
    except F3Failure:
        raise
    except Exception as exc:
        category = 'CONTRACT_INCONSISTENCY' if isinstance(exc, ValueError) else 'UNEXPECTED_TECHNICAL_FAILURE'
        raise F3Failure(category, 'INITIALIZATION_AFTER_PREFLIGHT_FAILED', 'prepare.generator',
                        input_id=req.input_id, details={'assessment': data,
                        'exception_type': type(exc).__name__, 'exception_message': str(exc)}) from exc
