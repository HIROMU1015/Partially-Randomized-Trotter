"""Common receipts over unchanged samplers. Entrance deliberately synthetic only.

Canonical runtime guards are NOT an F3 P0-P5 certificate. No H4 entry point.
"""
from dataclasses import asdict, is_dataclass
from fractions import Fraction as F
from support import source_rows, sha
from trottertracks.algorithm_codesign.g6_f3_contract import Evidence, fingerprint
from trottertracks.algorithm_codesign.g6_f3_generator import prepare
from trottertracks.algorithm_codesign.g6_f3_precision import enclose_atan
from trottertracks.algorithm_codesign.g7_generator import make_generator
from trottertracks.algorithm_codesign.g9_p5 import P5Closed
from trottertracks.algorithm_codesign.return_aggregation import reduce_word, dyadic_index

POLICY = dict(root_bits=256, probability_bits=160, eta=F(1, 10**12), rho=F(1, 10**12),
              angle_tau=F(1, 10**12), angle_root_bits=256, angle_terms=64)
METHODS = ('ordinary', 'partial_return_tail', 'closed_P3_tail', 'full_return', 'closed_P5_full')

def require(test, reason):
    if not test:
        raise ValueError(reason)

def state(value):
    if is_dataclass(value):
        return state(asdict(value))
    if isinstance(value, (list, tuple)):
        return tuple(state(v) for v in value)
    if isinstance(value, dict):
        return {str(k): state(v) for k, v in value.items() if not callable(v)}
    return value

class Trace:
    def __init__(self, bits):
        self.provider, self.draws = bits, []

    def bits(self, width):
        value = self.provider.bits(width)
        require(type(value) is int and 0 <= value < 1 << width, 'INVALID_BIT_INTEGER')
        self.draws.append((width, value))
        return value

class Family:
    def __init__(self, method, p, x, model, labels, sigma=1, *, scope):
        require(scope == 'SYNTHETIC_ONLY' and model.startswith('synthetic:'), 'REAL_ENTRANCE_CLOSED')
        require(method in METHODS and 1 <= len(p) <= 3 and sigma in (-1, 1), 'SYNTHETIC_DOMAIN')
        self.source = source_rows()
        self.method, self.sigma = method, sigma
        self.policy = dict(POLICY)
        self.target = dict(p=tuple(p), x=x, m=5, model=model, labels=labels,
                           scalar_policy='separate controlled phase', sigma=sigma)
        self.target_id = fingerprint(self.target)
        kwargs = {k: v for k, v in self.policy.items() if not k.startswith('angle_')}
        self.session = prepare(p, x, 5, sigma=sigma, **self.policy) if method == 'full_return' else None
        self.g = self.session._generator if self.session else (
            P5Closed(p, x, **kwargs) if method == 'closed_P5_full' else
            make_generator(p, x, 5, method, **kwargs))
        self.sealed = fingerprint(state(vars(self.g))) if not self.session else None
        self.context = dict(scope=scope, target_id=self.target_id, method=method,
            source_id=fingerprint(self.source), policy_id=fingerprint(self.policy),
            certificate=('F3_P0_P5_SYNTHETIC_INPUT' if self.session else
                         'CONDITIONAL_EXISTING_NOT_F3_CERTIFIED'),
            provider='direct_DF_Z_ZZ_ideal_v1', measurement='X:H;Y:Sdg,H;0:+1;1:-1')
        self.context_seal = fingerprint(self.context)
        self.failed = None

    def guard(self):
        require(self.failed is None, 'FAMILY_TERMINAL')
        require(self.source == source_rows(), 'SOURCE_CHANGED')
        require(fingerprint(self.target) == self.target_id, 'MODEL_CHANGED')
        require(fingerprint(self.policy) == self.context['policy_id'], 'PRECISION_CHANGED')
        require(fingerprint(self.context) == self.context_seal, 'RECEIPT_POLICY_CHANGED')
        if not self.session:
            require(fingerprint(state(vars(self.g))) == self.sealed, 'GENERATOR_STATE_CHANGED')

    def wrap(self, event, trial_id, trace, *, reason=None, formal=False):
        data = dict(schema='c1_45_microstep_v1', context=self.context, trial_id=trial_id,
                    kind='ZERO' if event is None else 'EVENT', trace=trace,
                    selection_scope='FORMAL_QUERY' if formal else None,
                    fresh_independent_bits='CALLER_ASSUMPTION_NOT_VERIFIED')
        if event is None:
            require(reason in ('RAW_NONREDUCED', 'COIN_REJECT'), 'INVALID_ZERO_REASON')
            data.update(reason=reason, is_valid_zero_trial=True)
        else:
            require(all(type(event[k]) is F and event[k] > 0 for k in
                        ('proposal', 'weight', 'coefficient', 'ratio')), 'EXACT_POSITIVE_EVENT_REQUIRED')
            require(event['proposal'] * event['weight'] == event['coefficient'], 'PI_WEIGHT_MISMATCH')
            require(event['circuit_time_word'] == tuple(reversed(event['word'])), 'WORD_ORDER')
            data['event'] = event
        return Evidence.create(data)

    def sample(self, bits, trial_id):
        try:
            self.guard()
            require(type(trial_id) is str and bool(trial_id), 'TRIAL_ID')
            if self.session:
                raw = self.session.sample(bits, trial_id=trial_id)
                d = raw.read()
                record = self.wrap(d.get('event'), trial_id, d['trace'], reason=d.get('reason'))
                payload = record.read()
                payload['upstream_F3'] = raw.to_dict()
                return Evidence.create(payload)
            trace = Trace(bits)
            event = self.g.sample(trace)
            require(event is not None, 'CANONICAL_UNEXPECTED_ZERO')
            self.guard()
            return self.wrap(event, trial_id, trace.draws)
        except Exception as exc:
            self.failed = type(exc).__name__ + ':' + str(exc)
            raise

    def formal(self, event, trial_id):
        self.guard()
        return self.wrap(event, trial_id, None, formal=True)

    def validate(self, record):
        self.guard()
        data = record.read()
        require(data['context'] == self.context, 'SOURCE_MODEL_POLICY_OR_FAMILY_MISMATCH')
        require(data['kind'] in ('ZERO', 'EVENT'), 'FAILURE_IS_NOT_ZERO')
        if self.session and data['selection_scope'] is None:
            draws = iter(data['trace']['draws'])
            class FullReplay:
                def bits(self, n):
                    row = next(draws)
                    require(row['width'] == n, 'F3_TRACE_WIDTH')
                    return row['value']
            actual = self.session.sample(FullReplay(), trial_id=data['trial_id']).read()
            require(next(draws, None) is None and actual['kind'] == data['kind'] and
                    actual.get('event') == data.get('event') and actual.get('reason') == data.get('reason') and
                    actual['trace'] == data['trace'], 'FULL_TRACE_OR_EVENT_MISMATCH')
        if data['kind'] == 'EVENT':
            e = data['event']
            require(e['raw_word'] == tuple(e['raw_word']) and reduce_word(e['raw_word']) == e['word'], 'REDUCTION')
            require(len(e['raw_word']) in (0, 2, 4) and e['phase_i_power'] == len(e['raw_word']) % 4,
                    'RAW_DEGREE_PHASE')
            require(e['proposal'] * e['weight'] == e['coefficient'] and e['proposal'] > 0 and e['weight'] > 0,
                    'ACTUAL_WEIGHT_REQUIRED')
            require(e['circuit_time_word'] == tuple(reversed(e['word'])), 'ORDER')
            require(all(type(i) is int and 0 <= i < len(self.target['p']) for i in (*e['word'], e['child'])),
                    'LABEL_RANGE')
            if self.session:
                expected = self.g.event(e['word'], e['child'])
                require(fingerprint(e) == fingerprint(expected), 'ACTUAL_FULL_EVENT_MISMATCH')
            elif data['selection_scope'] is None:
                # Replay the already recorded bit map; acquire no fresh bits.
                draws = iter(data['trace'])
                class Replay:
                    def bits(self, n):
                        width, value = next(draws)
                        require(width == n and type(value) is int and 0 <= value < 1 << n, 'TRACE_BITS')
                        return value
                expected = self.g.sample(Replay())
                require(next(draws, None) is None and fingerprint(e) == fingerprint(expected),
                        'ACTUAL_CANONICAL_EVENT_MISMATCH')
        else:
            require(data.get('is_valid_zero_trial') is True and
                    data.get('reason') in ('RAW_NONREDUCED', 'COIN_REJECT') and self.session is not None,
                    'UNSUPPORTED_ZERO')
        return data

    def angle(self, record):
        data = self.validate(record)
        require(data['kind'] == 'EVENT', 'EVENT_REQUIRED')
        return enclose_atan(data['event']['ratio'], tau=self.policy['angle_tau'],
            root_bits=self.policy['angle_root_bits'], terms=self.policy['angle_terms'], sigma=self.sigma,
            context=dict(event_id=record.digest, target_id=self.target_id, method=self.method))

    def build(self, record, adapter):
        from qiskit import QuantumCircuit
        data = self.validate(record)
        require(data['kind'] == 'EVENT' and adapter.model_id == self.target['model'] and
                adapter.labels_id == self.target['labels'] and adapter.p == self.target['p'], 'DF_BINDING')
        event = data['event']
        angle = self.angle(record).read()
        circuit = QuantumCircuit(adapter.n + 1)
        if event['phase_i_power'] == 2:
            circuit.z(adapter.n)
        for label in event['circuit_time_word']:
            adapter.append_q(circuit, label)
        adapter.append_q(circuit, event['child'], rotation=float(angle['signed_representative']))
        return circuit

def compare_contracts(left, right):
    left.guard()
    right.guard()
    require(left.target_id == right.target_id, 'DIFFERENT_FINITE_TARGET')
    return dict(common_target=True, identical_digital_law=left.context == right.context,
                full_p5_same_ideal_accepted_ensemble=set((left.method, right.method)) ==
                    {'full_return', 'closed_P5_full'}, native_price_acquired=False)
