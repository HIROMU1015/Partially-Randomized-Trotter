"""Artificial SHA256-counter interface. No real namespace dispatch exists."""
from hashlib import sha256
from .codec import need, fingerprint
from ..g6_w2c_b3_api_binding_v1.ports import install_guard

SEED = 'ARTIFICIAL_REAL_ADAPTER_V1'
ARMS = ('ordinary', 'partial_return_tail', 'closed_P3_tail', 'full_return')


def domain(arm, axis, *, namespace=SEED):
    need(type(namespace) is str and namespace == SEED, 'ARTIFICIAL_NAMESPACE_ONLY')
    need(arm in ARMS and axis in ('X', 'Y'), 'BIT_DOMAIN')
    return namespace + '/' + arm + '/' + axis


class ArtificialBits:
    def __init__(self, arm, axis, *, namespace=SEED):
        install_guard()
        self.domain = domain(arm, axis, namespace=namespace)
        self.counter = self.consumed_bits = 0
        self.trace = []
        self.failure_position = None
        self.stopped = False

    def bits(self, width):
        need(not self.stopped, 'BITS_TERMINAL_STOP')
        try:
            need(type(self.domain) is str and self.domain in
                 {domain(a,x) for a in ARMS for x in ('X','Y')}, 'ARTIFICIAL_DOMAIN_MUTATION')
            need(type(self.counter) is int and 0 <= self.counter < 2**64 and
                 type(self.consumed_bits) is int and self.consumed_bits >= 0, 'BIT_STATE_TYPE')
            need(type(width) is int and width > 0, 'BIT_WIDTH_POSITIVE_INT')
            before, value, available = self.counter, 0, 0
            while available < width:
                raw = sha256(self.domain.encode() + self.counter.to_bytes(8, 'big')).digest()
                self.counter += 1
                value = (value << 256) | int.from_bytes(raw, 'big')
                available += 256
            value >>= available - width
            self.consumed_bits += width
            self.trace.append(dict(width=width, counter_before=before, counter_after=self.counter,
                                   value_hex=hex(value), discarded_bits=available-width))
            return value
        except BaseException:
            self.failure_position = dict(draw_index=len(self.trace), counter=self.counter,
                                         consumed_bits=self.consumed_bits)
            self.stopped = True
            raise


def validate_trace(trace, *, arm, axis):
    need(type(trace) is dict and set(trace) == {'schema', 'domain', 'draws', 'counter_after',
        'bits_consumed', 'bits_consumed_before', 'artificial_only', 'real_consumption_equivalence'}, 'BIT_TRACE_FIELDS')
    need(trace['schema'] == 'b3_ARTIFICIAL_adapter_consumption_v1' and
         trace['domain'] == domain(arm, axis) and trace['artificial_only'] is True and
         trace['real_consumption_equivalence'] is False, 'BIT_TRACE_DOMAIN')
    need(type(trace['draws']) is list and bool(trace['draws']), 'MISSING_DRAWS')
    previous = None
    for row in trace['draws']:
        need(type(row) is dict and set(row) == {'width','counter_before','counter_after',
             'value_hex','discarded_bits'}, 'BIT_DRAW_FIELDS')
        need(all(type(row[k]) is int for k in ('width','counter_before','counter_after','discarded_bits')),
             'BIT_DRAW_INTEGER')
        n, before = row['width'], row['counter_before']
        need(n > 0 and before >= 0 and (previous is None or before == previous), 'BIT_DRAW_SEQUENCE')
        blocks = (n+255)//256
        need(row['counter_after'] == before+blocks and row['discarded_bits'] == 256*blocks-n,
             'BIT_COUNTER_DISCARD')
        raw = b''.join(sha256(trace['domain'].encode()+i.to_bytes(8,'big')).digest()
                       for i in range(before,before+blocks))
        need(row['value_hex'] == hex(int.from_bytes(raw,'big') >> (256*blocks-n)), 'BIT_VALUE_MISMATCH')
        previous = row['counter_after']
    need(type(trace['counter_after']) is int and trace['counter_after'] == previous and
         type(trace['bits_consumed']) is int and type(trace['bits_consumed_before']) is int and trace['bits_consumed_before'] >= 0 and
         trace['bits_consumed'] == trace['bits_consumed_before'] + sum(r['width'] for r in trace['draws']),
         'BIT_TOTAL')
    return fingerprint(trace)


def bits_factory(arm, axis):
    return ArtificialBits(arm, axis)


def real_bits(*args, **kwargs):
    raise RuntimeError('NO_REAL_BITS_IMPLEMENTATION_IN_THIS_RELEASE')
