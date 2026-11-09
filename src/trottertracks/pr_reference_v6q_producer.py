"""The preserved integer-CDF law connected to the preserved RTE constructor.

Scientific use needs its separately authorized scope. There is no default RNG,
seed/reseed operation, event cache, reweighting or change to any circuit builder.
The probability theorem assumes fresh independent uniform bits from the caller;
an arbitrary getrandbits implementation does not prove this assumption.
"""
import math
from types import SimpleNamespace

from .track_c_h4_finite_bits import integer_cdf, draw_cdf
from . import pr_reference_v6q_schedule as caller
from . import pr_reference_v6q_consumer as consumer

SOURCE = 'src/trottertracks/pr_reference_v6q_producer.py:CanonicalFiniteBitProducer'


def _rte_runtime():
    from trotterlib.rte import _make_event, finite_rte_distribution, make_rte_config
    return SimpleNamespace(make_event=_make_event, distribution=finite_rte_distribution, config=make_rte_config)


class _CheckedBits:
    def __init__(self, source):
        if not callable(getattr(source, 'getrandbits', None)):
            raise TypeError('Supply an explicit getrandbits source; there is no default RNG')
        self.source = source

    def getrandbits(self, bits):
        value = self.source.getrandbits(bits)
        if type(value) is not int or not 0 <= value < 1 << bits:
            raise ValueError('Bit source must return a builtin integer in [0,2^bits)')
        return value


def _cdf(probabilities, bits):
    values = tuple(probabilities)
    if not values or any(not math.isfinite(float(v)) or float(v) < 0 for v in values):
        raise ValueError('CDF probabilities must be finite, nonnegative and nonempty')
    return integer_cdf(values, bits)  # No intermediate float normalization.


class EventKernel:
    """One order draw followed by n+1 component draws, in SOURCE index order."""
    def __init__(self, components, distribution, bits, bit_source):
        if type(bits) is not int or bits < 1:
            raise ValueError('bits must be a positive builtin integer')
        if distribution.finite_taylor_order != 2 or distribution.orders != (0, 2):
            raise ValueError('This fixed producer supports K2/order0,2 only')
        if len(distribution.order_probabilities) != 2:
            raise ValueError('Order probability shape mismatch')
        self.components = tuple(components)
        if not self.components:
            raise ValueError('The randomized tail must not be empty')
        self.distribution, self.bits = distribution, bits
        self.bit_source = _CheckedBits(bit_source)
        self.order_cdf = _cdf(distribution.order_probabilities, bits)
        self.component_cdf = _cdf((c.probability for c in self.components), bits)
        self.make_event = _rte_runtime().make_event

    def draw(self):
        order_index = draw_cdf(self.order_cdf, self.bits, self.bit_source)
        count = self.distribution.orders[order_index]+1
        indices = [draw_cdf(self.component_cdf, self.bits, self.bit_source) for _ in range(count)]
        # First index is rotation; remaining indices are ordered products.
        # _make_event emits products before rotation and preserves all phases.
        return self.make_event(indices, self.components, self.distribution, order_index)


def fixed_parameters(metadata, prep):
    """The same numeric cfg/dist construction as old transfer.events_for.

    RTEConfig's inherited seed/PRNG fields are legacy carrier metadata only;
    they do not initialize or describe the supplied bit source.
    """
    caller.bind_round(metadata.m, q=metadata.q, r=metadata.r, bits=metadata.bits)
    if prep.ld != 3 or prep.num_system_qubits != 8 or prep.product_formula != '2nd':
        raise ValueError('Fixed H4/Partial S2/LD3 preparation required')
    tail = prep.rte_preparation.symbolic_tail
    if tail.lambda_r != prep.exact_rte_lambda_r:
        raise ValueError('Tail lambda and preparation lambda differ')
    runtime = _rte_runtime()
    tolerance = runtime.distribution(prep.exact_rte_lambda_r*.02/metadata.r, 2).step_truncation_residual_bound
    return runtime.config(tail, evolution_time=.02, rte_steps=metadata.r,
                          truncation_tolerance=tolerance, finite_taylor_order=2)


class CanonicalFiniteBitProducer:
    """Supplier for consume_saved_round; retain the caller's continuing stream.

    Each invocation generates new events without memoizing a trajectory or
    resetting the bit source. Uniformity/independence remains a model/input
    condition, including across invocations, shots, axes and rounds.
    """
    def __init__(self, bit_source):
        self.bit_source = _CheckedBits(bit_source)
        self.bit_source_type = type(bit_source).__module__+'.'+type(bit_source).__qualname__

    def __call__(self, metadata, prep):
        caller.bind_round(metadata.m, q=metadata.q, r=metadata.r, bits=metadata.bits)
        cfg, dist = fixed_parameters(metadata, prep)
        kernel = EventKernel(prep.rte_preparation.symbolic_tail.components, dist, metadata.bits, self.bit_source)
        events = tuple(kernel.draw() for _ in range(metadata.q*metadata.r))
        return consumer.DeclaredCanonicalEvents(metadata, prep, cfg, dist, events,
                                               consumer.CANONICAL_LAW, SOURCE)


def consume_canonical_axis(inputs, m, *, q, r, bits, axis, producer, mode='legacy'):
    """Build one physical axis-shot request; never recycle the other axis.

    The preserved consumer builds both X/Y templates from one core. Return
    only the requested axis; the next physical shot must invoke this function
    again with the continuing source. No measurement/backend loop is added.
    """
    if axis not in ('X', 'Y'):
        raise ValueError('axis must be X or Y')
    if not isinstance(producer, CanonicalFiniteBitProducer):
        raise TypeError('Use the implemented fixed-law producer with an explicit bit source')
    result = consumer.consume_saved_round(inputs, m, q=q, r=r, bits=bits,
        canonical_source=producer, mode=mode)
    return dict(circuit=result['native_axes'][axis], axis=axis,
        metadata=result['metadata'], phase_mode=result['phase_mode'], input_provenance=result['input_provenance'],
        bit_source_type=producer.bit_source_type,
        bits_model_assumption='fresh independent uniform words; not proved by the bit-source class name')
