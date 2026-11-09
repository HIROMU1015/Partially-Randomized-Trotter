"""Artificial CDF/kernel/consumer fixtures only; no saved H4 input."""
from collections import Counter
from fractions import Fraction as F
import importlib.util
import itertools
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks import pr_reference_v6q_producer as producer
from trottertracks import pr_reference_v6q_schedule as caller
from trottertracks import pr_reference_v6q_consumer as consumer


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT/path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Only the existing RTE numerical dataclasses/constructor, with artificial
# components/tau. No trotterlib convenience imports, matrices or H4 cache.
rte = load_file('artificial_rte_constructor', 'src/trotterlib/rte.py')
outer = load_file('existing_outer_consumer_mocks', 'tests/test_pr_reference_v6q_consumer.py')
RUNTIME = NS(make_event=rte._make_event, distribution=rte.finite_rte_distribution, config=rte.make_rte_config)


class ScriptBits:
    def __init__(self, values):
        self.values = iter(values)
        self.calls = []

    def getrandbits(self, bits):
        self.calls.append(bits)
        return next(self.values)

    def seed(self, *_):
        raise AssertionError('Do not seed/reset a supplied source')


def components():
    return tuple(rte.RTEComponent(component_id=name, probability=p, coefficient_abs=p,
        coefficient_sign=sign, df_fragment_id='artificial-fragment', basis_id='artificial-basis',
        basis_hash='artificial-basis-hash', diagonal_pauli_support=support)
        for name,p,sign,support in [('a',.25,-1,(0,)), ('b',.75,1,(0,1))])


class ProducerTests(unittest.TestCase):
    def test_exact_normalization_boundaries_and_zero_cells(self):
        for weights in ((.1,.2,.7), (0.,.25,.75), (2.,6.)):
            bits = 4
            cdf = producer._cdf(weights, bits)
            values = [F.from_float(float(x)) for x in weights]
            running = F(0); expected = []
            for value in values:
                running += value
                z = running/sum(values)*16
                expected.append(z.numerator//z.denominator)
            self.assertEqual(cdf, tuple(expected))
            counts = Counter(producer.draw_cdf(cdf, bits, producer._CheckedBits(ScriptBits([u]))) for u in range(16))
            self.assertEqual([counts[i] for i in range(len(weights))],
                             [right-left for left,right in zip((0,*cdf[:-1]),cdf)])
            for i in range(len(cdf)):
                left = 0 if i == 0 else cdf[i-1]
                if left < cdf[i]:
                    for u in (left,cdf[i]-1):
                        self.assertEqual(producer.draw_cdf(cdf,bits,producer._CheckedBits(ScriptBits([u]))), i)

    def test_exhaustive_event_kernel_law_not_monte_carlo(self):
        comps = [NS(probability=.25), NS(probability=.75)]
        dist = NS(finite_taylor_order=2, orders=(0,2), order_probabilities=(.5,.5))
        counts = Counter()
        with patch.object(producer, '_rte_runtime', return_value=NS(make_event=lambda ids,c,d,o:(o,tuple(ids)))):
            for word in itertools.product(range(4), repeat=4):
                source = ScriptBits(word)
                event = producer.EventKernel(comps,dist,2,source).draw()
                counts[event] += 1
                self.assertEqual(len(source.calls), 2 if event[0] == 0 else 4)
        for (order_index,ids), count in counts.items():
            probability = F(1,2)
            for index in ids:
                probability *= (F(1,4),F(3,4))[index]
            self.assertEqual(F(count,256), probability)
        self.assertEqual(len(counts), 2+2**3)

    def test_existing_constructor_exact_order_sign_angle_and_normalization(self):
        comps = components()
        dist = rte.finite_rte_distribution(.125,2)
        with patch.object(producer, '_rte_runtime', return_value=RUNTIME):
            for bits in (4,66,88):
                temporary = producer.EventKernel(comps,dist,bits,ScriptBits([]))
                for order_index,ids in ((0,[0]), (1,[0,1,0])):
                    order_left = 0 if order_index == 0 else temporary.order_cdf[0]
                    lefts = [0 if i == 0 else temporary.component_cdf[i-1] for i in ids]
                    source = ScriptBits([order_left,*lefts])
                    event = producer.EventKernel(comps,dist,bits,source).draw()
                    expected = rte._make_event(ids,comps,dist,order_index)
                    self.assertEqual(event.to_dict(), expected.to_dict())
                    self.assertEqual(event.rotation_angle.hex(), expected.rotation_angle.hex())
                    self.assertEqual(event.event_normalization.hex(), dist.exact_finite_distribution.hex())
                    self.assertEqual(event.phase, complex((-1)**order_index))
                    self.assertEqual(event.unsigned_rotation_angle, -event.rotation_angle)
                    self.assertEqual(event.selected_component_ids, tuple(comps[i].component_id for i in [*ids[1:],ids[0]]))
                    self.assertEqual(source.calls, [bits]*(2 if order_index == 0 else 4))
        self.assertIs(temporary.components[0], comps[0])

    def test_rejection_does_not_retry_or_reweight(self):
        for weights in ((), (0.,0.), (-.1,1.), (float('nan'),1.), (float('inf'),1.)):
            with self.subTest(weights=weights), self.assertRaises((ValueError,ArithmeticError)):
                producer._cdf(weights,4)
        with self.assertRaises(ArithmeticError):
            producer._cdf((2.**-60,1.),2)  # Do not erase/retry a positive category.
        for value in (-1,16,True,1.5):
            source = ScriptBits([value])
            with self.subTest(value=value), self.assertRaises(ValueError):
                producer._CheckedBits(source).getrandbits(4)
            self.assertEqual(len(source.calls),1)
        with self.assertRaises(TypeError):
            producer.CanonicalFiniteBitProducer(None)
        dist = NS(finite_taylor_order=2,orders=(0,2),order_probabilities=(.5,.5))
        for bits in (0,-1,True,2.):
            with self.subTest(bits=bits), self.assertRaises(ValueError):
                producer.EventKernel(components(),dist,bits,ScriptBits([]))

    def fixture(self, m=0):
        f = outer.ConsumerTests().fixture(m)
        tail = f['context'].prep.rte_preparation.symbolic_tail
        tail.lambda_r, tail.components = .5, components()
        return f

    def test_supplier_calls_continue_stream_and_return_fresh_events(self):
        f = self.fixture()
        bits = ScriptBits([0,0,0,0])
        source = producer.CanonicalFiniteBitProducer(bits)
        with patch.object(producer, '_rte_runtime', return_value=RUNTIME):
            first = source(f['meta'],f['context'].prep)
            second = source(f['meta'],f['context'].prep)
        self.assertIsNot(first.events, second.events)
        self.assertIsNot(first.events[0], second.events[0])
        self.assertEqual(bits.calls,[66]*4)
        self.assertEqual(first.producer_source,producer.SOURCE)
        self.assertIs(first.prep,f['context'].prep)

    def test_real_supplier_to_mock_consumer_two_axis_shot_requests(self):
        f = self.fixture()
        bits = ScriptBits([0,0,0,0])
        source = producer.CanonicalFiniteBitProducer(bits)
        with patch.object(producer, '_rte_runtime', return_value=RUNTIME), \
             patch.object(consumer,'_runtime',return_value=f['runtime']), \
             patch.object(caller,'_legacy_modules',return_value=(f['pilot'],f['transfer'])):
            results = [producer.consume_canonical_axis(f['context'],0,q=1,r=1,bits=66,
                axis=axis,producer=source,mode='exact_binary_m17') for axis in ('X','Y')]
        self.assertEqual(bits.calls,[66]*4)
        self.assertEqual([r['axis'] for r in results],['X','Y'])
        self.assertEqual([r['phase_mode'] for r in results],['legacy','legacy'])
        self.assertIsNot(results[0]['circuit'],results[1]['circuit'])
        events = [c.args[4] for c in f['transfer'].assemble.call_args_list]
        self.assertIsNot(events[0],events[1])
        for result in results:
            self.assertEqual(result['circuit'].operations[0],('reset',tuple(range(8,16))))
            self.assertTrue(result['bit_source_type'].endswith('.ScriptBits'))
        self.assertEqual(f['runtime'].compile_reference.call_count,4)  # Existing two templates/request; only one returned.

    def test_m17_bits_binding_without_whole_event_generation(self):
        f = self.fixture(17)
        source = producer.CanonicalFiniteBitProducer(ScriptBits([]))
        sentinel = RuntimeError('Stop artificial test before the whole event loop')
        cfg,dist = f['cfg'],NS(finite_taylor_order=2,orders=(0,2),order_probabilities=(.5,.5))
        with patch.object(producer,'fixed_parameters',return_value=(cfg,dist)), \
             patch.object(producer,'EventKernel',side_effect=sentinel) as kernel:
            with self.assertRaisesRegex(RuntimeError,'before the whole'):
                source(f['meta'],f['context'].prep)
        self.assertEqual(kernel.call_args.args[2],88)
        result = dict(native_axes={'X':'artificial-X','Y':'artificial-Y'},metadata=f['meta'],
                      phase_mode='exact_binary_m17',input_provenance={})
        with patch.object(consumer,'consume_saved_round',return_value=result) as consume:
            picked = producer.consume_canonical_axis(f['context'],17,q=131072,r=32,bits=88,
                axis='Y',producer=source,mode='exact_binary_m17')
        self.assertEqual(picked['circuit'],'artificial-Y')
        self.assertIs(consume.call_args.kwargs['canonical_source'],source)

    def test_metadata_and_invalid_axis_rejected_before_draws(self):
        f = self.fixture()
        bits = ScriptBits([])
        source = producer.CanonicalFiniteBitProducer(bits)
        for metadata in (caller.SavedRound(17,131072,32,87),caller.SavedRound(18,262144,32,88)):
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                source(metadata,f['context'].prep)
        with self.assertRaises(ValueError):
            producer.consume_canonical_axis(f['context'],0,q=1,r=1,bits=66,axis='Z',producer=source)
        self.assertEqual(bits.calls,[])


if __name__ == '__main__':
    unittest.main()
