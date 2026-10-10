"""Artificial JSON shapes only; no scientific event/provider calculation."""
from fractions import Fraction as F


def examples():
    shared = {'provider_calls': {2: 3, 0: 1}, 'fraction': F(2, 11)}
    return {
        'provider_integer_labels': {'event': {'provider_calls': {2: 3, 0: 1, 1: 2}}},
        'mixed_keys': {2: 'integer', 'label': None, F(2, 11): True},
        'collision_integer_first': {1: 'first', 'middle': 0, '1': 'last'},
        'collision_string_first': {'1': 'first', 'middle': 0, 1: 'last'},
        'multiple_collisions': {F(2, 11): 'old', 'x': 1, '2/11': 'new',
                                None: 0, 'None': 1},
        'insertion_order': {3: 'third', 1: 'first', 2: 'second'},
        'reverse_order': {2: 'second', 1: 'first', 3: 'third'},
        'bool_None_keys': {True: 1, False: 0, None: 'none'},
        'float_keys': {-0.0: 'negative zero', 1.25: 'fractional'},
        'tuple_bytes_keys': {(1, '漢'): 'tuple', b'a': 'bytes'},
        'nonfinite_keys_are_strings': {float('inf'): 1, float('nan'): 2},
        'nested': {'rows': ({'event': shared, 'native_ir': [('R', 0, 'mock', -1)]},),
                   'budget': {'mock': F(7, 11)}, 'confidence': (0.125, -0.0)},
        'fractions': {F(-7, 11): F(2**1600+1, 2**1200)},
        'shared_subtree': {'a': shared, 'b': [shared, shared]},
        'Unicode': {'研究😀': {'漢\n"': 'α\\😀', 7: '\t\r\n'}},
        'empty_and_scalars': [{}, [], (), None, True, False, 0, -7, 0.1, -0.0],
    }


def large_typed(count=50000):
    # Label/cost/angle fields are artificial serialization data, not results.
    return {'artificial_IO_only': [
        {'event': {'provider_calls': {2: 3, 0: 1}, 'fraction': F(i-25000, 7)},
         'collision': {1: 'discarded', 'middle': i, '1': ('😀漢', i, None)},
         'flags': [True, False]}
        for i in range(count)]}
