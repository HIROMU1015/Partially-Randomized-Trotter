"""G7 small enumeration oracle and cost reference ONLY; not a generator input."""
from fractions import Fraction as F
from itertools import product
from .g7_generator import FullReturnGenerator, reduce_word


def reference_events(generator):
    L = len(generator.p)
    if isinstance(generator, FullReturnGenerator):
        for l in range(0, generator.m, 2):
            for word in product(range(L), repeat=l):
                if reduce_word(word) == word:
                    for child in generator.kernel.parent(word).children:
                        yield generator.event(word, child)
    else:
        for index, group in enumerate(generator.groups):
            for word in product(range(L), repeat=group.degree):
                if group.mode == 'distinct' and word[0] == word[1]:
                    continue
                if group.mode == 'fixed_first' and (word[0] != group.first or word[1] == group.first):
                    continue
                for child in range(L):
                    if group.mode == 'fixed_first' and child == group.first:
                        continue
                    yield generator.event(index, word, child)


def resource_reference(generator, events, cache):
    acceptance = sum(e['proposal'] for e in events)
    if not 0 < acceptance <= 1 or (generator.arm != 'full_return' and acceptance != 1):
        raise ArithmeticError('reference proposal not normalized with zeros')
    moment = sum(e['proposal'] * e['weight'] ** 2 for e in events)
    per_trial = {'T_Rz': F(0), 'CX_fixed': F(0), '1Q_fixed_no_readout': F(0)}
    provider = [F(0)] * len(generator.p)
    for event in events:
        q = event['proposal']; row = cache[str(event['ratio'])]
        per_trial['T_Rz'] += q * 2 * row['T_count']
        per_trial['CX_fixed'] += q * 2  # controlled-Rz standard decomposition
        per_trial['1Q_fixed_no_readout'] += q * (2 * row['one_qubit_count'] + 4
                                                + int(event['phase_i_power'] == 2))
        for i, calls in event['provider_calls'].items():
            provider[i] += q * calls
    return {'reference_events': len(events), 'digital_acceptance': acceptance,
            'digital_weight_second_moment': moment, 'per_trial_fixed_cost': per_trial,
            'per_trial_provider_calls': provider,
            'reference_values_not_used_to_reduce_budget': True}
