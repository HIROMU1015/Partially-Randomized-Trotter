"""Share count probability tables between the partial and total Q3 evaluators."""
from contextlib import ExitStack, contextmanager
from unittest.mock import patch


def _identity(value):
    # Preserve binary64 signed zero and the provider's type checks.
    return type(value), value.hex() if isinstance(value, float) else value


@contextmanager
def reuse_round_tables(risk, extended_bound=None):
    original = risk.round_distribution
    tables = {}

    def round_distribution(row, *, h, target_energy, radius, probability_rule, maximum_outcomes):
        real, imag = row['raw']['real'], row['raw']['imag']
        key = tuple(map(_identity, (row['m'], real, imag, row['N_X'], row['N_Y'],
                                   h, target_energy, radius, probability_rule, maximum_outcomes)))
        if key not in tables:
            tables[key] = original(row, h=h, target_energy=target_energy, radius=radius,
                                   probability_rule=probability_rule, maximum_outcomes=maximum_outcomes)
        return tables[key]

    with ExitStack() as stack:
        stack.enter_context(patch.object(risk, 'round_distribution', round_distribution))
        if extended_bound is not None:
            # M=13's fixed bound is loaded with its own copy of the route's globals.
            stack.enter_context(patch.dict(extended_bound.__globals__, round_distribution=round_distribution))
        yield
