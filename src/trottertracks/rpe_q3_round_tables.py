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


# The fixed M<=12 and M=13 risk sources use this same certificate function.
# Only the evaluation order of its independent NumPy calls is changed.
_CLOSURE_SOURCE = 'c56b8030fcbcdaa3d3cb929efeeb593021ef3427766ba17a8c0ee5967d7b1ee7'


def _parallel_closure(risk, tables, maximum_pairs, workers):
    from concurrent.futures import ThreadPoolExecutor
    from itertools import islice
    import numpy as np

    risk.S._integer(maximum_pairs, 'maximum_pairs', 1)
    checked = 0
    if any(t['status'] != 'ENUMERATED' for t in tables):
        return dict(status='UNKNOWN', reason='ROUND_COUNT_SCOPE_INCOMPLETE', checked_pairs=0)
    # Share read-only arrays rather than forking the potentially huge exact
    # probability tables. Bound outstanding work and keep phase order for the
    # first failure and the original checked_pairs count.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for previous, current in zip(tables, tables[1:]):
            old = np.unique([v['oracle_estimate'] for v in previous['outcomes'] if v['good']])
            phases = {v['principal']: v['oracle_estimate'] for v in current['outcomes'] if v['good']}
            if checked + len(old)*len(phases) > maximum_pairs:
                return dict(status='UNKNOWN', reason='OUTSIDE_DECLARED_TRANSITION_VERIFICATION_SCOPE', checked_pairs=checked)
            if not len(old):
                continue

            def check(item):
                phase, oracle = item
                value, _, best, runner, _ = risk.F._nearest(np.full(len(old), phase), old, current['q'])
                return not (np.any(value != oracle) or np.any(np.abs(runner-best) <= risk.TIE_TOLERANCE))

            pending = iter(phases.items())
            # Small tables do not benefit from thread scheduling.
            parallel = len(old)*len(phases) >= 100_000 and len(phases) > 1
            while batch := list(islice(pending, 4*workers if parallel else 256)):
                results = pool.map(check, batch) if parallel else map(check, batch)
                for good in results:
                    checked += len(old)
                    if not good:
                        # Wait for speculative calls in this bounded batch before
                        # changing the captured arrays or leaving the context.
                        return dict(status='UNKNOWN', reason='GOOD_SET_NOT_CLOSED_UNDER_ORIGINAL_DECODER',
                                    m=current['m'], checked_pairs=checked)
    return dict(status='VERIFIED_FINITE_BINARY64_SUPPORT', checked_pairs=checked,
                method='ALL_ADJACENT_GOOD_SUPPORT_PAIRS_ORIGINAL_SCALAR_TIE_TOLERANCE')


@contextmanager
def parallel_good_transitions(risk, workers, extended_bound=None):
    """Reuse the exact decoder, support, limits and ordered certificate result.

    Unknown implementations fall back to their original evaluator. Threads use
    the existing Q3 allocation and finish before the total-DP pool starts.
    """
    import hashlib
    import inspect
    import os
    import textwrap
    original = getattr(risk, 'verify_good_transitions', None)
    if type(workers) is not int or workers < 2 or original is None:
        yield False
        return
    try:
        source = textwrap.dedent(inspect.getsource(original)).strip()
    except (OSError, TypeError):
        source = ''
    if hashlib.sha256(source.encode()).hexdigest() != _CLOSURE_SOURCE:
        yield False
        return
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count() or 1
    workers = min(workers, affinity)
    if workers < 2:
        yield False
        return

    def verify(tables, *, maximum_pairs):
        return _parallel_closure(risk, tables, maximum_pairs, workers)

    with ExitStack() as stack:
        stack.enter_context(patch.object(risk, 'verify_good_transitions', verify))
        if extended_bound is not None:
            stack.enter_context(patch.dict(extended_bound.__globals__, verify_good_transitions=verify))
        yield True
