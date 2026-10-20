"""C-0 toy-only mechanics. No molecular inputs, compiler, provider or grant issuer."""
from fractions import Fraction
from itertools import permutations
from math import factorial
import hashlib
import json
import os
import random
from pathlib import Path

TOY_P = (Fraction(1, 2), Fraction(1, 3), Fraction(1, 6))
TOY_R = 3
STRATA = ('S0_ONLY', 'S2_SINGLE', 'S2_MULTI', 'S4_PRESENT')


def member(n2, n4, stratum):
    if stratum == 'S0_ONLY': return n2 == n4 == 0
    if stratum == 'S2_SINGLE': return n2 == 1 and n4 == 0
    if stratum == 'S2_MULTI': return n2 >= 2 and n4 == 0
    if stratum == 'S4_PRESENT': return n4 >= 1
    raise ValueError('UNKNOWN_STRATUM')


def toy_count_law(stratum):
    """Exact rational test oracle for fixed R=3 synthetic probabilities only."""
    rows = []
    for n4 in range(TOY_R + 1):
        for n2 in range(TOY_R - n4 + 1):
            if member(n2, n4, stratum):
                n0 = TOY_R - n2 - n4
                count = factorial(TOY_R) // (factorial(n0)*factorial(n2)*factorial(n4))
                rows.append(((n2, n4), count*TOY_P[0]**n0*TOY_P[1]**n2*TOY_P[2]**n4))
    total = sum(w for _, w in rows)
    return [(counts, w/total) for counts, w in rows]


def toy_order_word(n2, n4, rng):
    positions = rng.sample(range(TOY_R), n2+n4)
    word = [0]*TOY_R
    for p in positions[:n2]: word[p] = 2
    for p in positions[n2:]: word[p] = 4
    return tuple(word)


def toy_words(n2, n4):
    return set(permutations((0,)*(TOY_R-n2-n4) + (2,)*n2 + (4,)*n4))


def toy_components(word):
    """Indices are symbolic; first index is rotation, the remaining are products."""
    rng = random.Random('SYNTHETIC_C0_COMPONENTS')
    return [dict(order=k, indices=tuple(rng.randrange(2) for _ in range(k+1))) for k in word]


def toy_identity(candidate='TOY_A', stratum='S2_SINGLE', replica=0, axis='cosine',
                 compiler='TOY_COMPILER_A', events='TOY_EVENT_A'):
    payload = ['SYNTHETIC_C0_ONLY', candidate, stratum, replica, axis, compiler, events]
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()


def toy_reserve(directory, key):
    """A mock O_EXCL reservation, explicitly unusable as a scientific grant."""
    path = Path(directory)/('MOCK_ONLY_' + key + '.json')
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as f:
        json.dump(dict(mock_only=True, scientific_authorization=False, nonrefundable=True), f)
    return path


def toy_admission(cpu_slots, memory_slots, pending_tasks, paired_parallel=True):
    """Synthetic admission tokens: each active task accounts for both axis slots."""
    axis_width = 2 if paired_parallel and cpu_slots >= 2 else 1
    return min(cpu_slots//axis_width, memory_slots, pending_tasks), axis_width


def toy_native_child(exit_code):
    """One synthetic process; native exit/reap checked by its creating parent."""
    payload = b'TOY_IMMUTABLE_COW_INPUT'
    before = hashlib.sha256(payload).hexdigest()
    child = os.fork()
    if child == 0:
        os._exit(exit_code if hashlib.sha256(payload).hexdigest() == before else 99)
    waited, status, _ = os.wait4(child, 0)
    return dict(mock_only=True, pid=waited, reaped=waited == child,
                exit_code=os.waitstatus_to_exitcode(status), attempts=1,
                parent_payload_unchanged=hashlib.sha256(payload).hexdigest() == before)


def refuse_scientific_execution(*args, **kwargs):
    raise PermissionError('C0_DESIGN_ONLY_NO_SCIENTIFIC_EXECUTION_OR_GRANT')
