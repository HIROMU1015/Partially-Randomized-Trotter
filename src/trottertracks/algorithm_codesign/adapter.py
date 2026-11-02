"""Signed F adapter and CPU matrix semantics, without circuit construction."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
import mpmath as mp
import numpy as np

from .domain import DPS, ExactTime, Point


@dataclass(frozen=True)
class Factor:
    generator: str
    time: ExactTime


def fuse(factors):
    stack = []
    for f in factors:
        if f.time.zero:
            continue
        if stack and stack[-1].generator == f.generator:
            f = Factor(f.generator, stack.pop().time+f.time)
        if not f.time.zero:
            stack.append(f)
    return tuple(stack)


def stage_list(point: Point, deterministic_count: int, q=1, *, simplify=True):
    if deterministic_count < 0 or q < 1:
        raise ValueError("Invalid stage count")
    factors = []
    for _ in range(q):
        for w in point.weights:
            half = w*Fraction(1, 2)
            factors.extend(Factor(f"D{i}", half) for i in range(deterministic_count))
            factors.append(Factor("R", w))
            factors.extend(Factor(f"D{i}", half) for i in reversed(range(deterministic_count)))
    return fuse(factors) if simplify else tuple(factors)


def stationary_template(point, deterministic_count, q):
    one = stage_list(point, deterministic_count)
    full = stage_list(point, deterministic_count, q)
    one_tail = [f.time for f in one if f.generator == "R"]
    full_tail = [f.time for f in full if f.generator == "R"]
    if full_tail != one_tail*q:
        raise ValueError("Tail fusion crosses an outer-step boundary; contract review required")
    return one, full


def allocate(point, tail_times, budget):
    if budget < len(tail_times):
        return None
    if not tail_times:
        return ()
    with mp.workdps(DPS):
        values = [abs(t.value(point.basis)) for t in tail_times]
        total = mp.fsum(values)
        x = [(budget-len(values))*v/total for v in values]
        floors = [int(mp.floor(v)) for v in x]
        allocation = [1+v for v in floors]
        order = sorted(range(len(x)), key=lambda i: (-(x[i]-floors[i]), i))
        for i in order[:budget-sum(allocation)]:
            allocation[i] += 1
    return tuple(allocation)


def tail_statistics(times, allocation, lambda_r, q, K):
    if K not in (2, 4):
        raise ValueError("BF-1 permits K=2 or trigger-eligible K=4")
    log_b, log_remainder, work, leading = 0., 0., 0., 0.
    for time, r in zip(times, allocation):
        tau = lambda_r*time/r
        absolute = abs(tau)
        a = [absolute**k/math.factorial(k)*math.sqrt(1+tau*tau/(k+1)**2)
             for k in range(0, K+1, 2)]
        b = math.fsum(a)
        log_b += r*math.log(b)
        work += r*math.fsum(v*(k+1) for v, k in zip(a, range(0, K+1, 2)))/b
        leading += r*tau*tau
        remainder = math.exp(absolute)*absolute**(K+2)/math.factorial(K+2)
        log_remainder += r*math.log1p(remainder)
    return dict(log_b=q*log_b, log_b_leading=q*leading, random_actions=q*work,
                tail_bound=math.expm1(q*log_remainder))


def polynomial(matrix, time, K):
    """Synthetic independent dense polynomial, K is EVEN event order."""
    x = -1j*time*matrix
    term = np.eye(len(matrix), dtype=complex)
    value = term.copy()
    for n in range(1, K+2):
        term = term@x/n
        value += term
    return value


def dense_operator(point, generators, total_time, q, *, allocation=None, K=2, construction="F", scalar=0.):
    """Small-matrix semantic oracle used only by synthetic tests."""
    from scipy.linalg import expm
    factors = stage_list(point, len(generators)-1, q, simplify=construction == "F")
    result = np.eye(len(generators["R"]), dtype=complex)
    tail_index = 0
    for f in factors:
        with mp.workdps(DPS):
            t = float(f.time.value(point.basis))*total_time/q
        if f.generator == "R" and allocation is not None:
            r = allocation[tail_index % len(allocation)]
            operator = np.linalg.matrix_power(polynomial(generators["R"], t/r, K), r)
            tail_index += 1
        else:
            operator = expm(-1j*t*generators[f.generator])
        result = operator@result
    return np.exp(-1j*scalar*total_time)*result
