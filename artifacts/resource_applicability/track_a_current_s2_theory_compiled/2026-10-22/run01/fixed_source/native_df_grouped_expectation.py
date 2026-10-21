"""Exact basis-grouped evaluation of the existing finite-RTE cost DP.

The component law, identity barriers, explicit lowering, phase bounds and closed
occurrence scope are unchanged. Only linear sums over equal basis states are
factored out; all probability/cost arithmetic remains Fraction arithmetic.
"""
from fractions import Fraction
import math


def expected_rte_grouped(m, components, orders, r, tau, ancilla, model='A',
                         max_transitions=1_000_000):
    components = tuple(components)
    m._validate_law(components, orders, ancilla)
    if (type(r) is not int or r <= 0 or model not in ('A','B') or
            not math.isfinite(tau) or type(max_transitions) is not int or max_transitions <= 0):
        raise ValueError('Invalid grouped expectation domain')
    bases = {}
    probabilities = {}
    for c in components:
        key = None if c.basis is None else (c.basis.basis_id,c.basis.source_hash)
        if key is not None:
            bases[key] = c.basis
        probabilities[key] = probabilities.get(key,Fraction(0)) + c.probability
    forward = {k:m.count(b.circuit()) for k,b in bases.items()}
    inverse = {k:m.count(b.circuit(True)) for k,b in bases.items()}
    center = {}
    cache = {}
    for k in orders:
        theta = math.atan(tau/(k+1))
        for role in ('product','rotation'):
            for c in components:
                key = None if c.basis is None else (c.basis.basis_id,c.basis.source_hash)
                # Cache central lowering only; Gaussian identity is never inferred
                # from support, approximate parameters or component names.
                ck = (c.basis is None,c.support,c.sign,role,theta.hex())
                if ck not in cache:
                    cache[ck] = m.count(m.application(c,role,theta,ancilla))
                lookup = k,role,key
                center[lookup] = center.get(lookup,m.Cost()) + cache[ck]*c.probability
    calls = 0
    def draw(states,role,k):
        nonlocal calls
        result = {}
        for active,(mass,weighted) in states.items():
            for key,p in probabilities.items():
                if p == 0:continue
                calls += 1
                if calls > max_transitions:
                    raise RuntimeError('DP transition cap; no approximation fallback')
                transition = m.Cost()
                if model == 'A':
                    if key is not None:transition = inverse[key]+forward[key]
                    end = None
                else:
                    if active != key:
                        if active is not None:transition += forward[active]
                        if key is not None:transition += inverse[key]
                    end = key
                op,ow = result.get(end,(Fraction(0),m.Cost()))
                result[end] = op+mass*p,ow+weighted*p+(transition*p+center[k,role,key])*mass
        return result
    total = m.Cost()
    if model == 'A':
        for k,p in orders.items():
            mean = m.Cost()
            for key,cp in probabilities.items():
                pair = inverse[key]+forward[key] if key is not None else m.Cost()
                mean += pair*(cp*(k+1))+center[k,'product',key]*k+center[k,'rotation',key]
            total += mean*(p*r)
    else:
        states = {None:(Fraction(1),m.Cost())}
        for _ in range(r):
            combined = {}
            for k,p in sorted(orders.items()):
                if p == 0:continue
                conditional = states
                for _ in range(k):conditional = draw(conditional,'product',k)
                conditional = draw(conditional,'rotation',k)
                for key,(mass,cost) in conditional.items():
                    op,ow = combined.get(key,(Fraction(0),m.Cost()))
                    combined[key] = op+mass*p,ow+cost*p
            states = combined
        if sum(v[0] for v in states.values()) != 1:
            raise ArithmeticError('Grouped probability mass loss')
        for key,(mass,cost) in states.items():
            total += cost+(forward[key]*mass if key is not None else m.Cost())
    lo,hi = Fraction(0),Fraction(1)
    if tau == 0 or not any(c.basis is None and c.probability for c in components):
        positive = sum(c.probability for c in components if c.sign == 1)
        zero = sum(p*positive**k for k,p in orders.items() if not (k//2)%2)
        lo = hi = 1-zero**r
    phase = m.count(m.Circuit((m.Gate('p',(ancilla,),1.0),)))
    return m.ExpectedCost(total+phase*lo,total+phase*hi,lo,hi,calls)
