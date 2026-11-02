"""Exact literal gate inverses and streaming pair kernel, for a future D2 run.

No provider, matrix, circuit compiler, synthesis, or sampler import. The kernel
may be tested off-domain; registered use requires separate one-shot authority.
"""
from fractions import Fraction as F
from math import lcm

SELF = {'H', 'X', 'Y', 'Z', 'CX', 'CZ'}
ADJOINT = {'T': 't', 't': 'T', 'S': 's', 's': 'S', 'W': 'w', 'w': 'W'}


def gate(g):
    g = tuple(g)
    if not g:
        raise ValueError('empty gate')
    name = g[0]
    arity = 4 if name == 'R' else 3 if name in ('CX', 'CZ') else 1 if name in ('W', 'w') else 2
    if name not in SELF | set(ADJOINT) | {'R'} or len(g) != arity:
        raise ValueError('unsupported literal gate')
    wires = g[1:3] if name in ('CX', 'CZ') else () if name in ('W', 'w') else g[1:2]
    if any(type(w) is not int or w < 0 for w in wires) or len(set(wires)) != len(wires):
        raise ValueError('invalid wire')
    if name == 'R' and (type(g[3]) is not int or g[3] not in (-1, 1) or F(g[2]) <= 0):
        raise ValueError('invalid saved signed rotation')
    return g


def inverse(g):
    g = gate(g)
    if g[0] == 'R':
        return g[:3] + (-g[3],)
    return (ADJOINT.get(g[0], g[0]),) + g[1:]


def simplify(gates):
    stack = []
    for raw in gates:
        g = gate(raw)
        if stack and inverse(stack[-1]) == g:
            stack.pop()
        else:
            stack.append(g)
    return stack


def cancellation_depth(left, right):
    """Boundary-only comparison; inputs must already be internally reduced."""
    k = 0
    while k < min(len(left), len(right)) and inverse(left[-1-k]) == tuple(right[k]):
        k += 1
    return k


def native_price(gates, cache):
    out = {'T': 0, 'CX': 0, '1Q': 0}
    for raw in gates:
        g = gate(raw)
        if g[0] == 'R':
            row = cache[g[2]]  # Missing saved key is fatal, never free.
            out['T'] += row['T_count']
            out['1Q'] += row['one_qubit_count']
        elif g[0] == 'CX':
            out['CX'] += 1
        elif g[0] == 'CZ':
            out['CX'] += 1
            out['1Q'] += 2
        else:
            out['T'] += g[0] in ('T', 't')
            out['1Q'] += g[0] not in ('W', 'w')
    return out


def aggregate_pairs(bindings, cache, check=lambda: None, max_pairs=300000,
                    denominator_bits=16384):
    """Enumerate costs once, streaming integer probability masses.

    No full concatenated circuits or per-pair JSONs are retained. Denominator
    lifting avoids repeated Fraction reduction in the inner loop. The exact
    same inverse pass is applied to all arms; no commutation or angle fusion.
    """
    irs, probabilities = [], []
    for b in bindings:
        ir = [gate(g) for g in b['native_ir']]
        if simplify(ir) != ir:
            raise ArithmeticError('saved step is not internally reduced')
        q = F(b['event']['proposal'])
        if q <= 0:
            raise ArithmeticError('positive support required')
        irs.append(ir); probabilities.append(q)
    if len(irs) ** 2 > max_pairs:
        raise RuntimeError('pair cap')
    D = 1
    for q in probabilities:
        D = lcm(D, q.denominator)
        if D.bit_length() > denominator_bits:
            raise RuntimeError('denominator bit cap')
    masses = [q.numerator * (D // q.denominator) for q in probabilities]
    Z = sum(probabilities, F(0))
    if Z > 1:
        raise ArithmeticError('acceptance exceeds one')
    prices, suffix_prices = [], []
    for ir in irs:
        prices.append(native_price(ir, cache))
        suffix = [dict(T=0, CX=0, **{'1Q': 0})]
        for g in reversed(ir):
            c = native_price([g], cache)
            suffix.append({k: suffix[-1][k] + c[k] for k in c})
        suffix_prices.append(suffix)
    saving = {k: 0 for k in ('T', 'CX', '1Q')}
    depth_mass = {}; count = 0; worst = {k: 0 for k in saving}
    for i, left in enumerate(irs):
        check()
        for j, right in enumerate(irs):
            if count % 256 == 0: check()
            n = cancellation_depth(left, right)
            mass = masses[i] * masses[j]
            depth_mass[n] = depth_mass.get(n, 0) + mass
            for k in saving:
                removed = 2 * suffix_prices[i][n][k]
                saving[k] += mass * removed
                worst[k] = max(worst[k], prices[i][k] + prices[j][k] - removed)
            count += 1
    denominator = D * D
    base = {k: 2 * Z * sum((q*p[k] for q, p in zip(probabilities, prices)), F(0)) for k in saving}
    reduction = {k: F(v, denominator) for k, v in saving.items()}
    assert sum(depth_mass.values()) == sum(masses) ** 2
    return {'positive_pairs': count, 'acceptance_squared': Z*Z,
            'additive_per_whole_trial': base,
            'cancellation_saving_per_whole_trial': reduction,
            'after_per_whole_trial': {k: base[k]-reduction[k] for k in base},
            'worst_nonzero_pair': worst,
            'depth_probability_unconditional': {str(k): F(v, denominator) for k, v in sorted(depth_mass.items())},
            'denominator_bits': D.bit_length(), 'pair_circuits_retained': 0,
            'probability_law_N_bias_unchanged': True}
