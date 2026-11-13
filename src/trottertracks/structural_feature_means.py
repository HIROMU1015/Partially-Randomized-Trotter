"""Supplied finite-K2 feature expectations; artificial evaluation in this study.

No physical reader, RNG, circuit, price fit or scientific runner. Counts describe
nonidentity tail applications, never full compiled RZ. The basis kernel retains
the active tail basis; it does not certify pending-Z or compiler-state closure.
"""
from fractions import Fraction

from .finite_k_sampler_law import _probabilities
from .mean_compiled_price import MeanKernel


def k2_linear_counts(component_weights, support_sizes, order_weights, *, q, R_outer):
    """Exact ratios of supplied weights, not analytic-input/libm error bounds.

    Assumes at each event: K2 order law and the same conditional component law
    for rotation/products. Cross-event independence is unnecessary for these
    first moments. All components must be nonidentity, with support 1 or 2.
    """
    if type(q) is not int or q < 1 or type(R_outer) is not int or R_outer < 1:
        raise ValueError('Positive integer q and R_outer required')
    p = _probabilities(component_weights)
    sizes = tuple(support_sizes)
    o = _probabilities(order_weights)
    if len(sizes) != len(p) or any(type(s) is not int or s not in (1, 2) for s in sizes) or len(o) != 2:
        raise ValueError('Explicit nonidentity Z/ZZ support and K2 order weights required')
    n = q * R_outer
    z = sum((w for w, s in zip(p, sizes) if s == 1), Fraction())
    zz = 1 - z
    return dict(events=Fraction(n), event_order_0=n * o[0], event_order_2=n * o[1],
                rotation_Z=n * z, rotation_ZZ=n * zz, product_applications=2 * n * o[1],
                product_CZ=2 * n * o[1] * (z + 2 * zz),
                identity_applications=Fraction(), all_applications=n * (1 + 2 * o[1]))


def k2_basis_counts(component_weights, basis_keys, order_weights, *, q, R_outer):
    """Exact active-basis group/share counts for IID nonidentity applications.

    Each tail starts with active=None as in Accounting.tail. Components must be
    IID conditional on order, with basis independent of the random word length;
    adjacent events must have the same product law. q tails need not be mutually
    independent for these means. Keys must retain source basis ID/hash identity.
    This counts active-basis branches, NOT emitted/cancelled native transforms.
    """
    if type(q) is not int or q < 1 or type(R_outer) is not int or R_outer < 1:
        raise ValueError('Positive integer q and R_outer required')
    p = _probabilities(component_weights)
    keys = tuple(basis_keys)
    o = _probabilities(order_weights)
    if len(keys) != len(p) or len(o) != 2 or any(k is None for k in keys):
        raise ValueError('Explicit basis keys and K2 order law required')
    rho = {}
    for key, weight in zip(keys, p):
        rho[key] = rho.get(key, Fraction()) + weight
    same = sum(v * v for v in rho.values())
    applications_per_tail = R_outer * (1 + 2 * o[1])
    return dict(tail_basis_groups=q * (1 + (applications_per_tail - 1) * (1 - same)),
                tail_shared_basis_pairs=q * (applications_per_tail - 1) * same)


def k2_basis_kernel(component_probabilities, basis_keys, order_probabilities, *, feature):
    """Minimal shared-basis reward kernel, reusing MeanKernel composition.

    For inhomogeneous pieces/laws use separate kernels and retain boundary state.
    None is the initial inactive basis. This is a float64 supplied-model kernel;
    no K0 parity certificate or physical numerical enclosure is generated here.
    The caller must justify IID conditional component/order laws. Physical use
    needs a separately fixed SOURCE, input scope and numerical contract.
    """
    keys = tuple(basis_keys)
    probabilities = tuple(component_probabilities)
    orders = tuple(order_probabilities)
    if (feature not in ('tail_basis_groups', 'tail_shared_basis_pairs') or not keys
            or len(keys) != len(probabilities) or any(k is None for k in keys)
            or len(orders) != 2):
        raise ValueError('Supported tail feature, nonidentity basis keys and K2 law required')
    states = (None, *dict.fromkeys(keys))
    def step(active, event):
        changed = active != keys[event]
        return keys[event], int(changed if feature == 'tail_basis_groups' else not changed)
    one = MeanKernel.from_events(states, range(len(keys)), probabilities, step)
    # Application order is product1, product2, rotation in order2. Same IID
    # basis marginal for each role allows this factorization without tuples.
    three = one.repeat(3)
    # Use the existing validator for order normalization; never repair inputs.
    MeanKernel.from_events((0,), (0, 2), orders, lambda s, e: (0, 0))
    return MeanKernel(states, orders[0] * one.transition + orders[1] * three.transition,
                      orders[0] * one.reward + orders[1] * three.reward)
