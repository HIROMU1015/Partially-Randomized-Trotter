"""H8 COST-only conditional orders; unchanged pinned K4 event construction.

For R independent draws conditioned on no 4 and at least one 2, draw the
positive binomial count, then a uniform subset. Component choices remain IID
under their original probabilities; _make_event retains all original weights.
No molecular imports or sampling take place on module import.
"""
import hashlib
import math
import random

DOMAIN = 'H8_CONDITIONAL_S2_NATIVE_COST_PILOT_20261017_V1'


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def seed(candidate, replica):
    require(type(replica) is int and 0 <= replica < 8, 'REPLICA')
    return int.from_bytes(hashlib.sha256(f'{DOMAIN}/{candidate}/{replica}'.encode()).digest()[:8], 'big')


def order_seed(master):
    require(type(master) is int and master >= 0, 'SEED')
    return int.from_bytes(hashlib.sha256(f'{DOMAIN}/ORDERS/{master}'.encode()).digest()[:8], 'big')


def step_seeds(master, q):
    require(type(master) is int and master >= 0 and type(q) is int and q >= 1, 'STEP_SEEDS')
    rng = random.Random(master)
    return (master,) if q == 1 else tuple(rng.randrange(0, 2**63) for _ in range(q))


def conditional_law(R, probabilities):
    require(type(R) is int and 1 <= R <= 64 and len(probabilities) == 3, 'LAW_DOMAIN')
    p0, p2, p4 = probabilities
    require(all(type(p) in (int, float) and math.isfinite(p) and 0 <= p <= 1 for p in probabilities)
            and p0 > 0 and p2 > 0 and math.isclose(math.fsum(probabilities), 1., abs_tol=2e-15, rel_tol=0), 'LAW_PROBABILITY')
    s = p0 + p2
    require(0 < s <= 1, 'NO4_MASS')
    t = p2 / s
    require(0 < t < 1, 'CONDITIONAL_TWO_PROBABILITY')
    log_no4 = R * math.log(s)
    positive_mass = -math.expm1(R * math.log1p(-t))
    pi2 = math.exp(log_no4) * positive_mass
    require(pi2 > 0 and math.isfinite(pi2), 'S2_MASS_UNREPRESENTABLE')
    logs = [math.log(math.comb(R, m)) + m * math.log(t) + (R-m) * math.log1p(-t)
            for m in range(1, R+1)]
    largest = max(logs)
    weights = [math.exp(v-largest) for v in logs]
    total = math.fsum(weights)
    return dict(R=R, original_probabilities=list(probabilities), conditional_p2=t,
                pi0=p0**R, pi2=pi2, pi4=-math.expm1(log_no4), positive_binomial_mass=positive_mass,
                counts=list(range(1, R+1)), count_probabilities=[v/total for v in weights])


def draw_orders(law, rng):
    R = law['R']
    m = int(rng.choice(law['counts'], p=law['count_probabilities']))
    positions = {int(x) for x in rng.choice(R, size=m, replace=False)}
    require(1 <= m <= R and len(positions) == m and all(0 <= i < R for i in positions), 'S2_COUNT_POSITIONS')
    return tuple(2 if i in positions else 0 for i in range(R))


def validate_orders(orders, q, r):
    require(type(q) is int and q in (4, 8) and r == 4 and len(orders) == q*r
            and all(type(k) is int and k in (0, 2) for k in orders) and 2 in orders, 'WHOLE_TRAJECTORY_S2')


def events_from_orders(orders, components, distribution, rngs, r):
    from trotterlib.rte import _make_event, _validate_components
    require(distribution.orders == (0, 2, 4) and distribution.finite_taylor_order == 4, 'ORIGINAL_K4_LAW')
    validate_orders(orders, len(rngs), r)
    total = _validate_components(components)
    probabilities = [c.probability/total for c in components]
    return tuple(_make_event(rngs[i//r].choice(len(components), size=k+1, p=probabilities),
                             components, distribution, distribution.orders.index(k))
                 for i, k in enumerate(orders))


def make_steps(preparation, *, q, master, expected):
    import numpy as np
    from trotterlib.rte import make_rte_config
    from trotterlib.pr2_matched_accuracy_m1_execution import _explicit_cutoff_tolerance
    from trotterlib.df_rte_circuit import DFRTEEventSequenceCircuitRequest
    from trotterlib.df_partial_s2 import DFPartialS2StepRequest
    require(q in (4, 8) and not preparation.is_deterministic_only, 'FIXED_CANDIDATE')
    delta = .8/q
    tau = preparation.exact_rte_lambda_r*delta/4
    config, distribution = make_rte_config(preparation.rte_preparation.symbolic_tail,
        evolution_time=delta, rte_steps=4, truncation_tolerance=_explicit_cutoff_tolerance(tau, 4),
        finite_taylor_order=4, seed=master)
    require(distribution.to_dict() == expected, 'FROZEN_FINITE_LAW_CHANGED')
    law = conditional_law(q*4, distribution.order_probabilities)
    orders = draw_orders(law, np.random.Generator(np.random.PCG64(order_seed(master))))
    seeds = step_seeds(master, q)
    events = events_from_orders(orders, preparation.rte_preparation.symbolic_tail.components, distribution,
        tuple(np.random.Generator(np.random.PCG64(s)) for s in seeds), 4)
    steps = []
    tail = preparation.rte_preparation.symbolic_tail
    for i in range(q):
        ev = events[i*4:(i+1)*4]
        for event in ev:
            preparation.rte_preparation.resolve_event_basis_definitions(event)
        occurrence = DFRTEEventSequenceCircuitRequest(events=ev,
            component_specs=preparation.rte_preparation.component_specs, controlled=True, ancilla_qubit=16,
            cancel_adjacent_equal_bases=True, tail_id=tail.tail_id, tail_hash=tail.tail_hash, occurrence_rte_steps=4)
        steps.append(DFPartialS2StepRequest(preparation=preparation, step_time=delta, rte_config=config,
            rte_distribution=distribution, rte_occurrence=occurrence, controlled=True,
            ancilla_qubit=16, seed=seeds[i], rte_basis_plan=None))
    log_probability = math.fsum(math.log(e.event_probability) for e in events)
    receipt = dict(schema='H8_CONDITIONAL_S2_SAMPLE_V1', q=q, r=4, K=4, R=q*4, T=.8, delta=delta,
        stratum='s2', seed=master, order_seed=order_seed(master), step_seeds=list(seeds), orders=list(orders),
        order2_count=orders.count(2), law=law, original_finite_law=expected,
        original_event_log_probability=log_probability,
        conditional_event_log_probability=log_probability-math.log(law['pi2']),
        component_law_changed=False, quantum_weights_changed=False, original_normalization_changed=False,
        construction_policy='raw_concatenation', rte_basis_plans=[None]*q)
    return tuple(steps), receipt
