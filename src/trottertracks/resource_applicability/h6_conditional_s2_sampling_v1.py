"""Conditional COST sampling only; original quantum event weights stay unchanged.

The pure helpers import no scientific libraries. Runtime construction is lazy
and is called only by the separately authorized pilot worker (or artificial
fixtures). No unconditional rejection loop and no forced single-order2 event.
"""
import hashlib
import math
import random

PHASE='conditional_s2_pilot_v1'
SEED_DOMAIN='H6_CONDITIONAL_S2_PILOT_V1_FRESH_COST'


def pilot_seed(cell_id,replica):
    if type(replica) is not int or not 0<=replica<8:
        raise ValueError('PILOT_REPLICA')
    return int.from_bytes(hashlib.sha256(f'{SEED_DOMAIN}/{cell_id}/{replica}'.encode()).digest()[:8],'big')


def order_seed(seed):
    if type(seed) is not int or seed<0:raise ValueError('PILOT_SEED')
    return int.from_bytes(hashlib.sha256(f'{SEED_DOMAIN}/ORDER_STREAM/{seed}'.encode()).digest()[:8],'big')


def component_step_seeds(seed,q):
    if type(seed) is not int or seed<0 or type(q) is not int or q<1:raise ValueError('COMPONENT_SEED_SCOPE')
    rng=random.Random(seed)
    return (seed,) if q==1 else tuple(rng.randrange(0,2**63) for _ in range(q))


def conditional_binomial(R,p2):
    if type(R) is not int or not 1<=R<=64 or type(p2) not in (int,float) or not math.isfinite(p2) or not 0<p2<1:
        raise ValueError('CONDITIONAL_BINOMIAL_DOMAIN')
    mass=-math.expm1(R*math.log1p(-p2))
    weights=[math.comb(R,m)*p2**m*(1-p2)**(R-m) for m in range(1,R+1)]
    total=math.fsum(weights)
    if not math.isclose(total,mass,rel_tol=2e-13,abs_tol=1e-300):
        raise ValueError('CONDITIONAL_BINOMIAL_NORMALIZATION')
    return dict(R=R,p2=p2,s2_probability=mass,counts=list(range(1,R+1)),
        conditional_probabilities=[w/total for w in weights],unconditional_probabilities=weights)


def draw_order_vector(R,p2,rng):
    law=conditional_binomial(R,p2)
    m=int(rng.choice(law['counts'],p=law['conditional_probabilities']))
    if not 1<=m<=R:raise ValueError('CONDITIONAL_COUNT_DRAW')
    positions=[int(x) for x in rng.choice(R,size=m,replace=False)]
    if len(positions)!=m or len(set(positions))!=m or any(not 0<=x<R for x in positions):
        raise ValueError('CONDITIONAL_POSITION_DRAW')
    chosen=set(positions)
    return tuple(2 if i in chosen else 0 for i in range(R))


def conditional_order_vector_probability(orders,p2):
    law=conditional_binomial(len(orders),p2)
    if not orders or any(type(x) is not int or x not in (0,2) for x in orders) or 2 not in orders:
        raise ValueError('S2_ORDER_VECTOR')
    m=orders.count(2)
    return p2**m*(1-p2)**(len(orders)-m)/law['s2_probability']


def events_from_orders(orders,components,distribution,component_rngs,r):
    """Reuse exact frozen _make_event; preserve signs, angles, weights and order."""
    from trotterlib.rte import _make_event,_validate_components
    if distribution.orders!=(0,2) or distribution.finite_taylor_order!=2:
        raise ValueError('PILOT_K2_DISTRIBUTION')
    if (type(r) is not int or r<1 or len(orders)!=len(component_rngs)*r
            or any(type(x) is not int or x not in (0,2) for x in orders) or 2 not in orders):
        raise ValueError('PILOT_WHOLE_TRAJECTORY_S2')
    total=_validate_components(components)
    probabilities=[c.probability/total for c in components]
    events=[]
    for i,k in enumerate(orders):
        indices=component_rngs[i//r].choice(len(components),size=k+1,p=probabilities)
        event=_make_event(indices,components,distribution,distribution.orders.index(k))
        events.append(event)
    return tuple(events)


def make_conditioned_steps(preparation,*,T,q,r,seed,ancilla_qubit,expected_distribution=None):
    """Same config and raw-concatenation step constructors; conditioned orders.

    Component streams retain the old master->step seed rule and PCG64 engine.
    A separate domain is used for the conditioned whole-trajectory order draw.
    Caller charges one trajectory and q occurrence calls BEFORE entering here.
    """
    import numpy as np
    from trotterlib.rte import make_rte_config
    from trotterlib.pr2_matched_accuracy_m1_execution import _explicit_cutoff_tolerance
    from trotterlib.df_rte_circuit import DFRTEEventSequenceCircuitRequest
    from trotterlib.df_partial_s2 import DFPartialS2StepRequest
    if (type(q) is not int or type(r) is not int or min(q,r)<1 or q*r>64
            or type(seed) is not int or seed<0 or not math.isfinite(T) or T<=0
            or preparation.is_deterministic_only):raise ValueError('CONDITIONAL_STEP_SCOPE')
    delta=T/q;tau=preparation.exact_rte_lambda_r*delta/r
    config,distribution=make_rte_config(preparation.rte_preparation.symbolic_tail,
        evolution_time=delta,rte_steps=r,truncation_tolerance=_explicit_cutoff_tolerance(tau,2),
        finite_taylor_order=2,seed=seed)
    if expected_distribution is not None and distribution.to_dict()!=expected_distribution:
        raise ValueError('PILOT_FROZEN_FINITE_LAW_CHANGED')
    p2=distribution.order_probabilities[distribution.orders.index(2)]
    orders=draw_order_vector(q*r,p2,np.random.Generator(np.random.PCG64(order_seed(seed))))
    step_seeds=component_step_seeds(seed,q)
    rngs=tuple(np.random.Generator(np.random.PCG64(s)) for s in step_seeds)
    events=events_from_orders(orders,preparation.rte_preparation.symbolic_tail.components,distribution,rngs,r)
    steps=[];tail=preparation.rte_preparation.symbolic_tail
    for i in range(q):
        ev=events[i*r:(i+1)*r]
        for event in ev:preparation.rte_preparation.resolve_event_basis_definitions(event)
        occurrence=DFRTEEventSequenceCircuitRequest(events=ev,
            component_specs=preparation.rte_preparation.component_specs,controlled=True,
            ancilla_qubit=ancilla_qubit,cancel_adjacent_equal_bases=True,
            tail_id=tail.tail_id,tail_hash=tail.tail_hash,occurrence_rte_steps=r)
        steps.append(DFPartialS2StepRequest(preparation=preparation,step_time=delta,rte_config=config,
            rte_distribution=distribution,rte_occurrence=occurrence,controlled=True,
            ancilla_qubit=ancilla_qubit,seed=step_seeds[i],rte_basis_plan=None))
    mass=conditional_binomial(q*r,p2)['s2_probability']
    probability=math.prod(e.event_probability for e in events)
    receipt=dict(schema='h6_conditional_s2_sampling_receipt_v1',stratum='s2',whole_trajectory=True,
        R=q*r,q=q,r=r,K=2,master_seed=seed,order_stream_seed=order_seed(seed),step_seeds=list(step_seeds),
        observed_orders=list(orders),order2_count=orders.count(2),s2_probability=mass,
        conditional_order_vector_probability=conditional_order_vector_probability(orders,p2),
        original_event_sequence_probability=probability,conditional_event_sequence_probability=probability/mass,
        original_finite_distribution=distribution.to_dict(),quantum_event_weights_changed=False,
        normalization_changed=False,component_selection_probability_changed=False,
        construction_policy='raw_concatenation',rte_basis_plans=[None]*q,
        conditioned_cost_sampling_only=True,measurement_shots_sampled=False)
    return tuple(steps),receipt
