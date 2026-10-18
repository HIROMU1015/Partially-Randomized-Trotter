"""Condition original H6 event counts, preserving the frozen quantum law."""
import importlib.util
from pathlib import Path
import sys
import h6_dba_contract as c

_spec=importlib.util.spec_from_file_location('_dba_existing_counts',
    c.ROOT/'src/trotterlib/h8_rare_tail_selection.py')
counts=importlib.util.module_from_spec(_spec);sys.modules[_spec.name]=counts;_spec.loader.exec_module(counts)


def count_law(R, distribution, stratum):
    probs=dict(zip(distribution.orders,distribution.order_probabilities,strict=True))
    law=counts.conditional_counts(R,tuple(probs.get(k,0.) for k in (0,2,4)),stratum)
    counts.require_representable(law)
    return law


def draw(law, rng):
    index=int(rng.choice(len(law.counts),p=law.probabilities))
    n2,n4=law.counts[index]
    positions=rng.choice(law.R,size=n2+n4,replace=False)
    out=[0]*law.R
    for i in positions[:n2]:out[int(i)]=2
    for i in positions[n2:]:out[int(i)]=4
    c.require(counts.in_stratum(out.count(2),out.count(4),law.stratum),'CONDITIONAL_MEMBERSHIP')
    return tuple(out)


def steps(preparation, cell, stratum, seed):
    import numpy as np
    from trotterlib.rte import make_rte_config,_make_event,_validate_components
    from trotterlib.pr2_matched_accuracy_m1_execution import _explicit_cutoff_tolerance
    from trotterlib.df_rte_circuit import DFRTEEventSequenceCircuitRequest
    from trotterlib.df_partial_s2 import DFPartialS2StepRequest
    q,r,K=cell['q'],cell['r'],cell['K'];delta=.8/q;tau=preparation.exact_rte_lambda_r*delta/r
    tail=preparation.rte_preparation.symbolic_tail
    config,distribution=make_rte_config(tail,evolution_time=delta,rte_steps=r,
        truncation_tolerance=_explicit_cutoff_tolerance(tau,K),finite_taylor_order=K,seed=seed)
    law=count_law(q*r,distribution,stratum)
    order_seed=int(c.digest(['H6_DBA_ORDER',seed])[:16],16)
    orders=draw(law,np.random.Generator(np.random.PCG64(order_seed)))
    stream_seeds=[int(c.digest(['H6_DBA_COMPONENT',seed,i])[:16],16) for i in range(q)]
    rngs=[np.random.Generator(np.random.PCG64(s)) for s in stream_seeds]
    components=tail.components;total=_validate_components(components)
    probs=[x.probability/total for x in components];events=[]
    for i,k in enumerate(orders):
        indices=rngs[i//r].choice(len(components),size=k+1,p=probs)
        events.append(_make_event(indices,components,distribution,distribution.orders.index(k)))
    out=[]
    for i in range(q):
        ev=tuple(events[i*r:(i+1)*r])
        for event in ev:preparation.rte_preparation.resolve_event_basis_definitions(event)
        ancilla=preparation.num_system_qubits
        occurrence=DFRTEEventSequenceCircuitRequest(events=ev,
            component_specs=preparation.rte_preparation.component_specs,controlled=True,ancilla_qubit=ancilla,
            cancel_adjacent_equal_bases=True,tail_id=tail.tail_id,tail_hash=tail.tail_hash,occurrence_rte_steps=r)
        out.append(DFPartialS2StepRequest(preparation=preparation,step_time=delta,rte_config=config,
            rte_distribution=distribution,rte_occurrence=occurrence,controlled=True,ancilla_qubit=ancilla,
            seed=stream_seeds[i],rte_basis_plan=None))
    receipt=dict(cell=cell,stratum=stratum,master_seed=seed,order_seed=order_seed,
        component_seeds=stream_seeds,orders=list(orders),n2=orders.count(2),n4=orders.count(4),
        original_distribution=distribution.to_dict(),log_stratum_mass=law.log_stratum_mass,
        events=[e.to_dict() for e in events],quantum_weights_changed=False,normalization_changed=False,
        count_distribution='original multinomial conditional on registered stratum',
        position_distribution='uniform subset',RNG='NumPy PCG64',sampling_is_not_quantum_shots=True)
    receipt['event_digest']=c.digest(receipt['events'])
    return tuple(out),receipt
