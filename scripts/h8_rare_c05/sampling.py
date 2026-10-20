"""Exact rational conditioning candidate; no production inputs at import."""
from fractions import Fraction
from functools import reduce,lru_cache
from math import comb, gcd, lcm
import hashlib
import json
from bisect import bisect_right

POLICY='EXACT_INTEGER_CONDITIONING_OF_STORED_BINARY64_V1_PROPOSED'
STRATA=('S2_SINGLE','S2_MULTI','S4_PRESENT')
CELLS={'B2_p8_q2_2nd_r4_K4':(8,2,4,4),'B2_p7_q16_2nd_r4_K4':(7,16,4,4)}


def require(ok,reason):
    if not ok: raise ValueError(reason)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def weights(values):
    """Fractions are exact; floats are interpreted as their exact binary64 value."""
    fs=[Fraction(x) for x in values]
    require(bool(fs) and all(x>=0 for x in fs) and sum(fs)>0,'PROBABILITY_DOMAIN')
    denominator=lcm(*(x.denominator for x in fs))
    ints=[x.numerator*(denominator//x.denominator) for x in fs]
    common=reduce(gcd,ints)
    return tuple(x//common for x in ints)


def in_stratum(n2,n4,s):
    if s=='S2_SINGLE': return n2==1 and n4==0
    if s=='S2_MULTI': return n2>=2 and n4==0
    if s=='S4_PRESENT': return n4>=1
    raise ValueError('UNKNOWN_STRATUM')


@lru_cache(maxsize=None)
def count_weights(R,p,s):
    require(type(R) is int and R>0,'EVENT_COUNT')
    require(len(p)==3 and s in STRATA,'COUNT_LAW_SCOPE')
    w0,w2,w4=weights(p);out=[]
    for n4 in range(R+1):
        for n2 in range(R-n4+1):
            if in_stratum(n2,n4,s):
                n0=R-n2-n4
                weight=comb(R,n4)*comb(R-n4,n2)*w0**n0*w2**n2*w4**n4
                if weight:out.append(((n2,n4),weight))
    require(bool(out),'EMPTY_CONDITIONAL_SUPPORT')
    return tuple(out)


def below(n,rng):
    require(type(n) is int and n>0,'INTEGER_CATEGORY_TOTAL')
    bits=(n-1).bit_length()
    while True:
        value=rng.getrandbits(bits) if bits else 0
        if value<n:return value


class IntegerCategorical:
    def __init__(self,items):
        require(bool(items) and all(type(w) is int and w>0 for _,w in items),'CATEGORY_WEIGHTS')
        total=0;self.values=[];self.cumulative=[]
        for value,w in items:
            total+=w;self.values.append(value);self.cumulative.append(total)
        self.total=total
    def draw(self,rng):return self.values[bisect_right(self.cumulative,below(self.total,rng))]


def choose(items,rng):return IntegerCategorical(items).draw(rng)


@lru_cache(maxsize=None)
def count_category(R,p,s):return IntegerCategorical(count_weights(R,p,s))


def order_word(R,p,s,rng):
    n2,n4=count_category(R,p,s).draw(rng)
    # Partial Fisher-Yates: all labelled position permutations equally likely.
    positions=list(range(R))
    for i in range(n2+n4):
        j=i+below(R-i,rng);positions[i],positions[j]=positions[j],positions[i]
    out=[0]*R
    for i in positions[:n2]:out[i]=2
    for i in positions[n2:n2+n4]:out[i]=4
    return tuple(out)


class PCG64Bits:
    """Provider supplies an already authorized numpy PCG64 bit_generator."""
    def __init__(self,bit_generator): self.generator=bit_generator
    def getrandbits(self,k):
        require(type(k) is int and k>=0,'BIT_COUNT')
        out=0;offset=0
        while offset<k:
            take=min(64,k-offset)
            out|=(int(self.generator.random_raw())&((1<<take)-1))<<offset
            offset+=take
        return out


def register_streams(tasks,source,binding,epoch,occupied=()):
    """Registry is private. No dependence on execution order, PID or current time."""
    used=set(occupied);registry={}
    for task in sorted(tasks,key=lambda t:t['id']):
        require(task['id'] not in registry,'DUPLICATE_TASK')
        cell=CELLS.get(task['candidate']);require(cell is not None,'FIXED_CANDIDATE')
        require(task['stratum'] in STRATA and task['phase'] in ('variance_load_pilot','production'),'TASK_DOMAIN')
        require(type(task['replica']) is int and task['replica']>=0,'REPLICA_INDEX')
        streams={}
        for label in ['order']+['component_'+str(i) for i in range(cell[1])]:
            key=['H8_RARE_C1_NEW_COHORT',source,binding,epoch,task['candidate'],task['stratum'],task['phase'],task['replica'],label]
            seed=int(digest(key),16);require(seed not in used,'SEED_COLLISION_OR_OLD_COHORT')
            used.add(seed);streams[label]=seed
        registry[task['id']]=streams
    return registry


def requests(preparation,task,law,config,streams,api):
    """All physics remains in pinned provider APIs. Tests supply symbolic mocks."""
    prefix,q,r,K=CELLS[task['candidate']]
    require((task['q'],task['r'],task['K'])==(q,r,K),'Q_R_K_SCOPE')
    require(tuple(law.orders)==(0,2,4),'FINITE_K4_SUPPORT')
    require(all(x>0 for x in law.order_probabilities),'POSITIVE_K4_SUPPORT_ALREADY_LOST')
    p=tuple(law.order_probabilities)
    orders=order_word(q*r,p,task['stratum'],api.rng(streams['order']))
    components=preparation.rte_preparation.symbolic_tail.components
    if getattr(api,'prepared_components',None) is not None:
        saved_components,categorical=api.prepared_components
        require(saved_components is components,'PARENT_COMPONENT_IDENTITY_CHANGED')
    else:
        total=api.validate_components(components)
        component_weights=weights([c.probability/total for c in components])
        categories=tuple((i,w) for i,w in enumerate(component_weights) if w)
        require(len(categories)==len(components),'COMPONENT_SUPPORT_ALREADY_LOST')
        categorical=IntegerCategorical(categories)
    rngs=[api.rng(streams['component_'+str(i)]) for i in range(q)]
    events=[]
    for i,k in enumerate(orders):
        indices=tuple(categorical.draw(rngs[i//r]) for _ in range(k+1))
        event=api.make_event(indices,components,law,law.orders.index(k))
        preparation.rte_preparation.resolve_event_basis_definitions(event)
        events.append(event)
    out=[]
    for i in range(q):
        occurrence=api.sequence(events=tuple(events[i*r:(i+1)*r]),
            component_specs=preparation.rte_preparation.component_specs,controlled=True,
            ancilla_qubit=preparation.num_system_qubits,occurrence_rte_steps=r,
            preserve_event_order=True,cancel_adjacent_equal_bases=True,
            control_diagonal_only=True,identity_as_relative_ancilla_phase=True,
            tail_id=preparation.rte_preparation.symbolic_tail.tail_id,
            tail_hash=preparation.rte_preparation.symbolic_tail.tail_hash)
        out.append(api.step(preparation=preparation,step_time=.8/q,rte_config=config,
            rte_distribution=law,rte_occurrence=occurrence,controlled=True,
            ancilla_qubit=preparation.num_system_qubits,seed=streams['component_'+str(i)],rte_basis_plan=None))
    return tuple(out),orders


def numerical_law_diagnostic(saved_probabilities,canonical_reference,tolerance=Fraction('1e-12')):
    """Engineering comparison ONLY. No cost-bias or interval certificate."""
    values=[Fraction(x) for x in saved_probabilities];total=sum(values)
    require(total>0 and len(values)==len(canonical_reference),'LAW_DIAGNOSTIC_DOMAIN')
    differences=[abs(x/total-Fraction(y)) for x,y in zip(values,canonical_reference)]
    return dict(normalized_probability_differences=[str(d) for d in differences],
        engineering_gate_passed=max(differences)<=tolerance,tolerance=str(tolerance),
        compiled_cost_bias_bounded=False,bit_level_RNG_equivalence=False,certificate=False)
