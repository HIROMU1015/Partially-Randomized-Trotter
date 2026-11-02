"""Literal RZ recipe tokens, independent scan, and expected-price state machine.

No compiled-cost prediction/bound. Unknown prices raise; never zero-fill.
"""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Callable


class UnpricedError(ValueError):
    pass


def literal_rz_count(circuit):
    # Fixed literal decomposition grammar, NOT transpilation or native synthesis.
    tokens={'rz':1,'p':1,'s':1,'sdg':1,'z':1,'h':2,'sx':0,'sxdg':0,
            'x':0,'cx':0,'cz':4,'cp':3,'crz':2,'u':3,'u3':3,'u2':2,
            'u1':1,'rx':3,'ry':3,'id':0}
    def instruction(gate,stack):
        if gate.name in tokens: return tokens[gate.name]
        if gate.name=='unitary': raise UnpricedError('Dense oracle forbidden')
        if gate.name in stack or gate.definition is None:
            raise UnpricedError('No literal recipe for '+gate.name)
        return sum(instruction(x.operation,stack+(gate.name,)) for x in gate.definition.data)
    return sum(instruction(x.operation,()) for x in circuit.data)


@dataclass(frozen=True)
class AtomPrice:
    atom_id: str
    basis_key: str | None
    identity: bool
    negative: bool
    full: int | None
    singleton: int | None
    product: int | None
    rotation: int | None

    def validate(self):
        values=(self.product,self.rotation) if self.identity else (
            self.full,self.singleton,self.product,self.rotation)
        if any(type(x) is not int or x<0 for x in values):
            raise UnpricedError('Missing/noninteger/negative literal price')

    def central(self,role):
        self.validate()
        if role not in ('product','rotation'): raise ValueError('Role')
        cost=self.product if role=='product' else self.rotation
        return cost+(1 if role=='product' and self.negative else 0)


def recipe_prices(atoms,provider):
    prices={}
    for a in atoms:
        ident=a.kind=='IDENTITY'
        full=single=None
        if not ident:
            b=provider.basis(a,False); s=provider.basis(a,True)
            full=literal_rz_count(b)+literal_rz_count(b.inverse())
            single=literal_rz_count(s)+literal_rz_count(s.inverse())
        # Central Z product is CZ (H-CX-H); rotation is two RZs/two CXs.
        prices[a.atom_id]=AtomPrice(a.atom_id,a.basis_key,ident,a.coefficient<0,
                                    full,single,0 if ident else 4,1 if ident else 2)
    return prices


def direct_literal_price(applications,prices,*,event_count,scalar_reserves=0,
                         fixed_price=0):
    """Independent run scan; None is an explicit D/flush boundary."""
    total=fixed_price+event_count+scalar_reserves; pos=0
    while pos<len(applications):
        item=applications[pos]
        if item is None: pos+=1; continue
        aid,role=item; p=prices[aid]; p.validate()
        if p.identity:
            total+=p.central(role); pos+=1; continue
        stop=pos+1
        while stop<len(applications) and applications[stop] is not None:
            other=prices[applications[stop][0]]
            if other.identity or other.basis_key!=p.basis_key: break
            stop+=1
        total+=p.singleton if stop-pos==1 else p.full
        for atom_id,atom_role in applications[pos:stop]:
            total+=prices[atom_id].central(atom_role)
        pos=stop
    return total


def _transition(state,aid,role,prices):
    p=prices[aid]; p.validate()
    if p.identity:
        pending=0 if state[0]!='S' else prices[state[1]].singleton+prices[state[1]].central(state[2])
        return ('E',),pending+p.central(role)
    if state[0]=='E': return ('S',aid,role),0
    old=prices[state[1]] if state[0]=='S' else None
    key=old.basis_key if old else state[1]
    if p.basis_key==key:
        cost=p.central(role)+(old.full+old.central(state[2]) if old else 0)
        return ('M',key),cost
    pending=old.singleton+old.central(state[2]) if old else 0
    return ('S',aid,role),pending


def _flush(states,prices):
    mass=0; value=0
    for state,(prob,moment) in states.items():
        cost=prices[state[1]].singleton+prices[state[1]].central(state[2]) if state[0]=='S' else 0
        mass+=prob; value+=moment+prob*cost
    return {('E',):(mass,value)}


def expected_literal_price(prices,component_probabilities,order_probabilities,*,
                           q,r,fixed_price,scalar_reserves=1):
    """Propagate probability and nonnormalized cost; keep state across microsteps."""
    if q<1 or r<1: raise ValueError('q/r')
    for p in prices.values(): p.validate()
    if set(prices)!=set(component_probabilities): raise ValueError('Price/law mismatch')
    # Match _make_event's finite-bit component normalization exactly.
    values=tuple(component_probabilities.values())
    mass=math.fsum(values) if all(isinstance(x,float) for x in values) else sum(values)
    component_probabilities={k:v/mass for k,v in component_probabilities.items()}
    states={('E',):(1,0)}
    for _macro in range(q):
        for _micro in range(r):
            merged={}
            for n,pn in order_probabilities.items():
                if n<0 or n%2: raise ValueError('Even paired cutoff')
                branch={s:(prob*pn,moment*pn+prob*pn) for s,(prob,moment) in states.items()}
                # One potential event phase reserve per event, including n=0.
                for role in ('product',)*n+('rotation',):
                    next_states={}
                    for state,(prob,moment) in branch.items():
                        for aid,pa in component_probabilities.items():
                            target,cost=_transition(state,aid,role,prices)
                            op,ov=next_states.get(target,(0,0))
                            next_states[target]=(op+prob*pa,ov+pa*(moment+prob*cost))
                    branch=next_states
                for state,(prob,moment) in branch.items():
                    op,ov=merged.get(state,(0,0)); merged[state]=(op+prob,ov+moment)
            states=merged
        states=_flush(states,prices)
    mass,moment=states[('E',)]
    return mass,moment+mass*(fixed_price+scalar_reserves)


@dataclass(frozen=True)
class ScoreInterval:
    lower: Fraction
    upper: Fraction
    provider_identity: str
    enclosure_qualified: bool

    def validate(self):
        if self.lower<0 or self.upper<self.lower:
            raise UnpricedError('Invalid score interval')
        if not self.enclosure_qualified or len(self.provider_identity)!=64:
            raise UnpricedError('Unqualified score interval provider')


def strictly_improves(trial, incumbent):
    trial.validate(); incumbent.validate()
    if trial.provider_identity!=incumbent.provider_identity:
        raise UnpricedError('Different interval providers')
    return trial.upper<incumbent.lower


def select_dictionary(evaluate: Callable[[str],ScoreInterval]):
    """Exactly two endpoints, one pass F0/F1 each; at most six requests."""
    cache={}; visited=[]
    def request(arm):
        visited.append(arm)
        if arm not in cache: cache[arm]=evaluate(arm)
        cache[arm].validate()
        return cache[arm]
    try:
        endpoints=[]
        for start in ('SS','PP'):
            current=start; score=request(current)
            for i in range(2):
                trial=current[:i]+('P' if current[i]=='S' else 'S')+current[i+1:]
                value=request(trial)
                if strictly_improves(value,score): current,score=trial,value
            endpoints.append(current)
        # Retain both original endpoints and final endpoints as specified in 2C.
        champion='SS'
        for arm in sorted(set(('SS','PP',*endpoints))):
            if strictly_improves(cache[arm],cache[champion]): champion=arm
        return dict(status='INTERVAL_PRICED',arm=champion,requests=tuple(visited),cache=cache)
    except UnpricedError as error:
        return dict(status='UNPRICED_FALLBACK',arm='SS',requests=tuple(visited),
                    cache=cache,reason=str(error))


def exhaustive_four(evaluate,*,cache=None):
    cache={} if cache is None else cache
    try:
        for arm in ('SS','SP','PS','PP'):
            if arm not in cache: cache[arm]=evaluate(arm)
            cache[arm].validate()
        best='SS'
        for arm in ('SP','PS','PP'):
            if strictly_improves(cache[arm],cache[best]): best=arm
        return dict(status='KNOWN_FINITE_SEARCH',arm=best,cache=cache)
    except UnpricedError as error:
        return dict(status='UNPRICED_FALLBACK',arm='SS',cache=cache,reason=str(error))
