"""Closed P3 sidecar for the fixed Track C H4 model; no old runner is invoked.

G7 constructs every coefficient/proposal. This adapter caches its finite laws,
uses its exact dyadic maps, and exposes explicit angles/applications rather than
pretending RA samples obey the old order-only RTEEvent distribution.
"""
from __future__ import annotations
from bisect import bisect_right
from dataclasses import dataclass, replace
from fractions import Fraction as F
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time
import numpy as np
import mpmath as mp
from .track_c_h4 import ROOT, common, save_json, finite_rte_distribution
from .algorithm_codesign.g7_generator import CanonicalGenerator, _law
from .algorithm_codesign.return_aggregation import dyadic_index
from .algorithm_codesign.g6_f3_precision import enclose_atan
from . import track_c_source_guarantee as sg

INPUT=ROOT/'.cache/closed_p3/inputs'
OUT=Path(os.environ.get('TRACK_C_CLOSED_P3_OUTPUT',str(ROOT/'artifacts/track_c_closed_p3/2026-11-08')))
SCIENCE_BASE='94b6499e4ce3675dd6eabb0e7c0e3657e1911dc9'
BITS=256
ETA=F(1,10**24)
RHO=F(1,10**24)


def provenance():
    return dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),
                science_base=SCIENCE_BASE,common_sha256=hashlib.sha256((INPUT/'common.pkl').read_bytes()).hexdigest())


def iv(v):
    if isinstance(v,F):return mp.iv.mpf(v.numerator)/v.denominator
    return mp.iv.mpf(v)


def interval_json(v):return [sg.lower(v),sg.up(v)]


@lru_cache(None)
def policy(family='PR',primary='RZ'):
    return next(p for p in json.loads((INPUT/'policy.json').read_text())['policies'] if p['family']==family and p['primary']==primary)


@lru_cache(None)
def probabilities():
    symbols=common()['preps'][3].rte_preparation.symbolic_tail.components
    ps=tuple(F.from_float(x.probability) for x in symbols);s=sum(ps)
    assert len(ps)==218 and min(ps)>0
    return tuple(x/s for x in ps)


def cells():
    return sorted({(float(r['time']),int(r['r'])) for p in [policy()] for s in p['selected'] for r in s['signal']['recipe']})


def cellkey(t,n):return f'{float(t).hex()}_{n}'


class CachedCanonical(CanonicalGenerator):
    """Only cache existing conditional laws/event packets; coefficients unchanged."""
    @lru_cache(None)
    def conditional(self,excluded):return super().conditional(excluded)
    @lru_cache(None)
    def first_distribution(self):return super().first_distribution()
    @lru_cache(None)
    def event(self,group_index,word,child):return super().event(group_index,word,child)


@lru_cache(None)
def generator(t,n):
    x=F.from_float(common()['preps'][3].exact_rte_lambda_r)*F.from_float(t)/n
    return CachedCanonical(probabilities(),x,3,'closed_P3_tail',root_bits=BITS,probability_bits=BITS,eta=ETA,rho=RHO)


@lru_cache(None)
def angle(ratio):
    d=enclose_atan(ratio,tau=F(1,10**28),root_bits=BITS,terms=64).read()
    value=float(d['representative']);lo,hi=d['phi_interval']
    error=max(abs(F.from_float(value)-lo),abs(F.from_float(value)-hi))
    return value,error


def cdf(law):
    cumulative=[];running=0
    for p in law:
        c=p*(1<<BITS);assert c.denominator==1 and c>0
        running+=int(c);cumulative.append(running)
    assert running==(1<<BITS)
    return tuple(cumulative)


class BitStream:
    """Seeded pseudo-random trace for tests; fresh uniform bits remain a model assumption."""
    def __init__(self,seed):self.rng=random.Random(seed);self.draws=0
    def bits(self,width):self.draws+=1;return self.rng.getrandbits(width)


class DigitalSampler:
    def __init__(self,g):
        self.g=g;self.groups=cdf(g.group_law);self.labels=cdf(g.label_law)
        self.root=cdf(_law(g.root_child,g.H,g.eta));self.conditional={}
    def draw(self,table,bits):return bisect_right(table,bits.bits(BITS))
    def conditional_draw(self,j,bits):
        if j not in self.conditional:
            ids,exact,law=self.g.conditional(j);self.conditional[j]=(ids,cdf(law))
        ids,table=self.conditional[j];return ids[self.draw(table,bits)]
    def sample(self,bits):
        gi=self.draw(self.groups,bits);g=self.g.groups[gi]
        if g.mode=='root_collected':word=();child=self.draw(self.root,bits)
        elif g.mode=='fixed_first':
            word=(g.first,self.conditional_draw(g.first,bits));child=self.conditional_draw(g.first,bits)
        else:raise ValueError('closed P3 adapter does not extend to other m/families')
        return self.g.event(gi,word,child)


def certify(g):
    """Finite m3 group/conditional support certification, not the Full F3 preflight.

    Group intervals are already validated by G7. All other laws are finite:
    certify all L conditional laws and the collected root, without word expansion.
    Up to 4 categorical factors occur in an event. Each midpoint root has relative
    error <=RHO; pi*W=alpha_mid is exact by G7. Hence Wmax <= B*(1+RHO)/(1-ETA)^4.
    """
    mp.iv.prec=192
    root=_law(g.root_child,g.H,g.eta)
    for exact,actual in [(g.root_child,root),(g.p,g.label_law)]:
        assert sum(actual)==1 and all((1-ETA)*p<=q<=(1+ETA)*p for p,q in zip(exact,actual))
    for j in range(len(g.p)):
        ids,exact,actual=g.conditional(j)
        assert j not in ids and sum(actual)==1 and all((1-ETA)*p<=q<=(1+ETA)*p for p,q in zip(exact,actual))
    for r in g.roots:assert r.hi-r.lo<=2*RHO*r.lo
    B=sum((mp.iv.sqrt(iv(q.square)) for q in g.groups),iv(0))
    ordB=mp.iv.sqrt(1+iv(g.x)**2)+iv(g.x)**2/2*mp.iv.sqrt(1+(iv(g.x)/3)**2)
    w=B*iv(1+RHO)/iv(1-ETA)**4
    errors=[angle(q.ratio)[1] for q in g.groups]
    # Coefficient approximation and angle rounding target corrected P3, not raw M/B.
    per_step=B*(iv(RHO)+iv(1+RHO)*iv(max(errors)))
    return dict(x_exact=str(g.x),x=float(g.x),domain=bool(0<g.x<=1),p_sum_exact=str(sum(g.p)),labels=len(g.p),
        probability_bits=BITS,root_bits=BITS,eta=str(ETA),rho=str(RHO),
        B=interval_json(B),ordinary_B=interval_json(ordB),log_B=interval_json(mp.iv.log(B)),
        log_Wmax=interval_json(mp.iv.log(w)),Wmax=sg.up(w),M2_upper=sg.up(w*w),
        root_relative_error_upper=str(RHO),maximum_angle_error_upper=sg.up(iv(max(errors))),
        corrected_mean_bias_step_upper=sg.up(per_step),conditional_laws_checked=len(g.p),
        rejection_probability=0,zero_probability=0,
        estimator='Y = exact event weight * signed Hadamard outcome; normalize by certified Wmax before bounded Hoeffding',
        scope='all m3 G7 groups/conditional labels, fresh independent bits, ideal native model')


@dataclass(frozen=True)
class RAEvent:
    phase: complex
    unsigned_rotation_angle: float
    application_sequence: tuple


def application_events(t,n,records,sigma=1):
    from trotterlib.rte import RTEEventApplication
    assert sigma in (-1,1)
    symbols=common()['preps'][3].rte_preparation.symbolic_tail.components
    result=[]
    for record in records:
        word=tuple(record['word']);child=int(record['child']);assert len(word) in (0,2)
        assert record['proposal']*record['weight']==record['coefficient']
        applications=[]
        for index,label in enumerate((*reversed(word),child)):
            s=symbols[label]
            applications.append(RTEEventApplication(application_index=index,role='rotation' if index==len(word) else 'product',
                component_id=s.component_id,coefficient_abs=s.coefficient_abs,coefficient_sign=s.coefficient_sign,
                is_identity=s.is_identity,df_fragment_id=s.df_fragment_id,basis_id=s.basis_id,basis_hash=s.basis_hash,
                diagonal_pauli_support=s.diagonal_pauli_support,basis_change_operations=s.basis_change_operations))
        theta,_=angle(record['ratio']);signed=sigma*symbols[child].coefficient_sign*theta
        result.append(RAEvent(1j**record['phase_i_power'],signed,tuple(applications)))
    return result


def flattened_runs(events):
    flat=[(e,a) for e in events for a in e.application_sequence];runs=[];start=0
    while start<len(flat):
        stop=start+1;basis=flat[start][1].basis_id
        while stop<len(flat) and flat[stop][1].basis_id==basis:stop+=1
        support=tuple(sorted({q for _,a in flat[start:stop] for q in a.diagonal_pauli_support}))
        runs.append((start,stop,basis,support));start=stop
    return flat,runs


@lru_cache(None)
def catalogue():return json.loads((INPUT/'catalogue.json').read_text())['entries']


def build_events(events,mode='network',basis_mode='joint'):
    from qiskit import QuantumCircuit
    from .track_c_bottleneck_basis import configuration,circuit
    from .track_c_joint_isometry import decode
    from .track_c_diagonal_fusion import emit,diagonal_word
    from trotterlib.df_rte_qiskit import _event_phase_angle
    flat,runs=flattened_runs(events);q=QuantumCircuit(9);phase=sum(_event_phase_angle(e.phase) for e in events);stats=[]
    for start,stop,basis,support in runs:
        mapping=None;entry=catalogue().get(basis+'|'+','.join(map(str,support)))
        if basis_mode=='joint' and entry is not None and entry['adopt']:
            ops=decode(entry['operations']);mapping=dict(zip(entry['support'],entry['targets']))
        elif basis_mode in ('joint','singletons') and stop==start+1:
            lookup=configuration()[2];d=configuration()[0].definition(lookup[flat[start][1].component_id].basis_id)
            ops=d.runtime_operations;mapping=dict(zip(support,range(len(support))))
        else:ops=common()['preps'][3].rte_preparation.basis_registry.definition(basis).runtime_operations
        rotations,products,p=diagonal_word(flat[start:stop],mapping);phase+=p
        q.compose(circuit(ops,9,True),inplace=True);s=emit(q,rotations,products,mode=mode);q.compose(circuit(ops,9),inplace=True)
        stats.append(dict(start=start,stop=stop,basis=basis,support=support,**s))
    if phase:q.p(phase,8)
    return q,stats
