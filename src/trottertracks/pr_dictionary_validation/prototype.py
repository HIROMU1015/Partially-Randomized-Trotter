"""Typed atoms, exact arm canonicalization and existing finite-RTE bridge."""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Any
from .fixture import EvolutionContext, digest


@dataclass(frozen=True)
class MixedAtom:
    atom_id: str
    kind: str
    coefficient: Fraction
    origins: tuple[str,...]
    word: tuple[str,...]
    basis_key: str | None
    frame_id: str | None = None

    def __post_init__(self):
        if self.kind not in ('FRAME_Z','PHYSICAL_PAULI','IDENTITY'):
            raise ValueError('Unknown typed atom kind')
        if not isinstance(self.coefficient,Fraction) or not self.coefficient:
            raise ValueError('Only exact nonzero rational coefficients')
        if not self.atom_id or not self.origins or len(set(self.origins))!=len(self.origins):
            raise ValueError('Missing/duplicate atom provenance')
        if not self.word or any(p not in 'IXYZ' or len(p)!=1 for p in self.word):
            raise ValueError('Invalid Pauli word')
        if (self.kind=='IDENTITY') != all(p=='I' for p in self.word):
            raise ValueError('Identity kind/word mismatch')
        if self.kind=='FRAME_Z' and (any(p not in 'IZ' for p in self.word)
                or not 1<=sum(p!='I' for p in self.word)<=2 or self.frame_id is None):
            raise ValueError('FRAME_Z requires a real frame and Z/ZZ support')
        if self.kind!='IDENTITY' and not self.basis_key:
            raise ValueError('Nonidentity basis key required')
        if self.kind=='PHYSICAL_PAULI' and self.frame_id is not None:
            raise ValueError('Physical Pauli must not impersonate a DF frame')

    @property
    def support(self):
        return tuple(i for i,p in enumerate(self.word) if p!='I')


def atom_record(a: MixedAtom):
    return dict(atom_id=a.atom_id,kind=a.kind,coefficient=str(a.coefficient),
                origins=a.origins,word=a.word,basis_key=a.basis_key,frame_id=a.frame_id)


@dataclass(frozen=True)
class FragmentAlternatives:
    fragment_id: str
    frame: tuple[MixedAtom,...]
    pauli: tuple[MixedAtom,...]


def make_fragment_alternatives(materialized):
    out=[]
    for f, matrix, smap, pmap in materialized:
        frame_key='frame:'+digest([[str(x) for x in row] for row in matrix])
        sides=[]
        for side,terms in (('S',smap),('P',pmap)):
            atoms=[]
            for word,c in sorted(terms.items()):
                is_i=all(p=='I' for p in word)
                kind='IDENTITY' if is_i else 'FRAME_Z' if side=='S' else 'PHYSICAL_PAULI'
                key=None if is_i else frame_key if side=='S' else 'local:'+''.join(
                    p if p in 'XY' else 'Z' for p in word)
                atoms.append(MixedAtom(f'{side}:{f.fragment_id}:{"".join(word)}',kind,c,
                    (f.fragment_id,),word,key,f.fragment_id if kind=='FRAME_Z' else None))
            sides.append(tuple(atoms))
        out.append(FragmentAlternatives(f.fragment_id,*sides))
    return tuple(out)


def canonicalize_arm(alternatives, arm: str):
    if arm not in ('SS','SP','PS','PP') or len(alternatives)!=2:
        raise ValueError('Exactly the four P0 arms')
    if tuple(f.fragment_id for f in alternatives)!=('F0','F1'):
        raise ValueError('Stable F0/F1 ordering required')
    result=[]; collected={}
    for side,f in zip(arm,alternatives,strict=True):
        if side=='S': result.extend(f.frame)
        else:
            for a in f.pauli:
                old,origins=collected.get(a.word,(Fraction(0),set()))
                collected[a.word]=(old+a.coefficient,origins|set(a.origins))
    for word,(c,origins) in sorted(collected.items()):
        if c==0: continue
        ident=all(p=='I' for p in word)
        key=None if ident else 'local:'+''.join(p if p in 'XY' else 'Z' for p in word)
        result.append(MixedAtom('P:collect:'+''.join(word),
            'IDENTITY' if ident else 'PHYSICAL_PAULI',c,tuple(sorted(origins)),word,key))
    return tuple(sorted(result,key=lambda a:a.atom_id))


@dataclass(frozen=True)
class MixedTailView:
    tail_id: str
    tail_hash: str
    lambda_r: float
    components: tuple[Any,...]
    atoms: tuple[MixedAtom,...]
    exact_lambda: Fraction

    @property
    def deterministic_only_certified(self):
        # Exact zero norm, NOT a small floating point norm.
        return self.exact_lambda==0 and not self.atoms and not self.components


def make_tail(atoms, tail_id):
    from trotterlib.rte import RTEComponent
    atoms=tuple(atoms)
    if len({a.atom_id for a in atoms})!=len(atoms):
        raise ValueError('Duplicate atom ID')
    lam=sum((abs(a.coefficient) for a in atoms),Fraction(0))
    components=[]
    for a in atoms:
        value=float(abs(a.coefficient)); prob=float(abs(a.coefficient)/lam)
        if value<=0 or prob<=0 or not math.isfinite(value):
            raise ValueError('Finite-bit coefficient loss/overflow')
        components.append(RTEComponent(component_id=a.atom_id,probability=prob,
            coefficient_abs=value,coefficient_sign=1 if a.coefficient>0 else -1,
            df_fragment_id=None,basis_id=a.basis_key,basis_hash=a.basis_key,
            is_identity=a.kind=='IDENTITY',
            diagonal_pauli_support=a.support if a.kind=='FRAME_Z' else None))
    return MixedTailView(tail_id,digest([atom_record(a) for a in atoms]),
                         float(lam),tuple(components),atoms,lam)


def configure(tail: MixedTailView, context: EvolutionContext, *, truncation_tolerance: float):
    from trotterlib.rte import make_rte_config
    context.validate()
    if tail.deterministic_only_certified:
        return None,None
    if tail.lambda_r<=0:
        raise ValueError('Uncertified zero tail')
    return make_rte_config(tail,evolution_time=float(context.delta),rte_steps=context.r,
        truncation_tolerance=truncation_tolerance,finite_taylor_order=context.cutoff,seed=0)


def ordered_event(tail, distribution, order: int, products, child):
    from trotterlib.rte import _make_event
    indices={a.atom_id:i for i,a in enumerate(tail.atoms)}
    if len(products)!=order or order not in distribution.orders:
        raise ValueError('Unregistered order/product count')
    return _make_event((indices[child],*(indices[p] for p in products)),tail.components,
                       distribution,distribution.orders.index(order))


def finite_events(tail, distribution):
    from trotterlib.rte import iter_rte_events
    count=sum(len(tail.atoms)**(n+1) for n in distribution.orders)
    return iter_rte_events(tail.components,distribution,max_events=count)
