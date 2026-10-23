"""Explicit P0 motif and catalog definitions. Never enumerated at import."""
from __future__ import annotations
from fractions import Fraction
from .prototype import MixedAtom, ordered_event


def bind_motif_labels(atoms):
    ordered=sorted(atoms,key=lambda a:a.atom_id)
    identity=next((a.atom_id for a in ordered if a.kind=='IDENTITY'),None)
    u=next((a for a in ordered if a.kind!='IDENTITY'),None)
    negative=next((a.atom_id for a in ordered if a.kind!='IDENTITY' and a.coefficient<0),None)
    v=next((a.atom_id for a in ordered if a.kind!='IDENTITY' and u and a.basis_key!=u.basis_key),None)
    if identity is None or u is None or negative is None or v is None:
        raise ValueError('INPUT_COVERAGE_FAILURE; no automatic replacement')
    return identity,u.atom_id,negative,v


def motif_events(tail,distribution,index):
    if index not in range(8): raise ValueError('Exactly M0-M7')
    ident,u,s,v=bind_motif_labels(tail.atoms)
    # Records are (product IDs in time order, child ID).
    definitions=(
        (((),u),((),v),((),u),((),v)),
        (((),s),((),u),((),s),((),u)),
        (((u,v),s),((),u),((),u),((),u)),
        (((ident,s),ident),((),v),((),v),((),v)),
        (((),u),((),u),((),u),((),u)),
        (((u,u),u),((),u),((),u),((),u)),
        (((),u),((),u),((),v),((),v)),
        (((),ident),((),ident),((),ident),((),ident)))
    events=tuple(ordered_event(tail,distribution,len(products),products,child)
                 for products,child in definitions[index])
    return (events[:2],events[2:])


def catalog_atoms(frame_key):
    # P0 catalog only; constructing it is B science and is not done in I.
    return (
        MixedAtom('A','FRAME_Z',Fraction(1,8),('F0',),('Z','I','I','I'),frame_key,'F0'),
        MixedAtom('B','FRAME_Z',Fraction(-1,10),('F0',),('I','Z','I','I'),frame_key,'F0'),
        MixedAtom('C','PHYSICAL_PAULI',Fraction(1,12),('catalog',),('X','I','Y','Z'),'local:XZYZ'),
        MixedAtom('I','IDENTITY',Fraction(1,20),('catalog',),('I','I','I','I'),None))
