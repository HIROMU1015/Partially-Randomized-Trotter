"""Small descriptor-only complete-step cut selector, no circuit or price access.

The caller supplies the frozen C7-4 event_features function. Only known
precompile fields are projected to it. No model fit or path reordering.
"""
from dataclasses import dataclass
import hashlib
import json
import math

RULE="neighbor_basis_support_load_v1"


def descriptors(ordered_events):
    """Project actual event objects/dictionaries onto compile-free metadata."""
    def get(x,k):return x[k] if isinstance(x,dict) else getattr(x,k)
    out=[]
    for step in ordered_events:
        if not step:raise ValueError("EMPTY_COMPLETE_STEP")
        rows=[]
        for event in step:
            theta=float(get(event,"rotation_angle"))
            if not math.isfinite(theta):raise ValueError("NONFINITE_DESCRIPTOR")
            applications=[]
            for app in get(event,"application_sequence"):
                basis=get(app,"basis_id");support=tuple(get(app,"diagonal_pauli_support"))
                sign=get(app,"coefficient_sign")
                if (not isinstance(basis,str) or not basis or type(sign) is not int or sign not in (-1,1)
                    or len(support) not in (1,2) or len(set(support))!=len(support)
                    or any(type(q) is not int or q<0 for q in support)):
                    raise ValueError("UNSUPPORTED_APPLICATION_DESCRIPTOR")
                applications.append(dict(basis_id=basis,diagonal_pauli_support=list(support),
                                         coefficient_sign=sign))
            if not applications:raise ValueError("EMPTY_APPLICATIONS")
            rows.append(dict(rotation_angle=theta,application_sequence=applications))
        out.append(rows)
    if len(out)<2:raise ValueError("AT_LEAST_TWO_COMPLETE_STEPS")
    return out


def identity(rows):
    return hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(",",":"),
                                    allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class CutDecision:
    rule:str
    descriptor_sha256:str
    q:int
    center:int
    selected:int
    step_loads:tuple
    candidates:tuple


def select_cut(ordered_events,event_features):
    rows=descriptors(ordered_events);q=len(rows);center=q//2;loads=[]
    for i,step in enumerate(rows):
        f=event_features(dict(q=1,sample_id=i,ordered_events=[step]))
        runs,work=f["basis_runs"],f["support_work"]
        if any(type(x) is not int or x<1 for x in (runs,work)):
            raise ValueError("INVALID_FROZEN_FEATURE_RESULT")
        # One fixed unit for step/head work. No learned weights or gate price.
        loads.append(1+runs+work)
    candidates=[]
    for cut in range(max(1,center-1),min(q-1,center+1)+1):
        left,right=sum(loads[:cut]),sum(loads[cut:])
        candidates.append((cut,max(left,right)+1,left,right))
    # One global cut loss, common to each complete-step candidate.
    best=min(candidates,key=lambda row:(row[1],abs(row[0]-center),row[0]))[0]
    return CutDecision(RULE,identity(rows),q,center,best,tuple(loads),tuple(candidates))


def build_selected(context,steps,decision):
    """Reuse C3 unchanged, validate the cut/descriptor binding before building."""
    from . import track_d_c3_whole_path as c3
    rows=descriptors([r["events"] for r in steps])
    if (not isinstance(decision,CutDecision) or decision.rule!=RULE
        or decision.q!=len(steps) or decision.center!=len(steps)//2
        or identity(rows)!=decision.descriptor_sha256
        or type(decision.selected) is not int or not 0<decision.selected<len(steps)
        or abs(decision.selected-decision.center)>1):
        raise ValueError("CUT_OR_DESCRIPTOR_BINDING")
    return c3.prepare_steps(context,steps,decision.selected,
        ordered_samples=tuple(r["sample"] for r in steps),
        fixture_selection="descriptor-only neighbor cut frozen before P0",
        cohort="O5_CUT_ARTIFICIAL_ONLY")
