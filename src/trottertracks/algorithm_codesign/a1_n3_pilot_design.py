"""Pilot design helpers, no loader, sampler execution, fit, engine or runner.

The proposed budget is not an execution grant. Only invented data are used in
this task. No finite-confidence calibration or sufficient-N calculation here.
"""
from dataclasses import dataclass
from fractions import Fraction as F

from .parent_allocation import rational, integer
from .return_aggregation import root_interval
from . import a1_n3_connection as n3


@dataclass(frozen=True)
class PilotBudget:
    axes: int = 2
    cells: int = 5
    train_per_cell: int = 16
    probe_per_cell: int = 8
    evaluation_per_method_axis: int = 128
    evaluation_floor: int = 4

    @property
    def counts(self):
        return dict(A1_train=self.axes*self.cells*self.train_per_cell,
                    B1_OL_probe=self.axes*self.cells*self.probe_per_cell,
                    A1_evaluation=self.axes*self.evaluation_per_method_axis,
                    B1_OL_evaluation=self.axes*self.evaluation_per_method_axis)

    @property
    def total(self):
        return sum(self.counts.values())


BUDGET = PilotBudget()


def allocate_evaluation(masses, sample_variances, mean_rz, *, total=128, floor=4):
    """Heuristic cost-aware quotas, preserving five positive exact masses.

    v_proxy=(1+min(s2,1))/2 is a fixed shrinkage proxy, NOT a confidence bound.
    The midpoint of the existing 128-bit root enclosure is only a deterministic
    score. Largest remainders apportion the 108 extra shots; cell-order ties.
    No mass is rounded/renormalized and no zero empirical variance is certified.
    """
    p=tuple(map(rational,masses));v=tuple(map(rational,sample_variances));c=tuple(map(rational,mean_rz))
    integer(total,'total',1);integer(floor,'floor',2)
    if len(p)!=5 or len(v)!=5 or len(c)!=5 or min(p)<=0 or sum(p)!=1:
        raise ValueError('all five positive exact masses summing to one required')
    if min(v)<0 or min(c)<0 or total<5*floor:
        raise ValueError('nonnegative variances/prices and feasible floor required')
    scores=[]
    for mass,variance,price in zip(p,v,c):
        proxy=(1+min(variance,F(1)))/2
        enclosure=root_interval(proxy/max(F(1),price),128)
        scores.append(mass*(enclosure.lo+enclosure.hi)/2)
    remaining=total-5*floor
    ideal=tuple(remaining*s/sum(scores) for s in scores)
    extra=[x.numerator//x.denominator for x in ideal]
    for i in sorted(range(5),key=lambda i:(-(ideal[i]-extra[i]),i))[:remaining-sum(extra)]:
        extra[i]+=1
    result=tuple(floor+x for x in extra)
    if sum(result)!=total or min(result)<floor:raise AssertionError('quota invariant')
    return result


@dataclass(frozen=True)
class PilotMeanPlan:
    """Only aggregation fields; deliberately has no epsilon/alpha/guarantee."""
    method: str
    scope: object
    cells: tuple
    antecedents: tuple
    model: object = None


def pilot_mean(sampler, method, scope, batches, frozen_counts, *, antecedents=(), model=None):
    """Reuse existing five-cell/independence/measurement validation and mean.

    Existing cv.stratified_mean groups by control patterns and cannot represent
    both distinct B1-OL cells sharing (0,0,0); n3.production_mean uses cell IDs.
    Legacy stage='prod' means fresh evaluation here, not a confidence claim.
    """
    if method not in ('A1','B1-OL'):raise ValueError('pilot supports A1 and B1-OL only')
    if (method=='A1') != (model is not None):raise ValueError('A1 requires frozen controls')
    cells=sampler.cells(method);counts=tuple(frozen_counts)
    if len(counts)!=len(cells):raise ValueError('complete frozen quota vector required')
    for count in counts:integer(count,'evaluation count',2)
    if method=='A1':
        if model.bank!=sampler.bank:raise ValueError('control bank mismatch')
        antecedents=tuple(antecedents)+(model.training,)
    plan=PilotMeanPlan(method,scope,tuple((k,p,n,None) for (k,p),n in zip(cells,counts)),tuple(antecedents),model)
    return n3.production_mean(plan,sampler,batches)
