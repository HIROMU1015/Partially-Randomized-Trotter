"""Supplied-model expectation algebra and passive B price lookup.

No molecular kernel builder, sampler, compiler, fitting, C1 coefficient changes,
or automatic model evaluation. Real-data evaluation needs its own science scope.
Finite-state closure is a caller assumption, not a property certified for C1.
"""
from copy import deepcopy
from dataclasses import dataclass
import math

import numpy as np

from . import pf_pr_rpe_pipeline as p
from .r_variable_compiled_price import describe_request

MASS_TOLERANCE = 1e-10  # Numerical failure check, not a scientific error bound.


@dataclass(frozen=True)
class MeanKernel:
    """P[i,j] and expected additive reward g[i], retaining boundary state.

    P/g are supplied finite-state model quantities, never compiled-price data.
    Composition is (P1 P2, g1 + P1 g2); no independent-boundary reset.
    """
    states: tuple
    transition: object
    reward: object

    def __post_init__(self):
        states=tuple(self.states);n=len(states)
        P=np.array(self.transition,dtype=float,copy=True);g=np.array(self.reward,dtype=float,copy=True)
        if not n or len(set(states))!=n or P.shape!=(n,n) or g.shape!=(n,):
            raise ValueError('Finite, uniquely labelled state space and matching P/g required')
        if (not np.isfinite(P).all() or not np.isfinite(g).all() or (P<0).any()
                or not np.allclose(P.sum(axis=1),1,atol=MASS_TOLERANCE,rtol=0)):
            raise ValueError('Invalid transition/reward or lost probability mass; no normalization fallback')
        P.setflags(write=False);g.setflags(write=False)
        object.__setattr__(self,'states',states);object.__setattr__(self,'transition',P);object.__setattr__(self,'reward',g)

    @classmethod
    def identity(cls,states):
        return cls(tuple(states),np.eye(len(states)),np.zeros(len(states)))

    @classmethod
    def from_events(cls,states,events,probabilities,step):
        """Enumerate states × a finite event alphabet, never complete trajectories.

        step(state,event) returns (next_state,reward) for the supplied model.
        Non-IID laws need an adequate augmented state or a different kernel.
        """
        states=tuple(states);events=tuple(events);probabilities=tuple(probabilities)
        if (len(events)!=len(probabilities) or not events
                or any(not math.isfinite(v) or v<0 for v in probabilities)
                or not math.isclose(math.fsum(probabilities),1,abs_tol=MASS_TOLERANCE,rel_tol=0)):
            raise ValueError('Explicit normalized finite event probabilities required')
        lookup={s:i for i,s in enumerate(states)};P=np.zeros((len(states),len(states)));g=np.zeros(len(states))
        for i,state in enumerate(states):
            for event,probability in zip(events,probabilities):
                if probability==0:continue
                end,cost=step(state,event)
                if end not in lookup or not math.isfinite(cost):raise ValueError('State closure or finite reward failed')
                P[i,lookup[end]]+=probability;g[i]+=probability*cost
        return cls(states,P,g)

    def then(self,following,stats=None):
        if self.states!=following.states:raise ValueError('Boundary state labels/order differ')
        if stats is not None:stats['kernel_compositions']=stats.get('kernel_compositions',0)+1
        return MeanKernel(self.states,self.transition@following.transition,
                          self.reward+self.transition@following.reward)

    def repeat(self,count,stats=None):
        if type(count) is not int or count<0:raise ValueError('Nonnegative integer repetition required')
        result=MeanKernel.identity(self.states);base=self
        while count:
            if count&1:result=result.then(base,stats)
            count>>=1
            if count:base=base.then(base,stats)
        return result

    def expected(self,initial,terminal):
        if terminal is None:raise ValueError('Explicit terminal boundary reward required; unknown is not zero')
        initial=np.asarray(initial,dtype=float);terminal=np.asarray(terminal,dtype=float)
        n=len(self.states)
        if (initial.shape!=(n,) or terminal.shape!=(n,) or not np.isfinite(initial).all()
                or not np.isfinite(terminal).all() or (initial<0).any()
                or not math.isclose(float(initial.sum()),1,abs_tol=MASS_TOLERANCE,rel_tol=0)):
            raise ValueError('Explicit initial law and terminal rewards required')
        mass=initial@self.transition
        if not math.isclose(float(mass.sum()),1,abs_tol=MASS_TOLERANCE,rel_tol=0):
            raise ValueError('Lost terminal probability mass')
        value=float(initial@self.reward+mass@terminal)
        if not math.isfinite(value):raise ValueError('Nonfinite model expectation')
        return value,mass.tolist()


@dataclass(frozen=True)
class MeanPiece:
    signed_weight: float
    events: int
    left: MeanKernel
    event: MeanKernel
    right: MeanKernel
    law_source: object

    def __post_init__(self):
        if (not math.isfinite(self.signed_weight) or self.signed_weight==0
                or type(self.events) is not int or self.events<1 or not self.law_source):
            raise ValueError('Signed piece, positive allocation and sourced event law required')


def full_wrapper_mean(q,pieces,prefix,suffix,initial,terminal):
    """Closed-state model only: prefix (product_j L_j E_j^r_j R_j)^q suffix.

    Scalar-once, measurement axis and final flushing belong to supplied
    prefix/suffix/terminal. Pieces keep their original signed order/law.
    The same outer kernel must apply on every repetition. No qR-sized arrays.
    """
    if type(q) is not int or q<1 or not pieces:raise ValueError('Positive q and ordered pieces required')
    stats={'kernel_compositions':0};outer=MeanKernel.identity(prefix.states)
    for piece in pieces:
        part=piece.left.then(piece.event.repeat(piece.events,stats),stats).then(piece.right,stats)
        outer=outer.then(part,stats)
    whole=prefix.then(outer.repeat(q,stats),stats).then(suffix,stats)
    value,mass=whole.expected(initial,terminal)
    if value<0:raise ValueError('Negative full-wrapper structural model mean; no clamping')
    return dict(mean=value,terminal_state_mass=mass,**stats,state_count=len(prefix.states),
        pieces=len(pieces),event_sequence_materialized=False,
        semantics='EXPECTATION_OF_SUPPLIED_CLOSED_STATE_REWARD_MODEL',
        compiled_price_qualified=False,numerical_roundoff_bound=None)


def local_residual_mean(event_count,order_probabilities,increment_means,*,target):
    """Reuse old k1..k3 IID window forms for C-S, not an extra correction to S.

    Output is an internal RTE residual component, not full-wrapper price.
    Missing positive-weight coefficients fail; no independent-SE assumption.
    """
    if target!='C_MINUS_S':raise ValueError('Use paired C-S increments; raw compiled clusters would double count S')
    if (set(order_probabilities)!={0,2} or any(not math.isfinite(v) or v<0 for v in order_probabilities.values())
            or not math.isclose(math.fsum(order_probabilities.values()),1,abs_tol=MASS_TOLERANCE,rel_tol=0)):
        raise ValueError('Explicit K2 order law required; unknown probabilities are not zero')
    # Pure old helper, imported lazily. No old calibration/compile entry is run.
    from trotterlib.rte_connected_cluster_cost_validation import _iid_sequence_form
    form=_iid_sequence_form(event_count,order_probabilities)
    values=[]
    for key,weight in form.items():
        if weight==0:continue
        if key not in increment_means or not math.isfinite(increment_means[key]):
            raise ValueError('Missing/nonfinite positive-weight residual increment: '+key)
        values.append(weight*increment_means[key])
    return dict(internal_residual_mean=math.fsum(values),form=form,target=target,
        boundary_residual=None,full_wrapper_mean=None,systematic_uncertainty=None,
        qualification='TRUNCATED_LOCAL_RESIDUAL_COMPONENT_ONLY')


def combine_mean(structural_mean,*,residual_terms):
    """One full-wrapper residual model: C1 affine OR paired local C-S.

    Never S + raw kappa(C), or C1 affine + an overlapping local correction.
    A local residual missing PF/piece/outer/wrapper boundaries stays unknown.
    """
    if len(residual_terms)!=1:raise ValueError('Choose one residual model; do not stack C1 and local corrections')
    term=residual_terms[0]
    if term.get('target')!='C_MINUS_S' or term.get('mode') not in ('C1_AFFINE','LOCAL_C_MINUS_S'):
        raise ValueError('Paired C-S residual target/mode required')
    if structural_mean is None or term.get('mean') is None or term.get('coverage')!='FULL_WRAPPER':
        return dict(mean=None,status='MISSING_FULL_WRAPPER_COMPONENT',qualified_for_candidate=False,total_CI=None)
    value=structural_mean+term['mean']
    if not math.isfinite(structural_mean) or structural_mean<0 or not math.isfinite(term['mean']) or not math.isfinite(value) or value<0:
        raise ValueError('Invalid model mean; no clamping')
    return dict(mean=value,status='UNQUALIFIED_SUPPLIED_MODEL',qualified_for_candidate=False,
                expectation='SUPPLIED_MODEL_EXPECTATION',total_CI=None)


class MeanPriceProvider:
    """B c_provider(task, requested_rows): copy supplied records or return missing.

    Does NOT call expectation algebra or read calibration/input files. Schedule
    context is a sourced B use-context, distinct from cluster kappa coefficients;
    it is echoed, never converted into T/q/N or certified as an RPE schedule.
    """
    def __init__(self,records=(),*,pipeline=p,schedule_context=None,selections=None):
        self.records=deepcopy(list(records));self.pipeline=p
        self.selections=deepcopy(selections or {})
        if schedule_context is not None and (not schedule_context.get('source') or not schedule_context.get('schedule_id')):
            raise ValueError('Sourced, identified external schedule context required')
        self.schedule_context=deepcopy(schedule_context)

    def __call__(self,task,requested_rows):
        api=self.pipeline;api.validate_task(task);output=[]
        for row in requested_rows:
            if not any(api.same(row,r) for r in task['candidates']):raise ValueError('Requested row outside task')
            for axis in api.AXES:
                identity=api.identity(task,row);reason=None;status='MISSING'
                try:scope=describe_request(task,row,axis,pipeline=api)
                except ValueError as exc:scope=None;reason=str(exc);status='OUTSIDE_DOMAIN'
                if api.arm(row['candidate'])!='PR':reason='PF_USES_EXISTING_PF_PROVIDER';status='OUTSIDE_DOMAIN'
                matches=[] if reason else [deepcopy(r) for r in self.records
                    if r['axis']==axis and api.same(r['identity'],identity)]
                if len({r['cohort'] for r in matches})!=len(matches):raise ValueError('Duplicate price cohort/axis')
                selection=self.selections.get((row['candidate']['id'],axis))
                if selection is not None:matches=[r for r in matches if r['cohort']==selection]
                elif len(matches)>1:matches=[];reason='EXPLICIT_COHORT_SELECTION_REQUIRED';status='SELECTION_REQUIRED'
                if not matches:
                    matches=[dict(**api.measurement(row,axis,saved_sampler=api.arm(row['candidate'])=='PR'),identity=identity,axis=axis,
                        classification='UNAVAILABLE',expectation='UNAVAILABLE',cohort='MEAN_PRICE_MISSING',n_paths=None,
                        mean={'rz_count':None},model=None,law_binding=None,source={'module':'trottertracks.mean_compiled_price'},
                        price_status=status,
                        reason=reason or 'NO_SUPPLIED_FULL_WRAPPER_MEAN_OR_COMPILED_CALIBRATION',
                        normalization_is_price_multiplier=False,
                        uncertainty=dict(joint_covariance=None,R_q_systematic=None,total_CI=None,all_law_expectation=None))]
                seen=set()
                for record in matches:
                    if record['cohort'] in seen:raise ValueError('Duplicate price cohort/axis')
                    seen.add(record['cohort']);model=record.get('model')
                    if model and model.get('qualified_for_candidate') is not False:
                        raise ValueError('This preparation cannot promote model qualification')
                    value=record['mean']['rz_count'];kind=record['classification']
                    if value is not None:
                        if record.get('price_status')=='FAILED':raise ValueError('Failed price cannot carry an accepted value')
                        if (not isinstance(value,(int,float)) or isinstance(value,bool) or not math.isfinite(value) or value<0
                                or not record.get('source') or record.get('normalization_is_price_multiplier') is not False):
                            raise ValueError('Invalid supplied price/source/unit')
                        law=record.get('law_binding');c=row['candidate']
                        if (not isinstance(law,dict) or not law.get('definition',{}).get('source')
                                or any(law.get(k)!=v for k,v in dict(candidate_id=c['id'],R_outer=c['R_outer'],K=c['K'],q_PF=c['q']).items())):
                            raise ValueError('Explicit sourced candidate/R sampling law binding required')
                        if kind=='DIRECT_COMPILED_FINITE_SAMPLE':
                            if record.get('expectation')!='FINITE_COHORT_ONLY' or type(record.get('n_paths')) is not int or record['n_paths']<1:
                                raise ValueError('Direct PR finite-cohort meaning required')
                            record['price_status']='SAVED_DIRECT_FINITE_COHORT'
                        elif kind=='MODEL_PREDICTION' and model:
                            if (not api.same(model.get('domain'),c) or not api.same(model.get('binding'),task['binding'])
                                    or model.get('compiler')!=task['profile']['compiler']):
                                raise ValueError('Foreign model candidate/binding/compiler domain')
                            if record.get('expectation') not in ('FINITE_COHORT_ONLY','SUPPLIED_MODEL_EXPECTATION'):
                                raise ValueError('Do not upgrade finite/model meaning to an all-law guarantee')
                            record['price_status']='UNQUALIFIED_MODEL'
                        else:raise ValueError('DIRECT finite cohort or explicitly unqualified MODEL required')
                    else:
                        if kind!='UNAVAILABLE':raise ValueError('Missing/failed price must stay UNAVAILABLE')
                        record.setdefault('price_status','MISSING')
                    record['mean_price_scope']=scope
                    record['schedule_mapping']=dict(status='NOT_SUPPLIED' if self.schedule_context is None else 'ECHOED_NOT_SCIENTIFICALLY_VALIDATED',
                                                    context=deepcopy(self.schedule_context))
                    record['full_RPE_price_adopted']=False
                    output.append(record)
        return output
