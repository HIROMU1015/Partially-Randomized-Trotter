"""Signed rotation support, caller-supplied metadata/bits only; passive import.

No file/model loader, sampler invocation, RNG, circuit, scientific runner or
new numeric acceptance threshold. Existing sources remain unchanged.
"""
from collections import Counter
from dataclasses import dataclass, field, replace
from fractions import Fraction as F
import json

from . import a1_control_variates as cv
from . import a1_n3_connection as n3
from . import scv_scalar as scalar
from .parent_allocation import rational
from .return_aggregation import Interval

FEATURE='rotation_coefficient_sign_times_diagonal_support_size_over_2_v1'
FEATURE_LOWER,FEATURE_UPPER=F(-1),F(1)


@dataclass(frozen=True)
class Component:
    component_id: str
    coefficient_sign: int
    diagonal_pauli_support: tuple

    def __post_init__(self):
        object.__setattr__(self,'diagonal_pauli_support',tuple(self.diagonal_pauli_support))
        s=self.diagonal_pauli_support
        if (not isinstance(self.component_id,str) or not self.component_id
            or type(self.coefficient_sign)is not int or self.coefficient_sign not in (-1,1)
            or any(type(i)is not int or i<0 for i in s)
            or len(s)>2 or s!=tuple(sorted(set(s)))):
            raise ValueError('label/sign/canonical I/Z/ZZ support required')

    @property
    def value(self):return F(self.coefficient_sign*len(self.diagonal_pauli_support),2)


@dataclass(frozen=True)
class Metadata:
    base: scalar.Metadata
    components: tuple
    system_qubits: int
    _identity: str = field(init=False,repr=False)

    def payload(self):
        return dict(base_identity=self.base.identity,system_qubits=self.system_qubits,
            components=tuple((r.component_id,r.coefficient_sign,r.diagonal_pauli_support) for r in self.components),
            feature=FEATURE)

    def __post_init__(self):
        object.__setattr__(self,'components',tuple(self.components))
        if (type(self.system_qubits)is not int or self.system_qubits<1
            or any(not isinstance(c,Component) for c in self.components)
            or tuple(c.component_id for c in self.components)!=tuple(l for l,_ in self.base.catalog)
            or any(i>=self.system_qubits for c in self.components for i in c.diagonal_pauli_support)):
            raise ValueError('complete label-aligned sign/support catalog and qubit domain required')
        object.__setattr__(self,'_identity',scalar.fingerprint(self.payload()))

    @property
    def identity(self):
        if scalar.fingerprint(self.payload())!=self._identity:raise ValueError('signed metadata identity changed')
        return self._identity

    @property
    def sampler(self):return self.base.sampler

    @property
    def law_sha256(self):return self.base.law_sha256

    def cell(self,event):
        self.identity
        return self.base.cell(event)

    def rotation(self,event):
        self.cell(event)  # validates phase, all labels/positions and positive grid support
        label=event.selected_component_ids[-1]
        return next(r for r in self.components if r.component_id==label)


def metadata_from_bytes(law_bytes,catalog_bytes,*,system_qubits,**identity):
    """Deferred pure saved-schema binder. Current tests supply invented bytes."""
    base=scalar.metadata_from_bytes(law_bytes,catalog_bytes,**identity)
    rows=json.loads(catalog_bytes)['components']
    components=tuple(Component(r['component_id'],r['coefficient_sign'],r['diagonal_pauli_support']) for r in rows)
    return Metadata(base,components,system_qubits)


@dataclass(frozen=True)
class SignedControl:
    metadata: Metadata
    means: tuple = field(init=False)
    second_moments: tuple = field(init=False)
    variances: tuple = field(init=False)
    identity: str = field(init=False)

    def __post_init__(self):
        md=self.metadata;md.identity;law=md.sampler.law.labels
        features={r.component_id:r.value for r in md.components}
        qa=law.mass(md.sampler.a);moments=[]
        for k in (1,2):
            total=sum((p*features[l]**k for l,p in zip(law.labels,law.masses)),F(0))
            fa=features[md.sampler.a]**k
            moments.append((fa,(total-qa*fa)/(1-qa),total,total,total))
        means,second=moments;variance=tuple(s-m*m for m,s in zip(means,second))
        if any(not -1<=m<=1 for m in means) or any(not 0<=s<=1 for s in second) or min(variance)<0:
            raise ValueError('invalid exact signed conditional moments')
        object.__setattr__(self,'means',means);object.__setattr__(self,'second_moments',second)
        object.__setattr__(self,'variances',variance)
        object.__setattr__(self,'identity',scalar.fingerprint(dict(metadata=md.identity,feature=FEATURE,
            means=means,second_moments=second,feature_bounds=(-1,1))))

    def value(self,event):return self.metadata.rotation(event).value

    def value_from_rte(self,event):
        # Existing explicit product/rotation consistency guard, then same feature.
        return self.value(cv.EventMetadata.from_rte_event(event))

    def snapshot(self):
        return dict(feature=FEATURE,metadata_identity=self.metadata.identity,control_identity=self.identity,
            conditional_means=self.means,second_moments=self.second_moments,variances=self.variances,
            feature_bounds=(FEATURE_LOWER,FEATURE_UPPER),range_guard='NOT_ADOPTED',science_ready=False)


@dataclass(frozen=True)
class ConditionalBank(scalar.ConditionalBank):
    def __post_init__(self):
        if not isinstance(self.control,SignedControl) or self.cell not in scalar.CELLS:
            raise ValueError('signed conditional bank and original cell required')
        self.control.metadata.identity

    @property
    def controls(self):return ((FEATURE,self.cell,self.control.identity),)


def residual_range(coefficient,mean,*,feature_lower=FEATURE_LOWER,feature_upper=FEATURE_UPPER):
    """Exact box for B in [-1,1]; never observed-sample min/max or a guard."""
    lam,nu,a,b=map(rational,(coefficient,mean,feature_lower,feature_upper))
    # This feature's fixed SOURCE enclosure is [-1,1]. Do not accept an
    # accidentally copied indicator box or a sample-derived tighter range.
    if (a,b)!=(FEATURE_LOWER,FEATURE_UPPER) or not a<=nu<=b:
        raise ValueError('SOURCE signed range [-1,1] and compatible exact mean required')
    ends=(lam*(nu-a),lam*(nu-b))
    interval=Interval(F(-1)+min(ends),F(1)+max(ends))
    if interval.hi-interval.lo!=2+2*abs(lam):raise ValueError('signed range identity failed')
    return interval


def validate_training(control,training):
    if training.stage!='train':raise ValueError('train-only fit required')
    original=n3.Domain('A1',training.scope.axis,'train','1').token
    if any(r.stream_id!=original for r in training.records):raise ValueError('h1 train acquisition-domain mismatch')
    projected=replace(training,records=tuple(replace(r,stream_id=n3.Domain('B1-OL',training.scope.axis,'train','1').token) for r in training.records))
    n3.validate_batch(control.metadata.sampler,projected,'B1-OL','1','train')


@dataclass(frozen=True)
class SignedFrozenCV(cv.FrozenCV):
    def __post_init__(self):
        if not isinstance(self.bank,ConditionalBank) or self.bank.cell!='1':
            raise ValueError('signed h1 bank required; indicator range adapter forbidden')
        super().__post_init__()
        validate_training(self.bank.control,self.training)
        self.output_range  # validates exact coefficient/mean/enclosure, no threshold

    @property
    def output_range(self):return residual_range(self.coefficients[0],self.bank.means[0])

    def apply(self,batch):
        if batch.stage!='prod':raise ValueError('fresh diagnostic evaluation only; cal not connected')
        n3.validate_batch(self.bank.control.metadata.sampler,batch,'B1-OL','1','prod')
        values=super().apply(batch)  # reuse frozen correction and stream disjointness
        interval=self.output_range
        if any(not interval.lo<=v<=interval.hi for v in values):raise ValueError('residual outside signed range')
        return values


@dataclass(frozen=True)
class SignedModels(scalar.CellModels):
    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.control,SignedControl) or not isinstance(self.models[1].frozen,SignedFrozenCV):
            raise ValueError('signed frozen adapter required; indicator range forbidden')
        if any(not isinstance(m.bank,ConditionalBank) for m in self.models):raise ValueError('signed cell banks required')


def fit_h1(control,training,*,sink=None):
    """Existing OLS once. Rank2/finite required; no numerical width guard chosen."""
    import numpy as np
    bank=ConditionalBank(control,'1');validate_training(control,training)
    fit=cv.fit_and_freeze(bank,training)  # never access its indicator-only range
    frozen=SignedFrozenCV(bank,training,fit.coefficients)
    design=np.asarray([[1,*bank.values(r.events)] for r in training.records],dtype=float)
    rank=int(np.linalg.matrix_rank(design))
    info=dict(rank=rank,coefficient=frozen.coefficients[0],range=frozen.output_range,
        width=frozen.output_range.hi-frozen.output_range.lo,range_guard='NOT_ADOPTED',
        feature_bounds=(-1,1),feature_counts=dict(Counter(str(bank.values(r.events)[0]) for r in training.records)),
        control_identity=control.identity,coefficients_by_cell=(F(0),frozen.coefficients[0],F(0),F(0),F(0)))
    if sink is not None:sink(info)
    if rank!=2:raise ValueError('rank deficient signed h1; STOP without refit')
    cells=tuple(scalar.CellModel(ConditionalBank(control,h),frozen if h=='1' else None) for h in scalar.CELLS)
    return SignedModels(control,training.scope,training,cells)


def paired_evaluate(models,rows,*,counts):
    """Caller fixes artificial quotas; no inherited 224-shot scientific defaults."""
    if not isinstance(models,SignedModels):raise ValueError('signed model with validated range required')
    rows=tuple(rows)
    result=scalar.paired_evaluate(models,rows,counts=tuple(counts))
    # Legacy arithmetic is generic, but its feature-count serializer lists
    # only indicator values 0/1. Replace just that field with all signed values.
    for record,(_,batch) in zip(result['cells'],rows):
        record['feature_counts']=dict(Counter(str(models.control.value(r.events[0])) for r in batch.records))
    result['feature']=FEATURE;result['range_guard']='NOT_ADOPTED'
    return result
