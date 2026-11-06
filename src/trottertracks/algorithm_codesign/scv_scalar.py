"""Scalar intrastatum control on caller-supplied metadata; passive import.

No file/model loader, RNG, state, circuit or compiler. Real H4 group/mean
evaluation is deliberately deferred to a separately authorized caller.
"""
from dataclasses import dataclass, field
from fractions import Fraction as F
from statistics import variance
import hashlib
import json

from . import a1_control_variates as cv
from . import a1_n3_connection as n3
from .a1_n3_pilot_design import PilotMeanPlan
from .parent_allocation import rational

CELLS=tuple(map(str,range(5)))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def fingerprint(data):
    return digest(json.dumps(data,sort_keys=True,separators=(',',':'),default=str).encode())


@dataclass(frozen=True)
class Metadata:
    sampler: object = field(compare=False,repr=False)
    catalog: tuple
    law_sha256: str
    catalog_sha256: str
    hamiltonian_hash: str
    preparation_hash: str
    _identity: str = field(init=False,repr=False)

    def payload(self):
        s=self.sampler
        return dict(law_id=s.bank.law_id,orders=s.law.orders.labels,
            order_masses=s.law.orders.masses,labels=s.law.labels.labels,
            masses=s.law.labels.masses,a=s.a,b=s.b,catalog=self.catalog,
            law_sha256=self.law_sha256,catalog_sha256=self.catalog_sha256,
            hamiltonian_hash=self.hamiltonian_hash,preparation_hash=self.preparation_hash)

    def __post_init__(self):
        object.__setattr__(self,'catalog',tuple(tuple(row) for row in self.catalog))
        labels,fragments=zip(*self.catalog)
        if (len(set(labels))!=len(labels) or set(labels)!=set(self.sampler.law.labels.labels)
            or any(not isinstance(f,str) or not f for f in fragments)
            or not all((self.law_sha256,self.catalog_sha256,self.hamiltonian_hash,self.preparation_hash))
            or self.sampler.bank.law_id!=self.law_sha256
            or self.sampler.a==self.sampler.b
            or tuple(k for k,_ in self.sampler.cells('B1-OL'))!=CELLS):
            raise ValueError('complete, identity-bound five-cell catalog required')
        object.__setattr__(self,'_identity',fingerprint(self.payload()))

    @property
    def identity(self):
        if fingerprint(self.payload())!=self._identity:
            raise ValueError('metadata/law identity changed')
        return self._identity

    @property
    def positive_fragments(self):
        self.identity
        lookup=dict(self.catalog)
        return tuple(sorted({lookup[l] for l,p in zip(self.sampler.law.labels.labels,
            self.sampler.law.labels.masses) if p>0}))

    def cell(self,event):
        self.identity
        return self.sampler.cell_of('B1-OL',n3.PathChoice(event.order,event.selected_component_ids))

    def rotation_fragment(self,event):
        self.cell(event)
        return dict(self.catalog)[event.selected_component_ids[-1]]


def metadata_from_bytes(law_bytes,catalog_bytes,*,law_sha256,catalog_sha256,
                        hamiltonian_hash,preparation_hash,a,b):
    """Pure deferred binding of exact saved schemas, tested only on toy bytes."""
    if digest(law_bytes)!=law_sha256 or digest(catalog_bytes)!=catalog_sha256:
        raise ValueError('LAW/catalog byte identity mismatch')
    law=json.loads(law_bytes);catalog=json.loads(catalog_bytes)
    if (catalog.get('hamiltonian_hash')!=hamiltonian_hash or
        catalog.get('preparation_hash')!=preparation_hash):
        raise ValueError('Hamiltonian/preparation identity mismatch')
    sampler=n3.ConditionalSampler.from_record(law,law_sha256,a,b)
    rows=tuple((r['component_id'],r['df_fragment_id']) for r in catalog['components'])
    return Metadata(sampler,rows,law_sha256,catalog_sha256,hamiltonian_hash,preparation_hash)


def conditional_indicator_mean(metadata,cell,local):
    """Reuse existing exact joint/inclusion-exclusion machinery, no path table."""
    metadata.identity
    if cell not in CELLS:raise ValueError('unknown positive cell')
    old=metadata.sampler.five_bank
    pattern=metadata.sampler.patterns[int(cell)]
    if local in old.controls:
        return F(pattern[old.controls.index(local)])
    bank=cv.ControlBank(old.law_id,old.occurrences,old.controls+(local,))
    return bank.pattern_mass(pattern+(1,))/dict(metadata.sampler.cells('B1-OL'))[cell]


@dataclass(frozen=True)
class FragmentControl:
    metadata: Metadata
    fragments: tuple
    means: tuple = field(init=False)
    identity: str = field(init=False)

    def __post_init__(self):
        groups=tuple(self.fragments)
        object.__setattr__(self,'fragments',groups)
        if groups!=tuple(sorted(set(groups))) or not set(groups)<set(self.metadata.positive_fragments) or not groups:
            raise ValueError('nonempty canonical proper fragment subset required')
        chosen={l for l,f in self.metadata.catalog if f in groups}
        means=[]
        for cell in CELLS:
            order=0 if int(cell)<2 else 2
            mean=sum((conditional_indicator_mean(self.metadata,cell,cv.LocalControl(0,order,((order,l),)))
                for l,p in zip(self.metadata.sampler.law.labels.labels,self.metadata.sampler.law.labels.masses)
                if l in chosen and p>0),F(0))
            if not 0<=mean<=1:raise ValueError('conditional mean outside indicator range')
            means.append(mean)
        object.__setattr__(self,'means',tuple(means))
        object.__setattr__(self,'identity',fingerprint(dict(metadata=self.metadata.identity,
            fragments=groups,means=means,feature='rotation_fragment_indicator_v1')))

    @property
    def second_moments(self):
        return self.means  # u²=u exactly

    @property
    def variances(self):
        return tuple(v*(1-v) for v in self.means)

    def value(self,event):
        return int(self.metadata.rotation_fragment(event) in self.fragments)

    def snapshot(self):
        return dict(metadata_identity=self.metadata.identity,control_identity=self.identity,
            fragments=self.fragments,conditional_means=self.means,
            second_moments=self.second_moments,conditional_variances=self.variances,
            range=(0,1),selection_rule='closest_to_half_then_increasing_integer_bitmask')


def select_fragment_control(metadata):
    """Pre-bit finite selection; canonical fragment index is bit i (LSB=0).

    Increasing integer masks specify the tie rule. No response argument exists.
    Fractions derive from actual grid masses; zero-grid labels are retained.
    """
    fragments=metadata.positive_fragments;lookup=dict(metadata.catalog)
    mass={f:F(0) for f in fragments}
    qa=metadata.sampler.law.labels.mass(metadata.sampler.a)
    for l,p in zip(metadata.sampler.law.labels.labels,metadata.sampler.law.labels.masses):
        if l!=metadata.sampler.a and p>0:mass[lookup[l]]+=p/(1-qa)
    best=None
    for mask in range(1,(1<<len(fragments))-1):
        groups=tuple(f for i,f in enumerate(fragments) if mask&(1<<i))
        mean=sum((mass[f] for f in groups),F(0))
        if F(1,8)<=mean<=F(7,8):
            row=(abs(mean-F(1,2)),mask,groups,mean)
            if best is None or row[:2]<best[:2]:best=row
    if best is None:raise ValueError('no eligible fragment group; STOP before measurement')
    control=FragmentControl(metadata,best[2])
    if control.means[1]!=best[3]:raise ValueError('conditional selection/moment mismatch')
    return control


@dataclass(frozen=True)
class ConditionalBank:
    control: FragmentControl
    cell: str

    @property
    def law_id(self):return self.control.metadata.law_sha256

    @property
    def controls(self):return (('rotation_fragment',self.cell,self.control.identity),)

    @property
    def means(self):return (self.control.means[int(self.cell)],)

    def values(self,events):
        if self.cell not in CELLS or len(events)!=1 or self.control.metadata.cell(events[0])!=self.cell:
            raise ValueError('conditional feature cell/order mismatch')
        return (self.control.value(events[0]),)


@dataclass(frozen=True)
class CellModel:
    bank: ConditionalBank
    frozen: object = None

    @property
    def coefficient(self):
        return self.frozen.coefficients[0] if self.frozen is not None else F(0)

    def apply(self,batch):
        if batch.stage!='prod':raise ValueError('fresh evaluation required')
        cv.raw_mean(batch,self.bank)
        values=self.frozen.apply(batch) if self.frozen is not None else tuple(r.raw for r in batch.records)
        if self.frozen is not None:
            bounds=self.frozen.output_range
            if any(not bounds.lo<=v<=bounds.hi for v in values):
                raise ValueError('residual outside frozen range')
        return values


@dataclass(frozen=True)
class CellModels:
    control: FragmentControl
    scope: object
    training: object
    models: tuple

    def __post_init__(self):
        if (tuple(m.bank.cell for m in self.models)!=CELLS
            or self.training.scope!=self.scope or self.training.stage!='train'
            or any(m.bank.control!=self.control for m in self.models)
            or self.models[1].frozen is None
            or any(m.frozen is not None for i,m in enumerate(self.models) if i!=1)
            or self.models[1].frozen.training!=self.training):
            raise ValueError('only h=1 is trained; other coefficients fixed zero')

    def apply(self,batch):
        if batch.scope!=self.scope:raise ValueError('model scope mismatch')
        cell=self.control.metadata.cell(batch.records[0].events[0])
        return self.models[int(cell)].apply(batch)


def fit_cells(control,training,*,range_guard=F(4),sink=None):
    """Existing intercept OLS/FrozenCV; exact guard on actual frozen slope."""
    import numpy as np
    bank=ConditionalBank(control,'1')
    if training.scope.weight_bound!=1:raise ValueError('fixed ±1 target required')
    cv.raw_mean(training,bank)
    frozen=cv.fit_and_freeze(bank,training)
    rank=int(np.linalg.matrix_rank([[1,*bank.values(r.events)] for r in training.records]))
    width=frozen.output_range.hi-frozen.output_range.lo
    info=dict(rank=rank,coefficient=frozen.coefficients[0],range=frozen.output_range,
        width=width,guard=rational(range_guard),training_count=len(training.records),
        training_feature_counts={str(u):sum(bank.values(r.events)[0]==u for r in training.records) for u in (0,1)},
        control_identity=control.identity,scope=training.scope,
        coefficients_by_cell=(F(0),frozen.coefficients[0],F(0),F(0),F(0)))
    if sink is not None:sink(info)  # preserve actual fit before failing guard
    if rank!=2 or width>rational(range_guard):
        raise ValueError('rank2/range guard failure; no refit')
    models=tuple(CellModel(ConditionalBank(control,h),frozen if h=='1' else None) for h in CELLS)
    return CellModels(control,training.scope,training,models)


def _covariance(x,y):
    mx=sum(x,F(0))/len(x);my=sum(y,F(0))/len(y)
    return sum(((a-mx)*(b-my) for a,b in zip(x,y)),F(0))/(len(x)-1)


def paired_evaluate(models,rows,counts=(4,64,4,4,4)):
    """Same fresh physical evaluation for baseline and CV; no fit or allocation."""
    sampler=models.control.metadata.sampler
    cells=tuple((k,p,n,None) for (k,p),n in zip(sampler.cells('B1-OL'),counts))
    if len(counts)!=5 or any(type(n)is not int or n<2 for n in counts):
        raise ValueError('five positive fixed quotas required')
    plan=PilotMeanPlan('B1-OL',models.scope,cells,(models.training,))
    raw_mean=n3.production_mean(plan,sampler,rows)
    from dataclasses import replace
    scv_mean=n3.production_mean(replace(plan,model=models),sampler,rows)
    vraw=vscv=cov=F(0);details=[]
    for (cell,batch),(_,p,n,_) in zip(rows,cells):
        x=tuple(r.raw for r in batch.records);z=models.apply(batch)
        vx,vz=variance(x),variance(z);c=_covariance(x,z)
        weight=p*p/n;vraw+=weight*vx;vscv+=weight*vz;cov+=weight*c
        details.append(dict(cell=cell,n=n,mass=p,coefficient=models.models[int(cell)].coefficient,
            raw_variance=vx,residual_variance=vz,paired_covariance=c,
            raw_all_equal=(vx==0),residual_all_equal=(vz==0),
            feature_counts={str(u):sum(models.control.value(r.events[0])==u for r in batch.records) for u in (0,1)}))
    return dict(status='PAIRED_DIAGNOSTIC_ONLY',precision='NOT_CERTIFIED',
        B1_OL=dict(mean=raw_mean,estimated_variance_of_mean=vraw),
        SCV=dict(mean=scv_mean,estimated_variance_of_mean=vscv),
        estimated_covariance_of_means=cov,
        estimated_variance_of_mean_difference=vraw+vscv-2*cov,
        cells=details,independent_method_campaigns=False,expected_cost_guarantee='UNKNOWN')
