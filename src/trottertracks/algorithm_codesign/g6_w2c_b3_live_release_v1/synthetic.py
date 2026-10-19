"""Owned API-shaped runtime objects. No G7 law, root, atan or scientific module.

The artificial order/label tables below are interface fixtures, independent of
the candidate P5 law. Receipt/input binding never certifies these as that law.
"""
from collections import Counter
from fractions import Fraction as F
from types import SimpleNamespace
from .guard import require
from ..g6_w2c_b3_real_adapter_v1 import codec as C, result as R
from .. import g6_w2c_cost_preparation as AUDIT

SOURCE='ARTIFICIAL_API_SHAPED_RUNTIME_RELEASE_V1'


def ids(tid):
    return dict(input_id='ARTIFICIAL_INPUT_BINDING',policy_id='ARTIFICIAL_PRECISION_BINDING',
        source_id=SOURCE,preflight_id='ARTIFICIAL_PREFLIGHT',trial_id=tid,sigma=1)


class Evidence:
    def __init__(self,value):
        d=C.Document.create(value);self.document,self.digest=d.document,d.digest
    def read(self):return C.Document(self.document,self.digest).read()


def index(law,width,value):
    require(type(value) is int and 0<=value<1<<width,'ARTIFICIAL_DRAW_INTEGER')
    cumulative=F(0)
    for i,q in enumerate(law):
        cumulative+=q
        if F(value,1<<width)<cumulative:return i
    raise ValueError('ARTIFICIAL_LAW_NORMALIZATION')


def event(word,child):
    word=tuple(word);calls=Counter(word);calls[child]+=2
    return dict(word=word,raw_word=word,child=child,ratio=F(1,4),coefficient=F(3,7),
        proposal=F(1,2),weight=F(6,7),phase_i_power=len(word)%4,provider_calls=dict(calls),helper=1,
        event_unitary='ARTIFICIAL_EVENT_INTERFACE_ONLY',circuit_time_word=tuple(reversed(word)))


class Generator:
    def __init__(self,p,x,m,arm,**precision):
        self.p,self.x,self.m,self.arm=tuple(p),x,m,arm
        self.H,self.K=precision['probability_bits'],precision['root_bits']
        self.eta,self.rho=precision['eta'],precision['rho']
        self.order=(F(1,2),F(1,2));self.labels=(F(1,2),F(1,4),F(1,4))
        self.group_law=(F(1),);self.last_word=()
    def packet(self,word):
        word=tuple(word)
        require(R.reduced(word)==word,'ARTIFICIAL_PACKET_NONREDUCED')
        children=tuple(i for i in range(3) if not word or i!=word[0])
        return dict(children=children,child_law=tuple(F(1,len(children)) for _ in children),
                    acceptance=getattr(self,'artificial_acceptance',F(3,4)))
    def event(self,*args):
        word,child=args if self.arm=='full_return' else args[1:]
        if self.arm=='full_return':self.packet(word)
        return event(word,child)
    def sample(self,bits):
        if self.arm!='full_return':
            index(self.group_law,self.H,bits.bits(self.H))
            child=index(self.labels,self.H,bits.bits(self.H))
            return self.event(0,(),child)
        degree=2*index(self.order,self.H,bits.bits(self.H))
        raw=tuple(index(self.labels,self.H,bits.bits(self.H)) for _ in range(degree))
        if R.reduced(raw)!=raw:return None
        packet=self.packet(raw)
        if F(bits.bits(self.H),1<<self.H)>=packet['acceptance']:return None
        child=packet['children'][index(packet['child_law'],self.H,bits.bits(self.H))]
        return event(raw,child)


def state(g):
    return dict(artificial=True,p=g.p,x=g.x,m=g.m,arm=g.arm,H=g.H,K=g.K,
                eta=g.eta,rho=g.rho,order=g.order,labels=g.labels,
                artificial_acceptance=getattr(g,'artificial_acceptance',F(3,4)))


def snapshot(g,law_builder=None):
    return dict(artificial_snapshot=True,held_state=state(g),
                scientific_canonical_snapshot_verified=False)


def angle(ratio,*,tau,root_bits,terms,sigma=1,context=None):
    lo,hi=F(1,8),F(1,8)+F(1,10**8);mid=(lo+hi)/2;radius=(hi-lo)/2
    return Evidence(dict(schema='g6_f3_angle_enclosure_v1',ratio=ratio,sigma=sigma,
        root_bits=root_bits,terms=terms,tau=tau,context=context,sqrt_interval=None,
        half_argument_interval=None,endpoint_series_evidence=None,phi_interval=(lo,hi),
        representative=mid,representative_error_upper=radius,signed_phi_interval=(lo,hi),
        signed_representative=mid,precision_check=dict(name='representative_radius',left=radius,
            operator='<=',right=tau,passed=True),native_rotation_acquired=False,
        physical_provider_certified=False,basis={'scope':'ARTIFICIAL_NOT_AN_ATAN_ENCLOSURE'}))


class Session:
    def __init__(self,p,x,m,**kw):
        params={k:kw[k] for k in ('root_bits','probability_bits','eta','rho')}
        self._generator=Generator(p,x,m,'full_return',**params)
        self.preflight=Evidence(dict(artificial=True,ready=True))
        self.kw=kw
        self.request=SimpleNamespace(input_id=ids('')['input_id'],policy_id=ids('')['policy_id'],sigma=1)
    def status(self):
        return dict(source_id=SOURCE,preflight_id=self.preflight.digest)
    def sample(self,bits=None,*,trial_id):
        from .bits import ObservedBits
        trace=bits if type(bits) is ObservedBits else ObservedBits(bits,self._generator,index,full=True)
        e=self._generator.sample(trace);raw=tuple(trace.raw)
        context={**ids(trial_id),**self.status()}
        data=dict(schema='g6_f3_trial_v1',**context,run_completion_or_confidence_certified=False,
                  trace=trace.f3_trace())
        if e is None:data.update(kind='ZERO',reason='RAW_NONREDUCED' if R.reduced(raw)!=raw else 'COIN_REJECT',is_valid_zero_trial=True)
        else:data.update(kind='EVENT',event=e)
        return Evidence(data)
    def angle_for_event(self,record):
        data=record.read();e=data['event'];query=self._generator.event(e['word'],e['child'])
        require(C.fingerprint(query)==C.fingerprint(e),'ARTIFICIAL_REQUERY_MISMATCH')
        return angle(e['ratio'],tau=self.kw['angle_tau'],root_bits=self.kw['angle_root_bits'],
            terms=self.kw['angle_terms'],sigma=1,context={**{k:data[k] for k in (*R.IDS,'sigma')},
                'event_id':record.digest,'parent':e['word'],'child':e['child']})


class Symbols:
    """Only this exact built-in type can enter the current executable wrapper."""
    def __init__(self):
        self.audit=SimpleNamespace(canonical_snapshot=snapshot,canonical_state=state,
            AuditParity=AUDIT.AuditParity,event_witness=AUDIT.event_witness)
        self.g7=SimpleNamespace(make_generator=lambda p,x,m,arm,**kw:Generator(p,x,m,arm,**kw),
            _law=lambda values,bits,eta:tuple(values))
        self.f3=SimpleNamespace(prepare=lambda p,x,m,**kw:Session(p,x,m,**kw),_state=state)
        self.precision=SimpleNamespace(enclose_atan=angle)
        self.dyadic_index=index
