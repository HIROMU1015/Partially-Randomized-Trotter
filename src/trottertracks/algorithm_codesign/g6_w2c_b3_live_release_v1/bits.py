"""Artificial counter stream and G7 observation capture; no science stream factory."""
from hashlib import sha256
from .guard import install,require
from .contract import ARTIFICIAL_SEED
from ..g6_w2c_b3_live_preflight_v1 import spec as S


class ArtificialBits:
    def __init__(self,arm,axis,*,namespace=ARTIFICIAL_SEED):
        install();require(namespace==ARTIFICIAL_SEED,'ARTIFICIAL_NAMESPACE_ONLY')
        require(arm in S.ARMS and axis in ('X','Y'),'SEED_CELL')
        self.domain=namespace+'/'+arm+'/'+axis
        self.counter=self.consumed_bits=0;self.trace=[];self.failure_position=None;self.stopped=False

    def bits(self,width):
        require(not self.stopped,'BITS_STOPPED')
        try:
            require(self.domain in {ARTIFICIAL_SEED+'/'+a+'/'+x for a in S.ARMS for x in ('X','Y')},'SEED_CHANGED')
            require(type(width) is int and width==160,'FIXED_REQUEST_WIDTH')
            require(type(self.counter) is int and 0<=self.counter<2**64,'COUNTER_RANGE')
            before=self.counter
            value=int.from_bytes(sha256(self.domain.encode()+before.to_bytes(8,'big')).digest(),'big')>>96
            self.counter+=1;self.consumed_bits+=width
            self.trace.append(dict(width=160,counter_before=before,counter_after=self.counter,
                                   discarded_bits=96,value_hex=hex(value)))
            return value
        except BaseException:
            self.stopped=True;self.failure_position=dict(draw_index=len(self.trace),counter=self.counter,
                                                       consumed_bits=self.consumed_bits)
            raise


def validate_trace(data,arm,axis):
    require(data['domain']==ARTIFICIAL_SEED+'/'+arm+'/'+axis and data['artificial_only'] is True,'BIT_DOMAIN')
    before=data['counter_before'];require(type(before) is int and before>=0,'COUNTER')
    for row in data['draws']:
        require(row['width']==160 and type(row['width']) is int and row['counter_before']==before
                and row['counter_after']==before+1 and row['discarded_bits']==96,'DRAW_POSITION_WIDTH')
        expected=int.from_bytes(sha256(data['domain'].encode()+before.to_bytes(8,'big')).digest(),'big')>>96
        require(row['value_hex']==hex(expected),'DRAW_VALUE');before+=1
    require(data['counter_after']==before and data['bits_consumed_before']==160*data['counter_before']
            and data['bits_consumed_after']==160*before,'CONSUMPTION_TOTAL')
    return S.digest(data)


class ObservedBits:
    """Observe existing draws; Full raw/order reconstruction uses bound dyadic_index."""
    def __init__(self,provider,generator,index,*,full):
        self.provider,self.generator,self.index,self.full=provider,generator,index,full
        self.start=len(provider.trace);self.counter_before=provider.counter
        self.bits_before=provider.consumed_bits;self.draws=[];self.raw=[];self.degree=None

    def bits(self,width):
        require(width==self.generator.H,'GENERATOR_WIDTH_CHANGED')
        value=self.provider.bits(width);i=len(self.draws)
        if self.full:
            stage='order' if i==0 else 'label' if i<=self.degree else 'acceptance' if i==self.degree+1 else 'child'
            require(self.degree is None or i<=self.degree+2,'EXTRA_DRAW')
            if i==0:self.degree=2*self.index(self.generator.order,width,value)
            elif i<=self.degree:self.raw.append(self.index(self.generator.labels,width,value))
        else:stage='canonical_draw'
        self.draws.append(dict(stage=stage,width=width,value=value));return value

    def f3_trace(self):
        require(self.full,'FULL_TRACE_REQUIRED')
        return dict(draws=list(self.draws),raw_word=tuple(self.raw),degree=self.degree,
            bits_consumed=sum(x['width'] for x in self.draws),draw_count=len(self.draws),
            completed_bit_map_calls_by_stage={s:sum(x['stage']==s for x in self.draws)
                for s in ('order','label','acceptance','child')},extra_random_draws_for_audit=0,
            fresh_IID_verified=False)

    def consumption(self):
        p=self.provider
        return dict(domain=p.domain,counter_before=self.counter_before,counter_after=p.counter,
            bits_consumed_before=self.bits_before,bits_consumed_after=p.consumed_bits,
            draws=list(p.trace[self.start:]),failure_position=p.failure_position,
            artificial_only=True,real_variable_consumption='NOT_ESTABLISHED')
