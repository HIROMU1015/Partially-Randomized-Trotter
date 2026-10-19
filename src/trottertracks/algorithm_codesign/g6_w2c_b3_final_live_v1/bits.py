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
            artificial_only=type(p) is ArtificialBits,real_variable_consumption='NOT_ESTABLISHED')


def validate_consumption(data,arm,axis,*,namespace,artificial):
    """Validate recorded positions only; does not generate scientific bits."""
    require(data['domain']==namespace+'/'+arm+'/'+axis and data['artificial_only'] is artificial,'CONSUMPTION_DOMAIN')
    position=data['counter_before']
    require(type(position) is int and position>=0,'CONSUMPTION_POSITION')
    for row in data['draws']:
        require(row['width']==160 and type(row['width']) is int and row['counter_before']==position
            and row['counter_after']==position+1 and row['discarded_bits']==96,'CONSUMPTION_DRAW')
        require(type(row['value_hex']) is str and 0<=int(row['value_hex'],16)<1<<160,'CONSUMPTION_INTEGER')
        position+=1
    require(data['counter_after']==position and data['bits_consumed_before']==160*data['counter_before']
        and data['bits_consumed_after']==160*position,'CONSUMPTION_TOTAL')
    return S.digest(data)


class CounterRecorder:
    """Observe a delegated G7-shaped counter. No extra hash/draw is performed."""
    def __init__(self,provider,domain):
        require(provider.counter==provider.consumed_bits==0,'FRESH_BITS_REQUIRED')
        self.provider=provider;self.domain=domain;self.trace=[]
        self.failure_position=None;self.stopped=False
    @property
    def counter(self):return self.provider.counter
    @property
    def consumed_bits(self):return self.provider.consumed_bits
    def bits(self,width):
        require(not self.stopped,'BITS_STOPPED')
        before,bits_before=self.counter,self.consumed_bits
        try:
            require(type(width) is int and width==160,'FIXED_REQUEST_WIDTH')
            require(type(before) is int and 0<=before<2**64,'COUNTER_RANGE')
            value=self.provider.bits(width)
            require(type(value) is int and 0<=value<1<<width,'BIT_INTEGER')
            require(self.counter==before+1 and self.consumed_bits==bits_before+160,'DELEGATED_CONSUMPTION')
            self.trace.append(dict(width=160,counter_before=before,counter_after=self.counter,
                discarded_bits=96,value_hex=hex(value)))
            return value
        except BaseException:
            self.stopped=True;self.failure_position=dict(draw_index=len(self.trace),counter=self.counter,
                consumed_bits=self.consumed_bits)
            raise


class ScientificBits(CounterRecorder):
    def __init__(self,arm,axis,c,m,permit):
        from .gate import require_live
        require_live(permit,c,m)
        require(arm in S.ARMS and axis in ('X','Y') and m['seed_namespace']==S.SEED,'SCIENCE_BITS_DOMAIN')
        from .resolver import resolve
        # Resolver checks source location/hash/signature; only after admission.
        symbols=resolve(c,m,permit)
        domain=m['seed_namespace']+'/'+arm+'/'+axis
        super().__init__(symbols.g7.DeterministicBits(domain),domain)
        self.c,self.m,self.permit=c,m,permit
    def bits(self,width):
        from .gate import require_live
        require_live(self.permit,self.c,self.m)
        return super().bits(width)


class ScientificBitsFactory:
    def __init__(self,c,m,permit):
        from .gate import require_live
        require_live(permit,c,m);self.c,self.m,self.permit=c,m,permit
    def __call__(self,arm,axis):return ScientificBits(arm,axis,self.c,self.m,self.permit)
