"""Bound ARTIFICIAL endpoint and bits; scientific factories cannot be supplied."""
from fractions import Fraction as F
from hashlib import sha256
import os
from . import g6_w2c_b3_parallel_scheduler as P
D=P.D

class Bits:
    def __init__(self,arm,axis):self.seed=('ARTIFICIAL_R1_PARALLEL/'+arm+'/'+axis).encode();self.counter=self.consumed_bits=0
    def bits(self,n):
        value=available=0
        while available<n:
            raw=sha256(self.seed+self.counter.to_bytes(8,'big')).digest();self.counter+=1
            value=(value<<256)|int.from_bytes(raw,'big');available+=256
        self.consumed_bits+=n;return value>>(available-n)

def bits_factory(arm,axis):return Bits(arm,axis)

class Endpoint:
    def __init__(self,task,settings):self.task,self.settings=task,settings;self.value={'artificial':True,'state':[F(i+1,i+31)for i in range(32)]}
    def shared_identity(self):return D.identity(self.value)
    def fault(self,stage,tid=None):
        return any(pair==self.task['pair_index'] and phase==self.task['phase_index'] and where==stage and (tid is None or tid.endswith('/X/'+str(trial))) for pair,phase,where,trial in self.settings.faults)
    def sample(self,bits,trial_id):
        if self.fault('exit',trial_id):os._exit(88)
        if self.fault('sample',trial_id):raise RuntimeError('ARTIFICIAL_SAMPLE_FAILURE')
        token=bits.bits(160);x=token
        for j in range(8):x=(x*6364136223846793005+j)&((1<<192)-1)
        kind=('ZERO_R','ZERO_C','EVENT','EVENT')[token%4]
        if kind!='EVENT':return D.Outcome(kind,evidence={'artificial':True,'token':x})
        ratio=F((1<<24)+(x&8191)+1,(1<<26)+57);weight=ratio*F(7,11)
        raw=(0,1)if self.task['arm']=='full_return'else(0,0);word=raw if self.task['arm']=='full_return'else()
        e=dict(raw_word=raw,word=word,child=2,phase_i_power=2,circuit_time_word=tuple(reversed(word)),ratio=ratio,
            proposal=F(1,2),coefficient=weight*F(1,2),weight=weight,provider_calls={0:1,1:1,2:2}if word else{2:2})
        if self.fault('after_event',trial_id):e['phase_i_power']=99
        return D.Outcome('EVENT',e,{'artificial':True,'token':x,'evidence':weight})
    def angle_for(self,result):
        if self.fault('angle'):raise RuntimeError('ARTIFICIAL_ANGLE_FAILURE')
        z=result.event['ratio'];partial=sum(((-1)**j*z**(2*j+1)/(2*j+1)for j in range(4)),F(0))
        return dict(artificial_not_a_real_enclosure=True,partial=partial,remainder=abs(z**9/9),event_requery_surrogate=D.identity(result.event))

class ArtificialFactory:
    artificial=True
    def __init__(self,runtime_manifest):self.runtime_manifest=runtime_manifest
    def __call__(self,task,settings):return Endpoint(task,settings)
