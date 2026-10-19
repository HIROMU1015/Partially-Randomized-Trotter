"""Artificial-only counterparts of the statically named ports."""
from dataclasses import dataclass
from fractions import Fraction as F
import hashlib
import importlib.abc
import sys
from .. import g6_w2c_b3_parallel_driver as D
from .static import ARMS, SCIENCE, SEED, Rejected, digest, need

class BlockScience(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.rsplit('.', 1)[-1] in SCIENCE:
            raise Rejected('SCIENTIFIC_IMPORT_BLOCKED_BEFORE_LOAD:'+fullname)
        return None

def install_guard():
    need(not [n for n in sys.modules if n.rsplit('.',1)[-1] in SCIENCE], 'SCIENCE_ALREADY_LOADED')
    if not any(type(x) is BlockScience for x in sys.meta_path):
        sys.meta_path.insert(0, BlockScience())

def forbidden_imports():
    return sorted(n for n in sys.modules if n.rsplit('.',1)[-1] in SCIENCE)

def artificial_domain(arm, axis, *, namespace=SEED):
    need(namespace == SEED, 'ARTIFICIAL_NAMESPACE_ONLY')
    need(arm in ARMS and axis in ('X','Y'), 'BIT_DOMAIN_IDENTITY')
    return namespace+'/'+arm+'/'+axis

class ArtificialBits:
    def __init__(self, arm, axis, *, namespace=SEED):
        self.domain = artificial_domain(arm,axis,namespace=namespace)
        self.counter = self.consumed_bits = 0
        self.trace = []

    def bits(self, n):
        need(type(n) is int and n == 160, 'FIXED_ARTIFICIAL_BIT_WIDTH')
        value = available = 0
        before = self.counter
        while available < n:
            raw = hashlib.sha256(self.domain.encode()+self.counter.to_bytes(8,'big')).digest()
            self.counter += 1
            value = (value << 256) | int.from_bytes(raw,'big')
            available += 256
        value >>= available-n
        self.consumed_bits += n
        self.trace.append(dict(width=n,counter_before=before,counter_after=self.counter,
                               value_hex=hex(value),discarded_bits=available-n))
        return value

def bits_factory(arm, axis):
    install_guard()
    return ArtificialBits(arm,axis)

@dataclass(frozen=True)
class TrialIdentity:
    run: str
    mode: str
    arm: str
    layer: str
    axis: str
    phase: str
    pass_name: str
    trial: int

    def global_id(self):
        need(self.mode in ('S','P1') and self.arm in ARMS and self.layer in ('A','B')
             and self.axis in ('X','Y') and self.phase in ('correctness','profile')
             and self.pass_name in ('baseline','instrumented') and type(self.trial) is int
             and self.trial >= 0 and bool(self.run), 'GLOBAL_IDENTITY')
        return '/'.join((self.run,self.mode,self.phase,self.pass_name,self.arm,
                         self.layer,self.axis,str(self.trial)))

    def seed_domain(self):
        self.global_id()
        return artificial_domain(self.arm,self.axis)

class ArtificialEndpoint:
    """An interface surrogate: no G7 law, F3 certificate or mathematical angle."""
    def __init__(self, task, settings=None, *, run='ARTIFICIAL_PORT', fault=''):
        install_guard()
        need(type(task) is dict and task['arm'] in ARMS and task['layer'] in ('A','B'), 'ENDPOINT_CELL')
        self.task, self.run, self.fault = dict(task), run, fault
        self.stopped = False
        self.records = []
        self.last = None

    def shared_identity(self):
        return digest(dict(kind='ARTIFICIAL_API_PORT', arm=self.task['arm'],layer=self.task['layer']))

    def sample(self, bits, trial_id):
        need(not self.stopped, 'TERMINAL_ENDPOINT_NO_RETRY')
        try:
            return self._sample(bits, trial_id)
        except BaseException:
            self.stopped = True
            raise

    def _sample(self, bits, trial_id):
        need(type(bits) is ArtificialBits, 'EXACT_ARTIFICIAL_BITS_REQUIRED')
        fields = trial_id.split('/')
        need(len(fields)==6, 'LOCAL_TRIAL_SCHEMA')
        phase,pass_name,arm,layer,axis,trial = fields
        need(arm==self.task['arm'] and layer==self.task['layer'] and
             phase==self.task['phase'] and pass_name==self.task['pass_name'], 'REQUEST_CELL_PHASE_MISMATCH')
        ident = TrialIdentity(self.run,self.task['condition'],arm,layer,axis,phase,pass_name,int(trial))
        need(bits.domain==ident.seed_domain(), 'SEED_DOMAIN_MISMATCH')
        if self.fault=='generator':
            raise Rejected('ARTIFICIAL_GENERATOR_FAILURE')
        begin = len(bits.trace)
        token = bits.bits(160)
        kind = ('ZERO_R','ZERO_C','EVENT','EVENT')[token%4] if arm=='full_return' else 'EVENT'
        event = None
        if kind!='ZERO_R':
            token ^= bits.bits(160)
        if kind=='EVENT':
            if arm=='full_return':
                token ^= bits.bits(160)
            ratio = F((1<<24)+(token&8191)+1,(1<<26)+57)
            weight = ratio*F(7,11)
            raw = (0,1) if arm=='full_return' else (0,0)
            word = raw if arm=='full_return' else ()
            event = dict(raw_word=raw,word=word,child=2,ratio=ratio,weight=weight,
                         coefficient=weight/2,proposal=F(1,2),phase_i_power=2,
                         circuit_time_word=tuple(reversed(word)),
                         provider_calls={0:1,1:1,2:2} if word else {2:2})
        if self.fault=='audit':
            raise Rejected('ARTIFICIAL_AUDIT_FAILURE')
        trace = dict(schema='b3_ARTIFICIAL_consumption_v1',domain=bits.domain,
                     draws=[dict(x) for x in bits.trace[begin:]],
                     counter_after=bits.counter, bits_consumed=bits.consumed_bits,
                     artificial_only=True, real_consumption_equivalence=False)
        result = D.Outcome(kind,event,dict(artificial=True,consumption=trace))
        validate_result(result,ident,trace)
        self.last = result
        self.records.append(dict(global_trial_id=ident.global_id(), local_trial_id=trial_id,
                                 domain=bits.domain,consumption=trace,
                                 event_digest=D.identity(dict(kind=kind,event=event))))
        return result

    def angle_for(self, result):
        need(not self.stopped, 'TERMINAL_ENDPOINT_NO_RETRY')
        try:
            need(self.task['layer']=='B' and result is self.last and result.kind=='EVENT', 'ANGLE_EVENT_IDENTITY')
            if self.fault in ('angle','requery'):
                raise Rejected('ARTIFICIAL_ANGLE_REQUERY_FAILURE')
            return dict(artificial=True,not_a_mathematical_enclosure=True,
                        event_requery_surrogate=D.identity(result.event), actual_angle=None)
        except BaseException:
            self.stopped=True
            raise

def validate_result(result, ident, trace):
    need(type(result) is D.Outcome, 'RESULT_EXACT_TYPE')
    result.validate()
    ident.global_id()
    need(type(result.evidence) is dict and result.evidence.get('artificial') is True, 'RESULT_EVIDENCE_SCHEMA')
    need(trace==result.evidence.get('consumption'), 'CONSUMPTION_EVIDENCE_MISMATCH')
    need(trace.get('schema')=='b3_ARTIFICIAL_consumption_v1' and trace.get('domain')==ident.seed_domain(), 'CONSUMPTION_DOMAIN_SCHEMA')
    expected = (1 if result.kind=='ZERO_R' else 2 if result.kind=='ZERO_C' else 3) if ident.arm=='full_return' else 2
    need(result.kind=='EVENT' or ident.arm=='full_return', 'CANONICAL_CANNOT_RANDOM_ZERO')
    draws = trace.get('draws')
    need(type(draws) is list and len(draws)==expected, 'CONSUMPTION_COUNT')
    for i,draw in enumerate(draws):
        need(all(type(draw.get(k)) is int for k in ('width','counter_before','counter_after','discarded_bits')), 'CONSUMPTION_INTEGER_TYPE')
        need(set(draw)=={'width','counter_before','counter_after','value_hex','discarded_bits'} and
             draw['width']==160 and draw['discarded_bits']==96 and
             draw['counter_after']==draw['counter_before']+1 and
             (i==0 or draw['counter_before']==draws[i-1]['counter_after']), 'CONSUMPTION_TRACE')
        need(type(draw['value_hex']) is str and 0<=int(draw['value_hex'],16)<(1<<160), 'BIT_INTEGER')
        expected=hashlib.sha256(trace['domain'].encode()+draw['counter_before'].to_bytes(8,'big')).digest()
        need(hex(int.from_bytes(expected,'big')>>96)==draw['value_hex'],'ARTIFICIAL_CONSUMPTION_VALUE')
    need(type(trace['counter_after']) is int and type(trace['bits_consumed']) is int and
         trace['counter_after']==draws[-1]['counter_after'] and trace['bits_consumed']==160*trace['counter_after'], 'CONSUMPTION_TOTAL')
    if result.kind=='EVENT':
        event=result.event
        need(type(event) is dict and {'raw_word','word','child','ratio','weight','coefficient','proposal',
            'phase_i_power','circuit_time_word','provider_calls'} <= set(event), 'MISSING_EVENT_FIELD')
        need(all(type(event[k]) is F for k in ('ratio','weight','coefficient','proposal')), 'EXACT_EVENT_RATIONAL')
        need(event['proposal']*event['weight']==event['coefficient'], 'EVENT_WEIGHT_IDENTITY')
        # Fixed structural audit only; never evaluates a scientific generator.
        D.audit.event_witness(event,full=ident.arm=='full_return',degree_cap=5,labels=3)

class ArtificialFactory:
    artificial=True
    def __call__(self,task,settings):
        return ArtificialEndpoint(task,settings)
