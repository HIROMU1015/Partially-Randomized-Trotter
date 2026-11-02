"""R1-bound, premeasurement tail diagnostic. Frozen samplers; no +/-1 observations."""
from collections import Counter
from fractions import Fraction as F
import math
from functools import reduce
from operator import mul
import time
from support import (MODEL, R1, canonical, digest, require, sha, prior_adapter,
                     Timers, save_blob)
from trottertracks.algorithm_codesign.g6_f3_contract import Evidence, fingerprint
from trottertracks.algorithm_codesign.g6_f3_generator import prepare
from trottertracks.algorithm_codesign.g6_f3_precision import enclose_atan
from trottertracks.algorithm_codesign.g7_generator import make_generator
from trottertracks.algorithm_codesign.g9_p5 import P5Closed
from trottertracks.algorithm_codesign.return_aggregation import reduce_word

METHODS = ('ordinary','full_return','closed_P5_full')
PRECISION = dict(root_bits=4096, probability_bits=2048, eta=F(1,10**12), rho=F(1,10**12),
                 angle_tau=F(1,10**12), angle_root_bits=256, angle_terms=64)

class Bits:
    """SHA256 counter keyed by method/trial/microstep, not a demonstrated IID source."""
    def __init__(self, method, trial, microstep):
        self.key = canonical(dict(schema='C1_5A_BITS_V1', seed=20261021,
                                  method=method, trial=trial, microstep=microstep))
        self.draws = []
        self.counter = 0

    def bits(self, width):
        require(type(width) is int and width > 0, 'INVALID_BIT_WIDTH')
        count = (width+255)//256
        raw = b''.join(__import__('hashlib').sha256(self.key + b'\0' +
            self.counter.to_bytes(8,'big') + j.to_bytes(4,'big')).digest() for j in range(count))
        value = int.from_bytes(raw,'big') >> (8*len(raw)-width)
        self.draws.append(dict(draw=self.counter,width=width,value=value))
        self.counter += 1
        return value

def load_model():
    import json
    core = json.loads((R1/'result/exact_model_core.json').read_text())
    require(sha(canonical(core)) == MODEL, 'R1_MODEL_SHA')
    require(core['partition_hash']=='16b35c1dde1a6807d39c733defd87e315e4ac7317a831b240d0f7214e744a834', 'R1_PARTITION')
    old = prior_adapter()
    restored, rows, p, registry, saved = old.restore_r1()
    require(restored == core and len(rows)==324 and len(saved)==13, 'LABEL_BASIS_COUNTS')
    adapter = old.DFAdapter(rows,p,registry,8,MODEL)
    scalar = F(core['constant_exact_from_binary64']) + F(core['tail_extracted_identity_exact_from_binary64'])
    return dict(core=core,p=p,x=F(core['lambda_exact'])/5,adapter=adapter,scalar=scalar,
                labels_id=adapter.labels_id,scope='H4_R1_EXPLORATORY')

class Sampler:
    def __init__(self, method, model, contract_id, directory, precision=None):
        require(method in METHODS, 'METHOD')
        self.method, self.model, self.contract_id, self.directory = method, model, contract_id, directory
        self.precision = dict(precision or PRECISION)
        args={k:v for k,v in self.precision.items() if not k.startswith('angle_')}
        self.session = prepare(model['p'],model['x'],5,sigma=1,**self.precision) if method=='full_return' else None
        self.g = self.session._generator if self.session else (
            P5Closed(model['p'],model['x'],**args) if method=='closed_P5_full' else
            make_generator(model['p'],model['x'],5,method,**args))
        require(self.g.p==model['p'] and self.g.x==model['x'] and self.g.m==5, 'COMMON_FINITE_TARGET')
        self.angles={}
        self.angle_hits=self.angle_misses=0
        self.event_ids=set()
        self.event_cache_hits=self.event_cache_misses=0
        self.calls=Counter()
        self.failed=None

    def sample(self, trial, microstep, supplied_bits=None):
        require(self.failed is None, 'SAMPLER_TERMINAL')
        require(not getattr(self,'abort',None) or not self.abort.is_set(),'JOB_ABORTED')
        bits=supplied_bits if supplied_bits is not None else Bits(self.method,trial,microstep)
        self.calls['sampler_calls']+=1
        tid=self.method+':'+str(trial)+':'+str(microstep)
        try:
            if self.session:
                record=self.session.sample(bits,trial_id=tid)
                d=record.read()
                event=d.get('event')
                header={k:v for k,v in d.items() if k not in ('event','trace')}
                trace=d['trace']
                f3_digest=record.digest
                if d.get('reason')!='RAW_NONREDUCED':
                    self.calls['Full_local_parent_packet_queries']+=1
            else:
                event=self.g.sample(bits)
                require(event is not None,'CANONICAL_UNEXPECTED_ZERO')
                header=dict(schema='C1_5A_CANONICAL_TRIAL_V1',kind='EVENT',trial_id=tid,
                            certification='UNCERTIFIED_EXPLORATORY')
                trace=None
                f3_digest=None
            if event is not None:
                require(all(type(event[k]) is F and event[k]>0 for k in
                    ('proposal','coefficient','weight','ratio')),'POSITIVE_EXACT_EVENT')
                require(event['proposal']*event['weight']==event['coefficient'],'ACTUAL_PI_WEIGHT')
                raw,word=event['raw_word'],event['word']
                require(len(raw) in (0,2,4) and reduce_word(raw)==word and
                    event['phase_i_power']==len(raw)%4 and event['circuit_time_word']==tuple(reversed(word)),
                    'RAW_DEGREE_PHASE_ORDER')
                require(all(type(i) is int and 0<=i<len(self.model['p']) for i in (*raw,event['child'])), 'LABEL_RANGE')
                if self.method in ('full_return','closed_P5_full'):
                    require(raw==word and (not word or event['child']!=word[0]),'RETURN_LEGALITY')
                event_record=Evidence.create(dict(contract_id=self.contract_id,model_id=self.model['adapter'].model_id,
                    labels_id=self.model['labels_id'],method=self.method,event=event))
                eid=event_record.digest
                if eid in self.event_ids:
                    self.event_cache_hits+=1
                else:
                    self.event_cache_misses+=1
                    save_blob(self.directory/'events',event_record)
                    self.event_ids.add(eid)
                if self.method=='closed_P5_full' and len(word)==2:
                    self.calls['P5_level2_queries_sample_plus_event']+=2
                if self.method=='closed_P5_full' and len(word)==4:
                    self.calls['P5_word_conditional_queries_sample_plus_event']+=6
            else:
                require(header['kind']=='ZERO' and header['is_valid_zero_trial'] and
                        header['reason'] in ('RAW_NONREDUCED','COIN_REJECT'),'VALID_ZERO')
                eid=None
            return dict(event=event,event_id=eid,header=header,F3_record_digest=f3_digest,F3_trace=trace,
                        bits_key_id=sha(bits.key) if hasattr(bits,'key') else 'ARTIFICIAL',bits_draws=bits.draws)
        except Exception as exc:
            self.failed=type(exc).__name__+':'+str(exc)
            raise

    def angle(self, event, timers):
        ratio=event['ratio']
        with timers.stage('angle_lookup'):
            cached=self.angles.get(ratio)
        if cached is not None:
            self.angle_hits+=1
            return cached
        self.angle_misses+=1
        with timers.stage('angle_query_and_Evidence'):
            rec=enclose_atan(ratio,tau=self.precision['angle_tau'],root_bits=self.precision['angle_root_bits'],
                terms=self.precision['angle_terms'],sigma=1,
                context=dict(contract_id=self.contract_id,kind='shared mathematical angle; not native provider'))
            angle=rec.read()['signed_representative']
            representative=float(angle)
            conversion_error=abs(F.from_float(representative)-angle)
            packet=Evidence.create(dict(angle=rec.to_dict(),representative_binary64=representative.hex(),
                conversion_error_exact=conversion_error,provider_native_certified=False))
            aid,_=save_blob(self.directory/'angles',packet)
        cached=(representative,aid)
        self.angles[ratio]=cached
        return cached

def logical_resources(circuit, events):
    counts=Counter()
    controls=0
    for item in circuit.data:
        size=len(item.qubits)
        counts['1Q' if size==1 else '2Q' if size==2 else 'multi_qubit']+=1
        controls+=int(getattr(item.operation,'num_ctrl_qubits',0)>0)
    return dict(scope='TAIL_ONLY_PLUS_SCALAR',signed_Q=sum(len(e['word']) for e in events),
        child_rotations=len(events),controlled_gate_instructions=controls,
        logical_1Q=counts['1Q'],logical_2Q=counts['2Q'],logical_multi_qubit=counts['multi_qubit'],
        instructions=len(circuit.data),depth=circuit.depth(),workspace_qubits=1,
        system_qubits=circuit.num_qubits-1,
        total_qubits=circuit.num_qubits,native_T_CX='UNKNOWN',deterministic_halves='UNKNOWN_NOT_BUILT')

def build_tail(adapter, scalar, events, rotations):
    from qiskit import QuantumCircuit
    require(len(events)==len(rotations)==4,'FOUR_PRESELECTED_EVENTS')
    circuit=QuantumCircuit(adapter.n+1)
    for event,angle in zip(events,rotations):
        if event['phase_i_power']==2:
            circuit.z(adapter.n)
        for label in event['circuit_time_word']:
            adapter.append_q(circuit,label)
        adapter.append_q(circuit,event['child'],rotation=angle)
    circuit.p(-float(F(4,5)*scalar),adapter.n)
    return circuit

def generate_path(sampler, trial, timers):
    steps=[]
    # Always four source samples, including after a legal zero; no method-specific early exit.
    for k in range(4):
        with timers.stage('sampling_checks_event_Evidence_storage'):
            steps.append(sampler.sample(trial,k))
    zero=any(step['event'] is None for step in steps)
    with timers.stage('weight_product'):
        weight=F(0) if zero else reduce(mul,(s['event']['weight'] for s in steps),F(1))
        proposal=None if zero else reduce(mul,(s['event']['proposal'] for s in steps),F(1))
    resources=None
    if not zero:
        rotations=[sampler.angle(s['event'],timers) for s in steps]
        with timers.stage('logical_build'):
            circuit=build_tail(sampler.model['adapter'],sampler.model['scalar'],
                [s['event'] for s in steps],[a for a,_ in rotations])
            for step,(_,aid) in zip(steps,rotations):
                step['angle_id']=aid
        with timers.stage('logical_accounting'):
            resources=logical_resources(circuit,[s['event'] for s in steps])
    with timers.stage('path_Evidence'):
        clean=[{k:v for k,v in s.items() if k!='event'} for s in steps]
        exact=Evidence.create(dict(schema='C1_5A_PREMEASUREMENT_PATH_V1',contract_id=sampler.contract_id,
            method=sampler.method,trial=trial,model_id=sampler.model['adapter'].model_id,
            labels_id=sampler.model['labels_id'],microsteps=clean,kind='ZERO' if zero else 'ACCEPTED',
            weight=weight,proposal=proposal,measurement_performed=False,coherent_signal_estimator=None,
            all_four_steps_generated=True,zero_quantum_calls=zero,
            scalar_phase_exact=F(4,5)*sampler.model['scalar']))
        exact.read()
        w=float(weight)
        require(math.isfinite(w) and math.isfinite(w*w),'NONFINITE_OBSERVED_WEIGHT')
    return dict(exact=exact.to_dict(),exact_digest=exact.digest,logical=resources,
                observed_weight_binary64=w,observed_weight_square_binary64=w*w,
                zero_reasons=[s['header']['reason'] for s in steps if s['event'] is None],
                reduced_lengths=[None if s['event'] is None else len(s['event']['word']) for s in steps],
                raw_degrees=[(s['F3_trace']['degree'] if s['event'] is None else len(s['event']['raw_word'])) for s in steps])
