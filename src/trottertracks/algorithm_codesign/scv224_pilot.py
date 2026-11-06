"""Dependency-injected 224-shot paired pilot; no activated science launcher.

Only fresh, caller-supplied services/streams/pool may be connected later. This
module creates none of them and never imports old results or science caches.
"""
from dataclasses import dataclass, replace
from fractions import Fraction as F
from pathlib import Path
import hashlib
import json
import time

from . import a1_control_variates as cv
from . import a1_n3_connection as n3
from . import a1_n3_campaign as legacy
from .a1_n3_managed_runtime import durable_new
from . import scv_scalar as scalar

PURPOSE='O2_SCV224_PAIRED_DIAGNOSTIC'
CAMPAIGN_CANDIDATE='O2-SCV224-H4-m0-q1-20261107-v1'
SEED_CANDIDATE=20261109  # proposal, never used/generated in preparation
EVAL_COUNTS=(4,64,4,4,4)
CALL_CAPS=dict(exact_engine=224,Born=224,selected_full_build=448,compile=224)


def domains():
    ds=[]
    for axis in ('X','Y'):
        ds.append(n3.Domain('A1',axis,'train','1'))
        ds.extend(n3.Domain('B1-OL',axis,'prod',h) for h in scalar.CELLS)
    return tuple(sorted(ds,key=lambda d:d.token))


def preparation_plan():
    return dict(status='PREPARATION_ONLY',scientific_execution_enabled=False,
        purpose=PURPOSE,campaign_candidate=CAMPAIGN_CANDIDATE,seed_candidate=SEED_CANDIDATE,
        train_per_axis=32,evaluation_per_axis=EVAL_COUNTS,total_physical_shots=224,
        paired_evaluation_shots=160,range_guard='4',domain_tokens=[d.token for d in domains()],
        child_streams=24,spawn_rule='lexical domain token, path child then bit child',
        path_words_per_trial=64,bit_words_per_trial=1,total_words_cap=14560,
        call_caps=CALL_CAPS,retry=0,precision='NOT_CERTIFIED',
        pending=['224-shot science scope and settings adoption','campaign identity and seed/spawn adoption',
                 'real metadata-only group/mean evaluation','activation SOURCE/environment/input lock',
                 'certified service, fresh streams and qualified manager activation'])


@dataclass(frozen=True)
class Envelope:
    campaign: str
    purpose: str
    metadata_identity: str
    control_identity: str
    axis: str
    stage: str
    rows: tuple
    costs: tuple
    cache_hits: object = None


def source_fields(campaign,control):
    return dict(campaign=campaign,purpose=PURPOSE,metadata_identity=control.metadata.identity,
                control_identity=control.identity)


def check_source(out,campaign,control):
    out=Path(out)
    if (out/'STOP.json').exists() or (out/'STOP_REQUESTED.json').exists():
        raise ValueError('STOP exists; resume forbidden')
    data=(out/'SOURCE.json').read_bytes();record=json.loads(data)
    if any(record.get(k)!=v for k,v in source_fields(campaign,control).items()):
        raise ValueError('fresh SCV SOURCE identity mismatch')
    return hashlib.sha256(data).hexdigest()


class LegacyCollector:
    """Reuse fixed group/worker/bit-before-price/journal; never legacy.run/learn.

    A1 is an acquisition-domain tag for conditional h1 train ONLY. The estimator
    and purpose are SCV. Evaluation uses B1-OL tags and a single physical dataset.
    """
    def __init__(self,raw,campaign,control,runtime):
        self.raw,self.campaign,self.control,self.runtime=raw,campaign,control,runtime
        if Path(runtime.out).resolve()!=Path(raw.out).resolve():raise ValueError('runtime output mismatch')
        self.source_hash=check_source(raw.out,campaign,control)
        if legacy._STOP_EVENT is None or runtime.stop is not legacy._STOP_EVENT:
            raise ValueError('shared lifecycle STOP event required')
        if raw.sampler is not control.metadata.sampler or raw.trials or raw.costs:
            raise ValueError('fresh bound sampler and empty acquisition required')
        if list(Path(raw.out).glob('worker_*.jsonl')) or any(p.is_file() for p in (Path(raw.out)/'cache').rglob('*')):
            raise ValueError('old journal/cache forbidden')

    def collect(self,axis,stage,quotas):
        if check_source(self.raw.out,self.campaign,self.control)!=self.source_hash:
            raise ValueError('SOURCE changed')
        method='A1' if stage=='train' else 'B1-OL'
        before,cost_before=len(self.raw.trials),len(self.raw.costs)
        rows,_=self.raw.group(method,axis,stage,quotas)
        new=self.raw.trials[before:]
        if any((r['domain'].method,r['domain'].axis,r['domain'].stage)!=(method,axis,stage) for r in new):
            raise ValueError('foreign acquisition domain')
        return Envelope(self.campaign,PURPOSE,self.control.metadata.identity,self.control.identity,
            axis,stage,tuple(rows),tuple(self.raw.costs[cost_before:]),sum(r['cache_hit'] for r in new))

    def request_stop(self,reason):self.runtime.request_stop(reason)

    def finalize(self):
        try:
            self.raw.pool.shutdown(wait=True,cancel_futures=True)
        finally:
            try:
                result=legacy.reconcile(self.raw.out)  # all durable tails, including failed groups
            finally:self.runtime.classify()
        counts=json.loads((Path(self.raw.out)/'CALL_COUNTS.json').read_text())
        for key,cap in CALL_CAPS.items():
            if counts['actual_worker_calls'].get(key,0)>cap:raise ValueError('scientific call cap exceeded')
        if counts['uniform53_words']>14560 or counts['trial_planned']>224:
            raise ValueError('word/trial cap exceeded')
        if any(result[m]['bits_saved'] for m in ('B0','B1-O')):
            raise ValueError('foreign method in SCV journal')
        return dict(authoritative_worker_reconciliation='ACCOUNTING.json',
            calls='CALL_COUNTS.json',attempts='ATTEMPTS.json',acquisition_tags_only=True,
            legacy_classical_timers=self.raw.classical,
            note='A1 tags denote h1 training, not the previous 3-control estimator')


def connect_legacy_pipeline(out,campaign,control,services,streams,pool,workers,runtime):
    """Later caller supplies already-qualified dependencies; no loader/RNG/pool."""
    raw=legacy.Campaign(out,services,streams,pool,workers)
    return LegacyCollector(raw,campaign,control,runtime)


def cost_summary(envelopes):
    """Physical spend once; deployment references overlap by explicit design."""
    costs=tuple(c for e in envelopes for c in e.costs)
    if not costs:
        return dict(physical_shots_returned=0,physical_RZ='UNKNOWN',cost_complete=False)
    accounted=cv.account_costs(costs)
    nbits=sum(len(b.records) for e in envelopes for _,b in e.rows)
    stages=accounted['quantum_by_stage'];total=accounted['quantum_total'][2]
    if len(costs)!=nbits:total=None
    return dict(physical_shots_returned=sum(len(b.records) for e in envelopes for _,b in e.rows),
        priced_records=len(costs),physical_RZ=total,cost_complete=(total is not None),
        known_RZ_sum=sum((c.quantum[2] for c in costs if c.quantum[2] is not None),F(0)),
        B1_OL_reference_evaluation_RZ=stages['prod'][2],
        SCV_training_plus_evaluation_RZ=total,train_RZ=stages['train'][2],
        cache_hits_returned=sum(e.cache_hits for e in envelopes) if all(type(e.cache_hits)is int for e in envelopes) else None,
        cache_hits_are_charged=True,reference_costs_are_not_additive=True,
        expected_cost_guarantee='UNKNOWN',
        classical_costs='Separate measured timers; legacy TrialCost zeros are not observations',
        worker_peak_RSS_bytes=max(c.memory_bytes for c in costs))


class Driver:
    """One dependency-injected campaign, fixed quotas, no recovery/refit path."""
    def __init__(self,out,campaign,control,scopes,collector):
        self.out=Path(out);self.campaign=campaign;self.control=control
        self.source_hash=check_source(out,campaign,control)
        if set(scopes)!={'X','Y'} or any(s.axis!=a or s.weight_bound!=1 or s.law_id!=control.metadata.law_sha256 for a,s in scopes.items()):
            raise ValueError('X/Y fixed-law scope required')
        if len({(s.target_id,s.circuit_id,s.round_id) for s in scopes.values()})!=1:
            raise ValueError('same target/circuit/round required')
        durable_new(self.out/'SCV_INVOCATION.json',source_fields(campaign,control))
        if not F(1,8)<=control.means[1]<=F(7,8):raise ValueError('selected h1 balance guard')
        self.scopes,self.collector=scopes,collector
        self.envelopes=[];self.seen=set();self.claimed=set();self.models={};self.results={};self.timings={};self.started=False

    def acquire(self,axis,stage):
        claim=(axis,stage)
        if axis not in ('X','Y') or stage not in ('train','prod') or claim in self.claimed:
            raise ValueError('outside single fixed acquisition/no retry')
        if check_source(self.out,self.campaign,self.control)!=self.source_hash:raise ValueError('SOURCE changed')
        quotas=(('1',32),) if stage=='train' else tuple(zip(scalar.CELLS,EVAL_COUNTS))
        self.claimed.add(claim)
        legacy.append(self.out/'SCV_JOURNAL.jsonl',dict(event='ACQUIRE_START',axis=axis,stage=stage,quotas=quotas))
        started=time.perf_counter()
        e=self.collector.collect(axis,stage,quotas)
        self.timings[axis+'_'+stage+'_wall_seconds']=time.perf_counter()-started
        # Retain received measurements before even validating prices/provenance.
        legacy.save(self.out/f'RECEIVED_{axis}_{stage}.json',e)
        self.envelopes.append(e)
        if check_source(self.out,self.campaign,self.control)!=self.source_hash:raise ValueError('SOURCE changed during acquisition')
        if (e.campaign,e.purpose,e.metadata_identity,e.control_identity,e.axis,e.stage)!=(
            self.campaign,PURPOSE,self.control.metadata.identity,self.control.identity,axis,stage):
            raise ValueError('envelope identity mismatch; reference data forbidden')
        if tuple(h for h,_ in e.rows)!=tuple(h for h,_ in quotas):raise ValueError('all positive cells/order required')
        expected={}
        for (cell,batch),(_,n) in zip(e.rows,quotas):
            if batch.scope!=self.scopes[axis] or batch.stage!=stage or len(batch.records)!=n:
                raise ValueError('scope/stage/quota mismatch')
            original=n3.Domain('A1',axis,'train',cell).token if stage=='train' else n3.Domain('B1-OL',axis,stage,cell).token
            if any(r.stream_id!=original for r in batch.records):raise ValueError('independent stream identity mismatch')
            projected=replace(batch,records=tuple(replace(r,stream_id=n3.Domain('B1-OL',axis,stage,cell).token) for r in batch.records))
            n3.validate_batch(self.control.metadata.sampler,projected,'B1-OL',cell,stage)
            for r in batch.records:
                if r.key in self.seen:raise ValueError('duplicate measurement')
                self.seen.add(r.key);expected[r.key]=batch.scope
        cv.account_costs(e.costs)
        if len(e.costs)!=len(expected) or {(c.stream_id,c.trial_id) for c in e.costs}!=set(expected):
            raise ValueError('missing/duplicate price; cost UNKNOWN')
        for c in e.costs:
            if c.scope!=expected[(c.stream_id,c.trial_id)] or c.stage!=stage or c.quantum[2] is None or F(c.quantum[2]).denominator!=1:
                raise ValueError('price scope/stage/missing RZ; cost UNKNOWN')
        legacy.append(self.out/'SCV_JOURNAL.jsonl',dict(event='ACQUIRE_RETURN',axis=axis,stage=stage,n=len(expected)))
        return e.rows

    def run(self):
        if self.started:raise ValueError('single run only; no retry')
        self.started=True
        status='STOPPED';reason='UNKNOWN';finalizer=None
        try:
            legacy.save(self.out/'CONTROL_FROZEN.json',self.control.snapshot())
            for axis in ('X','Y'):
                training=self.acquire(axis,'train')[0][1]
                started=time.perf_counter()
                self.models[axis]=scalar.fit_cells(self.control,training,
                    sink=lambda info,a=axis:legacy.save(self.out/f'{a}_FIT.json',info))
                self.timings[axis+'_fit_seconds']=time.perf_counter()-started
            # Both independent fits are frozen before any fresh evaluation.
            for axis in ('X','Y'):
                rows=self.acquire(axis,'prod');started=time.perf_counter()
                result=scalar.paired_evaluate(self.models[axis],rows)
                self.timings[axis+'_aggregation_seconds']=time.perf_counter()-started
                self.results[axis]=result;legacy.save(self.out/f'{axis}_PAIRED_RESULT.json',result)
            for axis,result in self.results.items():
                ledger=cost_summary([e for e in self.envelopes if e.axis==axis])
                result['realized_RZ_reference']=ledger
                result['exploratory_variance_times_realized_RZ']=dict(
                    B1_OL=result['B1_OL']['estimated_variance_of_mean']*ledger['B1_OL_reference_evaluation_RZ'],
                    SCV=result['SCV']['estimated_variance_of_mean']*ledger['SCV_training_plus_evaluation_RZ'],
                    interpretation='point diagnostics only; no expected cost guarantee')
                legacy.save(self.out/f'{axis}_PAIRED_RESULT.json',result)
            status='COMPLETE';reason='224 physical shots; paired exploratory diagnostic only'
        except BaseException as exc:
            reason=type(exc).__name__+': '+str(exc)
        finally:
            # Receipt and event before pool drain, on success as well as failure.
            if not (self.out/'STOP_REQUESTED.json').exists():
                durable_new(self.out/'STOP_REQUESTED.json',dict(reason=reason,retry=False))
            try:self.collector.request_stop(reason)
            except BaseException as exc:
                status='STOPPED';reason+='; stop hook '+type(exc).__name__+': '+str(exc)
            try:finalizer=self.collector.finalize()
            except BaseException as exc:
                status='STOPPED';reason+='; finalizer '+type(exc).__name__+': '+str(exc)
            try:costs=cost_summary(self.envelopes)
            except Exception as exc:costs=dict(status='UNKNOWN',reason=str(exc))
            legacy.save(self.out/'SCV_ACCOUNTING.json',costs)
            legacy.save(self.out/'SCV_CLASSICAL.json',dict(measured_driver_timings=self.timings,
                finalizer=finalizer,unmeasured_costs='UNKNOWN; metadata setup and manager costs require activation timers'))
            result=dict(status=status,reason=reason,axes=self.results,campaign=self.campaign,
                precision='NOT_CERTIFIED',expected_cost_guarantee='UNKNOWN',science_retry=False)
            legacy.save(self.out/'SCV_RESULT.json',result)
            durable_new(self.out/'STOP.json',dict(status=status,reason=reason,mandatory_STOP=True,retry=False))
        return result
