"""Prepared, dependency-injected 752-shot pilot; no real work on import.

There is intentionally no activated scientific launcher. Real binding, RNG and
pool creation remain a later SOURCE/adoption step. LegacyCollector reuses the
frozen journal/bit-before-price worker pipeline when explicitly supplied later.
"""
from dataclasses import dataclass,replace
from fractions import Fraction as F
from pathlib import Path
from statistics import variance
import hashlib
import json
import time

from . import a1_n3_campaign as legacy
from . import a1_n3_connection as n3
from . import a1_control_variates as cv
from . import a1_n3_pilot_design as design
from .a1_n3_managed_runtime import durable_new

CAMPAIGN_CANDIDATE='A1-N3-PILOT-H4-m0-q1-20261106-v1'
SEED_CANDIDATE=20261108
OUTPUT_CANDIDATE='artifacts/o2_a1_new_target/2026-11-07/n3_pilot_752_run01'
PURPOSE='A1_N3_PILOT_DIAGNOSTIC'
PENDING=('campaign_identity_and_seed','32_domains_and_spawn_order','allocation_proxy_and_floor',
         'range_guard_4','diagnostic_classification','guard_first_axis_schedule',
         'scientific_execution_authorization_and_final_SOURCE')


def preparation_plan():
    return dict(status='PREPARATION_ONLY_NOT_SCIENCE_EXECUTION_SOURCE',purpose=PURPOSE,
        methods=['A1','B1-OL'],axes=['X','Y'],shot_counts=design.BUDGET.counts,total=design.BUDGET.total,
        campaign_candidate=CAMPAIGN_CANDIDATE,seed_candidate=SEED_CANDIDATE,
        output_candidate=OUTPUT_CANDIDATE,cache_candidate=OUTPUT_CANDIDATE+'/cache',
        pending=list(PENDING),scientific_execution_enabled=False,precision='NOT_CERTIFIED')


def pilot_domains():
    ds=[]
    for axis in ('X','Y'):
        for cell in map(str,range(5)):
            ds.extend((n3.Domain('A1',axis,'train',cell),n3.Domain('B1-OL',axis,'train',cell),n3.Domain('B1-OL',axis,'prod',cell)))
        ds.append(n3.Domain('A1',axis,'prod','all'))
    return tuple(sorted(ds,key=lambda d:d.token))


@dataclass(frozen=True)
class Envelope:
    campaign: str
    purpose: str
    method: str
    axis: str
    stage: str
    rows: tuple
    rz_by_cell: dict


class LegacyCollector:
    """Adapter for an explicitly supplied fresh legacy Campaign, no creation.

    No old data/cache importer. SOURCE campaign/purpose must already be bound
    by a future scientific launcher before any legacy group is acquired.
    """
    def __init__(self,raw,campaign,runtime):
        self.raw,self.campaign,self.runtime=raw,campaign,runtime
        if Path(runtime.out).resolve()!=Path(raw.out).resolve():raise ValueError('runtime output mismatch')
        self.source=Path(raw.out)/'SOURCE.json'
        record=json.loads(self.source.read_text())
        if record.get('campaign')!=campaign or record.get('purpose')!=PURPOSE:
            raise ValueError('fresh pilot SOURCE required; old campaign rejected')
        if legacy._STOP_EVENT is None or runtime.stop is not legacy._STOP_EVENT:
            raise ValueError('shared lifecycle STOP event required')
        if raw.trials or raw.costs or list(Path(raw.out).glob('worker_*.jsonl')):
            raise ValueError('old measurements cannot enter a new pilot collector')
        if any(p.is_file() for p in (Path(raw.out)/'cache').rglob('*')):
            raise ValueError('fresh independent pilot cache required')
        self.source_hash=hashlib.sha256(self.source.read_bytes()).hexdigest()

    def collect(self,method,axis,stage,quotas):
        if hashlib.sha256(self.source.read_bytes()).hexdigest()!=self.source_hash:
            raise ValueError('pilot SOURCE changed before acquisition')
        before=len(self.raw.trials)
        rows,_=self.raw.group(method,axis,stage,quotas)
        trials=self.raw.trials[before:];rz={key:[] for key,_ in quotas}
        for row in sorted(trials,key=lambda r:int(r['trial'])):
            if (row['domain'].method,row['domain'].axis,row['domain'].stage)!=(method,axis,stage):
                raise ValueError('legacy result domain mismatch')
            rz[row['domain'].cell].append(row['metrics']['rz_count'])
        return Envelope(self.campaign,PURPOSE,method,axis,stage,tuple(rows),{k:tuple(v) for k,v in rz.items()})

    def request_stop(self,reason):
        self.runtime.request_stop(reason)

    def finalize(self):
        # Stop receipt is persisted by request_stop before worker drain. Existing
        # reconcile then includes ALL durable tails, including failed groups.
        self.raw.pool.shutdown(wait=True,cancel_futures=True)
        result=legacy.reconcile(self.raw.out)
        if any(result[m]['bits_saved'] for m in ('B0','B1-O')):
            raise ValueError('foreign method in pilot journal')
        return {m:result[m] for m in ('A1','B1-OL')}


class PilotDriver:
    def __init__(self,out,campaign,sampler,scopes,collector,engine_bound=None):
        if len(sampler.bank.controls)!=3 or len(sampler.cells('B1-OL'))!=5:raise ValueError('fixed 3 controls and 5 cells required')
        if set(scopes)!={'X','Y'} or any(v.axis!=k or v.law_id!=sampler.bank.law_id for k,v in scopes.items()):raise ValueError('X/Y target-law scope mismatch')
        self.engine_bound=engine_bound
        self.out=Path(out);self.out.mkdir(parents=True,exist_ok=True)
        durable_new(self.out/'PILOT_DRIVER_INVOCATION.json',dict(campaign=campaign,purpose=PURPOSE,attempt=1))
        self.campaign=campaign;self.sampler=sampler
        self.scopes,self.collector=scopes,collector
        self.envelopes=[];self.seen=set();self.models={};self.probes={};self.quotas={};self.results={}

    def acquire(self,method,axis,stage,quotas):
        if method not in ('A1','B1-OL') or axis not in ('X','Y') or stage not in ('train','prod'):
            raise ValueError('outside pilot method/axis/stage scope')
        expected_total=(80 if method=='A1' else 40) if stage=='train' else 128
        if sum(n for _,n in quotas)!=expected_total:raise ValueError('outside fixed pilot quota')
        envelope=self.collector.collect(method,axis,stage,tuple(quotas))
        if (envelope.campaign,envelope.purpose,envelope.method,envelope.axis,envelope.stage)!=(self.campaign,PURPOSE,method,axis,stage):
            raise ValueError('campaign/purpose/domain mismatch; reference data rejected')
        rows=envelope.rows
        if tuple(k for k,_ in rows)!=tuple(k for k,_ in quotas):raise ValueError('positive cells omitted/reordered')
        for (key,batch),(_,count) in zip(rows,quotas):
            if batch.scope!=self.scopes[axis] or len(batch.records)!=count:raise ValueError('quota/target mismatch')
            if method=='A1' and stage=='train':
                # A1 keeps its own stream identity, but its training acquisition
                # law is B1-OL. Validate a metadata-only projection; retain the
                # original records for fitting, storage and disjointness.
                token=n3.Domain('A1',axis,'train',key).token
                if any(r.stream_id!=token for r in batch.records):raise ValueError('A1 train stream mismatch')
                projected=replace(batch,records=tuple(replace(r,stream_id=n3.Domain('B1-OL',axis,'train',key).token) for r in batch.records))
                n3.validate_batch(self.sampler,projected,'B1-OL',key,stage)
            else:n3.validate_batch(self.sampler,batch,method,key,stage)
            for record in batch.records:
                if record.key in self.seen:raise ValueError('measurement reused across stages/methods')
                self.seen.add(record.key)
        # Preserve received bits before requiring complete price metadata.
        self.envelopes.append(envelope)
        legacy.save(self.out/f'{method}_{axis}_{stage}_DATA.json',envelope)
        if set(envelope.rz_by_cell)!=set(k for k,_ in quotas):raise ValueError('price cells mismatch')
        for key,count in quotas:
            costs=envelope.rz_by_cell[key]
            if len(costs)!=count or any(type(v) is not int or v<0 for v in costs):
                raise ValueError('price missing/invalid; durable bits retained; cost UNKNOWN')
        return rows,envelope.rz_by_cell

    def learn(self,axis):
        rows,_=self.acquire('A1',axis,'train',[(k,16) for k,_ in self.sampler.cells('B1-OL')])
        training=n3.merge_training([batch for _,batch in rows])
        # Same existing fit/rank/range functions as legacy.learn; provenance
        # validation above must run before fitting the new reduced-budget data.
        import numpy as np
        model=cv.fit_and_freeze(self.sampler.bank,training)
        rank=int(np.linalg.matrix_rank([[1,*self.sampler.bank.values(r.events)] for r in training.records]))
        width=model.output_range.hi-model.output_range.lo
        legacy.save(self.out/f'A1_{axis}_FROZEN.json',dict(coefficients=model.coefficients,rank=rank,
            range=model.output_range,width=width,guard=4,training_count=len(training.records)))
        if rank!=4 or width>4:raise ValueError('rank/range guard; no refit or extra shots')
        self.models[axis]=model

    def probe(self,axis):
        rows,prices=self.acquire('B1-OL',axis,'train',[(k,8) for k,_ in self.sampler.cells('B1-OL')])
        variances=tuple(variance(tuple(r.raw for r in batch.records)) for _,batch in rows)
        means=tuple(F(sum(prices[k]),len(prices[k])) for k,_ in rows)
        masses=tuple(p for _,p in self.sampler.cells('B1-OL'))
        counts=design.allocate_evaluation(masses,variances,means)
        self.probes[axis]=tuple(batch for _,batch in rows);self.quotas[axis]=counts
        legacy.save(self.out/f'B1-OL_{axis}_ALLOCATION.json',dict(cells=self.sampler.cells('B1-OL'),
            sample_variances=variances,probe_mean_RZ=means,counts=counts,heuristic=True,confidence_bound=False))

    def evaluate(self,method,axis):
        counts=(128,) if method=='A1' else self.quotas[axis]
        rows,prices=self.acquire(method,axis,'prod',[(key,n) for (key,_),n in zip(self.sampler.cells(method),counts)])
        model=self.models[axis] if method=='A1' else None
        antecedents=self.probes[axis]+(() if method=='A1' else (self.models[axis].training,))
        mean=design.pilot_mean(self.sampler,method,self.scopes[axis],rows,counts,antecedents=antecedents,model=model)
        sample_variances=[];vm=F(0);zero=[]
        for (key,batch),(_,mass),count in zip(rows,self.sampler.cells(method),counts):
            values=model.apply(batch) if model else tuple(r.raw for r in batch.records)
            v=variance(values);sample_variances.append(v);vm+=mass*mass*v/count
            if v==0:zero.append(key)
        rz=sum(sum(v) for v in prices.values())
        result=dict(status='PILOT_DIAGNOSTIC_ONLY',precision='NOT_CERTIFIED',mean=mean,
            sample_variances=tuple(sample_variances),estimated_variance_of_mean=vm,
            diagnostic_SE_preview=float(vm)**.5,observed_evaluation_RZ=rz,score=vm*rz,
            zero_variance_cells=zero,counts=counts,expected_cost_guarantee='UNKNOWN')
        self.results[method,axis]=result;legacy.save(self.out/f'{method}_{axis}_EVALUATION.json',result)

    def accounting(self,all_attempts=None):
        methods={method:dict(bits=0,priced=0,known_RZ=0,price_complete=True,stages={}) for method in ('A1','B1-OL')}
        for e in self.envelopes:
            semantic='train' if e.method=='A1' and e.stage=='train' else 'probe' if e.stage=='train' else 'evaluation'
            count=sum(len(b.records) for _,b in e.rows);costs=[v for vs in e.rz_by_cell.values() for v in vs]
            known=[v for v in costs if type(v) is int and v>=0];m=methods[e.method]
            m['bits']+=count;m['priced']+=len(known);m['known_RZ']+=sum(known)
            m['price_complete'] &= len(known)==count
            m['stages'][e.axis+':'+semantic]=dict(bits=count,priced=len(known),known_RZ=sum(known))
        for m in methods.values():m['total_RZ']=m['known_RZ'] if m['price_complete'] else 'UNKNOWN'
        legacy.save(self.out/'PILOT_ACCOUNTING.json',dict(received_group_methods=methods,
            authoritative_all_attempt_methods=all_attempts,
            scope='all-attempt reconciliation is authoritative, received groups are a diagnostic subset',
            all_attempts_unknown=(all_attempts is None),expected_cost_guarantee='UNKNOWN'))

    def run(self):
        started=time.monotonic();status='STOPPED';reason=None;all_attempts=None;finalization_errors=[]
        try:
            for axis in ('X','Y'):self.learn(axis)
            for axis in ('X','Y'):self.probe(axis)
            for method in ('A1','B1-OL'):
                for axis in ('X','Y'):self.evaluate(method,axis)
            status='PILOT_DIAGNOSTIC_COMPLETED'
        except BaseException as exc:
            reason=dict(type=type(exc).__name__,message=str(exc));self.collector.request_stop(reason)
        finally:
            try:all_attempts=self.collector.finalize()
            except BaseException as exc:
                finalization_errors.append(dict(type=type(exc).__name__,message=str(exc)))
                status='STOPPED';self.collector.request_stop('finalization failure')
            self.accounting(all_attempts)
        classifications={}
        for axis in ('X','Y'):
            a=self.results.get(('A1',axis));b=self.results.get(('B1-OL',axis))
            classifications[axis]=('INCONCLUSIVE' if not a or not b or a['zero_variance_cells'] or b['zero_variance_cells']
                else 'FAVOURABLE_PILOT_SIGNAL' if a['score']<b['score'] else 'NO_FAVOURABLE_SIGNAL')
        values=tuple(classifications.values())
        overall=('NOT_EVALUATED_DUE_STOP' if status!='PILOT_DIAGNOSTIC_COMPLETED' else
                 'FAVOURABLE_PILOT_SIGNAL' if all(v=='FAVOURABLE_PILOT_SIGNAL' for v in values) else
                 'NO_FAVOURABLE_SIGNAL' if all(v=='NO_FAVOURABLE_SIGNAL' for v in values) else 'INCONCLUSIVE')
        legacy.save(self.out/'PILOT_RESULT.json',dict(status=status,reason=reason,finalization_errors=finalization_errors,classifications=classifications,
            overall_classification=overall,engine_bound_exact=self.engine_bound if self.engine_bound is not None else 'UNKNOWN (no bound supplied)',
            classification_note='exploratory only; zero variance never proves improvement',
            formal_precision='NOT_CERTIFIED',elapsed_driver_seconds=time.monotonic()-started))
        legacy.save(self.out/'STOP.json',dict(status='MANDATORY_STOP',purpose=PURPOSE,
            campaign_status='COMPLETED' if status=='PILOT_DIAGNOSTIC_COMPLETED' else 'STOPPED',
            pilot_status=status,reason=reason,finalization_errors=finalization_errors,
            formal_precision='NOT_CERTIFIED',next_stage_authorized=False,automatic_restart=False))
        return status


def semantic_exit_code(out):
    path=Path(out)/'STOP.json'
    if not path.exists():return 1
    stop=json.loads(path.read_text())
    return 0 if stop.get('purpose')==PURPOSE and stop.get('campaign_status')=='COMPLETED' and stop.get('pilot_status')=='PILOT_DIAGNOSTIC_COMPLETED' else 1


def connect_legacy_pipeline(out,campaign,services,streams,pool,workers,runtime):
    """Deferred connector; caller must bind the fixed services/shared event.

    This does not load a model, create RNG/pool, compile, or submit a trial.
    The preparation CLI never calls it with real services.
    """
    if legacy._SERVICES is not services:raise ValueError('legacy worker services not bound to this pilot')
    raw=legacy.Campaign(out,services,streams,pool,workers)
    collector=LegacyCollector(raw,campaign,runtime)
    return PilotDriver(out,campaign,services[2],services[3],collector,services[4].get('total_engine_coordinate_upper_exact'))


def execution_blocked():
    raise RuntimeError('PREPARATION_ONLY: pending scientific settings/adoption and separate execution approval; no real binding/RNG/engine/compile started')
