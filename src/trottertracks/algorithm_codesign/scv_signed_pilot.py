"""Fixed 448-shot orchestration over existing signed kernels, no loader/RNG.

Only dependency-injected fresh acquisition is accepted. Full/half diagnostics
share measurements and prices. No scientific work occurs at import.
"""
from dataclasses import replace
from fractions import Fraction as F
from pathlib import Path
import hashlib
import json
import time

from . import a1_control_variates as cv
from . import a1_n3_connection as n3
from . import a1_n3_campaign as c
from . import scv_signed_support as signed
from . import scv_scalar as scalar
from . import scv224_pilot as old
from .a1_n3_managed_runtime import durable_new

CAMPAIGN = 'SCV-SIGNED-H4-H1-PAIRED-20261107-v1'
PURPOSE = 'SCV_SIGNED_SUPPORT_H1_FRESH_PAIRED_DIAGNOSTIC'
SEED = 20261110
COUNTS = (8,128,8,8,8)
HALF = (4,64,4,4,4)
CAPS = dict(physical_trial=448, conditional_path=448, probability=448, Born=448,
    bit_threshold=448, selected_full_build=896, price=448, compile=448,
    apply_exact=427840, word=29120, OLS=2, paired=6, saved_bundle_decode=1,
    saved_control_restore=1, stream_book=1)


def domains():return old.domains()  # same 12 schema tags; a distinct seed/namespace


def settings():
    return dict(campaign=CAMPAIGN,purpose=PURPOSE,seed=SEED,train_per_axis=64,
        science_scope=dict(model='saved linear H4',distance_angstrom='1.00',basis='STO-3G',DF='legacy DF12',
            system_qubits=8,number_sector=4,L_D=3,m=0,q=1,delta='0.02',r=1,K=2,coefficient_atol='0.0',
            law='N1 actual finite-grid Q',policy='selected full',state='N2 certified ground',engine='certified exact Fraction',axes=['X','Y']),
        compiler_profile=dict(qiskit_version='1.3.0',basis_gates=['rz','sx','x','cx'],optimization_level=2,
            transpiler_seed=17,backend_name=None,coupling_map=None,layout_method=None,routing_method=None,
            circuit_scope='full_no_state_preparation'),
        evaluation_per_axis=COUNTS,physical_shots=448,range_guard='6',
        feature=signed.FEATURE,domain_mapping=[dict(token=d.token,path_spawn_key=[2*i],
            bit_spawn_key=[2*i+1]) for i,d in enumerate(domains())],children=24,
        RNG='PCG64/SeedSequence',path_word_cap=64,bit_words=1,call_caps=CAPS,
        max_gate_count=955,science_retry=0,extra_campaigns=0,
        stage_order=['X_train','Y_train','X_fit','Y_fit','X_prod','Y_prod'],
        full_counts=COUNTS,half_counts=HALF,precision='NOT_CERTIFIED',
        statistical_significance='NOT_ASSESSED',expected_cost_guarantee='UNKNOWN')


def source_fields(campaign,control):
    return dict(campaign=campaign,purpose=PURPOSE,seed=SEED,
        metadata_identity=control.metadata.identity,control_identity=control.identity,
        feature=signed.FEATURE,settings=settings())


def check_source(out,campaign,control):
    out=Path(out)
    if any((out/n).exists() for n in ('STOP.json','STOP_REQUESTED.json')):
        raise ValueError('STOP exists; retry/resume forbidden')
    data=(out/'SOURCE.json').read_bytes();record=json.loads(data)
    expected=json.loads(c.dumps(source_fields(campaign,control)))
    if any(record.get(k)!=v for k,v in expected.items()):raise ValueError('signed SOURCE binding mismatch')
    return hashlib.sha256(data).hexdigest()


def signed_guard(model):
    if not isinstance(model,signed.SignedModels):raise ValueError('typed signed model required')
    active=model.models[1];width=active.frozen.output_range.hi-active.frozen.output_range.lo
    if width!=2+2*abs(active.coefficient) or width>6:
        raise ValueError('signed width guard exceeded; no clipping/refit')
    if any(m.coefficient!=0 for i,m in enumerate(model.models) if i!=1):raise ValueError('inactive coefficient changed')
    return dict(adopted_width_guard='6',actual_width=width,PASS=True)


def split_rows(rows):
    halves=[[],[]]
    if tuple(h for h,_ in rows)!=scalar.CELLS:raise ValueError('five ordered cells required')
    for (h,batch),n in zip(rows,COUNTS):
        if {r.trial_id for r in batch.records}!={str(i) for i in range(n)} or len(batch.records)!=n:
            raise ValueError('fixed evaluation trial IDs/quota required')
        records=tuple(sorted(batch.records,key=lambda r:int(r.trial_id)))
        for i in range(2):halves[i].append((h,replace(batch,records=records[i*n//2:(i+1)*n//2])))
    return tuple(tuple(v) for v in halves)


def diagnostic(full,a,b):
    delta=lambda r:r['B1_OL']['estimated_variance_of_mean']-r['SCV']['estimated_variance_of_mean']
    df,da,db=delta(full),delta(a)/2,delta(b)/2;spread=abs(da-db)/2;flags=[]
    if df>0 and (da<=0 or db<=0 or df<=spread):flags.append('INCONCLUSIVE_SPLIT_DIAGNOSTIC')
    for r in (full,a,b):
        cell=r['cells'][1]
        if (cell['raw_all_equal'] or cell['residual_all_equal'] or len(cell['feature_counts'])<2
            or r['B1_OL']['estimated_variance_of_mean']==0):
            flags.append('INCONCLUSIVE_DEGENERATE_OR_SPARSE');break
    return dict(delta_full=df,delta_half_A_full_scale=da,delta_half_B_full_scale=db,
        half_spread=spread,flags=sorted(set(flags)),point_improvement=(df>0),
        half_is_CI=False,statistical_significance='NOT_ASSESSED',precision='NOT_CERTIFIED',
        expected_cost_guarantee='UNKNOWN')


class Driver:
    def __init__(self,out,campaign,control,scopes,collector,gate):
        self.out=Path(out);self.campaign=campaign;self.control=control;self.collector=collector;self.gate=gate
        self.source_hash=check_source(out,campaign,control)
        if not isinstance(control,signed.SignedControl):raise ValueError('saved typed signed control required')
        if (set(scopes)!={'X','Y'} or any(s.axis!=a or s.weight_bound!=1 or
            s.law_id!=control.metadata.law_sha256 for a,s in scopes.items()) or
            len({(s.target_id,s.circuit_id,s.round_id) for s in scopes.values()})!=1):
            raise ValueError('same target/circuit/round, fixed-law X/Y scopes required')
        durable_new(self.out/'SIGNED_INVOCATION.json',dict(campaign=campaign,purpose=PURPOSE,retry=False))
        self.scopes=scopes;self.envelopes=[];self.seen=set();self.models={};self.results={};self.timings={}
        self.phase='READY';self.started=False;self.next_acquire=0

    def acquire(self,axis,stage):
        order=(('X','train'),('Y','train'),('X','prod'),('Y','prod'))
        if self.next_acquire>=4 or (axis,stage)!=order[self.next_acquire]:raise ValueError('stage order/no retry')
        if stage=='prod' and self.phase!='FROZEN':raise ValueError('both fits must pass before evaluation')
        if check_source(self.out,self.campaign,self.control)!=self.source_hash:raise ValueError('SOURCE changed')
        quotas=(('1',64),) if stage=='train' else tuple(zip(scalar.CELLS,COUNTS))
        self.next_acquire+=1
        c.append(self.out/'SIGNED_JOURNAL.jsonl',dict(event='ACQUIRE_START',axis=axis,stage=stage,quotas=quotas))
        t=time.perf_counter();e=self.collector.collect(axis,stage,quotas)
        self.timings[axis+'_'+stage+'_wall_seconds']=time.perf_counter()-t
        c.save(self.out/f'RECEIVED_{axis}_{stage}.json',e);self.envelopes.append(e)
        if check_source(self.out,self.campaign,self.control)!=self.source_hash:raise ValueError('SOURCE changed during acquisition')
        if (e.campaign,e.purpose,e.metadata_identity,e.control_identity,e.axis,e.stage)!=(
            self.campaign,PURPOSE,self.control.metadata.identity,self.control.identity,axis,stage):
            raise ValueError('foreign campaign/purpose/control/axis/stage')
        if tuple(h for h,_ in e.rows)!=tuple(h for h,_ in quotas):raise ValueError('all positive cells/order required')
        expected={}
        for (cell,batch),(_,n) in zip(e.rows,quotas):
            if (batch.scope!=self.scopes[axis] or batch.stage!=stage or len(batch.records)!=n
                or {r.trial_id for r in batch.records}!={str(i) for i in range(n)}):
                raise ValueError('scope/stage/quota/trial ID mismatch')
            original=n3.Domain('A1' if stage=='train' else 'B1-OL',axis,stage,cell).token
            if any(r.stream_id!=original for r in batch.records):raise ValueError('independent stream mismatch')
            projected=replace(batch,records=tuple(replace(r,stream_id=n3.Domain('B1-OL',axis,stage,cell).token) for r in batch.records))
            n3.validate_batch(self.control.metadata.sampler,projected,'B1-OL',cell,stage)
            for r in batch.records:
                global_id=(self.campaign,r.stream_id,r.trial_id)
                if global_id in self.seen:raise ValueError('duplicate global trial')
                self.seen.add(global_id);expected[r.key]=batch.scope
        cv.account_costs(e.costs)
        if len(e.costs)!=len(expected) or {(v.stream_id,v.trial_id) for v in e.costs}!=set(expected):
            raise ValueError('price missing/duplicated; UNKNOWN')
        for v in e.costs:
            if (v.scope!=expected[(v.stream_id,v.trial_id)] or v.stage!=stage or v.quantum[2] is None
                or F(v.quantum[2]).denominator!=1 or v.quantum[2]<0):raise ValueError('price scope/profile/missing RZ')
        return e.rows

    def run(self):
        if self.started:raise ValueError('single invocation; no retry')
        self.started=True;status='STOPPED';reason='UNKNOWN';finalizer=None
        try:
            c.save(self.out/'SAVED_CONTROL_FROZEN.json',self.control.snapshot())
            training={a:self.acquire(a,'train')[0][1] for a in ('X','Y')}
            for axis in ('X','Y'):
                self.phase=axis+'_FIT';t=time.perf_counter()
                with self.gate.call('OLS',axis=axis):
                    model=signed.fit_h1(self.control,training[axis],sink=lambda info,a=axis:c.save(self.out/f'{a}_FIT.json',info))
                guard=signed_guard(model)
                self.models[axis]=model;c.save(self.out/f'{axis}_GUARD.json',guard)
                self.timings[axis+'_fit_wall_seconds']=time.perf_counter()-t
            self.phase='FROZEN'
            c.save(self.out/'BOTH_AXES_FROZEN.json',dict(coefficients={a:tuple(x.coefficient for x in m.models) for a,m in self.models.items()},
                guard='6',control_identity=self.control.identity,quota=COUNTS,seed=SEED))
            for axis in ('X','Y'):
                rows=self.acquire(axis,'prod');ha,hb=split_rows(rows);t=time.perf_counter();parts=[]
                for name,part,counts in (('full',rows,COUNTS),('half_A',ha,HALF),('half_B',hb,HALF)):
                    with self.gate.call('paired',axis=axis,part=name):
                        result=signed.paired_evaluate(self.models[axis],part,counts=counts)
                    parts.append(result);c.save(self.out/f'{axis}_{name}.json',result)
                ledger=old.cost_summary([e for e in self.envelopes if e.axis==axis])
                result=dict(full=parts[0],half_A=parts[1],half_B=parts[2],diagnostic=diagnostic(*parts),
                    realized_RZ_reference=ledger,exploratory_variance_times_realized_RZ=dict(
                        B1_OL=parts[0]['B1_OL']['estimated_variance_of_mean']*ledger['B1_OL_reference_evaluation_RZ'],
                        SCV=parts[0]['SCV']['estimated_variance_of_mean']*ledger['SCV_training_plus_evaluation_RZ']))
                self.results[axis]=result;self.timings[axis+'_paired_wall_seconds']=time.perf_counter()-t
                c.save(self.out/f'{axis}_PAIRED_RESULT.json',result)
            status='COMPLETE';reason='448 physical shots; exploratory paired diagnosis only'
        except BaseException as exc:
            reason=type(exc).__name__+': '+str(exc)
        finally:
            self.phase='STOPPED'
            self.collector.request_stop(reason)  # shared linearized STOP before pool drain
            try:finalizer=self.collector.finalize()
            except BaseException as exc:status='STOPPED';reason+='; finalizer '+str(exc)
            try:costs=old.cost_summary(self.envelopes)
            except Exception as exc:costs=dict(status='UNKNOWN',reason=str(exc))
            costs['complete_campaign']=(status=='COMPLETE')
            costs['authoritative_all_inflight_tails']='ACCOUNTING.json / SIGNED_TAIL_ACCOUNTING.json'
            if status!='COMPLETE':costs['physical_RZ']='UNKNOWN; returned envelopes exclude possible tails'
            c.save(self.out/'SIGNED_ACCOUNTING.json',costs)
            c.save(self.out/'SIGNED_CLASSICAL.json',dict(parent_measured_intervals=self.timings,finalizer=finalizer,
                interpretation='Nested wall intervals are not additive; worker wall sum is not CPU time; unknown costs remain UNKNOWN'))
            result=dict(status=status,reason=reason,campaign=self.campaign,axes=self.results,
                precision='NOT_CERTIFIED',statistical_significance='NOT_ASSESSED',expected_cost_guarantee='UNKNOWN',retry=False)
            c.save(self.out/'SIGNED_RESULT.json',result)
            durable_new(self.out/'STOP.json',dict(status=status,reason=reason,mandatory_STOP=True,retry=False))
        return result
