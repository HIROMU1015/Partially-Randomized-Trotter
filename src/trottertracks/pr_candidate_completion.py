"""Complete B2-selected PR inputs; never choose LD, kappa, width or shot rules.

Scientific leaves are the published method-specific providers/acquisition paths.
The public API accepts the optimizer's input request/canonical configuration.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from copy import deepcopy
from fractions import Fraction
from . import pr_candidate_selection as Selection
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time

ROOT=Path(__file__).resolve().parents[2]
NATIVE='/home/AbeHiromu/projects/h6-environments/20261012/native-dependency-v2/venv/bin/python'
PRICE='/home/AbeHiromu/projects/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/venv/bin/python'
D2_VIEW=ROOT.parent/'b2-route-view-d2-c1f95e080'
B_VIEW=ROOT.parent/'d2-auto-b-intake-33e324b'
DF_Q3_VIEW=ROOT.parent/'b2-route-view-d2-23160cc87'
A_WORKSPACE=ROOT.parent/'d2-pauli-auto-nodrop-evidence-20261116'
GIT_LOCK=threading.RLock()
IDENTITY=['-c','user.name=HIROMU1015','-c','user.email=u411612g@ecs.osaka-u.ac.jp']


def git(*args,cwd=ROOT):return subprocess.check_output(['git',*args],cwd=cwd,text=True).strip()
def read(p):return json.loads(Path(p).read_text())
def value(loc):
    x=json.loads(subprocess.check_output(['git','show',loc['commit']+':'+loc['path']],cwd=ROOT))
    for k in loc.get('selector',[]):x=x[k]
    return x

def dec(v):return Fraction(int(v['numerator']),int(v['denominator']))if isinstance(v,dict)else Fraction(v)
def enc(v):v=Fraction(v);return dict(numerator=v.numerator,denominator=v.denominator)
def put(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);s=json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
    if p.exists()and p.read_text()==s:return
    tmp=p.with_name(p.name+'.tmp-'+str(os.getpid())+'-'+str(threading.get_ident()));tmp.write_text(s);tmp.replace(p)
def key(x):return hashlib.sha256(json.dumps(x,sort_keys=True,default=str).encode()).hexdigest()[:20]
def rel(p):return str(Path(p).resolve().relative_to(ROOT))


def publish(paths,message):
    with GIT_LOCK:
        selected=[rel(p)for p in paths if Path(p).exists()]
        if selected:
            git('add','--sparse','--',*selected)
            if git('diff','--cached','--name-only','--',*selected):
                git(*IDENTITY,'commit','--only','-q','-m',message+' [skip ci]','--',*selected)
        return git('rev-parse','HEAD')


def physical_evaluation_contract(config):
    """Exact schedule/family contract, independent of declared width and kappa.

    Requested parameters act through the mapper's physical rounds. Administrative
    configuration IDs are removed; H/state, dictionary and law SOURCE, target,
    error/decoder/profile contracts and actual steps remain part of equality.
    """
    spec=deepcopy(config['spec'])
    for field in ('kappa','delta_pf','delta_over_h'):spec.pop(field,None)
    # DF keeps the original mapper input as provenance as well as the resolved
    # spec. Its declared width/kappa are not extra physical conditions.
    if isinstance(spec.get('A_input'),dict):
        spec['A_input'].pop('kappa_paper',None)
        design=spec['A_input'].get('design',{})
        for field in ('delta','delta_pf','delta_over_h'):design.pop(field,None)
    rows=[]
    for row in sorted(config['rounds'],key=lambda r:r['m']):
        physical=deepcopy(row.get('physical_candidate',{}));physical.pop('id',None)
        candidate=deepcopy(row.get('candidate',{}));candidate.pop('id',None)
        rows.append(dict(m=row['m'],T=row['T'],q=row['q_PF'],R=row['R_outer'],
            actual_delta_pf=row['actual_delta_pf'],structure=row['structure'],
            law_binding=row.get('law_binding'),physical_candidate=physical,candidate=candidate))
    return dict(spec=spec,rounds=rows)


class Held(Exception):pass


class Completion:
    def __init__(self,directory,settings):
        self.directory=Path(directory).resolve();self.directory.mkdir(parents=True,exist_ok=True)
        self.settings=settings
        self.state_path=self.directory/'STATE.json'
        prior=read(self.state_path)if self.state_path.exists()else None
        self.source=settings.get('source')or (prior or{}).get('source')or git('rev-parse','HEAD')
        self.state=prior or dict(schema='PR_CANDIDATE_COMPLETION_V1',source=self.source,records={})
        imported=settings.get('resume_preparation_from')
        if not prior and imported:
            saved=value(imported)
            # Only terminal READY evidence crosses a technical source epoch.
            # Failed stages and their original STOPs remain in the old state.
            self.state['records']={k:dict(r,reused_from=imported) for k,r in saved['records'].items()
                                   if r['status']=='READY_FOR_SEARCH'}
        if self.state['source']!=self.source:raise ValueError('Completion SOURCE changed; preserve prior state')
        self.lock=threading.RLock()
        self.pauli_price_lock=threading.RLock()
        self.pauli_metadata_lock=threading.RLock()
        self.stage_executor=None
        self.shared_preparation_locks={}
        self.registry=self.directory/'PRICE_RULES_REGISTRY.json'
        if not self.registry.exists():
            provided=settings.get('price_registry_source')or settings.get('price_registry_file')
            if provided:
                initial=value(provided)if isinstance(provided,dict)else read(Path(provided))
                extra=dict(entries=[])
            else:
                initial=value(dict(commit='c1f95e0806f5478a1ee5ce9ecb148ce9d10461aa',path='configs/pr_price_rules/h4_saved_rules.json'))
                extra=value(dict(commit='54b4353b9',path='docs/research/b2_pauli_calibration_20261116/PRICE_RULES_REGISTRY_EXTENDED.json'))
            by={e['name']:e for e in initial['entries']};by.update({e['name']:e for e in extra['entries']})
            if imported:
                previous=Path(imported['path']).with_name('PRICE_RULES_REGISTRY.json')
                saved_registry=value(dict(commit=imported['commit'],path=str(previous)))
                if provided:
                    for entry in saved_registry['entries']:by.setdefault(entry['name'],entry)
                else:by.update({e['name']:e for e in saved_registry['entries']})
            put(self.registry,dict(initial,entries=list(by.values())))
        self.step_registry_path=self.directory/'STEP_PRICE_RULES.json'
        if settings.get('price_evidence_results'):
            self.ingest_price_results()

    def saved_signal_packets(self,method):
        name='saved_pauli_packets'if method=='PR_PAULI'else'saved_df_bundles'
        packets=list(getattr(self,'settings',{}).get(name,[]))
        for record in getattr(self,'state',{}).get('records',{}).values():
            loc=(record.get('signal')or{}).get('packet_source')
            if record.get('method')==method and loc and loc not in packets:packets.append(loc)
        return packets

    def _step_rules(self):
        """The partial-step registry is independent of the legacy 26-axis rule."""
        from .pr_price_rules import GitReader
        from .pr_step_price_registry import StepPriceRegistry
        path=getattr(self,'step_registry_path',self.directory/'STEP_PRICE_RULES.json')
        return StepPriceRegistry(GitReader(ROOT),read(self.registry),state=read(path)if path.exists()else None)

    def ingest_price_results(self,packages=None):
        """Accept fixed audited results once; never launch a calibration here."""
        packages=(getattr(self,'settings',{}).get('price_evidence_results',[])if packages is None else packages)
        if not packages:return []
        with self.lock:
            rules=self._step_rules();answers=[rules.ingest(item)for item in packages]
            put(getattr(self,'step_registry_path',self.directory/'STEP_PRICE_RULES.json'),rules.state)
        return answers

    def price_registry_digest(self):
        path=getattr(self,'step_registry_path',self.directory/'STEP_PRICE_RULES.json')
        return key(dict(legacy=read(self.registry),steps=read(path)if path.exists()else None))

    def quote_with_steps(self,config,quote):
        path=getattr(self,'step_registry_path',self.directory/'STEP_PRICE_RULES.json')
        if not path.exists():return quote
        return self._step_rules().quote(config,quote['implementation'],base_quote=quote)

    def save(self):
        with self.lock:put(self.state_path,self.state)

    def log(self,**event):
        with self.lock:
            with(self.directory/'PROGRESS.jsonl').open('a')as f:f.write(json.dumps(dict(epoch=time.time(),**event),default=str)+'\n')

    def bind_stage_executor(self,runtime):
        """Use the same durable host queue as Q3; metadata remains central."""
        self.stage_executor=runtime
        return self

    def shared_preparation_lock(self,identity):
        """One producer for an exact physical contract or calibration group."""
        token=key(dict(source=self.source,identity=identity))
        with self.lock:return self.shared_preparation_locks.setdefault(token,threading.RLock())

    def call(self,cmd,d,stage,output=None,science=False):
        d=Path(d);d.mkdir(parents=True,exist_ok=True);done=d/(stage+'.RETURN.json');started=d/(stage+'.STARTED.json')
        if done.exists():
            r=read(done)
            if r['returncode']!=0:raise Held('PRIOR_STAGE_FAILED_NO_RETRY:'+stage)
            return read(output)if output and Path(output).exists()else r
        runtime=getattr(self,'stage_executor',None)
        if runtime is not None and science and runtime.handles(stage):
            # Recovery is reconciled with the original host/attempt by the
            # shared executor. A local STARTED marker is never a retry grant.
            return runtime.call(self,cmd,d,stage,output,science=science)
        if started.exists():raise Held('IN_PROGRESS_OR_AMBIGUOUS_NO_DUPLICATE:'+stage)
        put(started,dict(command=cmd,source=self.source,science=science,automatic_retry=False))
        self.log(event='STAGE_START',stage=stage,science=science,directory=rel(d));t=time.time()
        with(d/(stage+'.log')).open('w')as f:
            p=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),stdout=f,stderr=subprocess.STDOUT)
            put(started,dict(command=cmd,source=self.source,pid=p.pid,science=science,automatic_retry=False))
            code=p.wait()
        put(done,dict(returncode=code if code>=0 else None,signal=-code if code<0 else None,seconds=time.time()-t,source=self.source))
        self.log(event='STAGE_RETURN',stage=stage,returncode=code,seconds=time.time()-t)
        if code!=0:
            detail=read(output)if output and Path(output).exists()else{}
            raise Held('STAGE_FAILED:'+stage+':'+str(detail.get('status'))+':'+str(detail.get('reason'))+':'+str(d/(stage+'.log')))
        return read(output)if output and Path(output).exists()else read(done)

    def service(self,mode,request,d,tag=None,python=NATIVE):
        tag=tag or mode;inp=d/(tag+'.INPUT.json');out=d/(tag+'.json');put(inp,request)
        return self.call([python,'-B',str(ROOT/'scripts/pr_candidate_services.py'),mode,'--input',str(inp),'--output',str(out)],d,tag,out)

    def saved_df_descriptor(self,request):
        if request.get('tail_descriptor'):return request['tail_descriptor']
        cfg=request.get('configuration')
        if cfg:
            cfg=value(cfg)if 'commit'in cfg else cfg
            source=cfg.get('spec',{}).get('split',{}).get('lambda_source',{})
            if source.get('commit')and source.get('path'):
                # The descriptor validator still checks its H/state/prefix.
                return source
        prefix=(request.get('design')or{}).get('L_D')
        for record in self.state['records'].values():
            source=record.get('request',{}).get('tail_descriptor')
            if record['method']=='PR_DF'and source and (record.get('design')or record.get('request',{}).get('design',{})).get('L_D')==prefix:
                return source
        for record in self.state.get('family_metadata',{}).values():
            if record.get('method')=='PR_DF'and record.get('L_D')==prefix and record.get('tail_descriptor'):
                return record['tail_descriptor']
        return None

    def refresh_price_hold(self,record):
        """Requery newly registered evidence, never retry a scientific leaf."""
        reason=str(record.get('reason',''))
        if record.get('scientific_failure')or record.get('status','').startswith('STOP')or any(
                token in reason for token in ('STAGE_FAILED:','PRIOR_STAGE_FAILED','NUMERICAL_GATE','NO_AUTOMATIC_RETRY')):
            return None
        if not all(record.get(k)for k in ('configuration','signal','intake')):return None
        digest=self.price_registry_digest()
        if record.get('registry_at_hold')==digest:return None
        d=self.directory/'refresh'/record['key']/digest
        registry=d/'PRICE_RULES.json';put(registry,read(self.registry))
        quote=self.service('quote',dict(configuration=record['configuration'],view=str(D2_VIEW),
            registry_file=str(registry),policy='MODEL'),d,tag='AVAILABLE_EVIDENCE_QUOTE',python=PRICE)
        quote=self.quote_with_steps(value(record['configuration']),quote)
        put(d/'AVAILABLE_EVIDENCE_QUOTE.json',quote)
        if not quote.get('complete'):
            record['registry_at_hold']=digest;self.save();return None
        fixed=publish([d],'Reuse newly available candidate prices without scientific retry')
        config=value(record['configuration'])
        record.update(previous_hold=dict(status=record['status'],reason=record.get('reason')),
            status='READY_FOR_SEARCH',reason=None,price_policy='MODEL',prices=[r['unit_price_RZ']for r in quote['rows']],
            quote=dict(commit=fixed,path=rel(d/'AVAILABLE_EVIDENCE_QUOTE.json')),
            design=dict(L_D=config['spec']['L_D'],formula=config['spec']['formula'],kappa=enc(dec(config['spec']['kappa']['value'])),delta_pf=config['spec']['delta_pf']),
            schedule=[dict(m=r['m'],T=r['T'],q=r['q_PF'],R=r['R_outer'],actual_delta_pf=r['actual_delta_pf'],structure=r['structure'],physical_id=r['candidate']['id'])for r in config['rounds']],
            recovered_by='METADATA_REQUERY_NO_SCIENTIFIC_RETRY')
        self.save();return record

    def complete(self,request):
        if request['method']not in ('PR_PAULI','PR_DF'):raise ValueError('PR_CANDIDATE_COMPLETION_ONLY')
        if request.get('plan_only')or getattr(self,'settings',{}).get('plan_only'):
            return self.plan(request)
        identifier=key(request);d=self.directory/'candidates'/identifier
        with self.lock:
            old=self.state['records'].get(identifier)
            if old and old['status']=='READY_FOR_SEARCH':return dict(old,resumed=True)
            if old and old['status'].startswith(('HELD','STOP','WAITING')):
                self.ingest_price_results()
                return dict(self.refresh_price_hold(old)or old,resumed=True)
            rec=dict(method=request['method'],selection=request.get('selection'),request=request,status='SELECTED',key=identifier,directory=rel(d))
            self.state['records'][identifier]=rec;self.save()
        self.log(event='SELECTED_BY_B2',method=rec['method'],selection=rec['selection'])
        physical_guard=None
        try:
            if request.get('selection_error'):raise Held('B2_SELECTION_NORMALIZATION:'+request['selection_error'])
            if request['method']=='PR_PAULI':config,prepared=self.pauli_configuration(request,d)
            else:
                saved_descriptor=self.saved_df_descriptor(request)
                if saved_descriptor:request['tail_descriptor']=saved_descriptor
                generated=self.service('df-generate',request,d)
                if generated['status']=='NEEDS_DF_TAIL_DESCRIPTOR':
                    prefix=request['design']['L_D'];descriptor=d/'DF_TAIL_DESCRIPTOR.json'
                    self.call([NATIVE,'-B',str(ROOT/'scripts/derive_pr_df_candidate_descriptor.py'),'--prefix',str(prefix),'--configuration','0defa286c53a1b9acf513be453907bcad83102b3:docs/research/d_pr_df_ld7_v0_results_20261116/generation/CONFIGURATION.json','--output',str(descriptor),'--source',self.source],d,'DF_DESCRIPTOR',descriptor,science=True)
                    dc=publish([d],'Fix derived DF tail descriptor and scientific SOURCE')
                    request['tail_descriptor']=dict(commit=dc,path=rel(descriptor))
                    generated=self.service('df-generate',request,d,tag='DF_GENERATE_WITH_DESCRIPTOR')
                if generated['status']!='CONFIGURATION_READY':raise Held('CONFIGURATION:'+str(generated))
                config=generated['configuration'];prepared=None
            physical_guard=self.shared_preparation_lock(dict(method=request['method'],physical=physical_evaluation_contract(config)))
            physical_guard.acquire()
            if request.get('design')and dec(config['spec']['kappa']['value'])!=dec(request['design']['kappa']):raise Held('ABSOLUTE_KAPPA_MISMATCH')
            put(d/'CONFIGURATION.json',config);commit=publish([d],'Fix selected PR candidate configuration')
            cref=dict(commit=commit,path=rel(d/'CONFIGURATION.json'));rec.update(configuration=cref,configuration_id=config['id'])
            # A declaration can differ while the complete physical schedule and
            # evaluation contract are identical. Reuse READY inputs before any
            # scientific leaf; install() retains both declarations as aliases.
            twin=next((old for old in self.state['records'].values()
                if old is not rec and old.get('method')==request['method']and old.get('status')=='READY_FOR_SEARCH'
                and old.get('configuration')and physical_evaluation_contract(value(old['configuration']))==physical_evaluation_contract(config)),None)
            if twin is not None:
                rec.update(status='READY_FOR_SEARCH',signal=deepcopy(twin['signal']),intake=deepcopy(twin['intake']),
                    quote=deepcopy(twin['quote']),prices=deepcopy(twin['prices']),price_policy=twin.get('price_policy','MODEL'),
                    reused_physical_configuration=twin['configuration'],new_scientific_calls=0,
                    design=dict(L_D=config['spec']['L_D'],formula=config['spec']['formula'],kappa=enc(dec(config['spec']['kappa']['value'])),delta_pf=config['spec']['delta_pf']),
                    schedule=deepcopy(twin.get('schedule',[])))
                self.save();return rec
            failed_twin=next((old for old in self.state['records'].values()
                if old is not rec and old.get('method')==request['method']and old.get('scientific_failure')
                and old.get('configuration')and physical_evaluation_contract(value(old['configuration']))==physical_evaluation_contract(config)),None)
            if failed_twin is not None:
                raise Held('PRIOR_PHYSICAL_PREPARATION_FAILED_NO_AUTOMATIC_RETRY:'+failed_twin['key'])
            initial_quote=self.initial_quote(config,cref,d)
            if request['method']=='PR_DF':
                if not initial_quote['complete']:
                    design=self.service('df-price-design',dict(configuration=config,price_quote=initial_quote),d)
                    bad=[g for g in design.get('groups',[])if g['status']!='CALIBRATION_DESIGN_READY']
                    if bad:raise Held('DF_MODEL_DESIGN:'+str(bad))
            def signal_work():
                return (self.pauli_signals(config,prepared,d)if request['method']=='PR_PAULI'
                        else self.df_signals(config,cref,d,request.get('tail_descriptor')))
            # Both branches depend on the frozen configuration, not on each
            # other's numerical result. They share host slots through the
            # executor rather than independently reserving CPU resources.
            if getattr(self,'stage_executor',None)is not None:
                with ThreadPoolExecutor(max_workers=2)as pool:
                    signal_future=pool.submit(signal_work)
                    price_future=pool.submit(self.price,config,cref,prepared,request,d,initial_quote)
                    signal=signal_future.result();rec.update(signal=signal);self.save()
                    intake=self.service('intake',dict(method=request['method'],configuration=signal.get('configuration_source',cref),packet=signal['packet_source'],view=str(B_VIEW if request['method']=='PR_PAULI'else DF_Q3_VIEW)),d)
                    rec['intake']=intake;self.save();quote=price_future.result()
            else:
                signal=signal_work();rec.update(signal=signal);self.save()
                intake=self.service('intake',dict(method=request['method'],configuration=signal.get('configuration_source',cref),packet=signal['packet_source'],view=str(B_VIEW if request['method']=='PR_PAULI'else DF_Q3_VIEW)),d)
                rec['intake']=intake;self.save()
                quote=self.price(config,cref,prepared,request,d,initial_quote=initial_quote)
            if not quote['complete']:raise Held('PRICE_RULE_INCOMPLETE:'+str(quote.get('classifications')))
            quote_commit=publish([d],'Publish completed candidate prices and receipt bindings')
            with self.lock:
                local_registry=d/'PRICE_RULES.json'
                if local_registry.exists():
                    current=read(self.registry);merged={e['name']:e for e in current['entries']};merged.update({e['name']:e for e in read(local_registry)['entries']});put(self.registry,dict(current,entries=list(merged.values())))
            rec.update(status='READY_FOR_SEARCH',quote=dict(commit=quote_commit,path=rel(d/'FINAL_QUOTE.json')),
                price_policy='MODEL',prices=[r['unit_price_RZ']for r in quote['rows']],
                design=dict(L_D=config['spec']['L_D'],formula=config['spec']['formula'],kappa=enc(dec(config['spec']['kappa']['value'])),
                    delta_pf=config['spec']['delta_pf']),schedule=[dict(m=r['m'],T=r['T'],q=r['q_PF'],R=r['R_outer'],actual_delta_pf=r['actual_delta_pf'],structure=r['structure'],physical_id=r['candidate']['id'])for r in config['rounds']])
        except Held as exc:
            text=str(exc)
            waiting=text.startswith(('PRICE_RULE_INCOMPLETE:','PRICE_EVIDENCE_REQUIRED:'))
            rec.update(status='WAITING_INPUTS'if waiting else'HELD_WITH_REASON',reason=text,
                       scientific_failure=any(t in text for t in ('STAGE_FAILED:','NUMERICAL_GATE','NO_AUTOMATIC_RETRY')))
        except Exception as exc:rec.update(status='HELD_TECHNICAL_FAILURE',reason=type(exc).__name__+': '+str(exc),scientific_failure=True)
        finally:
            if physical_guard is not None:physical_guard.release()
        if rec['status'].startswith(('HELD','STOP','WAITING')):rec['registry_at_hold']=self.price_registry_digest()
        self.save();self.log(event='CANDIDATE_COMPLETED',method=request['method'],status=rec['status'],reason=rec.get('reason'));return rec

    def plan(self,request):
        """Normal preparation entry without a scientific producer call.

        h/T and the requested width are passed to each existing mapper. A missing
        descriptor/dictionary or an unrepresentable step is returned as needs;
        another candidate can continue independently.
        """
        if request.get('method')not in ('PR_PAULI','PR_DF'):raise ValueError('PR_CANDIDATE_COMPLETION_ONLY')
        identifier=key(request);prior=self.state['records'].get(identifier)
        if prior and prior['status']=='READY_FOR_SEARCH':return dict(prior,resumed=True,plan_only=True)
        d=self.directory/'plans'/identifier/self.price_registry_digest()
        answer=d/'PLAN.json'
        if answer.exists():return dict(read(answer),resumed=True)
        d.mkdir(parents=True,exist_ok=True)
        out=dict(status='WAITING_INPUTS',method=request['method'],request=deepcopy(request),plan_only=True,
                 source=self.source,configuration=None,rounds=[],calibration_requests=[],new_scientific_calls=0)
        try:
            if request['method']=='PR_PAULI':
                config,prepared=self.pauli_configuration(request,d,allow_science=False)
                signals=prepared.get('signal_plan')or dict(status='SOURCE_UNRESOLVED',rounds=[],unsupported_rounds=list(range(13)))
            else:
                selected=deepcopy(request);descriptor=self.saved_df_descriptor(selected)
                if descriptor:selected['tail_descriptor']=descriptor
                generated=self.service('df-generate',selected,d)
                if generated.get('status')!='CONFIGURATION_READY':
                    out.update(reason=generated.get('reason',generated.get('status')),configuration_needs=generated)
                    put(answer,out);return out
                config=generated['configuration'];prepared=None
                signals=self.service('df-plan',dict(configuration=config,tail_descriptor=selected.get('tail_descriptor'),
                    saved_bundles=self.saved_signal_packets('PR_DF')),d)
            quote=self.initial_quote(config,config,d)
            if request['method']=='PR_PAULI':
                cal=self.service('pauli-price-plan',dict(configuration=config,prepared_item=prepared,price_quote=quote,
                    workspace=str(self.settings.get('pauli_workspace',A_WORKSPACE))),d)
            else:
                cal=self.service('df-price-design',dict(configuration=config,price_quote=quote),d)
            signal_rows={r['m']:r for r in signals.get('rounds',[])}
            out.update(configuration=config,configuration_id=config['id'],signal_plan=signals,price_quote=quote,
                calibration_plan=cal,calibration_requests=cal.get('calibration_requests',cal.get('groups',[])),
                rounds=[dict(m=r['m'],physical_id=r['candidate']['id'],T=r['T'],q=r['q_PF'],R=r['R_outer'],
                    actual_delta_pf=r['actual_delta_pf'],signal=signal_rows.get(r['m'],dict(decision='SOURCE_UNRESOLVED')),
                    price_axes=[row for row in quote['rows']if row['m']==r['m']])for r in config['rounds']],
                requested_delta_pf=config['spec']['delta_pf'],readiness=dict(configuration=True,
                    signal=not signals.get('request_rounds')and not signals.get('unsupported_rounds')
                        and len(signals.get('reuse_rounds',[]))==13,
                    prices=quote['complete']))
        except Held as exc:out['reason']=str(exc)
        put(answer,out)
        return out

    def prepare_family(self,request,allow_science=False):
        """Prepare a chosen split's mapper metadata only, never signal or price.

        The caller supplies a same-H template kappa solely to obtain that split's
        constants. Each split still derives its own descriptor/dictionary; the
        template never supplies another split's coefficients.
        """
        if request.get('method')not in('PR_PAULI','PR_DF'):raise ValueError('PR_CANDIDATE_COMPLETION_ONLY')
        d=self.directory/'family_metadata'/key(request);d.mkdir(parents=True,exist_ok=True)
        out=dict(status='NEEDS_FAMILY_PREPARATION',configuration=None,needs=[],
                 new_signal_calls=0,new_price_calls=0,scope='DICTIONARY_DESCRIPTOR_AND_MAPPER_METADATA_ONLY')
        try:
            if request['method']=='PR_PAULI':
                config,prepared=self.pauli_configuration(request,d,allow_science=allow_science)
                out.update(configuration=config,configuration_source=prepared.get('configuration_source'))
            else:
                selected=deepcopy(request);descriptor=self.saved_df_descriptor(selected)
                if descriptor:selected['tail_descriptor']=descriptor
                generated=self.service('df-generate',selected,d)
                if generated.get('status')=='NEEDS_DF_TAIL_DESCRIPTOR'and allow_science:
                    prefix=selected['design']['L_D'];target=d/'DF_TAIL_DESCRIPTOR.json'
                    template=selected.get('template_source')or dict(commit='0defa286c53a1b9acf513be453907bcad83102b3',path='docs/research/d_pr_df_ld7_v0_results_20261116/generation/CONFIGURATION.json')
                    self.call([NATIVE,'-B',str(ROOT/'scripts/derive_pr_df_candidate_descriptor.py'),'--prefix',str(prefix),
                        '--configuration',template['commit']+':'+template['path'],'--output',str(target),'--source',self.source],
                        d,'DF_DESCRIPTOR',target,science=True)
                    fixed=publish([d],'Fix chosen split descriptor for mapper metadata')
                    selected['tail_descriptor']=dict(commit=fixed,path=rel(target))
                    generated=self.service('df-generate',selected,d,tag='DF_GENERATE_WITH_DESCRIPTOR')
                if generated.get('status')!='CONFIGURATION_READY':
                    out.update(reason=generated.get('reason',generated.get('status')),needs=[generated])
                    put(d/'FAMILY.json',out);return out
                out.update(configuration=generated['configuration'],configuration_source=generated.get('configuration_source'),
                           tail_descriptor=selected.get('tail_descriptor'))
            out.update(status='CONFIGURATION_READY',needs=[])
            if out.get('tail_descriptor'):
                self.state.setdefault('family_metadata',{})[key(dict(method=request['method'],L_D=request['design']['L_D']))]=dict(
                    method=request['method'],L_D=request['design']['L_D'],tail_descriptor=out['tail_descriptor'])
                self.save()
        except Held as exc:out.update(reason=str(exc),needs=[str(exc)])
        put(d/'FAMILY.json',out);return out

    def pauli_configuration(self,r,d,allow_science=True):
        workspace=Path(self.settings.get('pauli_workspace',A_WORKSPACE));d.mkdir(parents=True,exist_ok=True)
        saved=[dict(commit='d7a976a629a6db68197b24f8f86734e109c8bf5d',path=p)for p in (
            'docs/research/rpe_split_preparation_20261116/LD4/PREPARATION_P1.0.CONFIGURATION.json',
            'docs/research/rpe_split_preparation_20261116/LD6/PREPARATION.0.CONFIGURATION.json')]
        def prepare(item,tag,science=False):
            inp=d/(tag+'.REQUEST.json');out=d/(tag+'.json')
            put(inp,dict(kind='pauli_native',items=[item],saved_configurations=saved,saved_packets=self.saved_signal_packets('PR_PAULI')))
            cache=(self.directory/'PAULI_SPLITS.json')if hasattr(self,'directory')else d/'SPLITS.json'
            cmd=[NATIVE,'-B',str(workspace/'scripts/prepare_pauli_candidate.py'),'--request',str(inp),'--output',str(out),'--cache',str(cache)]
            requested_prefix=(item.get('configuration')or{}).get('spec',{}).get('L_D',(item.get('configuration')or{}).get('L_D'))
            if requested_prefix in (7,8,9):cmd+=['--p0-builder','no-drop']
            if science:
                publish([d],'Fix selected Pauli P0 request')
                cmd+=['--prepare-missing','--source',git('rev-parse','HEAD',cwd=workspace),'--p0-directory',str(workspace/'docs/research/d2_auto_p0_20261116')]
            lock=getattr(self,'pauli_metadata_lock',None)
            with (lock if lock is not None else nullcontext()):
                response=self.call(cmd,d,tag,out,science=science)
            return response['items'][0]
        if r.get('configuration'):
            config=value(r['configuration'])if 'commit'in r['configuration']else r['configuration']
            item=prepare(dict(item_id='B2_SELECTED',configuration=config),'A_CANONICAL')
        else:
            design=r['design'];item=prepare(dict(item_id='B2_SELECTED_BASE',configuration=dict(L_D=design['L_D'])),'A_BASE')
            if item['status']=='NEEDS_SPLIT_PREPARATION':
                if not allow_science:raise Held('CONFIGURATION_PREPARATION_REQUIRED_NOT_EXECUTED:'+str(item))
                if design['L_D']in(7,8,9):raise Held('REGISTERED_NO_DROP_CONFIGURATION_UNRESOLVED_NO_P0_RETRY:'+str(design['L_D']))
                item=prepare(dict(item_id='B2_SELECTED_BASE',configuration=dict(L_D=design['L_D'])),'A_BASE_P0',True)
            if item['status']!='GENERATED':raise Held('A_PREPARATION:'+str(item))
            base=item['configuration'];changes=dict(formula=design['formula'],K=design.get('K',2),kappa_multiplier=str(dec(design['kappa'])/dec(base['spec']['kappa']['value'])))
            if design.get('delta_pf')is not None:changes['delta_pf']=design['delta_pf']
            else:changes['delta_over_h']=design['delta_over_h']
            item=prepare(dict(item_id='B2_SELECTED',configuration=base,changes=changes),'A_DESIGN')
        if item['status']!='GENERATED':raise Held('A_CONFIG_NOT_READY:'+str(item))
        return item['configuration'],item

    def pauli_signals(self,config,prepared,d):
        packets=self.saved_signal_packets('PR_PAULI')
        source=prepared['descriptor']['source']
        if 'commit'in source and 'path'not in source:raise Held('PAULI_DESCRIPTOR_SOURCE_UNRESOLVED')
        # Saved LD1/2/3/5 descriptors use A's index/context locator, not a
        # commit/path override. Let that same fixed entry resolve its saved
        # descriptor. Derived prefixes still require the explicit envelope.
        descriptor=(dict(prefix=config['spec']['L_D'],binding=config['spec']['binding'],
            descriptor=prepared['descriptor']['descriptor'],source=source)if 'commit'in source else None)
        if descriptor is None and(config['spec']['L_D']not in(1,2,3,5)or not all(k in source for k in('index','context_sha256'))):
            raise Held('PAULI_DESCRIPTOR_SOURCE_UNRESOLVED')
        request=dict(configuration=config,tail_descriptor=descriptor,saved_packets=packets,
            workspace=str(self.settings.get('pauli_workspace',A_WORKSPACE)),directory='docs/research/d2_automatic_pr_20261116/'+config['id'][:20]+'/'+self.settings.get('signal_epoch','descriptor_envelope_v2'),interpreter=NATIVE)
        inp=d/'PAULI_SIGNAL_REQUEST.json';out=d/'PAULI_SIGNAL_RESULT.json';put(inp,request)
        result=self.call([NATIVE,'-B',str(ROOT/'scripts/run_pr_pauli_candidate_signals.py'),'--request',str(inp),'--output',str(out),'--execute'],d,'PAULI_SIGNALS',out,science=True)
        if result['status']!='SIGNALS_READY':raise Held('PAULI_SIGNAL:'+str(result.get('reason')or result['status']))
        return result

    def df_signals(self,config,cref,d,descriptor=None):
        bundles=self.saved_signal_packets('PR_DF')
        plan=self.service('df-plan',dict(configuration=config,tail_descriptor=descriptor,saved_bundles=bundles),d)
        if plan.get('unsupported_rounds')or any(r['decision']=='UNSUPPORTED'for r in plan['rounds']):raise Held('DF_PROVIDER_UNSUPPORTED:'+str(plan))
        request=dict(configuration=cref,plan=plan,tail_descriptor=descriptor,execution_source=self.source,output_directory=rel(d/'df_signals'),runtime=dict(signal_view=str(ROOT.parent/'d2-pr-df-ld7-signal-source-fefd-20261116')))
        inp=d/'DF_SIGNAL_REQUEST.json';put(inp,request);input_commit=publish([d],'Fix missing DF round jobs')
        output=d/'df_signals/RESULT.json'
        result=self.call([NATIVE,'-B',str(ROOT/'scripts/run_pr_df_candidate_signal_jobs.py'),'--input',str(inp),'--inputs-commit',input_commit,'--source',self.source],d,'DF_SIGNALS',output,science=True)
        if result['status']not in ('REQUESTED_ROUNDS_COMPLETE','SIGNALS_READY','MISSING_ROUNDS_COMPLETE'):raise Held('DF_SIGNAL_RESULT:'+str(result['status']))
        raw=publish([d],'Publish missing DF round signals')
        obtained={str(x['m']):dict(commit=raw,path=x['raw_path'])for x in result['rows']if x.get('raw_path')}
        staged=d/'df_carrier'
        st=self.service('df-stage',dict(configuration=cref,directory=rel(staged),obtained=obtained,tail_descriptor=descriptor,saved_bundles=bundles),d)
        if st['status']!='CARRIERS_READY':raise Held('DF_CARRIER:'+str(st))
        carrier=publish([d],'Fix unified DF raw carrier')
        configref=dict(commit=carrier,path=rel(staged/'CONFIGURATION.json'))
        packet=self.service('df-bundle',dict(configuration_source=configref,tail_descriptor=descriptor,saved_bundles=bundles,raw_sources={str(m):dict(commit=carrier,path=rel(staged/'raw'/('m'+str(m)+'.json')))for m in range(13)}),d)
        pub=publish([d],'Assemble DF bundle for normal intake')
        return dict(status='SIGNALS_READY',configuration_source=configref,packet_source=dict(commit=pub,path=rel(d/'df-bundle.json')),new_signal_calls=result.get('new_provider_calls'))

    def initial_quote(self,config,cref,d):
        registry=d/'PRICE_RULES.json'
        if not registry.exists():put(registry,read(self.registry))
        req=dict(configuration=cref,view=str(D2_VIEW),registry_file=str(registry),policy='MODEL')
        quote=self.service('quote',req,d,tag='INITIAL_QUOTE',python=PRICE)
        return self.quote_with_steps(config,quote)

    def price(self,config,cref,prepared,r,d,initial_quote=None):
        # Pauli stages publish into one owned mutable checkout. Retain its
        # normal worker parallelism, but requery after another candidate has
        # registered the same calibration before launching a duplicate cell.
        lock=getattr(self,'pauli_price_lock',None)
        with (lock if r['method']=='PR_PAULI'and lock is not None else nullcontext()):
            return self._price(config,cref,prepared,r,d,initial_quote)

    def _price(self,config,cref,prepared,r,d,initial_quote=None):
        registry=d/'PRICE_RULES.json'
        req=dict(configuration=cref,view=str(D2_VIEW),registry_file=str(registry),policy='MODEL')
        quote=self.initial_quote(config,cref,d)if initial_quote is None else initial_quote
        quote=self.quote_with_steps(config,quote)
        if not quote['complete']:
            if r['method']=='PR_PAULI':
                inp=d/'PAULI_PRICE_REQUEST.json';out=d/'PAULI_PRICE_RESULT.json'
                put(inp,dict(configuration_source=cref,configuration=config,prepared_item=prepared,price_quote=quote,registry_file=str(registry),workspace=str(self.settings.get('pauli_workspace',A_WORKSPACE)),
                    directory='docs/research/d2_automatic_price_20261116/'+config['id'][:20],private_root=str(ROOT.parent/'d2-auto-private'/config['id'][:20]),d2_view=str(D2_VIEW),interpreter=PRICE,
                    scoped_registration=True,execution_source=self.source))
                try:
                    cmd=[PRICE,'-B',str(ROOT/'scripts/run_pr_pauli_candidate_prices.py'),'--request',str(inp),'--output',str(out),'--execute']
                    if getattr(self,'stage_executor',None)is not None:
                        compiled=self.call(cmd+['--phase','compile','--no-push'],d,'PAULI_COMPILE',out,science=True)
                        if compiled.get('status')not in('COMPILE_READY_FOR_AUDIT','STEP_COMPILES_READY_FOR_AUDIT','PRICES_READY','SCOPED_STEP_MODEL_READY','STEP_MODELS_READY'):
                            raise Held('PAULI_COMPILE:'+str(compiled))
                        result=self.call(cmd+['--phase','audit','--no-push'],d,'PAULI_AUDIT',out,science=True)
                    else:result=self.call(cmd,d,'PAULI_PRICES',out,science=True)
                except Held:
                    # Valid completed groups remain useful to independent
                    # candidates even if a later group reaches its own STOP.
                    if out.exists()and read(out).get('evidence_packages'):
                        self.ingest_price_results(read(out)['evidence_packages'])
                    raise
                if result.get('evidence_packages'):
                    self.ingest_price_results(result['evidence_packages'])
                if result['status']not in('PRICES_READY','PRICE_READY','COMPLETE_MODEL_26_AXES','STEP_MODELS_READY'):
                    raise Held(('PRICE_EVIDENCE_REQUIRED:'if result['status']in('WAITING_PRICE_EVIDENCE','CALIBRATION_PLANNED_NOT_EXECUTED')else'PAULI_PRICE:')+str(result))
            else:self.ingest_price_results(self.df_price(cref,quote,d,registry,r.get('tail_descriptor')))
            quote=self.quote_with_steps(config,quote)
            if not quote['complete']:
                quote=self.service('quote',req,d,tag='REQUOTE',python=PRICE)
                quote=self.quote_with_steps(config,quote)
        put(d/'FINAL_QUOTE.json',quote)
        return quote

    def df_price(self,cref,quote,d,registry,descriptor=None):
        config=value(cref)
        planned=self.service('df-price-design',dict(configuration=config,price_quote=quote),d,tag='DF_PRICE_GROUPS')
        groups=planned.get('groups',[])
        if any(g['status']!='CALIBRATION_DESIGN_READY'for g in groups):
            raise Held('DF_MODEL_DESIGN:'+str(groups))
        packages=[]
        for group in groups:
            with self.shared_preparation_lock(dict(stage='DF_PRICE_GROUP',group_id=group['group_id'])):
                # A different candidate may have completed this exact group
                # while we waited. Requote from its verified registry first.
                refreshed=self.quote_with_steps(config,quote)
                missing={(r['m'],r['axis'])for r in refreshed['rows']if r.get('unit_price_RZ')is None}
                if not any((r['m'],r['axis'])in missing for r in group['axis_keys']):continue
                package=self.df_price_group(cref,refreshed,d/'df_price_groups'/group['group_id'][:20],registry,group,descriptor)
                self.ingest_price_results([package]);packages.append(package)
        return packages

    def df_price_group(self,cref,quote,d,registry,group,descriptor=None):
        """One existing DF calibration/holdout campaign per exact actual step."""
        d.mkdir(parents=True,exist_ok=True);base=d/'df_prices';inp=base/'INPUTS.json'
        args=[PRICE,'-B',str(ROOT/'scripts/run_pr_df_candidate_prices.py')]
        descriptor_file=d/'DF_DESCRIPTOR_LOCATOR.json'
        if descriptor:put(descriptor_file,descriptor)
        quote_file=d/'AVAILABLE_PRICE_QUOTE.json';put(quote_file,quote)
        self.call(args+['prepare']+(['--tail-descriptor',str(descriptor_file)]if descriptor else[])+['--configuration',cref['commit']+':'+cref['path'],'--output',rel(base),'--cohort','D2_AUTO_DF_'+d.name,'--source',self.source,'--price-quote',str(quote_file),'--covered-rounds',json.dumps(group['covered_rounds'])],d,'DF_PRICE_PREPARE')
        if not inp.exists():raise Held('DF_PRICE_RULE_OR_RECEIPT_REUSE_REQUIRED:'+str(read(base/'NEEDS.json')))
        inputs=publish([d],'Fix candidate DF calibration and independent holdout inputs')
        common=['--input',str(inp),'--inputs-commit',inputs,'--source',self.source]
        self.call(args+['acquire',*common,'--stage','calibration'],d,'DF_CALIBRATION',science=True)
        raw=publish([d],'Save DF calibration wrapper receipts')
        self.call(args+['freeze',*common,'--raw-commit',raw],d,'DF_MODEL_FREEZE',science=True)
        frozen=publish([d],'Freeze candidate DF full-wrapper model before holdout')
        holdout=None
        if read(inp)['model_design'].get('holdout_round')is not None:
            self.call(args+['acquire',*common,'--stage','holdout','--model-commit',frozen],d,'DF_HOLDOUT',science=True)
            hold=publish([d],'Save independent DF holdout receipts')
            self.call(args+['compare-holdout',*common,'--raw-commit',hold,'--model-commit',frozen],d,'DF_HOLDOUT_COMPARE',science=True)
            compared=publish([d],'Publish scoped DF independent holdout comparison')
            holdout=dict(commit=compared,path=rel(base/'HOLDOUT_COMPARISON.json'))
        reg=read(registry)
        imp=next(e['implementation']for e in reg['entries']if e['name']=='DF_LD7')
        return dict(kind='PR_DF_STEP_MODEL',configuration_source=cref,implementation=imp,
            inputs_source=dict(commit=inputs,path=rel(inp)),model_source=dict(commit=frozen,path=rel(base/'FROZEN_MODELS.json')),
            price_source=dict(commit=frozen,path=rel(base/'AXIS_PRICES.json')),holdout_source=holdout)

    def common_route(self,method,cid,configuration,packet,binding):
        """The same fixed worker/contract for newly prepared and saved candidates."""
        view=B_VIEW if method=='PR_PAULI'else DF_Q3_VIEW
        return dict(kind='pauli_b_common_entry'if method=='PR_PAULI'else'prdf_d2_common_entry',interpreter=NATIVE,
            view=str(view),view_commit=git('rev-parse','HEAD',cwd=view),configuration_ids=[cid],
            contract_template=(dict(commit='b6d4dea49c39f364048af422bc2928ee7c169261',path='docs/research/pauli_ld5_two_vectors_q3_results_20261116/INPUTS.json')if method=='PR_PAULI'else dict(commit='0defa286c53a1b9acf513be453907bcad83102b3',path='docs/research/d_pr_df_ld7_v0_results_20261116/Q3_INPUTS.json',inputs_selector=['inputs'])),
            configurations={cid:dict(configuration=configuration,signal_packet=packet,
                binding_id=binding['binding_id'],total_configuration_id=binding['total_configuration_id'])})

    def install_saved_routes(self,context,routes):
        """Reconnect catalog-only PR cases through their own source-bound metadata.

        Existing usable routes remain unchanged, preserving prior-job reuse.
        Missing/conflicting metadata is a candidate-specific preparation need;
        neither signals nor numerical results are synthesized here.
        """
        from . import rpe_method_optimizer as O
        reader=context.get('reader')or O.Reader(ROOT)
        points=context.get('catalog',{}).get('candidates',[])
        for c in context['candidates']:
            method=c['method'];cid=c.get('configuration_id')
            if method not in('PR_PAULI','PR_DF')or not cid or not c.get('ready'):continue
            usable=False
            for rule in routes.get(method,[]):
                try:usable=O.route_supports_configuration(rule,cid,reader)
                except (KeyError,ValueError,TypeError,OSError,subprocess.CalledProcessError):continue
                if usable:break
            if usable:continue
            saved=[p for p in points if p.get('method')==method and p.get('configuration_id')==cid]
            try:
                if not saved:continue
                p=saved[0]
                fields=('configuration_source','signal_source','binding_id','total_configuration_id')
                if any(any(row.get(k)!=p.get(k)for k in fields)for row in saved):
                    raise ValueError('SAVED_ROUTE_BINDINGS_CONFLICT')
                if any(p.get(k)is None for k in fields):raise ValueError('SAVED_ROUTE_BINDING_INCOMPLETE')
                if c.get('signal_source')!=p['signal_source']:raise ValueError('SAVED_ROUTE_CANDIDATE_SIGNAL_DIFFERS')
                config=reader.read(p['configuration_source'])
                if config['id']!=cid:raise ValueError('SAVED_ROUTE_CONFIGURATION_ID_DIFFERS')
                inputs=reader.read(p['case_source'])
                cases=[row for row in inputs['cases']if(row.get('original_configuration_id')or row.get('configuration_id'))==cid]
                if not cases:raise ValueError('SAVED_ROUTE_CASE_MISSING')
                rule=self.common_route(method,cid,p['configuration_source'],p['signal_source'],p)
                loc=rule['contract_template'];contract=reader.read(dict(loc,selector=loc.get('inputs_selector',[])))
                if inputs['adopted_evaluation']!=contract['adopted_evaluation']:
                    raise ValueError('SAVED_ROUTE_EVALUATION_CONTRACT_DIFFERS')
                for case in cases:
                    if any(case.get(k)!=p[v]for k,v in [('configuration','configuration_source'),('signal_packet','signal_source'),('binding_id','binding_id'),('total_configuration_id','total_configuration_id')]):
                        raise ValueError('SAVED_ROUTE_CASE_BINDING_DIFFERS')
                    if case.get('evaluation_target')!=contract['cases'][0].get('evaluation_target'):
                        raise ValueError('SAVED_ROUTE_EVALUATION_TARGET_DIFFERS')
                expected=context['expected'].get(cid,{})
                if any(expected.get(k)!=p[k]for k in('binding_id','total_configuration_id')):
                    raise ValueError('SAVED_ROUTE_EXPECTED_BINDING_DIFFERS')
                routes.setdefault(method,[]).insert(0,rule)
                c['execution_binding_source']=p['case_source'];c.pop('execution_binding_hold',None)
            except (KeyError,ValueError,TypeError,OSError,subprocess.CalledProcessError)as exc:
                c['execution_binding_hold']=str(exc)
        return routes

    def install_snapshot(self,context,problem,keys):
        """Install immutable READY rows while the producer continues elsewhere.

        Do not mutate/save the producer's state from the queue thread. The
        existing installer and all physical/binding checks remain unchanged.
        """
        with self.lock:
            records=deepcopy([r for r in self.state['records'].values()if r.get('key')in keys and r['status']=='READY_FOR_SEARCH'])
        snapshot=object.__new__(type(self));snapshot.__dict__.update(self.__dict__)
        snapshot.state=dict(self.state,records={r['key']:r for r in records});snapshot.save=lambda:None
        return snapshot.install(context,problem,records),records

    def install(self,context,problem,records):
        from . import rpe_method_optimizer as O,rpe_candidate_preparation as R
        routes=deepcopy(problem['execution_routes'])
        for rec in records:
            if rec['status']!='READY_FOR_SEARCH':continue
            config=value(rec['configuration']);method=rec['method'];binding=rec['intake'];signal=rec['signal'];cid=config['id']
            requested=(problem.get('methods',{}).get(method,{})or{}).get('requested_delta_pf_values')
            if requested is not None and dec(config['spec']['delta_pf'])not in{dec(v)for v in requested}:
                continue
            # Populate original B2 physical keys from saved metadata only. R.register
            # owns the equivalence/alias decision; no new schedule rule is defined.
            for candidate in context['candidates']:
                if candidate['method']!=method or not candidate.get('configuration_id')or candidate.get('schedule_keys')is not None:continue
                try:existing,_=Selection.configuration_for(context,candidate['configuration_id'],value)
                except ValueError:continue
                candidate['schedule_keys']=O.schedule_keys(existing)
            def same_placeholder(candidate):
                if candidate.get('configuration_id')is not None:return False
                k=candidate.get('key',{})
                if k.get('L_D')!=config['spec']['L_D']or k.get('formula',config['spec']['formula'])!=config['spec']['formula']:
                    return False
                width=(dec(k['delta_pf'])if k.get('delta_pf')is not None else
                       dec(config['spec']['h'])*dec(k.get('delta_over_h','1')))
                return width==dec(config['spec']['delta_pf'])
            old=next((c for c in context['candidates']if c['method']==method and
                (c.get('configuration_id')==cid or same_placeholder(c))),None)
            ready_before={c['configuration_id']:c for c in context['candidates']if c['method']==method and c.get('ready')and c.get('configuration_id')}
            item=dict(status='GENERATED',configuration=deepcopy(config),configuration_id=cid,
                configuration_source=rec['configuration'],L_D=config['spec']['L_D'],
                readiness=dict(configuration_complete=True),requested_changes={})
            # Legacy register() uses a smaller schedule key. Do not let that
            # key alias different dictionaries, targets, SOURCE or law contracts.
            hidden=[];new_keys=O.schedule_keys(config)
            for candidate in context['candidates']:
                if candidate['method']!=method or candidate.get('schedule_keys')!=new_keys:continue
                try:existing,_=Selection.configuration_for(context,candidate['configuration_id'],value)
                except (ValueError,KeyError):existing=None
                if existing is None or physical_evaluation_contract(existing)!=physical_evaluation_contract(config):
                    if candidate.get('configuration_id')==cid:raise ValueError('CANONICAL_CONFIGURATION_PHYSICAL_CONTRACT_DIFFERS')
                    hidden.append((candidate,candidate.get('schedule_keys')));candidate['schedule_keys']=None
            try:
                disposition=R.register(method,old,item,context['candidates'],rec.get('directory',self.directory),
                    key=dict(L_D=config['spec']['L_D'],formula=config['spec']['formula'],
                        delta_over_h=str(dec(config['spec']['delta_pf'])/dec(config['spec']['h']))))
            finally:
                for candidate,keys in hidden:candidate['schedule_keys']=keys
            alias=disposition.split(':',1)[1]if disposition.startswith('PHYSICALLY_IDENTICAL_TO:')else None
            installed=alias or cid
            rec.update(installed_configuration_id=installed,installation_disposition=disposition)
            if alias is not None and installed in ready_before:
                # Keep the original candidate, its own prices/evaluations and route.
                # The new declaration remains visible as B2's physical_alias record.
                continue
            if alias:
                rec.update(status='HELD_PHYSICAL_ALIAS_CANONICAL_NOT_READY',
                    reason='Preserve existing partial prices/Q3; prepared metadata needs rebinding to the original canonical view')
                continue
            cand=next(c for c in context['candidates']if c['method']==method and c.get('configuration_id')==installed)
            cand.update(kappa_value=str(dec(config['spec']['kappa']['value'])),ready=True,needs=[],
                generated=True,generated_configuration=config,configuration_source=rec['configuration'],
                signal_source=signal['packet_source'],prices=[dec(x)for x in rec['prices']],
                price_status='MODEL_REFERENCE',price_quote=value(rec['quote']),
                schedule_keys=O.schedule_keys(config),prepared_by='D2_COMMON_CANDIDATE_COMPLETION')
            context['expected'][cid]=dict(configuration_id=cid,binding_id=binding['binding_id'],total_configuration_id=binding['total_configuration_id'])
            rule=self.common_route(method,cid,signal.get('configuration_source',rec['configuration']),signal['packet_source'],binding)
            routes[method]=[rule]+[old for old in routes.get(method,[])if old!=rule]
        routes=self.install_saved_routes(context,routes)
        self.save()
        return routes


def prepare_for_search(root,problem,context,spec,run,*,selection_results=None,completion=None,stage_executor=None):
    """V2 chooses missing inputs; this callback completes them and returns ready inputs.

    A finite demonstration may bound completions per method; unknown/failed
    stages are recorded and independent candidates retain their V2 request order.
    """
    from . import rpe_method_optimizer as O
    settings=deepcopy(spec['candidate_completion'])
    if not(settings.get('price_registry_source')or settings.get('price_registry_file')):
        supplied=(problem.get('price_rules')or{}).get('registry_file')
        if supplied:settings['price_registry_file']=str((Path(root)/supplied).resolve())
    if spec.get('science_approved')is not True:raise ValueError('Candidate acquisition requires the adopted science scope')
    old_state=Path(run)/'completion/STATE.json'
    settings.setdefault('source',read(old_state)['source']if old_state.exists()else git('rev-parse','HEAD'))
    completion=completion or Completion(Path(run)/'completion',settings)
    if stage_executor is not None:completion.bind_stage_executor(stage_executor)
    completion.ingest_price_results()
    for record in list(completion.state['records'].values()):
        if record['status'].startswith(('HELD','STOP','WAITING')):completion.refresh_price_hold(record)
    ready=[r for r in completion.state['records'].values()if r['status']=='READY_FOR_SEARCH']
    routes=completion.install(context,problem,ready)
    search=dict(problem['search'],**spec.get('search_overrides',{}))
    frozen=dict(methods=spec['methods'],method_scope={m:problem['methods'][m]for m in spec['methods']},
        design=settings.get('new_candidate_design'),max_completed_per_method=settings.get('max_completed_per_method'),source=settings['source'])
    put(completion.directory/'INPUTS.json',frozen);publish([completion.directory/'INPUTS.json'],'Fix finite PR preparation scope and SOURCE')
    def one(method):
        current=[r for r in completion.state['records'].values()if r['method']==method]
        maximum=settings.get('max_completed_per_method')
        if isinstance(maximum,dict):maximum=maximum.get(method)
        if maximum is not None and sum(r['status']=='READY_FOR_SEARCH'for r in current)>=maximum:return current
        attempted={Selection.identity(r['request']['selection'])for r in current if r['request'].get('selection')}
        limit=settings.get('max_attempted_per_method')
        remaining=None if limit is None else max(0,limit-len(current))
        while remaining is None or remaining>0:
            if remaining is not None:remaining-=1
            # Uses the published B2 rule verbatim; no local LD/kappa priority implementation.
            answer=(selection_results or{}).get(method)
            if answer is None:
                answer=O.optimize(root,dict(problem,search=search),context=context,methods=[method],max_new={method:0},science_approved=False,
                    initial=spec.get('initial','all'))[method]
            requests=answer.get('input_requests')or[]
            selection=next((r for r in requests if r['kind']in('NEW_L_D','CONFIGURATION_INPUTS_MISSING','KAPPA_PROPOSAL')and Selection.identity(r)not in attempted),None)
            if selection is None:break
            attempted.add(Selection.identity(selection))
            req=Selection.normalize(method,selection,context,settings.get('new_candidate_design',{}),value)
            put(completion.directory/(method+'_LATEST_SELECTION.json'),req)
            result=completion.complete(req);current.append(result)
            if result['status']=='READY_FOR_SEARCH':break
        return current
    preparation_methods=[m for m in spec['methods']if m in ('PR_PAULI','PR_DF')]
    if preparation_methods:
        with ThreadPoolExecutor(max_workers=len(preparation_methods))as pool:
            for _ in pool.map(one,preparation_methods):pass
    records=list(completion.state['records'].values())
    routes=completion.install(context,problem,records)
    put(completion.directory/'RETURN_TO_SEARCH.json',dict(source=completion.source,records=records,
        ready=[r['configuration_id']for r in records if r['status']=='READY_FOR_SEARCH'],selection_rule=search.get('rule')))
    if settings.get('evaluate_completed_only',False):
        # Integration acceptance scope: V2 still chooses both missing candidates
        # and their shot proposals. Existing ready baselines stay out of new Q3.
        allowed={r.get('installed_configuration_id',r['configuration_id'])for r in records if r['status']=='READY_FOR_SEARCH'}
        context['candidates']=[c for c in context['candidates']if c.get('configuration_id')in allowed]
    return context,routes,completion


def continue_selected_preparation(root,problem,context,spec,run,completion,executor,result,extra_results=(),optimizer=None):
    """Consume subsequent V2 requests with the same cumulative Q3 budget.

    No new chooser or independent scientific worker loop: each continuation is
    the normal optimizer with the same Executor/job keys. Failed requests are
    retained and skipped, while independent missing inputs remain eligible.
    """
    from . import rpe_method_optimizer as O
    cycles=[]
    while True:
        cycles.append(dict(ready=[r.get('configuration_id')for r in completion.state['records'].values()if r['status']=='READY_FOR_SEARCH'],
            methods={m:dict(stop=r['stop_reason'],new_calls=r['new_calls'],launched=executor.launched(m))for m,r in result.items()}))
        put(completion.directory/'SEARCH_CYCLES.json',dict(cycles=cycles,cumulative_budget=spec['max_new_q3_per_method']))
        remaining=[]
        for method in spec['methods']:
            maximum=spec['max_new_q3_per_method'][method]
            if maximum in (None,'UNLIMITED')or executor.launched(method)<maximum:remaining.append(method)
        pending={m:result[m]if m in remaining else dict(input_requests=[])for m in spec['methods']}
        if not any(r.get('input_requests')for r in pending.values()):break
        before={k for k,v in completion.state['records'].items()if v['status']=='READY_FOR_SEARCH'}
        context,routes,_=prepare_for_search(root,problem,context,spec,run,selection_results=pending,completion=completion)
        after={k for k,v in completion.state['records'].items()if v['status']=='READY_FOR_SEARCH'}
        if after==before:break
        executor.routes=routes;executor.expected=context['expected']
        O.apply_published(context['candidates'],routes)
        result=(optimizer or O.optimize)(root,problem,context=context,methods=spec['methods'],initial=spec.get('initial','all'),
            max_new=spec['max_new_q3_per_method'],executor=executor,extra_results=extra_results,science_approved=True)
    return result


def publish_comparison_exports(completion,run,context):
    """Publish each acquired candidate's actual Q3 answer for unchanged C sync."""
    from .rpe_method_optimizer import job_outcome
    run=Path(run);exports=[]
    for record in completion.state['records'].values():
        if record['status']!='READY_FOR_SEARCH':continue
        cid=record['configuration_id'];method=record['method']
        for job in sorted((run/'jobs'/method).glob('*')):
            if not(job/'OUTCOME.json').exists()or not(job/'REQUEST.json').exists():continue
            request=read(job/'REQUEST.json');outcome,outcome_path=job_outcome(job)
            if request['configuration_id']!=cid or outcome['status']not in ('MODEL_BOUND_MEETS_ADOPTED_THRESHOLD','MODEL_BOUND_EXCEEDS_ADOPTED_THRESHOLD','INCONCLUSIVE'):continue
            leaf=job/job.name;result=leaf/'RESULT.json'
            if not result.exists():continue
            raw=publish([job],'Publish automated candidate Q3 raw and individual receipts')
            d=completion.directory/'comparisons'/method/job.name;d.mkdir(parents=True,exist_ok=True)
            req=dict(configuration=record['intake']['configuration'],case_inputs=dict(commit=raw,path=rel(job/'CASE_INPUTS.json')),
                result=dict(commit=raw,path=rel(result)),outcome=dict(commit=raw,path=rel(outcome_path)),api=dict(commit=raw,path=rel(leaf/'API_RETURN.json')),
                quote=record['quote'],comparison_label=method+'_AUTO_'+cid[:12]+'_'+job.name)
            inp=d/'EXPORT_REQUEST.json';put(inp,req);script=str(ROOT/'scripts/export_pr_candidate_comparison.py')
            completion.call([NATIVE,'-B',script,'prices','--request',str(inp),'--output',str(d/'AXIS_PRICES.json')],d,'EXPORT_PRICES')
            pc=publish([d],'Bind automated candidate 26 axis prices for C')
            completion.call([NATIVE,'-B',script,'inputs','--request',str(inp),'--prices',json.dumps(dict(commit=pc,path=rel(d/'AXIS_PRICES.json'))),'--output',str(d/'INPUTS.json')],d,'EXPORT_INPUTS')
            ic=publish([d],'Fix automated candidate cost/Q3 consumer inputs')
            completion.call([NATIVE,'-B',script,'join','--inputs',json.dumps(dict(commit=ic,path=rel(d/'INPUTS.json'))),'--output-directory',str(d)],d,'EXPORT_JOIN')
            jc=publish([d],'Publish automated PR comparison through existing C schema')
            check=d/'C_SYNC_CHECK.json'
            completion.call([NATIVE,'-B',str(ROOT/'scripts/check_pr_candidate_c_sync.py'),'--join',jc+':'+rel(d/'C_RESULT_JOIN.json'),'--comparison-only','--output',str(check)],d,'C_READ_SYNC_CHECK',check)
            exports.append(dict(method=method,configuration_id=cid,job=job.name,source=dict(commit=jc,path=rel(d/'C_RESULT_JOIN.json')),C_sync_check=rel(check)))
    put(completion.directory/'C_EXPORTS.json',dict(exports=exports,source=completion.source,scientific_calls_in_export=0))
    return exports
