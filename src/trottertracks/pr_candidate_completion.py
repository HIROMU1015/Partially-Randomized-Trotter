"""Complete B2-selected PR inputs; never choose LD, kappa, width or shot rules.

Scientific leaves are the published method-specific providers/acquisition paths.
The public API accepts the optimizer's input request/canonical configuration.
"""
from concurrent.futures import ThreadPoolExecutor
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
A_WORKSPACE=ROOT.parent/'d2-pauli-auto-evidence-20261116'
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
        self.registry=self.directory/'PRICE_RULES_REGISTRY.json'
        if not self.registry.exists():
            initial=value(dict(commit='c1f95e0806f5478a1ee5ce9ecb148ce9d10461aa',path='configs/pr_price_rules/h4_saved_rules.json'))
            extra=value(dict(commit='54b4353b9',path='docs/research/b2_pauli_calibration_20261116/PRICE_RULES_REGISTRY_EXTENDED.json'))
            by={e['name']:e for e in initial['entries']};by.update({e['name']:e for e in extra['entries']})
            if imported:
                previous=Path(imported['path']).with_name('PRICE_RULES_REGISTRY.json')
                saved_registry=value(dict(commit=imported['commit'],path=str(previous)))
                by.update({e['name']:e for e in saved_registry['entries']})
            put(self.registry,dict(initial,entries=list(by.values())))

    def save(self):
        with self.lock:put(self.state_path,self.state)

    def log(self,**event):
        with self.lock:
            with(self.directory/'PROGRESS.jsonl').open('a')as f:f.write(json.dumps(dict(epoch=time.time(),**event),default=str)+'\n')

    def call(self,cmd,d,stage,output=None,science=False):
        d=Path(d);d.mkdir(parents=True,exist_ok=True);done=d/(stage+'.RETURN.json');started=d/(stage+'.STARTED.json')
        if done.exists():
            r=read(done)
            if r['returncode']!=0:raise Held('PRIOR_STAGE_FAILED_NO_RETRY:'+stage)
            return read(output)if output and Path(output).exists()else r
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

    def complete(self,request):
        identifier=key(request);d=self.directory/'candidates'/identifier
        with self.lock:
            old=self.state['records'].get(identifier)
            if old and old['status']=='READY_FOR_SEARCH':return dict(old,resumed=True)
            if old and old['status'].startswith(('HELD','STOP')):return dict(old,resumed=True)
            rec=dict(method=request['method'],selection=request.get('selection'),request=request,status='SELECTED',key=identifier,directory=rel(d))
            self.state['records'][identifier]=rec;self.save()
        self.log(event='SELECTED_BY_B2',method=rec['method'],selection=rec['selection'])
        try:
            if request.get('selection_error'):raise Held('B2_SELECTION_NORMALIZATION:'+request['selection_error'])
            if request['method']=='PR_PAULI':config,prepared=self.pauli_configuration(request,d)
            else:
                generated=self.service('df-generate',request,d)
                if generated['status']=='NEEDS_DF_TAIL_DESCRIPTOR':
                    prefix=request['design']['L_D'];descriptor=d/'DF_TAIL_DESCRIPTOR.json'
                    self.call([NATIVE,'-B',str(ROOT/'scripts/derive_pr_df_candidate_descriptor.py'),'--prefix',str(prefix),'--configuration','0defa286c53a1b9acf513be453907bcad83102b3:docs/research/d_pr_df_ld7_v0_results_20261116/generation/CONFIGURATION.json','--output',str(descriptor),'--source',self.source],d,'DF_DESCRIPTOR',descriptor,science=True)
                    dc=publish([d],'Fix derived DF tail descriptor and scientific SOURCE')
                    request['tail_descriptor']=dict(commit=dc,path=rel(descriptor))
                    generated=self.service('df-generate',request,d,tag='DF_GENERATE_WITH_DESCRIPTOR')
                if generated['status']!='CONFIGURATION_READY':raise Held('CONFIGURATION:'+str(generated))
                config=generated['configuration'];prepared=None
            if request.get('design')and dec(config['spec']['kappa']['value'])!=dec(request['design']['kappa']):raise Held('ABSOLUTE_KAPPA_MISMATCH')
            put(d/'CONFIGURATION.json',config);commit=publish([d],'Fix selected PR candidate configuration')
            cref=dict(commit=commit,path=rel(d/'CONFIGURATION.json'));rec.update(configuration=cref,configuration_id=config['id'])
            initial_quote=None
            if request['method']=='PR_PAULI':signal=self.pauli_signals(config,prepared,d)
            else:
                initial_quote=self.initial_quote(config,cref,d)
                if not initial_quote['complete']:
                    design=self.service('df-price-design',dict(configuration=config),d)
                    if design['status']!='CALIBRATION_DESIGN_READY':raise Held('DF_MODEL_DESIGN:'+str(design.get('reasons')))
                signal=self.df_signals(config,cref,d,request.get('tail_descriptor'))
            rec.update(signal=signal);self.save()
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
        except Held as exc:rec.update(status='HELD_WITH_REASON',reason=str(exc))
        except Exception as exc:rec.update(status='HELD_TECHNICAL_FAILURE',reason=type(exc).__name__+': '+str(exc))
        self.save();self.log(event='CANDIDATE_COMPLETED',method=request['method'],status=rec['status'],reason=rec.get('reason'));return rec

    def pauli_configuration(self,r,d):
        workspace=Path(self.settings.get('pauli_workspace',A_WORKSPACE));d.mkdir(parents=True,exist_ok=True)
        saved=[dict(commit='d7a976a629a6db68197b24f8f86734e109c8bf5d',path=p)for p in (
            'docs/research/rpe_split_preparation_20261116/LD4/PREPARATION_P1.0.CONFIGURATION.json',
            'docs/research/rpe_split_preparation_20261116/LD6/PREPARATION.0.CONFIGURATION.json')]
        def prepare(item,tag,science=False):
            inp=d/(tag+'.REQUEST.json');out=d/(tag+'.json')
            put(inp,dict(kind='pauli_native',items=[item],saved_configurations=saved,saved_packets=self.settings.get('saved_pauli_packets',[])))
            cmd=[NATIVE,'-B',str(workspace/'scripts/prepare_pauli_candidate.py'),'--request',str(inp),'--output',str(out),'--cache',str(d/'SPLITS.json')]
            if science:
                publish([d],'Fix selected Pauli P0 request')
                cmd+=['--prepare-missing','--source',git('rev-parse','HEAD',cwd=workspace),'--p0-directory',str(workspace/'docs/research/d2_auto_p0_20261116')]
            response=self.call(cmd,d,tag,out,science=science);return response['items'][0]
        if r.get('configuration'):
            config=value(r['configuration'])if 'commit'in r['configuration']else r['configuration']
            item=prepare(dict(item_id='B2_SELECTED',configuration=config),'A_CANONICAL')
        else:
            design=r['design'];item=prepare(dict(item_id='B2_SELECTED_BASE',configuration=dict(L_D=design['L_D'])),'A_BASE')
            if item['status']=='NEEDS_SPLIT_PREPARATION':
                if design['L_D']in(7,8,9):raise Held('PRIOR_PAULI_P0_NUMERICAL_GATE_FAILURE_NO_RETRY:'+str(design['L_D']))
                item=prepare(dict(item_id='B2_SELECTED_BASE',configuration=dict(L_D=design['L_D'])),'A_BASE_P0',True)
            if item['status']!='GENERATED':raise Held('A_PREPARATION:'+str(item))
            base=item['configuration'];changes=dict(formula=design['formula'],K=design.get('K',2),kappa_multiplier=str(dec(design['kappa'])/dec(base['spec']['kappa']['value'])))
            if design.get('delta_pf')is not None:changes['delta_pf']=design['delta_pf']
            else:changes['delta_over_h']=design['delta_over_h']
            item=prepare(dict(item_id='B2_SELECTED',configuration=base,changes=changes),'A_DESIGN')
        if item['status']!='GENERATED':raise Held('A_CONFIG_NOT_READY:'+str(item))
        return item['configuration'],item

    def pauli_signals(self,config,prepared,d):
        packets=list(self.settings.get('saved_pauli_packets',[]))
        for record in self.state['records'].values():
            loc=(record.get('signal')or{}).get('packet_source')
            if record['method']=='PR_PAULI'and loc and loc not in packets:packets.append(loc)
        descriptor=dict(prefix=config['spec']['L_D'],binding=config['spec']['binding'],
            descriptor=prepared['descriptor']['descriptor'],source=prepared['descriptor']['source'])
        request=dict(configuration=config,tail_descriptor=descriptor,saved_packets=packets,
            workspace=str(self.settings.get('pauli_workspace',A_WORKSPACE)),directory='docs/research/d2_automatic_pr_20261116/'+config['id'][:20]+'/'+self.settings.get('signal_epoch','descriptor_envelope_v2'),interpreter=NATIVE)
        inp=d/'PAULI_SIGNAL_REQUEST.json';out=d/'PAULI_SIGNAL_RESULT.json';put(inp,request)
        result=self.call([NATIVE,'-B',str(ROOT/'scripts/run_pr_pauli_candidate_signals.py'),'--request',str(inp),'--output',str(out),'--execute'],d,'PAULI_SIGNALS',out,science=True)
        if result['status']!='SIGNALS_READY':raise Held('PAULI_SIGNAL:'+str(result.get('reason')or result['status']))
        return result

    def df_signals(self,config,cref,d,descriptor=None):
        bundles=[record['signal']['packet_source']for record in self.state['records'].values()if record['method']=='PR_DF'and(record.get('signal')or{}).get('packet_source')]
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
        return self.service('quote',req,d,tag='INITIAL_QUOTE',python=PRICE)

    def price(self,config,cref,prepared,r,d,initial_quote=None):
        registry=d/'PRICE_RULES.json'
        req=dict(configuration=cref,view=str(D2_VIEW),registry_file=str(registry),policy='MODEL')
        quote=self.initial_quote(config,cref,d)if initial_quote is None else initial_quote
        if not quote['complete']:
            if r['method']=='PR_PAULI':
                inp=d/'PAULI_PRICE_REQUEST.json';out=d/'PAULI_PRICE_RESULT.json'
                put(inp,dict(configuration_source=cref,configuration=config,prepared_item=prepared,price_quote=quote,registry_file=str(registry),workspace=str(self.settings.get('pauli_workspace',A_WORKSPACE)),
                    directory='docs/research/d2_automatic_price_20261116/'+config['id'][:20],private_root=str(ROOT.parent/'d2-auto-private'/config['id'][:20]),d2_view=str(D2_VIEW),interpreter=PRICE))
                result=self.call([PRICE,'-B',str(ROOT/'scripts/run_pr_pauli_candidate_prices.py'),'--request',str(inp),'--output',str(out),'--execute'],d,'PAULI_PRICES',out,science=True)
                if result['status']not in('PRICES_READY','PRICE_READY','COMPLETE_MODEL_26_AXES'):raise Held('PAULI_PRICE:'+str(result))
            else:self.df_price(cref,quote,d,registry,r.get('tail_descriptor'))
            quote=self.service('quote',req,d,tag='REQUOTE',python=PRICE)
        put(d/'FINAL_QUOTE.json',quote)
        return quote

    def df_price(self,cref,quote,d,registry,descriptor=None):
        base=d/'df_prices';inp=base/'INPUTS.json'
        args=[PRICE,'-B',str(ROOT/'scripts/run_pr_df_candidate_prices.py')]
        descriptor_file=d/'DF_DESCRIPTOR_LOCATOR.json'
        if descriptor:put(descriptor_file,descriptor)
        self.call(args+['prepare']+(['--tail-descriptor',str(descriptor_file)]if descriptor else[])+['--configuration',cref['commit']+':'+cref['path'],'--output',rel(base),'--cohort','D2_AUTO_DF_'+d.name,'--source',self.source,'--price-quote',str(d/'INITIAL_QUOTE.json')],d,'DF_PRICE_PREPARE')
        if not inp.exists():raise Held('DF_PRICE_RULE_OR_RECEIPT_REUSE_REQUIRED:'+str(read(base/'NEEDS.json')))
        inputs=publish([d],'Fix candidate DF calibration and independent holdout inputs')
        common=['--input',str(inp),'--inputs-commit',inputs,'--source',self.source]
        self.call(args+['acquire',*common,'--stage','calibration'],d,'DF_CALIBRATION',science=True)
        raw=publish([d],'Save DF calibration wrapper receipts')
        self.call(args+['freeze',*common,'--raw-commit',raw],d,'DF_MODEL_FREEZE',science=True)
        frozen=publish([d],'Freeze candidate DF full-wrapper model before holdout')
        self.call(args+['acquire',*common,'--stage','holdout','--model-commit',frozen],d,'DF_HOLDOUT',science=True)
        hold=publish([d],'Save independent DF holdout receipts')
        self.call(args+['compare-holdout',*common,'--raw-commit',hold,'--model-commit',frozen],d,'DF_HOLDOUT_COMPARE',science=True)
        model=read(base/'FROZEN_MODELS.json');config=value(cref);reg=read(registry)
        imp=next(e['implementation']for e in reg['entries']if e['name']=='DF_LD7')
        entry=dict(name='AUTO_DF_'+config['id'][:16],configuration_source=cref,implementation=imp,
            price_source=dict(commit=frozen,path=rel(base/'AXIS_PRICES.json')),model_source=dict(commit=frozen,path=rel(base/'FROZEN_MODELS.json')),
            mode=model['mode'],calibration_domain=dict(q=[min(config['rounds'][m]['q_PF']for m in read(inp)['model_design']['calibration_rounds']),max(config['rounds'][m]['q_PF']for m in read(inp)['model_design']['calibration_rounds'])],R=[min(config['rounds'][m]['R_outer']for m in read(inp)['model_design']['calibration_rounds']),max(config['rounds'][m]['R_outer']for m in read(inp)['model_design']['calibration_rounds'])]),
            caveats=['candidate-owned finite-cohort model; independent holdout is descriptive; no all-law E[C]/CI'])
        reg['entries']=[e for e in reg['entries']if e['name']!=entry['name']]+[entry];put(registry,reg)

    def install(self,context,problem,records):
        from . import rpe_method_optimizer as O,rpe_candidate_preparation as R
        routes=deepcopy(problem['execution_routes'])
        for rec in records:
            if rec['status']!='READY_FOR_SEARCH':continue
            config=value(rec['configuration']);method=rec['method'];binding=rec['intake'];signal=rec['signal'];cid=config['id']
            # Populate original B2 physical keys from saved metadata only. R.register
            # owns the equivalence/alias decision; no new schedule rule is defined.
            for candidate in context['candidates']:
                if candidate['method']!=method or not candidate.get('configuration_id')or candidate.get('schedule_keys')is not None:continue
                try:existing,_=Selection.configuration_for(context,candidate['configuration_id'],value)
                except ValueError:continue
                candidate['schedule_keys']=O.schedule_keys(existing)
            old=next((c for c in context['candidates']if c['method']==method and(c['configuration_id']==cid or(c['configuration_id']is None and c['key']['L_D']==config['spec']['L_D']))),None)
            ready_before={c['configuration_id']:c for c in context['candidates']if c['method']==method and c.get('ready')and c.get('configuration_id')}
            item=dict(status='GENERATED',configuration=deepcopy(config),configuration_id=cid,
                configuration_source=rec['configuration'],L_D=config['spec']['L_D'],
                readiness=dict(configuration_complete=True),requested_changes={})
            disposition=R.register(method,old,item,context['candidates'],rec.get('directory',self.directory),
                key=dict(L_D=config['spec']['L_D'],formula=config['spec']['formula'],
                    delta_over_h=str(dec(config['spec']['delta_pf'])/dec(config['spec']['h']))))
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
            rule=dict(kind='pauli_b_common_entry'if method=='PR_PAULI'else'prdf_d2_common_entry',interpreter=NATIVE,
                view=str(B_VIEW if method=='PR_PAULI'else DF_Q3_VIEW),view_commit=git('rev-parse','HEAD',cwd=B_VIEW if method=='PR_PAULI'else DF_Q3_VIEW),
                configuration_ids=[cid],contract_template=(dict(commit='b6d4dea49c39f364048af422bc2928ee7c169261',path='docs/research/pauli_ld5_two_vectors_q3_results_20261116/INPUTS.json')if method=='PR_PAULI'else dict(commit='0defa286c53a1b9acf513be453907bcad83102b3',path='docs/research/d_pr_df_ld7_v0_results_20261116/Q3_INPUTS.json',inputs_selector=['inputs'])),
                configurations={cid:dict(configuration=signal.get('configuration_source',rec['configuration']),signal_packet=signal['packet_source'],binding_id=binding['binding_id'],total_configuration_id=binding['total_configuration_id'])})
            routes[method]=[rule]+routes.get(method,[])
        self.save()
        return routes


def prepare_for_search(root,problem,context,spec,run,*,selection_results=None,completion=None):
    """V2 chooses missing inputs; this callback completes them and returns ready inputs.

    A finite demonstration may bound completions per method; unknown/failed
    stages are recorded and independent candidates retain their V2 request order.
    """
    from . import rpe_method_optimizer as O
    settings=deepcopy(spec['candidate_completion'])
    if spec.get('science_approved')is not True:raise ValueError('Candidate acquisition requires the adopted science scope')
    old_state=Path(run)/'completion/STATE.json'
    settings.setdefault('source',read(old_state)['source']if old_state.exists()else git('rev-parse','HEAD'))
    completion=completion or Completion(Path(run)/'completion',settings)
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
        limit=settings.get('max_attempted_per_method',9)
        for _ in range(max(0,limit-len(current))):
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
    with ThreadPoolExecutor(max_workers=len(spec['methods']))as pool:
        for _ in pool.map(one,spec['methods']):pass
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


def continue_selected_preparation(root,problem,context,spec,run,completion,executor,result,extra_results=()):
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
        result=O.optimize(root,problem,context=context,methods=spec['methods'],initial=spec.get('initial','all'),
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
