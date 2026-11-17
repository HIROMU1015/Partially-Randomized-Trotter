"""Optional model-level checks; prices, scientific eligibility and search stay unchanged.

The existing PriceRules family/point functions define the physical domain.  A
single durable record is shared by the controller repository, not by allocation,
run or worker. No lease expiry/retry, fitting, compiler, or Q3 lives here.
"""
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import sqlite3
import subprocess
import threading

AXES = ('X', 'Y')
LABELS = {'CALIBRATION_ONLY':'較正のみ', 'PENDING':'検証待ち',
          'INDEPENDENT_MEASURED':'独立実測あり', 'INCOMPLETE':'検証未完了'}


def frac(v):
    return Fraction(int(v['numerator']), int(v['denominator'])) if isinstance(v,dict) else Fraction(v)


def enc(v):
    v=frac(v); return dict(numerator=v.numerator,denominator=v.denominator)


def canonical(v):
    if isinstance(v,dict):
        if set(v)=={'numerator','denominator'}:return enc(v)
        return {k:canonical(x)for k,x in sorted(v.items())}
    if isinstance(v,(list,tuple)):return [canonical(x)for x in v]
    if isinstance(v,Fraction):return enc(v)
    if isinstance(v,float):return enc(v)
    return v


def digest(v):
    return hashlib.sha256(json.dumps(canonical(v),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _coefficients(model):
    if 'models'in model:
        return {a:m.get('coefficients', {k:v for k,v in m.items()if k in('alpha','beta')}) for a,m in model['models'].items()}
    return model.get('coefficients')


def describe(rules, configuration, name, quote):
    """Expose frozen model/domain/evidence locators from an existing PriceRules.

    Read no holdout observations here: Service resolves them once, on first
    actual model use. DIRECT-only axes deliberately do not create a request.
    """
    e=next(e for e in rules.entries if e['name']==name)
    # Use functions from the fixed implementation, including its exact numbers.
    api=rules.quote.__func__.__globals__; groups={}
    rows={(r['m'],r['axis']):r for r in quote['rows']}
    for r in configuration['rounds']:
        used=[rows[r['m'],a]for a in AXES if rows[r['m'],a].get('classification')=='MODEL_REFERENCE']
        if not used:continue
        p=api['point'](r)
        identity=dict(family=e['family'],actual_step=p['step'],piece_pattern=p['piece_pattern'],
            event_recipe=p['event_recipe'],mode=e.get('mode'),coefficients=_coefficients(e.get('model',{})),
            model_source=e.get('model_source'))
        mid=digest(identity)
        if mid not in groups:
            # Calibration cells must be known before selecting a new circuit.
            calibration=[dict(m=d.get('m'),point=d.get('point'),source=d.get('source'))for d in e.get('direct',[])if d.get('role','').startswith(('CALIBRATION','OFF_SCHEDULE_CALIBRATION'))]
            for entry in e.get('model',{}).get('inputs',[]):
                rec=entry.get('record',entry);f=rec.get('features',{})
                q=f.get('q',f.get('pf_step_count'));qr=f.get('qR',f.get('event_count'))
                if q is not None and qr is not None:calibration.append(dict(q=int(frac(q)),R=int(frac(qr)/frac(q)),source=rec.get('source')))
            groups[mid]=dict(model_id=mid,identity=canonical(identity),registry=name,
                calibration_sources=[e[k]for k in('model_source','calibration_evidence','calibration_handoff','direct_path_means')if e.get(k)],
                calibration_points=calibration, evidence_sources=[deepcopy(d)for d in e.get('direct_sources',[])if d.get('role','').startswith('INDEPENDENT')],
                calibration_domain=e.get('calibration_domain'),configuration_source=e.get('configuration_source'),
                evidence_points=[dict(m=m,q=v['q'],R=v['R'])for m,v in e['points'].items()],
                configuration_id=configuration['id'],points=[],qualified_for_candidate=False)
        groups[mid]['points'].append(dict(m=r['m'],q=p['q'],R=p['R'],physical_id=r['candidate']['id'],
            T=r['T'],actual_step=p['step'],round=deepcopy(r),
            predicted_RZ={a:rows[r['m'],a].get('unit_price_RZ')for a in AXES}))
    return list(groups.values())


def describe_steps(registry,configuration,quote):
    """Reuse verified actual-step registry models, without another fitter."""
    result=[]
    for e in registry.state['entries']:
        if e['mode']=='DIRECT_FINITE_COHORT':continue
        used=[r for r in quote['rows']if r.get('registered_rule')=='ACTUAL_STEP:'+e['id'] and r.get('classification')=='MODEL_REFERENCE']
        if not used:continue
        domain=e['domain'];identity=dict(family=domain['family'],actual_step=domain['actual_step'],
            piece_pattern=domain['piece_pattern'],event_recipe=domain['event_recipe'],mode=e['mode'],
            coefficients=_coefficients(e['model']),model_source=e['provenance'].get('model_source')or e['provenance'].get('audit_source'))
        source=e['provenance'].get('holdout_source')
        result.append(dict(model_id=digest(identity),identity=canonical(identity),registry='ACTUAL_STEP:'+e['id'],
            calibration_sources=[v for k,v in e['provenance'].items()if v and k!='holdout_source'],
            calibration_points=e['model'].get('calibration',[]),
            evidence_sources=[]if not source else [dict(source=source,role='INDEPENDENT_HOLDOUT')],
            configuration_id=configuration['id'],configuration_source=None,calibration_domain=e.get('calibration_domain'),
            points=[dict(m=m,q=configuration['rounds'][m]['q_PF'],R=configuration['rounds'][m]['R_outer'],
                physical_id=configuration['rounds'][m]['candidate']['id'],round=deepcopy(configuration['rounds'][m]),
                predicted_RZ={a:next(r['unit_price_RZ']for r in quote['rows']if r['m']==m and r['axis']==a)for a in AXES})
                for m in sorted({r['m']for r in used})],qualified_for_candidate=False))
    return result


def saved_checks(description,reader):
    """Normalize supplied post-freeze comparisons. Never refit or infer a PASS."""
    points={};issues=[]
    for group in description.get('evidence_sources',[]):
        try:
            data=reader.read(group['source']);container=None
            if isinstance(data,dict):container=data;data=data.get('rows',data.get('units',[]))
            if not container:
                container=reader.read(dict(group['source'],selector=[]))
            for i,r in enumerate(data):
                if r.get('valid',True)is not True:continue
                if group.get('configuration_id')and r.get('configuration_id',group['configuration_id'])!=group['configuration_id']:continue
                if group.get('model_group')and r.get('model_group')!=group['model_group']:continue
                m=r.get('m',group.get('m'));axis=r.get('axis')
                if group.get('model_groups')and (r.get('model_group_id')or r.get('model_group'))!=group['model_groups'].get(axis):continue
                if m is None or axis not in AXES:continue
                observed=r.get('O',r.get('observed',r.get('observed_RZ',r.get('mean_C'))))
                predicted=r.get('P',r.get('prediction',r.get('predicted_RZ',r.get('predicted',r.get('frozen_prediction')))))
                relative=r.get('signed_relative_error',r.get('signed_relative_difference'))
                if observed is None or predicted is None:continue
                # A comparison for different frozen coefficients is not evidence.
                ref=r.get('prediction_source')or(container.get('prediction_source')if isinstance(container,dict)else None)or{}
                model=description['identity']['model_source']or{}
                if ref and (ref.get('commit'),ref.get('path'))!=(model.get('commit'),model.get('path')):
                    doc=reader.read(dict(ref,selector=[]))
                    candidates=[doc.get('model_source'),doc.get('model'),(doc.get('binding')or{}).get('model_source')]
                    locator=lambda x:(x.get('commit'),x.get('path'),x.get('selector')or[])if isinstance(x,dict)else None
                    if locator(model)not in [locator(x)for x in candidates]:
                        issues.append('DIFFERENT_FROZEN_PREDICTION_SOURCE');continue
                if relative is not None and frac(relative)!=(frac(predicted)-frac(observed))/frac(observed):
                    raise ValueError('Saved comparison relative difference differs')
                src=dict(group['source'],selector=group['source'].get('selector',[])+[i])
                paths=r.get('paths',r.get('receipts'))or[]
                if not paths and group.get('format')=='PF_HOLDOUT':
                    paths=[dict(index=0,axis=axis,compiled_RZ=observed,receipt=r.get('receipt')or src)]
                if not paths and isinstance(container,dict):
                    units=container.get('receipt_units',[])
                    if isinstance(units,dict):
                        units=[item for value in units.values()for item in(value if isinstance(value,list)else[value])]
                    paths=[x for x in units if isinstance(x,dict) and x.get('m',m)==m and x.get('axis')==axis]
                anchor=next((p for p in description.get('evidence_points',description['points'])if p['m']==m),None)
                p=points.setdefault(m,dict(m=m,q=r.get('q',None if anchor is None else anchor['q']),
                    R=r.get('R',None if anchor is None else anchor['R']),source=src,
                    observations=[],predicted_RZ={},observed_mean_RZ={},signed_relative_difference={},independent_paths=None))
                p['predicted_RZ'][axis]=enc(predicted);p['observed_mean_RZ'][axis]=enc(observed)
                p['signed_relative_difference'][axis]=enc((frac(predicted)-frac(observed))/frac(observed))
                for path in paths:
                    if path.get('valid',True)is not True:continue
                    p['observations'].append(dict(path_id=str(path.get('index',path.get('path_id'))),axis=axis,
                        compiled_RZ=path.get('compiled_RZ',path.get('C')),source=path.get('receipt',path.get('source',src))))
                n=r.get('n_paths',len(paths))
                if not n:issues.append('INDEPENDENT_PATH_COUNT_NOT_SOURCED:m'+str(m))
                p['independent_paths']=n if p['independent_paths']is None else min(n,p['independent_paths'])
        except (ValueError,KeyError,TypeError,subprocess.CalledProcessError)as error:
            issues.append(type(error).__name__+':'+str(error)[:200])
    return [p for p in points.values()if set(p['predicted_RZ'])==set(AXES) and p.get('independent_paths',0)>0],issues


class ModelStore:
    """One controller-repository namespace, shared by all runs/remote workers.

    No expiring lease: a lost owner remains PENDING/INCOMPLETE, never retried.
    Other controllers/clones must point at the same controller-owned store.
    """
    def __init__(self,path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.connect()as db:
            db.execute('CREATE TABLE IF NOT EXISTS models(id TEXT PRIMARY KEY, identity TEXT NOT NULL, record TEXT NOT NULL)')
    def connect(self):return sqlite3.connect(self.path,timeout=30)
    def register(self,desc,resolve):
        # Serialize the first-use evidence scan AND reservation, not just launch.
        with self.connect()as db:
            db.execute('BEGIN IMMEDIATE')
            if digest(desc['identity'])!=desc['model_id']:raise ValueError('Model identity digest differs')
            old=db.execute('SELECT record FROM models WHERE id=?',(desc['model_id'],)).fetchone()
            if old:
                record=json.loads(old[0])
                if record['identity']!=desc['identity']:raise ValueError('Shared model SOURCE/domain differs')
                return record
            resolved=resolve();checks,issues=resolved[:2];existing=resolved[2]if len(resolved)>2 else None
            status='PENDING'if existing else'INDEPENDENT_MEASURED'if checks else 'INCOMPLETE'if issues else 'CALIBRATION_ONLY'
            record={k:deepcopy(v)for k,v in desc.items()if k not in('points','configuration_id','configuration_source')}
            record.update(status=status,label=LABELS[status],validation_points=checks,issues=issues,evidence_parser_version=3)
            if existing:record.update(external_job=existing,pending_reason='REUSE_EQUIVALENT_ACTIVE_OR_WAITING_JOB')
            db.execute('INSERT INTO models VALUES(?,?,?)',(desc['model_id'],json.dumps(desc['identity'],sort_keys=True),json.dumps(record)))
            return record
    def row(self,mid):
        with self.connect()as db:
            row=db.execute('SELECT record FROM models WHERE id=?',(mid,)).fetchone()
            return None if row is None else json.loads(row[0])
    def change(self,mid,fn):
        with self.connect()as db:
            db.execute('BEGIN IMMEDIATE');record=json.loads(db.execute('SELECT record FROM models WHERE id=?',(mid,)).fetchone()[0])
            fn(record);record['label']=LABELS[record['status']]
            db.execute('UPDATE models SET record=? WHERE id=?',(json.dumps(record),mid));return record


def representative(desc,vector):
    known={p.get('m')for p in desc.get('calibration_points',[])if p.get('m')is not None}
    cells={(p.get('q',p.get('point',{}).get('q')if p.get('point')else None),p.get('R',p.get('point',{}).get('R')if p.get('point')else None))for p in desc.get('calibration_points',[])}
    candidates=[p for p in desc['points']if p['m']not in known and (p['q'],p['R'])not in cells and all(p['predicted_RZ'].get(a)is not None for a in AXES)]
    if not candidates:return None
    # Quoted full-wrapper size is an existing compile-load proxy, not a
    # performance bound. Comparable loads = within twice the cheapest one.
    load=lambda p:max(frac(v)for v in p['predicted_RZ'].values())
    cheapest=min(map(load,candidates));eligible=[p for p in candidates if load(p)<=2*cheapest]
    contribution=lambda p:sum(vector[2*p['m']+i]*frac(p['predicted_RZ'][a])for i,a in enumerate(AXES))
    selected=deepcopy(max(eligible,key=lambda p:(contribution(p),-load(p),-p['m'])))
    selected['selection_reason']=dict(rule='NON_CALIBRATION_COMPARABLE_WRAPPER_LOAD_THEN_WEIGHTED_CONTRIBUTION',
        cheapest_load=enc(cheapest),maximum_comparable_load=enc(2*cheapest),allocation_vector=list(vector),
        selected_weighted_contribution=enc(contribution(selected)),compile_load_proxy='MAX_QUOTED_XY_RZ')
    return selected


class Service:
    def __init__(self,root,run,spec,executor=None,problem=None):
        self.root=Path(root);self.run=Path(run);self.spec=spec;self.executor=executor;self.problem=problem or{}
        settings=spec.get('price_validation')or{};self.settings=settings
        common=subprocess.check_output(['git','rev-parse','--git-common-dir'],cwd=root,text=True).strip()
        default=(self.root/common).resolve()/'rpe-price-validations.sqlite3'
        self.store=ModelStore(settings.get('store',default));self.descriptions={};self.applications={};self.lock=threading.RLock()
        self.stopped=threading.Event();self.thread=None;self.evidence_cache={};self.issues={}
        self.implementation_source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=self.root,text=True).strip()
        self.dispatch_lock=threading.Lock()
    def observe(self,candidates):
        from .pr_price_rules import GitReader
        reader=GitReader(self.root)
        with self.lock:
            for c in candidates:
                if not c.get('prices'):continue
                descriptions=c.get('price_model_validations')or(c.get('price_quote')or{}).get('model_validations',[])
                if not descriptions and c.get('family')=='PF':
                    from .rpe_pf_price_validation import describe_candidate
                    try:descriptions=describe_candidate(self.root,c,reader)
                    except (ValueError,KeyError,TypeError)as error:
                        self.issues[c['configuration_id']]=dict(status='INCOMPLETE',label=LABELS['INCOMPLETE'],reason=str(error))
                        c['price_validation_issue']=self.issues[c['configuration_id']];continue
                    c['price_model_validations']=descriptions
                for desc in descriptions:
                    mid=desc['model_id']
                    first=mid not in self.descriptions
                    old=self.store.row(mid)if first else None
                    self.store.register(desc,lambda d=desc:self.resolve(d,reader))
                    if first and old and(old['status']!='INDEPENDENT_MEASURED'or old.get('evidence_parser_version')!=3):
                        checks,issues,external=self.resolve(desc,reader)
                        if checks:self.store.change(mid,lambda r:r.update(status='INDEPENDENT_MEASURED',validation_points=checks,issues=issues,evidence_parser_version=3))
                    self.descriptions[mid]=(desc,c)
                    self.applications[c['configuration_id'],mid]=dict(configuration_id=c['configuration_id'],model_id=mid,
                        points=[{k:p.get(k)for k in('m','q','R','physical_id')}for p in desc['points']],
                        prices=[enc(v)for v in c['prices']],signal_source=c.get('signal_source'))
                    c.setdefault('price_validation',{})[mid]=dict(model_id=mid,status=self.store.row(mid)['status'])
            self.export()
    def resolve(self,desc,reader):
        checks,issues=saved_checks(desc,reader)
        existing=[]
        for source in self.settings.get('existing_job_sources',[]):
            value=reader.read(source);existing+=value if isinstance(value,list)else value.get('jobs',[])
        if self.executor is not None and hasattr(self.executor,'store'):
            existing+=self.executor.store.rows()
        for job in existing:
            contract=job.get('contract')or{};model=contract.get('model_id',job.get('model_id'))
            if model==desc['model_id'] and job.get('state',job.get('status'))in('QUEUED','ASSIGNED','STARTING','RUNNING','UNKNOWN','COMPLETED_PENDING_TRANSFER','PENDING'):
                return checks,issues,job
        # Fixed C/publication indexes from normal sync are optional reusable
        # evidence. No scanning private host directories or large raw trees.
        sources=list(self.settings.get('published_validation_sources',[]))
        if not sources and not checks:
            ref='refs/heads/codex/h4-saved-rpe-cost-catalog-20261116'
            try:
                head=subprocess.check_output(['git','rev-parse','--verify',ref],cwd=self.root,stderr=subprocess.DEVNULL,text=True).strip()
                sources=[dict(commit=head,path='docs/research/h4_search_map_20261117/INDEX.json')]
            except subprocess.CalledProcessError:pass
        for source in sources:
            key=json.dumps(source,sort_keys=True)
            if key not in self.evidence_cache:self.evidence_cache[key]=reader.read(source)
            doc=self.evidence_cache[key];models=doc.get('models',doc.get('price_validation_models',{}))
            values=models.values()if isinstance(models,dict)else models
            for model in values:
                if model.get('model_id')==desc['model_id']:
                    if model.get('identity')!=desc['identity']:raise ValueError('Published model identity differs')
                    if model.get('validation_points'):return model['validation_points'],issues,None
                    if model.get('job_id')or model.get('external_job'):return checks,issues,model
        return checks,issues,None
    def poll(self):
        with self.lock:pending=list(self.descriptions.items())
        for mid,(desc,c)in pending:
            if self.stopped.is_set():break
            row=self.store.row(mid)
            if row['status']=='CALIBRATION_ONLY':
                vector=c.get('validation_vector')or[n for cap in c.get('caps',[1]*13)for n in(cap,cap)]
                chosen=representative(desc,vector)
                def freeze(r):
                    if r['status']!='CALIBRATION_ONLY':return
                    if chosen is None:r.update(status='INCOMPLETE',reason='NO_IDENTIFIABLE_NON_CALIBRATION_ROUND');return
                    r.update(status='PENDING',selected_round=chosen,selected_configuration=deepcopy(c.get('generated_configuration')),
                        selected_configuration_source=deepcopy(c.get('configuration_source')or desc.get('configuration_source')),
                        selected_configuration_id=c['configuration_id'],selected_signal_source=deepcopy(c.get('signal_source')),
                        automatic_retry=False,compile_calls_requested=4 if c['family']=='PR' else 2)
                self.store.change(mid,freeze)
            row=self.store.row(mid)
            if row['status']!='PENDING'or row.get('external_job'):continue
            if row.get('owner_run')not in(None,str(self.run.resolve())):continue
            if row.get('job_id'):
                self._collect(mid,row);continue
            # Scope comes from the adopted run, not from a Q3-only budget.
            prep=(self.spec.get('executor',{}).get('distributed')or{}).get('preparation')or{}
            budget=prep.get('budgets',{}).get('compile',0)
            in_scope=self.settings.get('compile_approved',budget=='UNLIMITED'or isinstance(budget,int)and budget>0)
            if self.spec.get('science_approved')is not True or in_scope is not True:
                self.store.change(mid,lambda r:r.update(pending_reason='COMPILE_NOT_IN_ADOPTED_RUN_SCOPE'));continue
            if self.executor is None or not hasattr(self.executor,'submit_preparation'):
                self.store.change(mid,lambda r:r.update(pending_reason='NO_DISTRIBUTED_COMPILE_EXECUTOR'));continue
            self._submit(mid,row,desc,c)
        self.export()
    def _submit(self,mid,row,desc,c):
        from .rpe_preparation_bridge import DistributedPreparationBridge
        acquisition=(self.settings.get('acquisition')or{}).get(desc.get('registry'))or(self.settings.get('providers')or{}).get(c['method'])
        if not acquisition and c['method']=='PR_PAULI':
            acquisition=dict(kind='PR_PAULI',backend='BUILTIN_PAULI_NORMAL_PATH',
                legacy_view=str(self.root.parent/'a-pauli-ld3-cal-source-13ab'),
                interpreter=(self.problem.get('price_rules')or{}).get('interpreter','python3'))
        if not acquisition and c['method']=='PR_DF':
            acquisition=dict(kind='PR_DF',backend='BUILTIN_DF_NORMAL_PATH',
                legacy_view=str(self.root.parent/'d2-pr-df-ld1-m3-acquisition-20261115'),
                interpreter=(self.problem.get('price_rules')or{}).get('interpreter','python3'))
        if not acquisition:
            self.store.change(mid,lambda r:r.update(pending_reason='ACQUISITION_PROVIDER_NOT_CONFIGURED'));return
        directory=self.run/'price_validation'/mid;directory.mkdir(parents=True,exist_ok=True)
        request=dict(schema='RPE_PRICE_VALIDATION_JOB_V1',model_id=mid,frozen_model=dict(source=desc['identity']['model_source'],coefficients=desc['identity']['coefficients']),
            prediction_source=desc['identity']['model_source'],implementation=desc['identity']['family'].get('implementation')or{a:v.get('implementation')for a,v in desc['identity']['family'].get('domains',{}).items()},
            selected_round=row['selected_round'],predicted_RZ=row['selected_round']['predicted_RZ'],
            configuration=row.get('selected_configuration'),configuration_source=row.get('selected_configuration_source'),
            signal_source=row.get('selected_signal_source')or(c.get('signal_source')if c['configuration_id']==row.get('selected_configuration_id')else None),acquisition=deepcopy(acquisition))
        configuration=request['configuration']
        if configuration is None:
            from .pr_price_rules import GitReader
            configuration=GitReader(self.root).read(request['configuration_source']);request['configuration']=configuration
        revision=digest(dict(acquisition=acquisition,source=self.implementation_source,signal=request['signal_source'],configuration_source=request['configuration_source']))
        if row.get('metadata_hold_revision')==revision:return
        if acquisition.get('backend')in('BUILTIN_PAULI_NORMAL_PATH','BUILTIN_DF_NORMAL_PATH'):
            from importlib.util import spec_from_file_location,module_from_spec
            worker_spec=spec_from_file_location('_validation_worker',self.root/'scripts/run_rpe_price_validation_job.py')
            worker=module_from_spec(worker_spec);worker_spec.loader.exec_module(worker)
            try:
                common=dict(root=self.root,configuration=configuration,legacy_view=acquisition['legacy_view'],interpreter=acquisition['interpreter'],
                    execution_source=dict(commit=self.implementation_source,
                        path='scripts/run_rpe_price_validation_job.py'),output_dir=directory,calibration_seed_sources=desc.get('calibration_sources',[]))
                if acquisition['kind']=='PR_PAULI':
                    request['acquisition']=worker.build_pauli_acquisition(request,signal_source=request['signal_source'],**common)
                else:request['acquisition']=worker.build_df_acquisition(request,configuration_source=request['configuration_source'],**common)
            except (ValueError,KeyError,OSError,ImportError,subprocess.CalledProcessError)as error:
                self.store.change(mid,lambda r:r.update(pending_reason='ACQUISITION_METADATA_HOLD:'+str(error),metadata_hold_revision=revision));return
        path=directory/'INPUTS.json';self._write(path,request)
        source=self.implementation_source
        runtime=(self.spec.get('executor',{}).get('distributed')or{}).get('preparation')or{}
        bridge=DistributedPreparationBridge(self.executor,self.root,runtime,approved=True)
        command=[self.settings.get('interpreter','python3'),'-B',str(self.root/'scripts/run_rpe_price_validation_job.py'),
            '--request',str(path),'--output',str(directory/'RESULT.json')]
        task=bridge.build_request(SimpleNamespace(source=source,preparation_root=self.root),command,directory,'PRICE_MODEL_VALIDATION',directory/'RESULT.json')
        task['preparation']['optional_price_validation']=True
        if request['acquisition'].get('backend')in('BUILTIN_PAULI_NORMAL_PATH','BUILTIN_DF_NORMAL_PATH'):
            legacy=Path(request['acquisition']['legacy_view']).resolve()
            pauli=acquisition['kind']=='PR_PAULI';commit=worker.PAULI_SOURCE if pauli else worker.DF_SOURCE
            files={}
            for name in (('scripts/run_pauli_b1_prices.py','src/trottertracks/pauli_tail_price_connection.py')if pauli else ('scripts/run_pr_df_candidate_prices.py',)):
                files[name]=hashlib.sha256((legacy/name).read_bytes()).hexdigest()
            task['preparation']['workspaces']['validation_legacy']=dict(source=dict(commit=commit,files=files),input_git_bundles=[],checkout_input_head=False)
            task['preparation']['path_bindings'][str(legacy)]='@WORKSPACE:validation_legacy@'
        from .rpe_preparation_queue import normalize_preparation_request
        task=normalize_preparation_request(task,source_manifest=self.executor.source_manifest)
        job=task['request_id']
        def intent(r):
            if r.get('owner_run')not in(None,str(self.run.resolve())):return
            if r.get('job_id'):return
            r.update(owner_run=str(self.run.resolve()),job_id=job,request=task,validation_request=request,result_path=str(directory/'RESULT.json'),
                pending_reason='FROZEN_DISPATCH_INTENT',compile_budget=runtime.get('budgets',{}).get('compile',0))
        fixed=self.store.change(mid,intent)
        if fixed.get('owner_run')==str(self.run.resolve()) and not self.stopped.is_set():self._collect(mid,fixed)
    def _collect(self,mid,row):
        if self.stopped.is_set():return
        # Re-submit only this exact frozen request; persistent queue rejects
        # any changed contract and preserves an assigned/unknown attempt.
        with self.dispatch_lock:
            if self.stopped.is_set():return
            job=self.executor.submit_preparation(row['request'],budget=row['compile_budget'],approved=True,
                contract=dict(model_id=mid,one_representative_only=True,automatic_retry=False))
        if job!=row['job_id']:raise ValueError('Frozen validation task ID changed')
        result=self.executor.preparation_result(row['job_id'],wait=False)
        if result is None:return
        if result.get('status')=='PREPARATION_COMPLETE':
            from .rpe_preparation_bridge import DistributedPreparationBridge
            DistributedPreparationBridge(self.executor,self.root)._import_result(result,row['request'])
            value=json.loads(Path(row['result_path']).read_text())
            if value.get('model_id')!=mid:raise ValueError('Validation returned another model')
            if value.get('frozen_model')!=row['validation_request']['frozen_model']or value.get('implementation')!=row['validation_request']['implementation']:raise ValueError('Returned frozen model/implementation differs')
            if value.get('status')not in('INDEPENDENT_MEASURED','COMPLETE'):raise ValueError('Validation worker incomplete')
            if value.get('selected_round')!=row['selected_round']:raise ValueError('Validation returned another circuit')
            if value.get('predicted_RZ')!=row['selected_round']['predicted_RZ']:raise ValueError('Validation prediction changed')
            checks=dict(m=value['selected_round']['m'],q=value['selected_round']['q'],R=value['selected_round']['R'],
                source=dict(path=row['result_path']),observations=[],predicted_RZ={},observed_mean_RZ={},signed_relative_difference={},independent_paths=value.get('independent_paths'))
            expected_paths=2 if self.descriptions[mid][1]['family']=='PR'else 1
            if len(value['rows'])!=2:raise ValueError('Validation axes incomplete')
            for axis in value['rows']:
                paths=axis['paths']
                if len(paths)!=expected_paths or len({p['path_index']for p in paths})!=expected_paths or any(p.get('valid')is not True for p in paths):raise ValueError('Validation paired paths invalid')
                mean=sum(frac(p['compiled_RZ'])for p in paths)/expected_paths
                if mean!=frac(axis['observed_mean_RZ']):raise ValueError('Validation mean differs')
                a=axis['axis'];checks['predicted_RZ'][a]=axis['predicted_RZ'];checks['observed_mean_RZ'][a]=axis['observed_mean_RZ']
                checks['signed_relative_difference'][a]=axis['signed_relative_difference']
                checks['observations']+=[dict(path_id=str(p['path_index']),axis=a,compiled_RZ=p['compiled_RZ'],source=p.get('receipt')or p['source'])for p in axis['paths']]
            if set(checks['predicted_RZ'])!=set(AXES)or checks['independent_paths']!=(2 if self.descriptions[mid][1]['family']=='PR'else 1):raise ValueError('Validation independent X/Y paths incomplete')
            self.store.change(mid,lambda r:r.update(status='INDEPENDENT_MEASURED',validation_points=[checks],result_source=row['result_path']))
        elif result.get('status')in('PREPARATION_FAILED','UNKNOWN'):
            self.store.change(mid,lambda r:r.update(status='INCOMPLETE',failure=result,automatic_retry=False))
    @staticmethod
    def _write(path,value):
        raw=json.dumps(value,ensure_ascii=False,indent=2,default=enc)+'\n';path=Path(path)
        if path.exists()and path.read_text()==raw:return
        path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp-'+str(threading.get_ident()));tmp.write_text(raw);tmp.replace(path)
    def export(self):
        self._write(self.run/'PRICE_MODEL_VALIDATIONS.json',dict(schema='RPE_PRICE_MODEL_VALIDATIONS_V1',
            models=[self.store.row(mid)for mid in sorted(self.descriptions)],applications=list(self.applications.values()),issues=self.issues,
            alters_model_coefficients=False,alters_qualification=False,alters_cost_or_ranking=False))
    def start(self):
        if self.thread is not None:return
        prep=(self.spec.get('executor',{}).get('distributed')or{}).get('preparation')or{}
        budget=prep.get('budgets',{}).get('compile',0)
        in_scope=self.settings.get('compile_approved',budget=='UNLIMITED'or isinstance(budget,int)and budget>0)
        if self.spec.get('science_approved')is not True or in_scope is not True:return
        def monitor():
            while not self.stopped.wait(.5):
                try:self.poll()
                except Exception as error:
                    # Optional acquisition/persistence issues remain visible but
                    # never turn a previously applicable price/Q3 into failure.
                    self._write(self.run/'PRICE_VALIDATION_ISSUE.json',dict(status='INCOMPLETE',reason=type(error).__name__+':'+str(error)))
        self.thread=threading.Thread(target=monitor,name='rpe-price-checks',daemon=True);self.thread.start()
    def close(self):
        self.stopped.set()
        with self.dispatch_lock:pass  # no late submit after executor shutdown
        if self.thread:self.thread.join(timeout=1)
        self.export()
