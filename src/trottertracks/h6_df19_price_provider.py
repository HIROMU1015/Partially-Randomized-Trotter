"""H6-only Affine price adapter; batch/quote only select frozen saved values.

predict_affine is an explicit numerical entry, never a lookup fallback. No
sampler, compiler, fitter, descriptor extraction, N solver or G accounting.
"""
from copy import deepcopy
import hashlib
import json
import math

RELEASE = '73dcbe23678bfc50d5654f09e600d4cfbe75e697'
FROZEN_SOURCE = 'b85837960c80d619a421c46235baf5ef2523d217'
V1 = 'a4d743cb5007d5fc6b6e93c3b43a4701ff956ac4'
PROVIDER_SOURCE = 'c31ef424175c02379eafebe8fc6c7548c1d9aabe'
BASE = 'docs/research/h6_df19_price_v2_20261110/'
V1_BASE = 'docs/research/h6_df19_price_v1_20261110/'
PROFILE = dict(compiler='1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b',
    control='symmetric_directional', wrapper='full_Hadamard_with_measurement',
    scope='single_hadamard_interrogation_without_state_preparation', P=0,
    qiskit_version='1.3.0', basis_gates=['rz','sx','x','cx'], optimization_level=1, seed_transpiler=17)
AXES = {'X':'cosine', 'Y':'sine'}
SELECTORS = {'kind','price_unit','cohort','path_index','event_digest','digest_kind','preparation_hash'}


def ref(commit, path, **selector):
    return dict(commit=commit, path=path, **selector)


def same(a, b):
    return json.dumps(a,sort_keys=True,allow_nan=False)==json.dumps(b,sort_keys=True,allow_nan=False)


def candidate_id(c):
    # Exact passive cid tuple/serialization from h46_long_time_signal at 155529e20.
    # In B, reuse B's own candidate_id instead of this standalone fallback.
    fields=(c['system'],c['method'],c['formula'],c['prefix'],c['T'],c['q'],c.get('R_outer'),c.get('K'))
    return hashlib.sha256(json.dumps(fields).encode()).hexdigest()[:20]


def _affine(coefficients, q, axis):
    return coefficients[0][axis]+coefficients[1][axis]*q


class H6AffinePriceProvider:
    def __init__(self, read_json, *, pipeline=None, selections=None):
        self._read=read_json;self.pipeline=pipeline;self.selections=deepcopy(selections or {})
        self._records=[];self._candidates={};self._validation={};self._statistics={};self._errors={}
        self._load_saved()

    @property
    def binding(self):return deepcopy(self._binding)

    @property
    def coefficients(self):
        return {family:deepcopy(model['coefficients']) for family,model in self._models.items()}

    def preparation_hash(self, family):return self._preparations[family]

    def _add(self, c, axis, cohort, unit, kind, value, source, *, n_paths, index=None, digest=None, role=None):
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
            raise ValueError('INVALID_SAVED_PRICE')
        self._candidates[c['id']]=deepcopy(c)
        self._records.append(dict(candidate=deepcopy(c),axis=axis,cohort=cohort,price_unit=unit,kind=kind,
            value=value,source=source,n_paths=n_paths,path_index=index,event_digest=digest,
            digest_kind='private_events_file_sha256' if digest else None,record_role=role))

    def _load_saved(self):
        inputs=self._read(ref(RELEASE,BASE+'PILOT_INPUTS.json'))
        self._binding=deepcopy(inputs['scientific_binding'])
        self._preparations={'PF_FULL19':inputs['preparation_hash']['19'],'PR_B':inputs['preparation_hash']['7']}
        if inputs['compiler_hash']!=PROFILE['compiler'] or inputs['provider_source']!=PROVIDER_SOURCE:
            raise ValueError('FROZEN_PROVIDER_COMPILER_BINDING')
        frozen=self._read(ref(FROZEN_SOURCE,BASE+'Q4_FROZEN_MODELS.json'))
        if frozen['q4_prices_observed'] or frozen['q4_refit']:raise ValueError('FROZEN_BEFORE_Q4_REQUIRED')
        self._models=deepcopy(frozen['families'])
        for model in self._models.values():
            if (model['model']!='AFFINE_FIXED_DELTA' or model['features']!=['intercept','q']
                    or model['response_axes']!=['cosine','sine'] or len(model['coefficients'])!=2
                    or any(len(row)!=2 or any(isinstance(v,bool) or not math.isfinite(v) for v in row) for row in model['coefficients'])):
                raise ValueError('FROZEN_AFFINE_SCHEMA')
        self._frozen_qualification=frozen['qualification']
        for family,model in self._models.items():self._statistics[family,1]=deepcopy(model['q1'])
        completion=self._read(ref(RELEASE,BASE+'COMPLETION.json'));self._orders=completion['observed_orders']
        paths={(r['candidate_id'],r['index']):r for r in completion['private_path_bindings']}
        validation=self._read(ref(RELEASE,BASE+'Q_VALIDATION.json'))
        for s in validation['summaries']:
            if s['model']=='Affine':self._validation[s['family'],s['q_PF'],s['axis']]=deepcopy(s)
        for s in validation['paired_covariances']:
            family,q=s['family'],s['q_PF']
            if s['model']=='DIRECT_PRICE':self._statistics[family,q]=deepcopy(s['statistics'])
            if s['model']=='Affine':self._errors[family,q]=deepcopy(s)
        for name in ('DEVELOPMENT','HOLDOUT'):
            source=ref(V1,V1_BASE+name+'.json');doc=self._read(source)
            if not same(doc['scientific_binding'],self._binding):raise ValueError('V1_SCIENTIFIC_BINDING')
            bindings={(b['family'],b['cohort'],b['index']):b for b in doc['private_trial_bindings']}
            for r in doc['rows']:
                if r['T']!=1.6 or r['q_PF']!=1 or r['family'] not in self._models:continue
                family=r['family'];pr=family=='PR_B'
                c=dict(system='H6',method='PR' if pr else 'FULL',formula='4th(m5_best)',prefix=7 if pr else 19,
                    T=1.6,q=1,delta=1.6,R_outer=16 if pr else None,K=2 if pr else None,id=r['candidate_id'])
                if c['id']!=candidate_id(c):raise ValueError('V1_CANDIDATE_BINDING')
                digest=bindings[family,r['cohort'],r['path_index']]['events_sha256'] if pr else None
                role='KNOWN_Q1_DATA_V1_USED' if pr and r['cohort']=='CONFIRMATION' else 'CALIBRATION_Q1'
                self._add(c,{'cosine':'X','sine':'Y'}[r['axis']],r['cohort'],'path' if pr else 'deterministic',
                    'direct_compiled',r['compiled_rz'],dict(source,candidate_id=c['id'],path_index=r['path_index'],axis=r['axis']),
                    n_paths=1,index=r['path_index'],digest=digest,role=role)
            for s in doc['summaries']:
                if s['family']!='PR_B':continue
                c=self._candidates[next(r['candidate_id'] for r in doc['rows'] if r['family']=='PR_B')]
                self._add(c,{'cosine':'X','sine':'Y'}[s['axis']],s.get('cohort','EXPLORATION' if name=='DEVELOPMENT' else 'CONFIRMATION'),
                    'cohort_mean','direct_compiled',s['mean_compiled'],dict(source,family='PR_B',axis=s['axis'],field='summaries.mean_compiled'),
                    n_paths=s['n_paths'],role='CALIBRATION_Q1' if name=='DEVELOPMENT' else 'KNOWN_Q1_DATA_V1_USED')
        for q in (2,4):
            filename='Q'+str(q)+'_DIRECT.json';doc=self._read(ref(RELEASE,BASE+filename))
            if doc['compiler_hash']!=PROFILE['compiler']:raise ValueError('DIRECT_COMPILER_BINDING')
            for r in doc['rows']:
                c=r['candidate'];family='PR_B' if c['method']=='PR' else 'PF_FULL19'
                if r['preparation_hash']!=self._preparations[family]:raise ValueError('DIRECT_PREPARATION_BINDING')
                unit='path' if family=='PR_B' else 'deterministic'
                for axis,original in AXES.items():
                    self._add(c,axis,r['cohort'],unit,'direct_compiled',r['prices'][original]['rz_count'],
                        ref(RELEASE,BASE+filename,candidate_id=c['id'],index=r['index'],axis=original),
                        n_paths=1,index=r['index'],digest=paths[c['id'],r['index']]['events_sha256'] if unit=='path' else None,
                        role='CALIBRATION_Q2' if q==2 else 'FINITE_INDEPENDENT_Q')
            for s in validation['summaries']:
                if s['q_PF']!=q or s['model'] not in ('M0','Affine'):continue
                family=s['family'];r=next(r for r in doc['rows'] if (r['candidate']['method']=='PR')==(family=='PR_B'))
                if family=='PF_FULL19' and s['model']=='M0':continue  # deterministic direct already copied
                modelled=s['model']=='Affine';field='predicted_mean' if modelled else 'observed_mean'
                self._add(r['candidate'],{'cosine':'X','sine':'Y'}[s['axis']],r['cohort'],
                    'cohort_mean' if family=='PR_B' else 'deterministic',
                    'model_prediction' if modelled else 'direct_compiled',s['metrics'][field],
                    ref(RELEASE,BASE+'Q_VALIDATION.json',family=family,q_PF=q,axis=s['axis'],model=s['model'],field='metrics.'+field),
                    n_paths=s['metrics']['n'],role=s['role'])

    def _domain(self, request):
        c=request['candidate'];family='PR_B' if c['method']=='PR' else 'PF_FULL19'
        if request['axis'] not in AXES:raise ValueError('X_OR_Y_REQUIRED')
        if type(c['q']) is not int or c['q']<=0:raise ValueError('POSITIVE_Q_PF_REQUIRED')
        if isinstance(c['T'],bool) or not math.isfinite(c['T']) or c['T']<=0 or c['delta']!=c['T']/c['q']:
            raise ValueError('Q_PF_TIME_DELTA_BINDING')
        cid=self.pipeline.candidate_id(c) if self.pipeline else candidate_id(c)
        if cid!=c['id']:return family,'CANDIDATE_ID_MISMATCH'
        if c['system']!='H6' or c['method'] not in ('FULL','PR') or c['formula']!='4th(m5_best)' or c['delta']!=1.6:
            return family,'OUTSIDE_MODEL_RECIPE'
        pr=family=='PR_B'
        if (type(c['prefix']) is not int or c['prefix']!=(7 if pr else 19)
                or not same([c['R_outer'],c['K']],[16,2] if pr else [None,None])):
            return family,'OUTSIDE_PREFIX_R_K'
        ignored={'T','profile','profile_is_reference_only','rank_is_saved_observation_not_generation_argument'}
        if any(not same(request['binding'].get(k),v) for k,v in self._binding.items() if k not in ignored):
            return family,'H_DF_STATE_SNAPSHOT_BINDING'
        if any(not same(request['profile'].get(k),v) for k,v in PROFILE.items()):return family,'COMPILER_CONTROL_WRAPPER_BINDING'
        if any(request['profile'].get(k) is not None for k in ('backend','backend_name','target','coupling_map','layout_method','routing_method','approximation_degree')):
            return family,'FOREIGN_COMPILER_TARGET'
        contract=request['contract'];prep=contract.get('preparation_binding',{})
        if (contract.get('completion')!='saved_weighted_Hermitian_completion'
                or contract.get('fragment_order')!='saved_snapshot_order' or not same(contract.get('state_preparation_cost'),0)
                or not contract.get('input_source') or prep.get('snapshot')!=self._binding['snapshot']
                or prep.get('provider_source')!=PROVIDER_SOURCE):return family,'PREPARATION_SOURCE_ORDER_SCALAR_BINDING'
        if request.get('preparation_hash',prep.get('preparation_hash',self._preparations[family]))!=self._preparations[family]:
            return family,'PREPARATION_HASH_MISMATCH'
        if 'q_RPE' in request and (type(request['q_RPE']) is not int or request['q_RPE']<=0 or request['q_RPE']&(request['q_RPE']-1)):
            raise ValueError('POSITIVE_SEPARATE_Q_RPE_REQUIRED')
        return family,None

    def _response(self, request, family, *, kind='unavailable', value=None, reason=None, record=None, outside=False):
        c=request['candidate'];q=c['q'];original=AXES[request['axis']];unit=request.get('price_unit','deterministic' if family=='PF_FULL19' else None)
        role={1:'CALIBRATION_Q1',2:'CALIBRATION_Q2',4:'FINITE_INDEPENDENT_Q'}.get(q,'UNVALIDATED_EXTRAPOLATION' if q>4 else 'UNVALIDATED_INTERPOLATION')
        validation='OUTSIDE_DOMAIN' if outside else role
        modelled=kind=='model_prediction';model=self._models[family]
        group='H6_DF19_AFFINE_'+family+'@'+FROZEN_SOURCE
        qualified=value is not None and (kind=='direct_compiled' or (modelled and q==4 and unit in ('deterministic','cohort_mean')))
        calibration_data=(modelled or (kind=='direct_compiled' and (q==2 or q==1 and (family=='PF_FULL19' or request.get('cohort')=='EXPLORATION'))))
        return dict(kind=kind,requested_kind=request.get('kind','direct_compiled'),value=value,resource='rz_count',units='gate_count',
            model='Affine' if modelled else None,price_unit=unit,reason=reason,conditions=deepcopy({k:v for k,v in request.items() if k!='ordered_events'}),validation=validation,
            record_role=(record or {}).get('record_role'),qualified_for_candidate=qualified,full_RPE_price_adopted=False,
            qualification_basis='SAVED_DIRECT_UNIT_ONLY' if kind=='direct_compiled' else 'FINITE_Q4_VALIDATION_ONLY',
            source=deepcopy(record['source'] if record else ref(RELEASE,BASE+'REPORT.md')),
            saved_record=deepcopy(record),preparation_hash=None if outside else self._preparations[family],
            preparation_evidence='OUTSIDE_MODEL_DOMAIN' if outside else 'FIXED_SNAPSHOT_PREFIX_PROVIDER_BINDING',
            model_provenance=dict(calibration=ref(FROZEN_SOURCE,BASE+'Q4_FROZEN_MODELS.json',family=family),
                validation=ref(RELEASE,BASE+'Q_VALIDATION.json',family=family),q_calibration=[1,2],q_independent_validation=[4],delta=1.6),
            uncertainty=dict(coefficient_uncertainty=None,nominal_coefficient_covariance=deepcopy(model['nominal_coefficient_covariance']) if modelled else None,
                nominal_covariance_qualification=self._frozen_qualification,covariance_order='feature_then_axis',response_axes=['X','Y'],
                saved_q4_prediction_mean_covariance=deepcopy(model['q4_predicted_mean_covariance_under_independent_IID_cohorts']) if modelled and q==4 else None,
                saved_validation=deepcopy(self._validation.get((family,q,original))) if modelled else None,
                saved_paired_error_covariance=deepcopy(self._errors.get((family,q))) if modelled else None,
                saved_direct_mean_statistics=deepcopy(self._statistics.get((family,q))) if unit=='cohort_mean' and kind=='direct_compiled' and (q!=1 or request.get('cohort')=='EXPLORATION') else None,
                shared_calibration_group=group if modelled else None,calibration_data_group=group if calibration_data else None,
                paired_XY=True,cross_round_independence_assumed=False,joint_covariance=None,long_q_systematic=None,
                all_law_expectation=None,unobserved_order2_price=None,total_CI=None,observed_pilot_orders=deepcopy(self._orders),
                observed_max_is_bound=False,alpha_is_price_confidence=False))

    def quote(self, request):
        """Lookup only: copy a saved scalar; never calculate a new mean or a+bq."""
        request=deepcopy(request);family,reason=self._domain(request)
        if reason:return self._response(request,family,reason=reason,outside=True)
        unit=request.get('price_unit','deterministic' if family=='PF_FULL19' else None)
        if unit is None:return self._response(request,family,reason='PR_PRICE_UNIT_REQUIRED')
        if unit=='law_mean':return self._response(request,family,reason='ALL_LAW_EXPECTATION_UNAVAILABLE')
        if unit=='model_mean':return self._response(request,family,reason='EXPLICIT_MODEL_EVALUATION_REQUIRED')
        if (family=='PF_FULL19' and unit!='deterministic') or (family=='PR_B' and unit not in ('path','cohort_mean')):
            return self._response(request,family,reason='PRICE_UNIT_MISMATCH')
        kind=request.get('kind','direct_compiled')
        if kind not in ('direct_compiled','model_prediction'):raise ValueError('EXPLICIT_PRICE_KIND')
        if family=='PR_B' and not request.get('cohort'):return self._response(request,family,reason='PR_COHORT_REQUIRED')
        if unit=='path' and 'ordered_events' in request:return self._response(request,family,reason='FILE_DIGEST_DOES_NOT_VALIDATE_SUPPLIED_EVENTS')
        if unit=='path' and kind=='model_prediction':return self._response(request,family,reason='AFFINE_IS_MEAN_NOT_PATH_MODEL')
        matches=[r for r in self._records if same(r['candidate'],request['candidate']) and r['axis']==request['axis']
            and r['kind']==kind and r['price_unit']==unit and (family=='PF_FULL19' or r['cohort']==request['cohort'])
            and (unit!='path' or r['path_index']==request.get('path_index'))]
        if not matches:return self._response(request,family,reason='NO_SAVED_PRICE_FOR_SELECTOR')
        record=matches[0]
        if any(r['value']!=record['value'] for r in matches):raise ValueError('AMBIGUOUS_SAVED_PRICE')
        if unit=='path' and (type(request.get('path_index')) is not int or request.get('event_digest')!=record['event_digest']
                or request.get('digest_kind')!=record['digest_kind']):
            return self._response(request,family,reason='PATH_EVENT_BINDING_MISMATCH')
        return self._response(request,family,kind=kind,value=record['value'],record=record)

    def predict_affine(self, request):
        """Explicit numerical API, requiring its own scientific execution scope.

        PR model_mean is not a sampled cohort mean, single path or law mean.
        This method is never called by quote/__call__/B run_missing.
        """
        request=deepcopy(request);family,reason=self._domain(request)
        if reason:return self._response(request,family,reason=reason,outside=True)
        if request.get('kind')!='model_prediction':raise ValueError('EXPLICIT_MODEL_KIND_REQUIRED')
        expected='model_mean' if family=='PR_B' else 'deterministic'
        if request.get('price_unit',expected)!=expected:raise ValueError('AFFINE_NUMERICAL_MEAN_UNIT_REQUIRED')
        request['price_unit']=expected
        value=_affine(self._models[family]['coefficients'],request['candidate']['q'],list(AXES).index(request['axis']))
        if not math.isfinite(value) or value<0:raise ValueError('INVALID_MODEL_PRICE')
        return self._response(request,family,kind='model_prediction',value=value,
            record=dict(source=ref(FROZEN_SOURCE,BASE+'Q4_FROZEN_MODELS.json',family=family),cohort='AFFINE_MODEL_SCENARIO',n_paths=None))

    def __call__(self, task, requested_rows):
        p=self.pipeline
        if p is None:raise ValueError('SUPPLY_B_PIPELINE_MODULE')
        p.validate_task(task);result=[]
        for row in requested_rows:
            if not any(p.same(row,r) for r in task['candidates']):raise ValueError('BATCH_ROW_OUTSIDE_TASK')
            selection=deepcopy(self.selections.get(row['candidate']['id'],{}))
            if set(selection)-SELECTORS:raise ValueError('SELECTOR_CANNOT_REPLACE_SCIENTIFIC_IDENTITY')
            for axis in p.AXES:
                request=dict(candidate=row['candidate'],axis=axis,q_RPE=row['q_RPE'],binding=task['binding'],profile=task['profile'],contract=task['contract'],**selection)
                result.append(self.pipeline_record(task,row,axis,self.quote(request)))
        return result

    def pipeline_record(self, task, row, axis, quote):
        p=self.pipeline
        if p is None:raise ValueError('SUPPLY_B_PIPELINE_MODULE')
        c=quote['conditions']
        if (not any(p.same(row,r) for r in task['candidates']) or axis not in p.AXES or c['axis']!=axis
                or not p.same(c['candidate'],row['candidate']) or any(not p.same(c[k],task[k]) for k in ('binding','contract','profile'))
                or c.get('q_RPE',row['q_RPE'])!=row['q_RPE']):raise ValueError('QUOTE_PIPELINE_IDENTITY_MISMATCH')
        unit=quote['price_unit'];saved=quote['saved_record'] or {};modelled=quote['kind']=='model_prediction'
        extrapolated=modelled and quote['validation']=='UNVALIDATED_EXTRAPOLATION'
        expectation=('UNAVAILABLE' if quote['value'] is None else 'UNQUALIFIED_MODEL' if extrapolated or unit=='model_mean'
            else 'FINITE_COHORT_ONLY' if unit=='cohort_mean' else 'FINITE_PATH_ONLY' if unit=='path'
            else 'EXPLICIT_CONDITIONAL_MODEL' if modelled else 'DETERMINISTIC')
        classification=('UNAVAILABLE' if quote['value'] is None else 'UNVALIDATED_EXTRAPOLATION' if extrapolated
            else 'MODEL_PREDICTION' if modelled else 'DIRECT_COMPILED_FINITE_SAMPLE' if unit in ('path','cohort_mean') else 'DIRECT_COMPILED_DETERMINISTIC')
        cohort=saved.get('cohort',c.get('cohort','NO_SAVED_RECORD'))
        label=('MODEL:' if modelled else 'DIRECT:')+cohort
        if quote['value'] is None:label='UNAVAILABLE:'+c.get('cohort','NO_SAVED_RECORD')
        if unit=='path':label+=':path:'+str(c.get('path_index'))
        measurement=p.measurement(row,axis,saved_sampler=row['candidate']['method']=='PR')
        if unit in ('path','model_mean'):measurement['law_context']['price_law']='explicit_single_ordered_path' if unit=='path' else 'Affine_model_mean_not_law_expectation'
        return dict(**measurement,identity=p.identity(task,row),axis=axis,classification=classification,cohort=label,
            source_cohort=saved.get('cohort',c.get('cohort')),n_paths=saved.get('n_paths'),mean=dict(rz_count=quote['value']),
            source=deepcopy(quote['source']),expectation=expectation,
            model=(dict(name='Affine',source=quote['model_provenance']['calibration'],validation_source=quote['model_provenance']['validation'],
                q_variable='q_PF',q=row['q_PF'],binding=deepcopy(task['binding']),compiler=task['profile']['compiler'],
                coverage=quote['validation'],qualified_for_candidate=quote['qualified_for_candidate']) if modelled else None),
            uncertainty=deepcopy(quote['uncertainty']),normalization_is_price_multiplier=False,
            preparation_evidence=quote['preparation_evidence'],peak_qubits=None,provider_quote=deepcopy(quote))


def h6_saved_price_provider(read_json, *, pipeline, selections=None):
    """Pass directly as B's c_provider; does not modify or wrap B accounting."""
    return H6AffinePriceProvider(read_json,pipeline=pipeline,selections=selections)
