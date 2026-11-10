"""H4-only C adapter. Lookup is passive; predict_path is an explicit numerical API.

No sampler/compiler/fitter, descriptor extraction, N solver, or cost aggregation.
The caller supplies a pinned JSON reader (e.g. B's SavedArtifacts.read).
"""
from copy import deepcopy
from functools import lru_cache
import hashlib
import importlib.util
import json
import math
from pathlib import Path

from . import h4_df10_price_v2_independent as mq

RELEASE = '1cf87201ebc6c9e0296fe647354981b6f7f8b0c1'
S1 = mq.S1
V1 = 'eec5bdf9cc391fe49b55b5673f3e385deeb20d42'
BASE = 'docs/research/h4_df10_price_v2_independent_q_20261110/'
S1_BASE = 'docs/research/h4_df10_price_v2_s1_20261110/'
V1_BASE = 'docs/research/h4_df10_price_v1_20261110/'
PROFILE = dict(compiler=mq.COMPILER, control='symmetric_directional',
    wrapper='full_Hadamard_with_measurement',
    scope='single_hadamard_interrogation_without_state_preparation', P=0,
    qiskit_version='1.3.0', basis_gates=['rz','sx','x','cx'],
    optimization_level=1, seed_transpiler=17)
PROVIDER_SOURCE = 'c31ef424175c02379eafebe8fc6c7548c1d9aabe'


def ref(commit, path, **selector):
    return dict(commit=commit, path=path, **selector)


def same(a, b):
    return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False)


def events_digest(events):
    return hashlib.sha256(json.dumps(events, sort_keys=True, separators=(',',':'),
                                    allow_nan=False).encode()).hexdigest()


@lru_cache(maxsize=1)
def _canonical_id():
    # Reuse the original passive function; importing this file runs no science.
    path = Path(__file__).resolve().parents[2]/'scripts/h46_long_time_signal.py'
    spec = importlib.util.spec_from_file_location('_h4_price_candidate', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.cid


class H4PriceProvider:
    def __init__(self, read_json, *, pipeline=None, selections=None):
        self._read = read_json
        self.pipeline = pipeline
        self.selections = deepcopy(selections or {})
        self._records = []
        self._metrics = {}
        self._preparations = {}
        self._load_saved()

    @property
    def binding(self):
        return deepcopy(self._binding)

    def preparation_hash(self, method):
        return self._preparations[method]

    def _load_saved(self):
        origin = ref(RELEASE, BASE+'completion01/PROVENANCE.json')
        provenance = self._read(origin)
        self._binding = deepcopy(provenance['scientific_binding'])
        self._b = dict(mq.B)
        if provenance['b'] != self._b or provenance['compiler'] != mq.COMPILER:
            raise ValueError('FROZEN_MQ_BINDING')
        evidence = self._read(ref(RELEASE, BASE+'completion01/RESULT.json'))
        predictions = self._read(ref(RELEASE, BASE+'PREDICTIONS.json'))
        s1 = self._read(ref(S1, S1_BASE+'RESULT.json'))
        for fit in s1['calibrations']:
            if fit['model']=='Mq' and fit['coefficients'] != [[self._b[fit['method']]]*2]:
                raise ValueError('S1_COEFFICIENT_CHANGED')
        bindings = {(b['candidate_id'],b['path_index']): b for b in provenance['trial_bindings']}
        forecast = {(p['candidate_id'],p['path_index'],p['axis']): p for p in predictions['predictions']}
        for b in bindings.values():
            previous = self._preparations.setdefault(b['method'], b['preparation_hash'])
            if previous != b['preparation_hash']: raise ValueError('PREPARATION_BINDING')
        for trial in evidence['trials']:
            for axis in ('cosine','sine'):
                key = trial['candidate_id'], trial['index'], axis
                p = forecast[key]; b = bindings[key[:2]]
                if b['canonical_events_sha256'] != p['events_sha256']:
                    raise ValueError('SAVED_EVENT_PREDICTION_BINDING')
                unit = 'path' if p['method']=='PR' else 'deterministic'
                for kind, value in [('direct_compiled',trial['prices'][axis]['rz_count']),
                                    ('model_prediction',p['Mq'])]:
                    self._add(p['method'],p['q'],axis,p['candidate_id'],mq.COHORT if unit=='path' else 'DETERMINISTIC',
                        unit,trial['index'],kind,value,
                        ref(RELEASE,BASE+('completion01/RESULT.json' if kind=='direct_compiled' else 'PREDICTIONS.json'),
                            candidate_id=p['candidate_id'],path_index=trial['index'],axis=axis),
                        event_digest=p['events_sha256'],digest_kind='canonical_events_sha256')
        for name in ('DEVELOPMENT','HOLDOUT'):
            source = ref(V1,V1_BASE+name+'.json'); doc = self._read(source)
            if not same(doc['scientific_binding'],self._binding): raise ValueError('V1_SCIENCE_BINDING')
            private = {r['index']:r for r in doc['private_trial_bindings']}
            for row in doc['rows']:
                method=row['method']; unit='path' if method=='PR' else 'deterministic'
                self._add(method,row['q'],row['axis'],row['candidate_id'],row['cohort'],unit,
                    row['path_index'],'direct_compiled',row['compiled_rz'],
                    dict(source,candidate_id=row['candidate_id'],path_index=row['path_index'],axis=row['axis']),
                    event_digest=private[row['path_index']]['events_sha256'] if method=='PR' else None,
                    digest_kind='private_events_file_sha256' if method=='PR' else None)
        for document, source in [(s1,ref(S1,S1_BASE+'RESULT.json')),
                                 (evidence,ref(RELEASE,BASE+'completion01/RESULT.json'))]:
            for summary in document['summaries']:
                method,q,axis,model = (summary[k] for k in ('method','q','axis','model'))
                if model=='Mq': self._metrics[method,q,axis,summary['cohort']] = dict(source=source,record=deepcopy(summary))
                if method=='PF' and model!='Mq': continue
                if method=='PR' and model not in ('M0','Mq'): continue
                cohort=summary['cohort']; unit='cohort_mean' if method=='PR' else 'deterministic'
                cid=next(r['candidate_id'] for r in self._records if r['method']==method and r['q']==q)
                kind='model_prediction' if model=='Mq' else 'direct_compiled'
                value=summary['metrics']['prediction_mean' if kind=='model_prediction' else 'observed_mean']
                self._add(method,q,axis,cid,cohort,unit,None if unit=='cohort_mean' else 0,kind,value,
                    dict(source,method=method,q=q,axis=axis,cohort=cohort,model=model),n_paths=summary['n_paths'])

    def _add(self, method, q, axis, cid, cohort, unit, path, kind, value, source, **extra):
        self._records.append(dict(method=method,q=q,axis={'cosine':'X','sine':'Y'}[axis],
            candidate_id=cid,cohort=cohort,price_unit=unit,path_index=path,kind=kind,
            value=value,source=source,**extra))

    def _domain(self, request):
        c=request['candidate']; method='PR' if c['method']=='PR' else 'PF'
        if request['axis'] not in ('X','Y'): raise ValueError('X_OR_Y_REQUIRED')
        if type(c['q']) is not int or c['q']<=0: raise ValueError('POSITIVE_Q_PF_REQUIRED')
        if (not math.isfinite(c['T']) or c['T']<=0 or c['delta']!=c['T']/c['q']):
            raise ValueError('Q_PF_TIME_DELTA_BINDING')
        if c['id'] != (self.pipeline.candidate_id(c) if self.pipeline else _canonical_id()(c)):
            return method,'CANDIDATE_ID_MISMATCH'
        if c['system']!='H4' or c['method'] not in ('B0','PR') or c['formula']!='2nd' or c['delta']!=.1:
            return method,'OUTSIDE_MODEL_RECIPE'
        if (type(c['prefix']) is not int or c['prefix']!=(1 if method=='PR' else 7)
                or (c['R_outer'],c['K'])!=((4,2) if method=='PR' else (None,None))
                or (method=='PR' and any(type(c[k]) is not int for k in ('R_outer','K')))):
            return method,'OUTSIDE_PREFIX_R_K'
        ignored={'T','profile','profile_is_reference_only','rank_is_saved_observation_not_generation_argument'}
        if any(not same(request['binding'].get(k),v) for k,v in self._binding.items() if k not in ignored):
            return method,'H_DF_STATE_SNAPSHOT_BINDING'
        if any(not same(request['profile'].get(k),v) for k,v in PROFILE.items()):
            return method,'COMPILER_CONTROL_WRAPPER_BINDING'
        if any(request['profile'].get(k) is not None for k in ('backend','target','coupling_map')):
            return method,'FOREIGN_COMPILER_TARGET'
        contract=request['contract'];prep=contract.get('preparation_binding',{})
        if (contract.get('completion')!='saved_weighted_Hermitian_completion'
                or contract.get('fragment_order')!='saved_snapshot_order'
                or not same(contract.get('state_preparation_cost'),0) or not contract.get('input_source')
                or prep.get('snapshot')!=self._binding['snapshot'] or prep.get('provider_source')!=PROVIDER_SOURCE):
            return method,'PREPARATION_SOURCE_ORDER_SCALAR_BINDING'
        supplied=request.get('preparation_hash',prep.get('preparation_hash'))
        if supplied is not None and supplied!=self._preparations[method]:
            return method,'PREPARATION_HASH_MISMATCH'
        if 'q_RPE' in request and (type(request['q_RPE']) is not int or request['q_RPE']<=0
                                   or request['q_RPE']&(request['q_RPE']-1)):
            raise ValueError('POSITIVE_SEPARATE_Q_RPE_REQUIRED')
        return method,None

    def _response(self, request, method, *, kind='unavailable', value=None, reason=None, record=None):
        c=request['candidate'];q=c['q'];axis=request['axis']
        independent = (method,q) in (('PF',32),('PR',16))
        prefix=(method,q,{'X':'cosine','Y':'sine'}[axis])
        evidence=[v for k,v in self._metrics.items() if k[:3]==prefix
                  and (not request.get('cohort') or k[3]==request['cohort'])]
        validation=('OUTSIDE_DOMAIN' if reason and reason.startswith(('OUTSIDE','FOREIGN','H_DF','COMPILER','PREPARATION','CANDIDATE','PATH_EVENT','ORDERED_EVENTS'))
                    else 'FINITE_INDEPENDENT_Q' if independent else 'S1_KNOWN_DATA' if (method,q) in (('PF',8),('PF',16),('PR',8))
                    else 'UNVALIDATED_EXTRAPOLATION')
        return dict(kind=kind,value=value,units='gate_count',resource='rz_count',model=None if kind=='direct_compiled' else 'Mq',
            price_unit=request.get('price_unit','deterministic' if method=='PF' else None),
            reason=reason,conditions=deepcopy({k:v for k,v in request.items() if k!='ordered_events'}),validation=validation,
            qualified_for_candidate=value is not None and (kind=='direct_compiled' or (independent and validation=='FINITE_INDEPENDENT_Q')),
            qualification_basis='SAVED_RECORD_AND_DECLARED_PRICE_UNIT' if kind=='direct_compiled' else 'EMPIRICAL_MODEL_DOMAIN_ONLY',
            full_RPE_price_adopted=False,preparation_hash=self._preparations[method],
            preparation_evidence='FIXED_SNAPSHOT_PREFIX_PROVIDER_BINDING',
            source=deepcopy(record['source'] if record else ref(RELEASE,BASE+'completion01/REPORT.md')),
            model_provenance=dict(structure=ref(V1,'src/trottertracks/h4_df10_price_v1.py'),
                q_scope_adapter=ref(RELEASE,'src/trottertracks/h4_df10_price_v2_independent.py'),
                calibration=ref(S1,S1_BASE+'RESULT.json'),validation=ref(RELEASE,BASE+'completion01/RESULT.json')),
            saved_record=deepcopy(record),
            uncertainty=dict(saved_validation=deepcopy(evidence),evidence_target_model='Mq',
                path_residual_reference=(ref(RELEASE if independent else S1,
                    BASE+'completion01/PATH_RESIDUALS.csv' if independent else S1_BASE+'PATH_RESIDUALS.csv',
                    method=method,axis={'X':'cosine','Y':'sine'}[axis],cohort=request.get('cohort'),path_index=request.get('path_index'),model='Mq')
                    if value is not None and request.get('price_unit')=='path' else None),
                coefficient_uncertainty=None,coefficient_uncertainty_applicable=kind=='model_prediction',
                long_q_systematic=None,all_law_expectation=None,joint_covariance=None,total_CI=None,
                shared_calibration_group='DF10_Mq_'+method+'@'+S1 if kind=='model_prediction' else None,
                paired_XY=True,cross_round_independence_assumed=False,
                observed_max_is_bound=False,alpha_is_price_confidence=False))

    def quote(self, request):
        """Saved scalar selection only. Never computes a fresh model or mean."""
        request=deepcopy(request);method,reason=self._domain(request)
        if reason:return self._response(request,method,reason=reason)
        unit=request.get('price_unit','deterministic' if method=='PF' else None)
        if unit is None:return self._response(request,method,reason='PR_PRICE_UNIT_REQUIRED')
        if unit=='law_mean':return self._response(request,method,reason='ALL_LAW_EXPECTATION_UNAVAILABLE')
        if method=='PF' and unit!='deterministic':return self._response(request,method,reason='PF_DETERMINISTIC_UNIT_REQUIRED')
        if method=='PR' and unit not in ('path','cohort_mean'):return self._response(request,method,reason='PR_PATH_OR_COHORT_UNIT_REQUIRED')
        kind=request.get('kind','direct_compiled')
        if kind not in ('direct_compiled','model_prediction'):raise ValueError('EXPLICIT_PRICE_KIND')
        matches=[r for r in self._records if r['method']==method and r['q']==request['candidate']['q']
            and r['candidate_id']==request['candidate']['id'] and r['axis']==request['axis']
            and r['price_unit']==unit and r['kind']==kind
            and (method=='PF' or r['cohort']==request.get('cohort'))
            and (unit!='path' or r['path_index']==request.get('path_index'))]
        if not matches:return self._response(request,method,reason='NO_SAVED_PRICE_FOR_SELECTOR')
        # Prefer the one independent source record over the copied n1 summary.
        record=matches[0]
        if any(r['value']!=record['value'] for r in matches):raise ValueError('AMBIGUOUS_SAVED_PRICE')
        if unit=='path':
            if request.get('event_digest')!=record.get('event_digest') or request.get('digest_kind')!=record.get('digest_kind'):
                return self._response(request,method,reason='PATH_EVENT_BINDING_MISMATCH')
            if 'ordered_events' in request and (record['digest_kind']!='canonical_events_sha256'
                    or events_digest(request['ordered_events'])!=record['event_digest']):
                return self._response(request,method,reason='ORDERED_EVENTS_MISMATCH')
        return self._response(request,method,kind=kind,value=record['value'],record=record)

    def predict_path(self, request, descriptors):
        """Explicit scientific numerical entry; not called by quote or batch.

        Tests use synthetic descriptors. Real calls need their own permitted
        scope. PR returns one ordered path, never a law/cohort expectation.
        """
        request=deepcopy(request);method,reason=self._domain(request)
        if reason:return self._response(request,method,reason=reason)
        expected=8 if method=='PF' else 2
        if (not same(descriptors['binding'],request['binding'])
                or descriptors['preparation_hash']!=self._preparations[method]
                or not descriptors.get('source') or len(descriptors['blocks'])!=expected
                or [b['block_id'] for b in descriptors['blocks']]!=['one_body']+['df-fragment-'+str(i) for i in range(expected-1)]
                or not math.isfinite(descriptors['scalar'])):
            raise ValueError('SOURCE_BOUND_DESCRIPTOR_ORDER_REQUIRED')
        events=request.get('ordered_events') if method=='PR' else None
        if method=='PF' and request.get('price_unit','deterministic')!='deterministic':
            raise ValueError('NUMERICAL_PF_DETERMINISTIC_ONLY')
        if method=='PR':
            if (request.get('price_unit')!='path' or type(request.get('path_index')) is not int
                    or request['path_index']<0):
                raise ValueError('NUMERICAL_PR_PATH_ONLY')
            if events is None or len(events)!=request['candidate']['q'] or any(len(b)!=4 for b in events):
                raise ValueError('ORDERED_Q_PF_R4_EVENTS_REQUIRED')
            if not request.get('event_source'):raise ValueError('ORDERED_EVENT_SOURCE_REQUIRED')
            for batch in events:
                for event in batch:
                    order=event.get('taylor_order');apps=event.get('application_sequence',[])
                    if (type(order) is not int or order not in (0,2) or len(apps)!=order+1
                            or [a.get('application_index') for a in apps]!=list(range(order+1))
                            or [a.get('role') for a in apps]!=['product']*order+['rotation']
                            or any(type(a.get('coefficient_sign')) is not int or a['coefficient_sign'] not in (-1,1) for a in apps)
                            or any(not a.get('is_identity') and (not a.get('basis_id') or not a.get('basis_hash')
                                or 'basis_change_operations' not in a) for a in apps)):
                        raise ValueError('ORDERED_K2_APPLICATION_BINDING')
            if request.get('event_digest')!=events_digest(events) or request.get('digest_kind')!='canonical_events_sha256':
                raise ValueError('ORDERED_EVENT_DIGEST_MISMATCH')
        # The existing implementation already exposes the unchanged V1 body.
        # No new decomposition, synthesis, or fitting is introduced here.
        c=request['candidate']
        baseline=mq._scope['predict'](descriptors['blocks'],c['q'],descriptors['scalar'],c['T'],
                                     {'X':'cosine','Y':'sine'}[request['axis']],events)
        value=baseline['structural_rz']+self._b[method]*c['q']
        if not math.isfinite(value) or value<0:raise ValueError('INVALID_MODEL_PRICE')
        return self._response(request,method,kind='model_prediction',value=value,
            record=dict(source=deepcopy(descriptors['source']),baseline=baseline,
                        coefficient=self._b[method],price_unit='path' if method=='PR' else 'deterministic'))

    def __call__(self, task, requested):
        """B-compatible c_provider(task, rows): saved lookup only, no G or N."""
        p=self.pipeline
        if p is None:raise ValueError('SUPPLY_B_PIPELINE_MODULE')
        p.validate_task(task)
        records=[]
        for row in requested:
            if not any(p.same(row,r) for r in task['candidates']):raise ValueError('BATCH_ROW_OUTSIDE_TASK')
            selection=deepcopy(self.selections.get(row['candidate']['id'],{}))
            if set(selection)-{'kind','price_unit','cohort','path_index','event_digest','digest_kind','ordered_events','preparation_hash'}:
                raise ValueError('SELECTOR_CANNOT_REPLACE_SCIENTIFIC_IDENTITY')
            for axis in p.AXES:
                request=dict(binding=task['binding'],contract=task['contract'],profile=task['profile'],
                    candidate=row['candidate'],q_RPE=row['q_RPE'],axis=axis)
                request.update(selection)
                records.append(self.pipeline_record(task,row,axis,self.quote(request)))
        return records

    def pipeline_record(self, task, row, axis, quote):
        """Map either a lookup or an explicitly computed quote to B's record."""
        p=self.pipeline
        if p is None:raise ValueError('SUPPLY_B_PIPELINE_MODULE')
        conditions=quote['conditions']
        if (not any(p.same(row,r) for r in task['candidates'])
                or axis not in p.AXES or conditions['axis']!=axis
                or any(not p.same(conditions[k],task[k]) for k in ('binding','contract','profile'))
                or not p.same(conditions['candidate'],row['candidate'])
                or conditions.get('q_RPE',row['q_RPE'])!=row['q_RPE']):
            raise ValueError('QUOTE_PIPELINE_IDENTITY_MISMATCH')
        saved=quote['saved_record'] or {};unit=quote['price_unit']
        modelled=quote['kind']=='model_prediction'
        extrapolated=modelled and quote['validation']=='UNVALIDATED_EXTRAPOLATION'
        expectation=('UNAVAILABLE' if quote['value'] is None else 'UNQUALIFIED_MODEL' if extrapolated
            else 'FINITE_COHORT_ONLY' if unit=='cohort_mean' else 'FINITE_PATH_ONLY' if unit=='path'
            else 'EXPLICIT_CONDITIONAL_MODEL' if modelled else 'DETERMINISTIC')
        classification=('UNAVAILABLE' if quote['value'] is None else 'UNVALIDATED_EXTRAPOLATION' if extrapolated
            else 'MODEL_PREDICTION' if modelled else 'DIRECT_COMPILED_FINITE_SAMPLE'
            if unit in ('path','cohort_mean') else 'DIRECT_COMPILED_DETERMINISTIC')
        cohort=saved.get('cohort',conditions.get('cohort','SUPPLIED_PATH' if unit=='path' else 'DETERMINISTIC'))
        label=('MODEL:' if modelled else 'DIRECT:')+cohort
        if quote['value'] is None:label='UNAVAILABLE:'+conditions.get('cohort','NO_SAVED_RECORD')
        if unit=='path':label+=':path:'+str(conditions['path_index'])
        measurement=p.measurement(row,axis,saved_sampler=row['candidate']['method']=='PR')
        if unit=='path':measurement['law_context']['price_law']='explicit_single_ordered_path'
        return dict(**measurement,identity=p.identity(task,row),axis=axis,classification=classification,
            cohort=label,source_cohort=saved.get('cohort',conditions.get('cohort')),
            n_paths=saved.get('n_paths',1 if quote['value'] is not None and unit!='cohort_mean' else None),
            mean=dict(rz_count=quote['value']),source=quote['source'],expectation=expectation,
            model=(dict(name='Mq',source=quote['source'],validation_source=ref(RELEASE,BASE+'completion01/REPORT.md'),
                q_variable='q_PF',q=row['q_PF'],binding=deepcopy(task['binding']),compiler=task['profile']['compiler'],
                coverage=quote['validation'],qualified_for_candidate=quote['qualified_for_candidate']) if modelled else None),
            uncertainty=deepcopy(quote['uncertainty']),normalization_is_price_multiplier=False,
            preparation_evidence=quote['preparation_evidence'],peak_qubits=None,provider_quote=deepcopy(quote))
