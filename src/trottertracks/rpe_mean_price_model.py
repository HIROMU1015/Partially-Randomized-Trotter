"""Supplied all-method price hypotheses and passive B V2 price inputs.

No calibration, molecular feature evaluation, sampling, compiler or G solver.
quote_model is an explicit numerical API: real inputs need a science scope.
PriceProvider copies already supplied records; it never calls quote_model.
"""
from copy import deepcopy

from . import rpe_optimization_pipeline as I

MODULE = 'src/trottertracks/rpe_mean_price_model.py'
QUANTITY = 'FULL_WRAPPER_MEAN_COMPILED_PRICE'
STRUCTURAL = 'EXPECTED_FULL_WRAPPER_STRUCTURAL_RZ_COUNT'
RESIDUAL = 'EXPECTED_FULL_WRAPPER_C_MINUS_S_RZ_COUNT'
FEATURE = 'EXPECTED_STRUCTURAL_FEATURE'
BOUNDARIES = {'SIGNED_PIECE_BOUNDARY', 'OUTER_BOUNDARY', 'SCALAR', 'WRAPPER'}


def context(root, request, m, axis, *, profile_extension):
    """Reuse canonical B identity and C physical binding; no request mutation."""
    I._validate(root, request); B = I.providers(str(root)).B
    if type(m) is not int or not 0 <= m <= request['configuration']['spec']['M'] or axis not in B.p.AXES:
        raise ValueError('Requested round/axis outside configuration')
    if not profile_extension.get('lowering_id') or not profile_extension.get('source'):
        raise ValueError('Explicit sourced lowering profile required')
    row = request['configuration']['rounds'][m]; profile = deepcopy(request['configuration']['spec']['profile'])
    if profile.get('lowering_id', profile_extension['lowering_id']) != profile_extension['lowering_id']:
        raise ValueError('Configuration and price lowering differ')
    profile['lowering_id'] = profile_extension['lowering_id']
    return dict(configuration_id=request['configuration']['id'], m=m, axis=axis,
        method=B.comparison_method(request['configuration']['spec']), identity=deepcopy(row['identity']),
        law_binding=deepcopy(row['law_binding']), physical_context=I.price_context(request,m),
        price_profile=profile, profile_extension=deepcopy(profile_extension))


def _supplied(entry, scope, quantity):
    if (entry.get('quantity') != quantity or not entry.get('source')
            or not I.C.p.same(entry.get('context'), scope)):
        raise ValueError('Supplied quantity/METHOD/law/SOURCE/profile context differs')
    return entry.get('value')


def quote_model(root, request, m, axis, *, model, profile_extension,
                features=None, structural=None, residuals=()):
    """Evaluate one sourced hypothesis; always return an unqualified MODEL.

    Additive coverage labels partition the full controlled wrapper. They are
    assumptions supplied with coefficients, not compiler closure proofs. For
    S + (C-S), reuse the existing combiner and allow only one residual model.
    """
    scope = context(root,request,m,axis,profile_extension=profile_extension)
    if (not model.get('source') or not I.C.p.same(model.get('context'),scope)
            or model.get('qualified_for_candidate',False) is not False):
        raise ValueError('Sourced exact model context required; qualification cannot be promoted')
    B=I.providers(str(root)).B; row=request['configuration']['rounds'][m]
    physical=scope['physical_context']['physical_candidate']; q=physical['q']
    builtins=dict(one=1, q=q, event_count=row['structure']['event_count'],
        outer_boundary_count=q-1, signed_piece_boundary_count=q*(len(row['structure']['signed_weights'])-1))
    if scope['method'] in ('FULL_PF','TRUNCATED_PF','PR'):builtins['pf_step_count']=q
    features=features or {}
    if set(features)&set(builtins):raise ValueError('Cannot replace canonical q/R/count metadata')
    mode=model.get('mode'); value=None; reason=None; details=[]
    if mode=='ADDITIVE':
        if structural is not None or residuals:raise ValueError('Do not stack additive C and S/C-S models')
        required=set(BOUNDARIES)
        if scope['method'] in ('FULL_PF','TRUNCATED_PF','PR'):required.add('PF_BODY')
        if scope['method'] in ('PR','FULL_H_RTE'):required.add('RTE_BODY')
        if scope['method']=='PR':required.add('PF_RTE_BOUNDARY')
        seen=set(); values=[]; missing=False
        for term in model.get('terms',()):
            covers=term.get('covers',()); labels=set(covers); feature=term['feature']
            if not labels or len(labels)!=len(covers) or labels-required or labels&seen:
                raise ValueError('Overlapping or foreign price contributions; do not double count')
            if labels&{'SCALAR','WRAPPER'} and feature!='one':
                raise ValueError('Scalar/wrapper must be counted once')
            seen.update(labels)
            weight=builtins.get(feature)
            if feature not in builtins:
                entry=features.get(feature)
                weight=None if entry is None else _supplied(entry,scope,FEATURE)
            coefficient=term.get('coefficient')
            details.append(dict(feature=feature,weight=deepcopy(weight),coefficient=deepcopy(coefficient),covers=list(covers)))
            if coefficient is None or weight is None:missing=True;continue
            weight=B.a.exact_scalar(weight)
            if weight<0:raise ValueError('Structural feature expectation must be nonnegative')
            values.append(B.a.exact_scalar(coefficient)*weight)
        if seen!=required or missing:reason='MISSING_FULL_WRAPPER_COMPONENT_OR_FEATURE'
        else:value=B.a.exact_value(sum(values))
    elif mode=='STRUCTURAL_PLUS_RESIDUAL':
        if model.get('terms') or features:raise ValueError('Do not add overlapping terms to C1 S or C-S')
        if structural is None or not residuals:reason='MISSING_FULL_WRAPPER_COMPONENT_OR_FEATURE'
        else:
            s=_supplied(structural,scope,STRUCTURAL)
            if s is not None:s=float(B.a.exact_scalar(s))
            for r in residuals:_supplied(r,scope,RESIDUAL)
            from .mean_compiled_price import combine_mean
            terms=[dict(target='C_MINUS_S',mode=r['mode'],mean=None if r.get('value') is None else float(B.a.exact_scalar(r['value'])),
                        coverage=r.get('coverage')) for r in residuals]
            result=combine_mean(s,residual_terms=terms);value=result['mean']
            reason=None if value is not None else result['status']
    else:raise ValueError('Explicit ADDITIVE or STRUCTURAL_PLUS_RESIDUAL mode required')
    if value is not None and B.a.exact_scalar(value)<0:raise ValueError('Negative compiled price hypothesis; no clamp')
    measurement=B.p.measurement(row,axis)
    measurement['law_context'].update(price_law='supplied_price_model_not_qualified_law_expectation',
                                      ideal_sampler_distance_qualified=None)
    return dict(**measurement, identity=deepcopy(row['identity']),law_binding=deepcopy(row['law_binding']),
        physical_context=deepcopy(scope['physical_context']),price_profile=deepcopy(scope['price_profile']),
        axis=axis,m=m,quantity=QUANTITY,cohort=model.get('cohort','SUPPLIED_MODEL'),n_paths=None,
        classification='MODEL_PREDICTION' if value is not None else 'UNAVAILABLE',
        expectation='SUPPLIED_MODEL_EXPECTATION' if value is not None else 'UNAVAILABLE',
        mean=dict(rz_count=value),source=deepcopy(model['source']),
        price_status='UNQUALIFIED_MODEL' if value is not None else 'MISSING',reason=reason,
        model=dict(name=model.get('name',mode),mode=mode,source=deepcopy(model['source']),
            context=scope,domain=deepcopy(row['candidate']),binding=deepcopy(request['configuration']['spec']['binding']),
            compiler=scope['price_profile']['compiler'],qualified_for_candidate=False,
            q_variable='event_count' if scope['method']=='FULL_H_RTE' else 'q_PF',
            q=row['structure']['event_count'] if scope['method']=='FULL_H_RTE' else row['q_PF'],
            coverage='EXACT_REQUEST_METADATA_ONLY_NOT_VALIDATION',validation_source=None,
            supplied_model=deepcopy(model),term_details=details),
        uncertainty=dict(joint_covariance=None,total_CI=None,long_q_R_systematic=None,law_distance=None,
            supplied_model=deepcopy(model.get('uncertainty')),features=deepcopy(features),
            structural=deepcopy(structural),residuals=deepcopy(residuals),independence_assumed=False),
        normalization_is_price_multiplier=False,full_RPE_price_adopted=False)


class PriceProvider:
    """Passive all-method record selection -> existing I.compare/B.cost inputs.

    DIRECT evidence remains externally supplied; no receipt acceptance audit or
    numeric model evaluation occurs. Missing cells reuse I's pending metadata.
    """
    def __init__(self,root,request,records=(),*,profile_extension,selections=None,comparison_policy=None):
        self.root=root;self.request=deepcopy(request);self.records=deepcopy(list(records))
        self.extension=deepcopy(profile_extension);self.selections=deepcopy(selections or {})
        self.comparison_policy=deepcopy(comparison_policy)

    def __call__(self):
        req=self.request;B=I.providers(str(self.root)).B;I._validate(self.root,req)
        pending=I.price_requests(self.root,req);scopes={(r['m'],r['axis']):context(self.root,req,r['m'],r['axis'],
            profile_extension=self.extension) for r in pending}
        grouped={k:[] for k in scopes};seen=set()
        for record in self.records:
            m=record.get('identity',{}).get('m');key=m,record.get('axis')
            if key not in scopes:raise ValueError('Foreign method/round/axis record')
            scope=scopes[key];row=req['configuration']['rounds'][m]
            if (any(not I.C.p.same(record.get(k),scope[k]) for k in ('identity','law_binding','physical_context','price_profile'))
                    or record.get('quantity')!=QUANTITY or not record.get('source') or not record.get('cohort')
                    or record.get('normalization_is_price_multiplier') is not False):
                raise ValueError('Price METHOD/law/SOURCE/compiler/quantity binding differs')
            measure=B.p.measurement(row,key[1])
            if any(not I.C.p.same(record.get(k),measure[k]) for k in ('measurement_axis','observable','outcome_convention','units')):
                raise ValueError('Price measurement/unit differs')
            if record.get('classification')=='MODEL_PREDICTION':
                model=record.get('model') or {}
                if (model.get('qualified_for_candidate') is not False or not I.C.p.same(model.get('context'),scope)
                        or not I.C.p.same(model.get('domain'),row['candidate'])):
                    raise ValueError('Experimental model qualification or context differs')
            signature=key+(record['cohort'],)
            if signature in seen:raise ValueError('Duplicate price cohort; do not pool')
            seen.add(signature);grouped[key].append(record)
        prices=[];bindings=[];cohorts={}
        for missing in pending:
            m=missing['m'];axis=missing['axis'];row=req['configuration']['rounds'][m];scope=scopes[m,axis]
            selector=(row['candidate']['id'],axis);matches=grouped[m,axis]
            if selector in self.selections:matches=[r for r in matches if r['cohort']==self.selections[selector]]
            if len(matches)==1:chosen=deepcopy(matches[0])
            else:
                chosen=deepcopy(missing)
                chosen.update(classification='UNAVAILABLE',expectation='UNAVAILABLE',mean=dict(rz_count=None),model=None,
                    cohort='MODEL_PRICE_MISSING',n_paths=None,price_profile=deepcopy(scope['price_profile']),
                    price_status='SELECTION_REQUIRED' if matches else 'MISSING',
                    uncertainty=dict(joint_covariance=None,total_CI=None,law_distance=None),
                    normalization_is_price_multiplier=False)
            prices.append(chosen);cohorts[selector]=chosen['cohort']
            bindings.append(dict(identity=deepcopy(row['identity']),law_binding=deepcopy(row['law_binding']),axis=axis,
                cohort=chosen['cohort'],price_source=deepcopy(chosen['source']),source=dict(module=MODULE),
                physical_context=deepcopy(scope['physical_context']),profile_extension=deepcopy(self.extension),mean_interval=None))
            for field in ('implementation','implementation_evidence'):
                if field in chosen:bindings[-1][field]=deepcopy(chosen[field])
        out=dict(prices=prices,price_bindings=bindings,cohorts=cohorts,price_extension=deepcopy(self.extension))
        if self.comparison_policy is not None:
            from .price_implementation_guard import check
            check(self.root,req['configuration'],prices,cohorts=cohorts,bindings=bindings,policy=self.comparison_policy)
            out['comparison_policy']=deepcopy(self.comparison_policy)
        return out
