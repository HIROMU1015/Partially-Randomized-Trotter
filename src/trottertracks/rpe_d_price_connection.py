"""Passive D request/path/externally supplied mean -> existing B price inputs.

No Runtime/Adapter construction, event sampling, preparation, compile, averaging,
fit, interval construction or scientific qualification takes place here.
"""
from copy import deepcopy

from . import rpe_optimization_pipeline as I

MODULE='src/trottertracks/rpe_d_price_connection.py'
PATH='PR_CANDIDATE_PAIRED_PATH_PRICE'
MEAN='FULL_WRAPPER_MEAN_COMPILED_PRICE'


def extension(d_request):
    return dict(lowering_id=d_request['physical_context']['price_profile']['lowering_id'],
        source=dict(commit=I.D_ADAPTER_SOURCE,path='src/trottertracks/efficient_controlled_zz.py'))


def pending_requests(root,request,*,input_source,execution_source,axes=('X','Y'),preparation_hashes=None,expected_scalar=None):
    """Use D.from_a_job (which calls from_b) and D.pending_price_records only."""
    I._validate(root,request);api=I.providers(str(root));hashes=preparation_hashes or {}
    if not input_source or not execution_source:raise ValueError('D input and execution SOURCE are required')
    if request['provider']!='A_PR':
        rows=I.price_requests(root,request,axes=axes)
        for row in rows:row.update(price_status='OUTSIDE_DOMAIN',reason='PUBLISHED_D_ADAPTER_IS_H4_PR_ONLY')
        return rows
    jobs=request['pending']['jobs']
    if not set(hashes)<=set(range(len(jobs))):raise ValueError('Foreign preparation-hash round')
    rows=[]
    for job in jobs:
        d=api.Price.from_a_job(request['configuration'],job,input_source=input_source,
            execution_source=execution_source,axes=axes,preparation_hash=hashes.get(job['m']),
            expected_scalar=expected_scalar)
        for row in api.Price.pending_price_records(d):
            row.update(m=job['m'],configuration_id=request['configuration']['id'],
                physical_context=I.price_context(request,job['m']),d_request=deepcopy(d),
                profile_extension=extension(d))
            rows.append(row)
    return rows


def _requests(root,request,pending):
    api=I.providers(str(root));result={};seen=set()
    for row in pending:
        d=row.get('d_request')
        if d is None:raise ValueError('No supported D request; preserve OUTSIDE_DOMAIN separately')
        api.Price.check_request(d);m=d['m']
        if type(m) is not int or m not in range(request['configuration']['spec']['M']+1):raise ValueError('Foreign D round')
        pc=d['physical_context']
        expected=api.Price.from_a_job(request['configuration'],request['pending']['jobs'][m],
            input_source=pc['input_source'],execution_source=pc['execution_source'],axes=d['axes'],
            preparation_hash=pc['preparation_hash'],expected_scalar=pc.get('expected_scalar'))
        if not I.C.p.same(d,expected):raise ValueError('D request does not belong to this physical configuration')
        if (row.get('axis') not in d['axes'] or not I.C.p.same(row.get('identity'),d['identity'])
                or not I.C.p.same(row.get('law_binding'),d['law_binding'])
                or not I.C.p.same(row.get('physical_context'),I.price_context(request,m))):
            raise ValueError('Pending record axis/physical/law differs from D request')
        key=(m,row['axis'])
        if key in seen:raise ValueError('Duplicate pending axis')
        seen.add(key);result[d['request_id']]=d
    for d in result.values():
        if {(d['m'],a) for a in d['axes']}-seen:raise ValueError('Missing pending record for a requested axis')
    # A schedule may change q/R/law by round, but cannot change its restored
    # input or preparation definition half way through the same price input.
    anchors=[dict({k:d['physical_context'][k] for k in ('binding','split','input_source','execution_source','implementation','price_profile','preparation_hash')},
                  expected_scalar=d['physical_context'].get('expected_scalar')) for d in result.values()]
    if anchors and any(not I.C.p.same(x,anchors[0]) for x in anchors[1:]):
        raise ValueError('Mixed preparation/input/execution/compiler/scalar domain in one schedule')
    return result


def _paths(requests,paths,exact_scalar):
    """Metadata binding only. Keep values and law diagnostics, never average."""
    by_request={k:[] for k in requests};seen=set();resolved=None
    for path in paths:
        d=path.get('request') or {};key=d.get('request_id')
        if key not in requests or not I.C.p.same(d,requests[key]):raise ValueError('Foreign path request/SOURCE/law')
        if (path.get('quantity')!=PATH or path.get('mean_compiled_price') is not None
                or path.get('confidence_interval') is not None or path.get('qualified_expectation') is not False):
            raise ValueError('A path record is not mean or interval evidence')
        pc=d['physical_context'];src=path.get('source') or {}
        if not I.C.p.same(src.get('execution'),pc['execution_source']) or not I.C.p.same(src.get('input'),pc['input_source']):
            raise ValueError('Path input/execution SOURCE mismatch')
        if (not path.get('cohort') or type(path.get('index')) is not int or path['index']<0
                or type(path.get('conditional')) is not bool):raise ValueError('Path cohort/index/conditional role missing')
        role=(key,path['cohort'],path['index'],path['conditional'])
        if role in seen:raise ValueError('Duplicate path identity; do not double-count')
        seen.add(role)
        axes=[]
        for record in path.get('records',[]):
            axis=record.get('axis')
            if (axis not in d['axes'] or axis in axes
                    or record.get('compiled_quantity')!='FULL_WRAPPER_PATH_COMPILED_RZ_COUNT'
                    or record.get('structural_quantity')!='FULL_WRAPPER_PATH_STRUCTURAL_RZ_COUNT'):
                raise ValueError('Path axis/quantity mismatch')
            axes.append(axis)
        r=path.get('resolved_context') or {}
        fields=('preparation_hash','tail_hash','lambda_R','scalar','deterministic_fragments','randomized_fragments')
        if any(r.get(k) is None for k in fields):raise ValueError('Resolved path preparation metadata missing')
        if pc['preparation_hash'] is not None and r['preparation_hash']!=pc['preparation_hash']:
            raise ValueError('Resolved preparation hash differs')
        if pc.get('expected_scalar') is not None and float(r['scalar']).hex()!=float(pc['expected_scalar']).hex():
            raise ValueError('Resolved scalar differs from the requested total scalar')
        prefix=pc['physical_candidate']['prefix'];rank=pc['binding']['rank']
        if (float(exact_scalar(r['lambda_R'])).hex()!=float(exact_scalar(pc['split']['lambda_R'])).hex()
                or r['deterministic_fragments']!=list(range(prefix)) or r['randomized_fragments']!=list(range(prefix,rank))):
            raise ValueError('Resolved lambda or fragment split differs from request')
        expected={k:r[k] for k in fields}
        if resolved is not None and not I.C.p.same(resolved,expected):raise ValueError('Mixed resolved preparation/tail within schedule')
        resolved=expected
        by_request[key].append(path)
    return by_request


def bind_results(root,request,pending,*,paths=(),prices=(),bindings=(),interval_contract=None):
    """Map external mean evidence, or return D's unavailable responses.

    bindings are the existing B sidecars plus d_request_id and d_context_key.
    Path/structural/finite-cohort data never become qualified expectations here.
    B retains final expectation/model/law/interval qualification checks.
    """
    I._validate(root,request);api=I.providers(str(root))
    requests=_requests(root,request,pending);grouped=_paths(requests,paths,api.B.a.exact_scalar)
    selected={};unmapped=[];out=[];out_bindings=[];cohorts={}
    for price in prices:
        m=price.get('identity',{}).get('m');axis=price.get('axis');key=(m,axis)
        if key in selected:raise ValueError('Select one mean cohort/SOURCE per axis; do not pool')
        selected[key]=price
    expected_cells={(d['m'],a) for d in requests.values() for a in d['axes']}
    if not set(selected)<=expected_cells:raise ValueError('External price has an unrequested round/axis')
    if any((b.get('identity',{}).get('m'),b.get('axis')) not in expected_cells for b in bindings):
        raise ValueError('External binding has an unrequested round/axis')
    for d in requests.values():
        pc=d['physical_context'];ext=extension(d)
        source_refs=[dict(source=p['source'],cohort=p['cohort'],index=p['index'],conditional=p['conditional'],
            axes=[r['axis'] for r in p.get('records',[])]) for p in grouped[d['request_id']]]
        unavailable=api.Price.pending_price_records(d,path_source=source_refs or None)
        for missing in unavailable:
            m=d['m'];axis=missing['axis'];candidate=selected.get((m,axis));chosen=missing;side=None
            matches=[b for b in bindings if b.get('axis')==axis and b.get('identity',{}).get('m')==m]
            if len(matches)>1:raise ValueError('Ambiguous external mean binding')
            if candidate is not None:
                if candidate.get('quantity')!=MEAN:raise ValueError('Path or structural quantity cannot supply a mean')
                if (any(p['conditional'] and p['cohort']==candidate.get('cohort') for p in grouped[d['request_id']])
                        and candidate.get('forced_cohort') is not True):
                    raise ValueError('Conditional path cohort cannot be relabelled as normal mean evidence')
                if (not I.C.p.same(candidate.get('identity'),d['identity'])
                        or not I.C.p.same(candidate.get('law_binding'),d['law_binding'])):raise ValueError('Mean identity/law mismatch')
                measurement=api.B.p.measurement(dict(identity=d['identity'],structure=d['identity']['structure']),axis)
                if any(not I.C.p.same(candidate.get(k),measurement[k]) for k in ('measurement_axis','observable','outcome_convention','units')):
                    raise ValueError('Mean axis/wrapper/unit mismatch')
                side=matches[0] if matches else None
                required=('source','price_source','profile_extension','physical_context','d_request_id','d_context_key')
                if not candidate.get('source') or side is None or any(side.get(k) is None for k in required):
                    unmapped.append(dict(m=m,axis=axis,reason='MEAN_SOURCE_OR_D_BINDING_MISSING'));side=None
                else:
                    if (not I.C.p.same(side['price_source'],candidate['source']) or side.get('cohort')!=candidate.get('cohort')
                            or not I.C.p.same(side.get('identity'),d['identity']) or not I.C.p.same(side.get('law_binding'),d['law_binding'])
                            or side['d_request_id']!=d['request_id'] or side['d_context_key']!=d['context_key']
                            or not I.C.p.same(side['physical_context'],I.price_context(request,m))
                            or not I.C.p.same(side['profile_extension'],ext)
                            or not I.C.p.same(candidate.get('price_profile'),pc['price_profile'])):
                        raise ValueError('Mean SOURCE/physical/compiler/lowering/cohort binding differs')
                    chosen=deepcopy(candidate);side=deepcopy(side)
            if side is None:
                side=dict(identity=deepcopy(d['identity']),law_binding=deepcopy(d['law_binding']),axis=axis,
                    cohort=chosen['cohort'],price_source=deepcopy(chosen['source']),
                    source=dict(module=MODULE,D=I.D_ADAPTER_SOURCE),profile_extension=deepcopy(ext),
                    physical_context=I.price_context(request,m),d_request_id=d['request_id'],d_context_key=d['context_key'],mean_interval=None)
            chosen.update(m=m,physical_context=I.price_context(request,m))
            out.append(chosen);out_bindings.append(side)
            cohorts[d['identity']['candidate']['id'],axis]=chosen['cohort']
    return dict(prices=out,cohorts=cohorts,price_extension=extension(next(iter(requests.values()))) if requests else None,
        price_bindings=out_bindings,interval_contract=deepcopy(interval_contract),paths=deepcopy(list(paths)),
        supplied_prices=deepcopy(list(prices)),supplied_bindings=deepcopy(list(bindings)),unmapped=unmapped,
        status='EXTERNAL_MEAN_EVIDENCE_OR_PENDING_NOT_QUALIFIED_BY_MAPPING',
        numerical_mean_or_interval_computation=False,source=dict(module=MODULE,D=I.D_ADAPTER_SOURCE))
