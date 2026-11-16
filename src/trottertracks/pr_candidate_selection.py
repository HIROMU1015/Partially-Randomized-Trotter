"""Normalize a B2-selected input request, without choosing a candidate or rank.

Legacy B2 KAPPA_PROPOSAL uses a multiplier coordinate. Resolve its absolute
value only from explicit B2 source candidates (or B2's generated canonical body).
Never substitute the new-split centre when that coordinate is unresolved.
"""
from copy import deepcopy
from fractions import Fraction
import hashlib
import json


def exact(value):
    return Fraction(int(value['numerator']),int(value['denominator']))if isinstance(value,dict)else Fraction(value)


def identity(selection):
    """Stable identity of the selected work, excluding changing cost diagnostics."""
    keys=('kind','candidate','configuration_id','configuration','family','kappa','kappa_multiplier',
          'base_configuration_id','base_configuration_source')
    value={k:selection[k]for k in keys if k in selection}
    if selection.get('kind')=='KAPPA_PROPOSAL'and selection.get('kappa')is None:
        value['coordinate_sources']=[{k:r[k]for k in ('from_configuration','from_kappa')if k in r}
            for r in selection.get('initial_candidates',[])]
    return hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest()[:24]


def configuration_for(context,cid,reader):
    candidate=next((c for c in context['candidates']if c.get('configuration_id')==cid),None)
    if candidate:
        if candidate.get('generated_configuration'):return candidate['generated_configuration'],candidate
        if candidate.get('configuration_source'):return reader(candidate['configuration_source']),candidate
    point=next((p for p in context.get('catalog',{}).get('candidates',[])if p.get('configuration_id')==cid),None)
    if point:return reader(point['configuration_source']),candidate
    raise ValueError('B2_SELECTED_CONFIGURATION_SOURCE_UNAVAILABLE:'+str(cid))


def _absolute_from_b2_sources(selection,context,reader,canonical_candidate=None):
    if selection.get('kappa')is not None:
        return exact(selection['kappa']),dict(kind='B2_ABSOLUTE_VALUE',target=str(exact(selection['kappa'])))
    if selection.get('kappa_multiplier')is None:raise ValueError('B2_KAPPA_VALUE_OR_MULTIPLIER_REQUIRED')
    multiplier=exact(selection['kappa_multiplier']);evidence=[]
    # A B2-generated canonical candidate identifies the precise base used by its
    # preparation adapter. That is stronger than choosing a neighbouring LD.
    base_id=(canonical_candidate or{}).get('generated_from')or selection.get('base_configuration_id')
    if base_id:
        base,_=configuration_for(context,base_id,reader)
        absolute=exact(base['spec']['kappa']['value'])*multiplier
        return absolute,dict(kind='B2_EXPLICIT_PREPARATION_BASE',base_configuration_id=base_id,
            base_kappa=str(exact(base['spec']['kappa']['value'])),multiplier=str(multiplier),target=str(absolute))
    if selection.get('base_configuration_source'):
        base=reader(selection['base_configuration_source']);absolute=exact(base['spec']['kappa']['value'])*multiplier
        return absolute,dict(kind='B2_EXPLICIT_BASE_LOCATOR',base_configuration_source=selection['base_configuration_source'],
            base_kappa=str(exact(base['spec']['kappa']['value'])),multiplier=str(multiplier),target=str(absolute))
    # Published P.kappa_plan carries from_configuration + from_kappa for its
    # transferred seeds. Both are supplied by B2; infer and cross-check its
    # coordinate origin rather than selecting a new base ourselves.
    for seed in selection.get('initial_candidates',[]):
        if not seed.get('from_configuration')or seed.get('from_kappa')is None:continue
        cfg,_=configuration_for(context,seed['from_configuration'],reader)
        coordinate=exact(seed['from_kappa'])
        if coordinate<=0:raise ValueError('B2_SOURCE_KAPPA_COORDINATE_NONPOSITIVE')
        absolute=exact(cfg['spec']['kappa']['value'])*multiplier/coordinate
        evidence.append(dict(configuration_id=cfg['id'],source_absolute=str(exact(cfg['spec']['kappa']['value'])),
                             source_coordinate=str(coordinate),target=str(absolute)))
    if evidence:
        targets={exact(x['target'])for x in evidence}
        if len(targets)!=1:raise ValueError('B2_KAPPA_COORDINATE_BASES_DISAGREE')
        target=next(iter(targets))
        return target,dict(kind='B2_TRANSFER_SOURCE_COORDINATE',multiplier=str(multiplier),target=str(target),sources=evidence)
    raise ValueError('B2_MULTIPLIER_REQUIRES_EXPLICIT_BASE_OR_PREPARED_CANONICAL_CONFIGURATION')


def normalize(method,selection,context,defaults,reader,prepare_kappa=None):
    """Return one completion request or an explicit held normalization request.

    prepare_kappa, if supplied, is the published B2 preparation adapter; it may
    mutate context/register canonical metadata but must not run scientific leaves.
    """
    selected=deepcopy(selection);request=dict(method=method,selection=selected,selection_identity=identity(selected))
    try:
        if selected['kind']=='KAPPA_PROPOSAL':
            prepared_id=(selected.get('preparation')or{}).get('configuration_id')
            if prepared_id is None and prepare_kappa is not None:
                selected=prepare_kappa(selected);request['selection']=selected
                prepared_id=(selected.get('preparation')or{}).get('configuration_id')
            cfg=candidate=None
            if prepared_id:cfg,candidate=configuration_for(context,prepared_id,reader)
            target,normalization=_absolute_from_b2_sources(selected,context,reader,candidate)
            if target<=0:raise ValueError('B2_TARGET_ABSOLUTE_KAPPA_NONPOSITIVE')
            request['kappa_normalization']=normalization
            if cfg is not None:
                if exact(cfg['spec']['kappa']['value'])!=target:raise ValueError('B2_PREPARED_ABSOLUTE_KAPPA_MISMATCH')
                family=selected.get('family')or{}
                if family.get('L_D')is not None and cfg['spec']['L_D']!=family['L_D']:raise ValueError('B2_PREPARED_SPLIT_MISMATCH')
                if family.get('formula')is not None and cfg['spec']['formula']!=family['formula']:raise ValueError('B2_PREPARED_FORMULA_MISMATCH')
                if family.get('delta_over_h')is not None and exact(cfg['spec']['delta_pf'])/exact(cfg['spec']['h'])!=exact(family['delta_over_h']):
                    raise ValueError('B2_PREPARED_REQUESTED_WIDTH_MISMATCH')
                request['configuration']=deepcopy(cfg);return request
            design=deepcopy(defaults);design.update(selected['family']);design.pop('kappa_multiplier',None);design['kappa']=str(target)
            request['design']=design;return request
        cid=selected.get('configuration_id')
        candidate=next((c for c in context['candidates']if c.get('name')==selected.get('candidate')),None)
        cid=cid or(candidate or{}).get('configuration_id')
        if cid:
            cfg,_=configuration_for(context,cid,reader);request['configuration']=deepcopy(cfg);return request
        design=deepcopy(defaults);design.update(selected.get('family')or selected.get('configuration')or{})
        if selected.get('kappa')is not None:design['kappa']=selected['kappa']
        if 'kappa_multiplier'in design:raise ValueError('NEW_SPLIT_MULTIPLIER_HAS_NO_EXPLICIT_B2_BASE')
        if not design.get('L_D')or design.get('kappa')is None:raise ValueError('B2_SELECTED_SPLIT_AND_ABSOLUTE_KAPPA_REQUIRED')
        request['design']=design;return request
    except (ValueError,KeyError,StopIteration)as exc:
        request['selection_error']=str(exc);return request
