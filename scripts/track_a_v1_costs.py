"""Future saved/model accounting, not executed on H6 during preparation."""
from copy import deepcopy
from fractions import Fraction
import math


def freeze_models(anchor_strata, *, contract, eligible_ids, transfer_receipts):
    """Freeze every S2 fit and separate S4 schedule hypothesis BEFORE compile.

    Each item supplies an independently verified stratum. Transfer receipts must
    cover the model's actual source/profile/control/phase scope; flags alone are
    not evidence. B0 and B0I have independent fits. S4 is not an S2 measurement.
    """
    models=[]
    for item in anchor_strata:
        fit=contract.affine_fit(item['rows'],stratum=item['stratum'])
        models.append(dict(stratum=fit['stratum'],coefficients=[str(v) for v in fit['coefficients']],
            maximum_anchor_relative_error=str(fit['maximum_anchor_relative_error']),
            anchor_ids=fit['anchor_ids'],validation_status='FROZEN_UNVALIDATED_MODEL'))
    axes=contract.select_validation_axes(eligible_ids);predictions=[]
    for axis in axes:
        c=axis['cell']
        matches=[m for m in models if m['stratum']['method']==c['method'] and m['stratum']['axis']==axis['axis']]
        if len(matches)!=1:raise ValueError('UNIQUE_S2_PARENT_STRATUM_REQUIRED')
        model=matches[0];a,b,d=map(Fraction,model['coefficients'])
        # Three signed Yoshida pieces. The native unmerged directional policy
        # has a single final wrapper/scalar; its per-step structure is tripled.
        value=a+3*c['q']*b+3*c['prefix']*c['q']*d
        if value<=0:raise ValueError('NONPOSITIVE_S4_HYPOTHESIS')
        predictions.append(dict(cell_id=c['id'],axis=axis['axis'],predicted=float(value),
            exact_prediction=str(value),kind='UNVALIDATED_S4_SCHEDULE_COMPOSITION_HYPOTHESIS',
            parent_stratum=model['stratum']))
    if not transfer_receipts or any(x.get('verified') is not True or not x.get('receipt_sha256') for x in transfer_receipts):
        raise ValueError('PHASE_PROFILE_COMPILER_TRANSFER_RECEIPTS_REQUIRED')
    return dict(schema='V1_frozen_cost_models_v1',models=models,validation_axes=axes,predictions=predictions,
                transfer_receipts=transfer_receipts,phase_and_profile_transfer_verified=True,
                fitted_after_validation=False,compiled_population_bound=False)


def map_costs(signal_records, cost_rows, *, contract, legacy_accounting, epsilon, factor, envelope=False):
    """One precision/scope; selected results preserve model/measured labels."""
    if epsilon not in contract.EPSILONS or factor not in contract.FACTORS:raise ValueError('PRECISION_DOMAIN')
    rows=[]
    for signal in signal_records:
        record=deepcopy(signal)
        if envelope:
            _,b=contract.signal_envelopes(record)
            if b is None:
                rows.append(dict(cell_id=record['cell']['id'],eligible=None,G=None,cost_status='UNRESOLVED_BIAS'))
                continue
            record['error_decomposition']['total_signed']=b
        accounting=legacy_accounting.accounting(record,epsilon,factor)
        usable=accounting['cost_acquisition_eligible']
        status=accounting['eligibility_under_declared_allowance']
        if not usable and status!='INELIGIBLE':
            rows.append(dict(cell_id=record['cell']['id'],eligible=None,G=None,cost_status='NUMERICALLY_UNRESOLVED'))
            continue
        selected=[x for x in cost_rows if x['cell_id']==record['cell']['id']]
        if usable and (len(selected)!=2 or {x['axis'] for x in selected}!=set(contract.AXES)):
            raise ValueError('ELIGIBLE_COST_MISSING')
        cost=None;kind='NOT_NEEDED_INELIGIBLE'
        if usable:
            if any(x['status'] not in ('MEASURED_COMPLETE','VALIDATED_MODEL_ESTIMATE') for x in selected):
                raise ValueError('UNVALIDATED_COST_CANNOT_SELECT')
            cost=sum(Fraction(x['RZ'])*accounting['axes']['real' if x['axis']=='cosine' else 'imag']['shots']
                     for x in selected)
            kind='VALIDATED_MODEL_ESTIMATE' if any(x['status']=='VALIDATED_MODEL_ESTIMATE' for x in selected) else 'MEASURED_COMPLETE'
        rows.append(dict(cell_id=record['cell']['id'],eligible=usable,G=cost,cost_status=kind,
                         accounting=accounting,selection_is_point_estimate=True,
                         conservative_bias_certified=False))
    return dict(rows=rows,best=contract.best(rows),epsilon=epsilon,factor=factor,
                no_cancellation_diagnostic=envelope,empirical_allowance=True)
