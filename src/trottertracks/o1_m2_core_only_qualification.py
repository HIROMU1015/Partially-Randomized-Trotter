"""Conditional model/omission budget arithmetic; no molecular I/O or adoption.

Accepts the existing core_only_bound result schema. Supplied model and omission
bounds must refer to the same B operator and saved sector. Their validity is
assumed, not established here; native/PF/signal/RPE qualification remains open.
"""
from collections.abc import Mapping
from fractions import Fraction


def _number(name, value, *, positive=False):
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(name+' must be an explicit numeric bound')
    try:
        value=Fraction(value)
    except (ValueError,TypeError,OverflowError) as exc:
        raise ValueError(name+' must be a finite numeric bound') from exc
    if value<0 or (positive and value==0):
        raise ValueError(name+' must be finite and '+('positive' if positive else 'nonnegative'))
    return value


def _sector(value):
    if value is None:
        return None
    if (not isinstance(value,(tuple,list)) or len(value)!=2
            or any(isinstance(x,bool) or not isinstance(x,int) or x<0 for x in value)):
        raise ValueError('Declared sector must be two nonnegative integers')
    return tuple(value)


def assess_model_omission_budget(omission_result=None, *, physical_reference=None,
        model_b_reference=None, model_sector=None, model_error_upper_bound=None,
        model_omission_allocation=None):
    """Check only an explicitly allocated model-plus-omission error component.

    Missing inputs/reference mismatches are UNASSESSED. An upper bound above
    allocation is INCONCLUSIVE, not proof of actual error or required PR.
    The exact sum of supplied binary64 upper bounds avoids rounding down at the
    budget boundary. No result file is read and no norm is recalculated.
    """
    if omission_result is not None and not isinstance(omission_result,Mapping):
        raise ValueError('Omission result must use the existing mapping schema')
    omission_result={} if omission_result is None else omission_result
    record=omission_result.get('energy_bound_reference_H_H_vs_C_H',{})
    if not isinstance(record,Mapping):
        raise ValueError('Omission energy bound must be a mapping')
    omission=_number('Omission upper bound',record.get('upper_bound'))
    model=_number('Model upper bound',model_error_upper_bound)
    allocation=_number('Model/omission allocation',model_omission_allocation,positive=True)
    b_reference=omission_result.get('reference')
    sector=_sector(omission_result.get('sector'))
    model_sector=_sector(model_sector)
    missing=[name for name,value in (
        ('physical_reference',physical_reference),('omission_B_reference',b_reference),
        ('model_B_reference',model_b_reference),('omission_sector',sector),
        ('model_sector',model_sector),('model_upper_bound',model),
        ('omission_upper_bound',omission),('model_omission_allocation',allocation))
        if value is None or (isinstance(value,str) and not value.strip())]
    for value in (physical_reference,b_reference,model_b_reference):
        if value is not None and not isinstance(value,str):
            raise ValueError('References must be explicit nonempty labels or None')
    incompatible=[]
    if b_reference and model_b_reference and b_reference!=model_b_reference:
        incompatible.append('B_reference_mismatch')
    if sector is not None and model_sector is not None and sector!=model_sector:
        incompatible.append('sector_mismatch')
    result=dict(scope='CONDITIONAL_MODEL_AND_CORE_OMISSION_ALLOCATION_ONLY',
        budget_status='UNASSESSED',missing=missing,incompatible=incompatible,
        combined_upper_bound_exact_fraction=None,
        physical_reference=physical_reference,b_reference=b_reference,sector=sector,
        input_bound_validity='ASSUMED_NOT_VALIDATED',
        physical_model_status='UNASSESSED',native_core_status='UNASSESSED',
        full_rpe_status='UNASSESSED',residual_randomization_needed=None)
    if missing or incompatible:
        return result
    combined=model+omission
    result['combined_upper_bound_exact_fraction']=str(combined)
    result['budget_status']=('CONDITIONALLY_WITHIN_DECLARED_MODEL_OMISSION_ALLOCATION'
        if combined<=allocation else 'BOUND_INCONCLUSIVE')
    return result
