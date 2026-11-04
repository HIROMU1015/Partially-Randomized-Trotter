"""IID-grid qualification and reference-transport use of the existing engine.

No old evidence lane or scientific SOURCE is changed. In particular no
IID-grid record is presented as actual_implemented_law evidence.
"""
from fractions import Fraction
import math
from . import c_all_law_signal_ca as previous
from .c_iid_grid_bridge import MODEL

PARTS=tuple(k for k in previous.PARTS if k!='stream_model')
LIMITS=previous.ACCURACY_LIMITS


def qualify(packet, *, expected_identity, observed_identity, components, coverage, model=MODEL):
    from . import c_pr_aware_df_stage_c2 as c2
    if model!=MODEL:raise ValueError('FIXED_IID_GRID_MODEL')
    if set(components)!=set(PARTS):raise ValueError('IID_DISJOINT_COMPONENTS')
    # Reuse all old unit/identity/round/disjointness/coverage validation, while
    # explicitly leaving every ACTUAL-stream premise unqualified.
    premises=dict(coverage,actual_joint_stream_covered=False,
        independent_shots_justified=False,independent_coherent_occurrences=False)
    actual,old=previous.attach_evidence(packet,expected_identity=expected_identity,
        observed_identity=observed_identity,components=dict(components,stream_model=None),premises=premises)
    expected_actual_gaps={'stream_model','premise:actual_joint_stream_covered',
        'premise:independent_shots_justified','premise:independent_coherent_occurrences'}
    missing=sorted(set(old['missing'])-expected_actual_gaps)
    rows=[]
    if not missing:
        for m in range(13):
            values={k:c2.rational(components[k][m]['bound']) for k in PARTS}
            rows.append(dict(m=m,synthesis=c2.upper_float(values['synthesis']),
                law_and_reference=c2.upper_float(sum((v for k,v in values.items() if k!='synthesis'),Fraction())),
                components={k:c2.upper_float(v) for k,v in values.items()}))
    assert actual['error_evidence'] is None
    return actual,dict(kind='IID_GRID_CONDITIONAL_NUMERICAL_BOUND' if not missing else 'UNQUALIFIED_IID_GRID',
        law_model=MODEL['id'],model=MODEL,round_errors=rows,missing=missing,
        iid_grid_error_evidence=not missing,actual_error_evidence=None,
        actual_law_qualified=False,PCG64_joint_stream_qualified=False,PCG64_joint_stream_error=None,
        actual_CA_achieved=None,rigorous_CA=None,source_qualification='saved PF/DF/reference numerical pilot retained')


def assess(target,packet,qualification,call=lambda k,l,f:f()):
    """Maximum16 calls; arithmetic reference law is distinct from observed law.

    The unchanged legacy engine recognizes three law tags. We use its IDEAL
    REFERENCE transport calculation with ALL known non-PF/RTE raw differences
    in its additive raw-error slot, then label the returned plan as IID-grid.
    The reference's self-law error is zero by definition, not an unknown
    PCG64 discrepancy assigned zero. Original component names/values remain
    attached to each plan; no actual-law tag is ever supplied to the engine.
    """
    from . import c1_stage3_accuracy as accuracy
    from . import c_pr_aware_df_stage_c as c1
    def gate():
        if qualification['model']!=MODEL or qualification['law_model']!=MODEL['id']:
            raise ValueError('IID_QUALIFICATION_MODEL')
        if qualification['actual_error_evidence'] is not None or packet.get('error_evidence') is not None:
            raise ValueError('DO_NOT_IMPERSONATE_ACTUAL_LAW')
        if qualification['kind']!='UNQUALIFIED_IID_GRID' and [x['m'] for x in qualification['round_errors']]!=list(range(13)):
            raise ValueError('ROUND_COVERAGE')
        return qualification
    q=call('signal_qualification','IID-grid only,actual stream unknown',gate)
    plans=[];energy=None
    if q['kind']=='IID_GRID_CONDITIONAL_NUMERICAL_BOUND' and not q['missing']:
        for row in packet['rounds']:
            def one(row=row):
                errors=q['round_errors'][row['m']];ref=row['reference']
                extra=math.fsum(errors['components'].values())
                ev=accuracy.RoundEvidence(m=row['m'],attenuation=ref['attenuation'],
                    pf_operator_error=ref['pf_operator_error'],rte_corrected_operator_error=ref['rte_corrected_operator_error'],
                    synthesis_signal_error=extra,finite_law_signal_error=0.,
                    target_id=packet['target_id'],state_model='own_DF_ground_state_oracle',
                    alias_branch_verified=True,iid_per_shot=True,
                    evidence_kind='numerically_qualified_pilot',law_model='ideal_finite_RTE_law')
                value=accuracy.assess_round(target,ev)
                return dict(value,law_model=MODEL['id'],
                    engine_reference_law='ideal_finite_RTE_law',
                    engine_raw_error_slot='sum of all four non-PF/RTE raw transport components',
                    raw_error_components=errors['components'],IID_assumptions=MODEL,
                    actual_CA_achieved=None,rigorous_CA=None,PCG64_joint_stream_qualified=False)
            value=call('round_accuracy',str(row['m']),one);plans.append(value)
            if value['status']!='CONDITIONAL_ROUND_PASS':break
        if len(plans)==13 and all(p['status']=='CONDITIONAL_ROUND_PASS' for p in plans):
            energy=call('energy_assessment','same existing DF once',lambda:c1.energy(target,packet,plans))
    return call('qualification_summary','conditional grid numerical sufficiency only',lambda:dict(
        status='IID_GRID_CONDITIONAL_NUMERICAL_SUFFICIENT_CONDITION' if energy and energy['same_accuracy']
            else 'UNRESOLVED_OR_INFEASIBLE_IID_GRID',model=MODEL,rounds=plans,energy=energy,
        actual_CA_achieved=None,rigorous_CA=None,PCG64_joint_stream_qualified=False,
        PCG64_joint_stream_error=None,quantum_shots_drawn=0,mandatory_stop=True))
