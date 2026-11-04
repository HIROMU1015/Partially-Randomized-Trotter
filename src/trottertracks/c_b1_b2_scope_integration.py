"""Pending-B2 evidence partitions; no H4 science or probability engine.

Finite diagnostics and confidence-qualified expectations are useful different
scopes, but are not silently reclassified as the old all-word evidence lane.
"""
from copy import deepcopy
from fractions import Fraction
from pathlib import Path

from . import c1_stage1_execution as io
from . import c_iid_grid_accuracy as iid
from .c_iid_grid_bridge import MODEL

BASE='69c9a764aee66cf2ca79ab29bd3145a027385bcc'
STAGE_A='ae9a4e13d1ab6f48fc4d38cbbcd0fddca30e9c53'
STAGE_A_SOURCE='ddd8333488e07a3b3af9095b5cb193ef9f90b3c3'
FAMILY='artifacts/research_c_b1_b2_scope_integration/2026-11-03'
CONTRACT='docs/research/research_c_b1_b2_accuracy_scope_design_unapproved_20261103.json'
ENTRY='scripts/prepare_research_c_b1_b2_scope_integration.py'
FORBIDDEN={'PF','corrected_RTE','DF','RPE_statistics','PCG64_joint_stream'}


def identity():
    from .c_all_law_signal_ca import CANDIDATE,PR_SOURCE,fixed
    from .c1_h4_native_fair import profile_dict
    from .c_pr_aware_df_stage_c import TARGET
    from .c1_h6_compiler_precision import VERSION as PHASE
    from .c_pr_aware_df_stage_b import NATIVE_POLICY
    return dict(candidate=CANDIDATE,PR_scientific_SOURCE=PR_SOURCE,
        stage_A_scientific_SOURCE=STAGE_A_SOURCE,stage_A_result=STAGE_A,
        model=deepcopy(MODEL),compiler=profile_dict(),conditions=fixed()['conditions'],
        accuracy_target=deepcopy(TARGET),phase_policy=PHASE,native_policy=NATIVE_POLICY)


def integrate(packet,records,publications,coverage):
    """Validate common scientific target,disjoint terms and scope,then reuse.

    Source identities here refer to the common scientific target. B1/B2
    implementation commits are different role-specific provenance, not an
    equality requirement. Publication pins establish availability, not proofs.
    """
    expected=identity()
    if set(records)!=set(iid.PARTS):raise ValueError('FOUR_DISJOINT_PARTITIONS')
    components={};classes={};missing=[];seen=set()
    for name,record in records.items():
        if record is None:
            components[name]=None;classes[name]='UNAVAILABLE';missing.append(name);continue
        if record['identity']!=expected:raise ValueError('CANDIDATE_SCIENTIFIC_SOURCE_MODEL_COMPILER')
        if record['quantity']!=name or record['norm']!='raw_complex_signal':raise ValueError('RAW_COMPLEX_UNITS')
        terms=record['included_error_terms']
        if (not terms or any(not isinstance(x,str) or not x for x in terms)
                or len(set(terms))!=len(terms) or seen.intersection(terms) or FORBIDDEN.intersection(terms)):
            raise ValueError('DOUBLE_COUNT_OR_NONPARTITION_ERROR')
        seen.update(terms)
        space={'synthesis':'full512_native','categorical':'ideal_trajectory_signal',
            'reference_bridge':'sector36_reference','reference_uncertainty':'sector36_reference'}[name]
        if record['space']!=space or record.get('phase_sensitive') is not True:
            raise ValueError('PHASE_OR_NATIVE_REFERENCE_SPACE')
        if name=='synthesis' and record.get('actual_inverse_and_shells') is not True:
            raise ValueError('ACTUAL_INVERSE_AND_SHELL_COVERAGE')
        if record.get('coherent_model')!='independent_IID_kernels':raise ValueError('COHERENT_OCCURRENCE_MODEL')
        rows=record['rows']
        if [x['m'] for x in rows]!=list(range(13)):raise ValueError('ALL_THIRTEEN_ROUNDS')
        for row in rows:
            if row['quantity']!=name or row['norm']!='raw_complex_signal':raise ValueError('ROUND_UNITS')
            if row.get('kind')=='rigorous_bound' and (record['qualification']!='bound' or row.get('rounding_enclosed') is not True):
                raise ValueError('ESTIMATE_IS_NOT_RIGOROUS_BOUND')
        qualified=record['qualification']=='bound' and all(x.get('kind') in ('numerical_bound','rigorous_bound') and x.get('bound') is not None for x in rows)
        scope=record['scope']
        if scope not in ('all_words','model_reference','fixed_finite_family','probabilistic_expectation'):
            raise ValueError('DECLARED_SCOPE')
        required_scope='all_words' if name=='synthesis' else 'model_reference'
        if scope!=required_scope:
            classes[name]=scope.upper()+'_NOT_OLD_ALL_WORD_EVIDENCE'
            missing.append(name+':scope_requires_separate_qualification')
            qualified=False
        else:classes[name]='SUPPLIED_BOUND' if qualified else record['qualification'].upper()
        components[name]=rows if qualified else None
        if not qualified:missing.append(name+':qualified_bound')
    if publications.get('B1_result')!=BASE:raise ValueError('FIXED_B1_PUBLIC_RESULT')
    if not publications.get('B2_result') or not publications.get('B2_SOURCE'):
        missing.append('B2_publication_and_SOURCE_not_observed')
    # Missing public B2 result blocks even wholly invented non-null values.
    # Do not call any accuracy calculation while availability/scope is pending.
    if missing:
        return dict(packet,error_evidence=None),dict(status='PENDING_B2_OR_BOUND_INPUTS',eligible=False,
            missing=sorted(set(missing)),classification=classes,iid_qualification=None,
            actual_CA_achieved=None,rigorous_CA=None,PCG64_joint_stream_qualified=False,
            PCG64_joint_stream_error=None,mandatory_stop=True)
    attached,q=iid.qualify(packet,expected_identity=expected,observed_identity=expected,
        components=components,coverage=coverage,model=MODEL)
    return attached,dict(status='CONDITIONAL_IID_INPUTS_ONLY' if not q['missing'] else 'NOT_READY_IID_QUALIFICATION',
        eligible=not q['missing'],iid_qualification=q,classification=classes,missing=q['missing'],
        actual_CA_achieved=None,rigorous_CA=None,PCG64_joint_stream_qualified=False,
        PCG64_joint_stream_error=None,mandatory_stop=True)


def assess_if_ready(target,packet,integration,call=lambda k,l,f:f()):
    if not integration['eligible']:
        return dict(status='NOT_READY_NO_STAGE_C_CALLS',rounds=[],energy=None,
            actual_CA_achieved=None,rigorous_CA=None,mandatory_stop=True)
    return iid.assess(target,packet,integration['iid_qualification'],call)


def confidence_schema(*,measurement_alpha,calibration_delta,reported_alpha,mode):
    """Metadata-only policy check;no new sampling,CI or shot formula.

    Existing measurement alpha is fixed by the target. A calibration event
    has an additional confidence level, not a free part of that alpha.
    """
    a=Fraction(str(measurement_alpha));d=None if calibration_delta is None else Fraction(str(calibration_delta))
    report=Fraction(str(reported_alpha))
    if not 0<a<1 or not 0<report<1 or d is not None and not 0<d<1:raise ValueError('FAILURE_PROBABILITIES')
    if mode not in ('conditional_on_calibration_event','unconditional_union'):raise ValueError('FAILURE_CLAIM_SCOPE')
    if d is None:return dict(status='CALIBRATION_CONFIDENCE_UNFIXED',compatible=False)
    need=a if mode=='conditional_on_calibration_event' else a+d
    return dict(status='PROPOSED_CONFIDENCE_SCHEMA_COMPATIBLE' if need<=report else 'REQUIRES_SCIENTIFIC_FAILURE_POLICY_DECISION',
        compatible=need<=report,measurement_policy_changed=False,
        calibration_is_separate=True,PCG64_IID_proven=False,adopted=False)


def fixed():
    return dict(scope='C_B1_B2_SCOPE_INTEGRATION_PENDING_B2_V1',base_commit=BASE,
        B2_result_commit=None,B2_SOURCE_commit=None,status='PENDING_B2_PUBLICATION',
        executable=False,maximum_logical_units=None,Stage_C='NOT_READY',
        new_scientific_claim_adopted=False,retry=0,resume=0,mandatory_stop=True,
        fresh_output=FAMILY+'/run01')


def preflight(root,path):
    root=Path(root).resolve();c=io.read(path)
    if any(c.get(k)!=v for k,v in fixed().items()):raise ValueError('FIXED_PENDING_B2_DESIGN')
    if io.sha(root/c['source_manifest'])!=c['source_manifest_sha256']:raise ValueError('SOURCE_MANIFEST')
    source=io.read(root/c['source_manifest'])
    for group in [source['files'],c['saved_inputs']]:
        for name,want in group.items():
            p=(root/name).resolve()
            if not p.is_relative_to(root) or io.sha(p)!=want:raise ValueError('SOURCE_INPUT:'+name)
    stop=io.read(root/c['stage_A_STOP'])
    if stop['scientific_units']!=38 or stop['failed'] or not stop['mandatory_stop'] or stop['retry'] or stop['resume']:
        raise ValueError('STAGE_A_PRESERVED_STOP')
    if (root/c['fresh_output']).exists():raise ValueError('FRESH_NO_RETRY_RESUME')
    return dict(status='PASSIVE_PENDING_INTEGRATION_CHECKS_PASS',B2='NOT_OBSERVED_PUBLISHED_AT_START',
        source_files=len(source['files']),saved_inputs=len(c['saved_inputs']),
        scientific_calls=0,Stage_C_calls=0,execution_ready=False,maximum_logical_units=None,mandatory_stop=True)


def run_science(*,execute_science=False):
    if not execute_science:raise PermissionError('SEPARATE_FIXED_SCIENCE_SCOPE_REQUIRED')
    raise NotImplementedError('B2_PENDING;NO_ADOPTED_FINITE_OR_PROBABILISTIC_SCIENCE_CONTRACT')
