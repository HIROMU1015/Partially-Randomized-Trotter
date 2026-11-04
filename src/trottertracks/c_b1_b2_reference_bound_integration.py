"""Read-only B2 evidence transport into the existing IID-grid adapter.

No molecular arrays, Arb engine, accuracy evaluation or probability estimator.
Exact rational arithmetic below checks recorded consistency, not new bounds.
The old pending-B2 adapter and its fixed design remain unchanged.
"""
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
from . import c_b1_b2_scope_integration as previous

io = previous.io
BASE = '90a2a7baf07b98108111126504d202294a884af5'
B2_RESULT = 'b7a3db09e395d36965e5b9dc28ae4a299db3b2f0'
B2_SOURCE = '01baf2b5e400773e00a5ae548941265f85b9882b'
B2_IMPLEMENTATION = '918c63cc2a959f59f2817388bf6be549b905e3e6'
B2_FAMILY = 'artifacts/research_c_reference_bound_contract/2026-11-03'
FAMILY = 'artifacts/research_c_b1_b2_reference_bound_integration/2026-11-03'
CONTRACT = 'docs/research/research_c_b1_b2_reference_bound_design_unapproved_20261103.json'
ENTRY = 'scripts/prepare_research_c_b1_b2_reference_bound_integration.py'
PARTS = dict(reference_bridge=('bridge', 'atom_to_mathematical_CAR'),
             reference_uncertainty=('reference_uncertainty', 'mathematical_CAR_to_binary64_reference'))
TARGETS = dict(
 atom='exact saved atom coefficients,dyadic saved U,saved angle bits;exact p;literal CAR Z/ZZ',
 CAR='exact dyadic saved sector generators and lambda/tau/identity;h=1/10;K0 polynomial;A=(1+tau^2)^(-16*q)',
 numeric='binary64 saved corrected matrix;shared NumPy power chain;each saved binary64 attenuation',
 state='all normalized sector36 vectors;no saved-state expectation or oracle-precision assertion')


def publications():
    return dict(B1_result=previous.BASE, B2_result=B2_RESULT, B2_SOURCE=B2_SOURCE,
                B2_implementation=B2_IMPLEMENTATION, Stage_A=previous.STAGE_A)


def evidence_paths():
    return {**{k:B2_FAMILY+'/run01/'+k+'.json' for k in ('result','STOP','SOURCE','contract')},
            'technical_SOURCE':B2_FAMILY+'/technical/SOURCE.json'}


def _rounds(rows):
    if [r['m'] for r in rows] != list(range(13)) or any(type(r['m']) is not int for r in rows):
        raise ValueError('THIRTEEN_REGISTERED_ROUNDS')


def _upper(value):
    # Do not float-cast/outward-round the saved exact endpoints during transport.
    if not isinstance(value,str):raise ValueError('EXACT_RATIONAL_BOUND_STRING')
    try:q=Fraction(value)
    except (ValueError,ZeroDivisionError) as exc:raise ValueError('EXACT_RATIONAL_BOUND') from exc
    if q<0:raise ValueError('NONNEGATIVE_BOUND')
    return q


def b2_records(result, *, pins):
    """Validate the fixed two legs; return original bound strings unchanged.

    This checks metadata/schema, not the mathematical proof or physical input
    accuracy. The recorded total/direct diagnostic never becomes a third term.
    """
    if pins != publications():raise ValueError('FIXED_PUBLIC_RESULT_SOURCE_ROLES')
    if (result['status']!='B2_REFERENCE_OPERATOR_BOUNDS_COMPLETE'
        or result['model']!=previous.MODEL or result['targets']!=TARGETS):
        raise ValueError('FIXED_B2_TARGET_MODEL')
    for key in ('Stage_C_ready','categorical_unitary_trajectory_premise_qualified',
                'scientific_input_uncertainty_covered','saved_ground_precision_qualified',
                'generator_construction_uncertainty_covered','PCG64_joint_stream_qualified'):
        if result[key] is not False:raise ValueError('B2_COVERAGE_OVERCLAIM:'+key)
    if result['actual_CA_achieved'] is not None or result['mandatory_stop'] is not True:
        raise ValueError('B2_PRESERVED_QUALIFICATION')
    if set(result['records'])!=set(PARTS):raise ValueError('TWO_LEGS_NO_TOTAL_COMPONENT')
    _rounds(result['rounds']);records=deepcopy(result['records'])
    for name,(leg,term) in PARTS.items():
        record=records[name]
        expected=dict(identity=previous.identity(),quantity=name,norm='raw_complex_signal',
            space='sector36_reference',phase_sensitive=True,coherent_model='independent_IID_kernels',
            scope='model_reference',reference_target='saved_CAR_raw_finite_RTE',
            raw_conversion='uniform over all normalized sector36 states;no saved-state approximation',
            qualification='bound',numerical_qualification='rigorous_enclosure',included_error_terms=[term])
        if any(record.get(k)!=v for k,v in expected.items()):raise ValueError('B2_IDENTITY_UNITS_PARTITION_COVERAGE:'+name)
        error=record['error_model']
        wanted=dict(targets=TARGETS,precision_bits=384,rounding_enclosed=True,saved_state_used=False,
            ground_oracle_precision_claimed=False,physical_input_error_included=False,
            generator_construction_error_included=False)
        if error!=wanted or record.get('proofs_verified_by_schema') is not False or not record.get('provenance'):
            raise ValueError('B2_ERROR_MODEL_QUALIFICATION')
        _rounds(record['rows'])
        for row,origin in zip(record['rows'],result['rounds']):
            bound=_upper(origin[leg]['upper_exact'])
            if (origin[leg]['norm']!='operator_2' or origin[leg]['upper_method']!='outward Frobenius;no numerical SVD'
                or origin['norm_conversion']!='operator2<=outward_Frobenius;|psi*D psi|<=operator2 for every normalized psi'
                or origin['rounding_enclosed'] is not True or origin['saved_state_used'] is not False
                or row.get('rounding_enclosed') is not True or row.get('kind')!='rigorous_bound'
                or row.get('quantity')!=name or row.get('norm')!='raw_complex_signal'
                or _upper(row['bound'])!=bound or _upper(row['value'])!=bound):
                raise ValueError('B2_EXACT_ENDPOINT_OR_CONVERSION')
            if name=='reference_bridge' and row.get('reference_uncertainty_included') is not False:
                raise ValueError('REFERENCE_UNCERTAINTY_DOUBLE_COUNT')
        record['fixed_publication']=dict(result_commit=B2_RESULT,science_SOURCE=B2_SOURCE,
                                         implementation_SOURCE=B2_IMPLEMENTATION)
    for row in result['rounds']:
        # Equality of existing fields only; do not publish a newly computed sum.
        if _upper(row['total_upper_exact'])!=_upper(row['bridge']['upper_exact'])+_upper(row['reference_uncertainty']['upper_exact']):
            raise ValueError('SAVED_TOTAL_LEGS_INCONSISTENT')
    return records


def categorical_record(stage_a):
    """Transport saved conditional TV leg; its unitary premise stays false."""
    if (stage_a['status']!='IID_GRID_REFERENCE_BRIDGE_NUMERICAL_PILOT_COMPLETE'
        or stage_a['candidate']!=previous.identity()['candidate']
        or stage_a['PR_scientific_SOURCE']!=previous.identity()['PR_scientific_SOURCE']
        or stage_a['model']!=previous.MODEL or stage_a['PCG64_joint_stream_qualified'] is not False):
        raise ValueError('STAGE_A_CANDIDATE_SOURCE_MODEL')
    _rounds(stage_a['rounds']);rows=[]
    for row in stage_a['rounds']:
        v=row['conditional_categorical'];_upper(v['conditional_raw_bound_exact'])
        if (row['model']!=previous.MODEL or v['m']!=row['m'] or v['random_model']!=previous.MODEL['id']
            or v['events']!=32*(1<<row['m']) or v['reference_bridge_included'] is not False
            or v['synthesis_included'] is not False or v['actual_implemented_law_error'] is not None):
            raise ValueError('CATEGORICAL_PARTITION_MODEL')
        rows.append(dict(m=row['m'],quantity='categorical',norm='raw_complex_signal',
            bound=v['conditional_raw_bound_exact'],kind='numerical_bound',
            provenance='fixed Stage A conditional categorical coupling;unitary premise unresolved'))
    return dict(identity=previous.identity(),quantity='categorical',norm='raw_complex_signal',
        included_error_terms=['categorical_p_to_q_only'],space='ideal_trajectory_signal',
        phase_sensitive=True,coherent_model='independent_IID_kernels',qualification='conditional_bound_pending_trajectory_premise',
        numerical_qualification='conditional_on_unitary_trajectory',scope='model_reference',rows=rows,
        categorical_unitary_trajectory_premise_qualified=False)


def integrate(packet, result, stage_a, *, pins, synthesis=None, coverage):
    records=b2_records(result,pins=pins)
    records.update(synthesis=synthesis,categorical=categorical_record(stage_a))
    # Reuse all existing identity/unit/disjoint-term/scope validators. A missing
    # categorical trajectory premise must block even invented synthesis bounds.
    declared=deepcopy(coverage)
    declared['categorical_unitary_trajectory_premise_qualified']=False
    attached,q=previous.integrate(packet,records,pins,declared)
    missing=set(q['missing'])
    missing.add('categorical_unitary_trajectory_premise')
    if synthesis is None:missing.add('whole_P0_uniform_synthesis')
    if not missing:raise AssertionError('FIXED_B2_PREMISE_MUST_REMAIN_UNQUALIFIED')
    return dict(attached,error_evidence=None),dict(q,status='NOT_READY_BOUND_INPUTS',eligible=False,
        iid_qualification=None,missing=sorted(missing),B1='P0_UNIFORM_NOT_ESTABLISHED',
        B2='FIXED_REFERENCE_LEGS_AVAILABLE',records=records,
        categorical_unitary_trajectory_premise_qualified=False,
        new_error_bounds_computed=False,Stage_C_calls=0)


def fixed():
    return dict(scope='C_B1_B2_FIXED_REFERENCE_BOUND_INTEGRATION_V1',base_commit=BASE,
        publications=publications(),identity=previous.identity(),status='NOT_READY_BOUND_INPUTS',
        executable=False,maximum_logical_units=None,Stage_C_maximum_adapter_units=16,
        scientific_calls_now=0,entry=ENTRY,retry=0,resume=0,mandatory_stop=True,
        fresh_output=FAMILY+'/run01',scientific_policy_adopted=False)


def preflight(root,path):
    root=Path(root).resolve();c=io.read(path)
    if any(c.get(k)!=v for k,v in fixed().items()):raise ValueError('FIXED_NONEXECUTABLE_DESIGN')
    if io.sha(root/c['source_manifest'])!=c['source_manifest_sha256']:raise ValueError('SOURCE_MANIFEST')
    source=io.read(root/c['source_manifest'])
    for group in (source['files'],c['saved_inputs']):
        for name,want in group.items():
            p=(root/name).resolve()
            if not p.is_relative_to(root) or io.sha(p)!=want:raise ValueError('SOURCE_INPUT:'+name)
    if c['B2_evidence']!=evidence_paths():raise ValueError('B2_SNAPSHOT_PATHS')
    d={k:io.read(root/p) for k,p in c['B2_evidence'].items()}
    contract,src,technical=d['contract'],d['SOURCE'],d['technical_SOURCE']
    if (contract['implementation_source_commit']!=B2_IMPLEMENTATION or src['implementation_commit']!=B2_IMPLEMENTATION
        or src['execution_commit']!=B2_SOURCE or {k:v for k,v in src.items() if k!='execution_commit'}!=technical
        or io.sha(root/c['B2_evidence']['contract'])!=c['B2_contract_sha256']
        or io.sha(root/c['B2_evidence']['technical_SOURCE'])!=contract['source_manifest_sha256']
        or contract['candidate_sha256']!=previous.identity()['candidate'] or contract['model']!=previous.MODEL
        or contract['stage_A_scientific_SOURCE']!=previous.STAGE_A_SOURCE
        or contract['stage_A_result']!=previous.STAGE_A or contract['targets']!=TARGETS):
        raise ValueError('B2_FIXED_SOURCE_INPUT_ROLES')
    # Bytes only, not decode of the preserved candidate/CAR array inputs.
    for p,want in contract['saved_inputs'].items():
        if c['saved_inputs'].get(p)!=want or io.sha(root/p)!=want:raise ValueError('B2_ORIGINAL_INPUT:'+p)
    for stop,n in ((d['STOP'],27),(io.read(root/c['stage_A_STOP']),38)):
        if (stop['failed'] or stop['scientific_units']!=n or stop['retry'] or stop['resume']
            or stop['mandatory_stop'] is not True):raise ValueError('PRESERVED_SCIENTIFIC_STOP')
    if sum(d['STOP']['call_counts'].values())!=27 or d['STOP']['first_failure'] is not None:
        raise ValueError('B2_CALL_COMPLETION')
    b1=io.read(root/c['B1_STOP'])
    if b1['method_status']!='P0_UNIFORM_NOT_ESTABLISHED' or b1['actual_H4_scientific_calls']!=0:
        raise ValueError('B1_STATUS')
    if (root/c['fresh_output']).exists():raise ValueError('FRESH_NO_RETRY_RESUME')
    # Missing synthesis ensures no qualification arithmetic or accuracy calls.
    _,q=integrate(dict(rounds=[]),d['result'],io.read(root/c['stage_A_result']),
                  pins=c['publications'],coverage={})
    return dict(status='PASSIVE_FIXED_EVIDENCE_CHECK_PASS',Stage_C=q['status'],
        execution_ready=False,scientific_calls=0,Stage_C_calls=0,H4_array_decode=0,
        B2=q['B2'],B1=q['B1'],missing=q['missing'],records=q['records'],
        actual_CA_achieved=None,PCG64_joint_stream_qualified=False,mandatory_stop=True)


def run_science(*,execute_science=False):
    if not execute_science:raise PermissionError('SEPARATE_FIXED_SCIENCE_SCOPE_REQUIRED')
    raise NotImplementedError('WHOLE_P0_AND_TRAJECTORY_PREMISES_NOT_QUALIFIED')
