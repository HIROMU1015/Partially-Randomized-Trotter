"""All-law signal/accuracy connection, passive until called with bound inputs.

This module propagates supplied scientific evidence; schemas do not prove it.
The current H4 design has no executable scientific entry: P0 coverage and the
actual joint random-stream/reference correspondence remain unresolved.
"""
from fractions import Fraction
import importlib.metadata
import math
from pathlib import Path

from . import c1_stage1_execution as io

BASE = '2f1c86f161fb76e84dfa6b31f6912eb8a01ccb8f'
PR_SOURCE = '4df8eca29c59ffd55e2dd898f05da16bb246ddbe'
CANDIDATE = 'a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a'
FAMILY = 'artifacts/research_c_all_law_signal_ca/2026-11-03'
FRESH = FAMILY + '/run01'
ENTRY = 'scripts/prepare_research_c_all_law_signal_ca.py'
CONTRACT = 'docs/research/research_c_all_law_signal_ca_contract_unapproved_20261103.json'
COVERAGE = 'all_trajectories_all_occurrences_and_shells'
KINDS = ('numerical_bound', 'rigorous_bound')
PARTS = ('synthesis', 'categorical', 'reference_bridge', 'reference_uncertainty', 'stream_model')
ACCURACY_LIMITS = dict(signal_qualification=1, round_accuracy=13,
    energy_assessment=1, qualification_summary=1)


def records():
    root = 'artifacts/research_c_near_practical_r32/2026-11-03/run01/'
    stat = 'artifacts/research_c_pr_xdf_statistics/2026-11-03/run01/'
    return dict(PR_result=root+'result.json', PR_contract=root+'contract.json',
        PR_SOURCE=root+'SOURCE.json', PR_STOP=root+'STOP.json',
        PR_law=root+'PR/A/law.json', PR_normalization=root+'PR/A/normalization.json',
        statistics_result=stat+'result.json', statistics_STOP=stat+'STOP.json',
        old_compiler='artifacts/research_c_pr_aware_df_c2_bridge/2026-10-27/technical/compiler_coverage.json',
        old_primitives='artifacts/research_c_pr_aware_df_c2_bridge/2026-10-27/technical/primitive_coverage.json')


def fixed():
    return dict(scope='C_PR_R32_ALL_LAW_SIGNAL_CA_READINESS_V1', base_commit=BASE,
        status='UNAPPROVED_DESIGN_NOT_EXECUTABLE', executable=False, entry=ENTRY,
        candidate_sha256=CANDIDATE, PR_scientific_SOURCE=PR_SOURCE,
        records=records(), fresh_output=FRESH, retry=0, resume=0, mandatory_stop=True,
        error_evidence=None, maximum_logical_units=None,
        accuracy_adapter_call_limits=ACCURACY_LIMITS,
        conditions=dict(molecule='linear H4',distance_A=1.,basis='STO-3G',Nalpha=2,Nbeta=2,
            sector_dimension=36,controlled_dimension=512,rank=8,ld=2,s=1,r=32,K=0,
            h='1/10',rounds=list(range(13)),oracle='own_DF_ground_state_oracle',
            state_preparation='EXCLUDED',CA_Ha=.00159,DF_reserve_Ha=.00019875),
        blockers=['whole-kernel P0 uniform transformation/reference coverage',
            'current-candidate atom/angle/normalization to saved CAR reference bound',
            'actual PCG64 joint stream versus IID-grid model and independent-shot premise'])


def _m(m):
    if type(m) is not int or not 0 <= m <= 12: raise ValueError('REGISTERED_ROUND')


def categorical_preview(table, m, *, independent_occurrences=True):
    """Generalize old r16*s4 coupling to current r32*s1, still IID-only.

    Uses C2's exact categorical TV result; never evaluates a real table here.
    No attenuation multiplier: this is already raw complex signal distance.
    """
    from . import c_pr_aware_df_stage_c2 as c2
    _m(m)
    if independent_occurrences is not True: raise ValueError('COHERENT_OCCURRENCE_LAW_CHANGED')
    if table['status'] != 'EXACT_CATEGORICAL_TABLE_DISTANCE_UNDER_IID_GRID_MODEL':
        raise ValueError('CATEGORICAL_TABLE_SCOPE')
    eta = c2.rational(table['eta_exact'])
    if not 0 <= eta <= 1: raise ValueError('TV_RANGE')
    n = 32*(1 << m)
    bound = min(Fraction(2), 2*n*eta)
    return dict(m=m, events=n, random_model=c2.IID,
        conditional_raw_bound_exact=str(bound), conditional_raw_bound_upper=c2.upper_float(bound),
        actual_implemented_law_error=None, reference_bridge_included=False,
        synthesis_included=False, PCG64_IID_proven=False)


def numerical_synthesis_rows(kernel, shells, coverage):
    """Reuse both-norm-growth full-Fock propagation on uniform inputs only.

    Each input must already bound its WHOLE final compiled segment versus the
    exact saved-atom map for every relevant word. Finite sample residuals and
    event-wise compiled residuals cannot establish this precondition.
    The supplied kernel bound/norm guard must cover both forward and actual
    inverse outputs wherever the protocol uses them, not a substituted adjoint.
    """
    from .c1_h4_full_synthesis import propagate
    required = dict(scope=COVERAGE, compiler_path='whole_kernel_P0',
        final_transform_covered=True, phase_sensitive=True,
        actual_adjoint_covered=True, dimension=512)
    if any(coverage.get(k) != v for k,v in required.items()) or not coverage.get('provenance'):
        raise ValueError('UNIFORM_FINAL_P0_COVERAGE_REQUIRED')
    if coverage.get('kind') != 'all_law_numerical_bound':
        raise ValueError('NUMERICAL_INPUT_NOT_A_RIGOROUS_ENCLOSURE')
    result=[]
    for m in range(13):
        # PF is deliberately absent from this synthesis-only propagation.
        value=propagate(kernel,shells,s=1,m=m,saved_pf_error=0.)
        result.append(dict(m=m,kind='numerical_bound',quantity='synthesis',
            norm='raw_complex_signal',bound=value['complex_signal_bound_pilot'],
            provenance=coverage['provenance'],rounding_enclosed=False))
    return result


def exact_unitary_preview(factors, shells, m):
    """Existing exact rational proof-input algebra, not validation of proofs."""
    from .c_pr_aware_df_stage_c2 import synthesis_preview
    _m(m)
    return synthesis_preview(factors,path='whole_kernel_P0',occurrences=1 << m,shells=shells)


def attach_evidence(packet, *, expected_identity, observed_identity, components, premises):
    """Attach disjoint raw-signal terms to the existing C1 evidence hook.

    Law slot = categorical + reference bridge + its uncertainty + stream gap.
    Saved PF/RTE remain in RoundEvidence and DF stays in the energy stage.
    Unknown/conditional-only/observed terms do not become zero or actual-law
    evidence. Supplied coverage/premises are scientific inputs, not proofs.
    """
    from . import c_pr_aware_df_stage_c as c1
    from . import c_pr_aware_df_stage_c2 as c2
    if observed_identity != expected_identity: raise ValueError('CANDIDATE_SOURCE_COMPILER_LAW_IDENTITY')
    if [r['m'] for r in packet['rounds']] != list(range(13)): raise ValueError('ROUND_COVERAGE')
    if set(components) != set(PARTS): raise ValueError('DISJOINT_COMPONENT_SCHEMA')
    missing=[];values={};rigorous=True
    for name, rows in components.items():
        if rows is None:
            missing.append(name);continue
        if [r['m'] for r in rows] != list(range(13)): raise ValueError('ROUND_COVERAGE:'+name)
        bound=[]
        for row in rows:
            if row.get('kind') not in KINDS or row.get('bound') is None:
                missing.append(name);break
            if (row.get('norm') != 'raw_complex_signal' or row.get('quantity') != name
                    or not row.get('provenance')): raise ValueError('RAW_DISJOINT_BOUND_SCOPE')
            if name == 'reference_bridge' and row.get('reference_uncertainty_included') is not False:
                raise ValueError('REFERENCE_UNCERTAINTY_DOUBLE_COUNT')
            x=c2.rational(row['bound'])
            if x < 0: raise ValueError('NEGATIVE_ERROR')
            bound.append(x)
            rigorous = rigorous and row['kind']=='rigorous_bound' and row.get('rounding_enclosed') is True
        if len(bound)==13: values[name]=bound
    needed = dict(coverage=COVERAGE, actual_joint_stream_covered=True,
        independent_shots_justified=True, independent_coherent_occurrences=True,
        final_whole_kernel_P0_covered=True, phase_sensitive=True,
        actual_adjoint_covered=True, controlled_dimension=512,
        reference_bridge_target='saved_CAR_raw_finite_RTE',
        synthesis_target='exact_saved_atom_map_under_same_actual_stream')
    missing.extend('premise:'+k for k,v in needed.items() if premises.get(k) != v)
    if not premises.get('provenance'): missing.append('premise:provenance')
    attached=dict(packet,error_evidence=None)
    if not missing:
        rows=[dict(m=m,synthesis=c2.upper_float(values['synthesis'][m]),
            law=c2.upper_float(sum((values[n][m] for n in PARTS if n!='synthesis'),Fraction()))) for m in range(13)]
        attached['error_evidence']=dict(kind='rigorous_all_law_bound' if rigorous else 'all_law_numerical_bound',
            norm='raw_complex_signal',law_model='actual_implemented_law',coverage=COVERAGE,
            provenance=premises['provenance'],round_errors=rows,
            component_bounds=components,proofs_verified_by_schema=False)
    q=c1.qualification(attached,'evidence')
    return attached,dict(q,missing=sorted(set(missing)),
        actual_CA_achieved=None,guaranteed_CA=None,ideal_law_cost_transfer_qualified=False)


def assess_ready_packet(target, packet, qualification, call=lambda k,l,f:f()):
    """Existing C1 round/energy engines; at most 16 logical calls.

    Current saved PF/DF/reference remain numerical pilots even when supplied
    synthesis/law inputs are rigorous. No shot execution or actual CA claim.
    """
    from . import c_pr_aware_df_stage_c as c1
    q=call('signal_qualification','existing evidence hook',lambda:c1.qualification(packet,'evidence'))
    plans=[];energy=None
    if q['kind']!='UNQUALIFIED' and not qualification['missing']:
        for row in packet['rounds']:
            value=call('round_accuracy',str(row['m']),lambda row=row:c1.assess_round(target,packet,q,row))
            plans.append(value)
            if value['status']!='CONDITIONAL_ROUND_PASS':break
        if len(plans)==13 and all(x['status']=='CONDITIONAL_ROUND_PASS' for x in plans):
            energy=call('energy_assessment','existing own-DF/alias energy',lambda:c1.energy(target,packet,plans))
    def summary():
        ready=energy is not None and energy['same_accuracy']
        return dict(status='NUMERICAL_IMPLEMENTED_LAW_SUFFICIENT_CONDITION' if ready else 'UNRESOLVED_OR_INFEASIBLE_IMPLEMENTED_LAW',
            rounds=plans,energy=energy,actual_CA_achieved=None,guaranteed_CA=None,
            quantum_shots_drawn=0,state_preparation='ORACLE_COST_EXCLUDED',mandatory_stop=True)
    return call('qualification_summary','A/B/C/D kept separate',summary)


def preflight(root, path):
    """Technical design/input binding only; no propagation, table or shots."""
    root=Path(root).resolve();c=io.read(path)
    for k,v in fixed().items():
        if c[k]!=v:raise ValueError('FIXED_DESIGN:'+k)
    if io.sha(root/c['source_manifest'])!=c['source_manifest_sha256']:raise ValueError('SOURCE_MANIFEST')
    source=io.read(root/c['source_manifest'])
    needed=set(records().values())|set(c['scientific_input_roles'].values())
    if not needed<=c['saved_inputs'].keys() or ENTRY not in source['files']:raise ValueError('UNPINNED_INPUT_OR_ENTRY')
    for group in (source['files'],c['saved_inputs']):
        for name,digest in group.items():
            p=(root/name).resolve()
            if not p.is_relative_to(root) or io.sha(p)!=digest:raise ValueError('SOURCE_INPUT:'+name)
    for name,v in c['packages'].items():
        if importlib.metadata.version(name)!=v:raise ValueError('ENVIRONMENT:'+name)
    d={k:io.read(root/p) for k,p in records().items()}
    pr=d['PR_result']['arms']['PR']
    old=d['PR_contract']
    identity_inputs={old['endpoints']['PR']['candidate']:CANDIDATE,
        old['input']:'1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db'}
    if any(c['saved_inputs'].get(p)!=v for p,v in identity_inputs.items()):raise ValueError('CANDIDATE_INPUT_BINDING')
    if (pr['candidate']!=CANDIDATE or pr['PR']['r']!=32 or not pr['PR']['native_admitted']
            or d['PR_contract']['candidate_sha256']['PR']!=CANDIDATE
            or old['source_manifest_sha256']!='103c8ba7ecda6e5da48cc0e73f267befcf104c4ad3c18c32de875289fa532739'
            or pr['PR']['all_law_synthesis'] is not None or pr['PR']['implemented_law_error'] is not None
            or pr['PR']['actual_CA_achieved'] is not None or d['PR_law']['atoms']!=216
            or d['PR_contract']['compiler']!=c['compiler']):raise ValueError('SAVED_CANDIDATE_LAW_OR_QUALIFICATION')
    for key,n in [('PR_STOP',1231),('statistics_STOP',7)]:
        s=d[key]
        if s['failed'] or s['scientific_units']!=n or s['retry'] or s['resume'] or not s['mandatory_stop']:
            raise ValueError('SAVED_STOP')
    if (root/FRESH).exists():raise ValueError('FRESH_NO_RETRY_RESUME')
    return c,dict(status='PASSIVE_DESIGN_CHECK_PASS',scientific_calls=0,new_H4_error_or_shots=0,
        scientific_execution_ready=False,contract_status=c['status'],blockers=c['blockers'],
        current_error_evidence=None,actual_CA_achieved=None)


def run(root, path, *, execute_science=False):
    if not execute_science:raise PermissionError('SEPARATE_SCIENTIFIC_PERMISSION_REQUIRED')
    # Permission cannot supply the missing scientific method/coverage.
    raise RuntimeError('DESIGN_NOT_EXECUTABLE_NO_SCIENTIFIC_CALLS_OR_OUTPUT')
