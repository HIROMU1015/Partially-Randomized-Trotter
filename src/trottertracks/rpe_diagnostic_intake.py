"""A m1 and C kernel/structural-S evidence. No numerical evaluation or pricing."""
from copy import deepcopy
from pathlib import Path
import re

from . import abc_s0_published_intake as S0

B, p, H = S0.B, S0.p, S0.H
A_PREPARATION = 'ac873e7fcc42e937084fa153476cbd87565702b4'
A_PREFLIGHT = 'docs/research/h4_m1_rmse_full_pf_preparation_20261113/PREFLIGHT.json'
M1_ID = 'c00e00dc1e9c414fb7d9'
M1_CARRIER = dict(commit='89c91c692e1d35de94d2e7afbe890aabe6d80b7e',
    path='docs/research/h4_df10_rpe_batch_results_20261110/run01/result.json', selector=['records',5])
C_DIAGNOSTIC = 'a637ed1b813dc3952339cb05f36beb79168ace23'
C_DIR = 'docs/research/h4_pr_physical_closure_20261113/'
C_LOCAL_SOURCE = 'eeccdf18ee839ccddb824afbc5c8d0aa8b52b7b6'
C_REDUCTION_SOURCE = '2b3b0ee7ffea92de25a2c0c549d64f258c34a611'
STRUCTURAL = 'MEAN_STRUCTURAL_RZ_COUNT_NOT_COMPILED_C'


def validate_m1_packet(config, packet, previous, *, numerical_source, adapter_source):
    """A new m1 scope, preserving original m0/m6 and every whole-schedule ID.

    numerical_source is the explicitly selected published A execution SOURCE,
    never an assertion that its values establish sufficient N or global RMSE.
    """
    same = H.same
    B._validate(config)
    H.validate_A_v2_packet(config, previous, p.SavedArtifacts(Path(__file__).resolve().parents[2]).read(H.A_PACKET))
    same(previous['adapter_source'], S0.A_ADAPTER_SOURCE, 'M1_PREVIOUS_ADAPTER_SOURCE_MISMATCH')
    same(config['id'], S0.CONFIGURATION_ID, 'M1_FIXED_CANONICAL_CONFIGURATION_REQUIRED')
    same(packet['configuration_id'], config['id'], 'M1_CONFIGURATION_MISMATCH')
    same(packet['policy'], config['spec']['rmse_policy'], 'M1_POLICY_MISMATCH')
    same(previous['configuration_id'], config['id'], 'M1_PREVIOUS_CONFIGURATION_MISMATCH')
    same(packet['preserved_S0_packet'], dict(commit=S0.A_RELEASE,path=S0.ADIR+'V2_N_HANDOFF.json'), 'M1_PREVIOUS_S0_SOURCE_MISMATCH')
    same(packet['diagnostic_scope'], dict(newly_reanalyzed_rounds=[1],preserved_S0_rounds=[0,6],
        other_rounds_NOT_EVALUATED=True,whole_schedule_RMSE_qualified=False), 'M1_NEW_SCOPE_REQUIRED')
    same(packet['B_S0_only_validator_applicable'], False, 'M1_IS_NOT_OLD_S0_SCOPE')
    if 'pilot_scope' in packet:
        raise ValueError('M1_MUST_NOT_POSE_AS_OLD_S0_PILOT')
    same(packet['source'], numerical_source, 'M1_NUMERICAL_SOURCE_MISMATCH')
    if (not re.fullmatch('[0-9a-f]{40}', numerical_source.get('commit',''))
            or numerical_source.get('path') != 'scripts/run_h4_pr_m1_rmse.py'
            or numerical_source.get('scope') != 'H4_M1_SAVED_SCALAR_ONLY'):
        raise ValueError('M1_FIXED_EXECUTION_SOURCE_REQUIRED')
    same(numerical_source['saved_carrier'], M1_CARRIER, 'M1_SAVED_CARRIER_SOURCE_MISMATCH')
    same(packet['adapter_source'], adapter_source, 'M1_ADAPTER_SOURCE_MISMATCH')
    if packet['status'] not in ('UNQUALIFIED','NUMERICAL_UNSTABLE','INFEASIBLE','NOT_EVALUATED'):
        raise ValueError('M1_NOMINAL_DIAGNOSTIC_IS_NOT_QUALIFIED_SUFFICIENT_N')
    same(packet['evidence_kind'], 'SCIENTIFIC', 'M1_EVIDENCE_KIND_MISMATCH')
    same(packet['nominal_promoted_to_sufficient'], False, 'M1_NOMINAL_PROMOTED')
    same(packet['technical_identity_verified'], True, 'M1_IDENTITY_NOT_VERIFIED')
    expected_rmse = dict(previous['rmse_evidence'], source=deepcopy(numerical_source))
    same(packet['rmse_evidence'], expected_rmse, 'M1_GLOBAL_RMSE_OR_ERROR_CONTRACT_CHANGED')
    same(len(packet['rounds']), 13, 'M1_WHOLE_SCHEDULE_REQUIRED')
    for row, old, request in zip(packet['rounds'], previous['rounds'], config['rounds']):
        m = request['m']
        for key, value in dict(m=m, identity=request['identity'], law_binding=request['law_binding'],
                source_candidate_id=request['physical_candidate']['id'], source_request_id=old['source_request_id'],
                N_X=None,N_Y=None, diagnostic_source=numerical_source if m==1 else previous['source'] if m in (0,6) else None).items():
            same(row[key], value, 'M1_ROW_IDENTITY_LAW_SOURCE_OR_SUFFICIENT_N_MISMATCH:' + key)
        if m != 1:
            for key in ('nominal_N_X','nominal_N_Y','model_status'):
                same(row[key], old[key], 'M1_OUTSIDE_DIAGNOSTIC_SCOPE_CHANGED')
        else:
            same(request['physical_candidate']['id'], M1_ID, 'M1_PHYSICAL_CANDIDATE_CHANGED')
            for axis in p.AXES:
                value = row['nominal_N_'+axis]
                if value is not None and (type(value) is not int or value < 1):
                    raise ValueError('M1_NOMINAL_MUST_REMAIN_EXACT_POSITIVE_INTEGER_OR_NULL')
            status = row['model_status']
            if (status not in ('NOT_EVALUATED','UNQUALIFIED_MODEL_ASSUMPTIONS') and
                    not status.startswith(('NUMERICAL','INAPPLICABLE'))):
                raise ValueError('M1_MODEL_STATUS_IS_NOT_SCIENTIFIC_PASS')
            if any(row['nominal_N_'+axis] is not None for axis in p.AXES) and status != 'UNQUALIFIED_MODEL_ASSUMPTIONS':
                raise ValueError('M1_NOMINAL_QUALIFICATION_CHANGED')
    return True


def accept_c_published(config, result, checks, reduction, plan):
    """Bind the already published local diagnosis, not K0/K1 computations."""
    same = H.same
    S0.C.verify_B_publication(plan,config,integration=H.I)
    expected_match = dict(commit=S0.A_RELEASE,path=S0.ADIR+'V2_CONFIGURATION.json',
        configuration_id=config['id'],status='EXACT_PUBLISHED_B_METADATA_MATCH')
    same(result['B_match'],expected_match,'C_PHYSICAL_CONFIGURATION_SOURCE_MISMATCH')
    same(checks['B_official_configuration_exact_match'],expected_match,'C_CHECKS_CONFIGURATION_MISMATCH')
    same(result['baseline'],S0.C_RELEASE,'C_DIAGNOSTIC_BASELINE_SOURCE_MISMATCH')
    same(result['execution_source'],checks['diagnostic_source'],'C_KERNEL_DIAGNOSTIC_SOURCE_MISMATCH')
    same(reduction['analysis_source'],checks['reduction_source'],'C_REDUCTION_SOURCE_MISMATCH')
    same(result['execution_source'],C_LOCAL_SOURCE,'C_PUBLISHED_LOCAL_SOURCE_MISMATCH')
    same(reduction['analysis_source'],C_REDUCTION_SOURCE,'C_PUBLISHED_REDUCTION_SOURCE_MISMATCH')
    same(result['preparation']['H'],config['spec']['binding']['hamiltonian'],'C_PREPARATION_H_MISMATCH')
    same(B.a.exact_value(B.a.exact_scalar(result['preparation']['lambda_R'])),config['spec']['split']['lambda_R'],'C_PREPARATION_LAMBDA_METADATA_MISMATCH')
    same([r['m'] for r in result['laws']],[0,6],'C_DIAGNOSTIC_DOMAIN_MISMATCH')
    for law in result['laws']:
        row = config['rounds'][law['m']]
        for key,value in dict(candidate_id=row['physical_candidate']['id'],q_PF=row['q_PF'],
                R_outer=row['R_outer'],event_count=row['structure']['event_count'],
                sampler_ideal_distance=None,seed_used_for_sampling=False).items():
            same(law[key],value,'C_DIAGNOSTIC_LAW_OR_PHYSICAL_MISMATCH:' + key)
    for key in ('mean_structural_price','mean_compiled_price','full_rpe_resources'):
        same(result[key],None,'C_LOCAL_DIAGNOSTIC_IS_NOT_MEAN_PRICE')
    same(result['qualified_for_candidate'],False,'C_DIAGNOSTIC_PRICE_QUALIFICATION_CHANGED')
    same(reduction['target'],'STRUCTURAL_RZ_COUNT_ONLY_NOT_FULL_ACCOUNTING_REPORT_OR_COMPILED_C','C_STRUCTURAL_QUANTITY_MISMATCH')
    return dict(kind='PHYSICAL_PRICE_STATE_DIAGNOSTIC_ONLY',source=dict(commit=C_DIAGNOSTIC,path=C_DIR+'run02/RESULT.json'),
        kernel_diagnostic_source=result['execution_source'],reduction_source=reduction['analysis_source'],
        preparation_hash=result['preparation']['hash'],B_match=deepcopy(result['B_match']),
        domains=[dict(m=r['m'],physical_candidate_id=r['candidate_id'],q_PF=r['q_PF'],R_outer=r['R_outer'],
            event_count=r['event_count'],runtime_sampler=r['runtime_sampler'],sampler_ideal_distance=r['sampler_ideal_distance']) for r in result['laws']],
        state_equivalence=deepcopy(result['state_equivalence']),
        source_reduction_status=reduction['status'],physical_transition_reward_parity=checks['physical_kernel_transition_reward_parity'],
        K0_received=False,K1_received=False,mean_S=None,mean_C_minus_S=None,mean_compiled_C=None,
        qualified_for_candidate=False,uncertainty=dict(systematic=None,covariance=None,total_CI=None),G=None)


def accept_structural_packet(config, packet):
    """Small future K0/K1 envelope. E[S] cannot enter MeanPriceLookup/cost.

    Values and check statuses are copied, not calculated or certified by B.
    A source-specific adapter may be needed when C publishes its actual format.
    """
    same = H.same
    B._validate(config)
    same(packet['configuration_id'],config['id'],'C_KERNEL_CONFIGURATION_MISMATCH')
    if packet['stage'] not in ('K0','K1') or not packet.get('source') or not packet.get('kernel_source'):
        raise ValueError('C_KERNEL_STAGE_AND_SOURCE_REQUIRED')
    same(packet['quantity'],STRUCTURAL,'E_S_IS_NOT_COMPILED_C')
    same(packet['qualified_for_candidate'],False,'C_STRUCTURE_IS_NOT_QUALIFIED_COMPILED_PRICE')
    same(packet['mean_compiled_C'],None,'E_C_MINUS_S_IS_MISSING')
    same(packet['mean_C_minus_S'],None,'THIS_ENVELOPE_IS_STRUCTURAL_ONLY')
    if packet['stage']=='K1' and packet['K0_status']!='PASS':
        raise ValueError('K1_REQUIRES_PUBLISHED_K0_PASS')
    keys = [(r['m'],r['axis']) for r in packet['rows']]
    if len(set(keys)) != len(keys) or any(m not in (0,6) or axis not in p.AXES for m,axis in keys):
        raise ValueError('C_KERNEL_FIXED_FOUR_CELL_DOMAIN_ONLY')
    for record in packet['rows']:
        row = config['rounds'][record['m']]
        for key,value in dict(identity=row['identity'],law_binding=row['law_binding'],
                physical_candidate_id=row['physical_candidate']['id'],profile=config['spec']['profile']).items():
            same(record[key],value,'C_KERNEL_IDENTITY_LAW_SOURCE_PROFILE_MISMATCH:' + key)
        if packet['stage']=='K0':
            same(record['mean_S'],None,'K0_DOES_NOT_EVALUATE_MEAN_S')
        for key in ('source','law_context','checks','uncertainty'):
            if key not in record:
                raise ValueError('C_STRUCTURAL_EVIDENCE_METADATA_REQUIRED:' + key)
        same(record['source'],packet['source'],'C_STRUCTURAL_NUMERICAL_SOURCE_MISMATCH')
    return dict(diagnostic=deepcopy(packet),compiled_price_records=[],mean_compiled_C=None,G=None,
        mean_S_by_cell={str(r['m'])+'/'+r['axis']:deepcopy(r['mean_S']) for r in packet['rows']},
        complete_four_cell_values=(set(keys)=={(m,axis) for m in (0,6) for axis in p.AXES}
            and all(r['mean_S'] is not None for r in packet['rows'])),scientific_price_qualification=False)


def validate_completion_packet(config, packet, previous, *, numerical_source, adapter_source, previous_source):
    """New ten-round scope, distinct from the immutable S0/m1 validators.

    The caller first accepts the published previous m1 packet. No nominal N,
    mean, normalization, numerical gate or error bound is calculated here.
    """
    same=H.same;B._validate(config)
    same(config['id'],S0.CONFIGURATION_ID,'COMPLETION_CANONICAL_CONFIGURATION_MISMATCH')
    for key,value in dict(configuration_id=config['id'],policy=config['spec']['rmse_policy'],
            source=numerical_source,adapter_source=adapter_source,previous_packet=previous_source,
            status='UNQUALIFIED',evidence_kind='SCIENTIFIC',technical_identity_verified=True,
            nominal_promoted_to_sufficient=False,B_previous_scope_validator_applicable=False).items():
        same(packet[key],value,'COMPLETION_PACKET_SOURCE_OR_QUALIFICATION_MISMATCH:'+key)
    new=[2,3,4,5,7,8,9,10,11,12]
    same(packet['diagnostic_scope'],dict(newly_evaluated_rounds=new,saved_mean_rounds=[3,4],
        new_mean_rounds=[2,5,7,8,9,10,11,12],preserved_rounds=[0,1,6],whole_schedule_RMSE_qualified=False),
        'COMPLETION_NEW_TEN_ROUND_SCOPE_REQUIRED')
    if any(k in packet for k in ('pilot_scope','B_S0_only_validator_applicable')):
        raise ValueError('COMPLETION_MUST_NOT_POSE_AS_S0_OR_M1_SCOPE')
    same(previous['configuration_id'],config['id'],'COMPLETION_PREVIOUS_CONFIGURATION_MISMATCH')
    same(packet['rmse_evidence'],dict(previous['rmse_evidence'],source=deepcopy(numerical_source)),
        'COMPLETION_GLOBAL_RMSE_PROMOTED')
    same(len(packet['rounds']),13,'COMPLETION_WHOLE_SCHEDULE_REQUIRED')
    for row,old,request in zip(packet['rounds'],previous['rounds'],config['rounds']):
        m=request['m']
        for key,value in dict(m=m,identity=request['identity'],law_binding=request['law_binding'],
                source_candidate_id=request['physical_candidate']['id'],source_request_id=old['source_request_id'],
                N_X=None,N_Y=None,model_status='UNQUALIFIED_MODEL_ASSUMPTIONS',
                diagnostic_source=numerical_source if m in new else old['diagnostic_source']).items():
            same(row[key],value,'COMPLETION_ROUND_IDENTITY_LAW_SOURCE_OR_QUALIFICATION:'+key)
        if m in (0,1,6):
            same(row,old,'COMPLETION_PREVIOUS_ROUND_CHANGED')
        for axis in p.AXES:
            if type(row['nominal_N_'+axis]) is not int or row['nominal_N_'+axis]<1:
                raise ValueError('COMPLETION_NOMINAL_POSITIVE_EXACT_INTEGER_REQUIRED')
    return True
