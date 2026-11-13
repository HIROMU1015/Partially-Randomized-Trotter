"""Fixed H4 S0 metadata -> canonical V2 requests and evidence gaps.

No A V2 N packet is created: the V1 packet remains original evidence and its
nominal values are exposed only in a separate diagnostic view. No N/C/G leaf.
"""
from copy import deepcopy
import hashlib
from pathlib import Path
import subprocess

from . import pr_rpe_rmse_integration as I
from . import pr_rpe_rmse_evidence as E
from .bv2_mean_price import V2MeanPriceLookup, QUANTITY

B = I.B
p = I.p
A_COMMIT = '7da8a35f68e9fdc2b46c361dd2c68e2a799b1e75'
C_COMMIT = '1db4d6e7ebce872c069e337b76632046f60d8b56'
A_DIR = 'docs/research/pr_rpe_rmse_s0_results_20261113/run01/'
A_CONFIG = dict(commit=A_COMMIT, path=A_DIR + 'CONFIGURATION.json')
A_PACKET = dict(commit=A_COMMIT, path=A_DIR + 'N_HANDOFF.json')
A_INDEX = dict(commit=A_COMMIT, path=A_DIR + 'SOURCE_INDEX.json')
B_HISTORY = dict(commit='6a9d7fd36b687e58aecbed1278c6d7927d6827d6',
    path='docs/research/pr_rpe_method_coverage_20261113/COVERAGE.json')
C_HISTORY = dict(commit=C_COMMIT,
    path='docs/research/bv2_mean_price_validation_design_20261113/PRICE_COVERAGE.json')
SCIENCE = dict(commit='7c3e1e5db0c55dd8249e9030a4c7e9954496fa7e',
    path='scripts/run_pr_rpe_rmse_s0.py', plan='d97a40ab719bd71fee4ad41d5acc2abe3c89904d')


def same(left, right, label):
    if not p.same(left, right):
        raise ValueError(label)


def canonical(saved, original):
    """Fresh generation using S0 inputs; reject every unaccounted difference."""
    same(saved['source'], SCIENCE, 'S0_SOURCE_MISMATCH')
    args = saved['inputs']; old = saved['B_configuration']
    same(args, old['spec']['A_input'], 'S0_INPUT_OR_LEGACY_SPEC_MISMATCH')
    same(args['design'], saved['A_schedule']['design'], 'S0_A_DESIGN_MISMATCH')
    same(args['kappa_paper'], saved['conditions']['kappa_paper_exact'], 'S0_EXACT_KAPPA_MISMATCH')
    same(dict(M=args['M'], system=args['design']['system'], formula=args['design']['formula'],
        prefix=args['design']['prefix'], K=args['design']['K'], delta=args['design']['delta'],
        h=args['design']['base_time'], PF=args['design']['PF']),
        dict(M=12, system='H4', formula='2nd', prefix=1, K=2, delta=.1, h=.1,
             PF=dict(method='B0', prefix=7)), 'OUTSIDE_FIXED_H4_S0_SCOPE')
    same(args['policy']['source'], SCIENCE, 'S0_POLICY_SOURCE_MISMATCH')
    same(old['schema'], 'PR_RPE_RMSE_V1', 'S0_LEGACY_SCHEMA_MISMATCH')
    same(original['source'], SCIENCE, 'S0_N_SOURCE_MISMATCH')
    same(original['configuration_id'], old['id'], 'S0_N_CONFIGURATION_MISMATCH')
    same(original['policy'], old['spec']['rmse_policy'], 'S0_N_POLICY_MISMATCH')
    same((original['status'], original['evidence_kind']), ('UNQUALIFIED', 'SCIENTIFIC'), 'S0_N_QUALIFICATION_CHANGED')
    same(original['pilot_scope']['rounds'], [0, 6], 'S0_PILOT_SCOPE_CHANGED')
    same(original['pilot_scope']['whole_schedule_RMSE_qualified'], False, 'S0_PILOT_QUALIFICATION_CHANGED')
    same(original['nominal_promoted_to_sufficient'], False, 'S0_N_PROMOTED')
    same(original['rmse_evidence']['source'], SCIENCE, 'S0_RMSE_SOURCE_MISMATCH')
    same(original['rmse_evidence']['error_contract'], original['policy']['error_contract'], 'S0_RMSE_CONTRACT_MISMATCH')
    if (original['rmse_evidence']['rmse_bound_ha'] is not None or
            original['rmse_evidence']['covers_state_target_alias_prior_decoder_all_rounds'] is not False or
            any(original['rmse_evidence'][k] != 'UNQUALIFIED' for k in
                ('status', 'finite_K_status', 'signed_piece_kappa_status'))):
        raise ValueError('S0_GLOBAL_RMSE_QUALIFICATION_CHANGED')
    # The existing _assemble hashes these pre-identity fields in V1 and V2.
    unassembled = [{k:v for k,v in row.items() if k not in ('candidate','identity','law_binding')}
                   for row in old['rounds']]
    same(old['id'], B._id(dict(schema=old['schema'], spec=old['spec'], rounds=unassembled)), 'S0_V1_ID_MISMATCH')
    config = B.generate(I.a_spec(args, 'PR'))
    same(I.A.schedule(**args), saved['A_schedule'], 'S0_A_WHOLE_SCHEDULE_MISMATCH')
    expected_spec = deepcopy(old['spec'])
    same(expected_spec['split']['deterministic_scope'], 'A_SOURCE_BOUND_PF_BASELINE', 'S0_OLD_SPLIT_MISMATCH')
    expected_spec['split']['deterministic_scope'] = 'PREFIX'
    same(expected_spec, config['spec'], 'UNEXPECTED_V1_V2_SPEC_DIFFERENCE')
    same(len(original['rounds']), 13, 'S0_N_ROUND_COUNT')
    old_task = dict(binding=old['spec']['binding'], profile=old['spec']['profile'],
        configuration=dict(schema=old['schema'], configuration_id=old['id'], spec=old['spec']),
        contract=dict(split=old['spec']['split'], rmse_policy=old['spec']['rmse_policy'], law=old['spec']['law']))
    A_requests = {r['identity']['m']:r for r in saved['A_schedule']['rows'] if p.arm(r['identity']['candidate'])=='PR'}
    rows = []
    for before, after, n in zip(old['rounds'], config['rounds'], original['rounds']):
        m = after['m']
        same({k:v for k,v in before.items() if k not in ('candidate','identity','law_binding')},
             {k:v for k,v in after.items() if k not in ('candidate','identity','law_binding')}, 'S0_PHYSICAL_SCHEDULE_MISMATCH')
        same({k:v for k,v in before['candidate'].items() if k != 'id'},
             {k:v for k,v in after['candidate'].items() if k != 'id'}, 'S0_CANDIDATE_FIELDS_MISMATCH')
        same(before['candidate']['id'], 'RMSE:' + B._id([old['id'], m]), 'S0_V1_CANDIDATE_ID_MISMATCH')
        same(n['m'], m, 'S0_N_ROUND_ORDER_MISMATCH')
        same(before['identity'], p.identity(old_task, before), 'S0_V1_FULL_IDENTITY_MISMATCH')
        same(n['identity'], before['identity'], 'S0_N_IDENTITY_MISMATCH')
        same(n['law_binding'], before['law_binding'], 'S0_N_LAW_MISMATCH')
        expected_law = dict(after['law_binding'], configuration_id=old['id'], candidate_id=before['candidate']['id'])
        same(before['law_binding'], expected_law, 'S0_V1_LAW_ID_MISMATCH')
        same(n['source_request_id'], A_requests[m]['request_id'], 'S0_A_REQUEST_ID_MISMATCH')
        for key in ('definition', 'kappa', 'event_mapping'):
            same(before['law_binding'][key], after['law_binding'][key], 'S0_LAW_OR_SOURCE_MISMATCH')
        same([n['N_X'], n['N_Y']], [None, None], 'S0_SUFFICIENT_N_NOT_NULL')
        nominal = 64 if m == 0 else 38 if m == 6 else None
        same([n['nominal_N_X'], n['nominal_N_Y']], [nominal, nominal], 'S0_NOMINAL_DIAGNOSTIC_CHANGED')
        same(n['model_status'], 'UNQUALIFIED_MODEL_ASSUMPTIONS' if m in (0,6) else 'NOT_EVALUATED', 'S0_MODEL_STATUS_CHANGED')
        rows.append(dict(m=m, physical_candidate=deepcopy(after['physical_candidate']),
            V1_candidate_id=before['candidate']['id'], V2_candidate_id=after['candidate']['id'],
            source_request_id=n['source_request_id'], q_PF=after['q_PF'], R_outer=after['R_outer'],
            event_count=after['structure']['event_count'], physical_schedule_match=True,
            S0_diagnostic=dict(evaluated=m in (0,6), nominal_N_X=n['nominal_N_X'], nominal_N_Y=n['nominal_N_Y'],
                model_status=n['model_status'], source=deepcopy(original['source']),
                original_handoff=deepcopy(A_PACKET), original_configuration_id=old['id'],
                sufficient_N_available=False, global_RMSE_bound=None)))
    return config, dict(source=deepcopy(A_CONFIG), original_handoff=deepcopy(A_PACKET),
        V1_configuration_id=old['id'], V2_configuration_id=config['id'],
        spec_change=dict(field='split.deterministic_scope', before='A_SOURCE_BOUND_PF_BASELINE', after='PREFIX'),
        A_metadata_schedule_match=True, scientific_qualification='UNQUALIFIED', rows=rows)


def validate_A_v2_packet(config, packet, original):
    """Validate a future A-produced diagnostic packet; never create/relabel one."""
    B._validate(config)
    expected = deepcopy(original['rounds'][0]['identity']['configuration']['spec'])
    same(expected['split']['deterministic_scope'], 'A_SOURCE_BOUND_PF_BASELINE', 'A_V2_ORIGINAL_SPEC_MISMATCH')
    expected['split']['deterministic_scope'] = 'PREFIX'
    same(config['spec'], expected, 'A_V2_S0_SPEC_MISMATCH')
    same(packet['configuration_id'], config['id'], 'A_V2_CONFIGURATION_REQUIRED')
    same(packet['policy'], config['spec']['rmse_policy'], 'A_V2_POLICY_MISMATCH')
    for key in ('source','status','evidence_kind','rmse_evidence','pilot_scope'):
        same(packet[key], original[key], 'A_V2_S0_EVIDENCE_OR_SOURCE_CHANGED')
    same(len(packet['rounds']), len(config['rounds']), 'A_V2_ROUND_COUNT')
    for request, row, previous in zip(config['rounds'], packet['rounds'], original['rounds']):
        same(row['identity'], request['identity'], 'A_V2_FULL_IDENTITY_MISMATCH')
        same(row['law_binding'], request['law_binding'], 'A_V2_LAW_MISMATCH')
        for key in ('m','N_X','N_Y','nominal_N_X','nominal_N_Y','model_status','source_request_id'):
            same(row[key], previous[key], 'A_V2_S0_DIAGNOSTIC_CHANGED')
        if 'source_candidate_id' in row:
            same(row['source_candidate_id'], request['physical_candidate']['id'], 'A_V2_SOURCE_CANDIDATE_MISMATCH')
    return True


def historical(config, design, previous, prices):
    """Physical metadata comparison only; no old N/C is rebound to V2."""
    for archive in (previous, prices):
        same(archive['contexts']['H4']['binding'], design['binding'], 'HISTORICAL_H_STATE_SOURCE_MISMATCH')
        same(archive['contexts']['H4']['profile'], design['profile'], 'HISTORICAL_COMPILER_PROFILE_MISMATCH')
        same(archive['contexts']['H4']['contract'], design['contract'], 'HISTORICAL_HAMILTONIAN_CONTRACT_MISMATCH')
    out = []
    for row in config['rounds']:
        physical = row['physical_candidate']; cid = physical['id']
        recipe = prices['recipes'].get(cid)
        if recipe is not None:
            same(recipe['candidate'], physical, 'HISTORICAL_PHYSICAL_ID_COLLISION')
            same(recipe['structure'], row['structure'], 'HISTORICAL_SIGNED_STRUCTURE_MISMATCH')
        near = [dict(cell_id=v['cell_id'], candidate=deepcopy(v['identity']['candidate']),
            role=v['role'], cohort=v['cohort'], n_paths=v['n_paths'],
            model_qualified_for_candidate=v['model_qualified_for_candidate'],
            expectation='FINITE_COHORT_ONLY', source=deepcopy(prices['C1_source']),
            physical_match=p.same(v['identity']['candidate'], physical),
            differing_fields=[k for k in physical if not p.same(physical[k], v['identity']['candidate'].get(k))])
            for v in prices['C1_cells'] if v['identity']['candidate']['system']=='H4' and v['identity']['q_PF']==row['q_PF']]
        for axis in p.AXES:
            old_record = next((v for v in previous['rows'] if v['candidate_id']==cid and v['axis']==axis), None)
            old_n = None if old_record is None else old_record['N']
            old_c = next((v for v in prices['rows'] if v['candidate_id']==cid and v['axis']==axis), None)
            out.append(dict(m=row['m'], axis=axis, physical_candidate_id=cid,
                historical_physical_recipe_present=recipe is not None,
                saved_HP_N=deepcopy(old_n), saved_normalization=None if old_record is None else deepcopy(old_record['normalization']),
                saved_C=deepcopy(old_c), C1_same_q_cells=deepcopy(near),
                source=dict(N=deepcopy(B_HISTORY), C=deepcopy(C_HISTORY)),
                full_V2_configuration_match=False, law_SOURCE_equivalence='NOT_ESTABLISHED',
                reusable_as_V2_sufficient_N=False, reusable_as_V2_price=False))
    return out


def prepare(root):
    """Pinned Git records + existing A schedule/C passive lookup/planner only."""
    store = p.SavedArtifacts(root)
    saved = store.read(A_CONFIG); original = store.read(A_PACKET); index = store.read(A_INDEX)
    path = 'src/trottertracks/pr_rpe_rmse.py'
    same(hashlib.sha256((Path(root)/path).read_bytes()).hexdigest(), index['consumed_files'][path], 'S0_A_METADATA_SOURCE_CHANGED')
    # Validate historical provenance against the consumed Git blob. The live
    # shared pipeline may evolve; canonical() below checks every H4 schedule,
    # identity and law field through the current API instead of freezing it.
    path = 'src/trottertracks/pf_pr_rpe_pipeline.py'
    before = subprocess.check_output(['git','show',SCIENCE['commit']+':'+path],cwd=root)
    same(hashlib.sha256(before).hexdigest(), index['consumed_files'][path], 'S0_PIPELINE_PROVENANCE_CHANGED')
    cpath = 'src/trottertracks/bv2_mean_price.py'
    same((Path(root)/cpath).read_bytes().hex(), subprocess.check_output(['git','show',C_COMMIT+':'+cpath],cwd=root).hex(), 'C_PROVIDER_SOURCE_CHANGED')
    config, mapping = canonical(saved, original)
    history = historical(config, saved['inputs']['design'], store.read(B_HISTORY), store.read(C_HISTORY))
    quotes = V2MeanPriceLookup(config, integration=I)()
    same(len(quotes), 26, 'C_PASSIVE_LOOKUP_AXIS_COVERAGE_MISMATCH')
    if any(v['mean']['rz_count'] is not None for v in quotes):
        raise ValueError('UNSUPPLIED_C_VALUE')
    # Missing provider placeholders are not acquired N/C records. A owns the
    # next V2 packet; the original V1 packet is a diagnostic reference only.
    entry = dict(configuration=config, prices=[], diagnostic_references=[dict(source=deepcopy(A_PACKET),
        original_configuration_id=original['configuration_id'], quantity='S0_LOCAL_NOMINAL_N_NOT_SUFFICIENT')])
    plan = E.plan([entry], price_policy='qualified_expectation')
    requests = []
    for cell, diagnostic, archived, quote in zip(plan['configurations'][0]['rows'],
            [v for r in mapping['rows'] for v in (r['S0_diagnostic'],r['S0_diagnostic'])], history, quotes):
        same((quote['axis'],quote['identity']), (cell['axis'],cell['identity']), 'C_PASSIVE_LOOKUP_IDENTITY_MISMATCH')
        same((archived['m'],archived['axis']), (cell['m'],cell['axis']), 'HISTORICAL_AXIS_COVERAGE_MISMATCH')
        cell['S0_diagnostic'] = deepcopy(diagnostic); cell['saved_evidence'] = deepcopy(archived)
        cell['S0_error_contract_qualification'] = {k:deepcopy(original['rmse_evidence'][k]) for k in
            ('source','status','rmse_bound_ha','covers_state_target_alias_prior_decoder_all_rounds','finite_K_status','signed_piece_kappa_status')}
        cell['C_passive_lookup'] = dict(status=quote['price_status'], reason=quote['reason'],
            domain={k:deepcopy(v) for k,v in quote['mean_price_scope'].items() if k!='identity'},
            source=deepcopy(quote['source']), value_acquired=False)
        row = config['rounds'][cell['m']]; measurement = p.measurement(row, cell['axis'])
        measurement.pop('law_context')
        requests.append(dict(configuration_id=config['id'], m=cell['m'], axis=cell['axis'],
            physical_candidate=deepcopy(row['physical_candidate']), source_candidate_id=row['physical_candidate']['id'],
            T=deepcopy(row['T']), actual_delta_pf=deepcopy(row['actual_delta_pf']), q_PF=row['q_PF'],
            R_outer=row['R_outer'], event_count=row['structure']['event_count'],
            signed_weights=deepcopy(row['structure']['signed_weights']), allocation=deepcopy(row['structure']['allocation']),
            identity=deepcopy(row['identity']), law_binding=deepcopy(row['law_binding']),
            measurement=measurement, requested_quantity=QUANTITY,
            actual_price_law=None, law_distance_qualification=None,
            requested_mean=None, shared_calibration_covariance=None, joint_covariance=None, total_CI=None,
            source=deepcopy(A_CONFIG)))
    plan.update(S0_fixed_design_source=deepcopy(A_CONFIG), S0_diagnostic_source=deepcopy(A_PACKET),
                new_A_V2_packet_created=False, scientific_model_qualification=original['rmse_evidence']['status'],
                global_RMSE_bound=None, price_policy_adoption=None)
    return dict(configuration=config, identity_map=mapping, evidence_plan=plan,
        price_requests=requests, original_S0_handoff=original, sources=store.sources,
        C_implementation=dict(commit=C_COMMIT,path=cpath), original_S0_status_preserved=True,
        metadata_source_compatibility=dict(A='BYTE_IDENTICAL_TO_S0',
            pipeline='PINNED_S0_BLOB_VERIFIED_CURRENT_API_H4_METADATA_MATCH'),
        science_calls=0, actual_N_C_G_evaluated=False, mandatory_stop=True)
