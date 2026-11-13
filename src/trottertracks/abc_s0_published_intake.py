"""Accept published A/C metadata against B's stored S0 configuration.

Never calls S0 canonical/prepare, A.connect, C.prepare, a provider or cost.
Existing identity validators and the evidence planner remain the authority.
"""
from pathlib import Path
from fractions import Fraction

from . import h4_s0_v2_intake as H
from . import h4_s0_price_plan as C
from . import pr_rpe_rmse_evidence as E

B, p = H.B, H.p
B_RELEASE = '030f3f548a35d98a88e62e92a35f132e0243a293'
A_RELEASE = '75b5c89bddf8a1ffa4a8ac99b2c06d6e05fa012d'
C_RELEASE = 'd458b5e4f596d628d923263f3b2f1678960f2ffa'
CONFIGURATION_ID = '587d20f7ed906d25841615dc9746674af872675b895bd438c06d3e74abf0e911'
BDIR = 'docs/research/h4_s0_v2_integration_20261113/run01/'
ADIR = 'docs/research/pr_rpe_s0_v2_connection_20261113/metadata01/'
CPLAN = dict(commit=C_RELEASE, path='docs/research/h4_s0_mean_price_plan_20261113/PRICE_REQUESTS.json')
A_ADAPTER_SOURCE = dict(module='src/trottertracks/pr_rpe_rmse_s0_v2.py',
    B='af284d853890155bacd19e61850c196bd1d5c660', S0=H.A_COMMIT, S0_science=H.SCIENCE['commit'])


def accept(config, original, packet, a_config, c_plan, b_requests, identity_map):
    """Exact metadata checks only; nominal diagnostics never become sufficient N."""
    same = H.same
    same(config['id'], CONFIGURATION_ID, 'FIXED_B_CONFIGURATION_REQUIRED')
    same(config, a_config, 'A_PUBLISHED_CONFIGURATION_DIFFERS')
    H.validate_A_v2_packet(config, packet, original)
    same(packet['schema'], B.SCHEMA, 'A_SCHEMA_MISMATCH')
    same(packet['adapter_source'], A_ADAPTER_SOURCE, 'A_ADAPTER_SOURCE_MISMATCH')
    for key in ('A_original_packet', 'nominal_promoted_to_sufficient', 'technical_identity_verified'):
        same(packet[key], original[key], 'A_ORIGINAL_EVIDENCE_OR_DIAGNOSTIC_CHANGED')
    links = [dict(m=r['m'], source_candidate_id=r['physical_candidate']['id'],
        legacy_B_candidate_id=r['V1_candidate_id'], canonical_B_candidate_id=r['V2_candidate_id'])
        for r in identity_map['rows']]
    same(packet['connection_provenance'], dict(legacy_B_configuration_id=identity_map['V1_configuration_id'],
        canonical_B_configuration_id=config['id'], original_packet=H.A_PACKET,
        physical_links=links, nominal_recalculated=False, legacy_relabelled=False), 'A_CONNECTION_SOURCE_MISMATCH')

    c_verified = C.verify_B_publication(c_plan, config, integration=H.I)
    args = config['spec']['A_input']; design = args['design']
    for key, value in dict(binding=design['binding'], profile=design['profile'], structural_contract=design['contract'],
            M=12, h_exact=args['design']['base_time'], kappa_paper_exact=args['kappa_paper'],
            normalization_rule=args['policy']['normalization_rule'], A_source=H.A_CONFIG).items():
        # C publishes h as an exact fraction string, unlike the saved design float.
        if key == 'h_exact':
            same(B.a.exact_value(Fraction(c_plan[key])), config['spec']['h'], 'C_H_EXACT_MISMATCH')
        else:
            same(c_plan[key], value, 'C_SOURCE_OR_CONTEXT_MISMATCH:' + key)
    same(c_plan['saved_B_id'], identity_map['V1_configuration_id'], 'C_LEGACY_CONFIGURATION_MISMATCH')
    same(c_plan['saved_B_schema'], 'PR_RPE_RMSE_V1', 'C_LEGACY_SCHEMA_MISMATCH')
    for key, value in dict(price_values_copied=False, new_scientific_calls=0,
            mean_compiled_price=None, full_rpe_resources=None).items():
        same(c_plan[key], value, 'C_METADATA_ONLY_STATE_CHANGED:' + key)
    same([r['m'] for r in c_plan['focus_requests']], [0, 6], 'C_FOCUS_CHANGED')
    for request in c_plan['focus_requests']:
        row = config['rounds'][request['m']]
        same(request['source_law_declaration'], row['round_law_definition']['A_declaration'], 'C_SOURCE_LAW_MISMATCH')
        same(request['A_source'], H.A_CONFIG, 'C_A_SOURCE_MISMATCH')
        same(request['A_request_id'], packet['rounds'][request['m']]['source_request_id'], 'C_A_REQUEST_MISMATCH')
        same([request['price'], request['cohort'], request['model_qualified_for_candidate']],
             [None, None, False], 'C_UNQUALIFIED_PRICE_CHANGED')
        same(request['uncertainty'], dict(systematic=None, joint_covariance=None, total_CI=None), 'C_UNKNOWN_UNCERTAINTY_CHANGED')
        same(request['quantity'], 'FULL_WRAPPER_MEAN_COMPILED_PRICE', 'C_QUANTITY_MISMATCH')
    same([r['m'] for r in c_plan['round_price_coverage']], list(range(13)), 'C_ROUND_COVERAGE_MISMATCH')
    for saved, row in zip(c_plan['round_price_coverage'], config['rounds']):
        physical = row['physical_candidate']
        for key, value in dict(physical_candidate_id=physical['id'], B_candidate_id=row['candidate']['id'],
                T=physical['T'], q_PF=row['q_PF'], R_outer=row['R_outer'], event_count=row['structure']['event_count'],
                price=None, qualified_mean_compiled_price=False,
                source=dict(commit=H.C_COMMIT, path=C.PRICE_INVENTORY)).items():
            same(saved[key], value, 'C_ALL_ROUND_METADATA_MISMATCH:' + key)

    same([(r['m'], r['axis']) for r in b_requests], [(m, ax) for m in range(13) for ax in p.AXES], 'B_REQUEST_AXES_MISMATCH')
    for request in b_requests:
        row = config['rounds'][request['m']]
        for key, value in dict(configuration_id=config['id'], identity=row['identity'], law_binding=row['law_binding'],
                physical_candidate=row['physical_candidate'], source_candidate_id=row['physical_candidate']['id'],
                T=row['T'], actual_delta_pf=row['actual_delta_pf'], q_PF=row['q_PF'], R_outer=row['R_outer'],
                event_count=row['structure']['event_count'], signed_weights=row['structure']['signed_weights'],
                allocation=row['structure']['allocation'], source=H.A_CONFIG).items():
            same(request[key], value, 'B_FORMAL_C_REQUEST_MISMATCH:' + key)
        measurement = p.measurement(row, request['axis'])
        same(request['measurement'], {k:measurement[k] for k in
            ('measurement_axis','observable','outcome_convention','units')}, 'B_C_MEASUREMENT_MISMATCH')
        same(request['requested_quantity'], 'FULL_WRAPPER_MEAN_COMPILED_PRICE', 'B_C_QUANTITY_MISMATCH')
        for key in ('actual_price_law', 'law_distance_qualification', 'requested_mean',
                    'shared_calibration_covariance', 'joint_covariance', 'total_CI'):
            same(request[key], None, 'B_MISSING_C_CHANGED:' + key)
    # No mean-price lookup or real cost call: the existing planner sees missing C.
    plan = E.plan([dict(configuration=config, N_packet=packet)], price_policy='qualified_expectation',
        unexplored_scope='PUBLISHED_FIXED_S0_PR_ONLY_NO_NEW_MENU')
    return dict(configuration_id=config['id'], A_packet_accepted=True, C_publication=c_verified,
        A_packet_source=dict(commit=A_RELEASE, path=ADIR+'V2_N_HANDOFF.json'), C_plan_source=CPLAN,
        nominal_rounds=[0, 6], nominal_N_XY={'0': [64,64], '6': [38,38]}, unevaluated_rounds=[m for m in range(13) if m not in (0,6)],
        sufficient_N_available=False, global_RMSE_bound=None, model_qualification='UNQUALIFIED',
        C_formal_axis_requests=26, C_focus_rounds=[0,6], C_round_metadata_matches=13,
        qualified_compiled_price_axes=0, G=None, winner=None, planner=plan, scientific_calls=0, mandatory_stop=True)


def load_published(root=None):
    """Read pinned, published JSON only. No S0 artifact regeneration."""
    store = p.SavedArtifacts(root or Path(__file__).resolve().parents[2])
    def read(commit, path):
        return store.read(dict(commit=commit, path=path))
    values = dict(config=read(B_RELEASE, BDIR+'CONFIGURATION_V2.json'),
        original=store.read(H.A_PACKET), packet=read(A_RELEASE, ADIR+'V2_N_HANDOFF.json'),
        a_config=read(A_RELEASE, ADIR+'V2_CONFIGURATION.json'), c_plan=store.read(CPLAN),
        b_requests=read(B_RELEASE, BDIR+'C_REQUESTS.json'), identity_map=read(B_RELEASE, BDIR+'IDENTITY_MAP.json'))
    return values, store.sources


def published_result(root=None):
    values, sources = load_published(root)
    result = accept(**values)
    result['source_files'] = sources
    return result
