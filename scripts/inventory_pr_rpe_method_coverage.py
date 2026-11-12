"""Pinned saved metadata only. No schedule, normalization, N/C/G or optimizer calls."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from trottertracks.pf_pr_rpe_pipeline import SavedArtifacts, same

FIXED = dict(commit='ac3e1ce87b5ed7d9114abc316545366dd097209e',
             path='docs/research/rpe_provider_integration_20261111/fixed_coverage/COVERAGE.json')
R_COVERAGE = dict(commit='7dd61438b64a4b015ce0b7a7db81187d9867d8a2',
                  path='docs/research/rpe_c1_price_results_intake_20261111/COVERAGE.json')
C1 = dict(commit=R_COVERAGE['commit'], path='docs/research/rpe_c1_price_results_intake_20261111/INTAKE.json')
P1 = dict(commit='71ba49bb28b295aa28e51101bc3b1bf07d500767',
          path='docs/research/pr_r_detailed_n_p1_results_20261111/run01/N_HANDOFF.json')
A_MENU = 'f98982118324bcc8402ed513b5c720b37bb03e87'


def inventory(root=ROOT):
    """Existence/status inventory of old physical recipes, never new RMSE records."""
    saved = SavedArtifacts(root)
    fixed = saved.read(FIXED); old = saved.read(R_COVERAGE); c1 = saved.read(C1); p1 = saved.read(P1)
    p1_shots = {r['identity']['candidate']['id']: r for r in p1['shots']}
    rows = []; contexts = {}; recipes = {}
    for system in ('H4', 'H6'):
        menu_ref = dict(commit=A_MENU, path=f'docs/research/pr_r_screening_20261111/{system}_METADATA_PROPOSAL.json')
        menu = saved.read(menu_ref)['menu']; task = fixed['systems'][system]['task']
        contexts[system] = dict(binding=deepcopy(task['binding']), profile=deepcopy(task['profile']),
            configuration=deepcopy(task['configuration']), contract=deepcopy(task['contract']),
            identity_reference=menu_ref, N_reference=deepcopy(fixed[system + '_public_N_ref']))
        by_id = {}
        for row in menu['candidates'] + task['candidates']:
            key = row['candidate']['id']
            if key in by_id and not same(by_id[key]['candidate'], row['candidate']):
                raise ValueError('Saved candidate ID has conflicting physical recipes')
            by_id[key] = row
        for price in c1['prices']:
            ident = price['identity']; c = ident['candidate']
            if c['system'] == system:
                if c['id'] in by_id and not same(by_id[c['id']]['candidate'], c):
                    raise ValueError('C1 candidate differs from saved physical recipe')
                by_id.setdefault(c['id'], ident)
        fixed_rows = {r['candidate']['id']: r for r in fixed['systems'][system]['rows']}
        for row in old['systems'][system]['rows']:
            cid = row['candidate_id']; physical = by_id[cid]; c = physical['candidate']
            method = 'PR' if c['method'] == 'PR' else 'FULL_PF' if c['method'] == 'FULL' else 'TRUNCATED_PF'
            recipes[cid] = dict(candidate=deepcopy(c), m=row['m'], q_RPE=physical['q_RPE'],
                structure=deepcopy(physical['structure']), context=system, method=method,
                M=12, common_kappa=None, kappa_status='HISTORICAL_ROUND_R_NOT_COMMON_KAPPA',
                SOURCE_reference=deepcopy(R_COVERAGE),
                current_RMSE_identity_available=False)
            for axis in ('X', 'Y'):
                previous = fixed_rows.get(cid, {})
                extra_n = p1_shots.get(cid)
                if extra_n and not same(extra_n['identity']['candidate'], c):
                    raise ValueError('P1 N physical candidate differs')
                has_n = row['N_status'] == 'SAVED_CONDITIONAL' or (extra_n is not None and extra_n['plan'].get('N_' + axis) is not None)
                scenarios = [v for v in row['price_scenarios'] if v['axis'] == axis]
                records = [v for v in c1['prices'] if v['axis'] == axis and v['identity']['candidate']['id'] == cid]
                exists = any(v['value_available'] for v in scenarios)
                outdomain = previous.get('provider_domain_matches') is False
                rows.append(dict(system=system, method=method, candidate_id=cid, m=row['m'], axis=axis,
                    N=dict(availability='AVAILABLE' if has_n else 'MISSING',
                        RMSE_qualification='UNQUALIFIED' if has_n else 'NOT_EVALUATED',
                        saved_policy='HIGH_PROBABILITY_CONDITIONAL_NOT_RMSE',
                        source=previous.get('N_source') or (extra_n.get('source') if extra_n else None),
                        P1_reference=deepcopy(P1) if extra_n else None,
                        P1_policy=p1['comparison_policy'] if extra_n else None,
                        reference=deepcopy(R_COVERAGE)),
                    C=dict(availability='AVAILABLE' if exists else 'MISSING',
                        RMSE_qualification='UNQUALIFIED' if exists else 'OUTSIDE_DOMAIN' if outdomain else 'NOT_EVALUATED',
                        existing_provider_domain='OUTSIDE_DOMAIN' if outdomain else 'NOT_QUALIFIED_FOR_NEW_RMSE_SCHEDULE',
                        scenarios=deepcopy(scenarios), reference=deepcopy(R_COVERAGE),
                        C1_sources=[dict(source=deepcopy(v['source']), cohort=v['cohort'],
                            classification=v['classification'], expectation=v['expectation'],
                            model_qualified_for_candidate=(v.get('model') or {}).get('qualified_for_candidate'),
                            uncertainty=dict(reference=deepcopy(C1),
                                nominal_mean_price_covariance_available=(v.get('uncertainty') or {}).get('nominal_mean_price_covariance') is not None,
                                paired_group=(v.get('uncertainty') or {}).get('paired_group'),
                                calibration_data_group=(v.get('uncertainty') or {}).get('calibration_data_group'),
                                joint_covariance=None, R_q_systematic=None, total_CI=None)) for v in records],
                        all_law_expected_price_qualified=False),
                    normalization=deepcopy(row['normalization']),
                    error_contract='NO_GLOBAL_RMSE_EVIDENCE',
                    G_status='NOT_EVALUATED', G=None, eligible_for_RMSE_comparison=False))
    # Missing routes remain route-level entries: no invented physical candidate,
    # event schedule, retained Hamiltonian, kappa, N or price is manufactured.
    missing_routes = [dict(system=system, method=method, candidate_id=None,
        status='NOT_EVALUATED', N='MISSING', C='MISSING', G=None,
        reason=reason) for system, method, reason in (
            ('H4', 'FULL_PF', 'FULL_H_RMSE_BASELINE_NOT_SUPPLIED'),
            ('H6', 'TRUNCATED_PF', 'RETAINED_H_AND_TRUNCATION_ERROR_CONTRACT_NOT_SELECTED'),
            ('H4', 'FULL_H_RTE', 'HISTORICAL_FIXED_T_DF_SOURCE_EXISTS_BUT_NO_RMSE_SCHEDULE_PROVIDER'),
            ('H6', 'FULL_H_RTE', 'HISTORICAL_FIXED_T_DF_SOURCE_EXISTS_BUT_NO_RMSE_SCHEDULE_PROVIDER'),
            ('PAULI_H4_H6', 'FULL_PF', 'HISTORICAL_PF_SOURCE_EXISTS_BUT_NO_RMSE_SCHEDULE_PROVIDER'),
            ('PAULI_H4_H6', 'TRUNCATED_PF', 'HISTORICAL_B0_SOURCE_EXISTS_BUT_NO_RMSE_TRUNCATION_CONTRACT'),
            ('PAULI_H4_H6', 'PR', 'HISTORICAL_SIGNAL_POLICY_IS_NOT_ALL_ROUND_ENERGY_RMSE'),
            ('PAULI_H4_H6', 'FULL_H_RTE', 'HISTORICAL_SIGNAL_POLICY_IS_NOT_ALL_ROUND_ENERGY_RMSE'))]
    return dict(status='SAVED_METADATA_ONLY_NOT_NEW_RMSE_EVIDENCE', contexts=contexts,
        recipes=recipes, rows=rows, missing_routes=missing_routes, sources=saved.sources,
        total_resources=None, winner=None, covariance=None, total_CI=None,
        saved_signal_N_C_G_values_copied=False, new_N_C_G_or_schedule_calls=0,
        model_qualified_for_candidate=False, S0a_is_mean_C=False, local_pilot_is_global_RMSE=False,
        full_RPE_qualified=False, mandatory_stop=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = inventory()
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
