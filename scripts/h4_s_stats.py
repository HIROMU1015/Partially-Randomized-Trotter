"""One authorized pass over saved H4 PR summary statistics; stdlib only."""
import argparse
from fractions import Fraction
import hashlib
import math
from pathlib import Path
from statistics import NormalDist
import subprocess
import time

import h4_dp_contract as c

ROOT = Path(__file__).resolve().parents[1]
FIXED_RESULT = '4e8ad1936363266d1b55bb689c36846e6c3dfcdb'
F_VALUES = (1, 2, 4)
ETA_VALUES = (.02, .01)
FORBIDDEN = ('raw_trajectory_aggregation', 'bootstrap', 'sampling', 'compile',
             'candidate_selection_or_reranking', 'reranking')


def partner(cell, cells):
    matches = [other['id'] for other in cells if other['K'] == 2 and all(
        other[k] == cell[k] for k in ('prefix', 'q', 'r', 'R', 'method', 'formula'))]
    c.require(cell['K'] == 4 and len(matches) == 1, 'FIXED_K2_PARTNER')
    return matches[0]


def call_budget(cells, n_fixed, endpoints):
    total = sum(n_fixed.values())
    return dict(trajectory=total, native_evolution=total, wrapper_compile=2*total,
        occurrence=sum(x['q']*n_fixed[x['id']] for x in cells),
        RTE_event=sum(x['R']*n_fixed[x['id']] for x in cells),
        control_probe_maximum=4*len(cells), independent_event_oracle_maximum=len(cells),
        endpoint_accounting_pass=1, endpoint_comparisons=endpoints,
        seed_slots=sum((1 if x['q'] == 1 else 1+x['q'])*n_fixed[x['id']] for x in cells),
        new_PF_compile=0, SCF=0, DF=0, state=0, reference=0, signal=0, shot_recalculation=0, fit=0)


def analyze(registration, pf_by_epsilon):
    """Caller must authorize real data first. Tests use artificial summaries."""
    cells = registration['candidates']
    by_id = {x['id']: x for x in cells}
    assignments = registration['assignments']
    keys = [(x['cell_id'], x['epsilon']) for x in assignments]
    stats = {(x['cell_id'], x['epsilon']): x for x in registration['saved_statistics']}
    c.require(len(keys) == len(set(keys)) == len(stats) and set(keys) == set(stats), 'FIXED_SUMMARY_COVERAGE')
    c.require(len(by_id) == len(cells) and set(by_id) == {x[0] for x in keys}, 'FIXED_CANDIDATE_COVERAGE')
    c.require(all(x['R'] == x['q']*x['r'] for x in cells), 'ORIGINAL_R_EVENT_COUNT')
    z_family, z_power = NormalDist().inv_cdf(1-.05/10), NormalDist().inv_cdf(.8)
    scenarios, directions = [], []
    for assignment in assignments:
        cid, eps = assignment['cell_id'], assignment['epsilon']
        saved, pf = stats[cid, eps], pf_by_epsilon[str(eps)]['best']
        c.require(all(saved[k] == assignment[k] for k in assignment), 'SAVED_ASSIGNMENT_BINDING')
        n0, se = int(saved['saved_n']), float(saved['saved_paired_SE'])
        c.require(n0 >= 2 and math.isfinite(se) and se > 0, 'POSITIVE_SAVED_PAIRED_SE')
        cost = int(pf['G_exact'])
        c.require(cost > 0, 'POSITIVE_FIXED_PF_COST')
        delta = Fraction(cost)-Fraction(saved['saved_mean_exact'])
        direction = 1 if delta > 0 else -1 if delta < 0 else 0
        directions.append(dict(assignment, saved_mean_exact=str(saved['saved_mean_exact']),
            saved_paired_SE=se, saved_n=n0, PR_saved_shots=saved['saved_shots'],
            PF_cell_id=pf['cell_id'], PF_G_exact=str(cost), PF_saved_shots=pf['shots'],
            PF_minus_saved_PR_exact=str(delta), expected_direction=direction,
            baseline='PR_CHEAPER' if direction > 0 else 'PF_CHEAPER' if direction < 0 else 'EXACT_TIE',
            population_mean_certified=False))
        sd = math.sqrt(n0)*se
        for f in F_VALUES:
            for eta in ETA_VALUES:
                n = c.normal_n_formula(f*sd, eta*cost, z_family, z_power)
                scenarios.append(dict(cell_id=cid, epsilon=eps, role=assignment['role'],
                    saved_n=n0, saved_paired_SE=se, s_plan_from_saved_SE=sd,
                    PF_cell_id=pf['cell_id'], PF_G_exact=str(cost), f=f, eta=eta, n=n))
    sensitivity = []
    for f in F_VALUES:
        for eta in ETA_VALUES:
            own = {cid: max(x['n'] for x in scenarios if x['cell_id'] == cid
                           and x['f'] == f and x['eta'] == eta) for cid in by_id}
            n_fixed = dict(own)
            recommendations = []
            for cid, cell in by_id.items():
                roles = {x['role'] for x in assignments if x['cell_id'] == cid}
                c.require(len(roles) == 1, 'SINGLE_REGISTERED_ROLE_PER_CANDIDATE')
                diag = roles == {'BOUNDARY_K_PARTNER_DIAGNOSTIC'}
                paired = partner(cell, cells) if diag else None
                if diag:
                    c.require({x['role'] for x in assignments if x['cell_id'] == paired} == {'PRIMARY'},
                              'DIAGNOSTIC_PARTNER_IS_PRIMARY')
                    n_fixed[cid] = max(own[cid], own[paired])
                else:
                    c.require(roles == {'PRIMARY'}, 'REGISTERED_ROLE')
                recommendations.append(dict(cell=cell, role=next(iter(roles)), own_endpoint_max_n=own[cid],
                    K2_partner=paired, K2_partner_n=None if paired is None else own[paired],
                    recommended_n=n_fixed[cid], limiting_own_epsilons=[x['epsilon'] for x in scenarios
                        if x['cell_id'] == cid and x['f'] == f and x['eta'] == eta and x['n'] == own[cid]]))
            sensitivity.append(dict(f=f, eta=eta, n_fixed=n_fixed, candidates=recommendations,
                proposed_P_calls=call_budget(cells, n_fixed, len(assignments))))
    recommended = next(x for x in sensitivity if x['f'] == 4 and x['eta'] == .02)
    return dict(settings=dict(alpha_family=.05, primary_family_size=10, planning_power=.8,
        z_one_sided_Bonferroni=z_family, z_power=z_power, f_values=list(F_VALUES), eta_values=list(ETA_VALUES),
        recommendation_f=4, recommendation_eta=.02, minimum_n=2, n_cap=None),
        normal_scenarios=scenarios, reference_directions=directions,
        sensitivity=sensitivity, recommended=recommended,
        law_visibility={cid: dict(pi_nonzero=None, pi4=None, visibility=None, n_visibility=None,
            reason='Candidate-specific saved p2/p4 absent from approved summary inputs; no law reconstructed.') for cid in by_id},
        unknown_tail_cost=None, population_mean_certified=False, CI_or_p_values_computed=False,
        raw_samples_read=False, old_and_new_cohorts_pooled=False, new_seeds_generated=0)


def verify_bytes(spec):
    for item in spec['inputs']:
        c.require(c.sha(item['path']) == item['sha256'], 'INPUT_CHANGED:'+item['role'])
        raw = subprocess.check_output(['git', 'show', FIXED_RESULT+':'+item['repo_path']], cwd=ROOT)
        c.require(hashlib.sha256(raw).hexdigest() == item['sha256'], 'FIXED_RESULT_INPUT_BINDING')
    for name, h in spec['source_hashes'].items():
        c.require(c.sha(ROOT/name) == h, 'SOURCE_CHANGED')
        raw = subprocess.check_output(['git', 'show', spec['SOURCE']+':'+name], cwd=ROOT)
        c.require(hashlib.sha256(raw).hexdigest() == h, 'SOURCE_GIT_BINDING')
    for name, h in spec['runtime_hashes'].items():
        c.require(c.sha(name) == h, 'STATS_RUNTIME_CHANGED')
    c.require(c.sha(__import__('sys').executable) == spec['python_sha256'], 'STATS_PYTHON_CHANGED')


def authorize(spec, grant):
    c.require(spec.get('stage') == grant.get('stage') == 'S_STATS', 'S_STATS_ONLY')
    c.require(spec.get('status') == 'SEALED_FOR_EXECUTION' and spec.get('technical_sealed') is True,
              'DRAFT_UNAUTHORIZED')
    c.require(grant.get('approved_by_user') is True and grant.get('runtime_authorization') is True,
              'S_STATS_APPROVAL_REQUIRED')
    c.require(grant.get('one_shot') is True and grant.get('retry') is False
              and grant.get('next_stage_authorized') is False, 'ONE_SHOT_MANDATORY_STOP')
    c.require(spec.get('fixed_result_commit') == FIXED_RESULT and spec['SOURCE'] == grant['SOURCE']
              and c.digest(spec) == grant['spec_digest'], 'S_STATS_CONTRACT_BINDING')
    c.require(c.sha(grant['approval_receipt']) == grant['approval_receipt_sha256'], 'APPROVAL_RECEIPT_CHANGED')
    c.require(not Path(spec['output']).exists() and not Path(spec['lock']).exists(), 'USED_S_STATS_ONE_SHOT')
    verify_bytes(spec)
    inputs = {x['role']: c.read(x['path']) for x in spec['inputs'] if x['role'] != 'design_document'}
    draft, registration = inputs['original_S_STATS_contract'], inputs['P_registration']
    c.require(draft['stage'] == 'S_STATS' and draft['numeric_design_pass'] == 1
              and draft['maximum_normal_scenarios'] == 84 and draft['fixed_reference_sign_comparisons'] == 14
              and all(draft[k] == 0 for k in FORBIDDEN), 'ORIGINAL_S_STATS_SCOPE')
    c.require(len(registration['candidates']) == 9 and len(registration['assignments']) == 14
              and sum(x['role'] == 'PRIMARY' for x in registration['assignments']) == 10,
              'PREREGISTERED_9_CANDIDATES_14_COMPARISONS')
    d, audit, p = inputs['D_result'], inputs['D_completion_audit'], inputs['P_after_D_contract']
    d_sha = next(x['sha256'] for x in spec['inputs'] if x['role'] == 'D_result')
    c.require(audit['status'] == 'COMPLETE_MANDATORY_STOP' and audit['native_driver_exit'] == 0
              and audit['known_worker_pids_absent_at_audit'] is True and audit['result_sha256'] == d_sha
              and p['D_result_sha256'] == d_sha, 'ACCEPTED_FIXED_D_RESULT')
    pf = {str(w['epsilon']): w for w in d['winners'] if w['bias_mode'] == 'USUAL_SIGNED'}
    c.require(pf == p['PF_by_epsilon'] and all(w['best'] is not None for w in pf.values()), 'FIXED_PF_REFERENCES')
    c.require(len(registration['saved_statistics']) == 14, 'SAVED_STATISTICS_ONLY')
    return registration, pf


def execute(spec, grant):
    registration, pf = authorize(spec, grant)
    output, lock = Path(spec['output']), Path(spec['lock'])
    c.require(output.parent == lock.parent, 'EXCLUSIVE_S_STATS_RUN')
    with lock.open('x') as f:
        f.write(c.encoded(dict(SOURCE=spec['SOURCE'], spec_digest=c.digest(spec))).decode())
    output.mkdir(exist_ok=False)
    ledger = dict(numeric_design_pass=1, normal_scenarios_reserved=84, reference_sign_comparisons_reserved=14,
                  nonrefundable=True, retry=False, next_stage_authorized=False)
    c.save(output/'calls_ledger.json', ledger)
    started = time.monotonic()
    try:
        result = analyze(registration, pf)
        c.require(len(result['normal_scenarios']) == 84 and len(result['reference_directions']) == 14
                  and len(result['recommended']['n_fixed']) == 9, 'COMPLETE_FIXED_S_STATS_SCOPE')
        verify_bytes(spec)
        result.update(SOURCE=spec['SOURCE'], fixed_result_commit=FIXED_RESULT,
            D_result_sha256=next(x['sha256'] for x in spec['inputs'] if x['role'] == 'D_result'),
            actual_calls=dict(numeric_design_pass=1, normal_scenarios=84, fixed_reference_sign_comparisons=14,
                              **{k: 0 for k in FORBIDDEN}), P_executed=False)
        c.save(output/'result.json', result)
        c.save(output/'terminal.json', dict(status='COMPLETE_MANDATORY_STOP', SOURCE=spec['SOURCE'],
            result_sha256=c.sha(output/'result.json'), numeric_wall_seconds=time.monotonic()-started,
            final_closure_checked=True, retry=False, next_stage_authorized=False))
        return result
    except BaseException as error:
        c.save(output/'STOP.json', dict(status='TECHNICAL_STOP', reason=type(error).__name__+':'+str(error),
            reservations_retained=True, retry=False, next_stage_authorized=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--grant', type=Path, required=True)
    args = parser.parse_args()
    execute(c.read(args.spec), c.read(args.grant))


if __name__ == '__main__':
    main()
