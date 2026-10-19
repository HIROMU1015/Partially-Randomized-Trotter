"""Rank10 H4 protocol metadata. No numerical provider is imported here."""
from copy import deepcopy
import ast
import os
from fractions import Fraction
from pathlib import Path
import subprocess
import sys

import h4_dp_contract as old
from h4_native_contract import cell, Writer as old_writer

require, digest, sha, read, save = old.require, old.digest, old.sha, old.read, old.save
ROOT = Path(__file__).resolve().parents[1]
SCOPE = 'H4_TOL10_MAIN_V1_20261019'
EPS = [.05, .01, .005, .001, .0005]
Q = (1, 2, 4, 8, 16)
BINDING = dict(system='linear_H4_1Angstrom', basis='STO-3G', rank=10,
    rank_policy='tol_only_1e-8', T=.8, P=0, control='symmetric_directional',
    scalar='original_plus_tail_once', reference_rotated=False,
    hamiltonian='c1a6c4281f11bba90df2fd18a998bdec96bed00379116537d22faf548c525ea0',
    state='9bcc0b0ecf44440c4411bcb76d652d78ff0595f036144580660c56c47c78e80b',
    snapshot='12219b7d14dc2a275853314269e5e8628005ed72b42d16bc4d3831c8ea515c96',
    compiler='1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b')
FP_PR = 'B2_p3_q4_2nd_r4_K4'
FP_DETERMINISTIC = ['B0I_p8_q4_2nd', 'B0I_p8_q4_4th', 'B1_p10_q4_2nd', 'B1_p10_q4_4th']
FP_SIGNALS_D = FP_DETERMINISTIC + ['B0_p8_q4_2nd', 'B0_p8_q4_4th']
KERNEL_FILES = ('scripts/h4_tol_main_contract.py', 'scripts/h4_tol_main_science.py',
                'scripts/run_h4_tol_main.py')


def source_closure(root=ROOT):
    pending = list(KERNEL_FILES); found = {}
    while pending:
        name = pending.pop()
        if name in found: continue
        path = Path(root)/name
        require(path.is_file(), 'SOURCE_CLOSURE:'+name)
        found[name] = sha(path)
        for node in ast.walk(ast.parse(path.read_text())):
            modules = [a.name for a in node.names] if isinstance(node, ast.Import) else (
                [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            for module in modules:
                candidate = 'scripts/'+module.split('.')[0]+'.py'
                if (Path(root)/candidate).is_file(): pending.append(candidate)
    return found


def registrations():
    # Fixed structural rules only; no saved bias/cost is used to select this grid.
    d = [cell(method, p, q, f) for p in range(1, 10) for q in Q
         for f in ('2nd', '4th') for method in ('B0', 'B0I')]
    d += [cell('B1', 10, q, f) for q in Q for f in ('2nd', '4th')]
    center = 10 // 4
    a = [cell('B2', p, q, r=4, K=k) for p in (center-1, center, center+1)
         for q in Q for k in (2, 4)]
    a += [cell('B2', center, q, r=r, K=k) for q in Q for r in (2, 8) for k in (2, 4)]
    require(len({x['id'] for x in a}) == len(a) == 50 and len(d) == 190, 'UNIQUE_REGISTERED_GRID')
    return dict(D=d, A=a, rule='D all nonempty truncated prefixes and full rank; A floor(rank/4)±1 with central r sensitivity',
        result_dependent_grid=False, global_optimum_certified=False,
        q_boundary='q16 winners remain registered-boundary results; no automatic extension')


def budgets():
    reg = registrations()
    new_a = [x for x in reg['A'] if x['id'] != FP_PR]
    return dict(D=dict(new_full_H_reference=0, SCF=0, integral_build=0, DF=0, saved_state_solver=0,
        new_base_signal=96, analytic_B0I_overlay=88, new_signed_shot_endpoints=920,
        independent_forward_signal_maximum=96, exact_truncated_reference_maximum=88,
        native_evolution_maximum=186, wrapper_compile_maximum=372, control_probe_maximum=744,
        preparation_reconstruction_maximum=10, primitive_probe_maximum=660, endpoint_accounting_pass=1,
        saved_signal_records=6, saved_deterministic_configurations=4, saved_axes=8),
        A=dict(new_full_H_reference=0, SCF=0, integral_build=0, DF=0, saved_state_solver=0,
        new_base_signal=49, new_signed_shot_endpoints=245, trajectory_maximum=8*len(new_a),
        independent_forward_signal_maximum=49, independent_tail_matrix_maximum=49,
        occurrence_maximum=8*sum(x['q'] for x in new_a), RTE_event_maximum=8*sum(x['R'] for x in new_a),
        native_evolution_maximum=8*len(new_a), wrapper_compile_maximum=16*len(new_a),
        control_probe_maximum=4*len(new_a), independent_event_oracle_maximum=len(new_a),
        preparation_reconstruction_maximum=3, primitive_probe_maximum=330, endpoint_accounting_pass=1,
        reused_FP_anchor_n=32, reused_FP_axes=64, new_exploration_n=8,
        old_FP_raw_statistics_recomputed=False),
        S_STATS=dict(status='SEPARATE_NUMERICAL_ANALYSIS_APPROVAL_REQUIRED', numerical_pass=None,
            model='approximate normal fixed-gap planning over saved mean/SE/n only',
            family_alpha=.05, power=.8, primary_endpoint_multiplicity=2,
            SD_inflation_sensitivity=[1, 2, 4, 8], relative_gap_sensitivity=[.01, .02, .05],
            proposed_reference_scenario=dict(SD_inflation=4, relative_gap=.02),
            formula='max(2,ceil(((z_(1-alpha/m)+z_power)*inflation*saved_SE*sqrt(saved_n)/(eta*PF_G))**2))',
            shared_candidate_n='maximum over its primary endpoints',
            rank12_variance_transferred=False, unseen_tail_mean_certified=False),
        P=dict(status='PENDING_D_A_RESULT', trajectory=None, wrapper_compile=None, n=None))


def plan(stage):
    require(stage in ('D', 'A', 'S_STATS', 'P'), 'SCOPED_STAGE')
    reg = registrations()
    return dict(schema=SCOPE, stage=stage, binding=deepcopy(BINDING), epsilons=EPS,
        bias_mode='USUAL_SIGNED', allowance_factor=1., compiler_options=dict(
            qiskit='1.3.0', basis_gates=['rz', 'sx', 'x', 'cx'], optimization_level=1,
            seed_transpiler=17, backend=None, coupling_map=None, layout=None, routing=None),
        cells=reg[stage] if stage in ('D', 'A') else None, calls=budgets()[stage],
        acquisition_rule='at least one saved eligible signed/factor1 endpoint; unresolved is never ineligible',
        reuse_FP_PR_as='SAVED_FP_PANEL_N32_ANCHOR' if stage == 'A' else None,
        new_cohort='EXPLORATION_N8' if stage == 'A' else None,
        seed_policy=dict(namespace=SCOPE+'_EXPLORATION' if stage == 'A' else SCOPE+'_CONFIRMATION',
            master='SHA256 canonical [namespace,SOURCE,plan_digest,input_binding,cell_id,replica] first64bit',
            steps='original direct_master_to_step_seeds_v1', values=None,
            old_inventory='legacy H4, geometry, conditional pilots, F/FP, rank12 P and accepted rank10 A',
            collision_receipt='PENDING_FROZEN_SOURCE_AND_INTEGER_INVENTORY_CHECK',
            cross_candidate_CRN=False, cosine_sine_shared_trajectory=True,
            collision_zero_is_independence_proof=False),
        P_contract=dict(primary_epsilons=[.05, .0005], status='PENDING_D_A_RESULT',
            candidates=None, PF=None, n=None, SOURCE=None, seeds=None,
            selection='lowest complete registered A point cost at each primary epsilon; lexical exact ties; deduplicate same candidate',
            absent_eligible_PR='NO_ELIGIBLE_PR; no replacement or endpoint expansion',
            absent_complete_PF='INCONCLUSIVE_MISSING_PF; no P launch',
            missing_eligible_cost='block final selection; never exclude missing as costly',
            sample_size='separate S_STATS over saved mean/SE/n and recorded rare information; no raw aggregation in Phase0',
            sensitivity='predeclare variance inflation and target gaps before numerical analysis; unseen S4 cost remains unknown',
            decision='sign(PF_G - new paired PR mean) agrees with fixed exploration direction; zero is INCONCLUSIVE',
            multiplicity='family design declared before S_STATS; shared-candidate endpoints use same n with covariance',
            no_optional_stopping=True, no_candidate_reselection=True,
            no_pooling=True, population_mean_certified=False),
        resources=dict(agent_imposed_CPU_RAM_worker_wall_caps=None,
            admission='fresh available physical CPUs, effective memory/quota, known concurrent allocations',
            worker_threads=1, preparation_cache=True, paired_evolution_shared=True,
            fingerprint_reuse=True, PSI_diagnostic_only=True, unrelated_worktree_changes_are_gates=False),
        strict_bias_certificate=None, population_mean_certified=False,
        new_scientific_results_in_Phase0=False, next_stage_authorized=False,
        automatic_retry=False, mandatory_STOP=True, proposal_is_authorization=False)


def same_binding(value):
    require(all(value.get(k) == v for k, v in BINDING.items()), 'RANK10_INPUT_COMPILER_CONTROL_BINDING')


def selected(records, cells):
    require(len(records) == len(cells) and len({x['cell']['id'] for x in records}) == len(cells), 'COMPLETE_SIGNAL_GRID')
    expected = {x['id']: x for x in cells}
    out = []
    for record in records:
        require(record['cell'] == expected.get(record['cell']['id']), 'FIXED_CELL')
        flags = []
        for eps in EPS:
            endpoint = record['precision_rows'][str(eps)]['1.0']
            flag = endpoint['cost_acquisition_eligible']
            require(type(flag) is bool, 'SAVED_ELIGIBILITY')
            if flag:
                require(all(type(endpoint['axes'][axis]['shots']) is int and endpoint['axes'][axis]['shots'] > 0
                            for axis in ('real', 'imag')), 'SAVED_SHOTS')
            else:
                require(endpoint['eligibility_under_declared_allowance'] == 'INELIGIBLE', 'UNRESOLVED_IS_NOT_EXCLUSION')
            flags.append(flag)
        if any(flags): out.append(deepcopy(record['cell']))
    return out


def seed(namespace, source, plan_digest, binding, candidate, replica):
    same_binding(binding)
    require(namespace in (SCOPE+'_EXPLORATION', SCOPE+'_CONFIRMATION') and len(source) == 40, 'SEED_SCOPE')
    return int(digest([namespace, source, plan_digest, binding, candidate, replica])[:16], 16)


def reuse_measurement(row, binding, cell_, *, accepted_result_sha256, source):
    """Accept only the same rank10 saved circuit identity; no inferred unit cost."""
    same_binding(binding)
    require(row['cell_id'] == cell_['id'] and len(source) == 40 and len(accepted_result_sha256) == 64, 'REUSE_SOURCE_RESULT')
    require(set(x['axis'] for x in row['rows']) == {'cosine', 'sine'}, 'PAIRED_REUSE_AXES')
    for axis in row['rows']:
        require(axis['compiler_hash'] == binding['compiler'] and len(axis['fingerprint']) == 64
                and type(axis['metrics']['RZ']) is int and axis['metrics']['RZ'] > 0, 'EXACT_SAVED_COMPILE_IDENTITY')
    return dict(cell_id=row['cell_id'], replica=row['replica'], cohort='SAVED_FP_PANEL_N32_ANCHOR'
        if cell_['R'] else 'SAVED_FP_DETERMINISTIC', SOURCE=source, result_sha256=accepted_result_sha256,
        axes={x['axis']: deepcopy(x) for x in row['rows']}, newly_compiled=False)


def invalid_grant(stage, contract=None):
    return dict(schema='INVALID_GRANT', scope=SCOPE, stage=stage, approved_by_user=False,
        runtime_authorization=False, SOURCE=None if contract is None else contract.get('SOURCE'),
        contract_digest=None if contract is None else digest(contract), approval_receipt=None,
        approval_receipt_sha256=None, one_shot=True, automatic_retry=False,
        next_stage_authorized=False, absolute_launch_command=None)


def validate_seed_registry(contract):
    registry = read(contract['seed_registry'])
    require(registry['SOURCE'] == contract['SOURCE'] and registry['plan_digest'] == digest(contract['plan'])
            and registry['binding'] == BINDING, 'SEED_REGISTRY_BINDING')
    cells = {x['id']: x for x in registrations()['A'] if x['id'] != FP_PR}
    tasks = registry['tasks']
    require(len(tasks) == 392 and {(x['cell']['id'], x['replica']) for x in tasks}
            == {(cid, i) for cid in cells for i in range(8)}, 'EXACT_EXPLORATION_N8_TASKS')
    excluded = set(read(contract['old_seed_exclusions'])['values']); new = set()
    for task in tasks:
        cid = task['cell']['id']
        require(task['cell'] == cells[cid] and task['seed'] == seed(SCOPE+'_EXPLORATION',
            contract['SOURCE'], digest(contract['plan']), BINDING, cid, task['replica']), 'EXACT_SEALED_SEED')
        hierarchy = old.seed_hierarchy(task['seed'], task['cell']['q'])
        require(len(hierarchy) == len(set(hierarchy)) and not set(hierarchy)&(excluded|new), 'SEED_COLLISION')
        new.update(hierarchy)
        require(task['step_seeds'] == (hierarchy if task['cell']['q'] == 1 else hierarchy[1:]), 'STEP_SEED_HIERARCHY')
    receipt = read(contract['seed_inventory_receipt'])
    require(receipt['status'] == 'COMPLETE_COLLISION_ZERO' and receipt['all_registered_scopes_covered']
        and receipt['new_seed_values_digest'] == digest(sorted(new))
        and receipt['old_explicit_exclusions_sha256'] == sha(contract['old_seed_exclusions'])
        and all(not x['collisions'] for x in receipt['geometry_checks']), 'COMPLETE_OLD_GEOMETRY_FP_P_SEED_RECEIPT')


def verify_provider_tree(contract):
    roots = contract['provider_roots']
    require(roots, 'COMPLETE_PROVIDER_ROOTS')
    actual = {str(p.resolve()) for root in roots for p in Path(root).rglob('*') if p.is_file()}
    require(actual == set(contract['provider_hashes']), 'COMPLETE_PROVIDER_FILE_SET')


def authorize(contract, grant, root=ROOT):
    # Approval must be checked before any scientific input, provider, output or lock.
    require(grant.get('approved_by_user') is True and grant.get('runtime_authorization') is True,
            'SEPARATE_EXPLICIT_SCIENTIFIC_APPROVAL_REQUIRED')
    require(grant.get('scope') == SCOPE and grant.get('stage') == contract['stage'], 'EXACT_STAGE_SCOPE')
    require(grant.get('one_shot') is True and grant.get('automatic_retry') is False
            and grant.get('next_stage_authorized') is False, 'ONE_SHOT_MANDATORY_STOP')
    require(contract['status'] == 'SEALED_FOR_EXECUTION' and contract.get('pending') == [], 'CONTRACT_PENDING')
    require(contract['plan'] == plan(contract['stage']), 'FIXED_PLAN')
    require(grant['contract_digest'] == digest(contract) and grant['SOURCE'] == contract['SOURCE'], 'SOURCE_CONTRACT_BINDING')
    require(len(contract['SOURCE']) == 40 and sha(grant['approval_receipt']) == grant['approval_receipt_sha256'], 'APPROVAL_RECEIPT')
    same_binding(contract['binding'])
    for name, expected in contract['source_hashes'].items():
        require(sha(Path(root)/name) == expected, 'SOURCE_CHANGED:'+name)
        blob = subprocess.check_output(['git', 'show', contract['SOURCE']+':'+name], cwd=root)
        require(__import__('hashlib').sha256(blob).hexdigest() == expected, 'SOURCE_GIT_BINDING')
    require(contract['source_hashes'] == source_closure(root), 'COMPLETE_OWN_SOURCE')
    for row in contract['inputs']:
        require(sha(row['path']) == row['sha256'], 'INPUT_CHANGED:'+row['role'])
    for group in ('runtime_hashes', 'provider_hashes'):
        require(bool(contract[group]), 'COMPLETE_EXECUTION_CLOSURE')
        for path, expected in contract[group].items():
            require(sha(path) == expected, group+'_CHANGED')
    verify_provider_tree(contract)
    bound_paths = {r['path'] for r in contract['inputs']}
    require({contract['native_config'], contract['FP_evidence'], contract['saved_reference'],
             contract['font_seed']['path']} <= bound_paths, 'ACTUALLY_USED_METADATA_INPUT_BINDING')
    require(sha(sys.executable) == contract['python_sha256'], 'PINNED_PYTHON')
    require(contract['fresh_resources']['admitted'] is True and contract['worker_cpus'], 'FRESH_RESOURCE_ADMISSION')
    require(len(set(contract['worker_cpus'])) == len(contract['worker_cpus'])
            and set(contract['worker_cpus']) <= os.sched_getaffinity(0), 'OWN_ALLOWED_CPU_ASSIGNMENT')
    require(sha(contract['fresh_resources']['receipt']) == contract['fresh_resources']['sha256'], 'FRESH_RESOURCE_RECEIPT')
    require(not Path(contract['output']).exists() and not Path(contract['lock']).exists(), 'USED_ATTEMPT_NAMESPACE')
    if contract['stage'] == 'A':
        require(contract.get('accepted_D') and contract['accepted_D']['binding'] == BINDING, 'PENDING_D_RESULT')
        require(sha(contract['accepted_D']['path']) == contract['accepted_D']['sha256'], 'ACCEPTED_D_CHANGED')
        require(contract.get('seed_inventory_complete') is True and contract.get('seed_registry'), 'COMPLETE_NEW_SEED_INVENTORY')
        require({contract['seed_registry'], contract['old_seed_exclusions'], contract['seed_inventory_receipt'],
                 contract['accepted_D']['path']} <= bound_paths, 'FINAL_A_SEED_AND_D_INPUT_BINDING')
        validate_seed_registry(contract)
    if contract['stage'] in ('P', 'S_STATS'):
        # Their SOURCE, scientific n/model and final selectors require accepted
        # D/A evidence and a new reviewed contract. Phase0 cannot invent them.
        require(False, 'PENDING_D_A_RESULT_AND_SEPARATE_FINAL_CONTRACT')


def account_stage(records, pairs, *, saved_FP_rows=(), d_result=None):
    """Future authorized accounting, or artificial tests; never a Phase0 real caller."""
    rows = []
    for record in records:
        cid = record['cell']['id']
        for eps in EPS:
            if cid == FP_PR:
                candidates = [r for r in saved_FP_rows if r['cell_id'] == cid and r['epsilon'] == eps
                              and r['bias_mode'] == 'USUAL_SIGNED' and r['allowance_factor'] == 1.]
                require(len(candidates) == 1, 'SAVED_FP_STATISTIC_REQUIRED')
                row = deepcopy(candidates[0]); row['cohort'] = 'SAVED_FP_PANEL_N32_ANCHOR'
                row.setdefault('G_RZ_exact', None)
                ep = record['precision_rows'][str(eps)]['1.0']
                require(row['eligible'] == ep['cost_acquisition_eligible'], 'SAVED_FP_ELIGIBILITY')
                if row['eligible']:
                    require(row['n'] == 32 and row['G_RZ_exact'] is not None
                        and row['G_RZ_SE'] is not None, 'COMPLETE_SAVED_FP_STATISTICS')
                row['statistics_recomputed'] = False; rows.append(row); continue
            ep = record['precision_rows'][str(eps)]['1.0']
            row = dict(cell_id=cid, epsilon=eps, eligible=ep['cost_acquisition_eligible'], G_RZ_exact=None,
                G_RZ_SE=None, n=None, cohort='EXPLORATION_N8' if record['cell']['R'] else 'DETERMINISTIC')
            if row['eligible']:
                values = pairs.get(cid)
                require(values, 'MISSING_ELIGIBLE_COST')
                weights = {a: ep['axes'][b]['shots'] for a, b in zip(old.AXES, ('real', 'imag'))}
                if record['cell']['R']:
                    require(len(values) == 8 and all(v['cohort'] == 'EXPLORATION_N8' for v in values), 'EXACT_NEW_A_COHORT')
                    require({v['replica'] for v in values} == set(range(8)), 'EXACT_NEW_A_REPLICAS')
                    for value in values: old.check_pair(value, BINDING['compiler'])
                    stats = old.evaluate_pairs(values, weights, 1, 1)
                    row.update(G_RZ_exact=stats['G_exact'], G_RZ_SE=stats['paired_SE'],
                        sample_SD=stats['sample_SD'], sample_variance_exact=stats['sample_variance_exact'], n=8)
                else:
                    require(len(values) == 1, 'DETERMINISTIC_SINGLE_COST')
                    row.update(G_RZ_exact=str(sum(weights[a]*values[0]['axes'][a]['RZ'] for a in old.AXES)), n=1)
                row['shots'] = weights
            rows.append(row)
    covariance = []
    for cid, values in pairs.items():
        endpoints = [r for r in rows if r['cell_id'] == cid and r['cohort'] == 'EXPLORATION_N8' and r['eligible']]
        if endpoints:
            covariance.append(dict(cell_id=cid, epsilons=[r['epsilon'] for r in endpoints], n=8,
                sample_covariance_exact=old.within_candidate_covariance(values, [r['shots'] for r in endpoints])))
    winners = []
    for eps in EPS:
        available = [r for r in rows if r['epsilon'] == eps and r['G_RZ_exact'] is not None]
        winner = min(available, key=lambda r: (Fraction(r['G_RZ_exact']), r['cell_id'])) if available else None
        winners.append(dict(epsilon=eps, best=winner))
    comparisons = []
    if d_result is not None:
        accepted = read(d_result['path'])
        require(accepted['binding'] == BINDING and accepted['registered_set_complete'], 'COMPLETE_RANK10_D_REQUIRED')
        for winner in winners:
            deterministic = next(x['best'] for x in accepted['winners'] if x['epsilon'] == winner['epsilon'])
            pr = winner['best']
            direction = None if pr is None or deterministic is None else (
                1 if Fraction(deterministic['G_RZ_exact']) > Fraction(pr['G_RZ_exact']) else
                -1 if Fraction(deterministic['G_RZ_exact']) < Fraction(pr['G_RZ_exact']) else 0)
            comparisons.append(dict(epsilon=winner['epsilon'], PR=pr, PF=deterministic,
                point_direction=direction, status='NO_ELIGIBLE_PAIR' if direction is None else 'FINITE_SAMPLE_POINT'))
    return dict(rows=rows, winners=winners, comparisons=comparisons, registered_set_complete=True,
        scope=SCOPE, binding=deepcopy(BINDING), saved_FP_raw_aggregation=False,
        new_cohort_covariance=covariance, saved_FP_covariance='NOT_RECOMPUTED; missing saved covariance needs separate approval',
        population_mean_certified=False, strict_bias_certificate=None,
        D_result_sha256=None if d_result is None else d_result['sha256'])
