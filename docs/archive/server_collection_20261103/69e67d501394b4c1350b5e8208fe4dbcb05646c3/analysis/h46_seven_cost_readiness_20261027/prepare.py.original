"""Phase A: saved moments, input bytes, deterministic seeds; no science calls."""
import ast
from collections import defaultdict
import csv
from fractions import Fraction
import hashlib
import io
import json
import math
from pathlib import Path
from statistics import NormalDist
import struct
import subprocess
import sys
import tarfile
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import h46_long_time_report as report
import h46_long_time_signal as study
import h46_long_time_native as native
from h46_m5_contract import core

BASE = '2e518b86e7c052dafb20dece0de7ffa6c7c85f8f'
PROVIDER = 'c31ef424175c02379eafebe8fc6c7548c1d9aabe'
TASK = ROOT.parents[1] / 'h46-seven-cost-speedup-20261027'
# ROOT is .../.server-preparation/worktrees/<worktree>; TASK is a sibling
# private task under .server-preparation, outside the versioned worktree.
PYTHON = Path('/home/AbeHiromu/projects/partially-randomized-trotter-wp11/.venv-server-opt2/bin/python')
IDS = ('d748ee85d02193720b34', '205f25b8cb63949b4b18', '53acfe65c49018ff2506',
       '6eb81b93d048deeeb63f', '54510520e99d85af4690', 'cf094355c78d50ae45f7',
       '45f9fc0a53c1a37c7345')
COHORTS = ('SEVEN_20261027_EXPLORATION', 'SEVEN_20261027_CONFIRMATION')
PUBLIC_INPUTS = ROOT / 'artifacts/resource_applicability/track_a_h246_reconstructed_confirmation/2026-10-22/published_inputs'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n')


def csv_write(name, values):
    fields = list(dict.fromkeys(k for v in values for k in v))
    with (HERE / name).open('w', newline='') as stream:
        out = csv.DictWriter(stream, fields, lineterminator='\n');out.writeheader();out.writerows(values)


def fraction(v):
    return Fraction(v['numerator'], v['denominator'])


def variance(stats, nc, ns):
    return (nc * nc * fraction(stats['cosine']['variance_exact'])
            + ns * ns * fraction(stats['sine']['variance_exact'])
            + 2 * nc * ns * fraction(stats['covariance_exact']))


def sample_size(cv2, relative_se):
    # Inflation applies to variance, not SE. n is fixed before acquisition.
    return max(33, math.ceil(4 * cv2 / relative_se ** 2))


def npy_record(data):
    """Header/payload byte inspection only; no array or Hamiltonian creation."""
    assert data[:6] == b'\x93NUMPY'
    width = 2 if data[6] == 1 else 4
    length = struct.unpack('<H' if width == 2 else '<I', data[8:8 + width])[0]
    header = ast.literal_eval(data[8 + width:8 + width + length].decode('latin1'))
    assert not header['fortran_order']
    # Existing receipt array_record uses np.ascontiguousarray, whose metadata
    # shape for a scalar is [1]. This does not reshape the saved scientific input.
    return dict(shape=list(header['shape']) or [1], dtype=header['descr'],
                sha256=hashlib.sha256(data[8 + width + length:]).hexdigest())


def input_info():
    old = study.read(study.PUBLIC / 'scientific_provenance.json')['inputs']
    systems = {}
    for s in ('H4', 'H6'):
        names = ({'H4_tol_snapshot.npz': 'tol_only_snapshot.npz', 'H4_tol_receipt.json': 'tol_only_receipt.json'}
                 if s == 'H4' else {'H6_snapshot.npz': 'h6_input_snapshot.npz',
                     'H6_snapshot_receipt.json': 'snapshot_receipt.json', 'H6_df_receipt.json': 'df_receipt.json'})
        inputs = {role: dict(path=str(PUBLIC_INPUTS / s / name), sha256=sha(PUBLIC_INPUTS / s / name))
                  for role, name in names.items()}
        binding = old[s]['binding']
        if s == 'H4':
            receipt = study.read(PUBLIC_INPUTS / s / 'tol_only_receipt.json')
            assert receipt['hamiltonian_hash'] == binding['hamiltonian']
            assert receipt['state_hash'] == binding['state'] and receipt['rank'] == binding['rank']
            with zipfile.ZipFile(PUBLIC_INPUTS / s / 'tol_only_snapshot.npz') as z:
                actual = {p[:-4]: npy_record(z.read(p)) for p in z.namelist()}
            assert actual == receipt['arrays']
            evidence = study.read(PUBLIC_INPUTS.parent / 'input_reconstruction.json')['H4']
            assert all(evidence['array_hash_comparisons'].values())
            assert evidence['old_snapshot_sha256'] == binding['snapshot']
            assert evidence['new_snapshot_sha256'] == inputs['H4_tol_snapshot.npz']['sha256']
            equivalence = 'SAVED_20261022_RECONSTRUCTION; all seven scientific array hashes equal original; NPZ container differs'
        else:
            assert inputs['H6_snapshot.npz']['sha256'] == binding['snapshot']
            for name in inputs:
                assert inputs[name]['sha256'] == old[s]['files'][name]['sha256']
            equivalence = 'ORIGINAL_SNAPSHOT_AND_RECEIPTS_BYTE_IDENTICAL; saved weighted-Hermitian completion'
        systems[s] = dict(binding=binding, inputs=inputs, equivalence=equivalence)
    return systems


def materialize_provider():
    target = TASK / 'provider'
    data = subprocess.check_output(['git', 'archive', PROVIDER, 'src'], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        members = [m for m in archive.getmembers() if m.isfile()]
        target.mkdir(parents=True, exist_ok=True)
        for member in members:
            p = target / member.name
            assert p.resolve().is_relative_to(target.resolve())
            value = archive.extractfile(member).read()
            if p.exists():
                assert p.read_bytes() == value, 'Existing private provider differs'
            else:
                p.parent.mkdir(parents=True, exist_ok=True);p.write_bytes(value)
    return target


def historical_seed_files():
    root = ROOT.parents[1] / 'h246-reconstructed-confirmation-20261022/confirmation_run02'
    files = []
    for s in ('H4', 'H6'):
        config = study.read(root / (s + '_CONFIRMATION/config_APPROVED_PRIVATE.json'))
        files.extend(Path(config[k]) for k in ('registry_path', 'excluded_seeds_path'))
    return sorted(set(files))


def integer_leaves(v):
    if type(v) is int:
        yield v
    elif isinstance(v, dict):
        for x in v.values():
            yield from integer_leaves(x)
    elif isinstance(v, list):
        for x in v:
            yield from integer_leaves(x)


def seed_audit(rows, known_files):
    # Enumerating hash-derived seed identifiers is not RTE random sampling.
    occupied = set()
    for p in known_files:
        saved = study.read(p)
        occupied.update(saved['values'] if isinstance(saved,dict) and 'values' in saved else integer_leaves(saved))
    # Reconstruct original long-time seed identifiers from saved cohort/index
    # metadata and its unchanged hash recipe; do not reopen raw or resample.
    saved_groups = list(report.costs().values())
    for name in ('confirmation_costs.json.gz', 'reused_confirmation_costs.json.gz'):
        saved_groups += list(study.read(study.PUBLIC / name).values())
    trials = set()
    for group in saved_groups:
        for path in group.get('trial_files', []):
            c = group['candidate'];cohort = group['cohort'];index = int(Path(path).parent.name)
            identity = (c['id'], cohort, index)
            if identity in trials:
                continue
            trials.add(identity)
            for piece in core.schedule(c['formula'], c['q'], 'ABS_TIME_PER_OUTER', c['R_outer'], T=c['T']) if c['method']=='PR' else []:
                occupied.add(native.trial_seed(c, cohort, index, piece['outer'], piece['piece']))
    original_count = len(occupied);new_count = 0;digest = hashlib.sha256()
    for cohort in COHORTS:
        for item in rows:
            c = item['candidate'];n = item['n_exploration' if cohort == COHORTS[0] else 'n_confirmation']
            for i in range(n):
                for piece in core.schedule(c['formula'], c['q'], 'ABS_TIME_PER_OUTER', c['R_outer'], T=c['T']):
                    v = native.trial_seed(c, cohort, i, piece['outer'], piece['piece'])
                    assert v not in occupied, 'Seed identifier collision'
                    occupied.add(v);new_count += 1;digest.update(str(v).encode() + b'\n')
    return dict(known_identifiers=original_count, original_long_time_trials_identified=len(trials), new_piece_seed_identifiers=new_count,
                new_seed_sequence_sha256=digest.hexdigest(), collisions=0,
                complete_historical_seed_coverage=False, PRNG_independence_proved=False)


def main():
    endpoints, prices = report.endpoints(), report.costs()
    candidates = {e['candidate']['id']: e['candidate'] for e in endpoints}
    selected = {x['id']: x for x in csv.DictReader((ROOT / 'analysis/h46_unpriced_screening_20261026/promising_candidates.csv').open())
                if x['recommendation'] == 'INITIAL'}
    assert set(selected) == set(IDS) and not (set(IDS) & set(prices))
    best = {(b['system'], b['T'], b['epsilon'], b['family']): b['best']
            for b in study.read(study.PUBLIC / 'best_configurations.json') if b['formula_group'] == 'ALL'}
    anchors = [(candidates[i], p) for i, p in prices.items() if candidates[i]['method'] == 'PR']
    confirmation = study.read(study.PUBLIC / 'confirmation_costs.json.gz')
    confirmation.update(study.read(study.PUBLIC / 'reused_confirmation_costs.json.gz'))
    z = NormalDist().inv_cdf(1 - .05 / (2 * 11))
    rows = [];diagnostics = [];comparisons = []
    for index, identity in enumerate(IDS):
        c = candidates[identity];e = float(selected[identity]['epsilon'])
        ep = next(v for v in endpoints if v['candidate']['id'] == identity and v['epsilon'] == e)
        assert ep['status'] == 'ELIGIBLE' and all(str(c[k]) == selected[identity][k] for k in ('system', 'method', 'formula'))
        assert all(c[k] == int(selected[identity][k]) for k in ('q','prefix','R_outer','K'))
        assert all(c[k] == float(selected[identity][k]) for k in ('T','delta'))
        assert (ep['N_cos'],ep['N_sin']) == (int(selected[identity]['N_cos']),int(selected[identity]['N_sin']))
        same_p = [(a, p) for a, p in anchors if all(a[k] == c[k] for k in ('system', 'formula', 'prefix'))]
        pool = [(a, p) for a, p in same_p if a['q'] == c['q'] and a['R_outer'] == c['R_outer']]
        rule = 'same_prefix_q_R; other_saved_T_K'
        q_adjust = False
        if not pool:
            pool = [(a, p) for a, p in same_p if a['q'] == c['q']]
            rule = 'same_prefix_q; different_R; no_R_variance_guarantee'
        if not pool:
            pool = same_p;q_adjust = True
            rule = 'same_prefix_R; q_transfer_CV_times_sqrt_q_ratio; additive_independent_step_planning_assumption'
        assert pool
        cv2s = []
        for a, p in pool:
            stats = p['unit_statistics'];nc, ns = ep['N_cos'], ep['N_sin']
            mean = nc * fraction(stats['cosine']['mean_exact']) + ns * fraction(stats['sine']['mean_exact'])
            cv2 = float(variance(stats, nc, ns) / mean ** 2)
            if q_adjust:
                cv2 *= max(1, a['q'] / c['q'])
            cv2s.append(cv2)
            diagnostics.append(dict(target_id=identity, analogue_id=a['id'], T=a['T'], q=a['q'], R_outer=a['R_outer'], K=a['K'],
                n=stats['cosine']['n'], C_mean=stats['cosine']['mean'], C_min=stats['cosine']['minimum'],
                C_max=stats['cosine']['maximum'], C_SE=stats['cosine']['SE'],
                cosine_variance=stats['cosine']['variance'], sine_variance=stats['sine']['variance'], covariance=stats['covariance'],
                planning_CV=math.sqrt(cv2), transfer_rule=rule))
        cv2 = max(cv2s)
        ne = sample_size(cv2, .01)
        nh = .005 if index not in (2, 5) else .01
        nf = sample_size(cv2, nh / z)
        pf, pr = best[(c['system'], c['T'], e, 'PF')], best[(c['system'], c['T'], e, 'PR')]
        item = dict(candidate=c, primary_epsilon=e, saved_endpoints=[v for v in endpoints if v['candidate']['id'] == identity],
            n_exploration=ne, n_confirmation=nf, variance_inflation=4, analogue_CV_max=math.sqrt(cv2),
            analogue_ids=[a['id'] for a, _ in pool], transfer_rule=rule,
            exploration_relative_SE_target=.01, confirmation_nominal_relative_halfwidth=nh,
            primary_PF=pf, primary_PR=pr, fixed_equal_axis_PF_break_even=pf['G'] / (ep['N_cos'] + ep['N_sin']),
            saved_independent_PR_price=confirmation.get(pr['id']),
            fixed_equal_axis_PR_break_even=pr['G'] / (ep['N_cos'] + ep['N_sin']))
        rows.append(item)
        for reference in ([pf, pr] if index < 2 else [pr]):
            independent = confirmation.get(reference['id'])
            new_sd = nh / z * min(pf['G'], pr['G'])
            saved_se = reference.get('G_SE') or 0.
            independent_se = None
            if independent:
                # Saved reference is evaluated using its own saved shots.
                ref_ep = next(v for v in endpoints if v['candidate']['id'] == reference['id'] and v['epsilon'] == e)
                independent_se = report.cost_for(ref_ep, independent)['G_SE']
            comparisons.append(dict(target_id=identity, reference_id=reference['id'], family='PF' if reference['method'] != 'PR' else 'PR',
                system=c['system'], T=c['T'], epsilon=e, saved_G=reference['G'], saved_G_SE=saved_se,
                saved_reference_halfwidth_floor=z * saved_se,
                nominal_combined_halfwidth= z * math.hypot(new_sd, saved_se),
                saved_independent_reference_available=independent is not None,
                saved_independent_reference_G_SE=independent_se,
                interpretation='fixed_saved_point_primary; two_mean_intervals_are_diagnostics_with_selection_and_tail_limits'))
    assert len(comparisons) == 9
    provider = materialize_provider();systems = input_info();seed_files = historical_seed_files()
    plan = dict(status='PHASE_A_FIXED_AWAITING_USER_EXECUTION_PERMISSION', base_commit=BASE,
        original_results_commit='155529e20a8fac39e00786c40b57830b71203c4f',
        previous_execution_SOURCE='1255f6e60656494a45aebceeefe5224313e4355f',
        provider_commit=PROVIDER, provider_root=str(provider), python=str(PYTHON),
        private_output_root=str(TASK / 'run01'), public_results_root=str(HERE / 'results'),
        cohorts=list(COHORTS), independent_confirmation='ALL_SEVEN_UNCONDITIONAL_AFTER_EXPLORATION_SUCCESS',
        confirmation_topup=False, new_candidate_expansion=False, baseline_PR_lower_cost_cells=33,
        primary_comparisons=11, candidate_vs_saved_comparisons=9, R_unit_price_contrasts=[[IDS[0], IDS[1]], [IDS[2], IDS[5]]],
        nominal_familywise_alpha=.05, normal_bonferroni_z=z, confidence_guaranteed=False,
        systems=systems, candidates=rows, seed_audit=seed_audit(rows, seed_files),
        known_seed_sources=[dict(path=str(p), sha256=sha(p)) for p in seed_files],
        compiler=dict(qiskit='1.3.0', numpy='1.26.4', basis=['rz','sx','x','cx'], optimization_level=1,
                      seed_transpiler=17, settings_hash=native.COMPILER_HASH),
        environment_policy='reuse_existing_environment_read_only; minor nonsemantic runtime differences accepted; no global installation',
        parallel_policy='heavy_work_first; shared_contexts_and_bulk_pool_across_cohorts; adaptive_inflight_idle_plus_owned_CPU; nested_compiler_BLAS_threads_1; representative_RSS_memory_admission',
        execution_memos='per_worker_schedule_memo_and_one_exact_previous_evolution; original_sampler_and_correctness_checks_unchanged',
        scientific_calls=dict(trajectories_exploration=sum(x['n_exploration'] for x in rows),
            trajectories_confirmation=sum(x['n_confirmation'] for x in rows),
            trajectories_total=sum(x['n_exploration'] + x['n_confirmation'] for x in rows),
            logical_wrapper_axes=2 * sum(x['n_exploration'] + x['n_confirmation'] for x in rows),
            logical_RTE_events=sum((x['n_exploration']+x['n_confirmation'])*x['candidate']['q']*x['candidate']['R_outer'] for x in rows),
            additional_compiler_anchors=0, maximum_new_transpiles=2 * sum(x['n_exploration'] + x['n_confirmation'] for x in rows),
            Hamiltonian_generation=0, state_generation=0, reference_generation=0, signal_recomputation=0,
            new_shots_or_allowance=0, actual_calls_in_Phase_A=0),
        stop_rules=['PHASE_A_END', 'ANY_CORRECTNESS_OR_IDENTITY_FAILURE', 'PARTIAL_OR_STOP_TRIAL_FOUND',
                    'SEED_COLLISION', 'SHARED_RESOURCE_INTERFERENCE', 'FIXED_N_COMPLETE'],
        saved_data_hashes={str(path.relative_to(ROOT)):sha(path) for path in [
            *(study.PUBLIC/name for name in ('signal_endpoints.json.gz','extension_signal_endpoints.json.gz',
                'best_configurations.json','reused_compiled_prices.json.gz','exploration_costs.json.gz',
                'extension_parallel_costs.json.gz','fixed_delta_exploration_costs.json.gz',
                'confirmation_costs.json.gz','reused_confirmation_costs.json.gz','scientific_provenance.json')),
            ROOT/'analysis/h46_unpriced_screening_20261026/promising_candidates.csv']},
        source_hashes={str(p.relative_to(ROOT)): sha(p) for p in {
            *(ROOT / name for name in study.read(study.PUBLIC / 'completion_provenance.json')['source_files']),
            ROOT / 'scripts/h46_m5_contract.py', ROOT / 'scripts/h46_pr_pilot_repair.py',
            HERE / 'prepare.py', HERE / 'execute.py', HERE / 'performance.py'} if p.exists()})
    previous = json.loads(subprocess.check_output(['git','show',
        '1255f6e60656494a45aebceeefe5224313e4355f:analysis/h46_seven_cost_readiness_20261027/PLAN.json'],cwd=ROOT))
    for name in ('candidates','cohorts','seed_audit','compiler','scientific_calls','R_unit_price_contrasts',
                 'normal_bonferroni_z','independent_confirmation','confirmation_topup'):
        assert plan[name] == previous[name], 'Optimization changed fixed scientific plan: ' + name
    plan['fixed_scientific_plan_unchanged_from_previous_SOURCE'] = True
    write('PLAN.json', plan)
    csv_write('sample_plan.csv', [dict(order=i + 1, id=x['candidate']['id'], **{k:v for k,v in x['candidate'].items() if k!='id'},
        primary_epsilon=x['primary_epsilon'], n_exploration=x['n_exploration'], n_confirmation=x['n_confirmation'],
        analogue_CV_max=x['analogue_CV_max'], confirmation_nominal_relative_halfwidth=x['confirmation_nominal_relative_halfwidth'],
        transfer_rule=x['transfer_rule']) for i,x in enumerate(rows)])
    csv_write('analogue_moments.csv', diagnostics);csv_write('baseline_precision.csv', comparisons)
    print(json.dumps(dict(samples=[(x['candidate']['id'], x['n_exploration'], x['n_confirmation']) for x in rows],
        calls=plan['scientific_calls'], z=z, seed_audit=plan['seed_audit']), indent=2))


if __name__ == '__main__':
    main()
