"""Metadata-only contract. A sealed plan is never permission to compute."""
from __future__ import annotations
import hashlib
import os
import subprocess
from pathlib import Path
from .ax2a_preparation import digest
from .ax2b_h6_pilot_contract_v2 import verify_parent, environment, safe_path, file_hash
from .ax2b_h6_contract import primitive_time_schedule

KIND = 'H6_MATCHED_SIGNAL_RESOURCE_V1'
RUNNER = 'scripts/resource_applicability/run_track_a_h6_matched_v1.py'
PREPARER = 'scripts/resource_applicability/prepare_track_a_h6_matched_v1.py'
AUDITOR = 'scripts/resource_applicability/audit_track_a_h6_matched_v1.py'
NAMESPACE = 'artifacts/resource_applicability/track_a_h6_matched_accuracy_v1/2026-10-11/'
PREPARATION_PATH = 'artifacts/resource_applicability/track_a_h6_matched_preparation_v1/2026-10-11/'
EPSILONS = (.05, .01, .005, .001)


def cells():
    rows = []
    def add(method, prefix, q, formula='2nd', r=None, K=None):
        name = f'{method}_p{prefix}_q{q}_{formula}' + (f'_r{r}_K{K}' if r else '')
        rows.append(dict(id=name, method=method, prefix=prefix, q=q, formula=formula,
                         r=r, R=q*r if r else None, K=K))
    for q in (1, 2, 4, 8):
        for prefix in (5, 10, 15):
            add('B0', prefix, q)
        for formula in ('2nd', '4th'):
            add('B1', 19, q, formula)
        for prefix in (5, 10, 15):
            for r in (1, 2, 4):
                for K in (2, 4):
                    add('B2', prefix, q, r=r, K=K)
    return rows


def resources(cpus):
    if (not isinstance(cpus, list) or len(cpus) != 4 or
            any(type(x) is not int or x < 0 for x in cpus) or cpus != sorted(set(cpus)) or
            not set(cpus) <= os.sched_getaffinity(0)):
        raise ValueError('MATCHED_FOUR_CPU_ASSIGNMENT')
    return dict(assigned_cpus=cpus, signal_threads=4, cost_workers=2,
                cost_cpu_sets=[cpus[:2], cpus[2:]], cost_threads=1,
                signal_address_space_bytes=8*2**30, cost_address_space_bytes=8*2**30,
                coordinator_address_space_bytes=2*2**30,
                aggregate_rss_bytes=18*2**30, memory_is_reservation=False, gpu=False)


def plan():
    cs = cells()
    pairs = set()
    schedules = []
    for c in cs:
        s = primitive_time_schedule(c, T=.8)
        s['registered_validation_times_v2'] = s['unique_primitive_times']
        schedules.append(dict(cell_id=c['id'], schedule=s))
        pairs.update(('one' if i == 0 else str(i-1), t) for i, t in s['unique_primitive_times'])
    return dict(schema='h6_matched_plan_v1', kind=KIND, T=.8, actual_rank=19,
        target=dict(model='linear_H6', geometry_angstrom=1., basis='sto-3g', n_qubits=12,
                    nelec_alpha=3, nelec_beta=3, sector_dimension=400),
        df_policy=dict(df_tol=1e-8, final_rank_supplied=False, cutoff=0.,
                       signed_generation_order_preserved=True, fragment_deletion=False),
        input_policy='WEIGHTED_HERMITIAN_PROJECTION_FROM_SAVED_RAW_V1',
        epsilons=list(EPSILONS), cells=cs,
        coverage=dict(cells=schedules, physical_times=sorted(pairs),
                      primitive_actions=3*len(pairs), primitive_probe_count=3),
        matrix_free=dict(backend='numba', num_threads=4, block_chunk_size=1),
        compiler_proposed=dict(basis_gates=['rz','sx','x','cx'], optimization_level=1,
                               seed_transpiler=17, backend=None, coupling_map=None),
        cost_primary='symmetric_directional',
        sensitivity_cells=['B1_p19_q1_4th', 'B2_p10_q2_2nd_r1_K2'],
        cost_rule=dict(acquire='all cells eligible at any epsilon under declared empirical allowance',
            exploratory_random_trajectories=8, confirmation_random_trajectories=32,
            confirmation_top_per_epsilon=2, confirmation_union_max=8,
            deterministic_replicas=1, seed_namespace='H6_MATCHED_V1_FRESH_COST',
            ordinary_sensitivity='replica 0 of registered cells, only if eligible',
            validation='saved state branches and X/Y on replica 0 per cell; paired ordinary on anchors',
            retry=False, resume=False, trajectory_sample_is_quantum_shot=False),
        numerical=dict(kind='EMPIRICAL', safety_factor=10., floor=1e-12,
            roundoff_multiplier=64., maximum_u_over_headroom=.01,
            sensitivity_factors=[1.,10.], precision='binary64',
            scope='independent occupation/expm/eigh and native/Horner/forward agreement; not a bound',
            certified=False),
        shot_rule=dict(alpha_axis=.025, axis_epsilon='epsilon/sqrt(2)',
                       estimator='B times fresh whole-trajectory Hadamard outcome',
                       quantum_shots='analytic Hoeffding allocation; no quantum shots sampled'),
        b3_diagnostic=dict(prefix=0, q=[1,2,4,8], r=[1,2,4,8,16,32,64], K=[2,4,6],
                           signal=False, sampling=False, compile=False),
        gates=dict(norm_and_leakage=1e-12, absolute_agreement=1e-9,
                   relative_agreement=1e-10, reference_expm_eigh=1e-10),
        caps_proposed=dict(phase_wall_seconds=None, total_wall_seconds=None,
            output_bytes=2*2**30, log_bytes=2**20, diagnostics=5000, progress_records=5000,
            primitive=max(10000, 3*len(pairs)), control_probe=1024, compile=1712,
            trajectory=832, occurrence=6656, reference_matvec=20000,
            reference_matvec_per_action=20000, deterministic_actions_per_cell=100000,
            tail_matvecs_corrected_and_raw={'B2':512,'B3':0},
            untranspiled_instructions=1000000, transpiled_instructions=5000000,
            integral_build=0, df_decomposition=0, state_solver=0, solver_matvec=0),
        maximum_cost_groups=852, maximum_primary_wrappers=1704,
        maximum_sensitivity_wrappers=4, maximum_wrappers=1708,
        common_preparation_cost='P excluded; report G(P)=G(0)+N_total*P and pairwise crossings',
        formal_winner_certified=False, ground_state_certified=False,
        mandatory_stop=True, next_stage_authorized=False,
        H6_status='H6_NOT_AUTHORIZED', contract_status='DRAFT_NOT_AUTHORIZATION')


def source_paths(root):
    root = Path(root)
    return sorted({str(p.relative_to(root)) for d in ('src/trotterlib', 'src/trottertracks')
                   for p in (root/d).rglob('*.py')} | {RUNNER, PREPARER, AUDITOR})


def verify_sources(root, commit, hashes):
    if not isinstance(commit, str) or len(commit) != 40 or any(x not in '0123456789abcdef' for x in commit):
        raise ValueError('MATCHED_SOURCE_COMMIT')
    if not isinstance(hashes, dict) or set(hashes) != set(source_paths(root)):
        raise ValueError('MATCHED_SOURCE_CLOSURE')
    subprocess.run(['git','merge-base','--is-ancestor',commit,'HEAD'], cwd=root, check=True,
                   stdout=subprocess.DEVNULL)
    for p, sha in hashes.items():
        blob = subprocess.check_output(['git','show',commit+':'+p], cwd=root)
        if file_hash(safe_path(root,p)) != sha or hashlib.sha256(blob).hexdigest() != sha:
            raise ValueError('MATCHED_SOURCE_CHANGED:'+p)


def preparation():
    return dict(schema='h6_matched_preparation_v1', kind=KIND, plan=plan(),
        source_commit=None, source_hashes=None, input_identity=None, environment=None,
        assigned_resources=None, exclusive_output=None, execution_plan_sealed=False,
        status='PREPARATION_ONLY_NOT_AUTHORIZED', science_authorized=False, launch_allowed=False,
        H6_status='H6_NOT_AUTHORIZED', contract_status='DRAFT_NOT_AUTHORIZATION',
        mandatory_stop=True, next_stage_authorized=False)


def validate_launch(root, manifest, grant, output, *, requested=False):
    if requested is not True or not isinstance(grant, dict) or grant.get('approved_by_user') is not True:
        raise ValueError('NEW_MATCHED_RUN_GRANT_REQUIRED')
    if grant.get('schema') != 'h6_matched_authorization_v1' or grant.get('kind') != KIND:
        raise ValueError('MATCHED_GRANT_SCHEMA')
    expected = preparation()
    for k in ('schema','kind','status','science_authorized','launch_allowed','H6_status',
              'contract_status','mandatory_stop','next_stage_authorized'):
        if type(manifest.get(k)) is not type(expected[k]) or manifest[k] != expected[k]:
            raise ValueError('MATCHED_MANIFEST_FLAG:'+k)
    if manifest.get('execution_plan_sealed') is not True or digest(manifest.get('plan')) != digest(plan()):
        raise ValueError('MATCHED_FIXED_PLAN')
    if (grant.get('manifest_digest') != digest(manifest) or grant.get('source_commit') != manifest.get('source_commit')
            or grant.get('one_shot') is not True or grant.get('retry') is not False or grant.get('resume') is not False):
        raise ValueError('MATCHED_GRANT_BINDING')
    if manifest.get('assigned_resources') != resources(grant.get('assigned_cpus')):
        raise ValueError('MATCHED_RESOURCES')
    intended = manifest.get('exclusive_output') or {}
    name = intended.get('repository_path', '')
    path = Path(output).resolve()
    if (not name.startswith(NAMESPACE) or safe_path(root,name) != path or intended.get('absolute_path') != str(path)
            or grant.get('exclusive_output') != str(path) or path.exists()):
        raise ValueError('MATCHED_EXCLUSIVE_OUTPUT')
    verify_sources(root, manifest.get('source_commit'), manifest.get('source_hashes'))
    if manifest.get('input_identity') != verify_parent(root):
        raise ValueError('MATCHED_INPUT_BINDING')
    if manifest.get('environment') != environment():
        raise ValueError('MATCHED_ENVIRONMENT')
    return manifest['assigned_resources']
