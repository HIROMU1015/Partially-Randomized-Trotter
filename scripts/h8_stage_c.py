"""Prepare a fixed H8 confirmation design; scientific execution is opt-in.

Preparation reads published aggregate fields verbatim, without reaggregating
prices or numerically redesigning n. --execute is dispatch, not user permission.
"""
import argparse
from copy import deepcopy
from fractions import Fraction
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess

import h8_long_time_adapter as a

PUBLICATION = '7fb1d125e05238a7d932fc89b8dab6ac6a7f8c84'
SIGNAL_SOURCE = 'd1272016592ff06c0567862f770ed796fd61b366'
NATIVE_SOURCE = 'df93fbb88761b61c2691bf01f761ed20a7987d7f'
PHASE_SOURCE = '18ce41942051065fa097644a8a937bd11adba708'
NAMESPACE = 'H8_STAGE_C_CONFIRMATION_V1'
SIGNAL = 'artifacts/resource_applicability/track_a_h8_long_time/2026-11-03/stage_A_combined.json.gz'
EXPLORATION = 'artifacts/resource_applicability/track_a_h8_long_time/2026-11-04/exploration.json.gz'
EVIDENCE_HASHES = {'stage_A': '9d5dcc17fd42b2f8e3f07da37784ea4d88e1dc6bb7f7f599d9794f10edc316ae',
                   'stage_B': 'd1e844b02960797d43854b0de296017a283cc9071e6c6066d878640dec1c0b29'}
SUPPLEMENT_FILES = ('scripts/h8_stage_c.py', 'scripts/h8_compiler_phase_bookkeeping.py',
                    'configs/h8_long_time/stage_c_proposal_20261104.json')
CONFIG = SCOPE = OUTPUT = None


def load(path):
    data = Path(path).read_bytes()
    return json.loads(gzip.decompress(data) if str(path).endswith('.gz') else data)


def proposal(signal, exploration, evidence):
    """Copy saved identities and aggregate values, not individual prices."""
    a.require(signal['status'] == 'PASS' and exploration['status'] == 'COMPLETE_EXPLORATION_ONLY'
              and signal['binding'] == exploration['binding'], 'COMPLETED_FIXED_EVIDENCE')
    a.require(signal['SOURCE'] == SIGNAL_SOURCE and exploration['native_SOURCE'] == NATIVE_SOURCE,
              'FIXED_BASE_SOURCES')
    selected = {}
    conditions = {'PR': dict(method='PR', formula='4th(m5_best)', prefix=8, q=1, R_outer=32, K=4),
                  'PF': dict(method='B0I', formula='4th(m5_best)', prefix=15, q=1)}
    for role, wanted in conditions.items():
        rows = [r for r in exploration['rows'] if all(r['candidate'].get(k) == v for k, v in wanted.items())]
        a.require(len(rows) == 1, 'UNIQUE_SAVED_COMPARISON:' + role)
        row = rows[0]
        a.validate_candidate(row['candidate'])
        endpoint = next(r['endpoint'] for r in signal['results'] if r['endpoint']['candidate']['id'] == row['candidate']['id'])
        a.require(endpoint['status'] == 'ELIGIBLE' and row['cohort'] == 'EXPLORATION'
                  and (row['N_cos'], row['N_sin']) == (endpoint['N_cos'], endpoint['N_sin']), 'SAVED_FIXED_SHOTS')
        selected[role] = dict(candidate=deepcopy(row['candidate']), endpoint=deepcopy(endpoint),
                              saved_aggregate=deepcopy(row))
    pr = selected['PR']['saved_aggregate']
    a.require(type(pr['n']) is int and pr['n'] >= 2 and math.isfinite(pr['sample_SE_G'])
              and pr['sample_SE_G'] > 0, 'PUBLISHED_REPLICATION_PRECISION')
    return dict(schema='H8_STAGE_C_PROPOSAL_V1', status='PREPARED_NOT_EXECUTED',
        publication_SOURCE=PUBLICATION, signal_SOURCE=SIGNAL_SOURCE, native_SOURCE=NATIVE_SOURCE,
        phase_SOURCE=PHASE_SOURCE, evidence=deepcopy(evidence), binding=deepcopy(signal['binding']),
        native_source_files=deepcopy(exploration['source_files']), comparison=selected,
        comparison_ids={role: r['candidate']['id'] for role, r in selected.items()},
        comparison_direction='PR_LOWER', cohort='CONFIRMATION',
        n=pr['n'], absolute_SE_G_target=pr['sample_SE_G'],
        precision_choice='REPEAT_PUBLISHED_EXPLORATION_N_AND_ITS_SAVED_SE',
        numerical_n_formula_evaluated=False, confidence_level=None,
        claim='INDEPENDENT_SAMPLE_COMPARISON_AND_DESCRIPTIVE_SE', population_guarantee=False,
        fixed_PF_policy='REUSE_SAVED_DETERMINISTIC_PRICE_NO_PF_SAMPLING_OR_COMPILATION',
        seed_namespace=NAMESPACE, pooled=False, automatic_top_up=False, conditional_sampling=False,
        T=1.6, epsilon=.0005, state_preparation_cost=0)


def published_proposal():
    paths = {'stage_A': SIGNAL, 'stage_B': EXPLORATION}
    evidence = {name: dict(path=path, sha256=a.sha(a.ROOT / path)) for name, path in paths.items()}
    a.require({name: value['sha256'] for name, value in evidence.items()} == EVIDENCE_HASHES,
              'PUBLISHED_STAGE_A_B_BYTES')
    return proposal(load(a.ROOT / SIGNAL), load(a.ROOT / EXPLORATION), evidence)


def validate_design(design):
    a.require(design == published_proposal(), 'FIXED_PUBLISHED_DESIGN')
    a.require(design['cohort'] == 'CONFIRMATION' and design['seed_namespace'] == NAMESPACE
              and design['comparison_direction'] == 'PR_LOWER' and not design['pooled']
              and not design['automatic_top_up'] and not design['conditional_sampling'], 'CONFIRMATION_POLICY')


def fresh_seed_map(row, n, source, binding, run_tag, occupied):
    """Called only by scientific dispatch, or with artificial test fixtures."""
    a.require(type(n) is int and n >= 2, 'FIXED_CONFIRMATION_N')
    used, result = set(occupied), {}
    for index in range(n):
        for piece in a.schedule(row):
            key = (row['id'], index, piece['outer'], piece['piece'])
            seed = int(a.digest([NAMESPACE, source, binding, run_tag, 'CONFIRMATION', *key])[:16], 16)
            a.require(seed not in used, 'EXPLORATION_OR_CONFIRMATION_SEED_COLLISION')
            used.add(seed)
            result[key] = seed
    return result


def compare_fixed_PF(pr, baseline, target):
    """A saved deterministic constant is not a new confirmation observation."""
    a.require(pr['cohort'] == 'CONFIRMATION' and not pr['pooled']
              and baseline['cohort'] == 'EXPLORATION' and baseline['n'] == 1, 'FIXED_BASELINE_NOT_POOLED')
    delta = Fraction(pr['mean_G_exact']) - Fraction(baseline['mean_G_exact'])
    se = pr['sample_SE_G']
    return dict(PR_minus_fixed_PF_exact=str(delta), observed_PR_lower=delta < 0,
        fixed_direction='PR_LOWER', absolute_SE_G_target=target, observed_sample_SE_G=se,
        descriptive_precision_met=se is not None and math.isfinite(se) and se <= target,
        PF_origin='SAVED_STAGE_B_DETERMINISTIC_PRICE', PF_observation_cohort='EXPLORATION',
        PF_confirmation_sampling=False, confidence_level=None, population_guarantee=False,
        automatic_top_up=False, pooled=False)


def preflight(design, config, source, run_tag, workers=None):
    """Metadata only: no Context, seeds, matrices, sampling, or compilation."""
    validate_design(design)
    a.preflight(config)
    a.require(config['SOURCE'] == NATIVE_SOURCE and config['source_files'] == design['native_source_files']
              and config['binding'] == design['binding'], 'NATIVE_CONFIG_SOURCE_INPUT')
    a.require(len(source) == 40 and all(c in '0123456789abcdef' for c in source), 'PUBLISHED_EXECUTION_SOURCE')
    files = {}
    for path in SUPPLEMENT_FILES:
        data = subprocess.check_output(['git', 'show', source + ':' + path], cwd=a.ROOT)
        files[path] = hashlib.sha256(data).hexdigest()
        a.require(a.sha(a.ROOT / path) == files[path], 'EXECUTION_SOURCE_BYTES:' + path)
    phase_bytes = subprocess.check_output(['git', 'show', PHASE_SOURCE + ':scripts/h8_compiler_phase_bookkeeping.py'], cwd=a.ROOT)
    a.require(hashlib.sha256(phase_bytes).hexdigest() == files['scripts/h8_compiler_phase_bookkeeping.py'], 'FIXED_PHASE_HELPER')
    for record in config['stage_B_seed_files']:
        a.require(a.sha(record['path']) == record['sha256'], 'ORIGINAL_SEED_FILE_BYTES')
    a.require(len(config['stage_B_seed_files']) == 2 and run_tag and Path(run_tag).name == run_tag
              and run_tag not in ('.', '..'), 'FRESH_RUN_NAMESPACE')
    a.require(Path(config['confirmation_run_root']).resolve().is_relative_to(a.ROOT.resolve()),
              'DEDICATED_CONFIRMATION_PROJECT_OUTPUT')
    count = min(design['n'], len(os.sched_getaffinity(0))) if workers is None else workers
    a.require(type(count) is int and count >= 1, 'WORKER_COUNT')
    return dict(stage='C', cohort='CONFIRMATION', SOURCE=source, source_files=files,
        native_SOURCE=NATIVE_SOURCE, signal_SOURCE=SIGNAL_SOURCE, phase_SOURCE=PHASE_SOURCE,
        binding=design['binding'], candidates=[design['comparison']['PR']['candidate']],
        n=design['n'], workers=count, comparison_ids=design['comparison_ids'],
        comparison_direction='PR_LOWER', absolute_SE_G_target=design['absolute_SE_G_target'],
        design_digest=a.digest(design), evidence=design['evidence'], run_tag=run_tag,
        seed_namespace=NAMESPACE, pooled=False, automatic_top_up=False, preflight_scientific_calls=0)


def confirmation_price_job(job):
    """Reuse the original seven checks with the fixed phase helper."""
    import h46_long_time_native as native
    import h8_long_time_runtime as runtime
    import h8_compiler_phase_bookkeeping as phase
    from qiskit import qpy
    row, endpoint, cohort, index, directory = job
    old_compile = native.compile_cached
    small_gate_cache, phases, reuse = {}, {}, []

    def compile_and_account(wrapper, fingerprint, representative):
        cached = None
        for folder in CONFIG.get('read_only_compiled_caches', []):
            path = Path(folder) / (fingerprint + '.json')
            qpath = Path(folder) / (fingerprint + '_compiled.qpy.gz')
            if path.exists() and (not representative or qpath.exists()):
                record = a.read(path)
                a.require(record['fingerprint'] == fingerprint and record['compiler'] == a.COMPILER, 'REUSED_CACHE_IDENTITY')
                qc = qpy.load(io.BytesIO(gzip.decompress(qpath.read_bytes())))[0] if representative else None
                reuse.append(dict(fingerprint=fingerprint, metrics_sha256=a.sha(path),
                                  raw_qpy_sha256=a.sha(qpath) if representative else None))
                cached = record, qc, True
                break
        record, qc, hit = cached if cached is not None else old_compile(wrapper, fingerprint, representative)
        if representative:
            if qc is None:
                from trotterlib.rte_compiled_cost import transpile_and_measure_cost
                compiled = transpile_and_measure_cost(wrapper, runtime.CONTEXT.compiler,
                                                       actual_circuit_fingerprint=fingerprint)
                a.require(compiled.rz_count == record['metrics']['rz_count'], 'CACHED_RECOMPILE_PRICE')
                qc = compiled.transpiled_circuit
            a.require(qc.count_ops().get('rz', 0) == record['metrics']['rz_count'], 'RAW_COMPILED_RZ')
            proof = phase.certify(wrapper, small_gate_cache, workers=max(1, len(os.sched_getaffinity(0)) // SCOPE['workers']))
            qc, accounting = phase.correct_phase(wrapper, qc, proof)
            a.save(Path(directory) / 'phase' / (fingerprint + '_certificate_PRIVATE.json'), proof)
            stream = io.BytesIO()
            qpy.dump(qc, stream)
            qpath = Path(directory) / 'phase' / (fingerprint + '_corrected.qpy.gz')
            with qpath.open('xb') as output:
                output.write(gzip.compress(stream.getvalue(), compresslevel=1, mtime=0))
            phases[fingerprint] = dict(accounting, corrected_qpy_sha256=a.sha(qpath))
        return record, qc, hit

    native.compile_cached = compile_and_account
    try:
        value = runtime.price_job(job)
        return dict(value, compiler_phase_source=PHASE_SOURCE, compiler_phase_accounting=phases,
                    read_only_cache_reuse=reuse)
    finally:
        native.compile_cached = old_compile


def execute(design, config, scope):
    """Not called by preparation. Requires separate scientific user permission."""
    global CONFIG, SCOPE, OUTPUT
    import h46_long_time_native as native
    import h8_long_time_runtime as runtime
    CONFIG, SCOPE = config, scope
    OUTPUT = Path(config['confirmation_run_root']).resolve() / scope['run_tag']
    a.require(not OUTPUT.exists(), 'FRESH_CONFIRMATION_OUTPUT')
    pr = design['comparison']['PR']
    occupied = set(config['excluded_seed_values'])
    for record in config['stage_B_seed_files']:
        occupied.update(a.read(record['path'])['values'])
    if config['historical_seed_tasks']:
        from pr_cost_native import seed_hierarchy
        for task in config['historical_seed_tasks']:
            occupied.update(seed_hierarchy(task['seed'], task['q']))
    seeds = fresh_seed_map(pr['candidate'], scope['n'], scope['SOURCE'], scope['binding'], scope['run_tag'], occupied)
    OUTPUT.mkdir(parents=True)
    a.save(OUTPUT / 'scope.json', scope)
    a.save(OUTPUT / 'seeds_PRIVATE.json', dict(cohort='CONFIRMATION', values=list(seeds.values())))
    try:
        runtime.CONTEXT = runtime.Context(config, OUTPUT, [pr['candidate']['prefix']], signal=False)
        native.PRIVATE = OUTPUT
        native.trial_seed = lambda row, cohort, index, outer, piece: seeds[(row['id'], index, outer, piece)]
        jobs = [(pr['candidate'], pr['endpoint'], 'CONFIRMATION', i,
                 str(OUTPUT / 'prices' / pr['candidate']['id'] / str(i))) for i in range(scope['n'])]
        trials = runtime.map_jobs(confirmation_price_job, jobs, scope['workers'])
        summary = a.paired_G(trials, pr['endpoint'], 'CONFIRMATION')
        observed = {str(k): sum(t['orders'].get(str(k), 0) for t in trials) for k in (0, 2, 4)}
        summary['rarity'] = native.rarity(trials[0]['laws'], runtime.CONTEXT.preps[pr['candidate']['prefix']],
                                         pr['candidate']['K'], scope['n'], observed)
        comparison = compare_fixed_PF(summary, design['comparison']['PF']['saved_aggregate'], scope['absolute_SE_G_target'])
        a.save(OUTPUT / 'result.json', dict(status='CONFIRMATION_COMPUTATION_COMPLETE', scope=scope,
            trials=trials, PR_summary=summary, fixed_PF=design['comparison']['PF']['saved_aggregate'],
            comparison=comparison, statistical_claim='DESCRIPTIVE_SAMPLE_ONLY', population_guarantee=False))
    except BaseException as exc:
        a.save(OUTPUT / 'HOLD.json', dict(reason=type(exc).__name__ + ':' + str(exc), automatic_retry=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-proposal', type=Path)
    parser.add_argument('--proposal', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--source')
    parser.add_argument('--run-tag', default='H8_M5_P8_Q1_CONFIRMATION_01')
    parser.add_argument('--workers', type=int)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.write_proposal:
        a.require(not args.execute, 'PREPARATION_ONLY')
        a.save(args.write_proposal, published_proposal())
        print(json.dumps(dict(status='PROPOSAL_WRITTEN', scientific_calls=0)))
        return
    design = a.read(args.proposal) if args.proposal else published_proposal()
    validate_design(design)
    if args.config:
        a.require(args.source is not None, 'FIXED_SOURCE_FOR_PREFLIGHT')
        config = a.read(args.config)
        scope = preflight(design, config, args.source, args.run_tag, args.workers)
        if args.execute:
            execute(design, config, scope)
            return
        print(json.dumps(dict(status='PASS_METADATA_ONLY', scope=scope, scientific_calls=0)))
    else:
        a.require(not args.execute, 'CONFIG_AND_SOURCE_REQUIRED_FOR_EXECUTION')
        print(json.dumps(dict(status='PREPARED_NOT_EXECUTED', n=design['n'],
            absolute_SE_G_target=design['absolute_SE_G_target'], scientific_calls=0,
            population_guarantee=False, separate_scientific_permission_required=True)))


if __name__ == '__main__':
    main()
