"""Four approved saved-value shot/necessary-price calculations, no circuits.

Uses the unchanged c640 helper and immutable JSON snapshots. No molecular load,
new signal, price, sampling or full-round optimization is performed.
"""
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'c64007e7a3310bddcc5c65b04463f5f98986f1e8'
SIGNALS = '185104f64a4594355a3813d04d9e50bd4abdb6b5'
C0 = '50ee55bc30f30838e1ac0704bc94f720914640de'
INPUT_SHA = '8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50'
PHASE = dict(PF=.06, RTE=.03, stat=.15, model=.03, implementation=.01, reserve=.02)
SAVED = (
    ('probe_result.json', SIGNALS,
     'artifacts/track_a_o1_m1_attenuation_science_20261106/run01/result.json'),
    ('signal_r4.json', SIGNALS,
     'artifacts/track_a_o1_m1_attenuation_science_20261106/run01/signal_r4.json'),
    ('signal_r8.json', SIGNALS,
     'artifacts/track_a_o1_m1_attenuation_science_20261106/run01/signal_r8.json'),
    ('c0_result.json', C0,
     'artifacts/track_a_o1_m1_c0_parallel_science_20261106/run01/result.json'))
CORE = ('src/trotterlib/o1_m1_cost_feasibility.py',
        'src/trotterlib/o1_m1_resource_bridge.py',
        'src/trotterlib/rpe_resource_accounting.py',
        'src/trotterlib/finite_rte_signal_validation.py',
        'src/trotterlib/rpe_hadamard_interrogation.py',
        'src/trotterlib/rte.py')


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def run(inputs):
    started = datetime.now(timezone.utc).isoformat()
    tick = time.perf_counter()
    hashes = {}
    for path in CORE:
        content = (ROOT/path).read_bytes()
        require(content == git('show', SOURCE+':'+path), 'Fixed SOURCE differs: '+path)
        hashes[path] = hashlib.sha256(content).hexdigest()
    hashes['scripts/run_o1_m1_saved_cost_feasibility.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    snapshots, provenance = {}, []
    for name, commit, path in SAVED:
        content = (inputs/name).read_bytes()
        require(content == git('show', commit+':'+path), 'Saved JSON differs: '+name)
        snapshots[name] = json.loads(content)
        provenance.append(dict(name=name, commit=commit, path=path,
            sha256=hashlib.sha256(content).hexdigest(),
            git_blob=git('rev-parse', commit+':'+path).decode().strip()))
    probe, c0 = snapshots['probe_result.json'], snapshots['c0_result.json']
    require(probe['status'] == 'LIMITED_SIGNAL_PROBE_COMPLETE' and probe['mandatory_stop'] is True,
            'Saved signal probe was not completed')
    require(c0['status'] == 'C0_CONDITIONAL_PILOT_COMPLETE' and c0['mandatory_stop'] is True,
            'Fixed C0 pilot was not completed')
    conditions, profile = probe['conditions'], c0['configuration']
    require((conditions['q'], conditions['delta'], conditions['K'], conditions['r_values'],
             conditions['dictionaries']) == (64, .05, 2, [4, 8], ['P', 'DF']), 'Probe cohort differs')
    require(conditions['state_field'] == profile['state_field'] == 'state'
            and conditions['state_preparation_included'] is False
            and profile['state_preparation_included'] is False, 'Original-state/no-prep condition differs')
    require(conditions['phase_budgets'] == profile['phase_budgets'] == PHASE
            and conditions['beta_RPE'] == profile['beta_RPE'] == .30, 'Fixed C0 phase allocation differs')
    require((profile['delta'], profile['K'], profile['alpha_total'], profile['alpha_axis'],
             profile['q_values'], profile['epsilon_E'], profile['energy_interval'])
            == (.05, 2, .05, .05/14, [1, 2, 4, 8, 16, 32, 64], .1, [-10., 0.]),
            'Fixed C0 error/horizon profile differs')
    require(profile['compiler'] == dict(version='1.3.0', basis=['rz','sx','x','cx'],
            optimization=1, seed=7, backend=None), 'C0 compiler profile differs')
    require(profile['baseline_substeps'] == 4 and profile['relative_fusion'] is True
            and profile['baseline_basis_policy'] == 'exact_spin_lift_spatial_eigh_v1', 'Baseline differs')
    require(profile['identity_policy'] == 'faithful_identity_in_tail', 'Tail identity policy differs')
    p_input, c_input = probe['provenance']['input'], c0['provenance']['input']
    for key in ('commit', 'path', 'sha256'):
        require(p_input[key] == c_input[key], 'Saved physical input binding differs: '+key)
    require(p_input['sha256'] == INPUT_SHA, 'Saved H4 snapshot binding differs')
    require(probe['provenance']['source_commit'] == '8b58cff3679c10aee0dd950874e5e344d0844b7a',
            'Unexpected signal-generating SOURCE')
    require([row['r'] for row in probe['rows']] == [4, 8], 'Missing/duplicated saved rows')

    sys.path.insert(0, str(ROOT/'src'))
    package = types.ModuleType('trotterlib')
    package.__path__ = [str(ROOT/'src/trotterlib')]
    sys.modules['trotterlib'] = package
    from trotterlib.o1_m1_cost_feasibility import account_saved_c0_signal, optimistic_unit_price_ceiling
    from trotterlib.rte import RTEConfig
    from trotterlib.rpe_resource_accounting import RPERoundSpecification
    runtime = {name:importlib.metadata.version(name) for name in
               ('numpy','scipy','openfermion','qiskit','qiskit-nature')}
    require(runtime == probe['provenance']['runtime'], 'Saved-signal runtime differs')
    baseline = c0['total_cost']['strong_DF_PF']
    require(baseline['total_cost'] == 9827008128., 'Fixed all-round baseline budget differs')
    old_contexts = c0['signal_rounds'][6]
    require(old_contexts['q'] == 64 and old_contexts['round_index'] == 6, 'C0 context round differs')
    rows = []
    for row in probe['rows']:
        r = row['r']
        raw = snapshots[f'signal_r{r}.json']
        require(raw['status'] == 'PASS' and raw['checks']
                and all(check['passed'] for check in raw['checks']), 'Saved validator was not PASS')
        rc = raw['conditions']
        require((rc['q'], rc['r'], rc['step_times'], rc['K_values'], rc['full_dimension'])
                == (64, r, [.05], [2], 256), 'Raw signal condition differs')
        require(rc['original_state_field'] == 'state' and rc['sector_projection'] is False
                and rc['identity_policy'] == 'faithful_identity_in_tail', 'Raw physical convention differs')
        require(rc['state_squared_norm'] == c0['original_state']['squared_norm'], 'State norm binding differs')
        require(len(raw['records']) == 1, 'Unexpected raw signal count')
        record = raw['records'][0]
        require((record['q'], record['r'], record['K'], record['delta']) == (64, r, 2, .05),
                'Raw record differs')
        for dictionary in ('P', 'DF'):
            saved_dictionary = record['dictionaries'][dictionary]
            config = RTEConfig(**saved_dictionary['config'])
            require(config.lambda_r == saved_dictionary['lambda_r'], 'Saved lambda differs')
            for raw_key, row_key in (('attenuation','attenuation'),('normalization','normalization_product')):
                require(math.isclose(saved_dictionary[raw_key], row['arms'][dictionary][row_key],
                                     rel_tol=1e-12, abs_tol=0.), 'Saved raw/row normalization differs')
            context = old_contexts['arms'][dictionary]
            require(context['context_hash'] == baseline['context']['context_hash'], 'Physical split context differs')
            account = account_saved_c0_signal(saved_row=row, dictionary=dictionary, config=config,
                specification=RPERoundSpecification(6, .05),
                context_hash=context['context_hash'], arm_hash=context['arm_hash'])
            require(account['feasible'] and not account['reasons'], 'New shot account was not feasible')
            require(account['shots']['cosine'] == account['shots']['sine'],
                    'Equal C0 axis allocations should give equal shot counts')
            ceiling = optimistic_unit_price_ceiling(account, baseline,
                comparison_compiler_hash=baseline['context']['compiler_hash'])
            fraction = Fraction(**ceiling['strict_unit_price_ceiling'])
            rows.append(dict(r=r, dictionary=dictionary, restored_config=saved_dictionary['config'],
                saved_signal_checks=len(raw['checks']), account=account, necessary_price_condition=ceiling,
                price_ceiling_display=float(fraction)))
    return dict(status='FOUR_POINT_SAVED_COST_SCREEN_COMPLETE', mandatory_stop=True,
        science_source=dict(fixed_commit=SOURCE, files_sha256=hashes), saved_inputs=provenance,
        original_snapshot_binding={key:p_input[key] for key in ('commit','path','sha256')},
        profile=profile, baseline=baseline, rows=rows,
        limitation='Necessary unit-price conditions only; no price lower bound or full-candidate total',
        config_scope='Saved numerical normalization config only; not a reconstructed pricing law/cohort',
        full_rpe_improvement=None, impossibility_status='UNASSESSED', price_lower_bound=None,
        guarantee_status='not_certified', price_uncertainty_qualified=False,
        runtime=dict(python=sys.version, packages=runtime, host=platform.node()),
        started_utc=started, finished_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.perf_counter()-tick,
        scope='Four saved q64/r4,r8 P/DF shot and necessary-unit-price calculations only')


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--saved-inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        try:
            result = run(args.saved_inputs)
        except Exception as exc:
            json.dump(dict(status='ERROR_STOP', mandatory_stop=True,
                error_type=type(exc).__name__, error=str(exc)), stream, indent=2)
            stream.write('\n')
            raise
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    for row in result['rows']:
        print(row['dictionary'], 'r='+str(row['r']), 'shots='+str(row['account']['shots']),
              'strict CX ceiling='+repr(row['price_ceiling_display']))
    print(result['status'], 'MANDATORY STOP')


if __name__ == '__main__':
    main()
