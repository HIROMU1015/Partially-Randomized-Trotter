"""Design from saved pilot metadata only. Cannot sample or compile anything."""
import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value


def build(fixture):
    d = module('confirmation_design', 'src/trotterlib/h46_confirmation_design.py')
    k = module('existing_planning', 'src/trotterlib/higher_pf_readiness.py')
    pilots = fixture['pilots']; resources = fixture['resources']
    if set(pilots) != set(d.EXPANDED) or set(resources) != set(d.EXPANDED):
        raise ValueError('FIXED_SEVEN_PILOTS_REQUIRED')
    specifications = [
        ('observed_gap_power80', {}), ('observed_gap_power90', {'power': .9}),
        ('half_observed_gap_power80', {'effect_scale': .5}),
        ('variance_x2_power80', {'variance_factor': 2}),
        ('variance_x4_power80', {'variance_factor': 4}),
        ('difference_halfwidth_half_gap', {'halfwidth_fraction': .5}),
        ('difference_halfwidth_quarter_gap', {'halfwidth_fraction': .25})]
    cases = []
    for family, ids, pairs in [('MINIMAL3', d.MINIMAL, d.PRIMARY_PAIRS),
                               ('EXPANDED7', d.EXPANDED, d.EXPANDED_PAIRS)]:
        for name, kwargs in specifications:
            item = d.design(pilots, ids, pairs, k.planning_equal_n_difference, **kwargs)
            item.update({'case': family + '_' + name, 'family': family, 'candidates': list(ids)})
            # A separately labeled proposal, not a mathematical minimum:
            # replicate at least the pilot size and also target 1% relative MCSE.
            factor = item['variance_factor_sensitivity']
            item['reporting_guard_allocation_plugin'] = {
                candidate: max(item['endpoint_allocation_plugin'][candidate], pilots[candidate]['pilot_n'],
                               k.planning_n(pilots[candidate]['variance'] * factor, relative_mcse=.01,
                                            reference_mean=pilots[candidate]['mean'])['n_plugin'])
                for candidate in ids}
            item['reporting_guard_policy_UNADOPTED'] = {
                'minimum_fresh_replication': 'at least the saved pilot n8',
                'relative_MCSE_target_plugin': .01,
                'not_a_necessity_or_precision_guarantee': True}
            item['resource_projections'] = {
                kind: [d.resource_projection(item[kind], resources, workers)
                       for workers in fixture['worker_scenarios']]
                for kind in ('equal_allocation_plugin', 'endpoint_allocation_plugin', 'reporting_guard_allocation_plugin')}
            cases.append(item)
    return {'status': 'TECHNICAL_DESIGN_ONLY_NOT_EXECUTION_READY',
            'input_result_commit': fixture['input_result_commit'], 'cases': cases,
            'adopted_case': None, 'adopted_n': None, 'new_trajectories': 0,
            'new_compile': 0, 'new_signal': 0, 'confirmation_started': False,
            'mandatory_STOP': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--pilot', type=Path); parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.execute:
        parser.exit(2, 'EXPLICIT_CONFIRMATION_APPROVAL_AND_NATIVE_ADAPTER_REQUIRED\n')
    if not args.design:
        print(json.dumps({'status': 'DESIGN_ONLY', 'native_execution_supported': False})); return
    if args.pilot is None or args.output is None:
        parser.error('--design requires --pilot and --output')
    result = build(json.loads(args.pilot.read_text()))
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'design_scenarios.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'cases': len(result['cases']), 'confirmation_started': False}))


if __name__ == '__main__': main()
