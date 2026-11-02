"""Future bounded H8 D0 one-shot. Default draft contracts cannot execute."""
from __future__ import annotations
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys

def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

g = _load('h8_execution_guard', 'final_pre_review_execution_v1.py')
a = _load('h8_frozen_adapter', 'pre_review_h8_d0_adapter_v1.py')
d0 = a.d0
SCOPE = 'H8_D0_SAVED_VALUE_ONE_SHOT_V1'
SCOPES = ('H8_EXPLORATION92', 'H8_CONFIRMATION_UNION6')


def specs(binding, selected):
    common = dict(compiler_hashes=[binding['compiler']], point_rule='sum_axis_products')
    return {SCOPES[0]: dict(common, primary_phase='exploration'),
        SCOPES[1]: dict(common, primary_phase='confirmation', confirmation_ids=selected)}


def complete_old_map(records, old_map, signals, wrappers, spec, guard):
    expected = {(r['cell_id'], r['epsilon'], r['factor'], metric)
        for r in records if r['status'] == 'DERIVED_SAVED_SCALAR' for metric in d0.METRICS}
    keys = [(r['cell_id'], r['epsilon'], r['allowance_factor'], r['metric']) for r in old_map['rows']]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise g.Stop('COMPLETE_ORIGINAL_MAP_COVERAGE')
    if any(r['status'] == 'MISSING' for r in records):
        raise g.Stop('MISSING_ELIGIBLE_COST')
    result = d0.verify_original(signals, records, old_map)
    by = {(r['cell_id'], r['epsilon'], r['factor']): r for r in records}
    for old in old_map['rows']:
        guard(); new = by[(old['cell_id'], old['epsilon'], old['allowance_factor'])]
        phase = new['sample_phase']
        pairs = d0.paired_rows([r for r in wrappers if r['phase'] == phase], cell_id=new['cell_id'],
            phase=phase, compiler_hash=spec['compiler_hashes'][0])
        for axis, channel in d0.AXIS_CHANNEL.items():
            values = [p[axis][old['metric']] for p in pairs]
            sd = statistics.stdev(values) if len(values) > 1 else None
            # Exact old H6/H8 metric_summary serialization, not a new estimator.
            summary = dict(n=len(values), mean=statistics.mean(values), min=min(values), max=max(values),
                sample_sd=sd, standard_error=sd/len(values)**.5 if sd is not None else None,
                population_mean_certified=False, interval_is_confidence_bound=False,
                shots=new['accounting']['axes'][channel]['shots'])
            if old['axes'][axis] != summary:
                raise g.Stop('ORIGINAL_AXIS_STATISTICS_MISMATCH')
        if (old['method'] != new['method'] or old['phase'] != spec['primary_phase'] or
                old['common_preparation_slope'] != new['N_total'] or old['preparation_cost_P'] != 0 or
                old['formal_winner_certified'] is not False):
            raise g.Stop('ORIGINAL_MAP_METHOD_PHASE_STATUS')
    return result


def verify_old_selection(records, selection):
    union = set()
    for eps in a.ORIGINAL_EPSILONS:
        rows = [r for r in records if r['epsilon'] == eps and r['factor'] == '1.0' and
            r['method'] == 'B2' and r['status'] == 'DERIVED_SAVED_SCALAR']
        rows.sort(key=lambda r: (r['metrics']['RZ']['G_point'], r['cell_id']))
        union.update(r['cell_id'] for r in rows[:2])
    if sorted(union) != sorted(selection['selected_ids']):
        raise g.Stop('ORIGINAL_TOP2_RZ_SELECTION_MISMATCH')


def pipeline(grant, data, monitor, artifacts):
    grant.require(SCOPE)
    signals = d0.strict_json(data['signal_summary.json'])['signals']
    selection = d0.strict_json(data['selection_before_confirmation.json'])
    wrappers = d0.parse_wrappers(data['wrapper_metrics.csv'])
    completion = d0.strict_json(data['completion_audit.json'])
    if grant.synthetic and (completion.get('fixture_kind') != 'ARTIFICIAL' or
            any(s.get('fixture_kind') != 'ARTIFICIAL' for s in signals)):
        raise g.Stop('ARTIFICIAL_FIXTURES_ONLY')
    binding = grant.contract['binding']
    a.validate_inputs(signals, wrappers, selection, binding)
    if (completion['SOURCE'] != binding['source'] or completion['seed_source'] != binding['seed_source'] or
            completion['input_SHA'] != binding['input'] or completion['compiler_hash'] != binding['compiler'] or
            completion['selected_ids'] != selection['selected_ids']):
        raise g.Stop('RESULT_SOURCE_INPUT_COMPILER_SELECTION_BINDING')
    for s in signals:
        d0.finite(s['b'], 'saved_normalization', minimum=0.)
        if s['b'] == 0 or s['numerical_allowance_certified'] is not False:
            raise g.Stop('NORMALIZATION_OR_U_EVIDENCE')
        expected_log = s['cell']['R']*math.log(s['b']) if s['cell']['method'] == 'B2' else 0.
        if s['log_B'] != expected_log or (s['cell']['method'] != 'B2' and s['b'] != 1.):
            raise g.Stop('SAVED_NORMALIZATION_LOG_BINDING')
        for axis in ('real', 'imag'):
            if s['error_decomposition']['total_signed'][axis] != s['signals']['corrected'][axis]-s['reference'][axis]:
                raise g.Stop('SAVED_SIGNED_SIGNAL_BIAS_BINDING')
    phase_rows = {p: [r for r in wrappers if r['phase'] == p] for p in ('exploration', 'confirmation')}
    definitions = specs(binding, selection['selected_ids'])
    equality, original = {}, {}
    monitor.phase = 'A_ORIGINAL_EQUALITY'
    for scope, spec in definitions.items():
        monitor.tick(force=True)
        original[scope] = d0.analyze_scope(signals, phase_rows[spec['primary_phase']], spec,
            a.ORIGINAL_EPSILONS, a.FACTORS, phase_rows['exploration'], monitor.tick)
        old_name = 'exploration_resource_map.json' if scope == SCOPES[0] else 'confirmation_resource_map.json'
        equality[scope] = complete_old_map(original[scope], d0.strict_json(data[old_name]),
            signals, wrappers, spec, monitor.tick)
    verify_old_selection(original[SCOPES[0]], selection)
    cases = sum(x['shot_cases'] for x in equality.values())
    costs = sum(x['cost_cases'] for x in equality.values())
    if cases != grant.contract['expected_old_precision_cases'] or costs != grant.contract['expected_old_map_cases']:
        raise g.Stop('ORIGINAL_TOTAL_COVERAGE')
    artifacts.write('original_equality_v1.json', dict(scopes=equality, precision_cases=cases,
        cost_cases=costs, selection='EXACT_TOP2_RZ_UNION_PASS', status='PASS'), monitor.tick)
    monitor.phase = 'B_NEW_ACCOUNTING'
    all_records, selections, sensitivity, coverage = {}, {}, {}, {}
    for scope, spec in definitions.items():
        records = d0.analyze_scope(signals, phase_rows[spec['primary_phase']], spec,
            a.EPSILONS, a.FACTORS, phase_rows['exploration'], monitor.tick)
        if any(r['status'] == 'MISSING' for r in records):
            raise g.Stop('MISSING_ELIGIBLE_COST')
        all_records[scope] = records; selections[scope] = {}; sensitivity[scope] = {}
        for eps in a.EPSILONS:
            for factor in a.FACTORS:
                subset = [r for r in records if r['epsilon'] == eps and r['factor'] == str(factor)]
                for metric in d0.METRICS:
                    monitor.tick(); key = f'{eps}/{factor}/{metric}'
                    families = d0.family_selection(subset, metric)
                    selections[scope][key] = families
                    lines = [dict(cell_id=r['cell_id'], intercept=r['metrics'][metric]['G_point'],
                        slope=r['N_total']) for r in subset if r['status'] == 'DERIVED_SAVED_SCALAR']
                    fixed = []
                    for baseline in ('B0', 'B1'):
                        ids = [families[x]['RZ_selected_cell'] for x in ('B2', baseline)]
                        if None not in ids:
                            left, right = [next(l for l in lines if l['cell_id'] == cid) for cid in ids]
                            fixed.append(dict(B2=ids[0], baseline=ids[1], crossing=d0.fixed_crossing(left, right)))
                    sensitivity[scope][key] = dict(units=metric+' per shot',
                        fixed_pair_rule='P=0 RZ-selected representatives, no pair reselection',
                        fixed_pairs=fixed, lower_envelope=d0.lower_envelope(lines))
                    if scope == SCOPES[0]:
                        cid = families['B2']['metric_selected_cell']
                        coverage[key] = dict(cell_id=cid, in_original_union6=cid in selection['selected_ids'],
                            status='NO_ELIGIBLE_CANDIDATE' if cid is None else 'COVERED' if
                                cid in selection['selected_ids'] else 'OUTSIDE_ORIGINAL_UNION6')
    monitor.phase = 'C_OUTPUT_AUDIT'
    rare = {phase: [a.unobserved_layer(order, histogram.get(str(order), 0)) for order in (0, 2, 4)]
        for phase, histogram in completion['Taylor_orders'].items()}
    artifacts.write('candidate_accounting_v1.json', all_records, monitor.tick)
    artifacts.write('selection_metrics_v1.json', selections, monitor.tick)
    artifacts.write('preparation_sensitivity_v1.json', sensitivity, monitor.tick)
    artifacts.write('confirmation_coverage_v1.json', coverage, monitor.tick)
    artifacts.write('rare_cost_status_v1.json', dict(saved_observed_order_histograms=completion['Taylor_orders'],
        layers=rare, missing_histogram_orders_mean_zero_observations=True,
        population_mean_cost_bound=None, conditional_sampling_added=False), monitor.tick)
    artifacts.write('provenance_v1.json', dict(SOURCE=grant.contract['SOURCE'],
        science_binding=binding, fixed_D0_SOURCE=a.D0_SOURCE, contract_sha256=grant.contract_sha,
        authorization_sha256=grant.authorization_sha, input_receipts=grant.contract['input_receipts'],
        selection_digest=binding['selection_digest'], profile=grant.contract['profile'],
        carry=grant.contract['carry'], synthetic_only=grant.synthetic, new_scientific_calls=0,
        new_independent_evidence=False, population_winner_certified=False), monitor.tick)


def run(contract, authorization, *, contract_sha, authorization_sha, mapping_loader,
        output, root, synthetic=False, monitor_factory=g.Monitor):
    grant = g.authorization_gate(contract, authorization, contract_sha=contract_sha,
        authorization_sha=authorization_sha, output=output, scope=SCOPE, synthetic=synthetic)
    monitor = None
    artifacts = g.Artifacts(grant)
    first_stop = None
    result = 2
    try:
        monitor = monitor_factory(grant)
        if not synthetic: monitor.install()
        g.runtime_gate(grant, root)
        monitor.phase = 'INPUT_BYTES_SCHEMA'
        data = g.load_inputs(grant, mapping_loader())
        pipeline(grant, data, monitor, artifacts)
        # Recheck bytes after outputs; no numerical pipeline retry.
        g.load_inputs(grant, mapping_loader())
        g.runtime_gate(grant, root)
        monitor.tick(force=True); result = 0
    except Exception as error:
        first_stop = dict(reason=type(error).__name__+':'+str(error)[:1000],
            phase=monitor.phase if monitor else 'PRELAUNCH')
        if monitor is not None and monitor.first_stop is not None: first_stop = monitor.first_stop
    finally:
        if monitor is not None: monitor.close()
        if artifacts is not None:
            try:
                artifacts.write('terminal_v1.json', dict(status='COMPLETE_MANDATORY_STOP' if result == 0 else
                    'FAIL_CLOSED_STOP', first_stop=first_stop, pid=g.identity(),
                    intended_exit_code=result, native_exit_code=None,
                    native_exit_recording='EXTERNAL_WAIT_REQUIRED', scientific_children_started=0,
                    cleanup=dict(artifact_FDs_closed_on_return=True, timer_removed=True, child_wait_reap='NO_CHILDREN'),
                    observations=monitor.observations if monitor else 0,
                    max_interval_seconds=monitor.max_interval if monitor else None,
                    max_observation_seconds=monitor.max_observation if monitor else None, automatic_retry=False,
                    last_resource_observations=getattr(monitor, 'history', []),
                    output_file_receipts=artifacts.receipts,
                    next_stage_authorized=False), terminal=True)
            finally: artifacts.close()
    return result


def record_external_exit(contract, authorization, *, contract_sha, authorization_sha, output, wait_status):
    """Called sequentially by the same shell after it has waited for the driver."""
    grant = g.authorization_gate(contract, authorization, contract_sha=contract_sha,
        authorization_sha=authorization_sha, output=output, scope=SCOPE,
        synthetic=bool(contract.get('synthetic_only')), consumed=True)
    ticket = d0.strict_json(g.read_bytes(g.marker_path(grant.output)))
    if (ticket['contract_sha256'] != contract_sha or ticket['authorization_sha256'] != authorization_sha or
            ticket['SOURCE'] != contract['SOURCE'] or ticket['output'] != str(grant.output) or
            ticket['parent_identity'] != g.identity(os.getppid())):
        raise g.Stop('NATIVE_WAIT_PARENT_OR_RUN_BINDING')
    previous = ticket['identity']; current = g.identity(previous['pid'])
    if current is not None and all(current[k] == previous[k] for k in ('pid', 'start_ticks', 'boot_id', 'uid')):
        raise g.Stop('DRIVER_IDENTITY_STILL_PRESENT')
    if type(wait_status) is not int or not 0 <= wait_status <= 255:
        raise g.Stop('SHELL_NATIVE_WAIT_STATUS')
    terminal_path = grant.output/'terminal_v1.json'
    terminal = d0.strict_json(g.read_bytes(terminal_path)) if terminal_path.exists() else None
    value = dict(shell_wait_status=wait_status, waited_by_same_parent=True,
        original_driver_identity=previous, original_identity_absent=True,
        driver_first_stop=terminal.get('first_stop') if terminal else None,
        terminal_available=terminal is not None, exact_exit_timestamp=None,
        scientific_child_processes=0, reap='FOREGROUND_SHELL_WAIT_RETURNED', automatic_retry=False)
    fd = os.open(grant.output/'native_exit_v1.json', os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        os.write(fd, json.dumps(value, sort_keys=True).encode()+b'\n'); os.fsync(fd)
    finally: os.close(fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('contract', 'authorization', 'inputs', 'output'):
        parser.add_argument('--'+arg, required=True)
    parser.add_argument('--record-wait-status', type=int)
    args = parser.parse_args()
    raw_contract = g.read_bytes(args.contract); raw_auth = g.read_bytes(args.authorization)
    contract = d0.strict_json(raw_contract); authorization = d0.strict_json(raw_auth)
    if contract.get('synthetic_only'):
        raise g.Stop('PRODUCTION_CLI_REJECTS_ARTIFICIAL_GRANTS')
    kwargs = dict(contract_sha=g.sha(raw_contract), authorization_sha=g.sha(raw_auth), output=Path(args.output))
    if args.record_wait_status is not None:
        record_external_exit(contract, authorization, wait_status=args.record_wait_status, **kwargs)
        return 0
    return run(contract, authorization, mapping_loader=lambda: d0.strict_json(g.read_bytes(args.inputs)),
        root=Path(__file__).resolve().parents[1], **kwargs)


if __name__ == '__main__':
    try: sys.exit(main())
    except Exception as error:
        print('PRELAUNCH_STOP:'+type(error).__name__+':'+str(error)[:1000], file=sys.stderr)
        sys.exit(2)
