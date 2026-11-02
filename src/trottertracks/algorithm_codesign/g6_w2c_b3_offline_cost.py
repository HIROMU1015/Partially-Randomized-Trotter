"""Descriptive artificial costs, separate CPU/wall/memory/I/O coordinates."""
from .g6_w2c_b3_offline_result import Invalid, require, nonnegative, KINDS

SETUP = {'setup.source_check', 'setup.constructor_and_preflight', 'setup.share_identity',
         'setup.axis_identity', 'setup.final_identity'}
OUTPUT = {'save.serialize', 'save.hash', 'save.file_hash', 'save.write', 'save.flush', 'save.finalize'}
OBSERVE = {'observe.memory_baseline', 'observe.memory_final', 'observe.memory'}
ONLINE = {f'online.{k}.{part}' for k in KINDS for part in ('generator_and_audit', 'evidence')}
ONLINE.add('online.EVENT.angle_and_requery')
ALLOWED = SETUP | OUTPUT | OBSERVE | ONLINE | {'correctness.event_invariants'}
SHARED = {'shared.source_check', 'shared.saved_budget_load', 'shared.module_load',
          'shared.worker_dispatch_and_receive'}


def vector(values):
    require(type(values) is list and len(values) == 2, 'CPU_WALL_VECTOR')
    return [nonnegative(x) for x in values]


def partition(rows, reconciliation):
    require(type(rows) is dict and type(reconciliation) is dict, 'PARTITION_REQUIRED')
    for row in rows.values():
        require(set(row) == {'calls', 'cpu_ns', 'wall_ns'}, 'COST_ROW_FIELDS')
        for v in row.values():
            nonnegative(v)
    sums = [sum(v[k] for v in rows.values()) for k in ('cpu_ns', 'wall_ns')]
    outer = vector(reconciliation['outer_total_ns'])
    residual = vector(reconciliation['unattributed_and_observer_residual_ns'])
    require(sums == vector(reconciliation['exclusive_sum_ns']) and
            all(a+b == c for a, b, c in zip(sums, residual, outer)), 'DOUBLE_COUNTED_OR_UNRECONCILED_COST')
    require(reconciliation['unknown_bit_operations'] is None and
            reconciliation['includes_observer_effect'] is True, 'MEASUREMENT_SCOPE')
    return {'exclusive_sum_ns': sums, 'outer_total_ns': outer, 'residual_ns': residual, 'rows': rows}


def subtotal(rows, names):
    selected = [v for k, v in rows.items() if k in names]
    # Absence is not measured zero. Keep missing coordinates explicitly null.
    return [sum(v[k] for v in selected) for k in ('cpu_ns', 'wall_ns')] if selected else None


def cost_pair(pair, observed):
    require(observed['status'] == 'VERIFIED_COMPLETE', 'FAILED_OR_INCOMPLETE_PAIR_EXCLUDED')
    coarse = pair['coarse']
    require(set(coarse) <= ALLOWED, 'UNSUPPORTED_COARSE_STAGE_OR_DUPLICATE_PREPARE_CONSTRUCTOR')
    main = partition(coarse, pair['reconciliation'])
    fine = pair['functions']
    if pair['pass_name'] == 'baseline':
        require(fine is None and pair['fine_reconciliation'] is None, 'BASELINE_FINE_MIXING')
        fine_partition = None
    else:
        fine_partition = partition(fine, pair['fine_reconciliation'])
        require(fine_partition['outer_total_ns'] == main['outer_total_ns'], 'FINE_OUTER_SCOPE_MISMATCH')
    for k in KINDS:
        stage = coarse.get('online.' + k + '.generator_and_audit')
        if observed['counts'][k]:
            require(stage is not None and stage['calls'] == observed['counts'][k] and
                    [stage['cpu_ns'], stage['wall_ns']] == observed['generator_exclusive_ns_by_kind'][k],
                    'RAW_GENERATOR_COST_COUNT_BINDING')
    for field in ('memory', 'memory_baseline'):
        require(type(pair[field]) is dict, 'MEMORY_OBJECT')
        for v in pair[field].values():
            if v is not None:
                nonnegative(v)
    n, accepted = observed['committed_trials'], observed['counts']['EVENT']
    output = subtotal(coarse, OUTPUT)
    online = subtotal(coarse, ONLINE)
    setup = subtotal(coarse, SETUP)
    result = {k: pair[k] for k in ('phase', 'pass_name', 'arm', 'layer')}
    result.update(total_cpu_ns=main['outer_total_ns'][0], wall_ns=main['outer_total_ns'][1],
                  online_cpu_ns=online[0] if online else None, setup_cpu_ns=setup[0] if setup else None,
                  output_cpu_ns=output[0] if output else None, trials=n, accepted=accepted,
                  cpu_per_trial_ns=main['outer_total_ns'][0]/n if n else None,
                  cpu_per_accepted_ns=main['outer_total_ns'][0]/accepted if accepted else None,
                  setup_cpu_wall_ns=setup, online_cpu_wall_ns=online, output_cpu_wall_ns=output,
                  by_outcome={k: subtotal(coarse, {f'online.{k}.generator_and_audit', f'online.{k}.evidence'} )
                              for k in KINDS},
                  generator_audit_cpu_wall_ns=subtotal(coarse, {f'online.{k}.generator_and_audit' for k in KINDS}),
                  angle_requery_cpu_wall_ns=subtotal(coarse, {'online.EVENT.angle_and_requery'}),
                  evidence_cpu_wall_ns=subtotal(coarse, {f'online.{k}.evidence' for k in KINDS}),
                  serialization_cpu_wall_ns=subtotal(coarse, {'save.serialize'}),
                  chain_file_hash_cpu_wall_ns=subtotal(coarse, {'save.hash', 'save.file_hash'}),
                  write_flush_finalize_cpu_wall_ns=subtotal(coarse, {'save.write', 'save.flush', 'save.finalize'}),
                  memory_observation_cpu_wall_ns=subtotal(coarse, OBSERVE),
                  coarse_partition=main, fine_alternative_partition=fine_partition,
                  memory=pair['memory'], memory_baseline=pair['memory_baseline'],
                  periodic_max_observed_RSS_bytes=pair.get('periodic_max_observed_RSS_bytes'),
                  RSS_sampling_is_not_continuous_peak=True, output_bytes=pair['output_bytes'],
                  output_rows=pair['output_rows'], peak_line_bytes=pair['peak_line_bytes'],
                  shared_XY_setup_charged_once=True,
                  full_prepare_included_in_constructor_and_preflight=True)
    return result


def aggregate(report):
    result = dict(schema='b3_offline_cost_v1', artificial_only=True, comparison=[], excluded=[],
                  ranking=None, pure_observer_overhead_certified=False, receipt_self_write_cost_ns=None,
                  RSS_sum_is_unique_physical_memory=False, parent_child_wall_added=False,
                  partitions_added=False, status='INSUFFICIENT_EVIDENCE')
    if report.get('status') != 'VERIFIED_COMPLETE' or report.get('artificial_only') is not True:
        result['excluded'].append('Bundle not verified complete; no comparison cells admitted.')
        return result
    try:
        final = report['summary']
        require(final['final_receipt_write_cost_ns'] is None and
                report['completion_receipt']['receipt_self_write_cost_ns'] is None, 'UNKNOWN_RECEIPT_COST_MUST_REMAIN_NULL')
        shared = final['shared_setup']
        require(set(shared['stages']) <= SHARED, 'UNSUPPORTED_SHARED_STAGE')
        result['parent_shared_separate'] = partition(shared['stages'], shared['reconciliation'])
        r = report['completion_receipt']
        require(set(r['shared_finalization_stages']) <= OUTPUT, 'UNSUPPORTED_FINAL_OUTPUT_STAGE')
        result['shared_summary_output_separate'] = partition(r['shared_finalization_stages'], r['reconciliation'])
        all_pairs = [cost_pair(p, v) for p, v in zip(final['pairs'], report['pairs'])]
        result['correctness_separate'] = [p for p in all_pairs if p['phase'] == 'correctness']
        result['comparison'] = [p for p in all_pairs if p['phase'] == 'profile']
        require(len(result['comparison']) == 16, 'MISSING_COMPARISON_CELL')
        result.update(status='COMPLETE_ARTIFICIAL_DESCRIPTIVE_COSTS',
                      accounting='Pair owns X/Y jointly. Fine is alternative. Parent and final summary output separate; no grand wall/RSS sum.')
    except (Invalid, KeyError, TypeError, ValueError) as exc:
        result.update(status='COST_INTEGRITY_MISMATCH', comparison=[], excluded=[str(exc)])
    return result
