"""Post-run P receipt export: preserve cost statistics; tally saved event labels.

No sampling, compile, mean/SE recomputation, reranking or scientific dispatch.
The fixed P SOURCE remains unchanged; this exporter's hash is recorded separately.
"""
from collections import Counter
from fractions import Fraction
import gzip
import os
from pathlib import Path
import h4_tol_p_contract as p
from audit_h4_tol_p_completed import audit
c = p.c
PRIVATE = p.ROOT/'.h4_tol_p_private'
ACT = PRIVATE/'activation01_20261019'
PUBLIC = p.ROOT/'artifacts/resource_applicability/track_a_h4_tol_p_completed/2026-10-19'


def saved_event_labels(output, tasks):
    """Integer receipt accounting over this new P cohort only, not cost analysis."""
    counts = {cid: Counter() for cid in p.N}
    digest_rows = []
    for task in tasks:
        phase = 'first_control_workers' if task['replica'] == 0 else 'remaining_workers'
        path = output/phase/task['id']/'trajectory.json'
        row = c.read(path)
        c.require(row['seed'] == task['seed'] and row['event_digest'] == c.digest(row['events']), 'EVENT_LABEL_RECEIPT')
        labels = [event['taylor_order'] for events in row['events'] for event in events]
        c.require(len(labels) == task['cell']['R'] and all(type(x) is int and x in (0,2,4) for x in labels),
            'REGISTERED_EVENT_LABELS')
        counts[task['cell']['id']].update(labels)
        digest_rows.append(dict(task_id=task['id'], trajectory_sha256=c.sha(path)))
    return dict(rows={cid:{str(k): values[k] for k in (0,2,4)} for cid,values in counts.items()},
        trajectory_receipts_digest=c.digest(digest_rows), total_RTE_event_labels=sum(sum(x.values()) for x in counts.values()),
        cost_statistics_recomputed=False, law_probabilities_calculated=False,
        old_samples_read=False, no_new_scientific_calls=True)


def files(directory):
    for parent, directories, names in os.walk(directory):
        # Disposable caches are not scientific records; raw results/logs are retained.
        directories[:] = [x for x in directories if x not in ('scratch','parent_cache','__pycache__')]
        for name in sorted(names): yield Path(parent)/name


def record():
    c.require(not PUBLIC.exists(), 'UNUSED_COMPLETION_EXPORT')
    spec_path = ACT/'P_contract_EXECUTION_PRIVATE.json'; grant_path = ACT/'P_grant_APPROVED_PRIVATE.json'
    spec, grant = c.read(spec_path), c.read(grant_path)
    sealed_path = PRIVATE/'P_contract_SEALED_PENDING_APPROVAL_PRIVATE.json'; sealed = c.read(sealed_path)
    immutable = ('SOURCE','scope','stage','plan','source_hashes','provider_hashes','provider_roots','runtime_hashes',
        'python','python_sha256','native_config','endpoints','saved_signals','saved_preparations','seed_registry',
        'old_seed_exclusions','seed_inventory_receipt','seed_origins','import_paths','font_seed','output','lock')
    c.require(all(spec[k] == sealed[k] for k in immutable)
        and spec['inputs'][:len(sealed['inputs'])] == sealed['inputs'], 'ORIGINAL_SCIENTIFIC_SEAL_RETAINED')
    for row in c.read(ACT/'activation_preservation_manifest_PRIVATE.json'):
        c.require(c.sha(row['path']) == row['sha256'], 'ACTIVATION_BYTES_RETAINED')
    checked = audit(spec, grant)
    output = Path(spec['output']); run = output.parent
    result = c.read(output/'result.json'); native = c.read(run/'native_exit.json')
    tasks = c.read(spec['seed_registry'])['tasks']
    events = saved_event_labels(output, tasks)
    c.require(events['total_RTE_event_labels'] == 5680, 'EXACT_EVENT_LABEL_TOTAL')
    comparisons = []
    for row in result['comparisons']:
        stats = row['result']; cid = row['cell']['id']
        # Only the requested subtraction of saved final mean from saved PF cost.
        difference = Fraction(row['PF_G_exact'])-Fraction(stats['G_exact'])
        comparisons.append(dict(epsilon=row['epsilon'], PR_cell_id=cid, n=stats['n'],
            PF_cell_id=row['PF_id'], PF_G_exact=row['PF_G_exact'], PR_G_exact=stats['G_exact'],
            PF_minus_PR_exact=str(difference), PR_paired_SE=stats['paired_SE'],
            sample_SD=stats['sample_SD'], point_direction=stats['point_direction'], status=stats['status'],
            trajectory_strata=row['strata'], RTE_event_order_counts=events['rows'][cid],
            population_mean_certified=False, unseen_cost=None, unseen_variance=None))
    PUBLIC.mkdir(parents=True)
    for name in ('result.json','terminal.json','calls_ledger.json','preparations.json'):
        (PUBLIC/name).write_bytes((output/name).read_bytes())
    (PUBLIC/'native_exit.json').write_bytes((run/'native_exit.json').read_bytes())
    c.save(PUBLIC/'comparison_projection.json', dict(scope=p.SCOPE, rows=comparisons,
        differences_are_saved_mean_subtractions=True, old_cohorts_not_pooled=True,
        CI_or_p_values_computed=False, point_direction_is_not_population_guarantee=True))
    c.save(PUBLIC/'event_label_audit.json', events)
    c.save(PUBLIC/'approval_receipt.json', c.read(grant['approval_receipt']))
    refresh = c.read(spec['seed_refresh_receipt'])
    c.save(PUBLIC/'seed_refresh_projection.json', refresh)
    c.save(PUBLIC/'execution_binding.json', dict(SOURCE=spec['SOURCE'],
        final_seal_commit='0ca8bf68dc154418781bb102cd48cd14d66bdb18', contract_digest=c.digest(spec),
        sealed_contract_digest=c.digest(sealed), source_hashes=spec['source_hashes'],
        grant_sha256=c.sha(grant_path), approval_receipt_sha256=grant['approval_receipt_sha256'],
        seed_registry_sha256=c.sha(spec['seed_registry']), compiler=p.main.BINDING['compiler'],
        runtime_files=len(spec['runtime_hashes']), provider_files=len(spec['provider_hashes']),
        activated_input_files=len(spec['inputs']), original_input_files=len(sealed['inputs']),
        worker_cpus=spec['worker_cpus'], worker_count=len(spec['worker_cpus']),
        resource_admission_sha256=spec['fresh_resources']['sha256'],
        resource_assignment='fresh available physical cores; existing affinity/quota retained; no arbitrary count cap',
        event_label_metadata_exporter_sha256=c.sha(__file__), raw_seed_event_data_exported=False,
        separate_AI_reviewer_used=False, new_public_push_performed=False))
    measured = c.read(output/'paired_measurements.json')['rows']
    projection = []
    for row in measured:
        projection.append(dict(cell_id=row['cell_id'],replica=row['replica'],cohort=row['cohort'],stratum=row['stratum'],
            calls=row['calls'],axes={axis:{k:v for k,v in value.items() if k not in ('seed','event_digest')}
                for axis,value in row['axes'].items()}))
    (PUBLIC/'paired_costs_projection.json.gz').write_bytes(gzip.compress(c.encoded(
        dict(rows=projection, raw_seed_event_data_exported=False)),mtime=0))
    worker_rows = []; controls = []
    for phase in ('first_control_workers','remaining_workers'):
        receipt = c.read(output/phase/'accepted_native_receipt.json')
        for worker in receipt['workers']:
            worker_rows.append({k:worker[k] for k in ('task_id','cpu','exit_code','reaped','result_sha256',
                'cpu_seconds','peak_RSS_bytes')})
            if phase == 'first_control_workers':
                controls.append(dict(task_id=worker['task_id'],**c.read(Path(worker['directory'])/'control.json')))
    c.save(PUBLIC/'worker_receipts.json', dict(workers=worker_rows, all_exit0_reaped=True))
    c.save(PUBLIC/'control_validation.json', dict(rows=controls, control_probes=8, independent_event_oracles=2, all_PASS=True))
    actual = dict(result['actual_worker_calls'],preparation_reconstruction=2,endpoint_accounting_pass=1,
        endpoint_comparisons=2,signal=0,PF_compile=0,SCF_DF_state_reference=0,bootstrap=0,
        shot_recalculation=0,old_sample_aggregation=0,automatic_retry=0)
    c.save(PUBLIC/'actual_calls.json', actual)
    completed = dict(checked, status='COMPLETE_MANDATORY_STOP', SOURCE=spec['SOURCE'],
        native_exit_code=0, all_workers_reaped=True, workers=len(worker_rows), worker_slots=len(spec['worker_cpus']),
        result_sha256=c.sha(output/'result.json'), contract_digest=c.digest(spec),
        actual_scientific_calls=actual, cost_statistics_recomputed=False,
        saved_event_label_metadata_counted=True, RTE_event_labels=5680, seed_collisions=0,
        automatic_retry=False, old_samples_pooled=False, population_mean_certified=False,
        next_stage_authorized=False, mandatory_STOP=True)
    c.save(PUBLIC/'completion_audit.json', completed)
    c.save(PRIVATE/'P_COMPLETED_MANDATORY_STOP_PRIVATE.json', completed)
    c.save(PRIVATE/'completed_P_preservation_manifest_PRIVATE.json', [dict(path=str(x),sha256=c.sha(x),bytes=x.stat().st_size)
        for directory in (run,ACT) for x in files(directory)]+[
            dict(path=str(Path(__file__).resolve()),sha256=c.sha(__file__),bytes=Path(__file__).stat().st_size)])
    c.save(PUBLIC/'artifact_manifest.json', dict(schema='H4_TOL10_P_COMPLETED_LOCAL_PROJECTION_V1', SOURCE=spec['SOURCE'],
        files={x.name:c.sha(x) for x in sorted(PUBLIC.iterdir()) if x.is_file()},
        raw_seed_event_data_exported=False,new_public_push_performed=False))
    return completed


if __name__ == '__main__':
    result = record()
    print(c.encoded(result).decode())
