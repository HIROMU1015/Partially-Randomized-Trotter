"""Saved H4 full512 evidence -> existing RPE/serial-resource functions.

Import/preflight are passive. Numerical joins occur only in a separately
permitted run. No arrays, QPY, operator, fitter or compiler are decoded.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
import importlib.metadata
from pathlib import Path
import resource
import subprocess
import time

from . import c1_h4_native_fair as prior
from . import c1_h6_ca_resources as reused

io = prior.io
integrated = prior.integrated
accuracy = reused.accuracy
TARGET = dict(reused.TARGET)
COST_SCOPE = reused.COST_SCOPE
CANDIDATES = prior.JOBS
SCOPE = 'C1_H4_CDF8_XDF8_SAVED_CA_RPE_SERIAL_RESOURCES_V1'
FAMILY = 'artifacts/research_c_c1_h4_ca_resources'
LIMITS = dict(saved_bundle_load=1, evidence_attachment=4, round_accuracy=52,
    energy_assessment=4, round_serial_cost=52, cost_aggregation=4, cost_comparison=2)
MAX_CALLS = sum(LIMITS.values())
RECORD_KEYS = {'native', 'native_STOP', 'native_SOURCE', 'native_contract',
    'identity', 'CDF_prepared', 'CDF_PF_s1', 'CDF_PF_s4', 'CDF_STOP', 'XDF', 'XDF_STOP'}
FORBIDDEN = ('new_fit', 'optimizer', 'rank_search', 'DF_recomputation',
    'PF_recomputation', 'synthesis_recomputation', 'state_action', 'build_compile',
    'quantum_sampling', 'retry', 'resume')
NATIVE_SOURCE = '862a97d4213121749448ca716cde820a44b99772'
NATIVE_CONTRACT_SHA = '4a99e76ea879226e077376dd69f12b529f9eb1a48e98dd048143892a59b0e578'
LEVEL = 'B_NUMERICAL_OPERATOR_UPPER_BOUND'


def preflight(root, path):
    """Check pinned bytes/policy/versions; never read saved scientific values."""
    root = Path(root).resolve()
    c = io.read(path)
    if (c['scope'] != SCOPE or c['status'] != 'UNAPPROVED_SCIENCE_CONTRACT'
            or c['conditions'] != prior.CONDITIONS or c['accuracy_target'] != TARGET
            or c['compiler'] != prior.profile_dict() or c['gaussian'] != prior.GAUSSIAN
            or c['compiler_phase'] != prior.PHASE
            or c['cost_scope'] != COST_SCOPE or c['call_limits'] != LIMITS
            or c['maximum_logical_units'] != MAX_CALLS
            or c['candidate_order'] != [f'{a}:s{s}' for a, s in CANDIDATES]
            or c['fresh_output'] != FAMILY + '/<execution-date>/run01'
            or c['native_SOURCE_commit'] != NATIVE_SOURCE
            or c['native_contract_sha256'] != NATIVE_CONTRACT_SHA
            or any(c[k] for k in FORBIDDEN)):
        raise ValueError('H4_CA_RESOURCE_SCOPE')
    manifest = (root / c['source_manifest']).resolve()
    if not manifest.is_relative_to(root) or io.sha(manifest) != c['source_manifest_sha256']:
        raise ValueError('SOURCE_MANIFEST')
    for group in (io.read(manifest)['files'], c['saved_inputs']):
        for name, want in group.items():
            p = (root / name).resolve()
            if not p.is_relative_to(root) or io.sha(p) != want:
                raise ValueError('IDENTITY: ' + name)
    if (set(c['records']) != RECORD_KEYS
            or not set(c['records'].values()) <= c['saved_inputs'].keys()
            or c['native_source_manifest'] not in c['saved_inputs']
            or c['saved_inputs'][c['records']['native_contract']] != NATIVE_CONTRACT_SHA):
        raise ValueError('UNPINNED_RECORD')
    for name, want in c['evaluation_packages'].items():
        if importlib.metadata.version(name) != want:
            raise ValueError('ENVIRONMENT: ' + name)
    return c


def fresh_output(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    parts = output.relative_to(root / FAMILY).parts
    if (len(parts) != 2 or date.fromisoformat(parts[0]).isoformat() != parts[0]
            or parts[1] != 'run01' or output.exists()):
        raise ValueError('FRESH_RUN01_NO_RETRY_RESUME')
    return output


def load_bundle(root, c):
    """Future numerical execution only; reuse native/PF/DF JSON evidence."""
    root = Path(root)
    b = {k: io.read(root / p) for k, p in c['records'].items()}
    old = b['native_contract']
    if (b['native']['status'] != 'H4_FAIR_NATIVE_FULL512_PILOT_COMPLETE'
            or b['native_STOP']['failed'] or not b['native_STOP']['mandatory_stop']
            or b['native_STOP']['scientific_units'] != prior.MAX_CALLS
            or b['native_STOP']['call_counts'] != prior.LIMITS
            or b['native_STOP']['retry'] or b['native_STOP']['resume']
            or b['native_SOURCE']['execution_commit'] != NATIVE_SOURCE
            or b['native_SOURCE']['files'] != io.read(root / c['native_source_manifest'])['files']
            or old['scope'] != prior.SCOPE or old['conditions'] != prior.CONDITIONS
            or old['allowances'] != prior.ALLOWANCES or old['compiler'] != prior.profile_dict()
            or old['gaussian'] != prior.GAUSSIAN or old['compiler_phase'] != prior.PHASE
            or old['synthesis'] != prior.synthesis.VERSION
            or b['identity']['status'] != 'PASS'
            or len(b['native']['local_reports']) != 18
            or set(b['native']['candidates']) != {f'{a}_s{s}' for a, s in CANDIDATES}
            or set(b['native']['shells']) != {'head', 'X', 'Y'}):
        raise ValueError('SAVED_NATIVE_SOURCE_OR_COVERAGE')
    for arm in ('CDF', 'XDF'):
        if b[arm + '_STOP']['failed'] or not b[arm + '_STOP']['mandatory_stop']:
            raise ValueError('SAVED_PF_STOP')
    if (b['CDF_prepared']['state_preparation'] != 'ORACLE_EXCLUDED'
            or b['XDF']['status'] != 'XDF8_STAGE1_COMPLETE'
            or b['XDF']['representation'] != 'PASS'
            or b['XDF']['ground']['status'] != 'PASS'):
        raise ValueError('SAVED_GROUND_ALIAS')
    if accuracy.AccuracyTarget(**c['accuracy_target']).horizon()['maximum_round_index_M'] != 12:
        raise ValueError('REGISTERED_HORIZON')
    return b


def attach_evidence(bundle, arm, s):
    label, key = f'{arm}_s{s}', f'{arm}:s{s}'
    record = bundle['native']['candidates'][label]
    segment = record['segment']
    if (record['arm'] != arm or record['s'] != s or segment['label'] != label
            or segment['input_dimension'] != 512 or segment['output_dimension'] != 512
            or segment['evidence_level'] != LEVEL or segment['floating_point_rounding_enclosed']
            or segment['rigorous_operator_bound'] is not None
            or segment['coverage'] != 'all full-Fock controlled input columns and all output components'):
        raise ValueError('FULL512_CANDIDATE_IDENTITY')
    df = bundle['identity']['DF_bounds_Ha'][arm]
    accuracy._finite(df, 'saved DF norm plus guard')
    saved_df = (bundle['CDF_prepared']['df_error_bounds_ha']['CDF'] if arm == 'CDF'
        else bundle['XDF']['saved_DF']['saved_norm_plus_allowance_Ha'])
    if df != saved_df or df > TARGET['energy_ha'] / 8:
        raise ValueError('SAVED_DF_RESERVE_OR_IDENTITY')
    if arm == 'CDF':
        pf = bundle[f'CDF_PF_s{s}']
        if (pf['candidate'] != dict(arm='CDF', family='D', s=s, r=0, K=0)
                or pf['status'] != 'POTENTIALLY_FEASIBLE'):
            raise ValueError('SAVED_CDF_PF_IDENTITY')
    else:
        pf = bundle['XDF']['candidates'][str(s)]
        if (bundle['XDF']['saved_DF']['status'] != 'PASS'
                or pf['summary']['status'] != 'POTENTIALLY_FEASIBLE'):
            raise ValueError('SAVED_XDF_PF_IDENTITY')
    rows = record['rounds']
    if [x['m'] for x in rows] != list(range(13)) or [x['m'] for x in pf['rounds']] != list(range(13)):
        raise ValueError('REGISTERED_ROUND_ORDER')
    for row, old in zip(rows, pf['rounds'], strict=True):
        if (row['s'] != s or row['occurrences'] != s * (1 << row['m'])
                or row['saved_PF_operator_error'] != old['pf_operator_error']
                or old['attenuation'] != 1 or old['rte_corrected_operator_error'] != 0
                or row.get('rigorous_signal_error_bound') is not None
                or row.get('floating_point_rounding_enclosed', False)):
            raise ValueError('SAVED_PF_SYNTHESIS_IDENTITY')
    metrics = reused.metric_record(record['metrics'], measurement=0)
    shells = {k: reused.metric_record(v, measurement=0 if k == 'head' else 1)
        for k, v in bundle['native']['shell_metrics'].items()}
    if set(shells) != {'head', 'X', 'Y'}:
        raise ValueError('SHELL_METRIC_COVERAGE')
    return dict(id=key, arm=arm, s=s, aliases=[label], df_error_with_guard=df,
        rounds=rows, kernel_metrics=metrics, shell_metrics=shells, evidence_level=LEVEL,
        state_model='own_DF_ground_state_oracle', alias_verified=True)


def source_commit(root):
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()


def run(root, path, output):
    root = Path(root).resolve()
    c = preflight(root, path)
    output = fresh_output(root, output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'candidates').mkdir()
    calls = integrated.Calls(output, LIMITS, MAX_CALLS)
    tick, cpu = time.monotonic(), time.process_time()
    try:
        io.save(output / 'contract.json', c)
        io.save(output / 'SOURCE.json', dict(io.read(root / c['source_manifest']),
            execution_commit=source_commit(root)))
        b = calls('saved_bundle_load', 'pinned H4 scalar evidence only', lambda: load_bundle(root, c))
        packets = {}
        for arm, s in CANDIDATES:
            key = f'{arm}:s{s}'
            packets[key] = calls('evidence_attachment', key, lambda arm=arm, s=s: attach_evidence(b, arm, s))
            io.save(output / 'candidates' / (key.replace(':', '_') + '.packet.json'), packets[key])
        target = accuracy.AccuracyTarget(**c['accuracy_target'])
        # Scalar tasks only: memory estimate uses dimension1, not dense H4 matrices.
        plan = integrated.resource_plan(task_count=4, full_dimension=1)
        plan['BLAS_threads_per_worker'] = None
        plan['policy'] = 'Independent scalar candidates; no BLAS kernels or shared settings changed'
        io.save(output / 'resource_plan.json', plan)

        def evaluate(key):
            packet, plans = packets[key], []
            for row in packet['rounds']:
                value = calls('round_accuracy', f'{key}:m{row["m"]}',
                    lambda row=row: reused.assess_saved_round(target, packet, row))
                plans.append(value)
                integrated.progress(output, key, 'round_accuracy', value)
            energy = calls('energy_assessment', key, lambda: accuracy.assess_energy(target,
                packet['df_error_with_guard'], plans, state_model=packet['state_model'],
                alias_branch_verified=packet['alias_verified']))
            integrated.progress(output, key, 'energy_assessment', energy)
            costs = []
            if energy.get('same_accuracy'):
                for row in plans:
                    pair = calls('round_serial_cost', f'{key}:m{row["m"]}:XY',
                        lambda row=row: reused.serial_round_cost(packet, row['m']))
                    costs.extend(pair)
                    integrated.progress(output, key, 'serial_cost', dict(m=row['m'], axes=pair))
            value = calls('cost_aggregation', key,
                lambda: reused.aggregate_candidate(packet, plans, energy, costs))
            io.save(output / 'candidates' / (key.replace(':', '_') + '.json'), value)
            return key, value

        results = {}
        with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
            futures = {pool.submit(evaluate, k): k for k in packets}
            for f in as_completed(futures):
                try:
                    key, value = f.result()
                    results[key] = value
                except BaseException:
                    calls.stop.set()
            # Drain already-started work before preserving the STOP.
        if calls.stop.is_set():
            raise ValueError('H4_CA_DEPENDENT_STOP')
        comparisons = {}
        for s in (1, 4):
            def compare(s=s):
                a, b = (results[f'{arm}:s{s}'] for arm in ('CDF', 'XDF'))
                return dict(s=s, comparison=accuracy.compare_totals(a['cost'], b['cost']),
                    qualification='conditional numerical pilot only', rigorous_same_accuracy=None,
                    guaranteed_total_resource_superiority=False)
            comparisons[str(s)] = calls('cost_comparison', f's{s}', compare)
        result = dict(status='H4_SAVED_CA_RPE_SERIAL_RESOURCES_COMPLETE', scope=SCOPE,
            candidates={f'{a}:s{s}': results[f'{a}:s{s}'] for a, s in CANDIDATES},
            comparisons=comparisons, scientific_units=sum(calls.counts.values()),
            call_counts=dict(calls.counts), CA_achieved=None, rigorous_enclosure=None,
            guaranteed_total_resource_superiority=False, quantum_shots_drawn=0,
            **{k: 0 for k in FORBIDDEN}, mandatory_stop=True)
        io.save(output / 'result.json', result)
        return result
    except BaseException as error:
        calls.stop.set()
        io.save(output / 'failure.json', dict(error=str(error), first_failure=calls.first_failure))
        raise
    finally:
        io.save(output / 'STOP.json', dict(failed=calls.stop.is_set(),
            scientific_units=sum(calls.counts.values()), call_counts=dict(calls.counts),
            retry=0, resume=0, mandatory_stop=True))
        io.save(output / 'resources.json', dict(wall_seconds=time.monotonic() - tick,
            CPU_seconds=time.process_time() - cpu,
            peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
