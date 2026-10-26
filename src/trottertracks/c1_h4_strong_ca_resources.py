"""Saved B2 CDF/B3 XDF -> existing full512 propagation/RPE/serial costs.

Import/preflight are passive. Only a separately permitted run joins real
saved values. No arrays, circuits, optimizer, or new numerical algorithm.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
import importlib.metadata
import math
from pathlib import Path
import resource
import time

from . import c1_h4_ca_resources as prior
from . import c1_h4_b3 as b3

io, integrated, accuracy, reused = prior.io, prior.integrated, prior.accuracy, prior.reused
synthesis = prior.prior.synthesis
SCOPE = 'C1_H4_B2_CDF_B3_XDF_S1_CA_RPE_SERIAL_V1'
FAMILY = 'artifacts/research_c_c1_h4_strong_ca_resources'
ROUTES = {'CDF': 'B2_CDF_relative', 'XDF': 'B3_XDF_spin_relative'}
CONDITIONS = dict(b3.CONDITIONS, route='B2_CDF_relative versus B3_XDF_spin_relative')
TARGET, COST_SCOPE, LEVEL = prior.TARGET, prior.COST_SCOPE, prior.LEVEL
LIMITS = dict(saved_bundle_load=1, evidence_attachment=2, round_synthesis=26,
    round_accuracy=26, energy_assessment=2, round_serial_cost=26,
    cost_aggregation=2, cost_comparison=1)
MAX_CALLS = sum(LIMITS.values())
FORBIDDEN = ('new_fit', 'optimizer', 'rank_search', 'DF_recomputation',
    'PF_recomputation', 'operator_regeneration', 'array_decode', 'state_action',
    'build_compile', 'quantum_sampling', 's4', 'PR', 'retry', 'resume')
RECORDS = {'strong', 'strong_SOURCE', 'strong_STOP', 'strong_contract',
    'B3', 'B3_SOURCE', 'B3_STOP', 'B3_contract'}
OLD_SOURCE = {'strong': b3.PREVIOUS_SOURCE,
              'B3': 'b40fce5c62c9fff97f68073cbc02182e4bbf2b9b'}


def preflight(root, path):
    """Pin bytes/policies/versions; do not attach saved numerical values."""
    root = Path(root).resolve(); c = io.read(path)
    if (c['scope'] != SCOPE or c['status'] != 'UNAPPROVED_SCIENCE_CONTRACT_DRAFT'
            or c['conditions'] != CONDITIONS or c['accuracy_target'] != TARGET
            or c['candidate_order'] != list(ROUTES.values())
            or c['call_limits'] != LIMITS or c['maximum_logical_units'] != MAX_CALLS
            or c['cost_scope'] != COST_SCOPE or c['synthesis'] != synthesis.VERSION
            or c['compiler'] != prior.prior.profile_dict()
            or c['gaussian'] != prior.prior.GAUSSIAN
            or c['compiler_phase'] != prior.prior.PHASE
            or c['fresh_output'] != FAMILY + '/<execution-date>/run01'
            or any(c[k] for k in FORBIDDEN)):
        raise ValueError('STRONG_CA_SCOPE')
    manifest = (root / c['source_manifest']).resolve()
    if not manifest.is_relative_to(root) or io.sha(manifest) != c['source_manifest_sha256']:
        raise ValueError('SOURCE_MANIFEST')
    for group in (io.read(manifest)['files'], c['saved_inputs']):
        for name, want in group.items():
            p = (root / name).resolve()
            if not p.is_relative_to(root) or io.sha(p) != want:
                raise ValueError('SOURCE_OR_INPUT: ' + name)
    if (set(c['records']) != RECORDS
            or not {*c['records'].values(), c['legacy_contract']} <= c['saved_inputs'].keys()):
        raise ValueError('UNPINNED_RECORD')
    # Reuse the prior scalar loader's byte/version checks, never its runner.
    prior.preflight(root, root / c['legacy_contract'])
    for p, v in c['evaluation_packages'].items():
        if importlib.metadata.version(p) != v: raise ValueError('ENVIRONMENT: ' + p)
    return c


def fresh_output(root, output):
    output = Path(output).resolve()
    parts = output.relative_to(Path(root).resolve() / FAMILY).parts
    if (len(parts) != 2 or date.fromisoformat(parts[0]).isoformat() != parts[0]
            or parts[1] != 'run01' or output.exists()):
        raise ValueError('FRESH_RUN01_NO_RETRY_RESUME')
    return output


def load_bundle(root, c):
    """Future science: saved JSON only; prior identity/DF/PF/alias reused."""
    legacy = io.read(Path(root) / c['legacy_contract'])
    b = prior.load_bundle(root, legacy)
    b.update({k: io.read(Path(root) / p) for k, p in c['records'].items()})
    for key, status, scope, units, limits in [
        ('strong', 'H4_STRONG_BASELINE_S1_NUMERICAL_COMPLETE', b3.strong.SCOPE, 227, b3.strong.LIMITS),
        ('B3', 'H4_B3_S1_NUMERICAL_COMPLETE', b3.SCOPE, 65, b3.LIMITS)]:
        r, stop, source, old = (b[key + suffix] for suffix in ('', '_STOP', '_SOURCE', '_contract'))
        if (r['status'] != status or source['execution_commit'] != OLD_SOURCE[key]
                or stop['failed'] or not stop['mandatory_stop'] or stop['retry'] or stop['resume']
                or stop['scientific_units'] != units or stop['call_counts'] != limits
                or old['scope'] != scope or old['compiler'] != c['compiler']
                or old['gaussian'] != c['gaussian'] or old['compiler_phase'] != c['compiler_phase']
                or old['allowances'] != b3.strong.ALLOWANCES):
            raise ValueError('SAVED_ROUTE_SOURCE_OR_CORRECTNESS: ' + key)
        expected = dict(CONDITIONS, route=(
            'B0 saved; B1 XDF spin; B2 both legacy bases relative fusion'
            if key == 'strong' else b3.ROUTE))
        if old['conditions'] != expected or old['conditions']['substeps'] != [1]:
            raise ValueError('SAVED_SCIENTIFIC_CONDITIONS')
        if source['files'] != io.read(Path(root) / old['source_manifest'])['files']:
            raise ValueError('SAVED_SOURCE_MANIFEST')
    if (b['B3_contract']['previous_contract'] != c['records']['strong_contract']
            or b['B3_contract']['previous_scientific_SOURCE'] != OLD_SOURCE['strong']
            or b['B3_contract']['original_input'] != b['strong_contract']['original_input']
            or b['strong_contract']['original_input'] != b['native_contract']['original_input']
            or b['B3_contract']['shells'] != b['strong_contract']['shells']
            or b['B3_contract']['shells'] != c['shells']
            or b['B3_contract']['reference'] != b['strong_contract']['references']['XDF']
            or b['B3_contract']['ground'] != b['native_contract']['ground_states']['XDF']):
        raise ValueError('SAVED_REFERENCE_SHELL_GROUND_BINDING')
    return b


def segment_ready(segment, label):
    """Validate coverage/qualification; missing bounds stay unresolved."""
    if (segment['label'] != label or segment['input_dimension'] != 512
            or segment['output_dimension'] != 512 or segment['evidence_level'] != LEVEL
            or segment['coverage'] != 'all full-Fock controlled input columns and all output components'
            or segment['global_phase_fit'] or segment['floating_point_rounding_enclosed']
            or segment['rigorous_operator_bound'] is not None):
        raise ValueError('FULL512_SCOPE_OR_QUALIFICATION')
    for k in ('raw_reference', 'compiler', 'compiled_reference'):
        x = segment[k]['max_entry']
        if not math.isfinite(x) or not 0 <= x <= 1e-10:
            raise ValueError('SAVED_COMPONENT_CORRECTNESS')
    values = [segment.get(k) for k in ('operator_bound_pilot', 'norm_upper_pilot',
                                      'compiled_reference_frobenius')]
    if any(x is None for x in values): return False
    if any(not math.isfinite(x) or x < 0 for x in values): raise ValueError('INVALID_BOUND')
    guards = segment.get('norm_guards')
    if guards is None or set(guards) != {'raw', 'compiled', 'reference'}: return False
    norms = [x.get('reference_norm_upper_pilot') for x in guards.values()]
    if any(x is None for x in norms): return False
    if (any(not math.isfinite(x) or x < 1 for x in norms)
            or values[1] < max(norms) or values[0] < values[2]):
        raise ValueError('NORM_GROWTH_OR_FROBENIUS_LOST')
    return True


def attach_evidence(bundle, arm):
    # The prior attachment checks own-ground/alias, DF reserve, PF13 identity,
    # and integer shell metrics. Discard its legacy synthesis rows/metrics.
    packet = prior.attach_evidence(bundle, arm, 1)
    route = ROUTES[arm]
    record = bundle['strong']['candidates'][route] if arm == 'CDF' else bundle['B3']
    if record['route'] != route or record['s'] != 1 or (arm == 'CDF' and record['arm'] != arm):
        raise ValueError('ROUTE_CANDIDATE_IDENTITY')
    for value in record['XY'].values():
        if not math.isfinite(value['error']) or not 0 <= value['error'] <= 2e-10:
            raise ValueError('SAVED_XY_CORRECTNESS')
    shells = bundle['native']['shells']
    if set(shells) != {'head', 'X', 'Y'}: raise ValueError('SHELL_COVERAGE')
    readiness = [segment_ready(record['segment'], route)]
    readiness.extend(segment_ready(shells[k], 'shell:' + k) for k in ('head', 'X', 'Y'))
    pf = [dict(m=r['m'], pf_operator_error=r['saved_PF_operator_error']) for r in packet['rounds']]
    packet.update(id=route, aliases=[route], rounds=[], PF_rounds=pf,
        segment=record['segment'], shells=shells, synthesis_ready=all(readiness),
        kernel_metrics=reused.metric_record(record['metrics'], measurement=0))
    return packet


def propagate_round(packet, pf):
    m = pf['m']
    if type(m) is not int or not 0 <= m <= 12: raise ValueError('REGISTERED_ROUND')
    def unresolved(reason):
        return dict(m=m, s=1, occurrences=1 << m, status='UNRESOLVED_SYNTHESIS',
            reason=reason, saved_PF_operator_error=pf['pf_operator_error'],
            complex_signal_bound_pilot=None, combined_complex_error_pilot=None,
            phase_bias_bound_pilot=None, evidence_level=LEVEL,
            rigorous_signal_error_bound=None, floating_point_rounding_enclosed=False,
            CA_achieved=None, shots=None, total_cost=None)
    if pf['pf_operator_error'] is None: return unresolved('MISSING_SAVED_PF')
    if not packet['synthesis_ready']: return unresolved('MISSING_SAVED_BOUND_OR_NORM_GUARD')
    try:
        return synthesis.propagate(packet['segment'], packet['shells'],
            s=1, m=m, saved_pf_error=pf['pf_operator_error'])
    except OverflowError:
        return unresolved('PROPAGATION_NUMERIC_OVERFLOW')
    except ValueError as error:
        if str(error) == 'PROPAGATION_OVERFLOW': return unresolved(str(error))
        raise


def run(root, path, output):
    root = Path(root).resolve(); c = preflight(root, path)
    output = fresh_output(root, output); output.mkdir(parents=True, exist_ok=False)
    (output / 'candidates').mkdir()
    calls = integrated.Calls(output, LIMITS, MAX_CALLS)
    tick, cpu = time.monotonic(), time.process_time()
    try:
        io.save(output / 'contract.json', c)
        io.save(output / 'SOURCE.json', dict(io.read(root / c['source_manifest']),
            execution_commit=prior.source_commit(root)))
        bundle = calls('saved_bundle_load', 'pinned scalar evidence', lambda: load_bundle(root, c))
        packets = {arm: calls('evidence_attachment', ROUTES[arm],
            lambda arm=arm: attach_evidence(bundle, arm)) for arm in ROUTES}
        for packet in packets.values(): io.save(output / 'candidates' / (packet['id'] + '.packet.json'), packet)
        target = accuracy.AccuracyTarget(**c['accuracy_target'])
        plan = integrated.resource_plan(task_count=2, full_dimension=1)
        plan.update(BLAS_threads_per_worker=None,
            policy='Independent scalar candidate chains; no BLAS or shared settings change')
        io.save(output / 'resource_plan.json', plan)

        def evaluate(arm):
            packet, rows, plans = packets[arm], [], []
            key = packet['id']
            for pf in packet['PF_rounds']:
                row = calls('round_synthesis', f'{key}:m{pf["m"]}', lambda pf=pf: propagate_round(packet, pf))
                rows.append(row); integrated.progress(output, key, 'synthesis', row)
                value = calls('round_accuracy', f'{key}:m{pf["m"]}',
                    lambda row=row: reused.assess_saved_round(target, packet, row))
                plans.append(value); integrated.progress(output, key, 'round_accuracy', value)
            energy = calls('energy_assessment', key, lambda: accuracy.assess_energy(target,
                packet['df_error_with_guard'], plans, state_model=packet['state_model'],
                alias_branch_verified=packet['alias_verified']))
            integrated.progress(output, key, 'energy_assessment', energy)
            costs = []
            if energy.get('same_accuracy'):
                for row in plans:
                    pair = calls('round_serial_cost', f'{key}:m{row["m"]}:XY',
                        lambda row=row: reused.serial_round_cost(packet, row['m']))
                    costs.extend(pair); integrated.progress(output, key, 'serial_cost', dict(m=row['m'], axes=pair))
            value = calls('cost_aggregation', key, lambda: reused.aggregate_candidate(packet, plans, energy, costs))
            value.update(synthesis_rounds=rows, failure_scope='one predeclared candidate; no joint or post-selection guarantee')
            io.save(output / 'candidates' / (key + '.json'), value)
            return arm, value

        results = {}
        with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
            for f in as_completed([pool.submit(evaluate, arm) for arm in ROUTES]):
                try:
                    arm, value = f.result(); results[arm] = value
                except BaseException: calls.stop.set()
        if calls.stop.is_set(): raise ValueError('STRONG_CA_DEPENDENT_STOP')
        comparison = calls('cost_comparison', 'B2 CDF versus B3 XDF',
            lambda: accuracy.compare_totals(results['CDF']['cost'], results['XDF']['cost']))
        comparison.update(qualification='conditional numerical pilot only',
            rigorous_same_accuracy=None, guaranteed_total_resource_superiority=False)
        result = dict(status='H4_STRONG_BASELINE_S1_CA_RESOURCES_COMPLETE', scope=SCOPE,
            candidates={ROUTES[a]: results[a] for a in ROUTES}, comparison=comparison,
            scientific_units=sum(calls.counts.values()), call_counts=dict(calls.counts),
            evidence_level=LEVEL, CA_achieved=None, rigorous_enclosure=None,
            guaranteed_total_resource_superiority=False, **{k: 0 for k in FORBIDDEN}, mandatory_stop=True)
        io.save(output / 'result.json', result); return result
    except BaseException as error:
        calls.stop.set(); io.save(output / 'failure.json', dict(error=str(error), first_failure=calls.first_failure)); raise
    finally:
        io.save(output / 'STOP.json', dict(failed=calls.stop.is_set(),
            scientific_units=sum(calls.counts.values()), call_counts=dict(calls.counts),
            retry=0, resume=0, mandatory_stop=True))
        io.save(output / 'resources.json', dict(wall_seconds=time.monotonic() - tick,
            CPU_seconds=time.process_time() - cpu, peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
