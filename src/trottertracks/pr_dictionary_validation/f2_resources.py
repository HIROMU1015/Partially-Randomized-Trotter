"""Independent per-shot resource contract; no statistical eligibility is issued."""
from pathlib import Path
import importlib.metadata
import json
import resource
import subprocess
import sys
import time
from .authorization import file_sha, _canonical, _under
from .f2_plan import ARMS, LENGTHS, CONTEXT, SAVED_INPUT_HASHES, COMPILER, THRESHOLDS, counts

STAGE = 'F2_RESOURCE'
COMPARISON = 'ACTUAL_PER_SHOT_RESOURCES_ONLY'


def resource_counts():
    result = counts()
    for key in ('shot_qualifications', 'conditional_selector', 'known_four_search'):
        result[key] = 0
    return result


def inspect_packet(packet_dir, source_root):
    """Separate eligibility; never edits or impersonates a qualified F2 packet."""
    packet = Path(packet_dir).resolve(); root = Path(source_root).resolve()
    c = json.loads((packet/'contract.json').read_bytes())
    if (c.get('stage'), c.get('comparison'), c.get('contract_status')) != (
            STAGE, COMPARISON, 'USER_APPROVED_F2_RESOURCE_FINAL'):
        raise PermissionError('Explicit limited resource contract required')
    expected = dict(context=CONTEXT, lengths=LENGTHS, counts=resource_counts(),
                    compiler=COMPILER, thresholds=THRESHOLDS, input_files=SAVED_INPUT_HASHES)
    for key, value in expected.items():
        if c.get(key) != value:
            raise PermissionError('Limited F2 fixed condition mismatch: '+key)
    if c.get('statistics') != dict(status='UNQUALIFIED_OUTSIDE_SCOPE', eta=None,
            alpha=None, gamma=None, N=None, weighted_costs=None, interval_qualified=False):
        raise PermissionError('Statistical certification must remain outside scope')
    inputs = {}
    for name, expected_hash in SAVED_INPUT_HASHES.items():
        data = _under(packet, name).read_bytes()
        if file_sha(data) != expected_hash:
            raise PermissionError('Saved B input mismatch: '+name)
        inputs[name] = data
    bundle = json.loads(inputs['handoff/f1_bundle.json'])
    if {a:len(bundle['arms'][a]) for a in ARMS} != LENGTHS:
        raise PermissionError('Saved dictionary lengths changed')
    if bundle.get('literal_prices') or bundle.get('score_intervals'):
        raise PermissionError('Input-only saved bundle required')
    sources = c.get('source_files', {})
    required = {str(p.relative_to(root)) for base in ('src/trotterlib', 'src/trottertracks')
                for p in (root/base).rglob('*.py')}
    required.add('scripts/run_pr_dictionary_f2_resources_saved.py')
    if not required.issubset(sources):
        raise PermissionError('Incomplete limited F2 SOURCE')
    for name, expected_hash in sources.items():
        if file_sha(_under(root, name).read_bytes()) != expected_hash:
            raise PermissionError('Frozen SOURCE mismatch: '+name)
    if c.get('source_closure_sha256') != file_sha(_canonical(sources)):
        raise PermissionError('SOURCE closure digest mismatch')
    if not c.get('source_commit'):
        raise PermissionError('Published SOURCE commit required')
    return dict(contract=c, inputs=inputs, source_root=root, packet=packet,
                technical_status='LIMITED_SOURCE_INPUT_PLAN_CHECKED')


def verify_packet(packet_dir, source_root, output):
    packet = Path(packet_dir).resolve()
    data = (packet/'authorization.json').read_bytes(); grant = json.loads(data)
    if (grant.get('stage'), grant.get('authority'), grant.get('status')) != (
            STAGE, 'USER_EXPLICIT_F2_RESOURCE', 'APPROVED_BOUND_ONE_SHOT'):
        raise PermissionError('New limited F2 one-shot grant required')
    if grant.get('retry_allowed') is not False or grant.get('mandatory_stop') is not True:
        raise PermissionError('No-retry and mandatory STOP required')
    v = inspect_packet(packet, source_root); c = v['contract']; root = v['source_root']
    for key, value in (
            ('contract_sha256', (packet/'contract.json').read_bytes()),
            ('approval_sha256', (packet/'user_approval.txt').read_bytes()),
            ('source_closure_sha256', _canonical(c['source_files']))):
        if grant.get(key) != file_sha(value):
            raise PermissionError('Limited grant binding mismatch: '+key)
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip() != c['source_commit']:
        raise PermissionError('Wrong frozen limited SOURCE commit')
    if not sys.flags.isolated or sys.prefix == sys.base_prefix or sys.version_info[:3] != (3,11,9):
        raise PermissionError('Dedicated isolated CPython 3.11.9 required')
    if not c.get('dependency_versions'):
        raise PermissionError('Fixed dependencies required')
    for name, version in c['dependency_versions'].items():
        if importlib.metadata.version(name) != version:
            raise PermissionError('Environment mismatch: '+name)
    out = Path(output).resolve(); marker = packet/('consumed-'+file_sha(data))
    if not out.is_relative_to(root/'artifacts/pr_algorithm_improvement_phase2d_f2_resources') or not out.parent.is_dir():
        raise PermissionError('Dedicated limited F2 output parent required')
    if str(out) != grant.get('fresh_output_namespace') or out.exists() or marker.exists():
        raise PermissionError('Fresh output and unused limited grant required')
    return dict(v, output=out, consumption_path=marker, authorization_digest=file_sha(data))


def execute_resources(verified):
    """Measured per-shot costs and numerical gates only; N remains null."""
    if verified['contract'].get('comparison') != COMPARISON:
        raise PermissionError('Resource-only contract required')
    from .f2_execution import measure_f2_circuits, check_source, write_json
    measured = measure_f2_circuits(verified)
    compiled = measured['compiled']; built = measured['built']
    residuals = dict(
        raw_operator=max(r['raw_operator']['max_entry'] for r in built),
        raw_signal=max(r['raw_signal_error'] for r in built),
        compiled_operator=max(r['operator']['max_entry'] for r in compiled),
        compiled_signal=max(r['signal_error'] for r in compiled),
        corrected_mean=max(r['max_entry'] for r in measured['means'].values()))
    result = dict(status='F2_RESOURCE_NUMERICAL_PASS_MANDATORY_STOP', counts=resource_counts(),
        actual_compile_misses=len(measured['groups']),
        compile_aliases=len(built)-len(measured['groups']),
        Phi={a:{axis:{m:str(v) for m,v in values.items()} for axis,values in axes.items()}
             for a,axes in measured['phis'].items()},
        per_circuit_records='compiled/*.json', mean_checks=measured['means'], max_residuals=residuals,
        N=None, weighted_costs=None, statistical_certification='UNQUALIFIED_OUTSIDE_SCOPE',
        comparison_scope=COMPARISON, physical_same_accuracy_claim=False,
        same_accuracy_total_resource_advantage_claim=False,
        dictionary_generation='NOT_REEXECUTED_SAVED_B_INPUT_REUSED', timings=measured['timings'],
        actual_transpile_seconds_sum=sum(r['compile_seconds'] for r in compiled),
        main_process_CPU_seconds=time.process_time()-measured['cpu'],
        worker_CPU_seconds=resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime+
            resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime-
            measured['children'].ru_utime-measured['children'].ru_stime,
        wall_seconds=time.monotonic()-measured['start'],
        peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        quantum_shots_drawn=0, general_algorithm_advantage_claim=False,
        novelty_status='INDEPENDENT_METHOD_DELTA_UNESTABLISHED', next_owner='USER_GPT', mandatory_stop=True)
    check_source(verified)
    write_json(verified['output']/'summary.json', result)
    return result
