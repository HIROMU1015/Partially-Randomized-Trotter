#!/usr/bin/env python3
"""One authorized post-hoc cache replay; never launch the science runner."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).absolute().parents[3]
CONTRACT = 'artifacts/track_b_bf1_read_only_recovery_contract/2026-10-05/contract_v1.json'
EXECUTION_ID = 'bf1-r0-20261005-read-only-v1'
OUTPUT = 'artifacts/track_b_bf1_read_only_recovery/2026-10-05/v1'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args]).decode().strip()


def checked_text(relative):
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts or path.suffix not in ('.py','.md','.json','.jsonl','.txt'):
        raise PermissionError('Only the declared text inputs are allowed')
    return ROOT/path


def main():
    if len(sys.argv) != 1:
        raise PermissionError('Recovery has no input/output/grid override arguments')
    contract_bytes = (ROOT/CONTRACT).read_bytes()
    contract = json.loads(contract_bytes)
    if not contract['read_only_recovery_authorized'] or contract['science_execution_authorized']:
        raise PermissionError('No science execution or draft recovery is permitted')
    if contract['execution_id'] != EXECUTION_ID or contract['output_relative'] != OUTPUT:
        raise PermissionError('Recovery identity changed')
    head = git('rev-parse', 'HEAD')
    if git('branch', '--show-current') != contract['branch'] or git('status', '--porcelain'):
        raise PermissionError('Recovery requires its clean, committed branch')
    if subprocess.check_output(['git','-C',str(ROOT),'show',head+':'+CONTRACT]) != contract_bytes:
        raise PermissionError('Contract is not bound to HEAD')
    identities = []
    for entry in contract['inputs']+contract['sources']:
        path = checked_text(entry['path'])
        raw = path.read_bytes()
        if len(raw) != entry['bytes'] or sha(raw) != entry['sha256']:
            raise PermissionError('Text-content seal changed: '+entry['path'])
        if subprocess.check_output(['git','-C',str(ROOT),'show',head+':'+entry['path']]) != raw:
            raise PermissionError('Text source/input not committed: '+entry['path'])
        if 'original_source_commit' in entry:
            old = subprocess.check_output(['git','-C',str(ROOT),'show',entry['original_source_commit']+':'+entry['path']])
            if entry.get('serialization_cast_only'):
                old = old.replace(
                    b"            score['union_rank'] = 1+sum(v < value for v in values) if score['feasible'] else None\n",
                    b"            # NumPy comparisons can promote the count to int64; JSON needs int.\n"
                    b"            score['union_rank'] = int(1+sum(v < value for v in values)) if score['feasible'] else None\n")
            if old != raw:
                raise PermissionError('Frozen rules changed: '+entry['path'])
        identities.append(entry)
    for name, expected in contract['environment']['packages'].items():
        if importlib.metadata.version(name) != expected:
            raise PermissionError('Numerical environment changed: '+name)
    if sys.version != contract['environment']['python'] or sys.executable != contract['environment']['executable']:
        raise PermissionError('Python environment changed')
    for name, expected in contract['environment']['threads'].items():
        if os.environ.get(name) != expected:
            raise PermissionError('Thread policy changed')
    common = Path(git('rev-parse', '--git-common-dir'))
    if not common.is_absolute():
        common = ROOT/common
    old_marker = common/contract['original_one_shot']['marker_relative_to_git_common']
    old_marker_bytes = old_marker.read_bytes()
    if sha(old_marker_bytes) != contract['original_one_shot']['marker_sha256']:
        raise PermissionError('Original consumed marker changed')
    old_value = json.loads(old_marker_bytes)
    if not old_value['consumed'] or old_value['retry_authorized']:
        raise PermissionError('Original science one-shot policy changed')
    sys.path.insert(0, str(ROOT/'src'))
    from trottertracks.algorithm_codesign.recovery import RecoveryFailure, parse_cells, replay
    from trottertracks.algorithm_codesign.freeze import canonical, write_new
    registry = common/'track-b-bf1-read-only-recovery'
    registry.mkdir(exist_ok=True)
    marker = registry/(EXECUTION_ID+'.json')
    write_new(marker, dict(execution_id=EXECUTION_ID, consumed=True, recovery_source_commit=head,
                          contract_sha256=sha(contract_bytes), science_execution_authorized=False,
                          retry_authorized=False))
    output = ROOT/OUTPUT
    output.mkdir(parents=True, exist_ok=False)
    caps = contract['resource_caps']
    started, cpu = time.monotonic(), time.process_time()
    class RecoveryLimits:
        def check(self):
            if (time.monotonic()-started > caps['wall_seconds'] or
                    time.process_time()-cpu > caps['cpu_seconds'] or
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > caps['rss_bytes']):
                raise RecoveryFailure('RESCUE_RESOURCE_CAP')
    def expired(*args):
        raise RecoveryFailure('RESCUE_WALL_TIME_CAP')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(caps['wall_seconds'])
    # After all imports/preflight, replay may open only explicitly sealed text
    # inputs and its new output. No subprocess or molecular path access.
    allowed_reads = {str(checked_text(e['path']).absolute()) for e in identities}
    allowed_reads.update((str(old_marker.absolute()), str(marker.absolute())))
    target = output/'result.json'
    opens = dict(reads=0, writes=0, denied=0)
    def io_audit(event, arguments):
        if event == 'subprocess.Popen' or event == 'os.system':
            opens['denied'] += 1
            raise RecoveryFailure('RESCUE_FORBIDDEN_SUBPROCESS')
        if event == 'open' and isinstance(arguments[0], (str, bytes, os.PathLike)):
            path = str(Path(os.fsdecode(arguments[0])).absolute())
            mode, flags = arguments[1], arguments[2]
            writing = (isinstance(mode,str) and any(c in mode for c in 'wax+')) or bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT))
            if (writing and path != str(target.absolute())) or (not writing and path not in allowed_reads):
                opens['denied'] += 1
                raise RecoveryFailure('RESCUE_UNDECLARED_FILE_ACCESS', path=path, writing=writing)
            opens['writes' if writing else 'reads'] += 1
    sys.addaudithook(io_audit)
    result = dict(status='BF1_READ_ONLY_RECOVERY_INCOMPLETE', primary_recovered=False,
                  mandatory_stop=True, science_execution_authorized=False, science_retry_authorized=False,
                  automatic_next_stage=None, BF2_authorized=False)
    try:
        original = json.loads((ROOT/contract['original_result_relative']).read_text())
        if (original['source_commit'] != contract['original_science_source_commit'] or
                original['status'] != 'BF1_INCOMPLETE_MANDATORY_STOP_NO_RETRY' or
                original['outcome'] != 'INCONCLUSIVE'):
            raise RecoveryFailure('RESCUE_ORIGINAL_RESULT_IDENTITY_MISMATCH')
        frozen = json.loads((ROOT/contract['domain_relative']).read_text())
        ideal, finite = parse_cells((ROOT/contract['original_cells_relative']).read_text())
        if dict(ideal=len(ideal), finite=len(finite)) != contract['original_cell_counts']:
            raise RecoveryFailure('RESCUE_ORIGINAL_CACHE_INVENTORY_MISMATCH')
        result = replay(original, frozen, ideal, finite, RecoveryLimits())
    except Exception as exc:
        result['failure'] = getattr(exc, 'detail', dict(reason='RESCUE_UNEXPECTED_FAILURE',
                               exception_type=type(exc).__name__, message=str(exc)))
    finally:
        signal.alarm(0)
        preserved = all(sha(checked_text(e['path']).read_bytes()) == e['sha256'] for e in identities)
        marker_unchanged = old_marker.read_bytes() == old_marker_bytes
        if not preserved or not marker_unchanged:
            result.update(status='BF1_READ_ONLY_RECOVERY_INCOMPLETE',
                          failure=dict(reason='RESCUE_ORIGINAL_INPUT_OR_MARKER_CHANGED'))
        result.update(execution_id=EXECUTION_ID, recovery_source_commit=head,
            contract_sha256=sha(contract_bytes), original_science_source_commit=contract['original_science_source_commit'],
            original_result_commit=contract['original_result_commit'],
            input_identities=contract['inputs'], original_inputs_unchanged=preserved,
            sealed_text_inputs_and_sources_unchanged=preserved,
            original_one_shot_marker_unchanged=marker_unchanged,
            recovery_marker_relative_to_git_common=str(marker.relative_to(common)),
            resource_caps=caps, wall_seconds=time.monotonic()-started, cpu_seconds=time.process_time()-cpu,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            runtime_io_audit=opens, forbidden_operation_counters=dict(molecular_inputs=0,
                Hamiltonian_or_state_reconstruction=0, exact_target_recalculation=0,
                science_signal_acquisition=0, new_physical_cells=0, additional_science_candidates=0,
                alternate_optimizer_search=0, trajectories=0, circuits=0, compilations=0, GPU=0))
        if len(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode())+1 > caps['output_bytes']:
            result = {k:v for k,v in result.items() if k not in ('primary_F','primary_reference','searches','coefficients','cross_objectives')}
            result.update(status='BF1_READ_ONLY_RECOVERY_INCOMPLETE', failure=dict(reason='RESCUE_OUTPUT_CAP'))
        write_new(target, result)
        print(result['status'])
        print('MANDATORY STOP: return recovery evidence to user/GPT research review.')
    return 0 if result['status'] == 'BF1_READ_ONLY_RECOVERY_COMPLETE' else 2


if __name__ == '__main__':
    raise SystemExit(main())
