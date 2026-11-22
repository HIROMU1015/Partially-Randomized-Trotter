"""Check a real Q3 request on one host without a queue or scientific execution.

Use the fixed worker's validate_assignment and the existing adapter's explicit
dry-check mode. Never construct a worker, submit an assignment, or collect an
old job. Output must be a new directory owned by this plan.
"""
import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback


def profile_helper():
    # Load the transport's pure metadata helper without importing its package:
    # validate_assignment below must still come from the fixed scientific root.
    path = Path(__file__).resolve().parents[1] / 'src/trottertracks/rpe_host_profiles.py'
    spec = importlib.util.spec_from_file_location('_readiness_host_profiles', path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


def select_profile(profile, assignment):
    return profile_helper().select_host_profile(profile, assignment)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assignment', required=True, type=Path)
    parser.add_argument('--profile', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--adapter-root', type=Path, help='Separately fixed adapter checkout; original assignment SOURCE is preserved')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = dict(status='REAL_HOST_DRY_CHECK_MISSING_OR_INVALID_INPUT',
                  scientific_evaluation_calls=0, queue_operations=0,
                  output=str(output), host=platform.node(), python=sys.executable,
                  python_version=sys.version)
    stage = 'READ_PLAN_INPUTS'
    try:
        if hasattr(sys, 'set_int_max_str_digits'):
            sys.set_int_max_str_digits(0)
        profile = json.loads(args.profile.read_text())
        assignment = json.loads(args.assignment.read_text())
        profile = select_profile(profile, assignment)
        if args.adapter_root:
            adapter_root = args.adapter_root.resolve()
            if profile.get('adapter_root') and Path(profile['adapter_root']).resolve() != adapter_root:
                raise ValueError('Adapter root differs from the selected published profile')
            profile['adapter_root'] = str(adapter_root)
            if not profile.get('adapter_manifest'):
                paths = ('scripts/run_rpe_optimizer_q3_job.py', 'src/trottertracks/rpe_dp_prefix_reuse.py')
                for helper in ('src/trottertracks/rpe_q3_evidence.py', 'src/trottertracks/rpe_q3_round_tables.py'):
                    if (adapter_root / helper).exists():
                        paths += (helper,)
                profile['adapter_manifest'] = dict(commit=subprocess.check_output(
                    ['git', 'rev-parse', 'HEAD'], cwd=adapter_root, text=True).strip(),
                    files={name: hashlib.sha256((adapter_root / name).read_bytes()).hexdigest() for name in paths})
        result.update(host_id=profile['host_id'], job_id=assignment['job_id'],
                      source=assignment['source_manifest']['commit'],
                      request_sha256=assignment['request_sha256'],
                      input_hashes=assignment.get('input_hashes', []))
        if assignment['request']['route']['kind'] == 'artificial':
            raise ValueError('This readiness check requires a real input route')
        root = Path(profile['root']).resolve()
        stage = 'VALIDATE_FIXED_SOURCE_INPUTS_AND_HOST_PATHS'
        sys.path.insert(0, str(root / 'src'))
        module = importlib.import_module('trottertracks.rpe_q3_worker')
        if Path(module.__file__).resolve() != root / 'src/trottertracks/rpe_q3_worker.py':
            raise ValueError('Worker validation module is outside fixed SOURCE')
        request = module.validate_assignment(profile, assignment)
        resolved = output / 'RESOLVED_REQUEST.json'
        resolved.write_text(json.dumps(request, indent=2, ensure_ascii=False) + '\n')
        result.update(route=request['route'], execution_profile=request.get('execution_profile'),
                      resolved_request_sha256=hashlib.sha256(resolved.read_bytes()).hexdigest())
        if profile_helper().needs_compact_q3_runtime(request):
            stage = 'COMPACT_Q3_RUNTIME_PRECHECK'
            profile_helper().check_compact_q3_runtime(profile['interpreter'])
            result['compact_q3_sqlite'] = 'PASS'
        if profile.get('runtime_checks'):
            stage = 'FIXED_RUNTIME_AST_PRECHECK'
            result['runtime_precheck'] = profile_helper().check_runtime_profile(profile)
            if result['runtime_precheck']['status'] != 'PASS':
                raise ValueError('Fixed execution runtime AST guard did not pass')
        stage = 'EXISTING_ADAPTER_DRY_CHECK'
        adapter_output = output / 'adapter'
        adapter = profile_helper().validate_adapter_profile(profile)
        adapter_root = Path(adapter['root']) if adapter else root
        command = [profile['interpreter'], '-B',
                   str(adapter_root / 'scripts/run_rpe_optimizer_q3_job.py'),
                   '--request', str(resolved), '--output', str(adapter_output),
                   '--mode', 'dry-check']
        if adapter:
            paths = output / 'ROUTE_VIEWS.json'
            paths.write_text(json.dumps(profile['route_views'], indent=2) + '\n')
            command += ['--route-views', str(paths)]
            result['adapter_source'] = dict(root=adapter['root'], commit=adapter['commit'],
                sha256=adapter['files']['scripts/run_rpe_optimizer_q3_job.py'], manifest=adapter,
                original_assignment_manifest_rewritten=False)
        result['command'] = command
        with (output / 'ADAPTER.log').open('w') as log:
            process = subprocess.run(command, cwd=adapter_root,
                                     env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
                                     stdout=log, stderr=subprocess.STDOUT)
        result['adapter_returncode'] = process.returncode
        dry = adapter_output / 'DRY_CHECK.json'
        if dry.is_file():
            result['adapter_result'] = json.loads(dry.read_text())
        if process.returncode == 0 and result.get('adapter_result', {}).get('status') == 'DRY_CHECK_PASS':
            result['status'] = 'REAL_HOST_DRY_CHECK_PASS'
        else:
            reason = result.get('adapter_result', {}).get('reason')
            result['error'] = reason or (output / 'ADAPTER.log').read_text()[-16000:]
    except Exception as exc:
        result.update(error_type=type(exc).__name__, error=str(exc),
                      traceback=traceback.format_exc())
    result.update(stage=stage, wall_seconds=time.monotonic() - started)
    (output / 'READINESS.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(dict(status=result['status'], output=str(output),
                          job_id=result.get('job_id'), error=result.get('error')), ensure_ascii=False))
    return 0 if result['status'] == 'REAL_HOST_DRY_CHECK_PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
