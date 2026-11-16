"""Prepare host-local Q3 settings; optionally run existing artificial fixtures.

Never connects to a controller or executes a scientific route. Use a dedicated
fixed common checkout for --worker-root, matching the server2 controller SHA.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return value


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def probe(interpreter):
    code = """import importlib.util,json,sys
print(json.dumps(dict(executable=sys.executable,version=sys.version,
    modules={x:bool(importlib.util.find_spec(x)) for x in
             ['numpy','scipy','psutil','flint','numba']})))
"""
    runtime = json.loads(subprocess.check_output([interpreter, '-B', '-c', code], text=True))
    mem = {}
    if Path('/proc/meminfo').exists():
        mem = {k: int(v.split()[0]) * 1024 for k, v in
               (line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
               if k in ('MemTotal', 'MemAvailable')}
    return dict(platform=platform.platform(),
                wsl='microsoft' in platform.release().lower() or bool(os.environ.get('WSL_DISTRO_NAME')),
                cpu_available=len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count(),
                memory_bytes=mem, disk_free_bytes=shutil.disk_usage(ROOT).free,
                tools={x: bool(shutil.which(x)) for x in ('ssh', 'rsync', 'wsl.exe', 'powershell.exe')},
                runtime=runtime)


def smoke(settings, output):
    """Reuse the existing fixture and executor, without the H4 problem/catalog."""
    sys.path[:0] = [str(ROOT / 'tests'), str(ROOT / 'src')]
    from test_rpe_optimizer_concurrent import fixture, canonical
    from trottertracks import rpe_optimizer_concurrent as C

    def run(folder, slots, workers):
        methods = ['PR_PAULI', 'PR_DF']
        problem, context, routes = fixture(methods=methods, configurations=1, lookahead=2, steps=4)
        assert all(r['kind'] == 'artificial' for rr in routes.values() for r in rr)
        for rr in routes.values():
            for r in rr:
                r['artificial']['sleep_seconds'] = 0.2
                r['artificial']['load'] = dict(chunks=4, units_per_chunk=1000, parallel_rounds=1)
        budgets = {method: 2 for method in methods}
        executor = C.ConcurrentExecutor(ROOT, folder, routes, {}, problem,
            max_concurrent=slots, dp_workers=workers, budgets=budgets, science_approved=True)
        result = C.optimize(ROOT, problem, context=context, executor=executor,
                            methods=methods, max_new=budgets)
        outcomes = {}
        for p in folder.glob('jobs/*/*/OUTCOME.json'):
            v = json.loads(p.read_text())
            outcomes[p.parent.name] = {k: v[k] for k in ('status', 'MSE_lower', 'MSE_upper')}
        return canonical(result), outcomes, executor.metrics()

    output.mkdir(parents=True, exist_ok=False)
    reference, ref_out, ref_metrics = run(output / 'single', 1, 1)
    local, local_out, metrics = run(output / 'configured', settings['slots'], settings['dp_workers'])
    requests = list((output / 'configured').glob('jobs/*/*/REQUEST.json'))
    checks = dict(same_adoption_order_and_stop=reference == local,
                  same_MSE_bounds_and_decisions=bool(ref_out) and ref_out == local_out,
                  no_real_Q3=ref_metrics['new_real_route_launches'] == metrics['new_real_route_launches'] == 0,
                  slots_respected=metrics['peak_active'] <= settings['slots'],
                  dp_workers_forwarded=bool(requests) and all(
                      json.loads(p.read_text())['execution_profile']['dp_workers'] == settings['dp_workers']
                      for p in requests))
    report = dict(scope='SINGLE_LINUX_HOST_ARTIFICIAL_ONLY_NOT_DISTRIBUTED_ACCEPTANCE',
                  checks=checks, passed=all(checks.values()), reference_metrics=ref_metrics, local_metrics=metrics,
                  settings=dict(slots=settings['slots'], dp_workers=settings['dp_workers']))
    write(output / 'CHECKS.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-dir', type=Path, required=True)
    parser.add_argument('--host-id', default='local')
    parser.add_argument('--slots', type=positive, required=True)
    parser.add_argument('--dp-workers', type=positive, required=True)
    parser.add_argument('--interpreter', default=sys.executable)
    parser.add_argument('--worker-root', type=Path, help='fixed common checkout, matching the server2 controller SHA')
    parser.add_argument('--host-dp-override', action='store_true',
                        help='include profile dp_workers; requires the local extension in the matching common SOURCE')
    parser.add_argument('--route-views', type=Path, help='JSON: full commit SHA -> local detached view path')
    parser.add_argument('--smoke-output', type=Path, help='new directory; existing synthetic fixture only')
    args = parser.parse_args()
    state = args.state_dir.expanduser().resolve()
    # Preserve virtualenv entry paths rather than resolving their interpreter symlink.
    interpreter = os.path.abspath(os.path.expanduser(shutil.which(args.interpreter) or args.interpreter))
    views = json.loads(args.route_views.read_text()) if args.route_views else {}
    for sha, value in views.items():
        view = Path(value).expanduser().resolve()
        if len(sha) != 40 or subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=view, text=True).strip() != sha:
            raise ValueError('Route view commit differs: ' + sha)
        if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=view, text=True):
            raise ValueError('Route view has tracked edits: ' + sha)
        views[sha] = str(view)
    state.mkdir(parents=True, exist_ok=True)
    worker_root = (args.worker_root or ROOT).expanduser().resolve()
    settings = dict(host_id=args.host_id, root=str(worker_root), work_dir=str(state / 'spool'),
                    interpreter=interpreter, slots=args.slots, dp_workers=args.dp_workers,
                    route_views=views, prefix_cache_directory=str(state / 'prefix-cache'))
    write(state / 'LOCAL_SETTINGS.json', settings)
    # 89f51a4 inherits REQUEST dp_workers; opt in only after the extension is
    # integrated in the fixed SOURCE used by BOTH controller and workers.
    write(state / 'worker.json', {k: v for k, v in settings.items()
                                if k != 'dp_workers' or args.host_dp_override})
    write(state / 'READINESS.json', probe(interpreter))
    fixture_path = ROOT / 'docs/research/rpe_distributed_q3/fixtures/ARTIFICIAL_REQUEST.json'
    request = json.loads(fixture_path.read_text())
    request['execution_profile'] = dict(dp_workers=args.dp_workers)
    write(state / 'ARTIFICIAL_REQUEST.json', request)
    print('Prepared', state / 'LOCAL_SETTINGS.json')
    print('worker.json uses host dp_workers; matching common extension required.' if args.host_dp_override else
          'worker.json inherits REQUEST dp_workers; LOCAL_SETTINGS dp_workers configures standalone smoke only.')
    print('No controller connection made.')
    if args.smoke_output:
        # Use the selected interpreter, also for child processes of the fixture.
        code = ('import runpy; m=runpy.run_path(' + repr(str(Path(__file__).resolve())) + '); '
                's=m["json"].loads(m["Path"](' + repr(str(state / 'LOCAL_SETTINGS.json')) + ').read_text()); '
                'r=m["smoke"](s,m["Path"](' + repr(str(args.smoke_output.resolve())) + ')); '
                'print(m["json"].dumps(r)); raise SystemExit(0 if r["passed"] else 1)')
        return subprocess.run([interpreter, '-B', '-c', code], cwd=ROOT).returncode
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
