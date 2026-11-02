"""Fresh optimization-only proposal; passive inspection does not decode NPZ."""
from dataclasses import asdict
from pathlib import Path
import importlib.metadata
import json
import resource
import subprocess
import time
from . import c_u_step_control as algorithm
from . import c1_stage1_execution as io
from .c1_stage3_integrated import Calls

BASE = 'a4d24682e79857f1e97f105cc03448d0e1786ea2'
SCOPE = 'C_PR_REDUCED_U_BOUNDED_ARMIJO_V1'
ENTRY = 'scripts/run_research_c_u_step_control.py'
FAMILY = 'artifacts/research_c_u_step_control/2026-11-03'
FRESH = FAMILY + '/run01'
OBJECTIVE = dict(rank=8, norb=4, ld=2, loss_scale=2.1839632053132894e-9,
                 mass_scale=2.2093719370722074, gamma=.1, epsilon=1e-8)
POINTS = 1 + algorithm.STEP.max_trials * algorithm.STEP.max_updates
LIMITS = dict(saved_load=1, initial_identity=1, physical_context=1,
              phase_A_assessment=1, objective_setup=1, optimization_driver=1,
              inner_J_solve=3 * POINTS, precision_state=2 * POINTS,
              reduced_admission=POINTS, independent_terminal_assessment=64,
              physical_assessment=65, export=1)
MAX_CALLS = sum(LIMITS.values())


def preflight(root, path, *, environment=False, evaluation=False):
    root = Path(root).resolve()
    c = io.read(path)
    for key, want in dict(base_commit=BASE, scope=SCOPE, entry=ENTRY,
                          objective=OBJECTIVE, policy=asdict(algorithm.DEFAULT),
                          step_policy=asdict(algorithm.STEP), DF_budget_Ha=.00019875,
                          fresh_output=FRESH, call_limits=LIMITS,
                          maximum_logical_units=MAX_CALLS, maximum_U_points=POINTS,
                          retry=0, resume=0, mandatory_stop=True).items():
        if c[key] != want:
            raise ValueError('FIXED_U_STEP_CONTROL:' + key)
    manifest_path = root / c['source_manifest']
    if io.sha(manifest_path) != c['source_manifest_sha256']:
        raise ValueError('SOURCE_MANIFEST')
    manifest = io.read(manifest_path)
    inherited = manifest['inherited_manifest']
    if io.sha(root / inherited['path']) != inherited['sha256']:
        raise ValueError('INHERITED_FROZEN_SOURCE_MANIFEST')
    if ENTRY not in manifest['files']:
        raise ValueError('UNPINNED_ENTRY')
    for group in (manifest['files'], io.read(root / inherited['path'])['files'], c['saved_inputs']):
        for name, digest in group.items():
            target = (root / name).resolve()
            if not target.is_relative_to(root) or io.sha(target) != digest:
                raise ValueError('SOURCE_INPUT:' + name)
    if not set(c['input_paths'].values()) <= c['saved_inputs'].keys():
        raise ValueError('UNPINNED_INPUT_ROLE')
    if not evaluation and (root / FRESH).exists():
        raise ValueError('FRESH_U_STEP_CONTROL_NO_RETRY_RESUME')
    if environment:
        for name, version in c['evaluation_packages' if evaluation else 'fit_packages'].items():
            if importlib.metadata.version(name) != version:
                raise ValueError('ENVIRONMENT:' + name)
        if not evaluation:
            dist = importlib.metadata.distribution('ffsim')
            for name, digest in c['ffsim_sources'].items():
                if io.sha(dist.locate_file('ffsim/' + name)) != digest:
                    raise ValueError('FFSIM_SOURCE:' + name)
            algorithm.stable.precision.require_extended()
    return c


class Service:
    """Same JSON pipe as the integrated run, with optimization-only worker."""
    def __init__(self, root, path, c, out):
        self.log = (out / 'evaluation_worker.log').open('x')
        self.process = subprocess.Popen(
            [c['evaluation_python'], '-B', str(root / ENTRY), '--contract', str(path), '--evaluation-worker'],
            cwd=root, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True)

    def request(self, operation, **values):
        if operation not in ('initialize', 'assess'):
            raise ValueError('OPTIMIZATION_ONLY_EVALUATION')
        self.process.stdin.write(json.dumps(dict(operation=operation, **values)) + '\n')
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('EVALUATION_WORKER_EXIT:see evaluation_worker.log')
        response = json.loads(line)
        if not response['ok']:
            raise RuntimeError('EVALUATION_FAIL:' + response['error'])
        return response['value']

    def close(self):
        try:
            if self.process.poll() is None:
                self.process.stdin.write('{"operation":"close"}\n')
                self.process.stdin.flush()
        except BrokenPipeError:
            pass
        code = self.process.wait()
        self.log.close()
        if code:
            raise RuntimeError('EVALUATION_WORKER_NONZERO_EXIT')


def evaluation_worker(root, c, out):
    import sys
    from .c_pr_integrated_evaluation import Evaluation
    context = None
    for line in sys.stdin:
        request = json.loads(line)
        operation = request['operation']
        if operation == 'close':
            break
        try:
            if operation == 'initialize':
                context = Evaluation(root, c, out)
                value = dict(status='READY')
            elif operation == 'assess' and context is not None:
                value = context.assess(request['label'], Path(request['path']))
            else:
                raise ValueError('OPTIMIZATION_ONLY_EVALUATION')
            print(json.dumps(dict(ok=True, value=value), allow_nan=False), flush=True)
        except BaseException as exc:
            print(json.dumps(dict(ok=False, error=repr(exc))), flush=True)
            raise


def run(root, path, *, service_factory=Service):
    root = Path(root).resolve()
    path = Path(path).resolve()
    c = preflight(root, path, environment=True)
    import numpy as np
    from .c_pr_aware_df_stage_b_runner import load, save_arrays
    out = root / FRESH
    out.mkdir(parents=True)
    call = Calls(out, LIMITS, MAX_CALLS)
    tick = time.monotonic()
    service = None
    result = None
    try:
        io.save(out / 'contract.json', c)
        io.save(out / 'SOURCE.json', io.read(root / c['source_manifest']))

        def inputs():
            p = c['input_paths']
            return (load(root / p['input'], ('g',)), load(root / p['initial'], ('J', 'U')),
                    load(root / p['selected'], ('J', 'U')), io.read(root / p['scales']))

        packet, cold, selected, scales = call('saved_load', 'original cold J and fixed selected', inputs)

        def identity():
            source = io.read(root / c['input_paths']['result'])['selected']
            stop = io.read(root / c['input_paths']['STOP'])
            if (source['id'] != c['initial_candidate']['id'] or
                    source['physical']['candidate_sha256'] != c['initial_candidate']['sha256'] or
                    io.sha(root / c['input_paths']['selected']) != c['initial_candidate']['sha256'] or
                    not source['physical']['budget_PASS'] or stop['failed'] or not stop['mandatory_stop']):
                raise ValueError('SELECTED_PROVENANCE')
            shape = (c['objective']['rank'], c['objective']['norb'], c['objective']['norb'])
            if any(x.shape != shape or x.dtype != np.float64 or not np.isfinite(x).all()
                   for x in (cold['J'], selected['J'], selected['U'])):
                raise ValueError('FIXED_INPUT_SHAPE_OR_FINITENESS')
            if (scales['initial_tensor_objective'] != c['objective']['loss_scale'] or
                    scales['initial_smooth_tail_mass'] != c['objective']['mass_scale']):
                raise ValueError('ORIGINAL_OBJECTIVE_SCALES')
            return dict(status='PASS', selected=source['id'],
                        candidate_sha256=c['initial_candidate']['sha256'],
                        cold_J='original initial.npz,not selected J', old_STOP_preserved=True)

        io.save(out / 'input_identity.json', call('initial_identity', 'fixed input and candidate binding', identity))
        service = service_factory(root, path, c, out)
        call('physical_context', 'original Hamiltonian and sector36', lambda: service.request('initialize'))

        def physical(label, J, U):
            target = out / (label + '_candidate.npz')
            save_arrays(target, J=J, U=U)
            return service.request('assess', label=label, path=str(target))

        phase = call('phase_A_assessment', 'saved selected J/U',
                     lambda: physical('phase_A', selected['J'], selected['U']))
        io.save(out / 'phase_A.json', phase)
        if not phase['budget_PASS']:
            result = dict(status='SAVED_SELECTED_DF_BUDGET_NOT_MET', phase_A=phase, trajectory=None)
        else:
            spec = call('objective_setup', 'original PR objective',
                        lambda: algorithm.stable.pr.Objective(**c['objective']))

            def record(kind, label, data, arrays):
                if arrays is not None:
                    save_arrays(out / (kind + '_' + label + '.npz'), **arrays)
                with (out / 'trajectory.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(kind=kind, label=label, **data), allow_nan=False) + '\n')

            trajectory = call('optimization_driver', 'fresh bounded recentered Armijo',
                              lambda: algorithm.trajectory(packet['g'], cold['J'], selected['U'], spec,
                                                           physical, call=call, record=record))
            terminal = trajectory['terminal']
            if terminal is not None:
                save_arrays(out / 'terminal.npz', J=terminal['J'], U=terminal['U'])
            result = dict(status=trajectory['status'], phase_A=phase,
                          trajectory={k: v for k, v in trajectory.items() if k != 'terminal'},
                          terminal_state=None if terminal is None else terminal['state'],
                          terminal_kind='last_physically_qualified_accepted_or_initial',
                          true_reduced_gradient_qualification=None, certified_descent=None)
        result.update(retry=0, resume=0, mandatory_stop=True, PF_RTE=None, shots=None,
                      native=None, total_resources=None, actual_CA_achieved=None)
        call('export', 'optimization-only result', lambda: io.save(out / 'result.json', result))
        return result
    except BaseException as exc:
        call.stop.set()
        io.save(out / 'failure.json', dict(error=repr(exc), first_failure=call.first_failure))
        raise
    finally:
        try:
            if service is not None:
                service.close()
        except BaseException:
            call.stop.set()
            raise
        finally:
            io.save(out / 'STOP.json', dict(failed=call.stop.is_set(), scientific_units=sum(call.counts.values()),
                    call_counts=dict(call.counts), maximum_logical_units=MAX_CALLS,
                    stop_reason='CORRECTNESS_OR_EXECUTION_FAILURE' if call.stop.is_set() else result['status'],
                    retry=0, resume=0, mandatory_stop=True))
            io.save(out / 'resources.json', dict(wall_seconds=time.monotonic() - tick,
                    CPU_seconds=time.process_time(), peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
