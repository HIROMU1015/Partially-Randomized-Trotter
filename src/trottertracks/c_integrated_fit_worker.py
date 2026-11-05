"""New execution plumbing for the fixed integrated fit APIs.

No numerical work on import. The parent owns scientific permission, fixed task
selection and the evaluation-environment physical RPC. This module supplies no
retry, resume, alternate initializer or scientific policy.
"""
from dataclasses import asdict
from pathlib import Path
import json
import math
import resource
import time
import traceback

from . import c_integrated_robustness as fit
from . import c_near_transfer as transfer
from . import c1_stage1_execution as io
from .c1_stage3_integrated import Calls

RESTORE_LIMITS = dict(start_generation=1, initialization_codec=1,
    tensor_only_restore=1, restored_physical_assessment=1, restored_export=1)


def _plain(value):
    """Serialize metadata only; raw arrays always go to their own NPZ files."""
    import numpy as np
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        raise ValueError('RAW_ARRAY_MUST_HAVE_SEPARATE_NPZ')
    return value


def _save(path, payload):
    io.save(path, _plain(payload))


def _path(root, name):
    path = (root/name).resolve()
    if not path.is_relative_to(root):
        raise ValueError('TASK_PATH_OUTSIDE_REPOSITORY')
    return path


def _load(path, keys):
    from .c_pr_aware_df_stage_b_runner import load
    return load(path, keys)


def _arrays(path, values):
    from .c_pr_aware_df_stage_b_runner import save_arrays
    save_arrays(path, **values)


def _identity(values):
    return {key: io.array_record(values[key]) for key in ('J', 'U')}


def _inputs(task, loader):
    root = Path(task['root']).resolve()
    input_path = _path(root, task['input_path'])
    expected_input = task.get('input_sha256', task['system']['input_sha256'])
    if expected_input != task['system']['input_sha256']:
        raise ValueError('TASK_SYSTEM_INPUT_IDENTITY')
    if io.sha(input_path) != expected_input:
        raise ValueError('TASK_INPUT_BYTE_IDENTITY')
    packet = loader(input_path, ('g',))
    cold = None
    if task.get('cold_path') is not None:
        cold_path = _path(root, task['cold_path'])
        if task.get('cold_sha256') is not None and io.sha(cold_path) != task['cold_sha256']:
            raise ValueError('TASK_COLD_BYTE_IDENTITY')
        cold = loader(cold_path, ('J', 'U'))
    return packet['g'], cold


def make_objective_specs(task, *, loader=None):
    """One paired objective setup; parent charges the common setup unit once.

    Return both serialized specs and pass each arm its own objective_spec in
    run_cell. This avoids counting or evaluating paired setup twice.
    """
    g, cold = _inputs(task, loader or _load)
    system = transfer.System(**task['system'])
    specs = fit.paired_specs(g, cold['J'], cold['U'], system,
        strength=task['strength'], loss_scale=task['loss_scale'])
    return {name: asdict(spec) for name, spec in specs.items()}


class Evidence:
    def __init__(self, task, limits, call_factory):
        self.task, self.root = task, Path(task['root']).resolve()
        self.output = _path(self.root, task['out'])
        self.output.mkdir(parents=True, exist_ok=False)
        (self.output/'raw').mkdir()
        (self.output/'checkpoints').mkdir()
        _save(self.output/'task.json', task)
        self.call = call_factory(self.output, limits, sum(limits.values()))
        self.sequence, self.checkpoints = 0, {}
        self.tick, self.cpu = time.monotonic(), time.process_time()
        self.result, self.failure = None, None

    def record(self, kind, label, data, arrays):
        self.sequence += 1
        row = dict(sequence=self.sequence, kind=kind, label=label, **_plain(data))
        if arrays is not None:
            if kind == 'checkpoint_export':
                destination = self.output/'checkpoints'/('budget'+str(label)+'.npz')
            else:
                destination = self.output/'raw'/f'{self.sequence:06d}.npz'
            _arrays(destination, arrays)
            row.update(array_path=str(destination.relative_to(self.root)),
                       array_file_sha256=io.sha(destination))
            if kind == 'checkpoint_export':
                row.update(candidate_path=row['array_path'],
                    candidate_sha256=row['array_file_sha256'], array_identity=_identity(arrays))
                physical = data.get('evidence', {}).get('physical')
                if physical is not None:
                    evaluated = physical.get('evaluated_array_identity')
                    if evaluated is not None and evaluated != row['array_identity']:
                        raise ValueError('CHECKPOINT_PHYSICAL_ARRAY_IDENTITY')
                    # Preserve the original assessment binding in evidence and
                    # explicitly bind the byte-identical J/U to its export.
                    row['physical'] = dict(physical,
                        evaluated_candidate_sha256=physical.get('candidate_sha256'),
                        evaluated_candidate_path=physical.get('candidate_path'),
                        candidate_sha256=row['candidate_sha256'], candidate_path=row['candidate_path'],
                        checkpoint_rebinding='same assessed J/U arrays',
                        evaluated_array_identity=row['array_identity'])
                _save(destination.with_suffix('.json'), row)
                self.checkpoints[str(label)] = row
        with (self.output/'trajectory.jsonl').open('a') as stream:
            stream.write(json.dumps(_plain(row), ensure_ascii=False, allow_nan=False)+'\n')

    def physical(self, callback):
        def assess(label, J, U):
            row = callback(label, J, U)
            return dict(row, evaluated_array_identity=_identity(dict(J=J, U=U)))
        return assess

    def finish(self, failure=None):
        self.failure = failure
        failed = failure is not None or self.call.stop.is_set()
        terminal_policy = dict(worker_terminal=True, mandatory_stop=failed,
                               integrated_continuation_allowed=not failed)
        partial = dict(checkpoints=self.checkpoints,
            actual_calls=dict(self.call.counts), scientific_units=sum(self.call.counts.values()),
            result=self.result, failure=failure, retry=0, resume=0, **terminal_policy)
        _save(self.output/'partial.json', partial)
        _save(self.output/'resources.json', dict(wall_seconds=time.monotonic()-self.tick,
            CPU_seconds=time.process_time()-self.cpu,
            peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            thread_CPU_caveat='Process CPU includes local numerical pools; not added across concurrent processes.'))
        _save(self.output/'STOP.json', dict(failed=failed,
            reason='CORRECTNESS_OR_EXECUTION_FAILURE' if failure else 'REGISTERED_TASK_COMPLETE',
            first_failure=self.call.first_failure, actual_calls=dict(self.call.counts),
            scientific_units=sum(self.call.counts.values()), maximum_logical_units=self.call.maximum,
            retry=0, resume=0, **terminal_policy))


def run_restore(task, physical, *, loader=None, initializer=None, restore=None, call_factory=Calls):
    """Exactly one new restored start, maximum five registered logical units.

    task.start is P1/P2 for fixed perturbation of cold_path, or C0 for the
    target's one explicit rank8 initializer. All physical matrices are evaluated
    by the supplied evaluation-environment callback(label,J,U).
    """
    evidence = Evidence(task, RESTORE_LIMITS, call_factory)
    loader, restore = loader or _load, restore or fit.neutral_restore
    failure = None
    restore_scale = {}
    try:
        system = transfer.System(**task['system'])
        def generate():
            g, cold = _inputs(task, loader)
            if task['start'] in ('P1', 'P2'):
                if cold is None:
                    raise ValueError('PERTURBATION_NEEDS_GEOMETRY_C0')
                seed = fit.SEEDS[('P1', 'P2').index(task['start'])]
                J, U, metadata = fit.perturb_initial(cold['J'], cold['U'], seed)
            elif task['start'] == 'C0':
                if initializer is None:
                    from .c1_molecule_extension import factor
                    J, U = factor(g, system.rank, optimize=False)
                else:
                    J, U = initializer(g, system.rank)
                if len(J) != system.rank:
                    raise ValueError('EXACT_INITIALIZATION_RANK_REQUIRED')
                from .c1_h6_initialization import tensor_metrics
                raw_tensor = tensor_metrics(g, J, U)
                scale = raw_tensor['objective_from_tensor']
                if not math.isfinite(scale) or scale <= 0:
                    raise ValueError('TARGET_INITIALIZER_TENSOR_SCALE_UNRESOLVED')
                restore_scale['value'] = scale
                metadata = dict(initialization='explicit_Cholesky_XDF8', optimize=False,
                    raw_tensor=raw_tensor, neutral_restore_loss_scale=scale)
            else:
                raise ValueError('FIXED_RESTORED_START')
            if task['start'] != 'C0':
                restore_scale['value'] = task['loss_scale']
            transfer.validate_arrays(g, J, U, system)
            evidence.record('start_generation', task['start'], metadata, dict(J=J, U=U))
            return g, J, U
        g, J, U = evidence.call('start_generation', task['start'], generate)
        hooks = dict(before=0, after=0)
        def before(fun, x, state):
            hooks['before'] += 1
            value, gradient = fun(x)
            raw_arrays = dict(x=x.copy(), gradient=gradient.copy())
            raw_arrays.update({key: state[key].copy() for key in ('J','U') if key in state})
            evidence.record('restore_initial', task['start'],
                dict(value=float(value), codec=state.get('codec'), evaluation='existing fitter VG'),
                raw_arrays)
        def after(fun, raw):
            hooks['after'] += 1
            evidence.record('restore_raw_terminal', task['start'],
                dict(success=bool(raw.success), status=int(raw.status), message=str(raw.message),
                     fun=float(raw.fun), nit=int(raw.nit), nfev=int(raw.nfev), njev=int(raw.njev),
                     nhev=int(getattr(raw, 'nhev', 0))),
                dict(x=raw.x.copy(), gradient=raw.jac.copy()))
        cold, report = restore(g, J, U, system, loss_scale=restore_scale['value'],
            physical=lambda a,b: evidence.physical(physical)('restored_'+task['start'],a,b),
            before_optimizer=before, after_optimizer=after, call=evidence.call)
        if hooks != dict(before=1, after=1):
            raise ValueError('RESTORE_HOOK_COVERAGE')
        def export():
            path = evidence.output/'restored.npz'
            _arrays(path, cold)
            summary = dict(report, start=task['start'],
                candidate_path=str(path.relative_to(evidence.root)), candidate_sha256=io.sha(path),
                array_identity=_identity(cold), input_sha256=system.input_sha256,
                source_sha256=task.get('source_sha256'), actual_CA_achieved=None,
                hook_counts=hooks, VG_calls_outside_optimizer=dict(initial=1,independent_terminal=1),
                actual_calls=dict(evidence.call.counts), scientific_units=sum(evidence.call.counts.values()),
                worker_terminal=True, mandatory_stop=False, integrated_continuation_allowed=True)
            _save(evidence.output/'result.json', summary)
            return summary
        evidence.result = evidence.call('restored_export', task['start'], export)
        return evidence.result
    except BaseException as error:
        failure = dict(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        evidence.call.stop.set()
        _save(evidence.output/'failure.json', failure)
        raise
    finally:
        evidence.finish(failure)


def run_cell(task, physical, *, loader=None, recipe=None, call_factory=Calls):
    """Run the unchanged integrated recipe API, at most3213logical units.

    objective_spec is required: parent generated the paired specs exactly once
    in its shared objective_setup unit. No extra scientific setup is hidden in
    this per-arm worker. No CLI permission switch or approval proof is added.
    """
    evidence = Evidence(task, fit.call_limits(), call_factory)
    loader, recipe = loader or _load, recipe or fit.recipe_cell
    failure = None
    try:
        g, cold = _inputs(task, loader)
        system = transfer.System(**task['system'])
        from . import c_near_stable as existing
        arm = task['arm']
        if arm not in ('PR', 'ENTRY'):
            raise ValueError('REGISTERED_ARM')
        constructor = existing.near.pr.Objective if arm == 'PR' else existing.near.Objective
        spec = constructor(**task['objective_spec'])
        actual_strength = spec.gamma if arm == 'PR' else spec.strength
        if actual_strength != task['strength'] or spec.loss_scale != task['loss_scale']:
            raise ValueError('TASK_OBJECTIVE_IDENTITY')
        result = recipe(arm, g, cold, spec, system, evidence.physical(physical),
                        evidence.call, evidence.record)
        summary = {key: value for key, value in result.items() if key != 'checkpoints'}
        summary.update(start=task['start'], arm=arm, strength=task['strength'],
            objective_spec=asdict(spec), input_sha256=system.input_sha256,
            source_sha256=task.get('source_sha256'), checkpoints=evidence.checkpoints,
            actual_calls=dict(evidence.call.counts), scientific_units=sum(evidence.call.counts.values()),
            retry=0, resume=0, worker_terminal=True, mandatory_stop=False,
            integrated_continuation_allowed=True)
        _save(evidence.output/'result.json', summary)
        evidence.result = summary
        return summary
    except BaseException as error:
        failure = dict(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        evidence.call.stop.set()
        _save(evidence.output/'failure.json', failure)
        raise
    finally:
        evidence.finish(failure)
