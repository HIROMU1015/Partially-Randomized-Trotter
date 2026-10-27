"""Thin Pauli composer. Only plans and explicitly artificial wiring can run."""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

from .contexts import PauliContext, PipelineError, ScientificExecutionDisabled, StageResult, PipelineResult

STAGES = ('input', 'method_law', 'event_trajectory', 'signal_wrapper', 'shot_price')
UNACQUIRED = ('B', 'bias', 'normalization', 'N', 'N_total', 'G', 'price', 'SE', 'rare_probability')


@dataclass(frozen=True)
class Providers:
    kind: str
    core: object
    prepare: Callable
    event: Callable
    signal: Callable
    account_price: Callable


def compose(c, adapter, *, mode='plan', data=None, providers=None):
    if mode not in ('plan', 'synthetic'):
        raise ScientificExecutionDisabled('scientific/real/resume execution is disabled in P2')
    if not isinstance(c, PauliContext):
        raise PipelineError('an explicit Pauli context is required')
    adapter.validate(c)
    if c.stop_status == 'STOP':
        return PipelineResult(mode, c, (StageResult('input', c.binding, 'STOP', {}, 'no resume'),), 'STOP')
    if mode == 'plan':
        return PipelineResult(mode, c, tuple(StageResult(s, c.binding, 'NOT_EXECUTED') for s in STAGES), 'PLAN_ONLY')
    if (c.input_kind != 'synthetic' or not isinstance(providers, Providers)
            or providers.kind != 'synthetic' or providers.core.kind != 'synthetic'
            or providers.core.scope != c.scope):
        raise ScientificExecutionDisabled('artificial input and study-matched mock providers required')
    for name in ('prepare', 'event', 'signal', 'account_price'):
        if not callable(getattr(providers, name)):
            raise PipelineError(f'explicit {name} callback required')
    completed = []; previous = None
    callbacks = (lambda: adapter.input(c, data, providers.core),
                 lambda: providers.prepare(c, previous), lambda: providers.event(c, previous),
                 lambda: providers.signal(c, previous), lambda: providers.account_price(c, previous))
    for stage, call in zip(STAGES, callbacks, strict=True):
        try:
            out = call()
            if not isinstance(out, StageResult) or out.stage != stage or out.binding != c.binding:
                raise PipelineError('stage/order/SOURCE/input/state/norm/cohort/compiler binding mismatch')
            if out.status not in ('SYNTHETIC', 'MISSING', 'UNQUALIFIED', 'STOP', 'FAILED'):
                raise PipelineError('mock stages cannot claim scientific PASS')
            if not isinstance(out.values, Mapping):
                raise PipelineError('stage record values must be a mapping')
            if any(out.values.get(name) is not None for name in UNACQUIRED):
                raise PipelineError('unacquired science must remain None, including rare-event probability')
            for name in ('effective_law', 'metric', 'cost_scope', 'bit_order', 'term_order_id',
                         'axes', 'normalization_scope', 'state_preparation', 'control', 'phase_scope',
                         'precision_bits', 'cohort_role', 'cohort_id', 'seed_id', 'norm_id',
                         'state_id', 'input_id', 'hamiltonian_id', 'scientific_source', 'representation'):
                if name in out.values and out.values[name] != getattr(c, name):
                    raise PipelineError(f'stage changed {name}')
        except Exception as exc:
            completed.append(StageResult(stage, c.binding, 'FAILED', {}, f'{type(exc).__name__}: {exc}'))
            break
        completed.append(out); previous = out
        if out.status != 'SYNTHETIC':
            break
    qualification = ('SYNTHETIC_END_TO_END_PASS' if len(completed) == len(STAGES)
                     and all(s.status == 'SYNTHETIC' for s in completed) else completed[-1].status)
    return PipelineResult(mode, c, tuple(completed), qualification)


def write_new_result(result, directory):
    try:
        from trotterlib.parallel_validation_executor import atomic_write_json
    except ModuleNotFoundError as exc:
        raise PipelineError(f'legacy writer dependency unavailable: {exc.name}; no directory created') from exc
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=False)
    target = path/'pauli_pipeline_result.json'
    atomic_write_json(target, result.to_dict())
    return target
