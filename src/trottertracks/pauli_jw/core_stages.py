"""Same-checkout existing API delegation; no new numerical kernel or loader."""
from dataclasses import asdict, dataclass
from importlib import import_module
from typing import Callable

from .contexts import PipelineError, ScientificExecutionDisabled, StageResult


@dataclass(frozen=True)
class CoreBindings:
    scope: str
    kind: str
    hamiltonian: Callable
    config: Callable
    law: Callable
    trajectory: Callable
    mean: Callable
    account: Callable
    compiler_settings: Callable
    compile_cost: Callable
    seed: Callable | None = None


def bind_existing_core(adapter, context):
    """Bind only this study's APIs. Never call them or start a provider.

    Normal legacy trotterlib initialization still applies transitively. No alias
    loading, global lazy facade, script import or dependency fallback is used.
    """
    paths = adapter.api_paths(context)
    try:
        p = import_module('trotterlib.pauli_signal_adapter')
        r = import_module('trotterlib.rte')
        price = import_module('trotterlib.rte_compiled_cost')
        modules = {}
        callbacks = {}
        for name, path in paths.items():
            if path is None:
                callbacks[name] = None
                continue
            module, attribute = path
            if module not in modules:
                modules[module] = import_module(module)
            try:
                callbacks[name] = getattr(modules[module], attribute)
            except AttributeError as exc:
                raise PipelineError(f'explicit study callback {name} is unavailable: {attribute}') from exc
            if not callable(callbacks[name]):
                raise PipelineError(f'explicit study callback {name} is unavailable')
    except ModuleNotFoundError as exc:
        raise PipelineError(f'legacy core dependency unavailable: {exc.name}; no fallback') from exc
    return CoreBindings(context.scope, 'real_unexecuted', p.PauliHamiltonian,
                        callbacks['config'], callbacks['law'], callbacks['trajectory'],
                        callbacks['mean'], callbacks['account'], r.CompilerSettings,
                        price.transpile_and_measure_cost, callbacks['seed'])


def prepare(core, c, previous):
    p = dict(previous.payload)
    if c.scope == 'h4-hybrid':
        cfg = core.config(c.prefix, c.q, c.r, c.K)
        law = core.law(p['ham'], cfg)
    elif c.scope == 'h6-hybrid':
        cfg = core.config(c.candidate_id)
        law = core.law(p['ham'], c.candidate_id)
    else:
        cfg = core.config(c.r, c.K)
        law = core.law(p['ham'], c.r, c.K)
    p.update(config=cfg, law=law)
    return StageResult('method_law', c.binding, 'SYNTHETIC', {'method': c.method, 'K': c.K,
        'PF_order': c.pf_order, 'effective_law': c.effective_law}, payload=p)


def event(core, c, previous):
    p = dict(previous.payload)
    if c.scope == 'h4-hybrid':
        if not callable(core.seed):
            raise PipelineError('explicit H4 seed callback required')
        seed = core.seed(c.prefix, c.q, c.r, c.K, c.cohort_role, c.trajectory_index)
        if seed != c.seed_value:
            raise PipelineError('seed address/value mismatch')
        body, wrapped, meta = core.trajectory(p['ham'], p['config'], seed=seed)
    elif c.scope == 'h6-hybrid':
        body, wrapped, meta = core.trajectory(p['ham'], c.candidate_id, c.cohort_role, c.trajectory_index)
    else:
        body, wrapped, meta = core.trajectory(p['ham'], c.molecule, c.r, c.K, c.cohort_role, c.trajectory_index)
    if set(wrapped) != {'cosine', 'sine'}:
        raise PipelineError('both original axis wrappers are required; no rewrapping')
    if c.scope != 'h4-hybrid' and meta.get('seed') != c.seed_value:
        raise PipelineError('returned seed does not match supplied cohort address')
    p.update(body=body, wrapped=wrapped, trajectory_metadata=meta)
    return StageResult('event_trajectory', c.binding, 'SYNTHETIC',
                       {'cohort_role': c.cohort_role, 'seed_id': c.seed_id}, payload=p)


def signal(core, c, previous):
    p = dict(previous.payload); data = p['data']
    if c.scope == 'h4-hybrid':
        target = core.mean(p['law'], data.raw_state, data.reference, c.precision_bits)
    elif c.scope == 'h6-hybrid':
        target = core.mean(data.model, data.reference, p['law'], c.candidate_id)
    else:
        target = core.mean(data.model, data.reference, p['law'])
    p['target'] = target
    return StageResult('signal_wrapper', c.binding, 'SYNTHETIC',
                       {'B': None, 'bias': None, 'normalization': None,
                        'axes': c.axes, 'norm_id': c.norm_id}, payload=p)


def account_price(core, c, previous):
    p = previous.payload
    compiler = core.compiler_settings(**asdict(c.compiler))
    prices = {axis: core.compile_cost(p['wrapped'][axis], compiler).rz_count
              for axis in ('cosine', 'sine')}
    if any(v is not None for v in prices.values()):
        raise PipelineError('mock compiled prices must remain unacquired')
    # These original study-specific accounts retain their B/u/rho/exact-integer
    # policies. Do not replace them by a generic or DF fixed-signal formula.
    accounting = core.account(p['target'], prices, c.epsilon_sig, p['law'])
    if any(accounting.get(k) is not None for k in (
            'N', 'N_total', 'N_cos', 'N_sin', 'G', 'G_exact', 'G_guard_range_exact',
            'actual_N', 'actual_G', 'SE')):
        raise PipelineError('mock accounting cannot acquire N/G/SE')
    return StageResult('shot_price', c.binding, 'SYNTHETIC',
                       {'N': None, 'G': None, 'price': None, 'SE': None,
                        'prices': prices, 'scientific_qualification': 'UNQUALIFIED',
                        'rare_probability': None}, 'all science callables are mocks')


def scientific_providers(*args, **kwargs):
    raise ScientificExecutionDisabled('P2 has no scientific provider or resume path')
