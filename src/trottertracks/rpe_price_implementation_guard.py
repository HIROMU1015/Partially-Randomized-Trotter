"""B direct entries for C's SOURCE-pinned price implementation guard.

C's guard (GUARD) runs byte-for-byte from its fixed commit; it is not copied
or edited. Its only C dependency, baseline_signal_pipeline, is bound here to
B's live controller (the object doing B's arithmetic), that controller's
pf_pr_rpe_pipeline helpers, and the same pinned D physical adapter C's guard
uses. Nothing here chooses a comparison policy, creates sidecars, changes
N/price arithmetic or qualifies prices, laws, intervals or N.
"""
from functools import lru_cache
import hashlib
from pathlib import Path
import subprocess
import sys
import types

from . import pr_rpe_rmse_controller as B

GUARD = dict(commit='3a8b37f1a282c0c06ac1a363b070f4376e9ff41a',
             path='src/trottertracks/price_implementation_guard.py')
D_ADAPTER = dict(commit='e7dca5f7301c8633b17a59b6a7161331303aef2d',
                 path='src/trottertracks/full_baseline_v2_requests.py')
MODULE = 'src/trottertracks/rpe_price_implementation_guard.py'


def blob(root, commit, path):
    try:
        return subprocess.check_output(['git', 'show', commit + ':' + path], cwd=root, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as error:
        raise ValueError('Pinned SOURCE unresolved: ' + commit + ':' + path) from error


def guard(root):
    """C's pinned guard module, with C's API hook bound to B's live controller."""
    return _guard(str(root))


@lru_cache(maxsize=None)
def _guard(root):
    name = 'trottertracks._b_price_guard_' + hashlib.sha256(root.encode()).hexdigest()[:12]
    package = types.ModuleType(name); package.__path__ = []
    sys.modules[name] = package

    def register(short, module):
        sys.modules[name + '.' + short] = module; setattr(package, short, module)
        return module

    def load(short, locator):
        module = register(short, types.ModuleType(name + '.' + short))
        module.__package__ = name; module.__file__ = str(Path(root) / locator['path'])
        code = blob(root, locator['commit'], locator['path'])
        exec(compile(code, locator['commit'] + ':' + locator['path'], 'exec'), module.__dict__)
        return module

    register('pr_rpe_rmse_controller', B); register('pf_pr_rpe_pipeline', B.p)
    api = types.SimpleNamespace(B=B, D=load('full_baseline_v2_requests', D_ADAPTER))

    def published_api(value):
        if str(value) != root:
            raise ValueError('Price implementation guard is bound to another repository root')
        return api
    shim = register('baseline_signal_pipeline', types.ModuleType(name + '.baseline_signal_pipeline'))
    shim.p, shim.blob, shim.published_api = B.p, blob, published_api
    return load('price_implementation_guard', GUARD)


def check(root, config, prices, *, cohorts, bindings, policy):
    """C's G.check before any B price arithmetic; ValueError on any mismatch."""
    return guard(root).check(str(root), config, prices, cohorts=cohorts, bindings=bindings, policy=policy)


def cost(root, config, n_packet, prices, *, cohorts, price_bindings, comparison_policy,
         price_policy='qualified_expectation'):
    """C's G.cost sequence (check, then unchanged B.cost), keeping the check."""
    checked = check(root, config, prices, cohorts=cohorts, bindings=price_bindings, policy=comparison_policy)
    out = B.cost(config, n_packet, prices, cohorts=cohorts, price_policy=price_policy)
    out['price_implementation_check'] = checked
    return out


def select(configs, evaluations, **options):
    """Unchanged B.select after requiring one common checked policy (C's I.select rule)."""
    policies = [(v.get('price_implementation_check') or {}).get('policy') for v in evaluations]
    if any(v is not None for v in policies):
        if any(v is None or not B.p.same(v, policies[0]) for v in policies):
            raise ValueError('One common SOURCE-bound implementation comparison policy required')
    elif any(v.get('cost') is not None and v.get('N_evidence_kind') != 'SYNTHETIC_ONLY' for v in evaluations):
        raise ValueError('Scientific selection requires implementation comparison policy')
    return B.select(configs, evaluations, **options)
