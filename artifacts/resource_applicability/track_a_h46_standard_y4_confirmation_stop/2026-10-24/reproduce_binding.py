"""Synthetic reproduction and unapplied candidate; no molecular work or retry.

Usage: python -B reproduce_binding.py --assets <recovered-private-assets>
The failed SOURCE files and all original run files stay unchanged.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

SOURCE = 'a94735946e8cf15ffea5c93ae99000f7ab83d23a'
REL = 'scripts/h46_standard_y4_runtime.py'


def load_module(name, source, filename):
    module = ModuleType(name)
    module.__file__ = filename
    exec(compile(source, filename, 'exec'), module.__dict__)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--assets', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(root / 'scripts'))
    import h46_standard_y4_contract as c
    c.install_paths(args.assets)
    source = subprocess.check_output(['git', 'show', SOURCE + ':' + REL], cwd=root).decode()
    assert (root / REL).read_text() == source
    candidate = source.replace('RT = None\nWARM = False',
        'RT = None\nACTIVE_RUNTIME = None\nWARM = False')
    assert candidate.count('RT.RUNTIME') == 3
    candidate = candidate.replace('RT.RUNTIME', 'ACTIVE_RUNTIME')
    candidate = candidate.replace('    global RT\n', '    global RT, ACTIVE_RUNTIME\n')
    candidate = candidate.replace('    r = RT.get_runtime(cfg)\n',
        '    r = RT.get_runtime(cfg)\n    ACTIVE_RUNTIME = r\n')
    assert 'RT.RUNTIME' not in candidate
    api = ModuleType('synthetic_native_api')
    api.build_partial_native = lambda *a, **k: None
    api.build_native_hadamard_wrapper = lambda *a, **k: None
    parent = ModuleType('trottertracks.resource_applicability')
    parent.ax2b_native_df_v5 = api
    minimal = dict(stage='H4_CONFIRMATION', signal_path='/synthetic/signal')
    with patch.object(c, 'authorize'), patch.dict(sys.modules,
            {'trottertracks.resource_applicability': parent}):
        old = load_module('failed_adapter_fixture', source, REL)
        old_runtime = old.runtime(minimal)
        assert old_runtime is old.RT.get_runtime(minimal)
        assert old.RT.RUNTIME is None  # stale SimpleNamespace copy
        assert old.RT.get_runtime.__globals__['RUNTIME'] is old_runtime
        try:
            old.worker({'id': 'toy'})
        except AttributeError as exc:
            assert str(exc) == "'NoneType' object has no attribute 'runner_namespace'"
        else:
            raise AssertionError('FAILED_SOURCE_DID_NOT_REPRODUCE')
        fixed = load_module('candidate_adapter_fixture', candidate, REL)
        live = fixed.runtime(minimal)
        assert fixed.ACTIVE_RUNTIME is live
        assert fixed.RT.RUNTIME is None
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            task = dict(id='toy', candidate=dict(formula='4th', q=4, method='PR'),
                sample=0, axes=['cosine', 'sine'], stage='H4_CONFIRMATION')
            cfg = dict(minimal, SOURCE='synthetic_source', plan_sha256='synthetic_plan',
                tasks=[task], grant={'synthetic': True}, output=str(p / 'output'),
                ledger=str(p / 'ledger'), marker=str(p / 'CONSUMED.json'),
                systems={'H4': dict(inputs={'synthetic': True})})
            (p / 'output/toy').mkdir(parents=True)
            (p / 'ledger').mkdir()
            c.save(cfg['marker'], dict(state='CONSUMED_NO_RETRY_NO_REFUND', grant=cfg['grant']))
            live.runner_namespace['WORKER_CFG'] = cfg
            live.runner_namespace['WORKER_STOP'] = SimpleNamespace(is_set=lambda: False)
            context = SimpleNamespace(synthetic=True)
            key = (cfg['SOURCE'], 'H4', c.digest(cfg['systems']['H4']['inputs']))
            live.native_namespace['CACHE'][key] = dict(prefixes={3: context})
            live.h46_context = lambda *a: context
            fixed.WARM = True
            assert fixed.guarded_context(cfg, 'H4', 3) is context

            def fake_task(actual_cfg, actual_task, stop):
                assert actual_cfg is cfg and actual_task is task
                for _ in range(12):
                    api.build_partial_native()
                for _ in range(2):
                    api.build_native_hadamard_wrapper()
                return {'id': 'toy', 'synthetic': True}

            live.native.run_task = fake_task
            assert fixed.worker(task) == {'id': 'toy', 'synthetic': True}
            dispatch = c.read(p / 'output/toy/build_dispatch.json')
            assert dispatch['completed'] and dispatch['counts'] == dict(
                partial_native_build=12, hadamard_wrapper_build=2)
            assert len(list((p / 'ledger').glob('*.json'))) == 2
    delta = ''.join(difflib.unified_diff(source.splitlines(keepends=True),
        candidate.splitlines(keepends=True), fromfile='a/' + REL, tofile='b/' + REL))
    delta = ''.join('\n' if line == ' \n' else line for line in delta.splitlines(keepends=True))
    out = Path(__file__).resolve().parent
    # These are new diagnostic artifacts, never replacements for sealed SOURCE.
    patch_path = out / 'binding_fix_UNAPPLIED.patch'
    with patch_path.open('x') as f:
        f.write(delta)
    report = dict(status='SYNTHETIC_REPRODUCTION_AND_CANDIDATE_PASS',
        failed_SOURCE=SOURCE, original_error_reproduced=True,
        real_rebind_lazy_state_exercised=True, concrete_runtime_retained=True,
        synthetic_worker_and_cache_lookup_pass=True,
        synthetic_partial_builds=12, synthetic_wrappers=2,
        scientific_calls=0, old_source_modified=False,
        candidate_applied_to_scientific_SOURCE=False,
        original_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        candidate_sha256=hashlib.sha256(candidate.encode()).hexdigest(),
        patch_sha256=hashlib.sha256(delta.encode()).hexdigest())
    c.save(out / 'binding_reproduction.json', report)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
