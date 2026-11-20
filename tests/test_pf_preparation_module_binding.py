"""Exercise the PF wrapper's actual child launch assembly with synthetic leaves.

The provider and child process are synthetic. No H4 inputs, signal, circuit
or compiler is used.
"""
import importlib.util
import hashlib
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import trottertracks


def load_wrapper():
    spec = importlib.util.spec_from_file_location('_pf_module_binding_wrapper', ROOT / 'scripts/run_pf_preparation_stage.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_provider(root):
    scripts = root / 'scripts'
    scripts.mkdir()
    (scripts / 'prepare_pf_candidate.py').write_text('''
from pathlib import Path
import json
def native_row(configuration, physical): return {'formula': physical['formula'], 'ARTIFICIAL': True}
def comparison_method(configuration): return 'FULL_PF' if configuration['spec']['L_D'] == 10 else 'TRUNCATED_PF'
def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
def load(path): return json.loads(Path(path).read_text())
def publish(paths, message): return 'ARTIFICIAL_PUBLICATION'
def head(): return 'ARTIFICIAL_SOURCE'
''')


def fake_modules():
    preparation = ModuleType('trottertracks.pf_evidence_preparation')
    def check(condition, reason):
        if not condition:
            raise ValueError(reason)
    preparation.check = check
    preparation.scalar_source = lambda prefix: {'ARTIFICIAL_tail': prefix}
    preparation.read = lambda *args: .25
    m0 = ModuleType('run_pf_m0_prices')
    m0.BASELINE = 'ARTIFICIAL_BASELINE'
    m0.S = SimpleNamespace(PLACEHOLDER='ARTIFICIAL_PLACEHOLDER')
    def bindings(*args, **kwargs):
        prices = []; sidecars = []
        for axis in ('X', 'Y'):
            prices.append({'axis': axis, 'diagnostic': {}, 'mean': {'rz_count': 123}})
            sidecars.append(dict(axis=axis, implementation_evidence={'selector': ['fixture_' + axis]},
                price_source=kwargs['price_source'], identity={'ARTIFICIAL': True}, law_binding={'ARTIFICIAL': True},
                cohort='ARTIFICIAL', physical_context={'ARTIFICIAL': True}, profile_extension={'ARTIFICIAL': True},
                implementation={'ARTIFICIAL': True}))
        return prices, sidecars, {}
    m0.F = SimpleNamespace(OUTPUT_ROOT='ARTIFICIAL_FULL', price_bindings=bindings)
    m0.T = SimpleNamespace(OUTPUT_ROOT='ARTIFICIAL_TRUNCATED', price_bindings=bindings)
    m0.read = lambda source: {'results': [{'facts': {'scalar': 1.25, 'tail_identity': .25},
                                          'trial': {'preparation_hash': 'ARTIFICIAL_PREPARATION'}}]}
    p7 = ModuleType('run_pf_p7_h_prices')
    resources = ModuleType('pf_common_run_contract')
    resources.resources = lambda: {'ARTIFICIAL': True}
    resources.workers = lambda *args: 1
    return preparation, {
        'trottertracks.pf_evidence_preparation': preparation,
        'run_pf_m0_prices': m0,
        'run_pf_p7_h_prices': p7,
        'pf_common_run_contract': resources,
    }


class PFPreparationModuleBinding(unittest.TestCase):
    def test_full_and_truncated_compile_use_the_loaded_provider_native_entry(self):
        for prefix in (10, 7):
            with self.subTest(prefix=prefix), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                fake_provider(root)
                wrapper = load_wrapper()
                with patch.object(sys, 'path', list(sys.path)):
                    provider = wrapper.load_provider(root)
                output = root / 'docs' / 'owned'
                value = dict(configuration={'id': 'ARTIFICIAL-' + str(prefix), 'spec': {'L_D': prefix}},
                    prepared={'pending': {'requests': [dict(m=2, physical_candidate={'formula': '2nd'})]}})
                source = {'commit': 'ARTIFICIAL_EXECUTION_SOURCE', 'caller': 'ARTIFICIAL_CALLER'}
                preparation, modules = fake_modules()
                child_calls = []
                def child(command, **kwargs):
                    child_calls.append(command)
                    self.assertEqual(Path(command[2]), ROOT/'scripts/run_pf_preparation_stage.py')
                    self.assertEqual(command[3:6], ['--provider-root', str(root), '--native-child'])
                    self.assertEqual(Path(kwargs['cwd']), root)
                    job = provider.load(command[6])
                    provider.write(Path(job['private']) / 'child.json', dict(results=[dict(ok=True,
                        cell=job['cells'][0], trial={'ARTIFICIAL': True}, wrappers={'cosine': {'ARTIFICIAL': 'X'},
                                                                               'sine': {'ARTIFICIAL': 'Y'}})]))
                    return SimpleNamespace(returncode=0)
                with patch.dict(sys.modules, modules), patch.object(trottertracks, 'pf_evidence_preparation', preparation, create=True), \
                     patch.object(wrapper.subprocess, 'run', side_effect=child):
                    result = wrapper.compile_receipts(value, output, 'simple_validation', [2], source)
                self.assertEqual(len(child_calls), 1)
                self.assertEqual(result['status'], 'COMPILE_RECEIPTS_READY')
                self.assertEqual(result['route'], 'FULL_PF' if prefix == 10 else 'TRUNCATED_PF')
                raw = output / 'simple_validation' / 'run01'
                prices = provider.load(raw / 'PRICES.json')
                self.assertEqual([p['axis'] for p in prices], ['X', 'Y'])
                self.assertEqual(provider.load(raw / 'RESULT.json')['child_exit_code'], 0)
                self.assertEqual(provider.load(raw / 'JOB.json')['driver_source'], source)
                self.assertTrue((raw / 'TRACE' / 'child.log').is_file())

    def test_native_child_uses_relocated_fixed_inputs_and_rejects_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = root/'snapshot'
            snapshot.write_bytes(b'fixed snapshot')
            config = root/'config.json'
            sha = hashlib.sha256(snapshot.read_bytes()).hexdigest()
            config.write_text(json.dumps(dict(provider_root='/original/provider',
                systems={'H4': {'inputs': {'snapshot': dict(path='/original/snapshot', sha256=sha)}}})))
            paths = {'PF_NATIVE_ORIGIN_CONFIG': '/original/config',
                     'PF_NATIVE_ORIGIN_SHA256': hashlib.sha256(config.read_bytes()).hexdigest(),
                     'PF_NATIVE_H4_INPUTS': ('snapshot',),
                     'resolve': {'/original/config': str(config), '/original/snapshot': str(snapshot),
                                 '/original/provider': str(root)}.__getitem__}
            legacy = ModuleType('h46_long_time_signal')
            wrapper = load_wrapper()
            with patch.dict(sys.modules, {legacy.__name__: legacy}), \
                 patch.object(wrapper.runpy, 'run_path', return_value=paths):
                wrapper.bind_native_compile_inputs()
                self.assertEqual(legacy.origin_config()['provider_root'], str(root))
                self.assertEqual(legacy.origin_config()['systems']['H4']['inputs']['snapshot']['path'],
                                 str(snapshot))
                snapshot.write_bytes(b'different snapshot')
                with self.assertRaisesRegex(ValueError, 'fixed SHA-256'):
                    wrapper.bind_native_compile_inputs()


if __name__ == '__main__':
    unittest.main()
