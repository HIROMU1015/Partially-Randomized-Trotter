"""Actual-interpreter binding regressions for deployment metadata, no Q3."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from collect_rpe_q3_deployment_readiness import bind_runtime


class DeploymentRuntimeBindingTests(unittest.TestCase):
    def setUp(self):
        self.profile=dict(interpreter='/owned/launcher')
        runtime=dict(executable='/owned/runtime/bin/python3', version='3.11.0rc1 fixed synthetic runtime',
                     interpreter_invocation='/owned/launcher', launcher_sha256='LAUNCHER',
                     actual_executable_realpath='/owned/runtime/bin/python3.11', actual_binary_sha256='BINARY')
        self.actual=dict(status='PASS',runtime=runtime,checks=[])
        self.raw=dict(runtime=dict(executable=runtime['executable'],version=runtime['version']))
        self.observed=dict(python_version='3.12.3 checker', adapter_result=dict(runtime=dict(
            executable=runtime['executable'],version=runtime['version'],guard_modified=False,constructor='PASS_UNCHANGED_GUARD')))

    def test_launcher_uses_actual_adapter_python_not_checker_version(self):
        result=bind_runtime(self.profile,self.raw,self.actual,self.observed)
        self.assertEqual(result['runtime']['version'],self.actual['runtime']['version'])
        self.assertEqual(result['runtime']['actual_binary_sha256'],'BINARY')
        self.assertNotIn('actual_binary_sha256',self.raw['runtime'])

    def test_changed_adapter_runtime_or_modified_guard_rejected(self):
        for key,value in [('version','3.12.3 checker'),('guard_modified',True),('constructor','GUARD_BYPASSED')]:
            observed=deepcopy(self.observed);observed['adapter_result']['runtime'][key]=value
            with self.subTest(key=key), self.assertRaises(ValueError):
                bind_runtime(self.profile,self.raw,self.actual,observed)

    def test_launcher_without_binary_binding_is_not_reused(self):
        actual=deepcopy(self.actual);actual['runtime'].pop('actual_binary_sha256')
        with self.assertRaisesRegex(ValueError,'launcher identity'):
            bind_runtime(self.profile,self.raw,actual,self.observed)

    def test_official_runtime_precheck_must_equal_actual_selected_precheck(self):
        self.profile['runtime_checks']=[dict(kind='SYNTHETIC_FIXED_GUARD')]
        self.observed['runtime_precheck']=deepcopy(self.actual)
        bind_runtime(self.profile,self.raw,self.actual,self.observed)
        self.observed['runtime_precheck']['checks']=[dict(status='CHANGED')]
        with self.assertRaisesRegex(ValueError,'Official runtime precheck differs'):
            bind_runtime(self.profile,self.raw,self.actual,self.observed)


if __name__=='__main__':unittest.main()
