"""Artificial definition/binding checks; no kernel, sampler or compiler calls."""
import ast
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks import pr_resource_wiring as w


def descriptor_api(tail, *, evolution_time, rte_steps, seed):
    raise AssertionError('API must only be bound')


def compatible_call(context, tail):
    return context.make_rte_config(tail, evolution_time=1, rte_steps=2, seed=0)


def obsolete_call(context, tail):
    return context.make_rte_config(tail, time=1, steps=2)


class WiringTests(unittest.TestCase):
    def test_actual_bytecode_scope_includes_nested_function_globals(self):
        scope = {}
        exec('def context():\n def inner(): return ast.parse("pass")\n return inner()\n', scope)
        with self.assertRaisesRegex(ValueError, 'unsupplied globals: ast'):
            w.check_globals(scope['context'])
        scope['ast'] = ast
        w.check_globals(scope['context'])

    def test_binding_checks_required_and_obsolete_arguments_without_call(self):
        w.bind_calls(compatible_call, 'make_rte_config', descriptor_api)
        with self.assertRaises(TypeError):
            w.bind_calls(obsolete_call, 'make_rte_config', descriptor_api)

    def test_api_disappearance_is_not_a_silent_pass(self):
        with self.assertRaisesRegex(ValueError, 'Expected API call missing'):
            w.bind_calls(compatible_call, 'foreign_api', descriptor_api)

    def test_full_fixed_commit_is_required_before_loading(self):
        for source in ('HEAD', '51d324ea', '../source', 'z'*40):
            with self.assertRaisesRegex(ValueError, 'fixed SOURCE'):
                w.preflight(ROOT, 'B', source)

    def test_changed_loader_boundary_stops_without_invoking_loader(self):
        def foreign_loader(root, plan, system):
            raise AssertionError('Do not run an unrecognized loader')
        with self.assertRaisesRegex(ValueError, 'boundary changed'):
            w.context_definition(foreign_loader, ROOT, {})


if __name__ == '__main__': unittest.main(verbosity=2)
