#!/usr/bin/env python3
"""Reuse R0 controller functions; specialize only R1 scope constants and binding."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
pkg = types.ModuleType('r1_private')
pkg.__path__ = [str(ROOT / 'src/trotterlib')]
sys.modules['r1_private'] = pkg
from r1_private import h4_r1_order2 as A

C, BASE = A.C, A.BASE
PYTHON = '/home/AbeHiromu/projects/h6-environments/20261012/native-dependency-v2/venv/bin/python'
SCHEMA = dict(version='R1_ORDER2_BINARY64_EMPIRICAL_V1', keys=list(A.KEYS), logical_diagnostics=1,
    initial_maximum_state_actions=3, axis='cosine',
    required=['m_C', 'm_input', 'm_output', 'delta', 'eta', 'd', 'rho_budget_exact', 'd_over_budget', 'budget_minus_d'],
    status='EMPIRICAL_ONLY_NO_RHO_UPPER', rho_upper=None, numerical_error_upper=None,
    PR_law_rho=None, formal_certificate_accepted=False)
FORBIDDEN = dict(new_compile=0, new_circuit_build=0, sampling=0, RNG_replay=0, Hamiltonian_state_generation=0,
    C_L_full_runs=0, N_G_reanalysis=0, Arb_interval=0, precision_escalation=0, sine=0,
    R0_reevaluation=0, additional_trajectory=0, R2=0, certificate_acceptance=0)


def controller():
    tree = ast.parse((ROOT / 'scripts/run_h4_r0_empirical.py').read_text())
    literal = dict(
        R0_BINARY64_SOURCE_INPUT_RUNTIME_SEAL_V1='R1_ORDER2_SOURCE_INPUT_RUNTIME_SEAL_V1',
        R0_BINARY64_EMPIRICAL_ACTIVE_V1='R1_ORDER2_BINARY64_EMPIRICAL_ACTIVE_V1',
        R0_BINARY64_ONE_SHOT='R1_ORDER2_BINARY64_ONE_SHOT',
        NEW_R0_BINARY64_EMPIRICAL_ONLY='NEW_R1_ORDER2_BINARY64_EMPIRICAL_ONLY')
    literal['2734b97c2dd5f017e93cd81619ff52a6a0c6f16b'] = 'f01c4018201de9335f20a8901ec5f40f9cf9e89b'
    literal['binary64 empirical state actions only; saved PF interval scalar conversion and exact residual/budget accounting, no new state/error enclosure'] = 'binary64 empirical selected PR1 only; reuse saved raw norm, exact residual/budget accounting; no new enclosure or R0 reevaluation'
    literal['five short 256/512-amplitude program walks; one worker avoids repeated parsing; no CPU/RAM/thread/affinity ceiling or shared changes'] = 'three short 256/512-amplitude program walks; one worker reuses parsed inputs, no CPU/RAM/thread/affinity ceiling or shared changes'

    class Scope(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str) and node.value in literal:
                node.value = literal[node.value]
            elif type(node.value) is int and node.value in (2, 5):
                node.value = {2: 1, 5: 3}[node.value]
            return node

        def visit_keyword(self, node):
            node = self.generic_visit(node)
            if node.arg == 'scientific_SOURCE':
                node.value.keywords.extend([
                    ast.keyword(arg='A_GATE', value=ast.Constant(A.P.AG)),
                    ast.keyword(arg='R0', value=ast.Constant('84a326598b006384d13c7a354a3eec6881ee841c'))])
            return node

    # Reuse the old CODE list and include both the reused controller and new files.
    code_node = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'CODE' for t in n.targets))
    env = dict(globals())
    exec(compile(ast.Module(body=[code_node], type_ignores=[]), '<R0 code inventory>', 'exec'), env)
    env['CODE'] += ['src/trotterlib/h4_r1_order2.py', 'scripts/run_h4_r1_order2.py',
                    'tests/test_h4_r1_order2.py', 'artifacts/non_df_h4_central_r1_order2/2026-10-24/prepare_metadata.py']
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('verify', 'prepare', 'check', 'worker', 'launch')]
    C.require(len(functions) == 5, 'R0_CONTROLLER_FUNCTIONS')
    module = ast.fix_missing_locations(Scope().visit(ast.Module(body=functions, type_ignores=[])))
    exec(compile(module, '<R0 controller with R1 scope constants>', 'exec'), env)
    return env


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    for mode in ('inspect', 'prepare', 'launch', 'worker'):
        group.add_argument('--' + mode, action='store_true')
    args = parser.parse_args()
    env = controller()
    if args.inspect:
        *_, result = A.preflight()
        C.save(BASE / 'structural_preflight.json', result)
        print(A.R.json.dumps(dict(status='PASS', state_actions=0)))
    elif args.prepare:
        env['prepare']()
    elif args.launch:
        sys.exit(env['launch']())
    else:
        env['worker']()
