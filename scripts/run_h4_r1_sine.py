#!/usr/bin/env python3
"""Specialize frozen R0 one-shot controller for three saved sine programs."""
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
pkg = types.ModuleType('sine_private')
pkg.__path__ = [str(ROOT / 'src/trotterlib')]
sys.modules['sine_private'] = pkg
from sine_private import h4_r1_sine as A

C, BASE = A.C, A.BASE
PYTHON = '/home/AbeHiromu/projects/h6-environments/20261012/native-dependency-v2/venv/bin/python'
SCHEMA = dict(version='R1_SINE_BINARY64_EMPIRICAL_V1', keys=list(A.KEYS), logical_diagnostics=3,
    initial_maximum_state_actions=6, axis='sine', required=['m_C', 'm_input', 'm_output', 'delta', 'eta', 'd', 'rho_budget_exact', 'd_over_budget', 'budget_minus_d'],
    status='EMPIRICAL_ONLY_NO_RHO_UPPER', rho_upper=None, numerical_error_upper=None, PR_law_rho=None, formal_certificate_accepted=False)
FORBIDDEN = dict(new_compile=0, new_circuit_build=0, sampling=0, RNG_replay=0, Hamiltonian_state_generation=0,
    new_C_target_state_actions=0, cosine_reevaluation=0, C_L_full_runs=0, N_G_reanalysis=0,
    Arb_interval=0, precision_escalation=0, additional_trajectory=0, R2=0, certificate_acceptance=0)


def controller():
    tree = ast.parse((ROOT / 'scripts/run_h4_r0_empirical.py').read_text())
    literal = dict(R0_BINARY64_SOURCE_INPUT_RUNTIME_SEAL_V1='R1_SINE_SOURCE_INPUT_RUNTIME_SEAL_V1',
        R0_BINARY64_EMPIRICAL_ACTIVE_V1='R1_SINE_BINARY64_EMPIRICAL_ACTIVE_V1',
        R0_BINARY64_ONE_SHOT='R1_SINE_BINARY64_ONE_SHOT',
        NEW_R0_BINARY64_EMPIRICAL_ONLY='NEW_R1_SINE_BINARY64_EMPIRICAL_ONLY', cosine='sine')
    literal['2734b97c2dd5f017e93cd81619ff52a6a0c6f16b'] = 'c7585f2d6e4a0fa3f0cc873fb6e468be1d42bce9'
    literal['binary64 empirical state actions only; saved PF interval scalar conversion and exact residual/budget accounting, no new state/error enclosure'] = 'binary64 sine empirical only; reuse PF saved imag interval and PR0/PR1 complex targets; no ideal target or cosine state action'
    literal['five short 256/512-amplitude program walks; one worker avoids repeated parsing; no CPU/RAM/thread/affinity ceiling or shared changes'] = 'six short 512-amplitude walks share parsed QPY/state in one worker; no CPU/RAM/thread/affinity ceiling or shared changes'

    class Scope(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str) and node.value in literal: node.value = literal[node.value]
            elif type(node.value) is int and node.value in (2, 5): node.value = {2: 3, 5: 6}[node.value]
            return node

        def visit_keyword(self, node):
            node = self.generic_visit(node)
            if node.arg == 'scientific_SOURCE':
                node.value.keywords.extend([ast.keyword(arg=k, value=ast.Constant(v)) for k, v in dict(
                    A_GATE=A.P.AG, R0='84a326598b006384d13c7a354a3eec6881ee841c', R1='ca9a1c0e2426565ba63cc8bfd0247f73339d95b8').items()])
            return node

    code = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'CODE' for t in n.targets))
    env = dict(globals())
    exec(compile(ast.Module(body=[code], type_ignores=[]), '<R0 code inventory>', 'exec'), env)
    env['CODE'] += ['src/trotterlib/h4_r1_sine.py', 'src/trotterlib/h4_r1_order2.py', 'src/trotterlib/rte_event_audit.py',
        'scripts/run_h4_r1_sine.py', 'tests/test_h4_r1_sine.py', 'artifacts/non_df_h4_central_r1_sine/2026-10-24/prepare_metadata.py']
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('verify', 'prepare', 'check', 'worker', 'launch')]
    C.require(len(functions) == 5, 'FIVE_REUSED_R0_CONTROLLER_FUNCTIONS')
    module = ast.fix_missing_locations(Scope().visit(ast.Module(body=functions, type_ignores=[])))
    exec(compile(module, '<R0 controller with sine scope>', 'exec'), env)
    return env


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    for mode in ('inspect', 'prepare', 'launch', 'worker'): group.add_argument('--' + mode, action='store_true')
    args = parser.parse_args()
    env = controller()
    if args.inspect:
        *_, result = A.preflight()
        C.save(BASE / 'structural_preflight.json', result)
        print(json.dumps(dict(status='PASS', state_actions=0, new_C_target_state_actions=0)))
    elif args.prepare: env['prepare']()
    elif args.launch: sys.exit(env['launch']())
    else: env['worker']()
