"""Reuse fixed D/Arb/V2 helpers without importing research pipelines."""
from __future__ import annotations
import importlib.util
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


D = module('rpe_D_connection', 'docs/research/rpe_branch_energy_preparation_20261104/adapter.py')
ball = D.ball
real, upper, encode, stop = D.real, D.upper, D.encode, D.stop
STATUS = dict(D.STATUS)


def excitation_functions():
    """Unchanged integer/sign SOURCE functions; constructor is structural only."""
    loader = module('rpe_V2_loader', 'docs/research/pr_rpe_v2_branch_success_20261103/run_saved_v2.py')
    source = (ROOT / 'src/trotterlib/df_hamiltonian.py').read_text()
    functions, records = loader.selected_functions(source, ('_before_mode_masks', '_build_excitation_table'))
    functions['_build_excitation_table'].__globals__['_ExcitationTable'] = SimpleNamespace
    return functions, records
