#!/usr/bin/env python3
"""User-directed new Stage 3 attempt; reuse old runner, preserve run01."""
import argparse
import importlib
from pathlib import Path
import run_h4_p3_stage3 as original

BASE='aab0b483fdd8e57fe17305a11c5cb750a9c80a4d'
_modules=original.modules
_write=original.write
_kernel=None
VARIANT='project-local number-conserving Givens; exact-zero elimination; atan2 angle; RZ/XXPlusYY'


def modules():
    global _kernel
    if _kernel is None:
        kernel=_modules()
        stable=importlib.import_module('stage1.stable_gaussian_basis')
        # Inject a project-local dependency into this process's reused kernel.
        # NumPy, Qiskit and qiskit-nature modules/packages are not monkeypatched.
        kernel.U_to_qiskit_ops_jw_givens=stable.stable_givens_ops
        basis=kernel.basis_from_saved
        def adapted_basis(u):
            result=basis(u);result.metadata['native_family']=VARIANT
            return result
        kernel.basis_from_saved=adapted_basis
        _kernel=kernel
    return _kernel


def write(path,data):
    if path=='source_and_inputs.json':
        data=dict(data,implementation_variant=VARIANT,
            prior_failed_result_commit=BASE,
            prior_failed_namespace='artifacts/df_pauli_matched_partition_validation/2026-11-02/stage3/run01',
            precision_thresholds_unchanged=True,compiler_seeds_tuple_unchanged=True,
            recovery_scope='user-directed technical correction and one new same-tuple Stage3 attempt; no automatic retry')
    if isinstance(data,dict) and 'counts' in data:
        data['counts'].update(scientific_retries=1,user_directed_followup_attempts=1,automatic_retries=0)
    _write(path,data)


def execute():
    original.OUT=original.ROOT/'artifacts/df_pauli_matched_partition_validation/2026-11-02/stage3/run02'
    original.OWN=(*original.OWN,'scripts/run_h4_p3_stage3_stable_basis.py',
        'src/trotterlib/stable_gaussian_basis.py','tests/test_stable_gaussian_basis.py',
        'docs/research/h4_p3_stage3_stable_basis_recovery_20261102.md')
    original.modules=modules;original.write=write
    original.execute()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--execute',required=True,action='store_true')
    parser.parse_args();execute()
