#!/usr/bin/env python3
"""Binding-only by default. --output runs the separately permitted scalar batch.

An invocation is not proof of permission; the user's separate instruction is
required. No grants, approval flags or authentication machinery are introduced.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
PREPARATION = ROOT / 'artifacts/non_df_h4_pr_mean_cost_scalar_preparation/2026-10-25'


def checked_inputs():
    """SOURCE/input bytes and quoted fields only. No log, sqrt, U or L."""
    binding = json.loads((PREPARATION / 'input_bindings.json').read_text())
    for path, expected in binding['sha256'].items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected:
            raise ValueError('SOURCE_OR_INPUT_MISMATCH:' + path)
    spec = json.loads((ROOT / binding['stage1_spec']).read_text())
    summary = json.loads((ROOT / binding['DF_summary']).read_text())['df_native_PR']
    scope = spec['fixed_scope']
    actual = dict(n=scope['n_system'], L=scope['L_Pauli_retained'], q=scope['q'],
                  r=scope['r'], K=scope['K'], n_samples=summary['n'],
                  c_min=summary['observed_min'],
                  delta_axis=spec['statistical_model']['delta_axis_exact_decimal'],
                  delta_cost=spec['statistical_model']['delta_cost_exact_decimal'])
    if actual != binding['inputs'] or spec['execution_enabled'] is not False:
        raise ValueError('INPUT_BINDING_OR_HISTORICAL_SPEC_CHANGED')
    return actual, binding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        help='After separate user permission only: create a new scalar-result JSON.')
    args = parser.parse_args()
    inputs, binding = checked_inputs()
    if args.output is None:
        print(json.dumps(dict(status='INPUTS_VERIFIED_NO_H4_EVALUATION',
                              execution_permission='SEPARATE_USER_INSTRUCTION_REQUIRED',
                              scalar_batch_calls=0, mandatory_STOP=True)))
        return
    # Exclusive creation prevents overwriting a successful prior result.
    with args.output.open('x') as stream:
        try:
            # Reuse the private namespace pattern only after binding succeeds;
            # the binding-only mode never imports a numerical kernel.
            pkg = types.ModuleType('h4_pr_price_private')
            pkg.__path__ = [str(ROOT / 'src/trotterlib')]
            sys.modules['h4_pr_price_private'] = pkg
            kernel = importlib.import_module('h4_pr_price_private.h4_pr_mean_cost_scalar')
            result = kernel.evaluate_scalar_batch(inputs)
        except Exception as error:
            json.dump(dict(status='STOPPED_ON_EXCEPTION', error_type=type(error).__name__,
                           error=str(error), inputs=inputs, results=None,
                           numerical_separation_status='STOPPED_ON_EXCEPTION',
                           count_premise_status='UNPROVED_PREMISES_UNCHANGED',
                           statistical_model_status='FIXED_IID_ASSUMPTION_UNCHANGED',
                           population_claim_scope='NO_NEW_POPULATION_CLAIM',
                           automatic_retry=False, mandatory_STOP=True), stream, indent=2)
            stream.write('\n')
            raise
        json.dump(dict(status='SCALAR_BATCH_COMPLETED', inputs=inputs,
                       input_SOURCE_sha256=binding['sha256'], results=result,
                       automatic_next_stage=False, mandatory_STOP=True), stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
