"""Resolve operational paths in original preparation inputs without editing them.

Numerical identities, Git locators, SOURCE and frozen input bytes remain intact.
Only explicitly named runtime/output paths are resolved by their caller.
"""
import json
import os

PF_NATIVE_ORIGIN_CONFIG = (
    '/home/AbeHiromu/projects/h46-pr-exploration-execution/20261021/'
    'private/authorization_run01/H4_PR_EXPLORATION/config_APPROVED_PRIVATE.json'
)
PF_NATIVE_ORIGIN_SHA256 = '52e221c7e3d56dc8c75f5df0b208c60bb0c3d3c8885c14a331e7cdc86c4e24c4'
PF_NATIVE_H4_INPUTS = ('H4_tol_snapshot.npz', 'H4_tol_receipt.json')

# The legacy price adapter requires this checked-out view next to its provider
# workspace. Keep the original adapter and all of its SOURCE checks unchanged.
PAULI_CALIBRATION_VIEW = 'a-pauli-ld3-cal-source-13ab'
PAULI_CALIBRATION_SOURCE = '13ab1acfaebe5f33958e1db67862f5c45725ab91'
PAULI_CALIBRATION_FILES = (
    'scripts/run_pauli_b1_prices.py',
    'scripts/run_pauli_ld1_m0_pilot.py',
    'src/trottertracks/pauli_tail_price_connection.py',
    'src/trottertracks/pauli_b1_receipts.py',
    'scripts/h46_same_event.py',
    'src/trottertracks/finite_price_design.py',
)


def resolve(value):
    if not isinstance(value, str):
        return value
    mapping = json.loads(os.environ.get('RPE_PREPARATION_PATH_MAP', '{}'))
    for old, new in sorted(mapping.items(), key=lambda pair: len(pair[0]), reverse=True):
        if value == old or value.startswith(old.rstrip('/') + '/'):
            return new.rstrip('/') + value[len(old.rstrip('/')):]
    return value


def runtime_request(request, fields):
    result = dict(request)
    for field in fields:
        if field in result:
            result[field] = resolve(result[field])
    if 'interpreter' in fields and os.environ.get('RPE_PREPARATION_PYTHON'):
        result['interpreter'] = os.environ['RPE_PREPARATION_PYTHON']
    if 'quote_interpreter' in fields and os.environ.get('RPE_PREPARATION_PYTHON'):
        result['quote_interpreter'] = os.environ['RPE_PREPARATION_PYTHON']
    return result
