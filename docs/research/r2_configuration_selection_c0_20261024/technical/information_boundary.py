"""Synthetic-only interface check. No numerical selector or scientific runner."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def field_registry():
    with (ROOT / 'information_tiers.csv').open() as stream:
        return {row['field']: row for row in csv.DictReader(stream)}


def validate_synthetic_view(kind, features, *, paid_fields=()):
    """Validate field ancestry, including renamed/derived feature declarations.

    This checks the declared interface, not arbitrary code or source truth. It
    does not certify independence of a fitted or externally computed proxy.
    """
    if kind != 'ARTIFICIAL':
        raise ValueError('NO_REAL_DATA_EXECUTION_INTERFACE_IN_C0')
    registry = field_registry()
    paid = set(paid_fields)
    if paid - set(registry):
        raise ValueError('UNKNOWN_PAID_FIELD')

    def visit(name, active):
        if name in active:
            raise ValueError('CYCLIC_FEATURE_LINEAGE')
        if name not in registry:
            raise ValueError('UNREGISTERED_SOURCE_FIELD')
        row = registry[name]
        if row['role'] == 'IDENTITY_ONLY':
            raise ValueError('IDENTITY_IS_NOT_PERFORMANCE_FEATURE')
        if row['tier'] == 'Oracle':
            raise ValueError('ORACLE_LEAKAGE')
        if row['tier'] == 'Tier2' and name not in paid:
            raise ValueError('UNPAID_TIER2_INFORMATION')
        for parent in filter(None, row['ancestors'].split(';')):
            visit(parent, active | {name})

    if len({f['name'] for f in features}) != len(features):
        raise ValueError('DUPLICATE_FEATURE_NAME')
    for feature in features:
        if not feature.get('ancestors'):
            raise ValueError('MISSING_SOURCE_LINEAGE')
        for ancestor in feature['ancestors']:
            visit(ancestor, set())
    return [feature['name'] for feature in features]


def validate_nonexecuting_contract(contract):
    if (contract['execution_authorized'] is not False
            or contract['enabled'] is not False
            or contract['grant'] is not None
            or contract['runner_SOURCE'] is not None
            or contract['internal_calls_verified'] is not False
            or any(v != 0 for v in contract['forbidden_scientific_calls'].values())):
        raise ValueError('C0_CONTRACT_MUST_REMAIN_NONEXECUTABLE')
    return True
