"""Strict metadata binding for the H6 runtime schedule, including registered times.

Only tuple/list serialization is normalized by the frozen coverage comparator.
Bounds and differences are saved before rejection. This is not a numerical PASS.
"""
from pathlib import Path
import hashlib
import json
from .ax2b_coverage_binding_v3 import canonical_coverage, assert_coverage_binding
from .ax2b_h6_input_generation_audit_v1 import read_json

BOUND_KEYS = {'primitive_actions', 'primitive_probe_count', 'cells', 'policy',
              'structural_bounds_are_not_compiled_costs'}
CELL_KEYS = {'cell_id', 'schedule', 'wrapper_instruction_upper_bounds'}
CONTROL_KEYS = {'ordinary', 'symmetric_directional'}
BOUND_POLICY = 'all actual unique times x saved state/first/last sector columns'


def expected_coverage(coverage):
    return {'cells': coverage['cells'], 'primitive_actions': coverage['primitive_actions'],
            'primitive_probe_count': coverage['primitive_probe_count'], 'bounds_schema_valid': True}


def actual_coverage(bounds):
    rows = bounds.get('cells')
    valid = (set(bounds) == BOUND_KEYS and type(rows) in (list, tuple)
             and bounds.get('policy') == BOUND_POLICY
             and bounds.get('structural_bounds_are_not_compiled_costs') is True)
    cells = []
    for row in rows if type(rows) in (list, tuple) else []:
        if type(row) is not dict:
            valid = False
            cells.append({'invalid_row_type': type(row).__name__})
            continue
        values = row.get('wrapper_instruction_upper_bounds')
        valid = valid and (set(row) == CELL_KEYS and type(values) is dict
                          and set(values) == CONTROL_KEYS
                          and all(type(v) is int and v >= 0 for v in values.values()))
        cells.append({'cell_id': row.get('cell_id'), 'schedule': row.get('schedule')})
    return {'cells': cells, 'primitive_actions': bounds.get('primitive_actions'),
            'primitive_probe_count': bounds.get('primitive_probe_count'),
            'bounds_schema_valid': bool(valid)}


def assert_actual_coverage(coverage, bounds, writer):
    """Persist full bounds first, then the comparator's actual/difference records."""
    writer.write('actual_bounds_v2.json', bounds, diagnostic=True)
    return assert_coverage_binding(expected_coverage(coverage), actual_coverage(bounds), writer)


def audit_coverage_records(output, coverage):
    """Read saved JSON only; do not recompute preparation, bounds or any signal."""
    out = Path(output)
    bounds = read_json(out / 'actual_bounds_v2.json')
    prepared = read_json(out / 'actual_prepared_representation.json')
    expected = canonical_coverage(expected_coverage(coverage))
    actual = canonical_coverage(actual_coverage(bounds))
    receipt = read_json(out / 'coverage_comparison_v3.json')
    stored = canonical_coverage(read_json(out / 'actual_coverage.json'))
    if (expected != actual or stored != actual
            or canonical_coverage(prepared.get('bounds')) != canonical_coverage(bounds)
            or receipt.get('schema') != 'track_a_ax2b_coverage_comparison_v3'
            or receipt.get('equal') is not True or receipt.get('differences') != []
            or receipt.get('differences_truncated') is not False
            or receipt.get('expected_sha256') != hashlib.sha256(expected).hexdigest()
            or receipt.get('actual_sha256') != hashlib.sha256(actual).hexdigest()
            or receipt.get('expected_canonical_bytes') != len(expected)
            or receipt.get('actual_canonical_bytes') != len(actual)):
        raise ValueError('PILOT_AUDIT_COVERAGE')
    return receipt
