"""Opt-in Stage R schema repair; the stopped V1 SOURCE remains unchanged.

The existing call dispatcher is the extension point for the event_catalog
unit. All CAR producers, signed-atom binding, norm arithmetic and call limits
are reused; no saved JSON is rewritten or coerced into the V1 expectation.
Passive preflight reads bytes/JSON and package metadata only.
"""
from fractions import Fraction
import importlib.metadata
import math
from pathlib import Path
import resource
import time
from types import SimpleNamespace

from . import c_per_word_range_execution as previous
from .c_iid_grid_bridge import MODEL

io = previous.io
BASE = 'e73f79de6e46d65e76ac7126f4eaf8a9b3568c0b'
FAMILY = 'artifacts/research_c_population_range_schema/2026-11-04'
CONTRACT = 'docs/research/research_c_population_range_schema_contract_unapproved_20261104.json'
ENTRY = 'scripts/prepare_research_c_population_range_schema.py'
SCOPE = 'C_PR_R32_LITERAL_POPULATION_NORM_STAGE_R_SCHEMA_V2'
SCHEMA = 'stage_A_event_tag_and_IID_model_v2'
IDEAL_EVENT_TAG = 'ideal_saved_atom'
LIMITS = dict(previous.LIMITS)
MAX_CALLS = sum(LIMITS.values())


def same_json(value, expected):
    """JSON identity includes types: bool/int and int/float are distinct."""
    if type(value) is not type(expected):
        return False
    if type(expected) is dict:
        return value.keys() == expected.keys() and all(same_json(value[k], v) for k, v in expected.items())
    if type(expected) is list:
        return len(value) == len(expected) and all(same_json(a, b) for a, b in zip(value, expected))
    return value == expected


def saved_metadata(root, c):
    """Validate producer-specific labels/configurations without array decode.

    Stage A writes an ideal tag to ideal_event_mean, an IID *ID string* to
    grid_event_mean, and the full IID configuration to result.model. These
    have different roles and must not be compared as interchangeable values.
    Saved numerical residuals/probabilities are neither recomputed nor used
    to infer a new qualification here.
    """
    if not same_json(c['identity']['model'], MODEL):
        raise ValueError('CONTRACT_IID_MODEL')
    ideal = io.read(Path(root) / c['event_mean'])
    grid_mean = io.read(Path(root) / c['grid_event_mean'])
    stage_a = io.read(Path(root) / c['stage_A_result'])
    for record, tag, label in ((ideal, IDEAL_EVENT_TAG, 'IDEAL_EVENT_TAG'),
                               (grid_mean, MODEL['id'], 'GRID_EVENT_MODEL_ID')):
        if type(record) is not dict or type(record.get('model')) is not str or record['model'] != tag:
            raise ValueError(label)
        report = record.get('report')
        if (type(report) is not dict or report.get('status') != 'K0_FINITE_MEAN_NUMERICAL_PASS'
                or type(report.get('angle')) is not float or not math.isfinite(report['angle'])
                or report.get('scalar_phase_excluded') is not True):
            raise ValueError(label + '_REPORT_SCHEMA')
    if (type(stage_a) is not dict or not same_json(stage_a.get('model'), c['identity']['model'])
            or stage_a.get('PCG64_joint_stream_qualified') is not False):
        raise ValueError('STAGE_A_IID_MODEL')
    if (stage_a.get('candidate') != c['identity']['candidate']
            or stage_a.get('PR_scientific_SOURCE') != c['identity']['PR_scientific_SOURCE']):
        raise ValueError('STAGE_A_TARGET_IDENTITY')
    if ideal['report']['angle'].hex() != grid_mean['report']['angle'].hex():
        raise ValueError('SAVED_EVENT_ANGLE_IDENTITY')
    atoms_record = io.read(Path(root) / c['atoms'])
    if type(atoms_record) is not dict or type(atoms_record.get('atoms')) is not list:
        raise ValueError('ATOM_SCHEMA')
    atoms = atoms_record['atoms']
    if len(atoms) != 216:
        raise ValueError('COMPLETE_ATOM_SOURCE')
    for row in atoms:
        if (type(row) is not dict or type(row.get('atom_id')) is not str
                or type(row.get('frame_id')) is not str or type(row.get('support')) is not list
                or any(type(x) is not int for x in row['support'])
                or type(row.get('coefficient')) is not str):
            raise ValueError('ATOM_SCHEMA')
    if len({row['atom_id'] for row in atoms}) != len(atoms):
        raise ValueError('COMPLETE_ATOM_SOURCE')
    table_record = io.read(Path(root) / c['grid_table'])
    if type(table_record) is not dict or type(table_record.get('grid')) is not dict:
        raise ValueError('GRID_TABLE_SCHEMA')
    table = table_record['grid']
    if (type(table.get('bits')) is not int or table['bits'] != MODEL['bits']
            or type(table.get('denominator')) is not int or table['denominator'] != 1 << MODEL['bits']
            or table.get('mapping') != 'searchsorted_right'
            or type(table.get('counts')) is not list or len(table['counts']) != len(atoms)
            or any(type(x) is not int for x in table['counts'])
            or table.get('zero_random_draws') is not True):
        raise ValueError('GRID_TABLE_SCHEMA')
    # Exact count sum/positive-q support remains the unchanged scientific
    # support_gate unit. Only types/shape/labels are checked passively.
    return dict(atoms=atoms, angle=ideal['report']['angle'])


def event_catalog(root, c, arm):
    data = saved_metadata(root, c)
    atoms, angle = data['atoms'], data['angle']
    actual = {a.atom_id: a for a in arm.plan.atoms}
    if set(actual) != {row['atom_id'] for row in atoms}:
        raise ValueError('COMPLETE_ATOM_SOURCE')
    for row in atoms:
        a = actual[row['atom_id']]
        if (a.frame_id != row['frame_id'] or list(a.support) != row['support']
                or Fraction(a.coefficient) != Fraction(row['coefficient'])):
            raise ValueError('SAVED_SIGNED_ATOM_BINDING')
    return dict(ids=[row['atom_id'] for row in atoms], catalog={row['atom_id']: SimpleNamespace(
        rotation_component_id=row['atom_id'], rotation_angle=angle,
        taylor_order=0, phase=1, product_component_ids=()) for row in atoms})


def preflight(root, path):
    root = Path(root).resolve()
    c = io.read(path)
    r = c['stages']['R']
    if (c['base_commit'] != BASE or c['schema'] != SCHEMA or c['target'] != previous.method.TARGET
            or c['precision_bits'] != 192 or c['retry'] != 0 or c['resume'] != 0
            or c['mandatory_stop'] is not True or c['maximum_logical_units'] != MAX_CALLS
            or r['scope'] != SCOPE or r['call_limits'] != LIMITS or r['maximum_logical_units'] != MAX_CALLS
            or r['fresh_output'] != FAMILY + '/stageR-run01' or r['entry'] != ENTRY + ' --execute-stage-R'
            or r['executable'] is not True
            or any(c['stages'][k]['executable'] is not False for k in ('P', 'C', 'A'))):
        raise ValueError('FIXED_SCHEMA_STAGE_R_CONTRACT')
    source = io.read(root / c['source_manifest'])
    if io.sha(root / c['source_manifest']) != c['source_manifest_sha256']:
        raise ValueError('SOURCE_MANIFEST')
    for group in (source['files'], c['saved_inputs']):
        for name, expected in group.items():
            p = (root / name).resolve()
            if not p.is_relative_to(root) or io.sha(p) != expected:
                raise ValueError('SOURCE_INPUT:' + name)
    for role in ('atoms', 'event_mean', 'grid_event_mean', 'grid_table', 'stage_A_result'):
        if c[role] not in c['saved_inputs']:
            raise ValueError('UNPINNED_METADATA:' + role)
    for name, expected in c['packages'].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError('ENVIRONMENT:' + name)
    if not same_json(c['identity'], previous.prior.integration.previous.identity()):
        raise ValueError('CANDIDATE_SOURCE_MODEL_COMPILER')
    saved_metadata(root, c)
    if (root / r['fresh_output']).exists():
        raise ValueError('FRESH_NO_RETRY_RESUME')
    return dict(status='PASSIVE_SCHEMA_STAGE_R_READY_FOR_SEPARATE_PERMISSION',
                schema=SCHEMA, scientific_calls=0, arrays_decoded=0, execution_started=False,
                Stage_R_maximum_logical_units=MAX_CALLS, Stage_P_executable=False,
                calibration_method_adopted=False, Stage_C_ready=False, mandatory_stop=True)


def execute_range(root, c, out, call):
    """Reuse V1 orchestration; explicitly replace only its event_catalog unit.

    The existing dispatcher hands each computation to the supplied call
    function. Capture its native_model result and supply the V2 catalog
    operation at that boundary. The old, incorrect metadata closure is never
    invoked. This is neither a JSON shim nor a fallback on failure.
    """
    state = {}

    def schema_call(kind, label, operation):
        if kind == 'event_catalog':
            if 'arm' not in state:
                raise ValueError('SCHEMA_CATALOG_REQUIRES_NATIVE_MODEL')
            operation = lambda: event_catalog(root, c, state['arm'])
        value = call(kind, label, operation)
        if kind == 'native_model':
            state['arm'] = value
        return value

    return previous.execute_range(root, c, out, schema_call)


def run(root, path, *, execute_science=False, executor=None):
    if not execute_science:
        raise PermissionError('SEPARATE_SCHEMA_STAGE_R_SCIENTIFIC_PERMISSION')
    root = Path(root).resolve()
    preflight(root, path)
    c = io.read(path)
    from .c_pr_aware_df_zsxx_enclosure import runtime_model
    from .c1_stage3_integrated import Calls
    runtime_model()
    out = root / c['stages']['R']['fresh_output']
    out.mkdir(parents=True, exist_ok=False)
    calls = Calls(out, LIMITS, MAX_CALLS)
    tick = time.monotonic()
    io.save(out / 'contract.json', c)
    io.save(out / 'SOURCE.json', io.read(root / c['source_manifest']))
    try:
        result = (executor or execute_range)(root, c, out, calls)
        io.save(out / 'result.json', result)
        return result
    except BaseException as error:
        calls.stop.set()
        if calls.first_failure is None:
            calls.first_failure = dict(kind='runner', error=type(error).__name__, message=str(error))
        io.save(out / 'partial_failure.json', dict(error=type(error).__name__, message=str(error)))
        raise
    finally:
        io.save(out / 'STOP.json', dict(scientific_units=sum(calls.counts.values()), call_counts=calls.counts,
                failed=calls.first_failure, retry=0, resume=0, mandatory_stop=True,
                Stage_P_executed=False, Stage_C_executed=False, actual_CA_achieved=None))
        io.save(out / 'resources.json', dict(wall_seconds=time.monotonic() - tick,
                peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
