"""Task-local HiGHS coefficient-retention profile; no molecular data at import.

The historical LP source, constraints, RHS and qualification tests are reused
unchanged. This module neither repairs returned duals nor retries a solver.
"""
from contextlib import contextmanager
import importlib.metadata
from pathlib import Path
import resource
import subprocess
import time
import warnings
from . import c_fixed_u_tail_lp_runner as previous

lp, io, save = previous.lp, previous.io, previous.save
BASE = '222330e60179873fb3753264720697677351bfaa'
VERSION = 'fixed_U_full_rows_small_matrix_1em12_v2'
SCOPE = 'C_FIXED_U_PR_TAIL_LP_NUMERICS_V2'
FAMILY = 'artifacts/research_c_fixed_u_tail_lp_numerics/2026-11-01'
FRESH = FAMILY + '/run01'
OPTIONS = dict(previous.lp.OPTIONS, small_matrix_value=1e-12)
LIMITS, MAX_CALLS, POLICY = dict(previous.LIMITS), previous.MAX_CALLS, dict(previous.POLICY)
WARNING_POLICY = 'scipy_1_14_1_duplicate_int_double_option_v1'


def expected_warnings():
    # v1.14.1 _highs_wrapper.pyx lists small_matrix_value in BOTH the int and
    # double loops. The first warns for a float; the second sets it correctly.
    return [
        ("Unrecognized options detected: {'small_matrix_value': 1e-12}. "
         'These will be passed to HiGHS verbatim.'),
        ('Option "small_matrix_value" is "1e-12", but only values in '
         '(1e-12, inf) are allowed. Using default: 1e-09.')]


def warning_gate(records):
    """Keep the known misleading warnings visible; reject extra native errors.

    An actual double-setter failure emits another warning, even if its text is
    identical. Counting occurrences matters. No broad warning suppression.
    """
    if ([x['message'] for x in records] != expected_warnings()
            or any(x['category'] != 'OptimizeWarning' for x in records)):
        raise ValueError('HIGHS_OPTION_HANDLING_UNQUALIFIED')


@contextmanager
def option_scope():
    """Dedicated runner process only; restore even on exception.

    No file, SciPy object, shared venv or other process is changed. Keeping the
    profile in scope also lets the reused runner record the actual options.
    The one LP occurs before its parallel CAR tasks. Concurrent LPs in this
    process are not supported.
    """
    original = lp.OPTIONS
    lp.OPTIONS = dict(OPTIONS)
    try:
        yield
    finally:
        lp.OPTIONS = original


class OptionCalls:
    """Use the existing call journal; qualify warnings after raw LP is saved."""
    def __init__(self, calls, output):
        self.calls, self.output, self.records = calls, Path(output), None
        self.stop = calls.stop

    def __call__(self, kind, label, function):
        if kind == 'LP_solve':
            def solve():
                with warnings.catch_warnings(record=True) as seen:
                    warnings.simplefilter('always')
                    try:
                        return function()
                    finally:
                        self.records = [dict(category=x.category.__name__,
                                             message=str(x.message)) for x in seen]
                        save(self.output/'solver_option_handling.json', dict(
                            profile=WARNING_POLICY, options=OPTIONS,
                            warnings=self.records, warnings_saved_verbatim=True,
                            qualification='checked in LP_diagnose after raw export'))
            return self.calls(kind, label, solve)
        if kind == 'LP_diagnose':
            def diagnose():
                warning_gate(self.records or [])
                return function()
            return self.calls(kind, label, diagnose)
        return self.calls(kind, label, function)


def preflight(root, path):
    """Hashes/JSON/headers/versions only; no NPZ values or LP evaluation."""
    root = Path(root).resolve()
    c = io.read(path)
    fixed = dict(scope=SCOPE, implementation=VERSION, base_commit=BASE,
        policy=POLICY, solver_options=OPTIONS, allowances=lp.ALLOWANCES,
        option_warning_policy=WARNING_POLICY, call_limits=LIMITS,
        maximum_logical_units=MAX_CALLS, fresh_output=FRESH, retry=0, resume=0)
    for key, wanted in fixed.items():
        if c[key] != wanted:
            raise ValueError('FIXED_LP_NUMERICS_SCOPE:'+key)
    if io.sha(root/c['source_manifest']) != c['source_manifest_sha256']:
        raise ValueError('SOURCE_MANIFEST')
    for group in (io.read(root/c['source_manifest'])['files'], c['saved_inputs']):
        for name, digest in group.items():
            target = (root/name).resolve()
            if not target.is_relative_to(root) or io.sha(target) != digest:
                raise ValueError('SOURCE_OR_INPUT:'+name)
    required = {c['input'], c['export'], c['reference'], *c['records'].values()}
    if not required <= c['saved_inputs'].keys():
        raise ValueError('UNPINNED_LP_INPUT')
    if c['identity'] != dict(role='pr', export=c['export'],
            sha256=c['saved_inputs'][c['export']], fit_source=previous.old.FIT_SOURCE,
            physical_input=c['saved_inputs'][c['input']]):
        raise ValueError('PR_IDENTITY')
    previous.old.old.export_headers(root/c['export'])
    saved = previous.evidence(root, c)
    previous.saved_gate(saved, c)
    parent, stop = saved['parent_LP_result'], saved['parent_LP_STOP']
    if (parent['status'] != 'UNRESOLVED_LP' or parent['candidate'] is not None
            or stop['failed'] or stop['scientific_units'] != 9
            or stop['call_counts'].get('LP_solve') != 1
            or stop['retry'] or stop['resume'] or not stop['mandatory_stop']
            or saved['parent_LP_SOURCE']['execution_commit'] !=
               '56a6acb8f7f48cd4792c560b64bc4f4aa270e0ad'):
        raise ValueError('PARENT_LP_STOP_IDENTITY')
    for name, wanted in c['packages'].items():
        if importlib.metadata.version(name) != wanted:
            raise ValueError('ENVIRONMENT:'+name)
    if (root/FRESH).exists():
        raise ValueError('FRESH_LP_NUMERICS_NO_RETRY_RESUME')
    return c


def run(root, path, output):
    """Dormant until separately permitted; exactly one new LP under this scope."""
    root = Path(root).resolve()
    c = preflight(root, path)
    output = Path(output).resolve()
    if output != root/FRESH or output.exists():
        raise ValueError('FRESH_LP_NUMERICS_NO_RETRY_RESUME')
    output.mkdir(parents=True)
    calls = previous.old.old.integrated.Calls(output, LIMITS, MAX_CALLS)
    started, cpu = time.monotonic(), time.process_time()
    try:
        save(output/'contract.json', c)
        save(output/'SOURCE.json', dict(io.read(root/c['source_manifest']),
            execution_commit=subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()))
        with option_scope():
            return previous.execute(root, c, output, OptionCalls(calls, output))
    except BaseException as error:
        calls.stop.set()
        save(output/'failure.json', dict(status='FAILED_STOP', error=str(error),
                                        first_failure=calls.first_failure))
        raise
    finally:
        save(output/'STOP.json', dict(failed=calls.stop.is_set(),
            scientific_units=sum(calls.counts.values()), call_counts=dict(calls.counts),
            retry=0, resume=0, mandatory_stop=True))
        save(output/'resources.json', dict(wall_seconds=time.monotonic()-started,
            CPU_seconds=time.process_time()-cpu,
            peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
