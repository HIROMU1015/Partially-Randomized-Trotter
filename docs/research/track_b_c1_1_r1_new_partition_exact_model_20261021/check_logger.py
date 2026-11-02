"""Artificial stdlib-only logger checks; never call scientific_diagnosis/main."""
import contextlib
import io
import json
from pathlib import Path
import runpy
import tempfile

HERE = Path(__file__).resolve().parent
module = runpy.run_path(str(HERE / 'support.py'))
records = []


def fail(_output):
    raise ValueError('synthetic primary exception')


def logger_fail(_event, _details):
    raise RuntimeError('synthetic terminal logger exception')


with tempfile.TemporaryDirectory(prefix='c1_1_r1_logger_') as temporary:
    cases = [
        ('match_with_stage', lambda _: {'classification': 'R1_LIMITED_REANALYSIS_COMPLETE', 'stage': 'synthetic', 'diagnosis_complete': True}, None, 0),
        ('mismatch_with_stage', lambda _: {'classification': 'HASH_MISMATCH_UNRESOLVED', 'stage': 'synthetic', 'diagnosis_complete': True}, None, 2),
        ('primary_exception', fail, None, 1),
        ('primary_and_logger_exception', fail, logger_fail, 1),
    ]
    for name, action, event_writer, expected_exit in cases:
        output = Path(temporary) / name
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            kwargs = {'event_writer': event_writer} if event_writer else {}
            actual_exit = module['run_guarded'](action, output, **kwargs)
        terminal = json.loads((output / 'terminal.json').read_text())
        assert actual_exit == expected_exit == terminal['expected_process_exit']
        assert terminal['attempts'] == 1 and terminal['retries'] == 0 and terminal['mandatory_STOP']
        failure = json.loads((output / 'failure.json').read_text()) if (output / 'failure.json').exists() else None
        if action is fail:
            assert failure['primary_error'] == 'synthetic primary exception'
            assert 'ValueError' in failure['traceback']
        if event_writer:
            assert terminal['secondary_failure_count'] == 1
            assert json.loads((output / 'secondary_failures.json').read_text())[0]['where'] == 'terminal_logger'
        records.append({'case': name, 'exit': actual_exit, 'passed': True, 'terminal': terminal,
                        'failure': failure, 'stdout': stdout.getvalue(), 'stderr': stderr.getvalue()})
module['save'](HERE / 'artificial_logger_checks.json', {'artificial_only': True,
    'science_imports_or_calls': 0, 'cases': records, 'all_passed': True})
print(json.dumps({'artificial_cases': len(records), 'all_passed': True, 'scientific_calls': 0}))
