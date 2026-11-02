"""Shared artificial/live failure journal: first failure is immutable."""
import datetime
import json
from pathlib import Path
import sys
import traceback

def exclusive_json(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write('\n')

def classify_failure(exc, stage):
    if getattr(exc, 'category', '') == 'TECHNICAL_PRECISION_FAILURE':
        return 'C1_3_PRECISION_INSUFFICIENT_STOP'
    reason = str(exc)
    if reason in ('CONTROLLED_OPERATOR_NUMERICAL_GATE_FAILED', 'ACTUAL_ADJOINT_FAILED'):
        return 'C1_3_OPERATOR_MISMATCH_STOP'
    if reason in ('P0_P5_NOT_READY', 'EXACT_PROPOSAL_WEIGHT_FAILED', 'ANGLE_GUARD_FAILED'):
        return 'C1_3_GUARANTEE_UNRESOLVED_STOP'
    if stage in ('R1_RESTORE', 'SCHEDULE') and isinstance(exc, (ValueError, RuntimeError)):
        return 'C1_3_INPUT_IDENTITY_STOP'
    return 'C1_3_TECHNICAL_INCONCLUSIVE_STOP'

class Journal:
    def __init__(self, directory, stream=None, scope='LIVE_H4'):
        self.directory = Path(directory)
        self.stream = sys.stdout if stream is None else stream
        self.first = None
        self.scope = scope

    def emit(self, event_name, /, **details):
        record = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'event': event_name, 'details': details}
        try:
            self.stream.write(json.dumps(record) + '\n')
            self.stream.flush()
        except Exception as exc:
            # Preserve stdout errors separately; never replace the scientific/technical cause.
            with (self.directory / 'logger_failures.jsonl').open('a') as handle:
                handle.write(json.dumps({'event': event_name, 'type': type(exc).__name__,
                                         'message': str(exc), 'primary_failure_replaced': False}) + '\n')

    def fail(self, exc, stage, classification):
        if self.first is None:
            self.first = {'schema': 'C1_3_R1_FAILURE_V1', 'stage': stage,
                'classification': classification, 'type': type(exc).__name__, 'message': str(exc),
                'traceback': ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
                'state_identity': getattr(exc, 'record', None),
                'F3_failure': exc.to_dict() if callable(getattr(exc, 'to_dict', None)) else None,
                'scope': self.scope, 'scientific_attempts': 1 if self.scope == 'LIVE_H4' else 0,
                'retry': 0, 'mandatory_STOP': True}
            exclusive_json(self.directory / 'failure.json', self.first)
        return self.first

    def terminal(self, classification, stage, **details):
        if self.first is not None:
            classification, stage = self.first['classification'], self.first['stage']
        result = {'schema': 'C1_3_R1_TERMINAL_V1', 'classification': classification,
                  'last_stage': stage, 'scope': self.scope,
                  'scientific_attempts': 1 if self.scope == 'LIVE_H4' else 0, 'retry': 0, **details}
        exclusive_json(self.directory / 'terminal.json', result)
        self.emit('TERMINAL', classification=classification)
        return result
