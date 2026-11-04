"""Stdlib-only observer policy; also importable by GDB's separate Python."""
import json
import os
from pathlib import Path
import signal


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def mode_supported(value):
    return type(value) is int and 0 <= value < (1 << 32) and value & 0xffc0 == 0x1f80


REQUIRED = ('cblas_zgemv64_', 'zgemv_t_PRESCOTT', 'zdotc_k_PRESCOTT',
            'CDOUBLE_multiply_FMA3__AVX2', 'CDOUBLE_subtract_FMA3__AVX2', '_Py_c_abs', 'hypot')


def validate_probe(counts, columns):
    if any(counts.get(name, 0) < 1 for name in REQUIRED) or columns != 70:
        raise ValueError('Unconfirmed numerical dispatch family')


class Checkpoints:
    def __init__(self, output):
        self.output, self.sequence = Path(output), 0

    def __call__(self, phase):
        self.sequence += 1
        write_json(self.output/'runtime_request.json', {'sequence': self.sequence, 'phase': phase})
        # Without an observing debugger this terminates before any real loading.
        os.kill(os.getpid(), signal.SIGTRAP)
        record = json.loads((self.output/'runtime.json').read_text())
        if (record.get('sequence') != self.sequence or record.get('phase') != phase
                or record.get('status') != 'RUNTIME_CONDITIONS_CONFIRMED'):
            raise ValueError('Runtime conditions unconfirmed; STOP')
        return record
