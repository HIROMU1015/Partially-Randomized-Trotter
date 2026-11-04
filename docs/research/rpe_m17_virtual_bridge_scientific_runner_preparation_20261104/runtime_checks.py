"""Same-process artificial probe and scientific array-layout checks."""
import ctypes
import hashlib
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def binary_profile():
    library = next((Path(np.__file__).parent.parent/'numpy.libs').glob('libopenblas*'))
    blas = ctypes.CDLL(str(library))
    core = blas.openblas_get_corename64_
    core.restype = ctypes.c_char_p
    # Resolve the actual mapped libm rather than a guessed loader symlink.
    candidates = {line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                  if '/' in line and Path(line.split()[-1]).name == 'libm.so.6'}
    if len(candidates) != 1:
        raise ValueError('Ambiguous or unavailable libm mapping')
    import numpy.core._multiarray_umath as module
    return {'numpy': np.__version__, 'blas_core': core().decode(), 'blas_sha256': sha(library),
            'numpy_binary_sha256': sha(module.__file__), 'libm_sha256': sha(candidates.pop())}


def validate_array(array, shape):
    if (not isinstance(array, np.ndarray) or array.dtype != np.complex128 or array.shape != shape
            or not array.flags.c_contiguous or not array.flags.aligned
            or array.strides != tuple(np.empty(shape, dtype=complex).strides)
            or not np.isfinite(array).all()):
        raise ValueError('Unsupported complex128 shape/stride/layout or nonfinite value')


def probe(checkpoint):
    checkpoint('probe_start')
    matrix = np.eye(70, dtype=complex)
    vector = np.asarray((np.arange(70)+1j)/128, dtype=complex)
    value = matrix @ vector
    np.vdot(vector, value)
    scaled = (.25-.5j)*value
    scaled-value
    abs(complex(.75, .25))
    return checkpoint('probe_complete')
