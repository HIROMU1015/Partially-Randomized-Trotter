"""Deterministic artificial dispatch probe. No saved arrays, RNG or real H4."""
import ctypes
from pathlib import Path

import numpy as np

library = next((Path(np.__file__).parent.parent/'numpy.libs').glob('libopenblas*'))
blas = ctypes.CDLL(str(library))
for name in ('openblas_get_config64_', 'openblas_get_corename64_'):
    function = getattr(blas, name)
    function.restype = ctypes.c_char_p
    print('PROFILE', name, function().decode(), flush=True)
print('PROFILE numpy', np.__version__, flush=True)
for layout in ('C', 'F', 'strided'):
    for size in (4, 70):
        raw = np.arange(size*size, dtype=np.float64).reshape(size, size)/1024
        matrix = np.array(raw+1j*raw.T, dtype=np.complex128, order=layout if layout != 'strided' else 'C')
        vector = (np.arange(size, dtype=np.float64)+1j)/16
        if layout == 'strided':
            large = np.zeros((size*2, size*2), complex)
            large[::2, ::2] = matrix
            matrix = large[::2, ::2]
        print('ARTIFICIAL', layout, size, 'matrix strides', matrix.strides,
              'vector strides', vector.strides, flush=True)
        value = matrix @ vector
        np.vdot(vector, value)
        np.linalg.norm(vector)
        scaled = (.25-.5j)*value
        scaled-value
print('ARTIFICIAL complex abs', flush=True)
abs(complex(.75, .25))
print('DONE', flush=True)
