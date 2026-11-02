"""Lossless saved-hex -> little-endian complex128, independent of gate output."""
import hashlib
import struct
import numpy as np

def raw_hash(matrix):
    return hashlib.sha256(matrix.tobytes(order='C')).hexdigest()

def parts(raw):
    shape = tuple(raw['shape'])
    if len(shape) != 2 or any(type(n) is not int or n < 1 for n in shape):
        raise ValueError('INVALID_SAVED_MATRIX_SHAPE')
    real, imag = raw['real_hex'], raw['imag_hex']
    if any(len(rows) != shape[0] or any(len(row) != shape[1] for row in rows) for rows in (real, imag)):
        raise ValueError('SAVED_HEX_SHAPE_MISMATCH')
    return shape, [[float.fromhex(v) for v in row] for row in real], [[float.fromhex(v) for v in row] for row in imag]

def restore_local(raw):
    shape, real, imag = parts(raw)
    matrix = np.empty(shape, dtype='<c16', order='C')
    matrix.real[...] = real
    matrix.imag[...] = imag
    if not np.all(np.isfinite(matrix.real)) or not np.all(np.isfinite(matrix.imag)):
        raise ValueError('NONFINITE_SAVED_MATRIX')
    return matrix

def independent_bytes(raw):
    _, real, imag = parts(raw)
    return b''.join(struct.pack('<dd', r, i) for rr, ii in zip(real, imag) for r, i in zip(rr, ii))

def legacy_local(raw):
    _, real, imag = parts(raw)
    return np.array(real) + 1j * np.array(imag)

def difference(expected, actual):
    if len(expected) != len(actual):
        return {'equal': False, 'length_mismatch': True, 'expected_bytes': len(expected), 'actual_bytes': len(actual)}
    offsets = [i for i, (a, b) in enumerate(zip(expected, actual)) if a != b]
    components = []
    for offset in range(0, len(expected), 8):
        left, right = struct.unpack('<Q', expected[offset:offset+8])[0], struct.unpack('<Q', actual[offset:offset+8])[0]
        if left != right:
            zero_sign_only = (left ^ right) == (1 << 63) and (left & ((1 << 63)-1)) == 0
            components.append({'byte_offset': offset, 'expected_binary64_bits': hex(left),
                               'actual_binary64_bits': hex(right), 'zero_sign_only': zero_sign_only})
    return {'equal': not offsets, 'first_difference_byte_offset': offsets[0] if offsets else None,
            'differing_byte_count': len(offsets), 'differing_binary64_components': components,
            'all_differences_are_zero_sign_bits': bool(offsets) and all(c['zero_sign_only'] for c in components)}
