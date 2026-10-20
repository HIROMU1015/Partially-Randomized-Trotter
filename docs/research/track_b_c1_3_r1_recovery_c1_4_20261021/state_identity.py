"""R1 formal array identity; separate from local-matrix raw-bytes hash."""
import hashlib
import json
import numpy as np

POLICY = 'R1_dtype_str__canonical_shape__C_order_bytes_SHA256_v1'

def array_identity_hash(array):
    # Same semantics as fixed pr2_s0_s1_validation._array_hash, without broad imports.
    normalized = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(normalized.dtype.str.encode())
    digest.update(json.dumps(list(normalized.shape), sort_keys=True,
                             separators=(',', ':'), allow_nan=False).encode())
    digest.update(normalized.tobytes(order='C'))
    return digest.hexdigest()

class StateIdentityError(ValueError):
    def __init__(self, record):
        super().__init__('STATE_IDENTITY_MISMATCH')
        self.record = record

def verify_state_identity(array, expected):
    array = np.asarray(array)
    record = {'schema': 'C1_3_R1_STATE_IDENTITY_V1', 'policy': POLICY,
              'expected': expected, 'actual': array_identity_hash(array),
              'dtype': array.dtype.str, 'shape': list(array.shape),
              'expected_policy': 'supplied fixed value, never replaced by observation',
              'NPZ_file_SHA256_is_a_separate_identity': True}
    record['matches'] = record['actual'] == expected
    if not record['matches']:
        raise StateIdentityError(record)
    return record
