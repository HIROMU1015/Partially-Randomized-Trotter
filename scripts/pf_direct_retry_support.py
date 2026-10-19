"""Small saved-format adapters; no real inputs or scientific imports at import."""
import hashlib
import math

from pf_direct_compile_contract import require


def array_record(array, np):
    # Match the original h2/h4 array_record exactly, including scalar -> shape [1].
    contiguous=np.ascontiguousarray(array)
    require(not contiguous.dtype.hasobject,'OBJECT_ARRAY_NOT_ALLOWED')
    return dict(shape=list(contiguous.shape),dtype=contiguous.dtype.str,
                sha256=hashlib.sha256(contiguous.tobytes()).hexdigest())


def scalar_value(original, receipt):
    require(isinstance(receipt,dict),'OWN_PREFIX_SCALAR_REQUIRED')
    selected=receipt
    if 'receipt' in receipt:
        require(receipt.get('policy')=='REUSED_SAVED_P21_FULL_FOCK_SCALAR'
                and isinstance(receipt['receipt'],dict),'UNKNOWN_NESTED_SCALAR_SCHEMA')
        selected=receipt['receipt']
    field='scalar' if 'scalar' in selected else 'C'
    require(field in selected and not isinstance(selected[field],bool),'SAVED_SCALAR_FIELD_REQUIRED')
    value=float(selected[field])
    require(math.isfinite(value) and math.isfinite(original),'SAVED_SCALAR_FINITE')
    result=value if field=='scalar' else float(original)+value
    require(math.isfinite(result),'SAVED_SCALAR_FINITE')
    return result
