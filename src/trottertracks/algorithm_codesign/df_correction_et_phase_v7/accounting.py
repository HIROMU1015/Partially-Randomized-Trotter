"""Saved-sequence checks before actual T accounting; no new synthesis."""
import hashlib
from fractions import Fraction as F
from ..df_correction_comparison.angles import Angle


def validate_saved_sequence(row, contract):
    Angle.load(row['angle'])
    sequence = row['sequence']
    if (not isinstance(sequence, str) or len(sequence) > contract['caps']['synthesis']['sequence_characters']
        or any(c not in 'HTtSXW' for c in sequence)
        or hashlib.sha256(sequence.encode()).hexdigest() != row['sequence_sha256']
        or sequence.count('T')+sequence.count('t') != row['T_count']
        or sequence.count('t') != row['Tdagger_count']
        or sequence.count('W') != row['W_count']
        or row.get('error_pass') is not True
        or not F(0) <= F(row['strict_operator_error_upper']) <= F(contract['T']['gate_precision'])):
        raise PermissionError('Saved strict synthesis sequence identity/error mismatch')
    return True

