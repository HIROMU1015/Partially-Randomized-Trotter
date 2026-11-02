"""Phase A: artificial bitwise/local-reference and existing operator regression."""
import hashlib
import importlib.util
import io
import json
import struct
import numpy as np
from support import HERE, ROOT, PRIOR, RECOVERY, digest, save
from bitwise import restore_local, independent_bytes, legacy_local, difference, raw_hash
from journal import Journal

def fixture(values):
    return {'shape': [2, 2],
            'real_hex': [[r.hex() for r, _ in row] for row in values],
            'imag_hex': [[i.hex() for _, i in row] for row in values]}

def main():
    tiny = float.fromhex('0x0.0000000000001p-1022')
    fixtures = [
        fixture([[(0., 0.), (-0., 0.)], [(0., -0.), (-0., -0.)]]),
        fixture([[(1.25, -2.5), (-3.5, 4.75)], [(-6., -8.), (10., 11.)]]),
        fixture([[(tiny, -tiny), (-tiny, tiny)], [(float.fromhex('0x1.fffffffffffffp+1023'), -0.), (-0., tiny)]]),
    ]
    checks = []
    for index, raw in enumerate(fixtures):
        matrix, expected = restore_local(raw), independent_bytes(raw)
        expected_hash = hashlib.sha256(expected).hexdigest()
        assert matrix.dtype.str == '<c16' and matrix.flags.c_contiguous
        assert matrix.shape == tuple(raw['shape']) and matrix.tobytes(order='C') == expected
        assert raw_hash(matrix) == expected_hash
        assert all(matrix.real[i,j].hex() == raw['real_hex'][i][j] and matrix.imag[i,j].hex() == raw['imag_hex'][i][j]
                   for i in range(2) for j in range(2))
        old = legacy_local(raw).tobytes(order='C')
        delta = difference(expected, old)
        checks.append({'fixture': index, 'saved_hex': raw, 'expected_sha256': expected_hash,
            'actual_sha256': raw_hash(matrix), 'expected_bytes_hex': expected.hex(),
            'new_bytes_hex': matrix.tobytes().hex(), 'legacy_bytes_hex': old.hex(),
            'legacy_difference': delta, 'new_exact_bytes_equal': True})
    assert checks[0]['legacy_difference']['all_differences_are_zero_sign_bits']
    # Shared first-failure journal; artificial scope must not count as H4 execution.
    directory = HERE / 'artificial_failure_case'; directory.mkdir(exist_ok=False)
    journal = Journal(directory, stream=io.StringIO(), scope='ARTIFICIAL_REFERENCE_FIXTURE')
    try: raise ValueError('ARTIFICIAL_REFERENCE_BYTES_MISMATCH')
    except ValueError as exc: journal.fail(exc, 'PHASE_A_FIXTURE', 'C1_3_INPUT_IDENTITY_STOP')
    before = (directory / 'failure.json').read_bytes()
    journal.emit('FAILURE', event='payload', step='payload')
    journal.fail(RuntimeError('secondary'), 'SECONDARY', 'C1_3_TECHNICAL_INCONCLUSIVE_STOP')
    journal.terminal('C1_3_TECHNICAL_INCONCLUSIVE_STOP', 'SECONDARY', mandatory_STOP=True)
    assert (directory / 'failure.json').read_bytes() == before
    assert json.loads((directory / 'terminal.json').read_text())['classification'] == 'C1_3_INPUT_IDENTITY_STOP'
    spec = importlib.util.spec_from_file_location('immutable_artificial_suite', PRIOR / 'test_artificial.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); module.main()
    import reference, adapter
    assert reference.__file__ == str(HERE / 'reference.py') and adapter.__file__ == str(PRIOR / 'adapter.py')
    save(HERE / 'phase_a_checks.json', {'status': 'PASS', 'artificial_only': True,
         'fixtures': checks, 'fixture_count': len(checks),
         'signed_zeros_real_imag_and_both': True, 'finite_subnormal_values': True,
         'exact_binary64_hex_and_complex_storage_C_order': True,
         'independent_expected': 'row-major struct.pack(<dd), no adapter gate or operator output',
         'failure_logger_terminal_preserves_first_failure': True,
         'connection_regression': '10 controlled comparisons and 63 artificial first-moment events, unchanged criteria',
         'reference_sha256': digest(HERE / 'reference.py'), 'bitwise_helper_sha256': digest(HERE / 'bitwise.py'),
         'adapter_sha256': digest(PRIOR / 'adapter.py'), 'H4_arrays_or_basis_opened': False,
         'H4_scientific_attempts': 0, 'legacy_H4_cause_inferred_from_artificial_only': False})
    print(json.dumps({'phase_A': 'PASS', 'bitwise_fixtures': len(checks), 'operator_comparisons': 10}), flush=True)

if __name__ == '__main__': main()
