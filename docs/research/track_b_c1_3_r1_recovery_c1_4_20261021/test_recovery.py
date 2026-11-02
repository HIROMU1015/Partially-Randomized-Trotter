"""Phase A: artificial arrays/journal plus unchanged artificial operator suite."""
import ast
import hashlib
import importlib.util
import io
import json
import numpy as np
from typing import Any
from support import HERE, ROOT, PRIOR, digest, save, git
from state_identity import array_identity_hash, verify_state_identity, StateIdentityError
from journal import Journal

EXPECTED = 'f48616dd87ae26671cdd548203399c5bdaa044db0f91f21e1c693dc124d357c1'

class BrokenStream:
    def write(self, value): raise OSError('artificial stdout failure')
    def flush(self): pass

def main():
    source = ROOT / 'src/trotterlib/pr2_s0_s1_validation.py'
    tree = ast.parse(source.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in ('_array_hash', '_canonical_json')]
    assert len(nodes) == 2
    environment = {'np': np, 'hashlib': hashlib, 'json': json, 'Any': Any}
    # Compile only two existing pure hash functions. No validation module imports/runners.
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), environment)
    official = environment['_array_hash']
    a = np.array([[1., -2.5, 0.]], dtype='<f8')
    literal = b'<f8' + b'[1,3]' + bytes.fromhex('000000000000f03f00000000000004c00000000000000000')
    assert hashlib.sha256(literal).hexdigest() == EXPECTED
    assert official(a) == array_identity_hash(a) == EXPECTED
    raw = hashlib.sha256(a.tobytes(order='C')).hexdigest()
    assert raw != EXPECTED
    assert a.view('<u8').tobytes() == a.tobytes()
    assert array_identity_hash(a.view('<u8')) != EXPECTED
    assert a.reshape(3, 1).tobytes() == a.tobytes()
    assert array_identity_hash(a.reshape(3, 1)) != EXPECTED
    b = np.array([[1., 2., 3.], [4., 5., 6.]], dtype='<f8')
    fixtures = [a, a.view('<u8'), a.reshape(3, 1), b, np.asfortranarray(b), b[:, ::2],
                np.array([1+2j, -3-4j], dtype='<c16'), np.array([1., -2.5], dtype='>f8')]
    for value in fixtures: assert official(value) == array_identity_hash(value)
    assert array_identity_hash(np.asfortranarray(b)) == array_identity_hash(b)
    assert array_identity_hash(b[:, ::2]) == array_identity_hash(np.array(b[:, ::2], order='C'))
    assert verify_state_identity(a, EXPECTED)['matches']
    directory = HERE / 'artificial_failure_case'
    directory.mkdir(exist_ok=False)
    journal = Journal(directory, stream=io.StringIO(), scope='ARTIFICIAL_HASH_FIXTURE')
    try: verify_state_identity(a, '0' * 64)
    except StateIdentityError as primary:
        journal.fail(primary, 'STATE_HASH_ARTIFICIAL', 'C1_3_INPUT_IDENTITY_STOP')
    else: raise AssertionError('mismatch not rejected')
    before = (directory / 'failure.json').read_bytes()
    journal.emit('FAILURE', event='payload does not replace event', step='payload does not collide', status='STOP')
    journal.stream = BrokenStream()
    journal.emit('LOGGING_FAILURE_FIXTURE', step='secondary')
    journal.fail(RuntimeError('secondary must not replace primary'), 'SECONDARY', 'C1_3_TECHNICAL_INCONCLUSIVE_STOP')
    terminal = journal.terminal('C1_3_TECHNICAL_INCONCLUSIVE_STOP', 'SECONDARY', mandatory_STOP=True)
    assert (directory / 'failure.json').read_bytes() == before
    assert terminal['classification'] == 'C1_3_INPUT_IDENTITY_STOP' and terminal['last_stage'] == 'STATE_HASH_ARTIFICIAL'
    failure = json.loads(before)
    assert failure['state_identity']['expected'] == '0' * 64 and failure['state_identity']['actual'] == EXPECTED
    assert (directory / 'logger_failures.jsonl').exists() and terminal['scientific_attempts'] == 0
    # The exact same immutable connection modules are resolved by the new support path.
    spec = importlib.util.spec_from_file_location('prior_artificial_suite', PRIOR / 'test_artificial.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.main()
    operator = json.loads((HERE / 'artificial_checks.json').read_text())
    assert operator['status'] == 'PASS'
    import adapter, reference
    assert str(PRIOR / 'adapter.py') == adapter.__file__ and str(PRIOR / 'reference.py') == reference.__file__
    save(HERE / 'phase_a_checks.json', {'status': 'PASS', 'artificial_only': True,
         'formal_hash_function_reference': str(source.relative_to(ROOT)), 'reference_source_sha256': digest(source),
         'official_function_reuse': 'AST extraction of _array_hash/_canonical_json only; no full science module import',
         'artificial_hash_fixture_count': len(fixtures), 'fixed_expected': EXPECTED,
         'checks': {'official_helper_equal': True, 'raw_bytes_hash_distinct': True, 'dtype_sensitive_with_same_raw_bytes': True,
                    'shape_sensitive_with_same_raw_bytes': True, 'C_order_normalization': True,
                    'fixed_expected_not_replaced': True, 'mismatch_failure_terminal_preserved': True,
                    'logger_does_not_mask_primary_failure': True, 'immutable_connection_operator_suite_PASS': True},
         'operator_event_comparisons': len(operator['event_comparisons']),
         'adapter_sha256': digest(PRIOR / 'adapter.py'), 'reference_sha256': digest(PRIOR / 'reference.py'),
         'H4_NPZ_model_basis_opened': False, 'H4_scientific_attempts': 0,
         'independent_external_reviewer': False})
    print(json.dumps({'phase_A': 'PASS', 'hash_fixtures': len(fixtures), 'operator_events': len(operator['event_comparisons'])}), flush=True)

if __name__ == '__main__': main()
