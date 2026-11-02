"""Read-only G10 preparation verification; standard library, no science imports.

Checks stored evidence and source identities. It never recomputes registered
coefficients, angles, budgets, or circuits and never creates a one-shot marker.
"""
import ast
import hashlib
import json
from fractions import Fraction as F
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PREP = Path('artifacts/track_b_g10_degree_preparation/2026-10-10')


def verify(root):
    root = Path(root)
    checks = []

    def require(condition, message):
        if not condition:
            raise AssertionError(message)
        checks.append(message)

    def read(relative):
        relative = Path(relative)
        if relative.suffix.lower() == '.npz':
            raise PermissionError('NPZ excluded before any access')
        return (root / relative).read_bytes()

    def data(name):
        return json.loads(read(PREP / name))

    c, a = data('contract_v1.json'), data('authorization.json')
    require(a['status'] == 'PENDING_SEPARATE_G10_AUTHORIZATION'
            and a['science_execution_authorized'] is False
            and a['source_commit'] is None and a['contract_sha256'] is None
            and a['explicit_execution_instruction'] is None,
            'separate G10 source review and authorization pending')
    require(a['runs'] == c['runs'] == 1 and a['retries'] == c['retries'] == 0
            and a['mandatory_STOP'] is c['mandatory_STOP'] is True,
            'one-shot/retry0/mandatory STOP fixed')
    require(c['degrees'] == [3, 5, 7] and c['new_degrees'] == [3, 7]
            and c['rows'] == 17 and c['new_rows'] == 11
            and c['saved_rebudget_rows'] == 6 and c['axes'] == 34,
            'registered finite comparison shape fixed, not evaluated')
    require(c['input']['p'] == ['1/5', '3/10', '1/2']
            and c['input']['x'] == '5/7' and c['input']['system_qubits'] == 3,
            'fixed G9 synthetic provider context retained')
    require(34 * F(c['alpha_axis']) == F(49, 1000)
            and 17 * F(c['resource_failure_per_row']) == F(1, 1000)
            and F(c['familywise_failure']) == F(1, 20),
            'familywise failure union arithmetic exact')
    require(c['finite_cost_aware_proposal_search'] is False
            and c['cache_entries'] >= c['static_cap_derivation']['total_angle_upper']
            and c['caps']['synthesis_keys'] == c['static_angle_upper_new'] == 162,
            'finite key/cache caps fixed without new proposal search')
    result = root / c['result_directory']
    require(not result.exists(), 'new G10 result directory and marker absent')

    m = data('source_manifest_v1.json')
    require(m['focused_tests_passed'] is True,
            'stored focused verification passed')
    for relative, digest in m['sha256'].items():
        raw = read(relative)
        require(hashlib.sha256(raw).hexdigest() == digest,
                'critical identity unchanged: ' + relative)
        if relative.endswith('.py'):
            ast.parse(raw, filename=relative)
    ledger = json.loads(read(c['protected_ledger']))
    for relative, record in ledger.items():
        raw = read(relative)
        if relative in c['append_only_paths']:
            require(len(raw) >= record['bytes'], 'old prefix retained: ' + relative)
            raw = raw[:record['bytes']]
        require(hashlib.sha256(raw).hexdigest() == record['sha256'],
                'protected history unchanged: ' + relative)
    require(hashlib.sha256(read(c['reuse_G9_result'])).hexdigest() == c['reuse_G9_sha256'],
            'G9 raw anchor unchanged')
    require(hashlib.sha256(read(c['scope_snapshot'])).hexdigest() == c['scope_sha256'],
            'adopted GPT review unchanged')
    require(hashlib.sha256(read(c['tool_identity']['path'])).hexdigest()
            == c['tool_identity']['sha256'], 'fixed tool identity retained')

    tests = data('focused_tests_v1.json')
    require(tests['passed'] is True and tests['tests_passed'] == 41
            and tests['new_live_synthesis_calls'] == 0
            and tests['registered_P3_P7_science_opened'] is False
            and tests['full_test_suite'] is False,
            '41 focused off-domain tests; no live backend/registered science')
    require(hashlib.sha256(read(PREP / 'focused_tests_v1.txt')).hexdigest()
            == tests['stdout_sha256'], 'focused test transcript identity retained')
    preflight = data('runtime_preflight_v1.json')
    require(preflight['pending_launch_refused'] is True
            and preflight['runtime']['packages_match'] is True
            and preflight['live_synthesis_calls'] == 0,
            'stored runtime preflight and pending launch refusal passed')
    saved, recount = data('saved_policy_audit_v1.json'), data('saved_native_recount_v1.json')
    require(saved['input_sha256'] == recount['input_sha256'] == c['reuse_G9_sha256']
            and recount['bindings_checked'] == 945
            and recount['no_circuit_build_matrix_synthesis_or_guard_execution'] is True,
            'saved-only audits bound to 945 original direct bindings')
    return {'status': 'G10_PREPARATION_IDENTITIES_AND_BOUNDARIES_PASS',
            'checks_passed': len(checks), 'critical_paths': len(m['sha256']),
            'protected_paths': len(ledger), 'registered_P3_P7_science_opened': False,
            'live_backend_calls': 0, 'source_only_next_gate': True}


if __name__ == '__main__':
    print(json.dumps(verify(ROOT), indent=2))
