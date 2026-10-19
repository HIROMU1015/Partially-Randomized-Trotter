"""Executable acceptance checks. Evidence is technical, not SOURCE adoption."""
from hashlib import sha256
from pathlib import Path
from .g6_w2c_b3_offline_result import BASE_COMMIT, canonical, projection

JUDGMENTS = ('PASS', 'FAIL', 'PENDING_EVIDENCE', 'REQUIRES_CONTRACT_CHANGE', 'NOT_APPLICABLE')
PARALLEL = {
 'P01': 'Global trial identity and per-pair sequential order',
 'P02': 'Worker failure/current attempt/inflight and multiple simultaneous failures',
 'P03': 'Global started/completed/failed/unknown call accounting within one-shot budget',
 'P04': 'Global output/log budget, output collision and unique namespaces',
 'P05': 'Worker abnormal exit and saved failure prefix without invented completion',
 'P06': 'Correctness/phase barrier and no unapproved early full-N calls',
 'P07': 'Atomic shared claim and reservation remains spent after STOP save failure',
 'P08': 'STOP notification: zero new trial starts after notification; already-started inflight counted',
 'P09': 'Worker CPU/wall allocation, overlap, observer paths and no parent-child wall sum',
 'P10': 'Global summary/receipt binds every pair and all worker failures',
 'P11': 'No retry/resume/claim reuse after any worker or output failure',
}


def record(check_id, target, fixed_commit, expected, observed, judgment, evidence, unresolved=None):
    if judgment not in JUDGMENTS:
        raise ValueError('Unknown acceptance judgment')
    return dict(check_id=check_id, target_SOURCE=target, fixed_commit=fixed_commit,
                contract_schema='serial_R1_A6_F01_admission_v2 / driver_result_v1',
                expected=expected, observed=observed, judgment=judgment,
                evidence=evidence, unresolved=unresolved)


def canonical_checks(candidate):
    """Candidate callback is supplied by artificial tests, never loaded by CLI."""
    from fractions import Fraction as F
    values = [None, True, 0, -(1 << 4096), F(-3, 7), ['é', '\ud800', '\x7f', '\u2028', '\n', '漢字'],
              {1: object(), '1': F(2, 3)}, {'phase': 2, 'weight': F(1, 2), 'word': (2, 1, 0)}]
    out = []
    for i, value in enumerate(values):
        try:
            wanted = canonical(value)
            got = candidate(value)
            ok = type(got) is str and got.encode() == wanted.encode()
            observed = sha256(got.encode()).hexdigest() if type(got) is str else str(type(got))
        except Exception as exc:
            ok, observed = False, type(exc).__name__ + ':' + str(exc)
        out.append(record('C_CODEC_' + str(i), 'supplied artificial canonical callback', BASE_COMMIT,
                          'Exact bytes including str-key collision, Unicode/surrogate/DEL, int/Fraction',
                          observed, 'PASS' if ok else 'FAIL', 'Compared to driver-compatible exact codec'))
    return out


def identity_checks(root, baseline, candidate, pins, *, candidate_commit):
    rows = []
    for name, expected in pins.items():
        path = (Path(root)/name).resolve()
        inside = path.is_relative_to(Path(root).resolve())
        actual = sha256(path.read_bytes()).hexdigest() if inside and path.is_file() else None
        rows.append(record('A_PIN_' + Path(name).stem, name, candidate_commit, expected, actual,
                           'PASS' if actual == expected else 'FAIL', 'Complete named SOURCE bytes SHA256'))
    for k, expected in projection(baseline).items():
        actual = projection(candidate).get(k)
        rows.append(record('A_CONTRACT_' + k, 'explicit contract projection', candidate_commit,
                           expected, actual, 'PASS' if actual == expected else 'REQUIRES_CONTRACT_CHANGE',
                           'Exact structure equality; trace equality does not waive contract changes'))
    return rows


def observer_check(baseline_paths, candidate_paths, baseline_functions, candidate_functions, commit):
    same = set(baseline_paths) == set(candidate_paths) and set(baseline_functions) == set(candidate_functions)
    return record('E_OBSERVER_SCOPE', 'FunctionObserver paths/functions', commit,
                  dict(paths=baseline_paths, functions=baseline_functions),
                  dict(paths=candidate_paths, functions=candidate_functions),
                  'PASS' if same else 'REQUIRES_CONTRACT_CHANGE',
                  'Explicit target sets; changed function tree requires mapping and new SOURCE pins',
                  None if same else 'Do not compare identical function labels across changed call trees.')


def build_report(root, baseline, pins, *, unit_evidence=None, verification=None, costs=None):
    """Current baseline acceptance + future candidate pending checks, never adoption."""
    rows = identity_checks(root, baseline, baseline, pins, candidate_commit=BASE_COMMIT)
    if unit_evidence is not None:
        for suite in unit_evidence:
            # Execute tests externally and bind their source/log bytes in Evidence Packet.
            # A test record is evidence, not authorization or adversarial attestation.
            for test_id in suite['collected']:
                passed = test_id in suite.get('passed_ids', [])
                rows.append(record('TEST:' + test_id, suite['source_path'], BASE_COMMIT,
                                   'Artificial regression succeeds; scientific modules absent',
                                   'passed' if passed else 'not established', 'PASS' if passed else ('FAIL' if test_id in suite.get('failed_ids', []) else 'PENDING_EVIDENCE'),
                                   dict(log_sha256=suite['log_sha256'], source_sha256=suite['source_sha256'])))
    else:
        rows.append(record('D_BASELINE_REGRESSION', 'Admission/Host/Boundary tests', BASE_COMMIT,
                           'Execute current artificial one-shot/STOP tests', None, 'PENDING_EVIDENCE', None,
                           'No runtime test evidence supplied.'))
    for cid, target, state in (('B_STREAM_EQUIVALENCE', 'Artificial A/B, X/Y, prefix, two-pass event hashes', verification),
                                ('E_COST_PARTITIONS', 'Coarse/fine, shared setup, parent/child, memory/I/O', costs)):
        expected = 'VERIFIED_COMPLETE' if cid.startswith('B') else 'COMPLETE_ARTIFICIAL_DESCRIPTIVE_COSTS'
        observed = state.get('status') if state else None
        rows.append(record(cid, target, BASE_COMMIT, expected, observed,
                           'PASS' if observed == expected else 'PENDING_EVIDENCE', 'Artificial example + runtime fault tests'))
    for cid, target in (('FUSED_CANDIDATE_RUNTIME', 'ad3117d287c4c28c9b861b3b8dffa7bfbba407e1'),
                        ('FUSED_OBSERVER_MAPPING', 'New helper SOURCE and FunctionObserver functions')):
        rows.append(record(cid, target, 'ad3117d287c4c28c9b861b3b8dffa7bfbba407e1',
                           'Run canonical/F01/source/guard tests against proposed integrated SOURCE', None,
                           'PENDING_EVIDENCE', 'Fixed published candidate documentation read-only',
                           'Candidate SOURCE not imported or integrated in this task.'))
    for cid, text in PARALLEL.items():
        rows.append(record(cid, 'unpublished parallel adapter', None, text, 'PENDING_SERVER_CONTRACT',
                           'PENDING_EVIDENCE', None, 'Published contract/schema/SOURCE and artificial logs required.'))
    return dict(schema='b3_integration_acceptance_v1', scientific_adoption=False,
                parallel_adapter_status='PENDING_SERVER_CONTRACT', checks=rows,
                totals={j: sum(r['judgment'] == j for r in rows) for j in JUDGMENTS})
