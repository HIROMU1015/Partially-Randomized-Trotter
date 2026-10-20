"""Pre-execution metadata-only seal; no H4 Green query, matrix or state evaluation."""
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
from support import HERE, ROOT, R1, BASE, FIXED, MODEL, canonical, digest, environment, git, save

def row(path):
    path = Path(path)
    return {'path': str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            'bytes': path.stat().st_size, 'sha256': digest(path)}

def main():
    assert not (HERE / 'result').exists(), 'science already attempted'
    artificial = json.loads((HERE / 'artificial_checks.json').read_text())
    assert artificial['status'] == 'PASS'
    for path in HERE.glob('*.py'): ast.parse(path.read_text(), filename=str(path))
    core = json.loads((R1 / 'result/exact_model_core.json').read_text())
    assert hashlib.sha256(canonical(core)).hexdigest() == MODEL
    labels = []
    for chunk in core['label_chunks_in_order']:
        path = R1 / 'result' / chunk['path']
        assert digest(path) == chunk['sha256']
        labels.extend(json.loads(path.read_text()))
    # Result-independent: stable label order, exact sign, saved support/basis metadata.
    a = next(r['label'] for r in labels if r['sign'] == -1)
    b = next(r['label'] for r in labels if r['sign'] == 1)
    c = next(r['label'] for r in labels if r['basis_id'] != labels[a]['basis_id'] and r['label'] not in (a, b))
    d = next(r['label'] for r in labels if r['sign'] == 1 and len(r['support']) == 2 and r['label'] not in (a, b, c))
    assert len({a, b, c, d}) == 4
    events = [{'word': list(u), 'child': i} for u, i in
              (((), a), ((), d), ((a, b), c), ((c, a), d), ((a, b, c, d), b), ((d, c, b, a), c))]
    save(HERE / 'execution_contract.json', {
        'schema': 'C1_3_FIXED_R1_SELECTED_EVENT_CORRECTNESS_V1', 'base_commit': BASE,
        'model_sha256': MODEL, 'partition_hash': core['partition_hash'],
        'snapshot': {'path': 'artifacts/pr2_s0_s1_validation/2026-09-28/h4_1p00_rank12_development_v1.npz',
                     'sha256': core['snapshot_sha256']},
        'model': 'linear H4 1.00 Angstrom / STO-3G / DF12 / R1 new partition / 8 system qubits / 324 labels / 13 bases',
        'T': '4/5', 'q_outer': 1, 'r_RTE': 4, 'm': 5, 'sigma': 1,
        'outer': 'symmetric second order PF; forward deterministic half, one tail occurrence with four substeps, reverse half',
        'adoption_scope': 'limited correctness only, not performance/optimal scientific condition',
        'precision': {'root_bits': 4096, 'probability_bits': 2048, 'eta': '1/1000000000000',
                      'rho': '1/1000000000000', 'angle_tau': '1/1000000000000',
                      'angle_root_bits': 256, 'angle_terms': 64},
        'precision_policy': 'one fixed supplied precision; F1/F3 P0-P5 exact checks inside one-shot; failure stops, no refinement',
        'event_selection': {'a': a, 'b': b, 'c': c, 'd': d,
             'rule': 'a first negative; b first positive; c first different-basis label from a excluding a,b; d first positive ZZ excluding a,b,c. Stable R1 label order. Two root, two length-2, two length-4 events as listed.',
             'numerical_results_consulted': False}, 'events': events,
        'reference': 'saved local binary64 matrices composed in time order independently of gate restoration; Q=sign B P B-adjoint; analytic cos/sin atan(s/a), ordered Q product, strict block diag(I,V), no global phase alignment',
        'axes': ['X', 'Y'], 'state_sha256': 'c9aca811b5c023772d148d0331c82958bac6824b593a367cb65a5f89937f4f63',
        'state_tolerance': 1e-12,
        'state_criterion_source': 'R1 result/state_checks.json existing_state_tolerance',
        'operator_tolerance': 1e-10,
        'operator_criterion_source': FIXED + ':src/trottertracks/algorithm_codesign/g10_comparison.py:73-85',
        'criterion_application': 'same strict controlled phase/order/composition diagnostic; Frobenius <=1e-10 is sufficient for the existing spectral-norm criterion; not rigorous arithmetic/native certificate',
        'XY_status': 'saved-state expectation diagnostic; no new rigorous tolerance or sampling/confidence PASS',
        'gates': ['immutable identities', 'exact p positive sum one', 'P0-P5 ready',
                  'exact proposal*weight=coefficient', 'F3 angle enclosure',
                  'controlled and actual-adjoint Frobenius errors <=1e-10'],
        'output': 'result/{attempt,execution_identity,schedule,preflight,event_NN_sidecar,event_NN_parent,event_NN_comparison,summary,failure,terminal}.json plus raw stdout/stderr',
        'runs': 1, 'retry': 0, 'failure': 'preserve log/failure/terminal; stop; do not enter C1-4',
        'success': 'C1-4 documentation/static survey only, then publish/preserve/mandatory STOP',
        'prohibited': ['molecular reconstruction', 'eigendecomposition', 'DF factorization', 'full H4 event enumeration',
                       'random trajectories', 'shots', 'native compile/transpile/synthesis', 'N/resource ranking', 'source repair/retry after science starts'],
        'resource_policy': 'no agent-added CPU/RAM/worker caps; task-local caches only; shared environment unchanged'})
    inputs = [row(p) for p in sorted((R1 / 'result').rglob('*')) if p.is_file()]
    inputs += [row(ROOT / 'artifacts/pr2_s0_s1_validation/2026-09-28/h4_1p00_rank12_development_v1.npz')]
    assert inputs[-1]['sha256'] == core['snapshot_sha256']
    save(HERE / 'input_manifest.json', {'base_commit': BASE, 'model_sha256': MODEL, 'files': inputs})
    save(HERE / 'environment_identity.json', environment())
    runtime_paths = set()
    previous_env = json.loads((R1 / 'environment_identity.json').read_text())
    for package in previous_env['packages'].values(): runtime_paths.update(package['metadata_file_sha256'])
    for record in json.loads((R1 / 'critical_runtime_files.json').read_text())['files']: runtime_paths.add(record['path'])
    qiskit = importlib.metadata.distribution('qiskit')
    for name in ('circuit/quantumcircuit.py', 'circuit/controlledgate.py', 'circuit/instruction.py',
                 'circuit/library/standard_gates/rz.py', 'circuit/library/standard_gates/rzz.py',
                 'circuit/library/standard_gates/xx_plus_yy.py', 'quantum_info/states/statevector.py'):
        runtime_paths.add(str(qiskit.locate_file('qiskit/' + name)))
    save(HERE / 'runtime_manifest.json', {'scope': 'fixed executable/version/metadata-RECORD plus critical matrix, gate, basis, and statevector implementations; no claim of a hermetic OS',
                                        'files': [row(p) for p in sorted(runtime_paths)]})
    baseline = json.loads((R1 / 'preservation_baseline.json').read_text())
    for record in baseline['private_originals']:
        assert digest(record['path']) == record['sha256']
    save(HERE / 'preservation_baseline.json', {'base_commit': BASE, 'rule': 'additions only in dedicated namespace; no existing tracked blobs may change',
         'private_originals': baseline['private_originals'], 'other_worktrees': 'untouched; unrelated HEAD/status changes do not invalidate this run'})
    science = []
    for name in git('ls-files', 'src/trotterlib', 'src/trottertracks').splitlines():
        path = ROOT / name
        if path.is_file():
            original = subprocess.check_output(['git', 'show', BASE + ':' + name], cwd=ROOT)
            assert path.read_bytes() == original, 'existing source changed: ' + name
            science.append(row(path))
    fixed_names = ['src/trottertracks/algorithm_codesign/' + s for s in
        ('return_aggregation.py', 'g7_generator.py', 'g7_provider.py', 'g6_f3_contract.py', 'g6_f3_precision.py', 'g6_f3_generator.py',
         'g8_bounds.py', 'g9_p5.py', 'g10_comparison.py', 'rte_development/accounting.py', 'rte_development/pipeline.py')]
    fixed_names += ['src/trotterlib/' + s for s in ('df_rte_tail.py', 'df_rte_qiskit.py', 'rpe_hadamard_interrogation.py', 'df_partial_s2.py')]
    for name in fixed_names:
        assert (ROOT / name).read_bytes() == subprocess.check_output(['git', 'show', FIXED + ':' + name], cwd=ROOT)
    local = [row(p) for p in sorted(HERE.iterdir()) if p.is_file() and p.name != 'source_manifest.json']
    save(HERE / 'source_manifest.json', {'base_commit': BASE, 'fixed_science_commit': FIXED,
          'fixed_highlighted_sources': fixed_names, 'files': science + local,
          'self_hash_bound_by_authorization': True})
    print(json.dumps({'metadata_sealed': True, 'events': events, 'source_files': len(science), 'input_files': len(inputs)}))

if __name__ == '__main__': main()
