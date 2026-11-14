"""Pauli B1 saved path/compile receipts and publication binding. No science on import.

Only JSON/Git and supplied counts are read. No sampler, circuit builder, model
fit or canonical-round identity is used by this non-configuration consumer.
"""
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

AXES = {'X': 'cosine', 'Y': 'sine'}
CHECKS = {'low_control', 'high_control_leakage', 'independent_full_space_vector',
          'Hadamard_cosine', 'Hadamard_sine', 'compiled_vector_cosine', 'compiled_vector_sine'}


def blob(root, commit, path):
    if (not re.fullmatch('[0-9a-f]{40}', commit or '') or not isinstance(path, str)
            or Path(path).is_absolute() or '..' in Path(path).parts):
        raise ValueError('Pinned commit and repository-relative receipt path required')
    try:
        return subprocess.check_output(['git', 'show', commit + ':' + path], cwd=root, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        raise ValueError('Published receipt missing: ' + path) from exc


def read(root, reference):
    value = json.loads(blob(root, reference['commit'], reference['path']))
    try:
        for key in reference.get('selector', []):
            value = value[key]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError('Published receipt selector missing') from exc
    return value


def locator(path, selector=(), commit=None):
    return dict(commit=commit, path=path, selector=list(selector))


def check_path(path, *, request, dictionary, cohort, index, seeds):
    result = path['result']
    if (path['index'] != index or path['cohort'] != cohort or path['request'] != request
            or path.get('dictionary_identity') != dictionary or result['seeds'] != seeds
            or result['status'] != 'PASS' or result['bit_order']['pass'] is not True):
        raise ValueError('Path request/dictionary/cohort/index/seed/gate differs')
    checks = {g['check']: g for g in result['checks']}
    if not CHECKS <= set(checks):
        raise ValueError('Path action/compiled-vector gate missing')
    for key in CHECKS:
        g = checks[key]; limit = 1e-7 if key.startswith('compiled_vector') else 1e-9
        if (not math.isfinite(g['error']) or not 0 <= g['error'] < g['tolerance'] <= limit):
            raise ValueError('Path action/compiled-vector gate failed')
    for key in AXES.values():
        f = result['fingerprints'][key]
        if not f['measurement'] or f['wrapper_qubits'] != result['n'] + 1:
            raise ValueError('Measured full wrapper required')


def read_parents(root, references, expected, *, source_files):
    """Resolve ordered published parent paths, retaining their original SOURCE.

    execution_commit is the parent's code SHA, commit is its later publication.
    SOURCE files must agree with the current replay code, even after doc-only
    publication commits. All checks precede context/build/compile in the caller.
    """
    if len(references) != len(expected):
        raise ValueError('Ordered parent receipt count differs')
    parents = []
    for ref, want in zip(references, expected, strict=True):
        if ref.get('selector') != ['child', 'paths', want['index']]:
            raise ValueError('Parent order/path selector differs')
        data = read(root, dict(ref, selector=[])); source = data['source']
        if (data['status'] != 'FINITE_CELL_COMPLETE' or data['child']['status'] != 'PASS'
                or source['commit'] != ref.get('execution_commit') or source['caller'] != want['caller']
                or source['result_path'] != ref['path'] or source['cell'] != want['cell_id']
                or source['cohort'] != want['cohort'] or source['step_seeds'] != want['cell_seeds']
                or source['prefix'] != want['request']['prefix'] or source['kind'] != 'B1-CAL'):
            raise ValueError('Parent SOURCE/result binding differs')
        for name, current in source_files.items():
            if blob(root, source['commit'], name) != current:
                raise ValueError('Parent execution SOURCE differs: ' + name)
        parent = read(root, ref)
        check_path(parent, **{k: want[k] for k in ('request', 'dictionary', 'cohort', 'index', 'seeds')})
        parent['receipt_source'] = deepcopy(ref)
        parents.append(parent)
    return parents


def compile_receipts(paths, source, dictionary, private):
    """Copy the existing native cache record into a public cell/path/axis receipt.

    QPY originals stay in the dedicated private cache with their hashes. C can
    inspect the public compile record without access to private paths.
    """
    receipts = {}
    for path in paths:
        result = path['result']
        for axis, key in AXES.items():
            fp = result['fingerprints'][key]['fingerprint']
            cache = Path(private) / 'compiled_cache' / (fp + '.json')
            if not cache.is_file():
                raise ValueError('Compiled cache receipt missing')
            raw = cache.read_bytes(); record = json.loads(raw)
            if record != result['compile_receipts'][key]:
                raise ValueError('Compiled cache differs from returned receipt')
            artifacts = []
            for suffix in ('_wrapper.qpy.gz', '_compiled.qpy.gz'):
                file = cache.with_name(fp + suffix)
                if not file.is_file():
                    raise ValueError('Saved wrapper/compiled artifact missing')
                artifacts.append(dict(path=str(file), bytes=file.stat().st_size,
                                      sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
            receipts['p%d%s' % (path['index'], axis)] = dict(cell_id=source['cell'], axis=axis,
                index=path['index'], cohort=path['cohort'], seeds=result['seeds'], request=path['request'],
                dictionary_identity=dictionary, execution_source=source, record=record,
                native_cache=dict(path=str(cache), sha256=hashlib.sha256(raw).hexdigest()), artifacts_private=artifacts)
    return receipts


def evidence(cell, paths, inputs, source, dictionary, facts, receipts):
    """Build source-bound proof/sidecars from actual saved path and cache records."""
    directory = str(Path(source['result_path']).parent); proofs, sides, prices = {}, [], []
    if len(paths) != cell['n_paths'] or len(inputs) != 2:
        raise ValueError('Incomplete cell/path/axis set')
    if len({p['result']['scalar'] for p in paths}) != 1:
        raise ValueError('Mixed path scalar quantities')
    for position, (axis, key) in enumerate(AXES.items()):
        record = inputs[position]
        if record['axis'] != axis or record['cell_id'] != cell['cell_id'] or record['source'] != source:
            raise ValueError('C record cell/axis/SOURCE differs')
        traces = []
        for index, path in enumerate(paths):
            check_path(path, request=paths[0]['request'], dictionary=dictionary, cohort=cell['cohort'],
                       index=index, seeds=source['step_seeds'][index])
            result = path['result']; fp = result['fingerprints'][key]['fingerprint']
            try:
                receipt = receipts['p%d%s' % (index, axis)]
            except KeyError as exc:
                raise ValueError('Compiled receipt missing') from exc
            expected = dict(cell_id=cell['cell_id'], axis=axis, index=index, cohort=cell['cohort'],
                seeds=result['seeds'], request=path['request'], dictionary_identity=dictionary, execution_source=source)
            if (any(receipt.get(k) != v for k, v in expected.items())
                    or receipt['record']['fingerprint'] != fp or receipt['record']['compiler'] != cell['profile']['compiler']
                    or receipt['record']['metrics'] != result['prices'][key]
                    or result['compile_receipts'][key] != receipt['record']):
                raise ValueError('Compiled receipt axis/path/fingerprint/metrics differs')
            trace = dict(index=index, axis=axis, metrics=result['prices'][key], fingerprint=fp,
                path_receipt=locator(source['result_path'], ['child', 'paths', index]),
                compile_receipt=locator(directory + '/COMPILE_RECEIPTS.json', ['p%d%s' % (index, axis)]))
            traces.append(trace)
            prices.append(dict(trace, cell_id=cell['cell_id'], request=path['request'], cohort=cell['cohort'],
                seeds=result['seeds'], classification='DIRECT_COMPILED_FINITE_SAMPLE', expectation='FINITE_COHORT_ONLY', source=source))
        mean = sum(Fraction(path['result']['prices'][key]['rz_count']) for path in paths) / len(paths)
        if record['rz_count'] != dict(numerator=mean.numerator, denominator=mean.denominator):
            raise ValueError('C price differs from its path metrics')
        proofs[axis] = dict(record='PAULI_B1_CELL_IMPLEMENTATION_V1', C_input=record, request=paths[0]['request'],
            dictionary_identity=dictionary, implementation=cell['implementation'], profile=cell['profile'],
            implementation_sources=facts, scalar=paths[0]['result']['scalar'],
            measurement=dict(axis=axis, include_measurement=True, state_preparation_cost_included=False),
            paths=traces, paired_dependence=cell.get('composed_of'), law_expectation=False)
        sides.append(dict(record='PAULI_B1_CELL_BINDING_V1', cell_id=cell['cell_id'], axis=axis,
            execution_source=source, result=locator(source['result_path']),
            C_input=locator(directory + '/C_INPUTS.json', [position]),
            implementation_evidence=locator(directory + '/IMPLEMENTATION_EVIDENCE.json', [axis]),
            compile_receipts=locator(directory + '/COMPILE_RECEIPTS.json')))
    return dict(implementation_evidence=proofs, sidecars=sides, prices=prices)


def bind(root, *, publication_commit, result_path):
    """Return separate published copies; never rewrite the raw C inputs/sidecars."""
    directory = str(Path(result_path).parent)
    inputs = read(root, locator(directory + '/C_INPUTS.json', commit=publication_commit))
    sides = read(root, locator(directory + '/SIDECARS_UNBOUND.json', commit=publication_commit))
    if len(inputs) != 2 or len(sides) != 2:
        raise ValueError('Both raw axes required')
    for record, side in zip(inputs, sides, strict=True):
        for name in ('result', 'C_input', 'implementation_evidence', 'compile_receipts'):
            if side[name]['commit'] is not None:
                raise ValueError('Expected an unbound raw locator')
            side[name]['commit'] = publication_commit
        if side['result']['path'] != result_path:
            raise ValueError('Published result path differs')
        record['source']['publication'] = deepcopy(side['C_input'])
        record['source']['implementation_evidence'] = deepcopy(side['implementation_evidence'])
    return dict(records=inputs, sidecars=sides)


def check(root, *, cell, request, dictionary, seeds, facts, records, sidecars):
    """Read/check one cell's two axes before fit. No fit/price qualification."""
    if (request['m'] is not None or len(records) != 2 or len(sidecars) != 2
            or [r['axis'] for r in records] != list(AXES) or [s['axis'] for s in sidecars] != list(AXES)):
        raise ValueError('Non-configuration cell and ordered X/Y required')
    checked = []
    for position, (record, side) in enumerate(zip(records, sidecars, strict=True)):
        axis = record['axis']; source = deepcopy(record['source']); publication = source.pop('publication', None)
        proof_reference = source.pop('implementation_evidence', None)
        if publication != side['C_input'] or proof_reference != side['implementation_evidence'] or source != side['execution_source']:
            raise ValueError('Execution/publication SOURCE differs')
        refs = [side[k] for k in ('result', 'C_input', 'implementation_evidence', 'compile_receipts')]
        commit = refs[0]['commit']; directory = str(Path(source['result_path']).parent)
        expected_refs = [locator(source['result_path'], commit=commit),
            locator(directory + '/C_INPUTS.json', [position], commit),
            locator(directory + '/IMPLEMENTATION_EVIDENCE.json', [axis], commit),
            locator(directory + '/COMPILE_RECEIPTS.json', commit=commit)]
        if refs != expected_refs or side['cell_id'] != cell['cell_id']:
            raise ValueError('Cell/axis publication locator differs')
        raw = read(root, side['C_input']); data = read(root, side['result'])
        unbound = deepcopy(record); unbound['source'] = source
        if (raw != unbound or data['status'] != 'FINITE_CELL_COMPLETE' or data['child']['status'] != 'PASS'
                or data['source'] != source or source['cell'] != cell['cell_id'] or source['cohort'] != cell['cohort']
                or source['step_seeds'] != seeds or source['prefix'] != request['prefix'] or source['kind'] != 'B1-CAL'):
            raise ValueError('Record/published result/SOURCE differs')
        if (record['record'] != 'PRICE_CALIBRATION_RECORD_V1' or record['role'] != cell['role']
                or record['n_paths'] != cell['n_paths'] or record['finite_sample_mean_is_law_expectation'] is not False
                or any(record[k] != cell[k] for k in ('route', 'law', 'implementation', 'profile', 'quantity', 'kind'))
                or record['features'] != {k: dict(numerator=v, denominator=1)
                    for k, v in dict(one=1, pf_step_count=cell['q'], event_count=cell['s']).items()}):
            raise ValueError('C record domain/features/qualification differs')
        paths = data['child']['paths']
        if len(paths) != cell['n_paths'] or any(p['request'] != request for p in paths):
            raise ValueError('Published physical request/path set differs')
        for i, path in enumerate(paths):
            check_path(path, request=request, dictionary=dictionary, cohort=cell['cohort'], index=i, seeds=seeds[i])
        inputs = read(root, locator(directory + '/C_INPUTS.json', commit=commit))
        receipts = read(root, side['compile_receipts'])
        want = evidence(cell, paths, inputs, source, dictionary, facts, receipts)['implementation_evidence'][axis]
        proof = read(root, side['implementation_evidence'])
        if proof != want:
            raise ValueError('Cell implementation evidence differs from path/receipt')
        trace = []
        for item in proof['paths']:
            path_ref = dict(item['path_receipt'], commit=commit)
            receipt_ref = dict(item['compile_receipt'], commit=commit)
            # Resolve the individual selectors too, not just the containing files.
            read(root, path_ref); read(root, receipt_ref)
            trace.append(dict(index=item['index'], axis=axis, path=path_ref, receipt=receipt_ref,
                              fingerprint=item['fingerprint']))
        checked.append(dict(cell_id=cell['cell_id'], axis=axis, paths=trace,
                            implementation_evidence=side['implementation_evidence']))
    return dict(status='CELL_RECEIPTS_CHECKED_NOT_SCIENTIFIC_QUALIFICATION', checked=checked,
                law_expectation=False, model_qualified=False, records=deepcopy(records))
