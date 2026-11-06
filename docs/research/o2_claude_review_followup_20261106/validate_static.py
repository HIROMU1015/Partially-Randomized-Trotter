"""Metadata/document validation only; never import or execute scientific code.

pointer/read_csv reused from the fixed O2 validate_catalog.py at 2779710a.
The original validator enforces that catalog's file set and cannot validate this
separate technical report. SOURCE inspection below uses AST and text only.
"""
import argparse
import ast
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess

BASE = 'b3b194dbf405285239326fd25b671d30957b801c'
PREFIX = 'docs/research/o2_claude_review_followup_20261106/'
FILES = {'README.md', 'REPORT.md', 'SOURCE_FORMULA_TRACE.md',
         'PRIOR_ART_MATRIX.md', 'MINIMAL_NEXT_VALIDATION.md',
         'METRIC_AVAILABILITY.csv', 'SOURCE_INDEX.json', 'VALIDATION.json',
         'validate_static.py'}
ENV = dict(os.environ, GIT_OPTIONAL_LOCKS='0', GIT_NO_LAZY_FETCH='1')


def pointer(doc, path):
    assert path == '' or path.startswith('/')
    for token in path.split('/')[1:]:
        token = token.replace('~1', '/').replace('~0', '~')
        doc = doc[int(token)] if isinstance(doc, list) else doc[token]
    return doc


def read_csv(path):
    reader = csv.DictReader(io.StringIO(path.read_text(), newline=''))
    rows = list(reader)
    assert len(reader.fieldnames) == len(set(reader.fieldnames))
    assert rows and all(None not in row for row in rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    args = parser.parse_args()
    out = Path(__file__).resolve().parent
    checks = []

    def git(*arguments):
        return subprocess.check_output(['git', '-C', str(args.repo),
                                        *arguments], env=ENV)

    def check(name, condition):
        assert condition, name
        checks.append(name)

    check('deliverable_file_set', {p.name for p in out.iterdir()
                                  if p.is_file()} == FILES)
    index = json.loads((out / 'SOURCE_INDEX.json').read_text())
    meta = json.loads((out / 'VALIDATION.json').read_text())
    cache = {}
    for entry in index['references']:
        check('fixed_sha:' + entry['commit'][:8] + ':' + entry['path'],
              bool(re.fullmatch('[a-f0-9]{40}', entry['commit'])))
        spec = entry['commit'] + ':' + entry['path']
        data = git('show', spec)
        blob = git('rev-parse', spec).decode().strip()
        check('fixed_blob:' + spec,
              blob == entry['blob'] and len(data) == entry['bytes']
              and hashlib.sha256(data).hexdigest() == entry['sha256'])
        cache[spec] = data

    def source(sha, path):
        return cache[sha + ':' + path].decode()

    def document(sha, path):
        data = cache[sha + ':' + path]
        if path.endswith('.gz'):
            data = gzip.decompress(data)
        return json.loads(data, parse_int=str, parse_float=str)

    provider = 'c31ef424175c02379eafebe8fc6c7548c1d9aabe'
    signal = 'e417ffc9f52b13c18f5493193ce510d858664259'
    native = '27ebabdd2943a4de787ec8b976c38e06fda5a497'
    rte = source(provider, 'src/trotterlib/rte.py')
    nodes = {n.name: n for n in ast.parse(rte).body
             if isinstance(n, ast.FunctionDef)}
    check('paired_definition_present', all(n in nodes for n in
          ['_paired_order_weight', 'finite_rte_distribution', '_make_event']))
    weight = ast.get_source_segment(rte, nodes['_paired_order_weight'])
    check('even_signed_normalization_structure',
          'magnitude = abs(raw_time)' in weight
          and 'math.hypot(1.0, magnitude / (order + 1))' in weight)
    core = source(signal, 'scripts/h46_new2_core.py')
    check('minimum_step_and_remainder_allocation_structure',
          'total < len(coefficients)' in core
          and '(total-len(coefficients))' in core
          and 'result = [1+math.floor(x) for x in quota]' in core)
    sig = source(signal, 'scripts/h46_long_time_signal.py')
    nat = source(native, 'scripts/h46_long_time_native.py')
    check('signal_log_product_structure',
          'logs[j] += q*count*math.log(b)' in sig
          and 'value/(b**count)' in sig)
    check('native_allocation_and_event_identity_structure',
          "'ABS_TIME_PER_OUTER'" in nat
          and "len(req.rte_occurrence.events)!=s['rte_steps']" in nat)

    for role in index['source_role_checks']:
        a = cache[role['commit'] + ':' + role['path']]
        carrier = cache['155529e20a8fac39e00786c40b57830b71203c4f:'
                        + role['path']]
        check('role_vs_carrier:' + role['role'] + ':' + role['path'],
              (a == carrier) == role['same_as_carrier'])

    metrics = {'rz_count', 'cx_count', 'rz_depth', 'cx_depth',
               'total_depth', 'circuit_size'}
    h8c = document('eb07156a37b1938ed3aed95c13abbb545466c573',
          'artifacts/resource_applicability/track_a_h8_long_time/'
          '2026-11-05/confirmation.json')
    check('H8_confirmation_six_metric_schema', all(
          set(t['prices'][axis]) == metrics
          for t in h8c['trials'] for axis in ['cosine', 'sine']))
    h8b = document('7fb1d125e05238a7d932fc89b8dab6ac6a7f8c84',
          'artifacts/resource_applicability/track_a_h8_long_time/'
          '2026-11-04/exploration.json.gz')
    pf_id = h8c['fixed_PF']['candidate']['id']
    pf = [t for t in h8b['trials'] if t['candidate_id'] == pf_id]
    check('H8_fixed_PF_identity_and_secondary_prices', bool(pf) and all(
          set(t['prices'][axis]) == metrics
          for t in pf for axis in ['cosine', 'sine']))
    p3 = document('e44e87a306d05c8549f1a044b7fd056c2e1055a3',
         'artifacts/df_pauli_matched_partition_validation/'
         '2026-11-02/stage3/run04/prices.json')
    check('P3_six_prices_and_saved_SX_histogram', all(
          {'RZ', 'CX', 'depth', 'rz_depth', 'cx_depth', 'size'} <= set(v)
          and 'sx' in v['output_histogram'] for v in p3.values()))
    p3code = source('a1d3f32aba6416fd1e200cd66a97d1a825b5dd73',
                    'src/trotterlib/matched_partition_stage4.py')
    check('P3_only_three_metrics_previously_accounted',
          "METRICS = ('RZ', 'CX', 'depth')" in p3code)

    rows = read_csv(out / 'METRIC_AVAILABILITY.csv')
    check('metric_inventory_identity_and_missingness', all(
          x['study_id'] and x['candidate_cohort_identity']
          and x['unverified'] and x['compile_requirement'] for x in rows))
    p3row = next(x for x in rows if x['comparison_id'] == 'P3_Stage4')
    check('P3_inventory_matches_accounting_source', all(
          'NO_STAGE4_G' in p3row[k]
          for k in ['RZ_depth', 'CX_depth', 'circuit_size']))
    check('no_scientific_execution_declared',
          all(v == 0 for v in meta['scientific_execution_counts'].values()))
    check('conditional_scope_and_user_handoff',
          meta['formula_status'] == 'CONSISTENT_WITH_ASSUMPTIONS'
          and meta['formal_adoption'] is False
          and meta['mandatory_STOP_after_publication'] is True)

    for path in out.glob('*.md'):
        body = path.read_text()
        for target in re.findall(r'\]\(([^)]+)\)', body):
            if not target.startswith(('https://', '#')):
                check('local_link:' + path.name + ':' + target,
                      (out / target.split('#')[0]).is_file())
    for path in out.iterdir():
        if not path.is_file():
            continue
        text = path.read_text()
        check('publication_hygiene:' + path.name,
              '/' + 'home/' not in text
              and 'BEGIN ' + 'PRIVATE KEY' not in text
              and 'GPUA100' + '-gleap' not in text)
    own = ast.parse(Path(__file__).read_text())
    imported = {a.name.split('.')[0] for n in ast.walk(own)
                if isinstance(n, ast.Import) for a in n.names}
    check('validator_imports_stdlib_only', not imported.intersection(
          {'numpy', 'scipy', 'qiskit', 'trotterlib', 'trottertracks'}))
    changed = git('diff', '--name-only', BASE).decode().splitlines()
    check('tracked_changes_confined_to_deliverable_directory',
          all(x.startswith(PREFIX) for x in changed))
    print(json.dumps({'status': 'PASS', 'check_count': len(checks),
          'scientific_modules_imported': False,
          'scientific_execution': False,
          'checks': checks}, ensure_ascii=False))


if __name__ == '__main__':
    main()
