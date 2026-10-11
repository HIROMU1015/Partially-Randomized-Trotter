#!/usr/bin/env python3
"""Read-only metadata check. No science import, solver, diagonalization or compile.

This does not validate the new mathematical design or an implemented decoder.
An optional fresh output path records the metadata audit, without overwriting.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = '10d91e518520e6f3fd853c85d8eff078357b7edb'
DOC = 'docs/research/'
ART = 'artifacts/n3_generator_n1_recovery/2026-10-11/preparation/'
CONTEXT = {'context','modes','particles','df_rank','weights','one_body','scalar',
           'input_factors','time','model_budget','q_grid','generator_candidates'}
CANDIDATE = {'id','family','eta','approx_eta','frame','ungauged_frame','ungauged_eta','g','groups'}


def audit(root):
    checks = []

    def require(ok, message):
        if not ok:
            raise ValueError(message)
        checks.append(message)

    def load(path):
        def constant(value):
            raise ValueError('nonfinite JSON token: ' + value)
        return json.loads((root / path).read_text(), parse_constant=constant)

    def sha(path):
        return hashlib.sha256((root / path).read_bytes()).hexdigest()

    def number(x, path):
        require(type(x) in (int, float) and math.isfinite(x), 'finite number: ' + path)

    def vector(x, path, object_form=False):
        if object_form:
            require(type(x) is dict and set(x) == {'real','imag'}, 'complex vector schema: ' + path)
            real, imag = x['real'], x['imag']
            require(type(imag) is list and len(imag) == 4, 'vector imag shape: ' + path)
            for value in imag:
                number(value, path + '/imag')
                require(value == 0, 'spectrum imag exactly zero: ' + path)
        else:
            real = x
        require(type(real) is list and len(real) == 4, 'vector real shape: ' + path)
        for value in real:
            number(value, path + '/real')

    def matrix(x, path):
        require(type(x) is dict and set(x) == {'real','imag'}, 'matrix object: ' + path)
        for field in ('real','imag'):
            rows = x[field]
            require(type(rows) is list and len(rows) == 4, 'matrix rows: ' + path + '/' + field)
            for row in rows:
                require(type(row) is list and len(row) == 4, 'matrix columns: ' + path)
                for value in row:
                    number(value, path + '/' + field)

    n3 = load(DOC + 'n3_generator_transfer_contract_v1_20261011.json')
    n1 = load(DOC + 'n1_decode_recovery_contract_v1_20261011.json')
    for contract in (n3, n1):
        require(contract['base_commit'] == BASE, 'published base identity')
        require(contract['mandatory_stop'] and contract['central_theme'] is None, 'STOP and no adopted theme')
        require(not contract['science_execution_authorized'] and not contract['implementation_authorized_this_preparation']
                and not contract['next_scientific_stage_authorized'], 'no implementation/science/next-stage authorization')
        require(contract['prior_gpt_scientific_review_complete'] and not contract['repeat_prior_review'], 'completed review not repeated')
        require(sha(contract['review_input']) == contract['review_sha256'], 'review byte identity')
        require(all(value == 0 for value in contract['counts_this_preparation'].values()), 'no scientific operations in this preparation')
        impl = contract['implementation']
        require(impl['status'] == 'not_implemented' and all(impl[key] is None for key in
                ('source_commit','source_bound_plan','execution_authorization')), 'future implementation pending')
        require(all(not (root / impl[key]).exists() for key in ('module','runner','test','saved_verifier')), 'future science paths absent')
        resources = contract['resources']
        require(resources['wall_seconds'] is None and resources['cpu_seconds'] is None, 'no scientific wall/CPU caps')
        require(resources['workers_max'] == 2 and resources['threads_per_worker'] == 1, 'coordinated CPU workers/threads')
        require(resources['worker_memory_bytes'] * resources['workers_max'] <= resources['aggregate_memory_bytes'], 'aggregate memory budget')

    old = load(DOC + 'n13_redesign_limited_scope_v1_20261011.json')
    require(n1['frozen_scientific_scope'] == old['N1'], 'N1 entire former scientific domain preserved')
    require(not n1['run_N3'] and not n1['run_N2'] and not n1['scientific_kernel_change_allowed'], 'N1 recovery excludes old N3/N2 and scientific-kernel edits')
    require(n1['scientific_attempts_max'] == 1 and n1['scientific_retries_max'] == 0 and not n1['resume'], 'N1 single attempt no retry/resume')
    scope = n1['frozen_scientific_scope']
    require(sha(scope['selector_input']) == scope['selector_input_sha256'], 'unchanged N1 real input pack')
    pack = load(scope['selector_input'])
    require(set(pack) == {'id','schema_version','scientific_evaluation_performed','source_path','source_sha256','contexts'}, 'pack top whitelist')
    require(pack['scientific_evaluation_performed'] is False and len(pack['contexts']) == 3, 'three coefficient-only contexts')
    counts = {}; ids = []
    for context in pack['contexts']:
        name = context['context']
        require(set(context) == CONTEXT, 'context whitelist: ' + name)
        require((context['modes'],context['particles'],context['df_rank']) == (4,2,2), 'fixed N1 dimensions: ' + name)
        require(context['q_grid'] == [1,2,4,8] and context['time'] == .6 and context['model_budget'] == .02, 'fixed q/time/budget: ' + name)
        require(context['weights'] == [.7,-.4] and len(context['input_factors']) == 2, 'signed rank2 factors: ' + name)
        number(context['scalar'], name + '/scalar')
        matrix(context['one_body'], name + '/one_body')
        for j, value in enumerate(context['input_factors']):
            matrix(value, name + '/factor' + str(j))
        candidates = context['generator_candidates']
        require(type(candidates) is list and len(candidates) == 2, 'two candidate lists: ' + name)
        counts[name] = [len(xs) for xs in candidates]
        for j, xs in enumerate(candidates):
            require(sum(x['family'] == 'exact' for x in xs) == 1, 'one exact fallback per factor')
            for k, value in enumerate(xs):
                ident = f'{name}/factor{j}/candidate{k}'
                require(set(value) == CANDIDATE and value['id'] == ident, 'candidate whitelist/order: ' + ident)
                ids.append(ident)
                require(value['family'] in {'exact','cluster','shifted_cutoff','zero_cutoff'}, 'fixed candidate family')
                vector(value['eta'], ident + '/eta')
                vector(value['approx_eta'], ident + '/approx_eta')
                vector(value['ungauged_eta'], ident + '/ungauged_eta', object_form=True)
                for field in ('frame','ungauged_frame','g'):
                    matrix(value[field], ident + '/' + field)
                groups = value['groups']
                require(type(groups) is list and all(type(xs) is list and xs for xs in groups), 'nonempty groups')
                flat = [i for xs in groups for i in xs]
                require(all(type(i) is int for i in flat) and sorted(flat) == [0,1,2,3], 'groups partition, no bool/duplicate')
    require(counts == {'planted_local_gauge':[9,9],'dense_group_frame':[9,9],'separated_spectrum':[4,4]}, 'all-context counts')
    require(len(ids) == len(set(ids)) == 44 and sum(a*b for a,b in counts.values()) == 178, '44 factors/178 pairs')
    requests = 2 * 178 * 4 * 2 + 24 + 48
    require(requests == n1['fixed_requested_wrappers'] == 2920 and requests <= n1['wrapper_compiles_max'] == 3072, 'unchanged wrapper accounting')
    require(n1['total_compiles_max'] == n1['wrapper_compiles_max'] + n1['basis_compiles_max'] == 3116, 'unchanged total compile cap')

    require(sha(n3['input_path']) == n3['input_sha256'], 'result-prior N3 coefficient pack identity')
    inputs = load(n3['input_path'])
    cases = inputs['cases']
    require(inputs['scientific_evaluation_performed'] is False and not inputs['blind_statistical_sample'], 'unevaluated literals, no statistical-blind claim')
    require(len(cases) == n3['input_count'] == 12 and [x['id'] for x in cases] == n3['input_ids'], '12 fixed transfer inputs/order')
    require(len({x['id'] for x in cases}) == 12, 'unique transfer ids')
    for case in cases:
        n = case['modes']; h = case['h']; d = case['external']; active = case['active']
        require(len(h) == n and all(len(row) == n for row in h), 'N3 literal matrix shape')
        require(sorted(active + [d]) == list(range(n)) and len(active) == n-1 and 1 <= case['particles'] <= n-1, 'one external and valid filling')
        require(all(h[i][j] == h[j][i] for i in range(n) for j in range(n)), 'real symmetric literal h')
        for row in h:
            for pair in row:
                require(type(pair) is list and len(pair) == 2 and all(type(x) is int for x in pair)
                        and pair[1] > 0 and math.gcd(*pair) == 1, 'canonical rational coefficient metadata')
        require(all(x['coefficient'][0] >= 0 and x['coefficient'][1] > 0 and x['i'] != x['j']
                    and 0 <= x['i'] < n and 0 <= x['j'] < n for x in case['density']), 'nonnegative density literal schema')
    require(12 * (len(n3['primary_methods']) + len(n3['secondary_methods'])) == n3['method_records_max'] == 72, 'bounded comparison records')
    require(n3['generator']['rational_witness_attempts_per_case_max'] == 10 and n3['generator']['search_eigenvalue_calls_per_case_max'] == 128, 'bounded generator calls')
    require(not n3['generator']['numerical_success_is_certificate'] and not n3['generator']['completeness_claim']
            and not n3['generator']['old_five_point_fallback_in_direct_method'], 'valid evidence distinguished from proposals/union')
    require(n3['old_development_scientific_replays'] == n3['old_GPT_appendix_replays'] == 0, 'no repeated former science/review')

    previous_raw = subprocess.check_output(['git','show',BASE+':artifacts/validation_manifest.json'], cwd=root, text=True)
    previous = json.loads(previous_raw); current_raw = (root/'artifacts/validation_manifest.json').read_text(); current = json.loads(current_raw)
    require(current['result_sets'][:88] == previous['result_sets'] and len(current['result_sets']) == 89, '88 old entries preserved; one new preparation entry')
    require(all(current[k] == v for k,v in previous.items() if k != 'result_sets'), 'all other manifest fields preserved')
    start = previous_raw.index('[', previous_raw.index('"result_sets"'))
    _, end = json.JSONDecoder().raw_decode(previous_raw,start)
    trim = end-1
    while previous_raw[trim-1].isspace():
        trim -= 1
    require(current_raw[:trim] == previous_raw[:trim] and current_raw.endswith(previous_raw[trim:]), 'old manifest raw prefix and suffix unchanged')
    prefixes = ['src/trottertracks/', 'tests/', 'artifacts/n13_redesign/', 'artifacts/hamiltonian_algorithm_design/']
    protected = subprocess.check_output(['git','ls-tree','-r','--name-only',BASE,'--',*prefixes],cwd=root,text=True).splitlines()
    protected += ['scripts/run_n13_redesign.py','scripts/verify_n13_redesign.py',
                  'scripts/audit_n13_redesign_saved_diagnostics.py', DOC+'n13_redesign_execution_plan_20261011.json',
                  DOC+'n13_redesign_execution_20261011.md', DOC+'n13_redesign_results_20261011.md']
    for path in protected:
        reference = subprocess.check_output(['git','show',BASE+':'+path],cwd=root)
        require((root/path).read_bytes() == reference, 'old frozen blob unchanged: '+path)
    own = ast.parse(Path(__file__).read_text())
    modules = {node.names[0].name.split('.')[0] for node in ast.walk(own) if isinstance(node,ast.Import)}
    modules |= {node.module.split('.')[0] for node in ast.walk(own) if isinstance(node,ast.ImportFrom)}
    require(modules <= {'argparse','ast','hashlib','json','math','pathlib','subprocess'}, 'metadata checker has no numerical/project/compiler imports')
    return {'status':'PREPARATION_METADATA_PASS','metadata_checks':len(checks),
            'claim_scope':'Identities, metadata, limits and preservation only; no mathematical, decoder implementation or scientific result validation.',
            'N1_pack_all_context_metadata':'PASS; decoder regression still pending',
            'N1_factor_candidates':44,'N1_pairs':178,'N3_coefficient_literals':12,
            'old_manifest_entries_preserved':88,'old_frozen_blobs_checked':len(protected),
            'source_base_commit':BASE,'review_sha256':n3['review_sha256'],
            'contract_sha256':{x['id']:sha(DOC+x['id']+'.json') for x in (n3,n1)},
            'scientific_calculations_run':0,'scientific_tests_run':0,
            'decoder_regression_tests_run':0,'mandatory_stop':True,'science_execution_authorized':False,
            'global_manifest_checker':'not rerun; known pre-existing description failure retained'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.output and args.output.exists():
        parser.error('fresh output path required; no overwrite')
    result=audit(ROOT)
    payload=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(payload)
    print(payload,end='')


if __name__=='__main__':
    main()
