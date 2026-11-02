#!/usr/bin/env python3
"""Read fixed Git JSON/CSV and prepare PF alignment. No science execution mode."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import gzip
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('pf_common_domain', ROOT/'src/trotterlib/pf_common_domain.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
BASE = '3eca46148070ba4d4b0f5815ef45943407cc959e'
H4 = '85cb5b4d1808434165c421115ed09dab3b63dd00'
H6 = 'fc6bd039dc7171d5674ac04f0b3ad427f1b6495b'
DIRECT = '80b7a35c4dbf683f952293eeb55944bc10b9efc8'
A = 'artifacts/resource_applicability/'
M = A+'track_a_h46_higher_pf_dataset_v1/2026-10-20/'
P = A+'track_a_h46_m5_tol_pilot_run01/2026-10-19/'
SOURCES = {
    'h4_registry': (H4, A+'track_a_h4_tol_main_phase0/2026-10-19/candidate_registrations.json'),
    'h4_signals': (H4, A+'track_a_h4_tol_d_completed/2026-10-19/signals.json.gz'),
    'h4_shots': (H4, A+'track_a_h4_tol_d_completed/2026-10-19/result.json'),
    'h4_pairs': (H4, A+'track_a_h4_tol_d_completed/2026-10-19/paired_measurements.json.gz'),
    'h4_legacy': (H4, A+'track_a_h4_native_validation/2026-10-18/D_result.json.gz'),
    'h4_legacy_registry': (H4, A+'track_a_h4_native_validation/2026-10-18/plan_v1.json'),
    'm5_legacy_plan_source': (BASE, 'src/trotterlib/h46_m5_plan.py'),
    'v2_registry': (BASE, A+'track_a_protocol_v2_draft/2026-10-19/candidate_registrations.json'),
    'h6_account': (H6, A+'track_a_v1_reaccount_result/2026-10-16/candidate_accounting.json.gz'),
    'h6_signals': (H6, A+'track_a_v1_phase_a_run02/2026-10-16/signal_results_v1.json.gz'),
    'h6_input_binding': (H6, A+'track_a_v1_h6_readiness/2026-10-16/input_binding_v1.json'),
    'm5_domain': (BASE, 'references/h46_m5_signal_domain_v1.json'),
    'm5_optional_domain': (BASE, A+'track_a_h46_m5_minimal_comparison_plan/2026-10-19/candidate_domain_DRAFT.json'),
    'm5_dataset': (BASE, M+'dataset.json'),
    'conditions': (BASE, M+'conditions.json'),
    'm5_signals': (BASE, A+'track_a_h46_m5_signal_run02_results/2026-10-20/signal_records.json'),
    'm5_saved_shots': (BASE, A+'track_a_h246_saved_cost_results/2026-10-21/candidate_precision_costs_1000.csv'),
    'm5_pilot_summary': (BASE, P+'summary.json'),
    'direct_axes': (DIRECT, A+'track_a_pf_direct_compile_retry106/2026-10-20/results01/combined_gate_counts.csv'),
    'direct_bindings': (DIRECT, A+'track_a_pf_q_model_comparison_preparation/2026-10-20/direct_system_bindings.json'),
    'legacy_models': (DIRECT, A+'track_a_pf_q_model_comparison_preparation/2026-10-20/model_registry.json'),
    'migrated_inputs': (BASE, A+'track_a_h246_reconstructed_confirmation/2026-10-22/input_reconstruction.json'),
    'migrated_resources': (BASE, A+'track_a_h246_reconstructed_confirmation/2026-10-22/controller_and_resources.json'),
}


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def read_source(label, source):
    commit, path = source
    raw = git('show', commit+':'+path)
    sha = hashlib.sha256(raw).hexdigest()
    blob = git('rev-parse', commit+':'+path).decode().strip()
    content = gzip.decompress(raw) if path.endswith('.gz') else raw
    data = (content.decode() if path.endswith('.py') else
            list(csv.DictReader(io.StringIO(content.decode()))) if path.endswith('.csv') else
            json.loads(content))
    return label, data, dict(label=label, commit=commit, path=path, bytes=len(raw), sha256=sha, git_blob=blob,
        url=f'https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/{commit}/{path}')


def pilot_group(system, mode, formula):
    row = dict(system=system, method=mode, formula=formula,
               prefix=c.RANKS[system] if mode == 'FULL' else (3 if system == 'H4' else 5),
               q=2, K=None, policy=None, r_value=None,
               input_series=f'{system}_TOL_ONLY_DF{c.RANKS[system]}')
    row['id'] = system+'_TOL_'+mode+'_'+c.digest(row)[:16]
    return row


def build():
    sources = dict(SOURCES)
    # Paths are reconstructed from the original stdlib candidate hash, then Git
    # existence is checked. A missing H6 full/m5 result retains its historical STOP.
    for system in c.RANKS:
        for formula in c.FORMULAS:
            for mode in ('B0', 'B0I', 'FULL'):
                group = pilot_group(system, mode, formula)
                directory = P+system+'_TOL_PILOT_DET/'+group['id']+'__0/'
                for kind in ('result', 'signal'):
                    path = directory+kind+'.json'
                    exists = subprocess.run(['git', 'cat-file', '-e', BASE+':'+path], cwd=ROOT,
                                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
                    if exists:
                        sources['pilot_'+group['id']+'_'+kind] = BASE, path
    with ThreadPoolExecutor(max_workers=min(len(sources), len(os.sched_getaffinity(0)))) as pool:
        loaded = list(pool.map(lambda item: read_source(*item), sources.items()))
    data = {label: d for label, d, _ in loaded}
    catalog = [entry for _, _, entry in loaded]
    for system, saved in [('H4', data['h4_signals']['binding']), ('H6', data['h6_input_binding']['binding'])]:
        binding = data['conditions']['systems'][system]['binding']
        for field in ('hamiltonian', 'state', 'compiler'):
            c.require(saved[field] == binding[field], 'SCIENTIFIC_INPUT_JOIN:'+system+':'+field)
        c.require(saved.get('snapshot', saved.get('input')) == binding['snapshot'], 'ORIGINAL_SNAPSHOT_JOIN:'+system)
    records = {}

    def get(system, row):
        r = c.normalized(system, row)
        k = c.key(r)
        if k not in records:
            records[k] = dict(r, id=c.identifier(r), input_comparable=True,
                input_series=f'{system}_TOL_ONLY_DF{c.RANKS[system]}',
                binding=data['conditions']['systems'][system]['binding'],
                registrations=[], signal_evidence=[], eligibility={}, shots={}, cost_evidence=[],
                same_native_cost_pair=False, any_direct_cost_pair=False,
                preparation_hashes=[], native_policy=c.NATIVE, compiler_hash=c.COMPILER)
        return records[k]

    def reg(system, row, label):
        r = get(system, row)
        r['registrations'].append(dict(source=label, original_id=row['id']))
        return r

    def signal(r, label, pointer, *, raw=True, prep=None):
        r['signal_evidence'].append(dict(source=label, pointer=pointer, raw=raw))
        if prep and prep not in r['preparation_hashes']:
            r['preparation_hashes'].append(prep)

    for row in data['h4_registry']['D']:
        reg('H4', row, 'h4_registry')
    for i, row in enumerate(data['h4_signals']['signals']):
        r = get('H4', row['cell']); signal(r, 'h4_signals', f'/signals/{i}')
        for ep in c.EPS:
            saved = row['precision_rows'][ep]['1.0']
            r['eligibility'][ep] = saved['eligibility_under_declared_allowance']
            r['shots'][ep] = {a: saved['axes'][a]['shots'] for a in ('real', 'imag')}
    for r in data['v2_registry']['registries']['H6_strengthened_S2_S4']['rows']:
        reg('H6', dict(r, id=r['id']), 'v2_registry')
    for i, row in enumerate(data['h6_account']['rows']):
        if row['method'] == 'B2' or row['factor'] != 1. or row['bias_rule'] != 'signed_signal':
            continue
        r = get('H6', row['cell'])
        signal(r, 'h6_account', f'/rows/{i}', raw=False)
        ep = str(row['epsilon']); ac = row['accounting']
        r['eligibility'][ep] = ac['eligibility_under_declared_allowance']
        r['shots'][ep] = {a: ac['axes'][a]['shots'] for a in ('real', 'imag')}
        if row.get('axis_costs'):
            r['cost_evidence'].append(dict(source='h6_account', pointer=f'/rows/{i}/axis_costs',
                kind=row['cost_status'], common_native_identity=False, axes=row['axis_costs']))
            if row['cost_status'] == 'MEASURED_COMPLETE':
                c.require(len(row['axis_costs']) == 2 and
                          {a['axis'] for a in row['axis_costs']} == {'cosine', 'sine'}, 'H6_DIRECT_PAIR')
                r['any_direct_cost_pair'] = True
    for i, row in enumerate(data['h6_signals']['signals']):
        signal(get('H6', row['cell']), 'h6_signals', f'/signals/{i}')
    for row in data['m5_domain']['rows']:
        if row['method'] != 'PR':
            reg(row['system'], row, 'm5_domain')
    for i, row in enumerate(data['m5_signals']['records']):
        cand = row['candidate']
        if cand['method'] == 'PR':
            continue
        r = get(cand['system'], cand)
        signal(r, 'm5_signals', f'/records/{i}', prep=row['preparation_hash'])
        for ep in c.EPS:
            vals = [x['status_under_proposed_empirical_policy'] for x in row['empirical_eligibility']['rows']
                    if str(x['epsilon']) == ep]
            c.require(len(vals) == 2, 'M5_TWO_AXES')
            r['eligibility'][ep] = ('ELIGIBLE' if all(v == 'ELIGIBLE' for v in vals)
                                    else 'INELIGIBLE' if 'INELIGIBLE' in vals else 'UNDETERMINED')
    for row in data['m5_dataset']['measured_gate_costs']:
        r = get(row['system'], row)
        r['same_native_cost_pair'] = True
        r['any_direct_cost_pair'] = True
        r['cost_evidence'].append(dict(source='m5_dataset', kind='DIRECT_PAIRED_ACCEPTED_NATIVE',
                                      common_native_identity=True, axis=row['axis'], receipt=row))
    for r in records.values():
        measured = [x for x in r['cost_evidence'] if x['source'] == 'm5_dataset']
        if measured:
            c.require(len(measured) == 2 and {x['axis'] for x in measured} == {'cosine', 'sine'},
                      'M5_COMPLETE_SAVED_COST_PAIR')
    for system in c.RANKS:
        for formula in c.FORMULAS:
            for mode in ('B0', 'B0I', 'FULL'):
                group = pilot_group(system, mode, formula)
                r = reg(system, group, 'm5_pilot_summary')
                base = 'pilot_'+group['id']
                if base+'_signal' in data:
                    signal(r, base+'_signal', '/')
                result = data.get(base+'_result')
                if result:
                    c.require(result['status'] == 'PASS' and result['native_policy'] == c.NATIVE,
                              'PILOT_ACCEPTANCE')
                    c.require(result['binding'] == r['binding'], 'PILOT_SCIENTIFIC_INPUT_IDENTITY')
                    axes = result['records']
                    c.require({x['axis'] for x in axes} == {'cosine', 'sine'} and
                              all(x['compiler_settings_hash'] == c.COMPILER for x in axes), 'PILOT_PAIRED_COMPILER')
                    r['same_native_cost_pair'] = r['any_direct_cost_pair'] = True
                    r['cost_evidence'].append(dict(source=base+'_result', kind='DIRECT_PAIRED_NATIVE_PILOT',
                                                  common_native_identity=True, axes=axes))
                    if result['preparation_hash'] not in r['preparation_hashes']:
                        r['preparation_hashes'].append(result['preparation_hash'])
    pattern = re.compile(r'^(B0I|B0|B1)_p(\d+)_q(\d+)_(2nd|4th)$')
    for row in data['direct_axes']:
        if row['system'] not in ('H4_DF10', 'H6_DF19'):
            continue
        m = pattern.fullmatch(row['candidate_id']); c.require(m is not None, 'DIRECT_DET_ID')
        mode, p, q, formula = m.groups()
        r = get(row['system'][:2], dict(method=mode, prefix=p, q=q, formula=formula))
        c.require(row['compiler_hash'] == c.COMPILER, 'DIRECT_COMPILER')
        r['cost_evidence'].append(dict(source='direct_axes', kind='DIRECT_MEASURED_PROFILE_BRIDGE_PENDING',
                                      common_native_identity=False, axis=row['axis'], receipt=row))
    for i, row in enumerate(data['h4_pairs']['rows']):
        m = pattern.fullmatch(row['cell_id']); c.require(m is not None, 'H4_DET_COST_ID')
        mode, p, q, formula = m.groups()
        r = get('H4', dict(method=mode, prefix=p, q=q, formula=formula))
        r['cost_evidence'].append(dict(source='h4_pairs', pointer=f'/rows/{i}',
            kind='DIRECT_LEGACY_NATIVE_BRIDGE_PENDING', common_native_identity=False, axes=row['rows']))
        r['any_direct_cost_pair'] = True
    for r in records.values():
        direct_axes = {x.get('axis') for x in r['cost_evidence'] if x['kind'].startswith('DIRECT_MEASURED')}
        if direct_axes == {'cosine', 'sine'}:
            r['any_direct_cost_pair'] = True
        raw = any(x['raw'] for x in r['signal_evidence'])
        r['signal_status'] = 'SAVED_RAW_SIGNAL' if raw else 'SAVED_ENDPOINT_ONLY_RAW_UNRECOVERED' if r['signal_evidence'] else 'MISSING_SIGNAL'
        r['eligibility_status'] = c.status_summary(r['eligibility'])
        r['shot_reuse'] = ('SAVED_VALUES_ONLY_REQUIRE_SAME_ALLOWANCE_AND_BINARY64_RULE'
                          if r['shots'] else 'NO_MATCHED_SAVED_SHOTS_IN_LEGACY_TABLE')
    # Future common rows include previously unregistered formula/prefix pairs.
    common = []
    for row in c.common_domain(list(records.values())):
        r = get(row['system'], row)
        if 'signal_status' not in r:
            r.update(signal_status='MISSING_SIGNAL', eligibility_status='ELIGIBILITY_UNCHECKED',
                     shot_reuse='NOT_ACQUIRED')
        common.append(r)
    for row in data['m5_saved_shots']:
        # CSV is an immutable saved projection. Record presence and its pointer,
        # without interpreting scientific totals or selecting any saved winner.
        cid = row.get('candidate_id', row.get('id'))
        if cid:
            for r in common:
                if any(x['original_id'] == cid for x in r['registrations']):
                    r.setdefault('saved_higher_pf_shot_rows', []).append(row)
    legacy = []
    legacy_cells = list(data['h4_legacy_registry']['D']['cells'])
    for row in data['h4_legacy']['rows']:
        m = pattern.fullmatch(row['cell_id'])
        if m:
            mode, p, q, formula = m.groups()
            legacy_cells.append(dict(method=mode, prefix=p, q=q, formula=formula))
    for row in legacy_cells:
        r = c.normalized('H4', row, rank=12)
        if c.key(r) not in {c.key(x) for x in legacy}:
            legacy.append(dict(r, input_series='H4_LEGACY_DF12', input_comparable=False,
                reason='DIFFERENT_SCIENTIFIC_INPUT_DF12_NOT_DF10', source='h4_legacy_registry',
                acquisition_status='OUTSIDE_CURRENT_COMPARISON;PARTIAL_MEASUREMENTS_IN_h4_legacy'))
    proposal = [x for x in data['m5_optional_domain']['rows'] if x['method'] != 'PR' and x.get('role') != 'PRIMARY']
    # The older m5 screen is an unexecuted blueprint, with H4 DF12. Preserve its
    # domain separately; it must never enlarge DF10 prefixes through p11.
    legacy_m5 = []
    for system, rank, cuts in [('H4', 12, (2, 3, 4, 6, 9, 11)), ('H6', 19, tuple(range(5, 16)))]:
        for q in c.QS:
            for mode, prefix in [('FULL', rank)]+[(m, p) for p in cuts for m in ('B0', 'B0I')]:
                legacy_m5.append(dict(system=system, method=mode, prefix=prefix, q=q,
                    formula='4th(m5_best)', source='m5_legacy_plan_source#screen',
                    status='HISTORICAL_PROPOSAL_NOT_EXECUTION_EVIDENCE',
                    input_series='H4_LEGACY_DF12' if system == 'H4' else 'H6_DF19_PROPOSED',
                    input_comparable=False if system == 'H4' else None))
    return dict(rows=common, registration_inventory=[r for r in records.values() if r['registrations']],
        excluded_legacy=legacy, optional_deterministic_proposals=proposal,
        historical_m5_proposals=legacy_m5,
        source_catalog=catalog, saved_conditions=data['conditions'],
        direct_bindings=data['direct_bindings'], legacy_models=data['legacy_models'])


def encode(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()


def prepare(output):
    result = build()
    domain = c.plan(result['rows'])
    output.mkdir(parents=True, exist_ok=False)
    files = {'inventory.json': result, 'execution_contract_UNAPPROVED.json': domain,
             'comparison_policy_DRAFT.json': c.comparison_policy(),
             'grant_INVALID.json': dict(valid=False, execution_authorized=False, SOURCE=None,
                 plan_digest=c.digest(domain), call_caps=None, grant_id=None,
                 reason='Separate explicit approval of the frozen additional call budget required', mandatory_STOP=True)}
    for name, value in files.items():
        (output/name).write_bytes(encode(value))
    rows = result['rows']
    columns = ['id', 'system', 'input_series', 'method', 'prefix', 'q', 'formula', 'signal_status',
               'eligibility_status', 'same_native_cost_pair', 'any_direct_cost_pair', 'compiler_hash']
    with (output/'common_candidates_504.csv').open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=columns, lineterminator='\n', extrasaction='ignore')
        w.writeheader(); w.writerows(rows)
    (output/'summary.json').write_bytes(encode({k: v for k, v in domain.items()
        if k not in ('signal_tasks', 'compile_candidate_pool')}))
    return domain


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--check', action='store_true', help='Read-only metadata reproduction; never a scientific execution')
    args = p.parse_args()
    if args.check:
        result = build(); d = c.plan(result['rows'])
        for name, value in [('inventory.json', result), ('execution_contract_UNAPPROVED.json', d),
                            ('comparison_policy_DRAFT.json', c.comparison_policy())]:
            c.require(json.loads((args.output/name).read_text()) == value, 'METADATA_REPRODUCTION:'+name)
        c.require(d['common_candidates'] == 504 and d['additional_new_signal_candidates'] == 150
                  and d['compile_axes_max'] == 746, 'FIXED_COUNTS')
        print(json.dumps(dict(status='METADATA_REPRODUCTION_PASS', scientific_calls=0), indent=2))
        return
    d = prepare(args.output)
    print(json.dumps({k: d[k] for k in ('status', 'common_candidates', 'additional_new_signal_candidates',
         'additional_base_signal_calls', 'additional_overlay_calls', 'compile_candidates_max', 'compile_axes_max')}, indent=2))


if __name__ == '__main__':
    main()
