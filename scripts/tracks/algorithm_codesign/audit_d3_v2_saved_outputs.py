#!/usr/bin/env python3
"""D3 v2 saved-only identity and scalar audit; no numeric/science imports."""
import argparse
import csv
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
RESULT = 'artifacts/track_b_d3_result/2026-10-14/v2'
SOURCE_DIR = 'artifacts/track_b_d3_source/2026-10-14/v2'
SOURCE = 'dceccaa774229527a46a1386131583a4f618f8ad'
AUTHORITY = 'daa2569208a945f8e41b3c4d0de715c1e11b104b'
COMPLETE = 'D3_GUARANTEE_PRICE_MAP_COMPLETE_AWAITING_GPT_REVIEW'
LEDGERS = ('G0_P0', 'G1_P0', 'G0_P1', 'G1_P1')
METRICS = ('T', 'CX', '1Q')
D1 = 'artifacts/track_b_rte_development_d1_result/2026-10-13/v1'
D2 = 'artifacts/track_b_d2_saved_boundary_result/2026-10-13/v1'


def read(p): return json.loads(p.read_text(encoding='utf-8'))
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def audit(snapshot):
    d = ROOT/RESULT
    checks = []
    def check(name, value): checks.append(dict(name=name, PASS=bool(value)))
    c = read(ROOT/SOURCE_DIR/'execution_contract_v2.json')
    a = read(ROOT/SOURCE_DIR/'authorization.json')
    manifest = read(ROOT/SOURCE_DIR/'source_manifest_v2.json')
    protected = read(ROOT/SOURCE_DIR/'protected_hashes_v2.json')
    for section, table in (('critical', manifest['critical_sha256']), ('protected', protected['sha256'])):
        for p, expected in table.items():
            check(section+':'+p, not p.lower().endswith('.npz') and digest(ROOT/p) == expected)
    for p, entry in protected['historical_source_blobs'].items():
        raw = subprocess.check_output(['git', 'show', entry['commit']+':'+p], cwd=ROOT)
        check('historical:'+p, len(raw) == entry['bytes'] and hashlib.sha256(raw).hexdigest() == entry['sha256'])
    for entry in snapshot['files']:
        p = ROOT/entry['path']
        check('raw:'+entry['path'], p.stat().st_size == entry['bytes'] and digest(p) == entry['sha256'])
    marker = read(d/'one_shot_consumed.json'); terminal = read(d/'terminal.json')
    inner = read(d/'inner_terminal.json'); outer = read(d/'outer_process_receipt.json')
    stop = read(d/'STOP.json'); rm = read(d/'resource_map.json'); em = read(d/'evidence_manifest.json')
    identity = dict(source_commit=SOURCE, execution_HEAD=AUTHORITY,
                    authorization_sha256=digest(ROOT/SOURCE_DIR/'authorization.json'),
                    contract_sha256=digest(ROOT/SOURCE_DIR/'execution_contract_v2.json'),
                    design_contract_sha256=a['design_contract_sha256'], inventory_sha256=a['inventory_sha256'],
                    source_manifest_sha256=digest(ROOT/SOURCE_DIR/'source_manifest_v2.json'),
                    tool_identity_sha256=c['acquisition']['tool_identity']['sha256'])
    for name, value in (('marker', marker), ('terminal', terminal), ('inner', inner), ('outer', outer),
                        ('map', rm), ('evidence manifest', em), ('job start', read(d/'job_start_receipt.json')),
                        ('ledger job', read(d/'ledger_job.json')), ('ledger start', read(d/'ledger_started.json')),
                        ('preflight', read(d/'preflight.json'))):
        check(name+':identity', all(value.get(k) == v for k, v in identity.items()))
        if name != 'marker': check(name+':marker', value['marker_sha256'] == digest(d/'one_shot_consumed.json'))
    check('direct S2 parent', subprocess.check_output(['git','show','-s','--format=%P',AUTHORITY],cwd=ROOT).decode().strip() == SOURCE)
    changed = subprocess.check_output(['git','diff','--name-only',SOURCE,AUTHORITY],cwd=ROOT).decode().splitlines()
    check('authority-only diff', set(changed) == {SOURCE_DIR+'/authorization.json',
                                               'docs/tracks/algorithm_codesign/d3_v2_execution_authorization_receipt.md'})
    check('all completion labels', terminal['status'] == inner['status'] == outer['inner_status']
          == outer['effective_status'] == rm['classification'] == em['classification'] == COMPLETE)
    check('outer exit0 / no cap / inner consistency', outer['exit_code'] == 0 and outer['signal'] is None
          and outer['resource_failure'] is None and outer['status_consistent'] is True and terminal['technical_reason'] is None)
    check('inner hash link', outer['inner_terminal_sha256'] == digest(d/'inner_terminal.json'))
    check('outer hash link', terminal['outer_process_receipt_sha256'] == digest(d/'outer_process_receipt.json'))
    check('STOP hash link', stop['terminal_sha256'] == digest(d/'terminal.json'))
    for name, value in (('inner',inner),('outer',outer),('terminal',terminal),('STOP',stop),('map',rm)):
        check(name+':STOP', value['retries'] == 0 and value['mandatory_STOP'] is True
              and value['next_science_authorized'] is False)
    for entry in em['files']:
        path = d/entry['path']
        check('writer:'+entry['path'], path.stat().st_size == entry['bytes'] and digest(path) == entry['sha256'])
        if entry['chunks']:
            h = hashlib.sha256(); count = 0
            for part in entry['chunks']:
                raw = (d/part['path']).read_bytes(); raw.decode('utf-8')
                check('chunk:'+part['path'], len(raw) <= 524288 and len(raw) == part['bytes']
                      and hashlib.sha256(raw).hexdigest() == part['sha256'])
                h.update(raw); count += len(raw)
            check('lossless:'+entry['path'], count == entry['bytes'] and h.hexdigest() == entry['sha256'])
    inventory = read(d/'frozen_inventory.json')
    original = read(ROOT/c['acquisition']['key_inventory']['path'])
    check('inventory snapshot exact values', inventory == original)
    cache = read(d/'synthesis_cache_new.json'); oldcache = read(ROOT/D1/'synthesis_cache.json')
    receipts = read(d/'synthesis_process_receipts.json')
    check('all36 cache / receipt coverage', len(cache) == len(receipts) == 36
          and set(cache) == set(oldcache) == {v['ratio'] for v in inventory['keys']})
    for i, key in enumerate(inventory['keys']):
        prefix = d/f'jobs/key_{i:03d}'
        job = read(prefix.with_suffix('.json')); start = read(prefix.with_suffix('.started.json'))
        token = read(prefix.with_suffix('.backend_call_consumed.json'))
        value = read(prefix.with_suffix('.result.json')); receipt = read(prefix.with_suffix('.process_receipt.json'))
        label = f'key{i:03d}'
        for role, record in (('job',job),('start',start),('token',token),('result',value)):
            check(label+':'+role+':identity', all(record.get(k) == v for k,v in identity.items())
                  and record['marker_sha256'] == digest(d/'one_shot_consumed.json'))
        check(label+':inventory', job['key'] == key['new_key'] and job['ratio'] == key['ratio'] and job['job_index'] == i)
        check(label+':owner/job hash', job['parent_pid'] == read(d/'ledger_started.json')['pid']
              and start['job_path_sha256'] == digest(prefix.with_suffix('.json')))
        check(label+':one complete call', value['status'] == 'KEY_COMPLETE' and value['synthesis_calls'] == 1
              and value['retries'] == 0 and token['backend_call_token'] == 1 and token['retries'] == 0)
        saved = cache[key['ratio']]; seq = saved['sequence']
        check(label+':cache=result', saved == value['saved'])
        check(label+':sequence identity/counts', hashlib.sha256(seq.encode()).hexdigest() == saved['sequence_sha256']
              and all(g in 'HTtSXW' for g in seq) and len(seq) <= c['caps']['sequence_characters']
              and saved['T_count'] == seq.count('T')+seq.count('t')
              and saved['Tdagger_count'] == seq.count('t') and saved['global_W_count'] == seq.count('W')
              and saved['one_qubit_count'] == len(seq)-seq.count('W'))
        check(label+':angle/precision identity', saved['angle_key'] == oldcache[key['ratio']]['angle_key']
              and saved['epsilon'] == '1/10000000' and saved['key'] == saved['angle_key']+':epsilon:1/10000000')
        check(label+':strict saved bound', saved['error_pass'] is True
              and 0 <= F(saved['strict_operator_error_upper']) <= F(1,10**7))
        check(label+':receipt identity/exit', receipt['result_sha256'] == digest(prefix.with_suffix('.result.json'))
              and receipt['key'] == job['key'] and receipt['ratio'] == job['ratio'] and receipt['exit_code'] == 0)
        check(label+':CPU/RSS caps', receipt['CPU_seconds'] <= 20 and receipt['peak_RSS_KiB'] <= 512*1024)
    for glob in ('*.started.json','*.backend_call_consumed.json','*.result.json','*.process_receipt.json'):
        check('36:'+glob, len(list((d/'jobs').glob(glob))) == 36)
    check('outer / inner exact coverage', terminal['rows'] == inner['rows'] == 48
          and terminal['new_synthesis_calls'] == inner['new_synthesis_calls'] == 36
          and outer['key_jobs_started'] == outer['key_result_files_saved'] == outer['backend_call_tokens_consumed'] == 36)
    check('scope counters', inner['event_metadata_reads'] == rm['event_metadata_reads'] == 1890
          and all(inner[k] == 0 for k in ('registered_pair_evaluations','provider_or_signal_matrix_evaluations',
                                         'circuit_builds','native_compile','sampling','GPU'))
          and rm['quantum_shots'] == 0 and inner['strict_guard_2x2_key_checks'] == 36)
    check('resource caps', outer['CPU_seconds_including_children'] <= 1800
          and outer['child_peak_RSS_KiB'] <= 512*1024
          and outer['observed_peak_process_RSS_bytes'] <= 512*1048576
          and outer['observed_peak_aggregate_RSS_bytes'] <= 1536*1048576
          and snapshot['total_bytes'] <= 16777216)
    check('map four ledgers /48', set(rm['rows']) == set(LEDGERS) and rm['rows_count'] == 48)
    check('no joint confidence/adoption', rm['four_ledger_joint_confidence_claim'] is False
          and rm['method_adoption'] is False and rm['all_native_error_accounts'] == '1/1000000')
    displayed = []; views = []
    for ledger in LEDGERS:
        rows = rm['rows'][ledger]; check(ledger+':12 rows', len(rows) == 12)
        compare = read(d/f'comparisons/{ledger}.json')
        views.append(dict(ledger=ledger, source='comparisons/'+ledger+'.json',
                          all12_T_min_h0=compare['all12']['T']['h0_ranking'][0],
                          primary_T_min_h0=compare['primary_equal_r_choice']['T']['h0_ranking'][0],
                          all12_h0_rankings={k:compare['all12'][k]['h0_ranking'] for k in METRICS},
                          primary_equal_r_choice=compare['primary_equal_r_choice'],
                          scientific_method_adoption=False))
        for i,row in enumerate(rows):
            old = read(ROOT/D1/f'rows/row_{i:02d}.json')
            r = row['steps']; n = row['budget']['N_per_axis']; z = F(row['step_acceptance'])
            boundary = read(ROOT/D2/f'rows/row_{i:02d}.json') if r == 2 else None
            label = ledger+f':row{i:02d}'
            check(label+':row file exact', row == read(d/f'rows/{ledger}/row_{i:02d}.json'))
            check(label+':input', row['row_index'] == i and row['ledger'] == ledger
                  and row['arm'] == old['arm'] and r == old['steps'] and row['degree'] == 5
                  and F(row['x']) == F(5,7)/r and row['events_sha256'] == old['events']['sha256'])
            check(label+':same law/bounds', row['production_step_bounds'] == old['production_step_bounds']
                  and z == F(old['reference_law_diagnostic_only']['acceptance'])
                  and F(row['acceptance_path']) == z**r)
            check(label+':fixed account/precision', F(row['native_error_account']) == F(1,10**6)
                  and row['new_sequence_precision'] == ('1/10000000' if ledger.endswith('P1') else '1/1000000'))
            check(label+':shot/axis contract', type(n) is int and 0 < n <= 10**8
                  and F(row['budget']['alpha_axis']) == F(49,24000)
                  and F(row['budget']['resource_failure_per_row']) == F(1,12000)
                  and row['budget']['quantum_shots_executed'] == 0 and row['budget']['uses_signal'] is False)
            check(label+':saved production moment/range', F(row['budget']['M2_upper']) == F(old['production_step_bounds']['M2_upper'])**r
                  and F(row['budget']['range_upper']) == F(old['production_step_bounds']['range_upper'])**r)
            check(label+':quantum calls/slope', F(row['prep_readout_slope']) == F(row['expected_quantum_calls']) == 2*n*z**r)
            check(label+':unknown classical', row['classical_online_per_trial_cost'] == 'NOT_ACQUIRED; never zero')
            for metric in METRICS:
                saving = F(boundary['cancellation_saving_per_whole_trial'][metric]) if boundary else F(0)
                price = F(row['per_whole_trial_native'][metric])
                check(label+':'+metric+':saved scalar cost', F(row['literal_boundary_saving_per_whole_trial'][metric]) == saving
                      and price == r*z**(r-1)*F(row['per_step_trial_native'][metric])-saving
                      and F(row['expected_native_intercept'][metric]) == 2*n*price)
                check(label+':'+metric+':tail/hard', F(row['accepted_tail_native'][metric]) == row['budget']['accepted_tail_calls']*r*row['worst_single_step_event_native'][metric]
                      and F(row['hard_native'][metric]) == 2*n*r*row['worst_single_step_event_native'][metric])
            if ledger == 'G0_P0':
                target = boundary['new_expected_native_intercept'] if boundary else old['expected_native_intercept']
                check(label+':legacy budget', row['budget'] == old['budget'])
                check(label+':legacy anchor costs', all(F(row['expected_native_intercept'][k]) == F(target[k]) for k in METRICS)
                      and F(row['prep_readout_slope']) == F(old['prep_readout_slope']))
            displayed.append(dict(ledger=ledger, row_index=i, arm=row['arm'], r=r, x=row['x'],
                                  N_per_axis=n, bias_upper=row['budget']['axis_bias_upper'], remaining=row['budget']['remaining'],
                                  M2_upper=row['budget']['M2_upper'], range_upper=row['budget']['range_upper'],
                                  acceptance_step=row['step_acceptance'], acceptance_path=row['acceptance_path'],
                                  T_intercept=row['expected_native_intercept']['T'], CX_intercept=row['expected_native_intercept']['CX'],
                                  oneQ_intercept=row['expected_native_intercept']['1Q'], prep_readout_slope=row['prep_readout_slope'],
                                  workspace_beyond_system=row['workspace_beyond_system'], native_error_account=row['native_error_account']))
    for i in range(12):
        for guarantee in ('G0','G1'):
            check(guarantee+f':row{i:02d}:no precision credit', rm['rows'][guarantee+'_P0'][i]['budget'] == rm['rows'][guarantee+'_P1'][i]['budget'])
    effects = read(d/'factorial_effects.json'); check('48 factorial effects', len(effects) == 48)
    for e in effects:
        i = e['row_index']; field = e['field']; metric = e['coordinate']
        def value(ledger):
            v = rm['rows'][ledger][i][field]
            return F(v[metric] if field == 'expected_native_intercept' else v)
        a0,b0,c0,d0 = map(value,LEDGERS)
        check('factorial:'+str(i)+':'+field+':'+metric,
              F(e['guarantee_only']) == b0-a0 and F(e['price_only_G0']) == c0-a0
              and F(e['price_only_G1']) == d0-b0 and F(e['interaction']) == (d0-b0)-(c0-a0))
    report = dict(kind='D3_V2_SAVED_ONLY_OUTPUT_AUDIT', date_JST='2026-10-14', checks=checks,
                  passed=sum(v['PASS'] for v in checks), failed=sum(not v['PASS'] for v in checks),
                  status='PASS_SAVED_OUTPUT_INTEGRITY' if all(v['PASS'] for v in checks) else 'AUDIT_FAILURE',
                  source_commit=SOURCE, execution_authorization_commit=AUTHORITY,
                  marker_sha256=digest(d/'one_shot_consumed.json'), final_classification_unchanged=COMPLETE,
                  raw_snapshot_reference='raw_output_snapshot_v2.json', rows=48, new_synthesis_calls=36,
                  saved_only_scalar_arithmetic=True, science_module_imports=0, new_backend_or_pair_or_matrix_calls=0,
                  reranked_or_reoptimized=False, mandatory_STOP=True, next_science_authorized=False)
    return report, displayed, dict(kind='DIRECT_PROJECTION_OF_PREREGISTERED_SAVED_COMPARISONS',
                                  views=views, h0_is_zero_prep_readout_cost=True, additional_optimization=False,
                                  scientific_interpretation_or_adoption=False, mandatory_STOP=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--snapshot',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True)
    args = p.parse_args(); report, rows, views = audit(read(args.snapshot))
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for name,value in (('saved_output_audit_v2.json',report),('saved_comparison_projection_v2.json',views)):
        with (args.output_dir/name).open('x',encoding='utf-8') as f:
            json.dump(value,f,ensure_ascii=False,indent=2); f.write('\n')
    with (args.output_dir/'resource_rows_exact_v2.csv').open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({k:report[k] for k in ('status','passed','failed','rows','new_synthesis_calls','mandatory_STOP')}))
    raise SystemExit(1 if report['failed'] else 0)
