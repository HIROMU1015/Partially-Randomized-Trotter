#!/usr/bin/env python3
"""Read-only D1 event/key price analysis. Never invokes boundary pair kernel."""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks.algorithm_codesign.rte_development.key_mechanisms import (
    inventory_terms, linear_price, one_key_boundary, symmetric_root_split)

RESULT_COMMIT = 'fc96571881f301435a655f057709df4d6c1678e0'
BASE = Path('artifacts/track_b_rte_development_d1_result/2026-10-13/v1')
OUTPUT = Path('artifacts/track_b_d2_key_mechanisms_preparation/2026-10-13/v1')


def identity(path):
    data = path.read_bytes()
    return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def dump(path, value):
    data = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       default=lambda x: str(x) if isinstance(x, F) else None)+'\n').encode()
    if len(data) > 524288: raise RuntimeError('review file >512 KiB')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f: f.write(data)
    return identity(path)


def load(path):
    return json.loads(path.read_text())


def checked_input(root, relative, inputs):
    path = root/relative
    current = identity(path)
    data = subprocess.check_output(['git', '-C', str(root), 'show', RESULT_COMMIT+':'+str(relative)])
    if current != dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest()):
        raise PermissionError('saved input changed: '+str(relative))
    inputs[str(relative)] = current
    return load(path)


def category(e):
    if 'pauli' in e: return 'CTS_rotation' if e['rotation_sign'] else 'CTS_even'
    return 'root' if not e['word'] else 'reduced_word_length_'+str(len(e['word']))


def analyze(root=ROOT, output=OUTPUT):
    start, cpu = time.monotonic(), time.process_time()
    directory = root/output
    directory.mkdir(parents=True, exist_ok=False)
    inputs = {}; summaries = []; roots = {}; all_events = 0
    rm = checked_input(root, BASE/'resource_map.json', inputs)
    cache = checked_input(root, BASE/'synthesis_cache.json', inputs)
    inv = checked_input(root, BASE/'inventory.json', inputs)
    checked_input(root, BASE/'saved_output_audit/saved_output_audit_v1.json', inputs)
    checked_input(root, BASE/'terminal.json', inputs)
    checked_input(root, BASE/'one_shot_consumed.json', inputs)
    checked_input(root, BASE/'STOP.json', inputs)
    prices = {k: v['T_count'] for k, v in cache.items()}
    keys = {v['ratio']: v for v in inv['keys']}
    if set(keys) != set(cache): raise ArithmeticError('fixed inventory coverage')
    for index, row in enumerate(rm['rows']):
        row = checked_input(root, BASE/f'rows/row_{index:02d}.json', inputs)
        bindings = checked_input(root, BASE/row['events']['path'], inputs)
        all_events += len(bindings)
        Z = F(row['reference_law_diagnostic_only']['acceptance'])
        N, r = row['budget']['N_per_axis'], row['steps']
        factor = 2*N*r*Z**r
        mu = defaultdict(F); lit = F(0); categories = defaultdict(lambda: F(0))
        event_map = []; root_rows = []; uses = defaultdict(list)
        for b in bindings:
            e = b['event']; q = F(e['proposal']) / Z
            literal, counts = inventory_terms(b['native_ir'])
            native = linear_price(literal, counts, prices)
            if native != b['native_cost']['T']: raise ArithmeticError('new linear price identity failed')
            lit += q*literal
            for k, m in counts.items():
                mu[k] += q*m; uses[k].append(b['index'])
            group = 'paired_root_first_three' if category(e)=='root' and b['index']<3 else category(e)
            categories[group] += q*native
            v = dict(index=b['index'], source_event_json_pointer=f'/{b["index"]}',
                     source_native_IR_pointer=f'/{b["index"]}/native_ir',
                     category=group, reduced_word=e.get('word'), raw_word=e.get('raw_word'),
                     child=e.get('child'), pauli=e.get('pauli'),
                     circuit_time_word=e.get('circuit_time_word'), phase_i_power=e['phase_i_power'],
                     proposal=e['proposal'], coefficient=e['coefficient'], weight=e['weight'],
                     conditional_probability=q, literal_T=literal, saved_native_T=native,
                     key_multiplicity=dict(counts),
                     signed_R_positions=[{'IR_index':i, 'ratio':g[2], 'sign':g[3],
                                          'inventory_key':keys[g[2]]['key']} for i,g in enumerate(b['native_ir']) if g[0]=='R'])
            event_map.append(v)
            if group=='paired_root_first_three': root_rows.append(v)
        d = linear_price(lit, mu, prices)
        if d*Z != F(row['per_step_trial_native']['T']): raise ArithmeticError('conditional price')
        if d*factor != F(row['expected_native_intercept']['T']): raise ArithmeticError('whole price')
        chunks = []
        for i in range(0, len(event_map), 48):
            path = f'events/row_{index:02d}_part{i//48:02d}.json'
            chunks.append({'path':path, 'index_start':i,'index_end_exclusive':min(len(event_map),i+48),
                           **dump(directory/path,event_map[i:i+48])})
        summary = dict(row=index, arm=row['arm'], steps=r, positive_events=len(bindings),
                       original_events={**row['events'], 'result_commit':RESULT_COMMIT},
                       event_map_chunks=chunks, acceptance=Z, N_per_axis=N,
                       literal_T_conditional=lit, key_multiplicity_conditional=dict(mu),
                       conditional_T=d, whole_multiplier=factor,
                       expected_native_intercept_T=d*factor,
                       category_conditional_T=dict(categories),
                       synthesis_T_conditional=d-lit, keys=[])
        for k in sorted(mu, key=F):
            saved = cache[k]
            summary['keys'].append(dict(ratio=k, inventory_key=keys[k]['key'],
                acquisition=keys[k]['acquisition'], T_price=saved['T_count'],
                negative_price='same T/Tdagger count of actual adjoint',
                sequence_sha256=saved['sequence_sha256'],
                strict_error_upper=saved['strict_operator_error_upper'],
                multiplicity_conditional=mu[k], unit_T_contribution=mu[k]*prices[k],
                whole_T_contribution=factor*mu[k]*prices[k], event_indices=uses[k],
                angle='signed atan(ratio), scale 1; native R target exp(-i sign*atan(ratio)*Z/2)'))
        dump(directory/f'rows/row_{index:02d}.json', summary)
        summaries.append(summary); roots[(r,row['arm'])]=root_rows
    pairs = []; display = []
    for r in (1,2):
        p = next(s for s in summaries if s['steps']==r and s['arm']=='partial_return_tail')
        c = next(s for s in summaries if s['steps']==r and s['arm']=='closed_P5_full')
        delta = c['conditional_T']-p['conditional_T']
        whole_delta = c['expected_native_intercept_T']-p['expected_native_intercept_T']
        cats = {k:c['category_conditional_T'].get(k,F(0))-p['category_conditional_T'].get(k,F(0))
                for k in set(c['category_conditional_T'])|set(p['category_conditional_T'])}
        assert sum(cats.values(),F(0))==delta
        mu = {k:c['key_multiplicity_conditional'].get(k,F(0))-p['key_multiplicity_conditional'].get(k,F(0)) for k in keys}
        whole_mu = {k:c['whole_multiplier']*c['key_multiplicity_conditional'].get(k,F(0))-
                         p['whole_multiplier']*p['key_multiplicity_conditional'].get(k,F(0)) for k in keys}
        literal = c['literal_T_conditional']-p['literal_T_conditional']
        whole_literal = c['whole_multiplier']*c['literal_T_conditional']-p['whole_multiplier']*p['literal_T_conditional']
        assert linear_price(literal,mu,prices)==delta
        assert linear_price(whole_literal,whole_mu,prices)==whole_delta
        root_split = []
        for child in range(3):
            a = next(e for e in roots[(r,p['arm'])] if e['child']==child)
            b = next(e for e in roots[(r,c['arm'])] if e['child']==child)
            part = symmetric_root_split(a['conditional_probability'],b['conditional_probability'],a['saved_native_T'],b['saved_native_T'])
            root_split.append(dict(child=child,partial=a,P5=b,split=part))
        root_delta = sum((v['split']['total'] for v in root_split),F(0))
        assert root_delta==cats['paired_root_first_three']
        by_key = []
        for k in sorted(keys,key=F):
            by_key.append(dict(ratio=k,inventory_key=keys[k]['key'],actual_price=prices[k],
                              unit_delta_coefficient=mu[k],unit_delta_contribution=mu[k]*prices[k],
                              whole_delta_coefficient=whole_mu[k],whole_delta_contribution=whole_mu[k]*prices[k],
                              unit_sensitivity=one_key_boundary(delta,mu[k],prices[k]),
                              whole_sensitivity=one_key_boundary(whole_delta,whole_mu[k],prices[k])))
        synthesis = sum((whole_mu[k]*prices[k] for k in keys),F(0))
        pair = dict(steps=r,direction='P5 minus partial',conditional_delta=delta,
                    conditional_literal_delta=literal,whole_delta=whole_delta,
                    whole_literal_delta=whole_literal,category_conditional_deltas=cats,
                    root_delta=root_delta,root_share_of_net_unit_delta=root_delta/delta,
                    root_decomposition='symmetric probability/price identity, not causal',
                    root_events=root_split,keys=by_key,
                    whole_hyperplane='whole_literal_delta + sum(whole_delta_coefficient[k]*t[k]) = 0',
                    common_price_scale_tie=(-whole_literal/synthesis if synthesis else None),
                    scope='symbolic prices >=0; actual law/angles/error/N frozen; no new sequence')
        dump(directory/f'P5_vs_partial_r{r}.json',pair);pairs.append(pair)
        display.append(dict(steps=r,unit_delta=float(delta),literal_unit_delta=float(literal),
                    root_unit_delta=float(root_delta),root_net_unit_share=float(root_delta/delta),
                    root_price_component=float(sum((v['split']['price'] for v in root_split),F(0))),
                    root_probability_component=float(sum((v['split']['probability'] for v in root_split),F(0))),
                    whole_delta=float(whole_delta),
                    sensitive_keys=[dict(ratio=v['ratio'],price=v['actual_price'],contribution=float(v['whole_delta_contribution']),
                        tie=float(v['whole_sensitivity']['tie_price']) if v['whole_sensitivity']['tie_price'] is not None else None,
                        flip=v['whole_sensitivity']['strict_sign_flip_integer'])
                        for v in sorted(by_key,key=lambda x:abs(x['whole_delta_contribution']),reverse=True) if v['whole_delta_coefficient']]))
    static = [dict(row=s['row'],arm=s['arm'],positive_step_support=s['positive_events'],
                   positive_pair_bound=s['positive_events']**2) for s in summaries if s['steps']==2]
    dump(directory/'static_boundary_workload_v1.json',{'rows':static,'total_pairs':sum(s['positive_pair_bound'] for s in static),
             'registered_pair_cost_evaluations':0,'registered_cancellation_passes':0,
             'source_IR_read_for_key_multiplicity_only':True})
    dump(directory/'display_summary_v1.json',{'primary':display,'exact_authority':'P5_vs_partial_r*.json and rows/*.json'})
    dump(directory/'input_identity_v1.json',{'result_commit':RESULT_COMMIT,'files':inputs,
             'all_match_committed_originals':True,'original_classification':rm['classification']})
    dump(directory/'analysis_receipt_v1.json',{'status':'SAVED_KEY_MECHANISMS_COMPLETE',
             'rows':len(summaries),'event_links':all_events,'prices':len(prices),
             'linear_price_and_root_conservation':True,'new_science':0,'new_boundary_pairs':0,
             'new_IR':0,'new_compile':0,'new_synthesis':0,'GPU':0,'sampling':0,
             'wall_seconds':time.monotonic()-start,'CPU_seconds':time.process_time()-cpu,
             'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
             'mandatory_STOP':True,'old_classification_changed':False})
    return display


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUTPUT)
    args=p.parse_args()
    print(json.dumps(analyze(output=args.output)))
