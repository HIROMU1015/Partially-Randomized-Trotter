"""Saved metadata/receipt join only. Never loads QPY/arrays or computes prices.

The current 156 axes stay a reference to B's fixed table. Alternative schedules
come from saved T2 metadata; saved means are indexed by A's existing reader.
No predict/fit, circuit, Q3, N, weighted sum or G implementation is imported.
"""
import ast
from collections import Counter
from copy import deepcopy
import csv
from fractions import Fraction
from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
T2 = '5fb2c32e39e63267ea1748861269c362a93a1b86'
META = 'docs/research/truncated_pf_optimization_plan_20261114/PLAN_METADATA.json'
B = '1002f1dfea34de439ca1b704a0192bca473c5387'
BPATH = 'docs/research/pf6_allround_cost_20261115/INPUTS.json'
ALIGN = '0b4b2d941cd669aaa45231e1236bc2d3158a9d12'
APATH = 'docs/research/h4_five_method_candidate_alignment_20261115/ALIGNMENT.json'
LONG = '155529e20a8fac39e00786c40b57830b71203c4f'
LPATH = 'artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/'
CAT = 'e7dca5f7301c8633b17a59b6a7161331303aef2d'
CPATH = 'docs/research/d_full_pf_full_h_v2_final_receipts_20261113/COVERAGE_V2.json'
COMMON = '0b1886f53f7c3a5e99e0c13cb5f7090581c5e38c'
PFPATH = 'artifacts/resource_applicability/track_a_pf_common_H4_results/2026-10-22/native_results.json.gz'
SIGNAL_AUDIT = 'c212cee857d639d2400819bbca3afa1dc052a9f4'
SA = 'docs/research/a_signal_candidate_identity_coverage_20261114/BUILD_SIGNAL_COVERAGE.py'
COMPLETE = '1adb7e65a9e7c3adc32491d853f31d124393ead7'
SCPATH = 'docs/research/pf_shortlist_mean_completion_20261115/PACKET_INDEX.json'
FROZEN = '12372fa6b53c878dc5dd124e036572ba0cd82061'
FPATH = 'docs/research/pf_stage1_model_freeze_20261115/run01/FROZEN_MODELS.json'
NATIVE = 'af284d853890155bacd19e61850c196bd1d5c660'
H = Fraction(.1)  # Saved exact binary64 h, matching the original T2 generator.
FIELDS = ('system','method','formula','prefix','T','q','delta','R_outer','K')


def require(ok, message):
    if not ok: raise ValueError(message)


def loc(commit, path, *selector):
    return dict(commit=commit, path=path, selector=list(selector))


@lru_cache(None)
def blob(commit, path):
    return subprocess.check_output(['git','show',commit+':'+path], cwd=ROOT)


@lru_cache(None)
def js(commit, path):
    raw = blob(commit,path)
    return json.loads(gzip.decompress(raw) if path.endswith('.gz') else raw)


def resolve(ref):
    v = js(ref['commit'],ref['path'])
    for key in ref.get('selector',[]): v = v[key]
    return v


def ls(commit,path):
    return subprocess.check_output(['git','ls-tree','-r','--name-only',commit,path],cwd=ROOT,text=True).splitlines()


def physical(c):
    # Exact binary64 comparison, not rounded display values or ID alone.
    return tuple(float(c[k]).hex() if k in ('T','delta') and c.get(k) is not None else c.get(k) for k in FIELDS)


def canonical_sha(v):
    return hashlib.sha256((json.dumps(v,sort_keys=True,indent=2)+'\n').encode()).hexdigest()


def saved_signal_index():
    """Reuse only A's metadata index function, not agreement/mean arithmetic."""
    tree = ast.parse(blob(SIGNAL_AUDIT,SA)); names = {'cid','saved_pf_records'}
    funcs = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    source = js(SIGNAL_AUDIT,'docs/research/a_signal_candidate_identity_coverage_20261114/SAVED_ROUND_AUDIT.json')['sources']
    env = dict(SRC=source,js=js,ls=ls,blob=blob,gzip=gzip,json=json,hashlib=hashlib)
    exec(compile(ast.Module(body=funcs,type_ignores=[]),SA,'exec'),env)
    records = env['saved_pf_records']()
    require(len(records)==710,'A saved-index scope changed')
    out = {}
    for v in records:
        out.setdefault(v['physical_id'],[]).append({k:v[k] for k in ('source','physical_id','method','prefix','formula','T','q','note','gate')})
    return out


def main():
    meta = js(T2,META); b = js(B,BPATH); alignment = js(ALIGN,APATH)
    require(b['direct_axes']==96 and b['closure_axes']==60 and not b['issues'],'Fixed B coverage differs')
    current_ids = {v['original_configuration_id'] for v in b['axes']}
    require(len(current_ids)==6,'Six current configurations required')
    contexts = {}
    for c in alignment['candidates']:
        if c['configuration_id'] in current_ids:
            contexts[(c['prefix'],c['formula'])] = dict(binding=c['binding'],profile=c['profile'] if 'profile' in c else None,
                                                      source=loc(ALIGN,APATH,'candidates',alignment['candidates'].index(c)))
    frozen = js(FROZEN,FPATH)
    for model in frozen['models'].values():
        v = model['input_locators'][0]; c = v['physical_candidate']; context = contexts[(c['prefix'],c['formula'])]
        context.update(scalar=v['scalar'],profile=model['domain']['profile'],fragment_order=v['fragment_order'],
                       split=v['split'],input_source=loc(FROZEN,FPATH,'models',model['model_group_id'],'input_locators',0),
                       model_width=float(Fraction(v['actual_delta']['numerator'],v['actual_delta']['denominator'])) if isinstance(v['actual_delta'],dict) else v['actual_delta'])
    current_direct = {}
    for i,r in enumerate(b['axes']):
        if r['price_class']=='DIRECT_COMPILED_DETERMINISTIC':
            c = next(a['physical_candidate'] for a in alignment['rows'] if a['configuration_id']==r['original_configuration_id'] and a['m']==r['m'] and a['axis']==r['axis'])
            current_direct[(physical(c),r['axis'])] = dict(receipt=r['source']['receipt'],proof_of_current_admission=loc(B,BPATH,'axes',i))
    # Keep all fixed current prices as references, never recompute or re-audit.
    current = dict(price_table=loc(B,BPATH,'axes'),price_axis_terms=loc(B,'docs/research/pf6_allround_cost_20261115/AXIS_TERMS.json','rows'),
                   direct_axes=96,conditional_axes=60,configuration_ids=sorted(current_ids),
                   adoption_policy='PF_FIXED_COMPILER_CLOSURE_CONDITIONAL_60_AXES_V1',
                   adopted_scope='SIX_FIXED_CONFIGURATIONS_AND_ORIGINAL_FIXED_SHOT_COMPARISON',
                   new_shot_allocation_note='Physical per-shot prices do not depend on shot counts; the old conditional-use policy/claim still needs explicit scope handling by B/C.',
                   Q3_signal_packets=loc(COMPLETE,SCPATH,'packets'))
    carriers = []; scans = []
    for filename in ('exploration_costs','extension_parallel_costs','confirmation_costs','fixed_delta_exploration_costs','reused_compiled_prices'):
        path = LPATH+filename+'.json.gz'; d = js(LONG,path); carriers.append((path,d));scans.append(dict(source=loc(LONG,path),records=len(d)))
    catalog = js(CAT,CPATH); provenance = js(LONG,LPATH+'scientific_provenance.json')
    require(catalog['receipt_source_checks']['status']=='EXACT_METADATA_MATCH','Existing SOURCE audit missing')
    require(hashlib.sha256(blob(NATIVE,'scripts/h46_long_time_native.py')).hexdigest()==provenance['source_files']['scripts/h46_long_time_native.py'],'Original long-time native bytes differ')
    pfc = js(COMMON,PFPATH); pfc_sha = {canonical_sha(r):(i,r) for i,r in enumerate(pfc)}
    signal_index = saved_signal_index()
    complete = js(COMPLETE,SCPATH)
    # Shared current round signal provenance, only existence/identity/gates read.
    for packet in complete['packets']:
        config = resolve(packet['configuration']); require(config['id']==packet['configuration_id'],'Signal configuration mismatch')
        for r in config['rounds']:
            # Physical IDs are already available in the fixed alignment.
            c = next(a['physical_candidate'] for a in alignment['rows'] if a['configuration_id']==config['id'] and a['m']==r['m'])
            signal_index.setdefault(c['id'],[]).append(dict(source=packet['signal_packet'],note='COMPLETE_CURRENT_PACKET_SAME_PHYSICAL_ROUND',gate=True,physical_id=c['id']))
    all_rows = []; candidates = []; checks = []; steps = {}
    for config_index, cfg in enumerate(meta['configurations_table']):
        if cfg['prefix'] not in (7,10) or cfg['configuration_id'] in current_ids: continue
        ctx = contexts[(cfg['prefix'],cfg['formula'])]; route = 'FULL_PF' if cfg['prefix']==10 else 'TRUNCATED_PF'
        width = Fraction(cfg['requested_delta_over_h']); group_rows = []
        for r in cfg['rounds']:
            m,q = r['m'],r['q_PF']; T = float(H*2**m); delta = float(H*Fraction(r['actual_delta_over_h']))
            c = dict(system='H4',method='FULL' if cfg['prefix']==10 else 'B0I',formula=cfg['formula'],prefix=cfg['prefix'],T=T,q=q,delta=delta,R_outer=None,K=None,id=r['physical_candidate_id'])
            require((T/q).hex()==delta.hex(),'T2 actual binary64 width differs')
            ident = hashlib.sha256(json.dumps(('H4',c['method'],c['formula'],c['prefix'],T,q,None,None)).encode()).hexdigest()[:20]
            require(ident==c['id'],'Saved physical ID does not describe its schedule')
            # Reuse saved signed weights/step scopes for both route families.
            w = Fraction(r['actual_delta_over_h']); token=str(w).replace('/','o'); form=cfg['formula'].replace('(','_').replace(')','')
            spath=f'docs/research/truncated_pf_t2_results_20261114/run01/cell_p7_{form}_{token}/CELL_SCOPE.json'
            step = js(T2,spath); require(step['actual_delta_over_h']==str(w) and step['formula']==cfg['formula'],'Step SOURCE mismatch')
            skey=cfg['formula']+'|'+str(w)
            times=[(T*coef/q).hex() for coef in step['signed_weights']]
            if skey in steps: require(steps[skey]['native_one_outer_signed_time_hex']==times,'Same-width signed step metadata is not bit-identical')
            steps[skey]=dict(source=loc(T2,spath),signed_weights=step['signed_weights'],
                            source_scope='PF_WEIGHTS_ONLY; Full/p7 Hamiltonian and blocks remain in each row context, not inherited from this p7 descriptor.',
                            native_one_outer_signed_time_hex=times,
                            repetition='repeat one-outer signed list q times in order; scalar at total T once')
            signal_hits=signal_index.get(c['id'],[])
            whole_packet = next((a for a in alignment['rows'] if a['configuration_id']==cfg['configuration_id'] and a['m']==m and a['signal'].get('packet_source')),None)
            if whole_packet:
                signal=dict(status='COMPLETE_SAVED_PACKET',packet=whole_packet['signal']['packet_source'],round_sources=signal_hits)
            elif any(v['note']=='COMPLETE_CURRENT_PACKET_SAME_PHYSICAL_ROUND' or v['note'] in ('A_HEFF_PILOT_DIRECT_MEAN','A_HEFF_SCAN_DIRECT_MEAN','A_FOLLOWUP_DIRECT_MEAN','T2_TWO_TIME_DIRECT_MEAN') and v.get('gate') in (True,None) for v in signal_hits):
                signal=dict(status='SAVED_CURRENT_MEAN_NEEDS_CANDIDATE_PACKET_BINDING',round_sources=signal_hits)
            elif signal_hits: signal=dict(status='LEGACY_MEAN_QUALIFICATION_PENDING',round_sources=signal_hits)
            else: signal=dict(status='NEW_MEAN_NEEDED',round_sources=[])
            for axis,native_axis in (('X','cosine'),('Y','sine')):
                price=None; evidence=[]; state='MISSING_VALIDATION_OR_ACQUISITION'; binding='NOT_AVAILABLE'
                known=current_direct.get((physical(c),axis))
                if known:
                    state='DIRECT_SAME_PHYSICAL_EVIDENCE';evidence=[known];binding='NEW_CONFIGURATION_METADATA_BINDING_REQUIRED'
                    # Copy the fixed per-shot price; do not use any N or N*C.
                    saved = resolve(known['receipt'])
                    price = saved.get('mean',{}).get('rz_count')
                    if price is None: price = saved.get('observed_RZ')
                    if price is None:
                        # C admitted registry records may retain the original
                        # provider's mean under a different key. Preserve its
                        # locator rather than inventing a new price.
                        price = None
                else:
                    matches=[(path,k,v) for path,d in carriers for k,v in d.items() if physical(v['candidate'])==physical(c)]
                    for path,k,v in matches: evidence.append(dict(carrier=loc(LONG,path,k),trial_files=v.get('trial_files',[])))
                    cat=[(i,a) for i,a in enumerate(catalog['rows']) if physical(a['physical_candidate'])==physical(c)]
                    for i,a in cat: evidence.append(dict(catalog=loc(CAT,CPATH,'rows',i),saved_axis_status=a[axis]))
                    trials=[]
                    for path,k,v in matches:
                        if v.get('result_sha256') in pfc_sha:
                            i,t=pfc_sha[v['result_sha256']]; trials.append((t,loc(COMMON,PFPATH,i),False))
                        for filename in v.get('trial_files',[]):
                            p=Path(filename)
                            if p.is_file():
                                raw=p.read_bytes(); t=json.loads(raw)
                                trials.append((t,dict(path=filename,sha256=hashlib.sha256(raw).hexdigest(),public_carrier=loc(LONG,path,k)),True))
                    if trials:
                        t,tl,is_long=trials[0]; scalar=ctx['scalar']['value']
                        cc=t['candidate']; require(all(cc.get(k)==c[k] for k in ('system','method','prefix','formula','q')),'Saved trial physical mismatch')
                        if is_long: require(physical(cc)==physical(c),'Long trial time/width mismatch')
                        else: require(T==.8,'PF common implicit T must be .8')
                        require(t['status']=='PASS' and t['scalar'].hex()==scalar.hex(),'Trial status/scalar mismatch')
                        require(t['preparation_hash']==('bb736dc3158aa52f205ae0c23e4be374ae4e5edfa614d3cbceda04dcd5a28cfd' if cfg['prefix']==10 else 'fc83f26f1612a41d1d34caae79e78d3b5ce8e961d0e442ca4a9e92168006cf31'),'Preparation differs')
                        require(all(g.get('PASS',True) and g['error']<g.get('tolerance',g.get('strictly_below',0)) for g in t['checks']),'Original trial gate differs')
                        require(any(g['check'].lower()=='compiled_vector_'+native_axis for g in t['checks']),'Axis compiled-vector gate missing')
                        metric=t['prices'][native_axis]; rawmetric=metric if is_long else metric['metrics']
                        price = rawmetric['rz_count']  # saved integer, no recount
                        if not is_long:
                            require(metric['compiler']==ctx['profile']['compiler'],'Compiler differs')
                            require(t['hamiltonian']==ctx['binding']['hamiltonian'] and t['state']==ctx['binding']['state'],'H/state mismatch')
                        else:
                            require(all(provenance['inputs']['H4']['binding'][k]==ctx['binding'][k] for k in ('hamiltonian','state','snapshot','rank')),'Original run input binding differs')
                        # Copy no new price; this locator names the saved metric.
                        evidence.append(dict(trial=tl,gate='ORIGINAL_AXIS_GATE_PASS',metrics_selector=['prices',native_axis] if is_long else ['prices',native_axis,'metrics'],
                                             scalar=t['scalar'],preparation_hash=t['preparation_hash'],generation_SOURCE=subprocess.check_output(['git','rev-parse',provenance['source_commits']['native']+'^{commit}'],cwd=ROOT,text=True).strip() if is_long else t['SOURCE']))
                        state='DIRECT_SAME_PHYSICAL_EVIDENCE';binding='NEW_CONFIGURATION_SIDECAR_AND_C_GUARD_PENDING'
                        checks.append(dict(physical_id=c['id'],axis=axis,trial=tl,original_gate_pass=True,source_audit=loc(CAT,CPATH,'receipt_source_checks')))
                    elif matches: state='SAVED_LOCATOR_BINDING_PENDING';binding='RECOVER_ORIGINAL_TRIAL_BEFORE_REUSE_OR_NEW_COMPILE'
                row=dict(configuration_id=cfg['configuration_id'],candidate_label=cfg['label'],route=route,formula=cfg['formula'],prefix=cfg['prefix'],
                         requested_width_over_h=str(width),m=m,axis=axis,physical_id=c['id'],physical_candidate=c,
                         T_binary64_hex=T.hex(),actual_delta_binary64_hex=delta.hex(),actual_delta_over_h=r['actual_delta_over_h'],
                         signed_S2_pieces=r['signed_pieces'],signed_schedule_key=skey,context_key=str(cfg['prefix'])+'|'+cfg['formula'],
                         schedule_source=loc(T2,META,'configurations_table',config_index,'rounds',m),
                         price_status=state,price_evidence=evidence,consumer_binding=binding,conditional_model_applicable=False,
                         model_reason='Use the same-physical direct receipt where present. For missing points, the frozen group/60-axis policy does not cover the different step width; no coefficient transfer.',
                         signal=signal,saved_compiled_RZ=price,price_computed=False)
                group_rows.append(len(all_rows));all_rows.append(row)
        mine=[all_rows[i] for i in group_rows];pr=Counter(v['price_status'] for v in mine)
        sig={s:sorted({r['m'] for r in mine if r['signal']['status']==s}) for s in sorted({r['signal']['status'] for r in mine})}
        candidates.append(dict(label=cfg['label'],route=route,prefix=cfg['prefix'],formula=cfg['formula'],configuration_id=cfg['configuration_id'],requested_width_over_h=str(width),
                               relative_to_current='FINER' if width<{'2nd':Fraction(16,7),'4th':Fraction(16,3),'4th(m5_best)':Fraction(16)}[cfg['formula']] else 'COARSER',
                               schedule_source=loc(T2,META,'configurations_table',config_index),axis_rows=group_rows,price_counts=dict(pr),
                               missing_price_axes=[dict(m=r['m'],axis=r['axis'],physical_id=r['physical_id']) for r in mine if r['price_status']!='DIRECT_SAME_PHYSICAL_EVIDENCE'],
                               saved_direct_rounds=sorted({r['m'] for r in mine if r['price_status']=='DIRECT_SAME_PHYSICAL_EVIDENCE'}),
                               conditional_price_axes=0,signal_rounds=sig,all_round_prices_available=False,all_round_signal_packet=any(r['signal']['status']=='COMPLETE_SAVED_PACKET' for r in mine)))
    require(len(candidates)==12 and len(all_rows)==312,'Alternative width scope differs')
    result=dict(schema='PF_WIDTH_PRICE_EVIDENCE_METADATA_V1',status='METADATA_ONLY_NO_NEW_SCIENCE',current_156_reference=current,
                contexts={str(p)+'|'+f:v for (p,f),v in contexts.items()},signed_schedule_templates=steps,
                candidates=candidates,rows=all_rows,counts=dict(Counter(r['price_status'] for r in all_rows)),scans=scans,
                scope_limit='Saved T2 ladder, current fixed B table, known long-time carriers/catalog and A saved-mean index. No private QPY or arrays read; unlisted archives/other hosts UNKNOWN.',
                scientific_calls=dict(price_prediction=0,RZ_recount=0,QPY_audit=0,build=0,compile=0,fit=0,Q3=0,N=0,G=0),mandatory_stop=True)
    (OUT/'PRICE_COVERAGE.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (OUT/'RECEIPT_CHECKS.json').write_text(json.dumps(dict(metadata_only=True,checks=checks),ensure_ascii=False,indent=2)+'\n')
    with (OUT/'PRICE_COVERAGE.csv').open('w',newline='') as f:
        fields=['configuration_id','candidate_label','route','formula','prefix','requested_width_over_h','m','axis','physical_id','T','q','actual_delta','T_binary64_hex','actual_delta_binary64_hex','price_status','saved_compiled_RZ','consumer_binding','conditional_model_applicable','signal_status','schedule_source','price_evidence']
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for r in all_rows:
            item={k:r[k] for k in fields if k in r}
            item.update(T=r['physical_candidate']['T'],q=r['physical_candidate']['q'],actual_delta=r['physical_candidate']['delta'],signal_status=r['signal']['status'])
            for k in ('schedule_source','price_evidence'):item[k]=json.dumps(item[k],ensure_ascii=False)
            writer.writerow(item)
    source_refs=[loc(T2,META),loc(ALIGN,APATH),loc(FROZEN,FPATH),loc('606881b042ed69f9a1e7f00311417f2b65855d7b','docs/research/d_pf_qpy_closure_audit_20261115/REPORT.md'),
                 loc('c3f761f0af5b21207e64051c9c619858064450fe','docs/research/pf_closure_price_recommendation_20261115/RECOMMENDATION.json'),
                 loc('e4458cd7548977e6e11a5a7b0c8dca0d049535d4','docs/research/pf_closure_prices_20261115/run01/PRICES.json'),
                 loc(B,BPATH),loc(COMPLETE,SCPATH),loc(SIGNAL_AUDIT,SA),loc(CAT,CPATH),loc(LONG,LPATH+'scientific_provenance.json')]
    (OUT/'SOURCE_INDEX.json').write_text(json.dumps(dict(sources=[dict(r,sha256=hashlib.sha256(blob(r['commit'],r['path'])).hexdigest()) for r in source_refs],
         excluded='Private QPY/NPZ/statevectors, price predictor/fit, Q3/N/G runners; current B weighted results never recalculated.'),indent=2)+'\n')
    first=next(c for c in candidates if c['label']=='B0I_P7_2nd_1h')
    targets=[dict(configuration_id=r['configuration_id'],physical_id=r['physical_id'],m=r['m'],axis=r['axis'],physical_candidate=r['physical_candidate']) for r in all_rows if r['configuration_id']==first['configuration_id'] and r['m'] in (1,2)]
    plan=dict(status='PROPOSED_NOT_APPROVED_NOT_EXECUTED',candidate=first['label'],configuration_id=first['configuration_id'],targets=targets,
              trigger='B marks this existing full-signal-packet candidate Q3-feasible and relevant to the width/shot frontier; no cost rank inferred from S or a transferred model.',
              reuse_axes=first['price_counts']['DIRECT_SAME_PHYSICAL_EVIDENCE'],reused_rounds=first['saved_direct_rounds'],new_compile_maximum=4,signal_calls=0,
              goal='Direct prices for q2/q4 at actual width .1, combined with saved q1/q8/q16 evidence; ten of twenty-six axes, no long-q/model qualification.',
              remaining_price_axes_after_success=16,scientific_retry=False,
              duplicate_preflight='Repeat only changes to PRICE_COVERAGE rows for these four axes; compare aliases/current public receipts/catalog and original trials, plus relevant running job metadata. Unresolved axes held; only missing axes compiled.',
              caller_source='Future commit after minimal selector/axis-scope tests; existing native/compiler/gates and per-axis C bindings reused.',
              stop='First acquisition/implementation/scalar/input failure preserves partial UNKNOWN; no retry/axis substitution. After at most four compiles, publish and STOP. No fit/QPY audit/other round/Q3/N/G.')
    (OUT/'FIRST_FINITE_PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
    print('Alternative candidates:',len(candidates),'axis rows:',len(all_rows),'categories:',result['counts'])
    for c in candidates: print(c['label'],c['price_counts'],'direct rounds',c['saved_direct_rounds'],'signal', {k:len(v) for k,v in c['signal_rounds'].items()})


if __name__=='__main__': main()
