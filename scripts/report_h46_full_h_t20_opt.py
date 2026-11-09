"""Saved-value accounting for the T=2.0 full-H RTE optimization."""
from collections import defaultdict
from decimal import Decimal,localcontext
from fractions import Fraction
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess

import h46_full_h_t20_opt as s


def best(rows):
    valid=[v for v in rows if v.get('G') is not None and v['status']=='ELIGIBLE']
    return min(valid,key=lambda v:Fraction(v['G_exact'])) if valid else None


def write_csv(path,rows):
    from h46_long_time_report import write_csv as existing
    existing(path,rows)
    # Keep the reused field/value serialization; use repository LF newlines.
    path.write_bytes(path.read_bytes().replace(b'\r\n',b'\n'))


def group_sources():
    """Never pool exploration, separate phases or independent confirmation."""
    result={};confirm={}
    for path in sorted(s.PUBLIC.glob('*_costs.json.gz')):
        for identity,value in s.read(path).items():
            target=confirm if 'CONFIRMATION' in value['cohort'] else result
            s.full.require(identity not in target,'DUPLICATE_PRICE_GROUP_NO_POOLING')
            target[identity]=dict(value,source_file=path.name)
    if (s.PUBLIC/'costs.json.gz').exists():
        for identity,value in s.read(s.PUBLIC/'costs.json.gz').items():
            s.full.require(identity not in result,'DUPLICATE_PRICE_GROUP')
            result[identity]=dict(value,source_file='costs.json.gz')
    return result,confirm


def account():
    s.setup()
    from h46_long_time_report import cost_for
    from h46_long_time_native import cdf_law
    records=s.read(s.PUBLIC/'signal_records.json.gz');endpoints=s.read(s.PUBLIC/'endpoints.json.gz')
    prices,confirmed=group_sources()
    complete=s.read(s.PUBLIC/'exploration_v2_completion.json')
    s.full.require(complete['status']=='PASS' and complete['all_workers_joined'] and complete['new_trials']==90,'EXPLORATION_DRAINED')
    if confirmed:
        complete=s.read(s.PUBLIC/'confirmation_h4_completion.json')
        s.full.require(complete['status']=='PASS' and complete['all_workers_joined'] and complete['new_trials']==66,'CONFIRMATION_DRAINED')
    confirmation=confirmation_account(confirmed,endpoints)
    data=[];trials=[];rarity=[]
    contexts={k:s.read(s.PRIVATE/(k+'_context.json')) for k in ('H4','H6')}
    for identity,group in prices.items():
        r=group['candidate'];sig=records[identity]['signal'];ts=[s.read(p) for p in group['trial_files']]
        s.full.require(len(ts)==9 and {t['index'] for t in ts}==set(range(9)),'N9')
        for t in ts:
            s.full.require(t['status']=='PASS' and not t['conditional'] and t['candidate']==r and t['cohort']==group['cohort'],'TRIAL_IDENTITY')
            s.full.require(abs(t['log_B']-sig['log_B'])<1e-9,'MEAN_NATIVE_LOG_B')
            s.full.require(abs(t['scalar']-contexts[r['system']]['scalar'])<1e-12,'SCALAR_ONCE')
            trials.append(t)
        for order in range(2,r['K']+1,2):
            with localcontext() as ctx:
                ctx.prec=80;ideal_no=Decimal(1);bit_no=Decimal(1)
                for law in ts[0]['laws']:
                    ideal=sum(Decimal.from_float(float(p)) for k,p in zip(law['orders'],law['probabilities']) if k>=order)
                    effective=cdf_law(law['probabilities'])['probabilities']
                    bit=sum(Decimal.from_float(float(p)) for k,p in zip(law['orders'],effective) if k>=order)
                    ideal_no*=(1-ideal)**law['steps'];bit_no*=(1-bit)**law['steps']
                rarity.append(dict(id=identity,system=r['system'],cohort=group['cohort'],order_or_higher=order,
                    saved_law_probability=str(1-ideal_no),effective_CDF_probability=str(1-bit_no),
                    probability_unobserved_n9=str(ideal_no**9),
                    observed_exact_order_events=group['rarity']['observed_order_counts'].get(str(order),0),
                    conditional_compiled_price=None,population_tail_cost_bound=None,unobserved_price_imputed=False))
    keys={(t['cohort'],t['candidate']['id'],t['index']) for t in trials}
    s.full.require(len(keys)==len(trials),'UNIQUE_TRIALS')
    confirmation_keys={(g['cohort'],g['candidate']['id'],i) for g in confirmed.values() for i in range(33)}
    s.full.require(len(confirmation_keys)==confirmation['trials'] and not keys.intersection(confirmation_keys),'SEPARATE_UNIQUE_CONFIRMATION')
    for ep in endpoints:
        r=ep['candidate'];sig=records[r['id']]['signal'];info=contexts[r['system']]
        out=dict(r,epsilon=ep['epsilon'],status=ep['status'],N_cos=ep['N_cos'],N_sin=ep['N_sin'],
            lambda_R=info['lambda_R'],log_B=ep['log_B'],B=math.exp(ep['log_B']),u=ep['u'],
            log_B_margin=ep['log_B_margin'],headroom_cos=ep['axes']['real']['headroom'],
            headroom_sin=ep['axes']['imag']['headroom'],proxy_G=ep['proxy_G'],proxy_C=ep['proxy_C'],
            price_status='ACQUIRED' if r['id'] in prices else 'NOT_ACQUIRED',G=None,G_SE=None,G_exact=None,
            C_cos=None,C_sin=None,CX_cos=None,CX_sin=None,n=0,cohort=None)
        for key,label in [('corrected','corrected'),('raw','raw'),('reference','reference'),('total_signed_bias','bias')]:
            out[label+'_cos']=sig[key]['real'];out[label+'_sin']=sig[key]['imag']
        if ep['status']=='ELIGIBLE' and r['id'] in prices:
            g=prices[r['id']];out.update(cost_for(ep,g))
            out['CX_cos']=g['mean_metrics']['cosine']['cx_count'];out['CX_sin']=g['mean_metrics']['sine']['cx_count']
            ts=[t for t in trials if t['candidate']['id']==r['id']]
            direct=Fraction(sum(ep['N_cos']*t['prices']['cosine']['rz_count']+ep['N_sin']*t['prices']['sine']['rz_count'] for t in ts),len(ts))
            s.full.require(direct==Fraction(out['G_exact']),'EXACT_INTEGER_G')
        data.append(out)
    s.save(s.PUBLIC/'accounting.json.gz',data)
    write_csv(s.PUBLIC/'RESULTS.csv',[v for v in data if v['price_status']=='ACQUIRED'])
    s.save(s.PUBLIC/'price_trials.json.gz',trials)
    s.save(s.PUBLIC/'rare_event_status.json',rarity)
    baselines={'B3':s.read(s.B3/'accounting.json.gz'),'B2_PF':s.read(s.BASE/'accounting.json.gz')}
    for name,folder in [('B3',s.B3),('B2_PF',s.BASE)]:
        provenance=s.read(folder/'scientific_provenance.json')
        for system in contexts:
            s.full.require(provenance['inputs'][system]['binding']==contexts[system]['binding'],'SAME_SAVED_INPUT')
        s.full.require(provenance['compiler']['hash']==contexts['H4']['binding']['compiler'],'SAME_COMPILER')
    nominees=[];comparisons=[]
    saved_contexts={system:s.read(s.full.BASE_PRIVATE/(system+'_context.json')) for system in contexts}
    for system in ('H4','H6'):
        for e in s.full.signal.EPS:
            domain=[v for v in data if v['system']==system and v['epsilon']==e]
            choice=best(domain)
            if choice is None:continue
            nominees.append(choice)
            bd=[v for v in baselines['B3'] if v['system']==system and v['T']==2.0 and v['epsilon']==e]
            old=[v for v in baselines['B2_PF'] if v['system']==system and v['T']==2.0 and v['epsilon']==e]
            b3match=next((v for v in bd if all(v[k]==choice[k] for k in ('q','R_outer','K','formula'))),None)
            initial=next((v for v in domain if v['q']==(8 if system=='H4' else 32) and v['R_outer']==512 and v['K']==2),None)
            transferred=next((v for v in domain if v['q']==(16 if system=='H4' else 32) and v['R_outer']==(16 if system=='H4' else 32) and v['K']==(4 if system=='H4' else 6)),None)
            refs={'INITIAL_GRID_NOMINEE':initial,'TRANSFERRED_T08_PARAMETERS_AT_T20':transferred,
                'SAVED_B3_BEST':best(bd),'SAVED_B2_BEST':best([v for v in old if v['method']=='PR']),
                'SAVED_FULL_PF_BEST':best([v for v in old if v['method']=='FULL']),
                'SAVED_PF_ENVELOPE_BEST':best([v for v in old if v['method']!='PR']),
                'MATCHED_B3':b3match}
            for role,b in refs.items():
                if b is not None:
                    for axis in ('cos','sin'):
                        s.full.require(abs(b['reference_'+axis]-choice['reference_'+axis])<1e-10,'SAME_REFERENCE_T20')
                g=None if b is None else b.get('G')
                lam=None if b is None else b.get('lambda_R')
                if b is not None and lam is None:
                    if 'B3' in role:lam=contexts[system]['lambda_R']-contexts[system]['one_body_lambda_R']
                    elif b['method']=='PR':lam=saved_contexts[system]['prefixes'].get(str(b['prefix']),{}).get('lambda_R')
                    else:lam=0.
                comparisons.append(dict(system=system,T=2.0,epsilon=e,role=role,id=choice['id'],
                    pure_q=choice['q'],pure_R=choice['R_outer'],pure_K=choice['K'],pure_G=choice['G'],
                    pure_lambda_R=choice['lambda_R'],pure_log_B=choice['log_B'],pure_B=choice['B'],
                    pure_N_cos=choice['N_cos'],pure_N_sin=choice['N_sin'],pure_C_cos=choice['C_cos'],pure_C_sin=choice['C_sin'],
                    baseline_id=None if b is None else b['id'],baseline_G=g,
                    baseline_lambda_R=lam,baseline_log_B=None if b is None else b['log_B'],
                    baseline_B=None if b is None else math.exp(b['log_B']),
                    baseline_N_cos=None if b is None else b['N_cos'],baseline_N_sin=None if b is None else b['N_sin'],
                    baseline_C_cos=None if b is None else b.get('C_cos'),baseline_C_sin=None if b is None else b.get('C_sin'),
                    baseline_q=None if b is None else b['q'],baseline_R=None if b is None else b.get('R_outer'),
                    baseline_K=None if b is None else b.get('K'),baseline_formula=None if b is None else b['formula'],
                    baseline_prefix=None if b is None else b['prefix'],baseline_n=None if b is None else b.get('n'),
                    baseline_cohort=None if b is None else b.get('cohort'),
                    G_ratio=None if g is None else choice['G']/g,
                    point_reduction_fraction=None if g is None else 1-choice['G']/g,
                    comparison_status='SAVED_PRICE_NOT_ACQUIRED' if g is None else 'FINITE_SAMPLE_POINT_COMPARISON',
                    method_superiority_confirmed=False,
                    independent_confirmation_scope='H4_FIXED_R448_VS_R512_ONLY' if choice['id'] in confirmed else 'NONE',
                    comparison_uses_exploration_prices=True))
    s.save(s.PUBLIC/'best_configurations.json',nominees)
    write_csv(s.PUBLIC/'best_configurations.csv',nominees)
    write_csv(s.PUBLIC/'comparisons.csv',comparisons)
    # Rebuild inventory directly from the accepted signal records, never from a price proxy.
    s.save(s.PUBLIC/'candidate_inventory.json',[dict(rec['candidate'],signal_status='EVALUATED',
        price_status='ACQUIRED' if rec['candidate']['id'] in prices else 'NOT_ACQUIRED') for rec in records.values()])
    boundary_record(data,nominees,comparisons)
    write_report(data,nominees,comparisons,contexts,prices,confirmed,confirmation)
    publish_index(records,prices,trials,confirmed)
    validation=dict(status='PASS',signal_candidates=len(records),precision_endpoints=len(data),
        eligible_endpoints=sum(v['status']=='ELIGIBLE' for v in data),
        unpriced_candidates=len(records)-len(prices),all_acquisition_workers_joined=True,
        priced_candidates=len(prices),price_trials=len(trials),unique_trials=len(keys),
        independent_confirmation_candidates=len(confirmed),independent_confirmation_trials=confirmation['trials'],
        exploration_confirmation_pooled=False,confirmation_after_selection_reselection=False,
        independent_confirmation_maximum_native_error=confirmation.get('maximum_native_vector_error'),
        independent_confirmation_maximum_compiled_error=confirmation.get('maximum_compiled_vector_error'),
        maximum_native_vector_error=max((c['error'] for t in trials for c in t['checks'] if not c['check'].startswith('compiled_')),default=None),
        maximum_compiled_vector_error=max((c['error'] for t in trials for c in t['checks'] if c['check'].startswith('compiled_')),default=None),
        integer_G_crosscheck=True,signal_native_log_B_match=True,comparison_input_reference_compiler_match=True,
        fixed_SOURCE_baseline_and_input_conservation=True,
        synthetic_checks='s-equivalence plus 3 corrected Aer gate/global-phase/thread tests',
        new_reference_calls=0,reference_counter_semantics='saved target reference generation; exact-tail control column is evaluated internally',input_generation=0,unobserved_cost_imputed=False)
    s.save(s.PUBLIC/'validation_summary.json',validation)
    print(json.dumps(validation))


def confirmation_account(groups,endpoints):
    from h46_long_time_report import cost_for
    rows=[];trials=[]
    for identity,g in groups.items():
        ts=[s.read(path) for path in g['trial_files']]
        s.full.require(len(ts)==33 and {t['index'] for t in ts}==set(range(33)),'FIXED_CONFIRMATION_N33')
        s.full.require(all(t['status']=='PASS' and not t['conditional'] and t['candidate']==g['candidate'] and t['cohort']==g['cohort'] for t in ts),'CONFIRMATION_BINDING')
        trials.extend(ts)
        ep=next(e for e in endpoints if e['candidate']['id']==identity and e['epsilon']==.0005)
        s.full.require(max(abs(t['log_B']-ep['log_B']) for t in ts)<1e-9,'CONFIRMATION_LOG_B')
        scalar=s.read(s.PRIVATE/(g['candidate']['system']+'_context.json'))['scalar']
        s.full.require(max(abs(t['scalar']-scalar) for t in ts)<1e-12,'CONFIRMATION_SCALAR_ONCE')
        value=dict(g['candidate'],epsilon=.0005,**cost_for(ep,g))
        direct=Fraction(sum(ep['N_cos']*t['prices']['cosine']['rz_count']+ep['N_sin']*t['prices']['sine']['rz_count'] for t in ts),33)
        s.full.require(direct==Fraction(value['G_exact']),'CONFIRMATION_INTEGER_G')
        rows.append(value)
    result=dict(trials=len(trials),candidates=len(groups),
        maximum_native_vector_error=max((c['error'] for t in trials for c in t['checks'] if not c['check'].startswith('compiled_')),default=None),
        maximum_compiled_vector_error=max((c['error'] for t in trials for c in t['checks'] if c['check'].startswith('compiled_')),default=None),
        rarity={identity:g['rarity'] for identity,g in groups.items()},pooled=False,reselection=False,top_up=False,primary_epsilon=.0005)
    if rows:
        a=next(r for r in rows if r['R_outer']==448);b=next(r for r in rows if r['R_outer']==512)
        delta=a['G']-b['G'];se=math.hypot(a['G_SE'],b['G_SE'])
        result.update(G_R448=a['G'],G_R512=b['G'],ratio=a['G']/b['G'],difference=delta,difference_MCSE=se,
            normal_95_diagnostic=[delta-1.96*se,delta+1.96*se],point_direction_reproduced=delta<0,
            population_tail_guarantee=False,statistical_superiority_claim=False)
    write_csv(s.PUBLIC/'confirmation_accounting.csv',rows)
    s.save(s.PUBLIC/'confirmation_summary.json',result)
    s.save(s.PUBLIC/'confirmation_trials.json.gz',trials)
    return result


def boundary_record(data,nominees,comparisons):
    entries=[]
    for n in nominees:
        same=[v for v in data if v['system']==n['system'] and v['epsilon']==n['epsilon'] and v['K']==n['K']]
        qs=[v['q'] for v in same];rs=[v['R_outer'] for v in same]
        initial=next(c for c in comparisons if c['system']==n['system'] and c['epsilon']==n['epsilon'] and c['role']=='INITIAL_GRID_NOMINEE')
        probes=[dict(q=v['q'],R=v['R_outer'],K=v['K'],status=v['status'],proxy_G=v['proxy_G'],G=v['G'],price_status=v['price_status']) for v in same if v['R_outer'] in (1024,2048) and v['q']==(8 if n['system']=='H4' else 32)]
        entries.append(dict(system=n['system'],epsilon=n['epsilon'],id=n['id'],q=n['q'],R=n['R_outer'],K=n['K'],
            initial_nominee_R_boundary=512,initial_nominee_K_boundary=2,
            R_signal_domain=[min(rs),max(rs)],q_signal_domain=[min(qs),max(qs)],
            R_best_inside_evaluated_domain=min(rs)<n['R_outer']<max(rs),
            K_at_mathematical_lower_bound=n['K']==0,
            initial_to_final_G_ratio=initial['G_ratio'],upper_boundary_probes=probes,
            state='EXTENSION_EVALUATED_ACQUIRED_MINIMUM_INSIDE_SIGNAL_DOMAIN_PRICE_SUBSET_ONLY',
            remaining_unpriced_candidates=sum(v['price_status']=='NOT_ACQUIRED' for v in same),
            global_compiled_optimality_confirmed=False))
    s.save(s.PUBLIC/'boundary_status.json',dict(entries=entries,
        chronology=['initial B3 q1..64/R4..512/K2,4,6','adjacent lower K0','fixed-q R1024 plus local R interpolation','H6 K0 R2048 headroom-only probe'],
        initial_total_step_domain=[4,32768],
        factorization_note='Most fixed-q R extensions alias s=qR already available at other initial q/R; R448 adds intermediate s; H6 K0 s65536 genuinely extends the total-step upper bound',
        headroom_only_probe=[dict(q=v['q'],R=v['R_outer'],K=v['K'],epsilon=v['epsilon'],status=v['status'],N_cos=v['N_cos'],N_sin=v['N_sin'],proxy_G=v['proxy_G'],price_status=v['price_status']) for v in data if v['system']=='H6' and v['q']==32 and v['R_outer']==2048 and v['K']==0],
        stopping_reason='Acquired minima lie inside the evaluated s/R range; adjacent upper probes have no promising proxy improvement. K0 reaches its mathematical lower bound. Unpriced actual G and global q/R optimum remain unresolved; no dominance inferred from proxy.',
        K8_not_evaluated_reason='K6 upper boundary was not a signal/proxy nominee; K2 and K0 were selected',
        q_extension_reason='no acquired best at q64 upper edge; q>=1 mathematical lower edge; same-s signal equivalence is not a compiled-price guarantee',
        no_unbounded_extension=True,unpriced_upper_probe_costs_imputed=False))

def write_report(data,nominees,comparisons,contexts,prices,confirmed,confirmation):
    lookup={(c['system'],c['epsilon'],c['role']):c for c in comparisons}
    lines=['# DF全Hamiltonian純RTE：T=2.0最適化・境界拡張','',
        'linear H4/H6、1.00 Å、STO-3G、保存tol-only DF10/19（H6 weighted-Hermitian completion維持）、T=2.0、P=0。',
        '**以下の最良値は今回価格取得した候補域のn=9点費用最良であり、全域最適・母平均費用保証・統計的優位ではない。**','',
        f"signalは{len(data)//5}候補・{len(data)}精度endpoint、うち適格{sum(v['status']=='ELIGIBLE' for v in data)}件。価格は{len(prices)}候補×n9、固定独立確認は{len(confirmed)}候補・{confirmation['trials']}標本。未価格化候補は{len(data)//5-len(prices)}件。",'',
        '## 改善量と精度別最良候補','',
        '前回のT=.8固定q/R/KをT=2.0へ移した対照も今回新規に価格化した。T=.8の価格をT=2.0へ無検証転用していない。',
        '初期格子nomineeは、B3 T=2.0 S2と同じ格子上でpure RTEのsignal/shot・proxyを評価し、価格取得前に選抜したH4 q8/R512/K2、H6 q32/R512/K2。','',
        '|系|ε|取得域最良 q/R/K|G ± MCSE|前回parameter移植対照比|初期格子nominee比|',
        '|---|---:|---|---:|---:|---:|']
    for n in nominees:
        a=lookup[n['system'],n['epsilon'],'TRANSFERRED_T08_PARAMETERS_AT_T20']['G_ratio']
        b=lookup[n['system'],n['epsilon'],'INITIAL_GRID_NOMINEE']['G_ratio']
        fmt=lambda x:'未取得' if x is None else f'{x:.6g}'
        lines.append(f"|{n['system']}|{n['epsilon']:g}|{n['q']}/{n['R_outer']}/{n['K']}|{n['G']:.7g} ± {n['G_SE']:.3g}|{fmt(a)}|{fmt(b)}|")
    for system in ('H4','H6'):
        n=next((v for v in nominees if v['system']==system and v['epsilon']==.0005),None)
        if n is not None:
            ratio=lookup[system,.0005,'TRANSFERRED_T08_PARAMETERS_AT_T20']['G_ratio']
            lines.insert(2,f"{system}・ε=.0005の取得域点最良G={n['G']:.7g}。前回parameterをT=2.0へ移した実測対照の{ratio:.7g}倍（約{1/ratio:.6g}分の1）。")
    lines += ['',
        '探索cohortの最小値を選んだ点比較であり、選抜biasを含み得る。独立確認を実施した場合は、その固定候補の別cohort結果を別表へ保存し、探索値へpoolしない。',
        f'今回の独立確認候補数：{len(confirmed)}。','',
        '## 実装・科学条件の再利用','',
        '固定出発点は`9b83b96b83f62142c9d87858dc24a94db7ac5d61`。前回の一体項Z成分追加、全DF成分、抽出scalar、保存Hamiltonian/state/referenceとfragment順を維持した。',
        '決定論blockはゼロ。一体項の決定論的時間発展を再導入していない。分子からの入力再生成・保存target referenceの再生成は0。signal kernel内部のexact-tail対照列は計算している。',
        'S2、paired-Taylor law（最大偶数次数K、numerator次数K+1）、normalization、共通u floor1e-9・alpha_axis=.025・headroom政策を再利用。',
        '価格は測定付きcosine/sine full Hadamard wrapper、Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、backend/coupling/layout/routing追加指定なし。P=0。',
        '前回correctnessを保持し、T=2.0の正負event・一体項/mixed成分・actual adjoint・qR縮約・normalizationを限定再確認した。各価格候補trial0で独立occupation oracle、両軸Hadamard、compiled作用を検証した。',
        '作用照合は既存Aer0.15.1のCPU double statevectorで実行し、native custom gateは正確な定義展開、global phaseは明示的に適用する。価格compiler/circuitは変更していない。',
        '既存B3の先行compileは固定n9の未着手seedだけでcacheを準備し、fingerprint排他lockで二重compileを防ぐ。標本追加は0で、本体の検査・受理手順は維持した。[先行cache記録](prefetch_status.json)。',
        '末尾5seedは、cache専用処理の実測ピークと実際の残作業数で資源admissionを行い、同じ既存cache関数で先行処理した。[末尾cache記録](prefetch_tail_status.json)。元の実行中SOURCE・controllerを変更していない。',
        'Aer呼出しは既存fingerprintのsingle-thread前提を守る同期実行。thread、OS、共有環境設定の変更はない。自分のprocessのlibrary検索pathだけを指定した。','',
        '## q/R/K探索と境界','',
        'B3 T=2.0 S2格子は各系192構成：q={1,2,4,8,10,16,32,64}、R={4,8,16,32,64,128,256,512}、K={2,4,6}。',
        'pure S2では決定論blockがないため、有限meanとBはs=qRとKだけに依存する。既存evolve_batchのq1/R=s形を再利用し、元の分割q/Rでの結果とも照合した。価格の同一性は仮定せず、価格候補は個別のseed/cohortで取得した。',
        'K2が下限かつ有望だったため隣接K0を追加し、緩い精度の低費用候補を得た。K0は数学的下限で、それより下へは拡張しない。厳しい精度ではbiasとの競合によりK2が有望。',
        'R512上限から固定qで隣接R1024を評価し、局所の中間R=192/224/320/384/448/640/768も評価した。追加全格子は作成していない。',
        '固定qでのR上限拡張の多くは、別の初期q/Rに既にあるs=qRと重なる。初期のs上限32768に対して、nomineeのsは既に内側だった。R448は局所の新しい中間sを加え、H6 K0 s65536が総step上限を拡張した。未価格化の上限probeはproxy評価までで、実際のGの劣位を確定していない。',
        'H6 K0のε=.0005不適格境界はR1024から隣接R2048（s65536）へ拡張して適格化を確認したが、proxyはK2より高く、その点は未価格化とした。',
        'K4/6は初期signal格子に含まれる。K6上限に改善するnomineeはなく、K8拡張は行っていない。qは探索域1..64。q/Rの同s signal等価性を、全compiled価格の等価性やqの全域最適性へ拡張しない。',
        '境界の取得・価格化・未解決状態は[boundary_status.json](boundary_status.json)を参照。未価格化候補の劣位は確定していない。','',
        '## λ_R・B・N・C・Gの競合','',
        '|系|ε|s|K|λ_R|log B|B|Ncos / Nsin|Ccos / Csin|CXcos / CXsin|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for n in nominees:
        lines.append(f"|{n['system']}|{n['epsilon']:g}|{n['s']}|{n['K']}|{n['lambda_R']:.8g}|{n['log_B']:.7g}|{n['B']:.7g}|{n['N_cos']} / {n['N_sin']}|{n['C_cos']:.7g} / {n['C_sin']:.7g}|{n['CX_cos']:.7g} / {n['CX_sin']:.7g}|")
    lines += ['',
        'sを増すとnormalization/shotが下がる一方、compiled回路価格が増える。K0はnormalizationが小さい一方で有限Taylor biasが大きく、εによって適したKとsが変わる。',
        'signal biasはcompile前の有限meanと保存referenceとの差であり、compiled回路の厳密bias保証とは区別する。uとlog B marginは既存の経験的会計。必要shot数は計算値で、量子shotの実測ではない。','',
        '## 保存B3/B2/PFとの比較','',
        '以下は各方式の保存済み価格取得域最良との比較で、公式/q/R/K/prefixは方式ごとに異なる。Full PFと、Full/B0/B0+Iを含む決定論PF envelopeを分ける。','',
        '|系|ε|pure/B3保存最良|pure/B2保存最良|pure/Full PF保存最良|pure/PF envelope|',
        '|---|---:|---:|---:|---:|---:|']
    for n in nominees:
        ratios=[lookup[n['system'],n['epsilon'],role]['G_ratio'] for role in ('SAVED_B3_BEST','SAVED_B2_BEST','SAVED_FULL_PF_BEST','SAVED_PF_ENVELOPE_BEST')]
        lines.append('|'+n['system']+'|'+f"{n['epsilon']:g}"+'|'+'|'.join('未取得' if x is None else f'{x:.6g}' for x in ratios)+'|')
    lines += ['', 'ε=.0005の保存最良との費用内訳。λ_Rは前回full-H/B3の保存値と、保存B2 prefix診断から参照した。PFのRTE weightは0。全精度の対応は[comparisons.csv](comparisons.csv)。','',
        '|系|保存方式|λ_R|log B|Ncos+Nsin|Ccos|G|',
        '|---|---|---:|---:|---:|---:|---:|']
    for system in ('H4','H6'):
        for role in ('SAVED_B3_BEST','SAVED_B2_BEST','SAVED_FULL_PF_BEST','SAVED_PF_ENVELOPE_BEST'):
            v=lookup[system,.0005,role]
            if v['baseline_G'] is not None:
                lines.append(f"|{system}|{role}|{v['baseline_lambda_R']:.8g}|{v['baseline_log_B']:.7g}|{v['baseline_N_cos']+v['baseline_N_sin']}|{v['baseline_C_cos']:.7g}|{v['baseline_G']:.7g}|")
    if confirmed:
        lines += ['', '## 固定H4 2候補の独立確認', '',
            '探索点差約0.66%がMCSEと同程度だったため、H4 q8/K2のR448とR512だけを、取得前に固定した各n=33の別cohortで確認した。主評価はε=.0005。確認後minへの選び直し、探索とのpooling、top-upは0。',
            f"独立確認のG比 R448/R512={confirmation['ratio']:.8g}。差G448−G512={confirmation['difference']:.7g} ± {confirmation['difference_MCSE']:.3g}（MCSE）。",
            f"正規近似95%の有限標本診断区間：{confirmation['normal_95_diagnostic']}。点方向再現：{confirmation['point_direction_reproduced']}。",
            'この区間は希少eventを含む母平均費用の厳密保証ではない。探索点最良候補は確認後に変更していない。H6とK0候補の価格は探索n9のみ。',
            '[確認会計](confirmation_accounting.csv) / [確認要約](confirmation_summary.json) / [独立trial](confirmation_trials.json.gz)。']
    lines += ['',
        '同じq/R/KのB3対照は[comparisons.csv](comparisons.csv)のMATCHED_B3に分離した。保存compiled価格がない場合は未取得とし、signalが存在するだけでGを補完しない。旧B2/B3/PF結果は読取専用。',
        f"最良点と同じq/R/KのB3保存価格がある比較は{sum(c['role']=='MATCHED_B3' and c['baseline_G'] is not None for c in comparisons)}/{len(nominees)}件。方式間の優位性を独立確認した比較は0件。",'',
        '## 限界・技術的中断・成果物','',
        'proxyは前回実測RZ/eventにsと期待event product次数を掛けた選抜用概算であり、compiled価格でも厳密下界でもない。価格未取得域を含むため全域最適性は未確認。',
        'n=9は有限標本。希少eventの保存law確率と実効float-CDF確率、観測数を[rare_event_status.json](rare_event_status.json)へ分離した。未観測条件付き価格・母平均tail費用上界はnullで、0補完していない。',
        '修正前の作用検証adapterはnative crzz未対応で価格受理前に技術停止した。人工検証でglobal-phaseと残留threadの問題も修正した。元SOURCE/STOP/失敗記録を保持し、新SOURCE/cohortへ移した。これらの技術trialを探索n9へ混入していない。',
        '[technical_history.json](technical_history.json)にSOURCE・失敗・受理範囲を記録する。科学的に不一致の結果をPASS扱いしていない。',
        '[RESULTS.csv](RESULTS.csv)は価格取得候補の5精度会計、[best_configurations.csv](best_configurations.csv)は10最良点、[candidate_inventory.json](candidate_inventory.json)とsignal/endpoints JSON.gzは全評価域と未価格化範囲。',
        '[SOURCE_INDEX.json](SOURCE_INDEX.json)はSOURCE/input/candidate/cohort/結果参照、[validation_summary.json](validation_summary.json)は必要な検証記録。新しい論文図表・論文本文・詳細科学レビューは作成していない。',
        'S1 Main Claimと論文構成は変更していない。GitHub公開照合後mandatory STOP。次の作業なし。','']
    (s.PUBLIC/'REPORT.md').write_text('\n'.join(lines))


def publish_index(records,prices,trials,confirmed):
    cfg=s.full.signal.origin_config()
    sources=['scripts/h46_full_h_pure_rte.py','scripts/h46_full_h_t20_opt.py',
        'scripts/h46_full_h_t20_aer.py','scripts/run_h46_full_h_t20_opt.py',
        'scripts/check_h46_t20_action_adapter.py','scripts/report_h46_full_h_t20_opt.py','scripts/check_h46_t20_fused_action.py',
        'scripts/prefetch_h46_full_h_t20.py','scripts/prefetch_h46_full_h_t20_tail.py','scripts/h46_b3_prefetch.py','scripts/h46_b3_durable.py',
        'scripts/h46_long_time_signal.py','scripts/h46_long_time_native.py',
        'scripts/h46_new2_core.py','scripts/h46_new2_native.py','scripts/h46_b3_fast_fingerprint.py',
        'scripts/run_pf_common_run.py','scripts/h46_long_time_report.py','scripts/h246_saved_cost_evaluation.py',
        'scripts/d0_frozen_shot_rule_v1.py','src/trotterlib/h46_m5_plan.py',
        'scripts/h46_m5_contract.py','scripts/h46_m5_tol_native.py',
        'scripts/h46_m5_tol_saved_input.py','scripts/h46_pr_pilot_repair.py',
        'tests/test_h46_full_h_t20_opt.py','tests/test_h46_full_h_t20_aer.py']
    prior=s.read(s.ROOT/'artifacts/resource_applicability/df_full_h_pure_rte_light/2026-11-09/SOURCE_INDEX.json')
    fixed={'scripts/h46_full_h_pure_rte.py':'9b83b96b83f62142c9d87858dc24a94db7ac5d61',
        'scripts/h46_full_h_t20_opt.py':'8ca723320',
        'scripts/h46_full_h_t20_aer.py':'abd0715a2',
        'scripts/run_h46_full_h_t20_opt.py':'93b4f13e0',
        'scripts/prefetch_h46_full_h_t20.py':'4af033c83',
        'scripts/prefetch_h46_full_h_t20_tail.py':'5cfbb5f8a'}
    for folder in (s.B3,s.BASE):
        fixed[str((folder/'accounting.json.gz').relative_to(s.ROOT))]='9b83b96b83f62142c9d87858dc24a94db7ac5d61'
    conservation={}
    for path,commit in fixed.items():
        expected=hashlib.sha256(subprocess.check_output(['git','show',commit+':'+path],cwd=s.ROOT)).hexdigest()
        actual=s.full.signal.sha(s.ROOT/path)
        s.full.require(actual==expected,'SOURCE_OR_BASELINE_MUTATED:'+path)
        conservation[path]=dict(reference_commit=commit,sha256=actual,match=True)
    provider=Path(cfg['provider_root']);seals={v['path']:v['sha256'] for v in cfg['sealed_files']}
    for path,want in prior['provider_files'].items():
        s.full.require(s.full.signal.sha(provider/path)==want==seals[str(provider/path)],'PROVIDER_SOURCE_UNCHANGED')
    for info in cfg['systems'].values():
        for item in info['inputs'].values():s.full.require(s.full.signal.sha(item['path'])==item['sha256'],'SAVED_INPUT_UNCHANGED')
    s.save(s.PUBLIC/'SOURCE_INDEX.json',dict(base_commit='9b83b96b83f62142c9d87858dc24a94db7ac5d61',
        publication_branch='codex/df-full-h-pure-rte-t20-opt-20261109',
        kernel_SOURCE='8ca723320',action_adapter_SOURCE='abd0715a2',
        source_files={f:s.full.signal.sha(s.ROOT/f) for f in sources},
        source_and_baseline_conservation=conservation,
        provider_SOURCE=cfg['runtime']['provider_SOURCE'],provider_files=prior['provider_files'],
        saved_inputs=prior['saved_inputs'],T20_reference_files={k:dict(path=str(s.full.BASE_PRIVATE/(k+'_context.json')),sha256=s.full.signal.sha(s.full.BASE_PRIVATE/(k+'_context.json'))) for k in ('H4','H6')},
        prior_correctness_reference='artifacts/resource_applicability/df_full_h_pure_rte_light/2026-11-09/',
        baseline_results={str((folder/'accounting.json.gz').relative_to(s.ROOT)):s.full.signal.sha(folder/'accounting.json.gz') for folder in (s.B3,s.BASE)},
        evaluation_T=2.0,historical_binding_T_is_input_reference_only=True,
        compiler=prior['compiler'],runtime=dict(prior['runtime'],qiskit_aer='0.15.1',simulation_device='CPU'),
        candidate_inventory='candidate_inventory.json',price_groups={identity:dict(cohort=g['cohort'],n=g['n'],source=g['source_file']) for identity,g in prices.items()},
        confirmation_groups={identity:dict(cohort=g['cohort'],n=g['n'],source=g['source_file']) for identity,g in confirmed.items()},new_reference_calls=0,reference_counter_semantics='saved target reference generation; internal exact-tail control is evaluated',input_generation=0,
        old_SOURCE_results_grants_ledgers_markers_STOPs_modified=False,
        seed_policy=dict(function='scripts/h46_full_h_t20_opt.py:trial_seed',
            namespace='FULL_H_T20_OPT_20261109',keys=['cohort','candidate_id','index','outer','piece'],
            exploration_and_confirmation_separate=True,prior_method_samples_pooled=False,
            finite_PRNG_independence_proved=False),
        private_assets='PRIVATE/cost, PRIVATE/compiled_cache; technical failures retained separately'))


if __name__=='__main__':account()
