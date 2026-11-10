"""Saved-only H8 pilot accounting; no new trajectories or reference calls."""
import argparse
import csv
from fractions import Fraction
import hashlib
import importlib.metadata
import math
from pathlib import Path
import subprocess
import sys
import h8_full_h_pure_rte as s


def write_csv(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(rows)


def main():
    construction=s.read(s.PUBLIC/'validation.json')
    product_check=s.read(s.PUBLIC/'conditional_product_compile_validation.json')
    s.require(product_check['status']=='PASS' and product_check['statistical_samples']==0,'CONDITIONAL_PRODUCT_CHECK')
    signals=s.read(s.PUBLIC/'signals.json');selection=s.read(s.PUBLIC/'selection.json')
    sources=s.read(s.PUBLIC/'execution_sources.json')
    if sources.get('technical_run')=='parallel_v1':
        proof=sources['parallel_validation']
        s.require(s.old.sha(s.ROOT/proof['path'])==proof['sha256']
            and s.read(s.ROOT/proof['path'])['status']=='PASS','PARALLEL_H8_PREFLIGHT')
    for path,want in sources['source_files'].items():
        s.require(s.old.sha(s.ROOT/path)==want,'FIXED_SCIENTIFIC_SOURCE_CHANGED:'+path)
        frozen=subprocess.check_output(['git','show',sources['price_SOURCE']+':'+path],cwd=s.ROOT)
        s.require(frozen==(s.ROOT/path).read_bytes(),'COMMITTED_PRICE_SOURCE:'+path)
    baseline_path=s.ROOT/'artifacts/resource_applicability/track_a_h8_long_time/2026-11-05/confirmation.json'
    baseline=s.read(baseline_path);b2=baseline['PR_summary'];pf=baseline['fixed_PF']
    original=subprocess.check_output(['git','show','eb07156a37b1938ed3aed95c13abbb545466c573:'+str(baseline_path.relative_to(s.ROOT))],cwd=s.ROOT)
    s.require(original==baseline_path.read_bytes(),'ORIGINAL_BASELINE_PRESERVED')
    s.require(baseline['scope']['binding']==signals['reference']['binding'],'SAME_H8_BASELINE_INPUT')
    design_path=s.ROOT/'configs/h8_long_time/stage_c_proposal_20261104.json'
    design=s.read(design_path)
    s.require(design['T']==1.6 and design['epsilon']==.0005
        and baseline['scope']['candidates'][0]['T']==pf['candidate']['T']==1.6,'SAME_H8_BASELINE_T_EPS')
    for role,base in [('PR',b2),('PF',pf)]:
        endpoint=design['comparison'][role]['endpoint']
        s.require((base['fixed_N_cos'],base['fixed_N_sin'])==(endpoint['N_cos'],endpoint['N_sin']),'SAVED_BASELINE_FIXED_N')
    prices=s.read(s.PUBLIC/'prices.json') if (s.PUBLIC/'prices.json').exists() else None
    s.require(prices is not None and prices['all_workers_joined'],'ACQUISITION_MUST_BE_FINISHED')
    ts=s.read(s.PUBLIC/'price_trials.json.gz') if (s.PUBLIC/'price_trials.json.gz').exists() else []
    s.require(len({(t['cohort'],t['candidate_id'],t['index']) for t in ts})==len(ts),'UNIQUE_PRICED_TRIALS')
    s.require(all(t['status']=='PASS' and not t['conditional'] and t['cohort']==s.COHORT for t in ts),'ACCEPTED_EXPLORATION_ONLY')
    s.require(sorted(t['index'] for t in ts)==list(range(9)),'COMPLETE_FIXED_INDICES')
    checked=[]
    for trial in ts:
        folder=s.PRIVATE/'cost'/s.COHORT/trial['candidate_id']/str(trial['index'])
        raw=s.read(folder/'result.json')
        for key,value in raw.items():
            s.require(trial[key]==value,'RAW_TRIAL_AGGREGATE_CONSISTENCY:'+key)
        for check in trial['checks']:
            s.require(math.isfinite(check['error']) and check['error']<check['tolerance']
                and check['phase_alignment'] is False,'ACCEPTED_CHECK_ERROR')
        ep=next(e for e in selection['endpoints'] if e['candidate']['id']==trial['candidate_id'])
        s.require(trial['endpoint_digest']==s.old.digest(ep),'SAME_FIXED_SHOT_ENDPOINT')
        s.require(abs(trial['log_B']-ep['log_B'])<=ep['log_B_margin'],'NATIVE_SIGNAL_NORMALIZATION')
        checked.append(trial['index'])
    ctx=s.read(s.PRIVATE/'contexts/price/full_h_context.json')
    provider=s.read(s.PUBLIC/'provider_sources.json')
    provider_root=Path(s.read(s.CONFIG)['provider_root'])
    for item in provider:
        s.require(item['match'] and s.old.sha(provider_root/item['path'])==item['sha256'],'PROVIDER_SOURCE_CONSERVATION')
    s.require(s.old.sha(s.ROOT/'scripts/h8_compiler_phase_bookkeeping.py')
        ==baseline['scope']['source_files']['scripts/h8_compiler_phase_bookkeeping.py'],'SAME_H8_PHASE_HELPER')
    packages={}
    for name in ('qiskit','numpy','scipy','numba','qiskit-aer','qiskit-aer-gpu'):
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:pass
    results=[];comparisons=[]
    for summary in prices['summaries']:
        row=summary['candidate'];ep=summary['endpoint'];cohort=[t for t in ts if t['candidate_id']==row['id']]
        raw=[ep['N_cos']*t['prices']['cosine']['rz_count']+ep['N_sin']*t['prices']['sine']['rz_count'] for t in cohort]
        s.require(Fraction(sum(raw),len(raw))==Fraction(summary['mean_G_exact']),'EXACT_INTEGER_PAIRED_G')
        mu=Fraction(sum(raw),len(raw));variance=sum((Fraction(g)-mu)**2 for g in raw)/(len(raw)-1)
        se=math.sqrt(float(variance/len(raw)))
        s.require(math.isclose(se,summary['sample_SE_G'],rel_tol=1e-14),'PAIRED_SAMPLE_SE')
        s.require(variance==Fraction(summary['sample_variance_G_exact']),'EXACT_PAIRED_VARIANCE')
        result=dict(method='FULL_H_PURE_RTE',id=row['id'],T=1.6,epsilon=.0005,formula=row['formula'],
            q=row['q'],R_outer=row['R_outer'],K=row['K'],s=row['s'],prefix=0,deterministic_blocks=0,
            lambda_R=construction['lambda_R'],log_B=ep['log_B'],B=math.exp(ep['log_B']),
            N_cos=ep['N_cos'],N_sin=ep['N_sin'],C_cos=float(Fraction(summary['mean_C_cos_exact'])),
            C_sin=float(Fraction(summary['mean_C_sin_exact'])),CX_cos=float(Fraction(summary['mean_CX_cos_exact'])),
            CX_sin=float(Fraction(summary['mean_CX_sin_exact'])),G=float(Fraction(summary['mean_G_exact'])),
            G_exact=summary['mean_G_exact'],G_SE=summary['sample_SE_G'],n=summary['n'],cohort=summary['cohort'],
            observed_one_body_applications=summary['observed_one_body_applications'],status='ACQUIRED',
            population_guarantee=False)
        signal=next(x['record']['signal'] for x in signals['items'] if x['record']['candidate']['id']==row['id'])
        for key in ('corrected','raw','reference','total_signed_bias'):
            for axis in ('real','imag'):result[key+'_'+axis]=signal[key][axis]
        results.append(result)
        for label,base in [('B2_CONFIRMATION',b2),('PF_B0I_SAVED',pf)]:
            value=Fraction(base['mean_G_exact']);comparisons.append(dict(pure_id=row['id'],baseline=label,
                pure_n=summary['n'],pure_cohort=s.COHORT,pure_G_exact=summary['mean_G_exact'],
                baseline_n=base['n'],baseline_cohort=base['cohort'],baseline_G_exact=str(value),
                G_ratio=float(Fraction(summary['mean_G_exact'])/value),same_q_R_K_control=False,
                comparison='FIXED_REPRESENTATIVE_S2_VS_SAVED_M5_POINT',pooled=False,population_superiority_claim=False))
    for label,base in [('B2',b2),('PF_B0I',pf)]:
        row=baseline['scope']['candidates'][0] if label=='B2' else base['candidate']
        baseline_trials=baseline['trials'] if label=='B2' else []
        n=base['n'];nc=base.get('fixed_N_cos',base.get('N_cos'));ns=base.get('fixed_N_sin',base.get('N_sin'))
        # Published paired trajectory fields provide the saved marginal prices.
        cc=(sum(t['prices']['cosine']['rz_count'] for t in baseline_trials)/n if baseline_trials else float(Fraction(base['mean_C_cos_exact'])))
        cs=(sum(t['prices']['sine']['rz_count'] for t in baseline_trials)/n if baseline_trials else float(Fraction(base['mean_C_sin_exact'])))
        saved_endpoint=design['comparison']['PR' if label=='B2' else 'PF']['endpoint']
        results.append(dict(method=label,id=row['id'],T=1.6,epsilon=.0005,formula=row['formula'],q=row['q'],
            prefix=row['prefix'],R_outer=row.get('R_outer'),K=row.get('K'),N_cos=nc,N_sin=ns,
            C_cos=cc,C_sin=cs,log_B=saved_endpoint['log_B'],B=math.exp(saved_endpoint['log_B']),G=float(Fraction(base['mean_G_exact'])),G_exact=base['mean_G_exact'],
            G_SE=base['sample_SE_G'],n=n,cohort=base['cohort'],status='SAVED_BASELINE',population_guarantee=False))
    signal_rows=[]
    for item in signals['items']:
        ep=item['endpoint'];r=ep['candidate'];rec=item['record'];sig=rec['signal']
        v=dict(r,status=ep['status'],lambda_R=construction['lambda_R'],log_B=ep['log_B'],B=math.exp(ep['log_B']),
            N_cos=ep['N_cos'],N_sin=ep['N_sin'],headroom_cos=ep['axes']['real']['headroom'],
            headroom_sin=ep['axes']['imag']['headroom'],u=ep['u'],proxy_G_work=ep['proxy_G_work'],
            proxy_is_compiled_price_or_lower_bound=False)
        for k in ('corrected','raw','reference','total_signed_bias'):
            for axis in ('real','imag'):v[k+'_'+axis]=sig[k][axis]
        signal_rows.append(v)
    write_csv(s.PUBLIC/'SIGNALS.csv',signal_rows);write_csv(s.PUBLIC/'RESULTS.csv',results)
    write_csv(s.PUBLIC/'comparisons.csv',comparisons)
    s.save(s.PUBLIC/'accounting.json',dict(results=results,comparisons=comparisons,pooled=False))
    trial_rows=[]
    for t in ts:
        ep=next(x for x in selection['endpoints'] if x['candidate']['id']==t['candidate_id'])
        trial_rows.append(dict(id=t['candidate_id'],cohort=t['cohort'],index=t['index'],
            RZ_cos=t['prices']['cosine']['rz_count'],RZ_sin=t['prices']['sine']['rz_count'],
            CX_cos=t['prices']['cosine']['cx_count'],CX_sin=t['prices']['sine']['cx_count'],
            N_cos=ep['N_cos'],N_sin=ep['N_sin'],
            G=ep['N_cos']*t['prices']['cosine']['rz_count']+ep['N_sin']*t['prices']['sine']['rz_count']))
    write_csv(s.PUBLIC/'paired_prices.csv',trial_rows)
    public_sources=dict(sources,report_source_file='scripts/report_h8_full_h_pure_rte.py',
        report_source_sha256=s.old.sha(Path(__file__)),base_H8_commit='eb07156a37b1938ed3aed95c13abbb545466c573',
        reused_full_H_commit='9b83b96b83f62142c9d87858dc24a94db7ac5d61',
        reused_Aer_commit='7a8bc7f132e166390c943e6b32c92cc6abe1baf4',
        compiler_phase_source='18ce41942051065fa097644a8a937bd11adba708',
        compiler=signals['reference']['binding']['compiler'],saved_inputs=ctx['saved_inputs'],
        saved_input_binding=signals['reference']['binding'],saved_input_loader_config=str(s.CONFIG),
        candidate_selection='selection.json',selected_candidates=selection['candidates'],
        runtime=dict(python=sys.version.split()[0],packages=packages,simulation_device='CPU'),
        baseline_runtime=baseline['environment'],
        conditional_product_compile=dict(source='8f5d99375',
            code='scripts/check_h8_full_h_product_compile.py',
            sha256=s.old.sha(s.ROOT/'scripts/check_h8_full_h_product_compile.py'),
            result='conditional_product_compile_validation.json',statistical_samples=0),
        reference=dict(path=str(s.SIGNAL_FILE.relative_to(s.ROOT)),sha256=s.old.sha(s.SIGNAL_FILE),T=1.6),
        baseline=dict(path=str(baseline_path.relative_to(s.ROOT)),sha256=s.old.sha(baseline_path)),
        baseline_design=dict(path=str(design_path.relative_to(s.ROOT)),sha256=s.old.sha(design_path)),
        provider_SOURCE=s.read(s.CONFIG)['provider_source'],
        provider_files=provider,
        saved_input_regeneration=0,new_target_reference=0,private_assets=str(s.PRIVATE.relative_to(s.ROOT)),
        private_trial_root=str((s.PRIVATE/'cost'/s.COHORT/selection['candidates'][0]['id']).relative_to(s.ROOT)),
        private_compiler_cache=str((s.PRIVATE/'compiled_cache').relative_to(s.ROOT)),
        public_trial_records='price_trials.json.gz',
        reuse_guide='docs/parallel_acceleration_reuse.md',
        seed_policy=dict(namespace='H8_FULL_H_LIGHT_20261110',keys=['cohort','candidate_id','index','outer','piece'],
            finite_PRNG_independence_proved=False),old_SOURCE_raw_results_grant_ledger_marker_STOP_modified=False)
    for x in s.read(s.CONFIG)['inputs']:
        if x['role'] in {v['role'] for v in ctx['saved_inputs']}:
            s.require(s.old.sha(x['path'])==x['sha256'],'SAVED_INPUT_CONSERVATION')
    s.save(s.PUBLIC/'SOURCE_INDEX.json',public_sources)
    checks=[c for t in ts for c in t['checks']]
    validation=dict(status='PASS' if not prices['errors'] and len(ts)==prices['planned'] else 'PARTIAL_OR_UNACQUIRED',construction='validation.json',signal_vector='signal_vector_validation.json',
        artificial_tests=10,priced_candidates=len(prices['summaries']),accepted_price_trials=len(ts),planned=prices['planned'],
        acquisition_status=prices['status'],acquisition_errors=prices['errors'],all_workers_joined=True,
        maximum_native_error=max((c['error'] for c in checks if not c['name'].startswith('COMPILED')),default=None),
        maximum_compiled_error=max((c['error'] for c in checks if c['name'].startswith('COMPILED')),default=None),
        SOURCE_input_conservation=True,exact_integer_G_check=True,pooled=False,new_target_reference=0)
    validation['conditional_product_compile']='PASS_EXCLUDED_FROM_PRICE_COHORT'
    validation['full_vector_checked_trial_indices']=[t['index'] for t in ts if len(t['checks'])==7]
    validation['all_price_trial_indices']=sorted(t['index'] for t in ts)
    validation.update(raw_trial_consistency=True,paired_variance_and_SE=True,
        fixed_endpoint_and_native_normalization=True,committed_price_SOURCE_bytes=True,
        saved_baseline_bytes_unchanged=True)
    s.save(s.PUBLIC/'validation_summary.json',validation)
    report(results,comparisons,construction,signals,prices,validation)
    print(validation)


def report(results,comparisons,construction,signals,prices,validation):
    pure=[r for r in results if r['method']=='FULL_H_PURE_RTE']
    lines=['# H8 DF全Hamiltonian純RTE：T=1.6小規模検証','',
        'Linear H8、隣接1.00 Å、STO-3G、保存tol-only1e-8 DF28、16 system qubit・spin sector4900、T=1.6、ε=.0005、P=0。','',
        f"Full-H RTE構築PASS。一体項16 Z成分と全28 DF fragmentをsamplingへ含め、決定論blockは0。λ_R={construction['lambda_R']:.12g}。Hamiltonian再構成最大誤差={max(construction['hamiltonian_errors']):.3g}。",'',
        f"signal評価は{len(signals['items'])}候補、適格{sum(x['endpoint']['status']=='ELIGIBLE' for x in signals['items'])}候補。価格取得前にS2 q32/R704/K2の1候補を固定し、取得済み価格は{len(pure)}候補・{prices['accepted']}/{prices['planned']} trajectory。広域最適化・境界の反復拡張・新独立確認cohortは行っていない。",'',
        f"固定代表のG={pure[0]['G']:.9g} RZ、記述的SE={pure[0]['G_SE']:.7g}。保存B2の{comparisons[0]['G_ratio']:.6g}倍、保存PFの{comparisons[1]['G_ratio']:.6g}倍だった。構築と適格性は確認できたが、この代表の点費用は保存比較より大きい。",'',
        '## 価格と保存B2/PF比較','',
        '|方式|構成|n / cohort|G（RZ）|記述的SE|',
        '|---|---|---|---:|---:|']
    for r in results:
        config=f"{r['formula']} p{r['prefix']} q{r['q']}"
        if r.get('R_outer') is not None:config+=f" R{r['R_outer']} K{r['K']}"
        se='—' if r['G_SE'] is None else f"{r['G_SE']:.7g}"
        lines.append(f"|{r['method']}|{config}|{r['n']} / {r['cohort']}|{r['G']:.9g}|{se}|")
    for r in comparisons:lines.append(f"\nFull-H pure RTE / {r['baseline']} = **{r['G_ratio']:.8g}**。")
    for r in pure:
        lines.extend(['',f"代表のB={r['B']:.10g}、Ncos={r['N_cos']:,}、Nsin={r['N_sin']:,}。平均compiled RZはcos={r['C_cos']:.9g} / sin={r['C_sin']:.9g}、平均CXはcos={r['CX_cos']:.9g} / sin={r['CX_sin']:.9g}。実観測の一体項applicationは{r['observed_one_body_applications']:,}件。"])
    for r in pure:
        for b in [x for x in results if x['method']!='FULL_H_PURE_RTE']:
            nr=(r['N_cos']+r['N_sin'])/(b['N_cos']+b['N_sin']);cr=r['C_cos']/b['C_cos']
            lines.append(f"\n対{b['method']}：shot合計比={nr:.7g}、平均RZ/回路比={cr:.7g}。今回の両軸価格が同じなので積がG比に一致する。保存log B={b['log_B']:.10g}、B={b['B']:.10g}。大きなG差には1回路価格の増加が強く寄与した。")
    rarity=prices['summaries'][0]['rarity'];orders=rarity['observed_order_counts']
    lines.extend(['',f"実観測event次数はorder0={orders.get('0',0):,}、order2={orders.get('2',0):,}。保存lawによる1 trajectory内のorder2出現確率は{rarity['probability_at_least_one_order_or_higher']['2']:.9g}、n9で未観測となる確率は{rarity['probability_unobserved_in_n']['2']:.9g}。order2は今回は観測されたが、2 eventだけから条件付き母平均価格を保証しない。未観測categoryがあれば価格はnullとし、0補完しない。"])
    lines+=['','固定S2代表と保存m5 B2/PFの点比較であり、同一q/R/Kによる因果比較ではない。cohortをpoolせず、全域最適性・統計的な方式間優位・母平均費用保証を主張しない。','',
        '## Signal・normalization・必要shot','',
        '|q/R/K|適格性|bias cos / sin|log B|Ncos / Nsin|',
        '|---|---|---:|---:|---:|']
    for item in signals['items']:
        e=item['endpoint'];r=e['candidate'];bias=item['record']['signal']['total_signed_bias']
        lines.append(f"|{r['q']}/{r['R_outer']}/{r['K']}|{e['status']}|{bias['real']:.5g} / {bias['imag']:.5g}|{e['log_B']:.9g}|{e['N_cos']} / {e['N_sin']}|")
    lines+=['','保存T=1.6 referenceと入力bindingを照合し、T=.8等のreferenceは使っていない。保存stateの固有残差を測り、有限Taylor多項式の固有state作用を計算した。代表全22,528 stepのsparse実ベクトルrecurrenceとの差は[signal_vector_validation.json](signal_vector_validation.json)。4900-square密行列・full-space unitaryは作っていない。',
        '共通u floor1e-9、alpha_axis=.025、headroom・shot政策は既存のqualify_recordを再利用。Bは有限K lawに基づき、raw×Bとcorrectedのclosure、native lawとのnormalizationを検証する。必要Nは計算値で、量子shotの実測ではない。Signal biasはcompile前の有限meanの差であり、compiled回路の厳密bias保証とは区別する。','',
        '## 回路・correctness・再利用','',
        'H8保存loader、Gaussian操作、fragment順0..27を再利用。一体項RTE追加はH4/H6 prepare_full_hへ保存済みbaseを渡す最小拡張。新SCF/DF/state・Gaussian circuit生成は0。scalarは一度だけ適用する。',
        '正負時間・actual adjointとone-body/mixed eventを確認。独立event演算は既存saved-basis algebraと同じ局所行列作用をベクトル化したもので、circuit loweringとは別経路だが独立したorbital分解ではない。',
        'Signal/referenceは保存spin sector4900を使用し、event/native/compiled検査はfull register上で行う。ランダムeventの中間状態をspin sectorへ投影していない。',
        '価格はmeasurement込みfull cosine/sine Hadamard wrapper、Qiskit1.3.0/rz,sx,x,cx/opt1/seed17、backend/topology指定なし。H8既存の小gate operator由来のglobal-phase会計を再利用し、statevectorから位相をfitしない。RZ/CX gate数は変更しない。',
        '現行Python3.12.3、保存Stage CはPython3.11.0rc1。NumPy1.26.4・Qiskit1.3.0とcompiler identity・保存入力は一致する。環境全体のbitwise一致や旧価格の再compileは主張しない。実行環境のmetadataはSOURCE_INDEX.json。',
        '作用照合は検証済みAer CPU doubleを使用。parallel_v1では価格用回路を維持し、検証用に限って最大3量子ビットgateの厳密Operatorをcall-local cacheし、実測で選んだ32 threadsを使用する。full-vectorの7検査は代表index0のみ。残り8価格trialでは回路identity・normalization・paired event等を確認した。全9のfull-vector個別検査を行った意味ではない。代表7検査の誤差は[validation_summary.json](validation_summary.json)、位相会計は[price_trials.json.gz](price_trials.json.gz)。主担当による実行・技術確認で、別担当者の科学レビューではない。','',
        '旧逐次workerはHIGH_CONTROL検査の途中で、承認された高速化のため同一標本を使うparallel_v1へ移行した。旧SOURCE・未完了出力を保全し、partial標本を集計しない。代表eventファイルのbyte一致は[representative_event_identity.json](representative_event_identity.json)。人工回路の約5.2倍は本番全体の短縮率を保証しない。',
        '固定代表のeventはorder0のみ。K2 productで加わるCZは、元helperの小operator検査を使って追加位相0を確認し、別processの診断許可集合へCZを加えた。固定候補と同じτの条件付き単event full wrapperも両軸compileし、[補助検証](conditional_product_compile_validation.json)でPASS。主実行SOURCE・元helperファイルは変更せず、補助をn9や条件付き母平均費用の標本へ混入していない。','',
        '## H4/H6との対応・限界・次段判断','',
        'H4/H6でも、λ_R増加でnormalizationを抑えるためのstep数が増え、shot削減とcompiled回路長が競合した。H8でもR576→704→832でlog BとNは減少する。H8はλ_R=66.0090で、H4の21.0279・H6の41.2279より大きい。H4/H6 T=2.0とは時間も異なるため、その数値を同一条件の因果比較へ読み替えない。',
        '価格候補選択はevent長×期待product次数×Nの未較正work proxy。compiled RZ価格でも厳密下界でもなく、未価格候補の劣位は未確定。今回のS2代表についての実装可能性と点費用が成果であり、最適pure RTEの優劣ではない。',
        'Event次数の観測数、保存law確率、未観測条件付き価格unknown/null・母平均tail費用上界nullは[prices.json](prices.json)。追加rare sampling・top-upは0。小さなSEだけで母平均を保証しない。',
        'この固定代表は保存B2/PFを約85～113倍上回った。今回は小規模検証で一区切りとするのが適切と判断する。追加研究を検討するなら、shot削減に加えてGaussian/basis作用を含む1回路価格を大幅に減らせる具体策があるかを先に考える価値がある。未価格候補の優劣・q/R/K全域最適性は未確認であり、広域探索は自動開始せず、次の研究範囲はユーザーへ戻す。',
        '最初の索引tuple形式による技術的IndexErrorは再構成検査前に停止し、SOURCE c6523c737と停止記録を保持した。修正後の別SOURCE・fresh出力でPASSし、価格標本へ混入していない。[technical_history.json](technical_history.json)、[元の技術STOP](../validate_STOP.json)。','',
        '[H4/H6 T=2.0保存報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7a8bc7f132e166390c943e6b32c92cc6abe1baf4/artifacts/resource_applicability/df_full_h_pure_rte_t20_opt/2026-11-09/REPORT.md) / [H8 Stage C保存報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eb07156a37b1938ed3aed95c13abbb545466c573/docs/research/h8_stage_c_execution_20261105/REPORT.md)。',
        '[RESULTS.csv](RESULTS.csv) / [SIGNALS.csv](SIGNALS.csv) / [paired_prices.csv](paired_prices.csv) / [SOURCE_INDEX.json](SOURCE_INDEX.json)。',
        '旧SOURCE・raw/QPY・結果・grant/ledger/marker・mandatory STOPは変更していない。S1 Main Claim・論文構成は変更せず、詳細科学レビュー・新しい広域探索・別分子・T変更は開始していない。公開remote SHA・主要成果物照合後mandatory STOP。',
        '次の担当：ユーザーが次の研究範囲を判断する。Codexによる追加計算は開始しない。','']
    (s.PUBLIC/'REPORT.md').write_text('\n'.join(lines))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--technical-run',choices=['original','parallel_v1'],default='original')
    args=parser.parse_args()
    if args.technical_run=='parallel_v1':
        s.PUBLIC=s.PUBLIC/'parallel_v1';s.PRIVATE=s.PRIVATE/'parallel_v1'
    main()
