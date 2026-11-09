"""Report the two frozen full-H points against compatible saved results only."""
from fractions import Fraction
from pathlib import Path
from decimal import Decimal, localcontext
import gzip
import json
import math
import subprocess
import h46_full_h_pure_rte as study


def winner(rows):
    rows=[r for r in rows if r.get('G') is not None and r['status']=='ELIGIBLE']
    return min(rows,key=lambda r:Fraction(r.get('G_exact') or str(r['G']))) if rows else None


def main():
    study.setup()
    from h46_long_time_report import cost_for, write_csv
    from h46_long_time_native import cdf_law
    p=study.PUBLIC
    rows=study.read(p/'selection.json')['candidates']
    b3=study.read(study.B3/'accounting.json.gz')
    base=study.read(study.BASE/'accounting.json.gz')
    b3pro=study.read(study.B3/'scientific_provenance.json')
    basepro=study.read(study.BASE/'scientific_provenance.json')
    results=[];comparisons=[];summaries={};trials_all=[];rare=[]
    for row in rows:
        s=row['system'];info=study.read(study.PRIVATE/(s+'_context.json'))
        require=study.require
        require(info['binding']==b3pro['inputs'][s]['binding']==basepro['inputs'][s]['binding'],'BASELINE_INPUT_BINDINGS')
        require(b3pro['compiler']['hash']==basepro['compiler']['hash']==info['binding']['compiler'],'BASELINE_COMPILER')
        require(study.read(p/(s+'_validation.json'))['status']=='PASS','CORRECTNESS')
        sig=study.read(p/(s+'_signal.json'))['signal']
        pricepath=p/(s+'_costs.json')
        costs=study.read(pricepath) if pricepath.exists() else None
        endpoints=study.read(p/(s+'_endpoints.json'))
        if costs:
            trials=[study.read(f) for f in costs['trial_files']]
            require(len(trials)==9 and all(t['status']=='PASS' and not t['conditional'] for t in trials),'NINE_UNCONDITIONAL_TRIALS')
            require(max(abs(t['log_B']-sig['log_B']) for t in trials)<1e-9,'SAME_NORMALIZATION')
            for t in trials:
                require(t['candidate']==row and t['scalar']==info['scalar'],'SAME_CANDIDATE_SCALAR')
                trials_all.append(t)
            law=trials[0]['laws']
            for order in range(2,row['K']+1,2):
                with localcontext() as ctx:
                    ctx.prec=80
                    ideal_no=Decimal(1);effective_no=Decimal(1)
                    for l in law:
                        chance=sum(Decimal.from_float(float(prob)) for k,prob in zip(l['orders'],l['probabilities']) if k>=order)
                        effective=cdf_law(l['probabilities'])['probabilities']
                        bit_chance=sum(Decimal.from_float(float(prob)) for k,prob in zip(l['orders'],effective) if k>=order)
                        ideal_no *= (1-chance)**l['steps']
                        effective_no *= (1-bit_chance)**l['steps']
                    rare.append(dict(system=s,order_or_higher=order,
                        saved_law_probability_per_trajectory=str(1-ideal_no),
                        effective_float_cdf_probability_per_trajectory=str(1-effective_no),
                        probability_none_in_n9=str(ideal_no**9),
                        observed_events_exact_order=costs['rarity']['observed_order_counts'].get(str(order),0),
                        conditional_compiled_cost=None,population_tail_cost_bound=None,
                        unobserved_price_imputed=False))
        for ep in endpoints:
            pure=dict(row,epsilon=ep['epsilon'],comparison_role='FULL_H_FIXED',status=ep['status'],
                lambda_R=info['lambda_R'],log_B=sig['log_B'],B=math.exp(sig['log_B']),
                N_cos=ep['N_cos'],N_sin=ep['N_sin'],u=ep['u'],log_B_margin=ep['log_B_margin'],
                headroom_cos=ep['axes']['real']['headroom'],headroom_sin=ep['axes']['imag']['headroom'],
                C_cos=None,C_sin=None,CX_cos=None,CX_sin=None,G=None,G_SE=None,n=0)
            for key,label in [('reference','reference'),('corrected','corrected'),('raw','raw'),('total_signed_bias','bias')]:
                pure[label+'_cos']=sig[key]['real'];pure[label+'_sin']=sig[key]['imag']
            if costs and ep['status']=='ELIGIBLE':
                pure.update(cost_for(ep,costs))
                pure.update(CX_cos=costs['mean_metrics']['cosine']['cx_count'],CX_sin=costs['mean_metrics']['sine']['cx_count'])
                # Direct integer per-trial weighted sum, distinct from mean formula.
                direct=Fraction(sum(ep['N_cos']*t['prices']['cosine']['rz_count']+
                    ep['N_sin']*t['prices']['sine']['rz_count'] for t in trials),9)
                require(direct==Fraction(pure['G_exact']),'G_INTEGER_CROSSCHECK')
            bd=[v for v in b3 if v['system']==s and v['T']==.8 and v['epsilon']==ep['epsilon']]
            old=[v for v in base if v['system']==s and v['T']==.8 and v['epsilon']==ep['epsilon']]
            matched=next(v for v in bd if v['id']==row['saved_B3_id'])
            for a in ('cos','sin'):
                require(abs(matched['reference_'+a]-pure['reference_'+a])<1e-10,'BASELINE_REFERENCE')
            refs={'MATCHED_B3':matched,'SAVED_B3_BEST':winner(bd),
                'SAVED_B2_BEST':winner([v for v in old if v['method']=='PR']),
                'SAVED_FULL_PF_BEST':winner([v for v in old if v['method']=='FULL']),
                'SAVED_PF_ENVELOPE_BEST':winner([v for v in old if v['method']!='PR'])}
            for role,value in refs.items():
                require(value is not None,'SAVED_COMPARATOR')
                if role=='MATCHED_B3':
                    require(all(value[k]==row[k] for k in ('q','R_outer','K','formula','T')),'MATCHED_Q_R_K')
                for a in ('cos','sin'):
                    require(abs(value['reference_'+a]-pure['reference_'+a])<1e-10,'COMPARATOR_REFERENCE')
                copy=dict(value,comparison_role=role,source='B3/accounting.json.gz' if 'B3' in role else 'B2_PF/accounting.json.gz')
                copy['family']='B3' if 'B3' in role else 'B2' if value['method']=='PR' else value['method']
                copy.setdefault('B',math.exp(value['log_B']))
                if 'B3' in role:copy['lambda_R']=info['lambda_R']-info['one_body_lambda_R']
                results.append(copy)
                ratio=None if pure['G'] is None or value.get('G') is None else pure['G']/value['G']
                pure['G_ratio_to_'+role]=ratio
                comparisons.append(dict(system=s,T=.8,epsilon=ep['epsilon'],role=role,
                    full_H_id=row['id'],baseline_id=value['id'],full_H_G=pure['G'],baseline_G=value['G'],
                    G_ratio=ratio,baseline_formula=value['formula'],baseline_q=value['q'],
                    baseline_R_outer=value.get('R_outer'),baseline_K=value.get('K'),baseline_prefix=value['prefix'],
                    baseline_n=value.get('n'),baseline_cohort=value.get('cohort'),
                    interpretation='same_q_R_K_control' if role=='MATCHED_B3' else 'fixed_full_H_vs_saved_obtained_best',
                    statistical_superiority_claim=False))
            results.append(pure)
        summaries[s]=dict(info,signal=sig,costs=costs)
    write_csv(p/'RESULTS.csv',results)
    write_csv(p/'comparisons.csv',comparisons)
    study.save(p/'price_trials.json',trials_all)
    study.save(p/'rare_event_status.json',rare)
    study.save(p/'input_and_validation_summary.json',{s:{k:v for k,v in info.items() if k not in ('signal','costs')} for s,info in summaries.items()})
    write_report(p,results,comparisons,summaries,rare)
    sources=['scripts/h46_full_h_pure_rte.py','tests/test_h46_full_h_pure_rte.py',
        'scripts/h46_b3.py','scripts/h46_long_time_signal.py','scripts/h46_long_time_native.py',
        'scripts/h46_new2_core.py','scripts/h46_new2_native.py','scripts/h46_m5_tol_native.py',
        'scripts/h46_m5_tol_saved_input.py','scripts/h46_pr_pilot_repair.py','scripts/h46_b3_fast_fingerprint.py',
        'scripts/run_pf_common_run.py','scripts/d0_frozen_shot_rule_v1.py','scripts/h246_saved_cost_evaluation.py',
        'scripts/h46_long_time_report.py','scripts/h46_m5_contract.py',
        'src/trotterlib/h46_m5_plan.py','scripts/report_h46_full_h_pure_rte.py']
    cfg=study.signal.origin_config()
    provider=Path(cfg['provider_root'])
    provider_files=['src/trotterlib/df_rte_tail.py','src/trotterlib/df_rte_circuit.py',
        'src/trotterlib/df_rte_qiskit.py','src/trotterlib/df_partial_s2.py','src/trotterlib/rte.py',
        'src/trotterlib/rte_compiled_cost.py','src/trotterlib/df_hamiltonian.py',
        'src/trotterlib/df_trotter/decompose.py','src/trotterlib/df_trotter/ops.py',
        'src/trotterlib/df_trotter/circuit.py',
        'src/trottertracks/resource_applicability/ax2b_h6_saved_completion_loader_v1.py',
        'src/trottertracks/resource_applicability/ax2b_native_df_v5.py']
    sealed={v['path']:v['sha256'] for v in cfg['sealed_files']}
    for name in provider_files:
        actual=study.signal.sha(provider/name)
        study.require(actual==sealed[str(provider/name)],'SAVED_PROVIDER_SOURCE:'+name)
    for info in cfg['systems'].values():
        for item in info['inputs'].values():
            study.require(study.signal.sha(item['path'])==item['sha256'],'END_INPUT_SHA')
    execution_file=subprocess.check_output(['git','show','ac33da5f0cb038a780a447746a38e478bb0cec61:scripts/h46_full_h_pure_rte.py'],cwd=study.ROOT)
    study.require(execution_file==(study.ROOT/'scripts/h46_full_h_pure_rte.py').read_bytes(),'EXECUTION_SOURCE_UNCHANGED')
    study.save(p/'SOURCE_INDEX.json',dict(
        baseline_commit='ebe9e2aa2c83cd71a6da0a885b78f32274bfb6ed',
        execution_SOURCE='ac33da5f0cb038a780a447746a38e478bb0cec61',
        provider_SOURCE=cfg['runtime']['provider_SOURCE'],
        source_files={s:study.signal.sha(study.ROOT/s) for s in sources},
        provider_files={s:study.signal.sha(provider/s) for s in provider_files},
        saved_inputs={s:dict(binding=cfg['systems'][s]['binding'],inputs=cfg['systems'][s]['inputs'],
            reference_root=cfg['systems'][s]['reference_root'],
            reference_sha256=summaries[s]['reference_file_sha256']) for s in summaries},
        saved_result_files={str(f.relative_to(study.ROOT)):study.signal.sha(f) for f in
            [study.B3/'accounting.json.gz',study.B3/'best_configurations.json',study.B3/'scientific_provenance.json',
             study.BASE/'accounting.json.gz',study.BASE/'scientific_provenance.json']},
        selection_sha256=study.signal.sha(p/'selection.json'),
        result_files={f.name:study.signal.sha(f) for f in p.iterdir() if f.is_file() and f.name in
            ('RESULTS.csv','comparisons.csv','price_trials.json','rare_event_status.json',
             'H4_signal.json','H6_signal.json','H4_validation.json','H6_validation.json','H4_costs.json','H6_costs.json',
             'H4_endpoints.json','H6_endpoints.json','input_and_validation_summary.json')},
        private_assets='PRIVATE/: event records, deterministic seed-derived trajectories and QPY/compiler cache retained locally',
        runtime=dict(python='3.12.3',numpy='1.26.4',scipy='1.14.1',qiskit='1.3.0'),
        compiler=b3pro['compiler'],P=0,full_hadamard_wrapper=True,new_reference_calls=0,input_regeneration=0,
        scope='two fixed T=0.8 S2 candidates; n9 each; five epsilon; no new search/confirmation',
        old_SOURCE_results_grants_ledgers_markers_STOPs_modified=False))
    print(json.dumps(dict(rows=len(results),comparisons=len(comparisons),price_trials=len(trials_all))))


def write_report(p,results,comparisons,summaries,rare):
    pure=[r for r in results if r['comparison_role']=='FULL_H_FIXED']
    lines=['# DF全Hamiltonian純RTE：固定代表2条件の軽量追加検証', '',
        'linear H4/H6、1.00 Å、STO-3G、保存tol-only DF10/19（H6の保存weighted-Hermitian completionを維持）、T=0.8、P=0。',
        '保存Hamiltonian・state・reference・DF fragment順を再利用し、入力生成と新reference計算は0。',
        '**両系とも全Hamiltonian純RTEの構築・一体項sampling・native/compiled full-wrapper整合を確認した。以下は固定候補各n=9の点費用と経験的shot会計である。全域最適性、母平均費用保証、統計的優位を示さない。**','',
        '## 実装と再利用','',
        '既存B3のL_D=0は一体項を決定論側に残すため、そのままpure RTEとは呼べない。新adapterは既存B3 preparationの一体項固有値ε_jとGaussian基底を再利用し、',
        '`H_one = (sum ε_j)/2 I - sum ε_j/2 U Z_j U†` の非scalar Z成分を既存DF tailの末尾へ追加した。',
        '一体項scalarを既存抽出identityへ合算し、元Hamiltonian constantとともに一回適用する。決定論blockは空。二体DF fragment順・既存component順・係数閾値0を維持した。',
        'この一体項symbolic追加と独立occupation oracleの対応だけが必要な科学実装拡張である。既存APIは一体項を必ず決定論blockに置くため、新adapterが必要だった。','',
        '`h46_b3.py`の構造、`h46_new2_native.prepare/OccupationOracle`、`h46_new2_core`のoccupation/spectral/polynomial、',
        '`h46_long_time_signal.evolve_batch/qualify`、既存DF RTE sampler、native compose/cost_trial、compiler cache、',
        '`h46_long_time_report.cost_for`を再利用した。高速fingerprintはB3でbyte一致検証済みのwriterを再利用。',
        'Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、backend/coupling/layout/routing追加指定なし。cosine/sineとも測定付きfull Hadamard wrapper。P=0なのでstate preparationは価格外。','',
        '## 事前固定候補と正しさ','',
        '[selection.json](selection.json)はpure RTEの出力を見る前に保存B3のT=0.8・S2・ε=0.0005探索nomineeから固定した。H4 q16/R16/K4（s256）、H6 q32/R32/K6（s1024）。',
        'Kは最大偶数event次数であり、paired Taylor numeratorはK+1次。各精度で同じsignalと価格標本を再利用し、回路再作成・再最適化は0。','',
        '|系|λ_R|一体項λ|一体項sampling質量|一体項成分数|全H再構成最大差|決定論block|実標本の一体項application|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for s,info in summaries.items():
        validation=study.read(p/(s+'_validation.json'))
        lines.append(f"|{s}|{info['lambda_R']:.9g}|{info['one_body_lambda_R']:.9g}|{info['one_body_probability_mass']:.6%}|{info['one_body_components']}|{max(validation['full_h_action_errors']):.3g}|0|{info['costs']['observed_one_body_applications'] if info['costs'] else '未取得'}|")
    lines += ['',
        '再構成は保存DF Hamiltonianの独立linear operatorとも照合し、全粒子数sectorでは保存stateと人工複素vector2本で全component作用を照合した。',
        '一体項自身の再構成、成分確率p=|a|/λ、一体項を含む正負時間・全採用偶数次数event、low/high control、actual adjointを検証した。',
        '人工2成分の全event列挙でも有限K mean×normalizationがTaylor polynomialに一致した。人工・必要回帰14件PASS。',
        '実候補ではraw/corrected vector closure、signal/native log B、scalarを照合し、各系trial0で独立occupation oracle、cosine/sine Hadamard期待値、compiled作用vectorを検証した。',
        '詳細は[H4検証](H4_validation.json)、[H6検証](H6_validation.json)、[価格trialとwrapper検査](price_trials.json)、[人工テスト](technical_tests.json)。条件付き正しさfixtureは価格標本へ混入していない。','',
        '## 固定候補のsignal・費用','',
        'B=exp(log B)。biasは保存referenceとの差の観測値。共通u floor=1e-9、alpha_axis=.025、log B marginとheadroom/u政策を維持し、厳密bias保証とは区別する。',
        'shot数は必要量子shotの計算値であり、実際の量子測定shotを実行した結果ではない。Cはn=9のcompiled RZ数平均、G=Ncos Ccos+Nsin Csin。',
        'biasはcompile前の有限meanに対する数値観測であり、compiled回路の厳密bias保証ではない。代表compiled vector差の許容値は既存条件の1e-7。','',
        '|系|log B|B|max軸bias|Ccos / Csin|CXcos / CXsin|',
        '|---|---:|---:|---:|---:|---:|']
    for s in summaries:
        r=next(v for v in pure if v['system']==s)
        lines.append(f"|{s}|{r['log_B']:.9g}|{r['B']:.9g}|{max(abs(r['bias_cos']),abs(r['bias_sin'])):.3g}|{r['C_cos']:.6g} / {r['C_sin']:.6g}|{r['CX_cos']:.6g} / {r['CX_sin']:.6g}|")
    lines += ['','|系|ε|適格性|Ncos / Nsin|G ± MCSE|','|---|---:|---|---:|---:|']
    for r in pure:
        lines.append(f"|{r['system']}|{r['epsilon']:g}|{r['status']}|{r['N_cos']} / {r['N_sin']}|{r['G']:.7g} ± {r['G_SE']:.3g}|")
    lines += ['','## 保存結果との比較','',
        '同じq/R/KのB3は制御比較。他の列は固定pure RTE候補と各方式の保存済み取得域最良（公式・q/R/K・prefixは方式ごとに異なる）の比較である。',
        'B2は一体項＋一部DFを保持、B3は一体項だけ保持。PF envelopeはFull/B0/B0+Iを含む価格取得済み適格候補から選ぶ。',
        '比較baselineは保存探索cohort、pure RTEは今回n=9。独立確認cohortは新たに取得せず、保存確認値とのpoolingも行っていない。','',
        '|系|ε|pure/B3同qRK|pure/B3保存最良|pure/B2保存最良|pure/Full PF保存最良|pure/PF envelope|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in pure:
        ratios=[r['G_ratio_to_'+key] for key in ('MATCHED_B3','SAVED_B3_BEST','SAVED_B2_BEST','SAVED_FULL_PF_BEST','SAVED_PF_ENVELOPE_BEST')]
        lines.append('|'+r['system']+'|'+f"{r['epsilon']:g}"+'|'+ '|'.join(f'{x:.5g}' if x is not None else '未取得' for x in ratios)+'|')
    lines += ['',
        'pure RTEでは決定論側との分割が消えるため、観測biasは両系とも丸め誤差水準まで低下した。一方、一体項をsamplingへ含めることでλ_RとBが増え、shot数が増加した。',
        'ここでの正味傾向は上表の固定点比較に限る。高費用でも「最適化pure RTEが不利」とは結論せず、低費用の比較があっても一般的優位とは扱わない。','',
        '## 希少event・未確認域・次の科学的価値','',
        '[rare_event_status.json](rare_event_status.json)は保存lawの確率と実効float CDFの確率、n=9で未観測となる確率、観測数を分離している。',
        '条件付き価格、未観測event費用、母平均tail費用上界はnull。未観測費用を0として補完していない。条件付きnative正しさPASSは条件付きcompiled価格の取得を意味しない。',
        'n=9のMCSEは有限標本の診断であり、希少eventを含む母平均保証や統計的順位保証ではない。すべての個別component列の条件付き価格は未取得。','',
        '実装が成立し、分割biasの除去とnormalization増大のトレードオフが見えたため、追加探索を検討する科学的価値はある。',
        'ただし、この2点から広域探索の費用対効果や優位性は判断できず、広域探索を優先する根拠は弱い。特に緩い精度でも最厳精度由来の同一候補を使ったため、費用比は精度別最適化の比較ではない。今回取得していないsやλ低減を扱う研究は新しい範囲となる。広域探索、R/K再最適化、別PF次数、40条件追加、詳細科学レビューは開始していない。',
        'S1論文Main Claimは変更していない。必要なテストとGitHub公開照合の完了後mandatory STOP。','',
        '## 成果物と再現','',
        '[RESULTS.csv](RESULTS.csv)はpure10行＋比較baseline50行、[comparisons.csv](comparisons.csv)は50比較のID・q/R/K・公式・cohortと点比率。',
        '[SOURCE_INDEX.json](SOURCE_INDEX.json)は実行SOURCE、再利用SOURCE、保存入力hash・reference、保存結果参照と出力hash。',
        '新科学実装は`scripts/h46_full_h_pure_rte.py`、人工テストは`tests/test_h46_full_h_pure_rte.py`、保存値会計は`scripts/report_h46_full_h_pure_rte.py`。',
        'private event・seed・QPY/cacheは専用directoryのPRIVATE/へ保持し、公開trial記録には価格・law・検査結果を含めた。旧SOURCE・結果・grant・ledger・marker・STOPを変更していない。',
        'branch: `codex/df-full-h-pure-rte-light-20261109`。実行SOURCE: `ac33da5f0cb038a780a447746a38e478bb0cec61`。',
        '次の作業なし。追加探索はユーザーが別途採否を判断する場合のみ。','']
    (p/'REPORT.md').write_text('\n'.join(lines))


if __name__=='__main__':
    main()
