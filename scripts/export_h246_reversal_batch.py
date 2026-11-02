"""Publish literal completed results; no scientific kernel or ranking is called."""
from copy import deepcopy
import csv
from fractions import Fraction
import json
from pathlib import Path
import h246_reversal_batch as c

OUT=c.ROOT/'artifacts/resource_applicability/track_a_h246_reversal_batch_run01/2026-10-24'
READY=c.OUT


def public(name,value):
    text=json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n'
    c.require('/home/' not in text and '"seed"' not in text,'NO_PRIVATE_PATH_OR_SEED')
    (OUT/name).write_text(text)


def write_csv(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def exact(r):
    return str(r['numerator'])+'/'+str(r['denominator']) if r is not None else None


def display(r):
    # Formatting an already evaluated exact result, not new scientific analysis.
    return f"{float(Fraction(r['numerator'],r['denominator'])):.6f}" if r is not None else '未成立'


def saved_unit_stats_only(u):
    out=deepcopy(u)
    if 'saved_axis_statistics' in out:
        stats=out.pop('saved_axis_statistics')
        if isinstance(stats,dict):
            for ax in ('cosine','sine'):
                if isinstance(stats.get(ax),dict):stats[ax].pop('shots',None)
        out['saved_unit_statistics_from_main_0005']=stats
        out['saved_unit_statistics_source_epsilon']=.0005
    return out


def projection(raw,rawsha):
    p=deepcopy(raw)
    for row in p['H2_new_endpoints']:
        row.pop('pairs',None);row.pop('account_inputs',None)
        row['saved_base_explanation_0005']=row.pop('explanation')
        row['uncertainty']=saved_unit_stats_only(row['uncertainty'])
        row['evaluated_endpoint_epsilon']=row['epsilon']
        row['headroom_source']='accounting.shots.axes; saved_base_explanation_0005 is context only'
    for r in p['H2_new_comparisons']:
        r['uncertainty']=[saved_unit_stats_only(u) for u in r['uncertainty']]
    for row in p['integration']['main_15_system_precision_cells']:
        if row['system']=='H2' and row['epsilon']!=.0005:
            for method in ('full','PR'):
                row[method]['uncertainty']=[saved_unit_stats_only(u) for u in row[method]['uncertainty']]
    p['publication_projection']=dict(private_scientific_result_sha256=rawsha,
        omitted_fields=['H2_new_endpoints.pairs','H2_new_endpoints.account_inputs','Old main-epsilon shot fields within quoted unit statistics'],
        scope_labels='Legacy copied explanation is explicitly named saved_base_explanation_0005; new headroom/log bound is accounting.shots.axes',
        scientific_numeric_values_modified=False,new_analysis_calls=0)
    return p


def comparison_row(r,scope,n_by):
    return dict(scope=scope,system=r['system'],method=r['method'],epsilon=r['epsilon'],q=r.get('q'),
        S2=r.get('S2'),Yoshida4=r.get('Yoshida4'),status=r['status'],direction=r.get('direction'),
        n_S2=n_by.get(r.get('S2')),n_Y4=n_by.get(r.get('Yoshida4')),
        G_S2=r.get('G_saved_unchanged',[None,None])[0],G_Y4=r.get('G_saved_unchanged',[None,None])[1],
        R_G_exact=exact(r.get('R_G')),R_C_effective_exact=exact(r.get('R_C_effective')),R_N_total_exact=exact(r.get('R_N_total')),
        C_term_exact=exact(r.get('additive',{}).get('C_at_S2_N')),N_term_exact=exact(r.get('additive',{}).get('N_at_S2_C')),
        cross_term_exact=exact(r.get('additive',{}).get('cross')),delta_G_exact=exact(r.get('additive',{}).get('total')),
        G_SE_S2=r.get('uncertainty',[{},{}])[0].get('G_SE'),G_SE_Y4=r.get('uncertainty',[{},{}])[1].get('G_SE'),
        population_guarantee=False)


def export():
    cfg=c.read(c.ASSETS/'BATCH_contract_PRIVATE.json');run=Path(cfg['run_root'])
    done=c.read(run/'completion.json');raw=c.read(run/'results/analysis.json')
    c.require(done['status']=='COMPLETE_MANDATORY_STOP' and done['used']==c.counts() and
        done['result_sha256']==c.sha(run/'results/analysis.json'),'COMPLETED_EXACT_BATCH_ONLY')
    native=c.read(OUT/'native_completion.json');c.require(native['native_exit_code']==0 and native['reaped'],'NATIVE_EXIT_REAP')
    inputs=c.read(cfg['manifest']);before=c.read(c.prior.ASSETS/'inputs_PRIVATE.json')
    key=lambda r:(r['system'],r['method'],r['epsilon'])
    quoted_in_result=[x for cell in raw['integration']['main_15_system_precision_cells'] for x in (cell['full'],cell['PR']) if not(x['system']=='H2' and x['epsilon']!=.0005)]
    c.require(sorted(inputs['quoted_main_comparisons'],key=key)==sorted(quoted_in_result,key=key),'OLD_22_LITERAL_QUOTE')
    p=projection(raw,done['result_sha256']);public('analysis.json',p)
    n_by=raw['integration']['n_by_candidate']
    write_csv('same_q_comparisons_8.csv',[comparison_row(r,'AUXILIARY_SAME_Q',n_by) for r in raw['same_q_comparisons']])
    main=[r for cell in raw['integration']['main_15_system_precision_cells'] for r in (cell['full'],cell['PR'])]
    write_csv('primary_comparisons_30.csv',[comparison_row(r,'PRIMARY_DIFFERENT_Q',n_by) for r in main])
    directions=[]
    for row in raw['integration']['main_15_system_precision_cells']:
        a,b=row['full'],row['PR']
        directions.append(dict(system=row['system'],epsilon=row['epsilon'],scope=row['scope'],
            Full_lower=a.get('direction'),PR_lower=b.get('direction'),reversal=row['reversal'],
            Full_G_ratio_exact=exact(a.get('R_G')),PR_G_ratio_exact=exact(b.get('R_G')),
            Full_status=a['status'],PR_status=b['status'],PR_n_S2=n_by[b['S2']],PR_n_Y4=n_by[b['Yoshida4']],
            PR_G_SE_S2=b['uncertainty'][0]['G_SE'],PR_G_SE_Y4=b['uncertainty'][1]['G_SE'],
            independent_standard_Y4_at_this_endpoint=row['independent_standard_Y4_at_this_endpoint'],
            H2_independent_candidate_law_support_only=row['H2_candidate_law_independent_support_only'],
            weak_H4_005=row['weak_H4_005'],evidence=row['evidence']))
    write_csv('primary_directions_15.csv',directions)
    write_csv('same_q_directions_4.csv',[dict(system=row['system'],q=row['q'],epsilon=row['epsilon'],
        scope=row['scope'],Full_lower=row['full']['direction'],PR_lower=row['PR']['direction'],reversal=row['reversal'],
        Full_G_ratio_exact=exact(row['full']['R_G']),PR_G_ratio_exact=exact(row['PR']['R_G']),pure_order_causal_effect=None)
        for row in raw['integration']['same_q_4_system_q_cells']])
    endpoints=before['rows']+raw['H2_new_endpoints'];c.require(len(endpoints)==60,'SIXTY_PRIMARY_ENDPOINTS')
    endpoint_rows=[]
    for r in endpoints:
        new=r.get('subscope')=='B_NEW_ENDPOINT'
        shot=r['accounting']['shots']['axes'] if new else None
        e=r['explanation'];st=r['uncertainty'].get('saved_axis_statistics') or {}
        def val(axis,k):return st.get(axis,{}).get(k)
        endpoint_rows.append(dict(system=r['system'],epsilon=r['epsilon'],candidate=r['candidate'],method=r['target']['method'],
            formula=r['target']['formula'],q=r['target']['q'],prefix=r['target']['prefix'],n=r['target']['n'],
            cohort=r['cohort'],status=r['status'],C_cos=r['C']['cosine'],C_sin=r['C']['sine'],
            N_cos=r['N']['cosine'],N_sin=r['N']['sine'],G=r['G_saved'],G_SE=r['uncertainty']['G_SE'],
            saved_unit_C_SE_cos=val('cosine','SE') if val('cosine','SE') is not None else val('cosine','standard_error'),
            saved_unit_C_SE_sin=val('sine','SE') if val('sine','SE') is not None else val('sine','standard_error'),
            bias_cos=e['signed_bias']['cosine'],bias_sin=e['signed_bias']['sine'],log_B_base=e['log_B_base'],
            log_B_margin=e['log_B_margin'],u_cos=e['allowance']['cosine'],u_sin=e['allowance']['sine'],
            headroom_cos=shot['real']['headroom'] if new else e['headroom']['cosine'],
            headroom_sin=shot['imag']['headroom'] if new else e['headroom']['sine'],
            origin='NEW_APPROVED_B_ENDPOINT' if new else 'LITERAL_SAVED_ENDPOINT',new_G_SE_evaluated=False,
            population_guarantee=False))
    write_csv('primary_endpoints_60.csv',endpoint_rows)
    write_csv('H2_new_endpoints_16.csv',[r for r in endpoint_rows if r['origin']=='NEW_APPROVED_B_ENDPOINT'])
    # Copy only accepted result receipts. Private grant/marker/ledger remain private.
    ledger=sorted((run/'ledger').glob('*.json'));counts=dict.fromkeys(c.KINDS,0);receipts=[]
    for path in ledger:
        r=c.read(path);counts[r['kind']]+=r['amount'];receipts.append(dict(sequence=path.stem,kind=r['kind'],sha256=c.sha(path)))
    c.require(counts==c.counts() and len(ledger)==83,'EXACT_COMPLETED_LEDGER')
    public('execution_ledger_receipt.json',dict(status='EXACT_83_PASS',used=counts,ledger=receipts,
        grant_consumed_by_new_marker=True,consumed_marker_sha256=c.sha(run/'CONSUMED.json'),
        old_grants_reused=False,attempts=1,retries=0))
    public('result_verification.json',dict(status='PASS_COMPLETED_VALUES_AND_SCOPE',SOURCE=cfg['source_commit'],
        contract_sha256=c.digest(cfg),input_manifest_sha256=cfg['manifest_sha256'],
        private_scientific_result_sha256=done['result_sha256'],public_projection_sha256=c.sha(OUT/'analysis.json'),
        exact_decompositions_completed=sum(x['status']=='EXACT_ACCOUNTING_DECOMPOSITION' for x in raw['same_q_comparisons']+raw['H2_new_comparisons']),
        new_H2_eligible_endpoints=sum(r['status']=='ELIGIBLE' for r in raw['H2_new_endpoints']),new_H2_endpoints=16,
        quoted_old_comparisons_unchanged=True,main_epsilon_0005_recomputed=False,
        actual_new_statistics=0,new_sampling=0,new_compile=0,new_simulation=0,global_ranking=0,
        public_projection_scope_labels='Main .0005 base explanation is kept separate from evaluated endpoint shot diagnostics',
        D_independent_execution=False,strict_COMPARE='BLOCKED',mandatory_STOP=True))
    main_table=['| 系 | ε | Full低費用 | PR低費用 | Full G比 | PR G比 | 点推定の逆転 |','|---|---:|---|---|---:|---:|---|']
    for row in raw['integration']['main_15_system_precision_cells']:
        def label(r):return {'S2_LOWER':'S2','Y4_LOWER':'Y4'}.get(r.get('direction'),'未成立')
        main_table.append('| '+' | '.join(map(str,[row['system'],row['epsilon'],label(row['full']),label(row['PR']),
            display(row['full'].get('R_G')),display(row['PR'].get('R_G')),'あり（弱い順位）' if row['weak_H4_005'] else 'あり' if row['reversal'] else 'なし']))+' |')
    control_table=['| 系 | q | Full低費用 | PR低費用 | Full G比 | PR G比 | 逆転 |','|---|---:|---|---|---:|---:|---|']
    for row in raw['integration']['same_q_4_system_q_cells']:
        control_table.append('| '+' | '.join(map(str,[row['system'],row['q'],label(row['full']),label(row['PR']),display(row['full']['R_G']),display(row['PR']['R_G']),'あり' if row['reversal'] else 'なし']))+' |')
    factor_table=['| 系 | 方法 | q | C_eff比 | N_total比 | G比 |','|---|---|---:|---:|---:|---:|']
    for r in raw['same_q_comparisons']:
        factor_table.append('| '+' | '.join(map(str,[r['system'],r['method'],r['q'],display(r['R_C_effective']),display(r['R_N_total']),display(r['R_G'])]))+' |')
    d=deepcopy(c.read(READY/'independent_confirmation_preparation.json'))
    d.update(status='PREPARATION_COMPLETE_AFTER_A_B_C_NO_NEW_INDEPENDENT_EXECUTION',
        observed_scope='Same-q .001 has no reversal in the four system/q cells; primary different-q reversals remain',
        necessity='No new acquisition required to describe conditional fixed-configuration point reversals. Independent acquisition only if independent standard-Y4 replication is part of the intended claim.',
        minimal_if_selected='Reuse existing independent S2 unit samples where the recorded static joins pass; acquire only H4/H6 standard-Y4/q4 groups; separate .001 S2 repricing and any G-SE/interval contract required.',
        scientific_pure_order_causality_claim_supported=False,new_valid_grants=0,n_selected=False,
        next_actor='User/GPT decide the intended claim; Codex fixes a separate design/source/input/call budget only if independent confirmation is selected')
    public('independent_confirmation_plan.json',d)
    report=f'''# H2/H4/H6 同一q対照・H2精度補完：承認済み一括解析 run01

**固定83イベントを一回正常終了した。同一qではH4/H6ともq4はFull/PR両方Yoshida4、q8は両方S2が低費用で、方向逆転はない。異なるqの主比較では逆転が残り、H2は今回補完した他4精度を含む登録5精度すべてでFull S2・PR Y4が低費用だった。** この結果は固定構成の費用方向に関するもので、qだけの因果効果や母平均優位を証明しない。

## 固定条件・実行証拠

linear H2/H4/H6、隣接1 Å、STO-3G、tol-only DF1e-8、rank3/10/19、T=.8、P=0、α_axis=.025、USUAL_SIGNED。Full prefix3/10/19、PR prefix1/3/5、K4・ABS_TIME_PER_OUTER32。主比較はH2 q4/q2、H4/H6 q8/q4。補助同一qはH4/H6 q4/q8、ε=.001。symmetric-directional、状態準備を除くfull Hadamard cosine/sine wrapperのcompiled RZ費用。候補を各精度で選び直していない。

| 実行識別子 | 値 |
|---|---|
| SOURCE | `{cfg['source_commit']}` |
| 契約SHA256 | `{c.digest(cfg)}` |
| 入力SHA256 | `{cfg['manifest_sha256']}` |
| 元科学結果SHA256 | `{done['result_sha256']}` |
| 実行 | native exit0/reaped、{native['wall_seconds']:.3f}秒、attempt1/retry0 |
| 実ledger | bootstrap1/load1/A分解8/B会計16/内部shot16/内部headroom32/B分解8/統合1＝83 |

[ユーザー承認](approval_receipt.json)、[直前preflight](fresh_preflight.json)、[終了](native_completion.json)、[台帳receipt](execution_ledger_receipt.json)、[固定契約・人工検査](../../track_a_h246_reversal_batch_ready/2026-10-24/REPORT.md)。新namespaceのgrantを承認後に一回消費し、旧grantは再使用していない。SOURCE/input/runtime1,007 fileと既知1,792 fileを保全。新sampling/build/compile/simulation/allowance/reference/Hamiltonian・標本統計・再ランキング0。旧strict COMPAREはBLOCKED、mandatory STOP。

## C：異なるqの主比較・3系5精度

比はYoshida4/S2。表示は保存されたexact比の小数6桁表記であり、exact値は[15セルCSV](primary_directions_15.csv)、C/N分解と差は[30比較CSV](primary_comparisons_30.csv)、単価・shot・bias/log_B/u/headroom・n/保存SEは[60 endpoint CSV](primary_endpoints_60.csv)。元22比較は引用し、H2他4精度の8比較だけを追加した。

{chr(10).join(main_table)}

H2 ε=.0005のFull G比は1に近い保存会計であり、PRの有限標本SEと同じ不確かさでは扱わない。H4 ε=.005のPR点差は各保存SEより小さく、頑健な順位例にはしない。H4/H6 ε=.0005ではFull/PRともY4が低費用であり、逆転の有無を系の名前だけで説明できない。登録点間の境界は補間しない。

H2の新16 endpointはすべて宣言allowance下で適格だった。[新16行](H2_new_endpoints_16.csv)。受入済みDIAGNOSTICSの通常signed bias/reference/allowance/log_B_marginと元compiled単価を使い、共通PF表の別政策を代入していない。旧ε=.0005は再計算0。H2 PRの単価標本はS2 n33/Y4 n34で固定、新GのSEは未計算/null。単価の保存SEは引用するが、旧主精度のG SEを他精度へ流用しない。

## A：同一qの補助対照・ε=.001

{chr(10).join(control_table)}

{chr(10).join(factor_table)}

全8比較でexact軸別積とC/N/crossの加法恒等式が成立した。[8比較の差・寄与・n/保存SE](same_q_comparisons_8.csv)、[4系qセル](same_q_directions_4.csv)。同一qではY4の単価増加をq4のN削減が上回る一方、q8では上回らなかった。Full/PRの両方でこの方向が揃った。元のS2/q8対Y4/q4では、異なるqでのN/Cを比べるため方向が異なる。

したがって、今回の逆転は「S2/q8とY4/q4を採った固定構成の資源比較」として成立し、同一qでFull/PRが逆方向になる現象は今回の保存範囲では支持されなかった。q/RTE構成の影響を限定する対照であり、signed piece・RTE配分・normalization・deterministic/random partitionも関わるため、qのみを原因と断定しない。主比較を同一q結果で置き換えない。

## C/N/Gと保存診断の対応

費用の正本は`G=Σ_a N_a C_a`。新16比較のexact分解は既存2e59の関数をそのまま使用した。H4/H6主比較のPR C比は元解析でH4が1未満、H6が1超だが、ε=.001では両者ともN削減によりY4 Gが低い。同一qではC比もN比も主比較とは異なり、Cだけの説明は成立しない。

Fullのbase log_B=0、PRはY4で保存normalizationが増える。保存bias/uとheadroomの改善、normalizationの増加を同時に含む元shot会計を維持した。新H2のheadroom/Nは各εの実評価出力から引用する。旧代表RTE6軌道では主比較S2→Y4でevent数が半減し、Y4各outerは10/12/10配分だったが、同一qはsigned piece構成が異なり、event数だけをcompiled RZへ換算しない。[元構成・診断](../../track_a_h246_order_cost_analysis_run01/2026-10-24/REPORT.md)。新しいsignal・回路・operation寄与の計算は行っていない。

## 証拠の強さ・Dの独立確認準備

主表は探索標本を選択後に固定した点推定である。H2の別確認には同じ標準Y4 candidate-lawの支持があるが、現DIAGNOSTICSの各endpointと同一policyの独立再現と認定していない。H4/H6の既存高次確認はm5_bestであり、今回の標準Y4/q4へ転用しない。PRの保存G SEは回路費用のMonte Carlo誤差であり、shot誤差・PF bias・母平均の保証ではない。新CI/検定/差SEを算出せず、候補間共分散を仮定せず、未観測stratumの費用や母確率を0にしない。

[D準備](independent_confirmation_plan.json)。既存独立S2はH4 n25/H6 n23、law/input/preparation/native数値SOURCE/compilerが静的対応したので、再取得を既定としない。H4のNPZ容器SHA差は科学配列一致と区別する。新ε=.001のS2 G/SEへ再会計する場合は別契約が必要。標準Y4を独立確認するなら最小新取得候補はH4/H6のq4の2群、必要軸は`2×(n_H4+n_H6)`。nは未決定で、求めるMC精度・統計目的・tail限界・cohort/seed除外範囲・非適応STOPを先に固定する。

今回の限定された主張を記述するために新規取得を必須とはしない。独立した標準Y4方向再現を主張に含めるかをユーザー・GPTが判断する。同一qで逆転が消えた結果に対し、標準Y4の独立標本追加だけで次数単独の因果効果が証明できるとはしない。新独立sampling/compileや詳細レビューは開始していない。

## 保全・公開範囲・次の担当

[検証](result_verification.json)、[保全監査](preservation_audit.json)、[file目録](file_inventory.json)。元科学結果はprivate領域でbyte保全し、[公開JSON](analysis.json)は軌道ごとのprivate単価配列/内部入力を省いたprojectionとした。元resultのH2行が保持する説明fieldはε=.0005の保存元contextであるため、公開側では`saved_base_explanation_0005`と明記した。新精度のheadroom/log boundは`accounting.shots.axes`が正本で、CSVもその値を引用する。科学値は変更せず、SOURCE/元結果/台帳/旧STOPを保全した。

Full/PRの保存reference末尾差、empirical allowance、有限標本・事後選択、H6旧中断stageの元parent exit/reap不明を保持する。新規結果を外部再現・独立担当の科学レビュー・厳密certificateへ格上げしない。論文主表/Figure2・共有環境・他ユーザーprocessを変更していない。

次：**ユーザー・GPT**が固定構成とq対照を踏まえた主張範囲、および標準Y4独立確認の必要性を判断する。**Codex**は追加実行を開始せず、今回のA/B/C・D準備を完了し[STOP](STOP.json)。独立確認を選ぶ場合のみ、別SOURCE/input/call/標本設計を固定して承認を受ける。
'''
    (OUT/'REPORT.md').write_text(report)
    print(json.dumps(dict(status='EXPORTED_WITHOUT_NEW_ANALYSIS',public_projection_sha256=c.sha(OUT/'analysis.json'),original_result_sha256=done['result_sha256'],primary_cells=15,same_q_cells=4,primary_endpoints=60)))


if __name__=='__main__':export()
