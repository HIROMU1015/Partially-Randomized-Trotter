# O1 M1：保存信号からの費用成立性screen準備

2026-11-06、SERVER 2（GPUA100-gleap02）。base `185104f64a4594355a3813d04d9e50bd4abdb6b5`。**純粋helper＋架空値のunit testのみ。実H4のshot・費用・必要単価上限の新計算、保存値数値再解析、H4 load／simulation／回路build/compile、価格MCは未実行。** 科学実行未承認、成果公開後mandatory STOP。

[C0負結果][C0]は維持する。[q64 r4/r8結果][SIG]の減衰軽減は総費用改善の証拠ではない。本書は次の最小計算の準備で、科学的GO・候補順位・新規性の採択ではない。

## 最小実装と再利用

追加は[helper](../../../src/trotterlib/o1_m1_cost_feasibility.py)、[unit test](../../../tests/test_o1_m1_cost_feasibility.py)、本書の3ファイル。既存SOURCE・結果は不変更。

- `account_saved_c0_signal`は既存`account_radius_round(cost=None)`／`RPEErrorAllocation`を再利用。明示finite-RTE config・round・保存signal row・physical／arm hashを受け取り、q/r/K/deltaと保存A／normalization／rhoの一致を確認する。PRの`config=None`＝単位attenuation経路は拒否する。SOURCE/input・元H state・faithful identityのbindingは呼出側が固定commit/pathで確認する前提で、helper自身は入力を再生成しない。A／normalization／rhoの`rel_tol=1e-12`はdata-law整合確認で、旧operator／phaseの科学許容値の代替ではない。
- `optimistic_unit_price_ceiling`は未価格のround accountと、全7roundの強DF-PF費用記録を接続する。physical／backend／delta／beta／metric／full no-prep scopeと、明示したprospective compiler hashの一致を要求する。baseline armは固定`strong_DF_PF_4_substeps_spin_fused`を要求し、同physical contextのP/DF PR aggregate誤選択も拒否する。部分baselineやprep込み価格を使わず、未取得のcompiler価格を取得済みと偽装しない。
- 後段の再利用先は既存`MatchedNoPrepCostProvider`／並列providerと`aggregate_registered_rounds`。今回は一度も呼んでいない。新law価格、全round資格・費用、不確かさが揃う前に欠落価格を0として集計しない。実行runnerは追加せず、保存値read/bindingの科学呼出は承認後に限定する。

## shot・必要単価の意味

比較profileはC0と同じH4-A・保存rank10・元H厳密基底state／no-prep、共通T_eff/V_square・faithful identity、0.1 Ha／成功目標95%、delta=.05、K2、全7round m0..6。PF/RTE/stat/model/実装/reserve枠`.06/.03/.15/.03/.01/.02`、beta_RPE=.30、**各軸alpha=.05/14を維持する**。q16/32/64中心でも`.05/6`へ再配分しない。

既存accountingは保存された同roundのPF半径下界・RTE信号diskとAから観測半径下界を作り、既存Hoeffding式でshot数を導出する。単位半径・旧r2 shot／価格・短時間4条件を新r4/8へ転用しない。`not_certified`を維持し、energy／alias・branch・IID等の包括的保証へ昇格しない。

全7round baseline費用をB、一roundのshotをN_c,N_s、両軸full no-prep期待単価をC_c,C_sとする。他round費用の非負性だけから

`G_candidate < B なら N_c C_c + N_s C_s < B`

が必要。従ってshot重み付き単価`C_weighted=(N_c C_c+N_s C_s)/(N_c+N_s)`には、`C_weighted < B/(N_c+N_s)`が必要となる。helperはこの閾値を、入力baseline floatが表す値に対する正確なFraction（分子・分母）で返す。巨大shot数でもunderflowで0にしない。**実H4の保存値による閾値は今回は計算していない。**

これは**全RPE改善の十分条件ではない**。他roundを無料と認定したaggregateではなく、必要条件だけを得る楽観的緩和である。`round_price_lower_bound`、`full_candidate_total_cost`、`full_rpe_improvement`は全て`None`。期待CXは整数とは限らず、閾値が1未満というだけでは実現不能とは言えない。特定trajectory・代表controlled検査・旧r2平均・未検証proxyは新law全体の正の価格lower boundにならない。

## 未承認の最小有限案と保存入力

**初回案は保存q64・r4/8・P/DFの4点のshot算術＋楽観必要単価条件だけ**。H4再load／新signal／build/compile／価格MC／sampling／全RPE総費用の新計算は含めない。

| 固定資産 | 再利用箇所と留保 |
|---|---|
| `185104f...`の[result][SAVED]と同directoryの`signal_r4.json`／`signal_r8.json` | `rows`の保存bounds／事前allowance・資格と辞書別config。入力hash・元state・分割・SOURCEを照合し、再simulationしない |
| 保存finite-RTE config | 既存`RTEConfig`へ復元しq/r/K/delta・lambda・A・normalizationを照合する案。validatorのnormalization snapshotを、価格providerのcanonical law／compiler cohortの代替としない |
| `50ee55b...`の[C0 result][C0RAW] | 同一snapshotのphysical／辞書context、`total_cost.strong_DF_PF`の全7round予算／compiler条件。r2の信号・shot・価格は引用対照のみ |
| 比較compiler／metric | Qiskit1.3.0、rz/sx/x/cx、optimization1、seed7、all-to-all、主CXを維持。prospective compiler hashをbaseline cohortへ合わせ、実compileしない |

C0 r2の`signal_rounds`は旧account形式で、新probeの`rows`＋`arms[辞書].bounds`とはschemaが異なる。本helperへ旧r2記録を無条件投入するadapterは用意しておらず、今回r2は既存accountの引用のみ。将来r2を数値入力として再利用する場合も、SOURCE／schema／同条件を明示固定する。

保存値を使うshot／必要単価算術も新しい科学的再解析のため、上記固定SOURCE・保存入力・4点・規則についてユーザーの明示承認後に行う。今回は実値を読んで再計算・再rankingしていない。

後続にq16／32／64、r2／4／8、K2、delta=.05、split固定の有限候補を検討し得るが、**q16/32 r4/8の信号は未検証**。q64の信号・boundsを転用しない。必要なら対象round／r／辞書、選択規則、独立参照／数値判定、baselineの適格PF step、価格SE／CI規則を先に固定し、新signal／build／shot／価格の範囲を別途明示承認する。今はその格子を評価していない。

全RPE比較には、選択した全mのlaw・信号・半径・誤差配分・分枝／成功条件・full no-prep価格が必要。q64必要条件の通過やsample SEを、全roundの費用優位性へ昇格しない。旧F2価格・単位半径・旧proxyで不足を埋めない。

## 人工テスト・GO/STOP

**32 passed in 1.90s**。架空scalar law・半径／phase bounds・費用だけで、既存shot式、非単位半径、r4/r8両辞書、config=None拒否、保存条件／A／rho不一致、個別phase枠、14軸alpha、比較scope／compiler／physical context、PR aggregate誤選択と部分baseline拒否を確認。非対称shotの重み付け、極大整数でもFraction保持、単一round必要条件を満たしても他roundで負ける架空反例も確認した。

fixtureで`np.load`、分子preparation、回路build、価格provider／compile、samplerを禁止。既存M1専用環境をread-onlyで再利用し、専用cache `/tmp/o1_m1_cost_feasibility_preparation_cache_20261106`、bytecode／pytest cache無効。人工PASSは科学性能の証拠ではなく、人工許容値をH4へ転用しない。

**GO条件**は保存SOURCE/input・profile・law・scopeの一致と限定算術の明示承認。結果は必要単価条件までで、価格資格がなければ改善はUNASSESSED。

**STOP**：現在未承認。将来もsignal／半径／phase不適格、SOURCE/input・比較不一致、必要signal／価格欠損なら止める。適切な新law全体の価格lower boundによってq64だけでも全baseline予算超過を正当化できれば、その候補を終了できるが、そのboundは本準備では得ていない。全round・同条件価格と不確かさを揃えた有限候補で改善根拠がなければ、その集合を終了し、格子・split／H／DFを自動拡張しない。未評価と一般のPR不可能性は区別する。

次は**ユーザー**が初回4点の保存値再解析・shot算術・必要単価評価を許可するか判断する。**Codex**は公開照合後mandatory STOP。C0負結果、旧SOURCE・入力・raw・result・grant・marker・STOPを保全し、新承認guard／grant／管理frameworkは作らない。

[C0]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/50ee55bc30f30838e1ac0704bc94f720914640de/docs/research/track_a_o1_m1_c0_parallel_science_20261106/REPORT.md
[SIG]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/185104f64a4594355a3813d04d9e50bd4abdb6b5/docs/research/track_a_o1_m1_attenuation_science_20261106/REPORT.md
[SAVED]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/185104f64a4594355a3813d04d9e50bd4abdb6b5/artifacts/track_a_o1_m1_attenuation_science_20261106/run01/result.json
[C0RAW]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/50ee55bc30f30838e1ac0704bc94f720914640de/artifacts/track_a_o1_m1_c0_parallel_science_20261106/run01/result.json
