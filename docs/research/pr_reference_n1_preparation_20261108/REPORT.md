# 新A N1：vacuum T価格境界の数値実行準備

2026-11-08 JST、SERVER 2（`GPUA100-gleap02`）。基準commit **`0dbf1fe265176677e026ea982c0bd435987f4338`**、専用branch `codex/pr-reference-n1-preparation-20261108`。

**Stage A〜Dの静的準備を完了し、mandatory STOP。** 保存価格・確率の対応と後続数値実行仕様を確定した。K、d0、d2、dmin、pointwise/期待閾値の数値は今回評価していない。新しい科学runner・合成・compile・simulationは実装／実行していない。

## 目的と固定条件

CT-5Bの条件付きT差から、Referenceが優越するためにvacuum価格へ要求する十分境界を、後続の限定再解析で数値化できるようにする。実T-countは採用strict wordの物理T+T†出現数であり、最小Tではない。

Linear H4・1.00 Å・STO-3G・DF12、元Hの厳密ground供給・状態準備費除外。Partial S2、LD3、h=.02、finite RTE K=2、ordinary `diag(I,V)`、canonical independent finite-bit law、**m7/q128/r1/保存73 bits**を固定する。一様strict epsilon=10^-11、Qiskit1.3.0/compiler A、rz/sx/x/cx、全結合・Targetなし。

B′の4-CX controlled-RZZ、ReferenceのF/G・actual SOURCE inverse・vacuum位相補正・2 register・8-CX CSWAP、既存D融合・basis共有・scalar/event位相・X/Y wrapperを維持する。エネルギー精度1.5936e-4 Ha、全失敗確率.05、phase cap .4、alpha1/720、保存199 shot/axisを変更しない。追加8 qubitやnative CX/RZ/depthをTと合算しない。

## Stage A〜Dの結果

| Stage | 静的に確認／確定した事項 | 資格 |
|---|---|---|
| A：入力対応 | CT-4Bの8保存行のrole名・符号・価格、中央4 keyのe=-11 record、SOURCEの対応 | `SAVED_INPUT_MAPPING_CONFIRMED`。価格の再集計ではない |
| A：確率 | B′保存`TOTALS_18ROUND.json`のm7行`contributions.p_order2`に厳密有理値あり。V3/CT-0のm7・73 bitsと整合 | `SAVED_PROBABILITY_LOCATED`。分布関数は実行していない |
| B：境界 | CT-5Bのpointwiseと期待十分条件を別々に継承 | `CONDITIONAL_ON_COMPILER_CONTEXT`。閾値は`NOT_EVALUATED` |
| C：実行仕様 | 入力hash、整数／Fraction算術、外向き表示、保存行との検算、出力・STOPを指定 | `PREPARATION_COMPLETE`。数値実行承認や科学的GOではない |
| D：不足資格 | vacuum・common frame・到達する場合のb2 word、全support CT RPE、18round総T/T-depthを分離 | 引き続き`UNQUALIFIED` |

### 保存確率の取得元

[B′ TOTALS_18ROUND.json](../../../artifacts/pr_reference_bprime/2026-11-07/TOTALS_18ROUND.json)の`rows[7]`をm=7/q=128/r=1/bits=73/shots_per_axis=199でも選択確認した。保存文字列は

```text
p2 = 642545340934082409/9444732965739290427392
```

である。これは既存値の引用であり、新しい確率計算ではない。`pr_reference_bprime_accounting.distributions`→`transfer_cost_bounds.distribution`→`track_c_h4_finite_bits.integer_cdf`の静的対応により、order2のfloor-CDF整数区間幅を表す。nominalな浮動確率、rare stratumの確率、保存2 trajectoryの頻度へ置換しない。

価格の正本はCT-4B [ROLE_T_COUNTS](../../../artifacts/pr_reference_ct4/2026-11-07/ROLE_T_COUNTS.json)、[PILOT_T_COUNTS](../../../artifacts/pr_reference_ct4/2026-11-07/PILOT_T_COUNTS.json)、[KEY_REUSE_AND_SYNTHESIS](../../../artifacts/pr_reference_ct4/2026-11-07/KEY_REUSE_AND_SYNTHESIS.json)。具体的な行・field・価格・keyは[N1_INPUT_AND_ACCOUNTING](N1_INPUT_AND_ACCOUNTING.md)。固定SOURCEと保存入力のSHA256/Git blobは[INPUT_REFERENCES.json](../../../artifacts/pr_reference_n1_preparation/2026-11-08/INPUT_REFERENCES.json)に記す。旧ファイルの内容を変更していない。

## 継承する構造と、まだ主張できないこと

[CT-5Bの条件C](../pr_reference_ct5b_20261107/STRUCTURAL_CERTIFICATES.md)と共通word policy Pの下で

\[
\Delta T(\omega)=K+\sum_{i=1}^{128}d_{o_i}+E_B(\omega)-v_P(\phi_\omega),
\quad K=D_B^\star-D_R^\star-C_{sw}^\star+S_B^\star,
\quad d_o=2t(h_o)-t(f_o).
\]

Dのstar価格は未融合roleとsigned境界補正の和。共通frame価格の取消し、product/wrapperの保存T=0価格、非負event価格の省略をCT-5Bの適用条件付きで引き継ぐ。D/frameの全context調査・compiler証明を今回やり直していない。

後続N1は、`K,d0,d2,dmin,V_threshold,U_threshold`だけを厳密整数／有理数で評価する。`Vmax < V_threshold`と`UV < U_threshold`は別の**保守的十分条件**である。未知のVmax・UVを求める作業はその数値実行にも含めない。閾値が非正でもReference劣位／no-goとは判断しない。

同一trajectoryのvacuum位相、order2、product符号を独立と仮定しない。期待の線形性はこの独立性を要求しないが、UVを別途評価する際には同じcanonical joint lawが必要。CT-4Bの保存2 vacuum価格を全support上界や期待値にしない。

未資格事項は、全support vacuum wordの取得・有限停止・strict保証、一様／期待vacuum価格上界、未取得common frame word、到達する場合のB′ b2 event word、全support CT RPE成功、全18round総T・T-depth。差から価格が消えても必要wordと誤差は消えない。CT-0/CT-4Bの保存shot条件を全support実装の成功へ格上げしない。

## 参照と確認範囲

正式共通運用指示2026-10-25、親／worktree AGENTS.md、PROJECT_MAP、研究概要を参照。固定CT-5Bの[REPORT](../pr_reference_ct5b_20261107/REPORT.md)・[STRUCTURAL_CERTIFICATES](../pr_reference_ct5b_20261107/STRUCTURAL_CERTIFICATES.md)・[FINITE_KEY_AND_UNKNOWN_PRICE](../pr_reference_ct5b_20261107/FINITE_KEY_AND_UNKNOWN_PRICE.md)・[CONDITIONAL_T_ADVANTAGE](../pr_reference_ct5b_20261107/CONDITIONAL_T_ADVANTAGE.md)・[NEXT_SCOPE](../pr_reference_ct5b_20261107/NEXT_SCOPE.md)の既存結論を再利用した。

[CT-5A REPORT](../pr_reference_ct5a_static_20261107/REPORT.md)／[条件式](../pr_reference_ct5a_static_20261107/CONDITIONAL_BOUNDS.md)、[CT-4B REPORT](../pr_reference_ct4b_20261107/REPORT.md)／[SCOPE](../pr_reference_ct4b_20261107/SCOPE.md)、[V3](../pr_reference_rotation_v3_20261107/REPORT.md)の保存角度台帳・role SOURCE、[CT-0](../pr_reference_ct0_20261107/REPORT.md)のphase/precision資料と照合した。保存record metadataとSOURCEを読んだのであって、guard・word価格・operator検証を再実行したのではない。新たな一次文献照合や独立科学レビューは実施していない。

## 完了・次担当

保存schema／field／metadata／hashと新規文書リンクを技術確認した。科学再解析、K/d/閾値の数値化、strict合成／guard、build/compile、sampling/simulation、最適化、18round会計はいずれも**0**。旧SOURCE/raw/result/grant/ledger/marker/STOP、共有設定、他ユーザー・別Trackは変更していない。[停止記録](../../../artifacts/pr_reference_n1_preparation/2026-11-08/STOP.json)。

**mandatory STOP。** 次担当は**ユーザー／GPT**：[NUMERIC_EXECUTION_SCOPE](NUMERIC_EXECUTION_SCOPE.md)に記す保存入力だけの限定数値再解析を実施するか判断する。**Codex**は、その明示的な実行承認後に限り同scopeを実施できる。N2/N3、SK/CT-3B、全support価格、全18round総Tへ自動進行しない。
