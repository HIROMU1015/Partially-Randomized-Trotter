# Bへ：B0I p7/S2・要求幅h=.1の直接価格

**q=1,2,4,8,16のX/Y価格が揃い、今回configurationで10/26軸を受入。保存6軸＋新規4 compile、対象の受入失敗/UNKNOWN/保留0。m5〜m12の16軸は未取得。mandatory STOP。**

| m | q | T | actual幅 | X compiled RZ | Y compiled RZ | 取得 |
|---|---:|---:|---:|---:|---:|---|
| 0 | 1 | .1 | .1 | 5,820 | 5,820 | 旧D receipt再利用 |
| 1 | 2 | .2 | .1 | 11,333 | 11,333 | 今回X/Y別compile |
| 2 | 4 | .4 | .1 | 22,359 | 22,359 | 今回X/Y別compile |
| 3 | 8 | .8 | .1 | 44,411 | 44,411 | PF common原trial再利用 |
| 4 | 16 | 1.6 | .1 | 88,515 | 88,515 | long-time原trial再利用 |

正本は[run01/AXIS_PRICES.json](run01/AXIS_PRICES.json)／[CSV](run01/AXIS_PRICES.csv)。全26軸を持ち、10行は`ACCEPTED_DIRECT`、残るm5〜m12の16行は`NOT_ACQUIRED`・RZ null。欠測を0や片軸価格で埋めていない。詳細は[REPORT](REPORT.md)。

元configurationは `e6754564a42fc59a7f7088708da23ee861ec193e18f50641eeafd1096dfef7bc`。各行を `(configuration_id, physical_id, m, axis)` でBのprecision側 `bec032fc26093177f529754060df06d5375944b4:docs/research/p7_s2_d01_same_shots_q3_results_20261115/C_RESULT_JOIN.json` へ結合する。original canonical IDも保存し、B total-view IDへ元recordを付け替えない。今回の取得は価格だけで、Q3・shot配分・N/Gの計算0。

## 固定locatorと受入

- 実行SOURCE：`be0f81ebd893415de031ee30ee0755f039adb823`、`scripts/run_pf_p7_h_prices.py`。既存native child・compiler・作用gate・価格builderを再利用した。
- raw / 10価格record / 元保存trialへの対応 / implementation proof：`302d63e9024a0c594d899f407af9331cc7174e86:docs/research/d_pf_p7_s2_h_direct_prices_20261115/run01/`。
- source-bound sidecar / C guard：`11cb034ef39b1796cf83958a405233d93af14a99` の同path。`PRICES.json`・`SIDECARS.json`・`GUARD_CHECK.json`を使用し、行のreceipt selectorから戻れる。
- 元configuration descriptor：`c5466946ab0ef9b50511210ca1e58fb7147bb0f0:docs/research/baseline_finite_signal_plan_20261113/BASELINE_FINITE_MANIFEST.json#candidates[1].configuration`。13roundの再生成・信号計算はしていない。

価格quantityは保存linear H4 / 1Å / STO-3G / tol-only DF10、元H/state/snapshot、P0除外、一体項＋fragment0..6保存順、S2 signed順、scalar c_H＋tail identityをtotal Tに一回適用した**測定付きfull controlled Hadamard wrapper1 shotのcompiled RZ数**。symmetric_directional、Qiskit1.3.0 / rz,sx,x,cx / opt1 / seed17を保持した。native policyは `H46_LONG_TIME_NATIVE_COMPOSE`、RTEは `NOT_APPLICABLE_NO_RTE`。

新規4軸は元index0の7作用gate（native <1e-9 / compiled <1e-7、位相合わせなし）とC guardを通過した。保存6軸も元trial・SOURCE・scalar・axisを保持したまま今回候補へ受入し、C guardのchecked軸数は10。分類 `DIRECT_COMPILED_DETERMINISTIC` / `DETERMINISTIC`、`model_qualified=false`。C guardはimplementation受入で、Nや全RPE科学資格を認定しない。

保存m0は元D receipt/cohort、m4は元EXPLORATIONを保持。m3原trialにはcohort fieldがなく、その欠如を`original_cohort=null`に保ち、既存carrierの `SAVED_T0.8_COMMON` を接続labelとして明示した。新cohortの独立標本と呼ばない。元SOURCEと今回のbinding SOURCEは別欄で保持し、原結果は変更していない。m3のmetrics viewは既存builderの形に合わせただけで、価格やgateの再計数・再実行ではない。

## この5点を次の価格規則へ使える範囲

この幅のq=1/2/4/8/16に対して、SOURCE・scalar・wrapper・compilerを結び付けた直接価格として使える。固定実幅の反復と端効果を扱う規則の**開発・整合確認の入力**にはなるが、今回fitや係数算出をしておらず、直線や任意qの閉包を認定しない。5点を見て選んだ規則に対して、この5点を未使用holdoutと呼ばない。

次にq方向の規則を提案する場合は、幅hでのsigned角度・境界・終端scalar phase・compilerの数値枝をSOURCEと保存回路でどこまで説明できるかを区別する。現行16h/7の閉包監査・係数・2%規則を幅hへ流用しない。保存QPYの新監査、fit、未観測q32〜4096への導出や独立検証が必要なら、別の有限scopeと採択判断へ戻す。今回の10直接価格だけでは残る16軸のCや全額を埋められない。

**435,481,090 RZは要求幅16h/7の保存達成費用として保持する。幅hの全額・優位性・winnerへ転用しない。** Bの保存Q3結果は幅hを候補に残す根拠として参照しただけで、再Q3、shot削減案の生成、価格加重、費用差の算出は行っていない。

## 資源とロードマップ差分

2 cellを2 workerで並列実行、取得wall25.706 s（binding/報告/公開は別）。最大worker RSS401.97 MiB、child/workerはすべて終了code0・signalなし、cache-hit0・科学retry0。段階時間は[run01/TIMING.json](run01/TIMING.json)。新規4 compile以外に保存6軸の再build/compile・再simulationは0。対象4軸の追加receipt・同等jobは事前差分で0と確認した。

統合担当へ：**p7/S2要求幅hの価格は所在のみ6/26から、今回候補での受入10/26へ前進。q1/2/4/8/16が直接価格で揃い、m5〜m12の16軸不足は維持。Q3適合の保存結果と価格の物理ID接続ができるが、幅hの価格規則・長q導出・全額・shot最適化は未実施。** ロードマップ本体は編集せず、この差分を渡す。

commit/push・remote SHA/tree/主要成果物の一致確認後**mandatory STOP**。追加fit・長round取得/予測・QPY監査・Q3/N/G・winnerへ進まない。
