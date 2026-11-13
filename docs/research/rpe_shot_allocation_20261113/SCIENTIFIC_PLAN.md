# 初回の有限whole-RPE shot配分・費用比較案

2026-11-13 JST。**未採択・未実行。** 技術adapter完成は実H4 MC・保存再解析・N/G最適化の許可ではない。前S3の許可を再利用/拡張しない。[HANDOFF](HANDOFF.md)を用い、追加の承認証明書は要求しない。

## 1. 推奨する初回範囲

**1つの外部供給physical V2 configuration×全schedule×最大3個の事前固定whole-shotベクトル**を初期案とする。既存H4を対象にする場合はM12/13roundを維持する案だが、今回対象ID・N・scale値を新規生成/採択していない。物理候補・signal生成、価格取得は担当側の別scope。

3個は有限科学scope案でありCPU/RAM/worker上限ではない。Nは同じdecoderを通る結合ベクトル。base×共通scale等を最初の候補とし、実scale値/ceil/軸・group制約/探索外域をユーザーが採択する。26個の独立N探索やN単調二分探索は行わない。

## 2. 実行前に必要な決定・入力

| 項目 | 所有者と必要な内容 |
|---|---|
| candidate/mean | A/Cのsignal担当：canonical＋元physical bridge、全round raw/corrected/logB、H/state/target/SOURCE、数値/物理/law不確かさ。Bで再計算しない |
| claim/誤差 | ユーザー＋A/B：条件付き経験的screenか全事象上側riskか、元H target/proxy、未解決loss、誤branch/alias、finite-K/raw/law差・数値誤差の取扱い |
| 配分menu | ユーザー＋B：最大3のexact整数ベクトル、allocation ID/SOURCE、制約と未探索域。旧nominalは出発候補であり十分Nでない |
| 統計精度 | ユーザー＋A/B：評価量、MSE/未解決率の誤差幅、confidence・同時family、loss support、固定反復数、事前seed。旧43,819反復を自動採用しない |
| N資格producer | A/B：採択契約に対する有効なrisk上側評価と入力移送。現在のMC summaryだけでは科学的十分Nを出せない。不足なら診断で停止 |
| 価格 | D：同law/compiler/control/full wrapper/lowering/P0の適格平均C、SOURCE、cohort/domain、数値/model/rare/law誤差、必要な同時区間。欠測を旧価格や構造Sで補完しない |
| 実行SOURCE | B：本adapter成果commit＋既存stats/R3/controller、確定入力とcaller・有限scope。source/datasetの変更時は元科学契約のまま継続しない |

L1経験的claim、十分N、global RMSE bound、価格の暫定精度/confidenceは未採択のまま。採択が必要な事項を技術flagで正式化しない。

## 3. 有限実行の二経路

### A：診断のみを先に行う

最大3配分に対し`diagnose`を各1回、事前固定n_screen/seedで実行する（最大3 simulator呼出し）。同じ1組の供給signalを再利用し、全trial/status、conditional RMSE、未解決率/CI、解決分MSE、prior外、入力資格を保存する。

lossが未定義又は入力移送未資格なら、`n_packet=None`を維持する。`compare`はG=UNKNOWNで返す。点RMSEが小さいだけで配分を十分N/正式winnerにしない。screen後の配分追加・新CI再解析・増分runは別有限scopeへ戻す。

### B：N資格と費用比較まで許可する

有効な全事象risk法と資格producerが準備できた場合に限り、次のどちらかを採択する。

- **事前有限menuの同時risk評価**：最大3配分それぞれに事前α/精度を配分し、各固定nの評価を一度ずつ実行。最大3 simulator呼出し。採択した入力移送・有限K/alias/target等も含むN packetを生成する。
- **screen＋独立確認**：最大3配分をscreenし、選抜規則を事前固定して1配分を独立seedで1回確認する。最大4 simulator呼出し。確認失敗後の別配分への乗換え・再試行はしない。この方式で資格化するのは確認した1配分で、全3配分の最小性は認定しない。

既存Hoeffdingの有限反復数は、採択したsupport U・MSE幅η・failure αから`trial_budget`等で決める。MCの反復数と量子shot Nを混同しない。rare-loss制御が非実用的ならINCONCLUSIVEで止め、別の解析risk法を人工検証する技術scopeを先に用意する。全13roundのcount直積列挙を代案としない。

Gの実算術・正式選抜を許可範囲へ明示的に含める場合だけ、資格付きNとDのCを`compare`へ渡す。最大3配分の費用評価、同じ価格を使った差分区間、1つの有限menu結果に限定し、他physical candidate/M/κや新C取得へ進まない。CIが重なれば点最小と優越不明を別表示する。

価格欠測ならUNKNOWN。N又は価格domain/全law/共同confidenceが不足する場合も、その科学資格を維持した条件付き結果に限定する。未探索配分・未検証方式へのglobal最適性は主張しない。

## 4. 保全とSTOP

新run directoryにconfiguration/whole allocation/mean/target/統計契約とSOURCE参照、全trial、summary、N証拠、元価格とextension/区間、coverage/未探索域を保存する。過去raw・結果・grant・ledger・marker・KEEP・STOPを上書きしない。他担当のworktree/processへ介入しない。

SOURCE/candidate/law/profile/target混在、欠測、数値範囲外、backend上限、loss/support違反、decoder逸脱、保存不能はSTOP。zero/tie/誤branchは元規則で会計し、捨てて再sampleしない。精度未達を理由に反復増・再seed・N変更を自動追加しない。

今回の技術成果は公開・remote照合後mandatory STOP。次は**ユーザー**のclaim/有限scope採択と実行許可、**A/C**の入力供給、**D**の価格資格、**B**の許可された評価。実N/G最適化には自動進行しない。
