# 最初の追加検証1案：LD1固定scheduleの残差モデルをm4へ移送できるか

**提案のみ。未採択・未承認・未実行。** 最初の問いは「保存P2 fresh m0/m6と新m3を使ったC−Sの3係数モデルが、同じLD1固定scheduleの未開封m4価格を予測できるか」。全law E[C]や全RPEの費用優位をこのpilotだけで認定する計画ではない。

## 入力・役割・仮の件数

同じH4 DF10/LD1/S2/Δ.1/K2、S0 exactκ、保存H/state/snapshot/preparation、fragment/basis順、新ZZ・scalar c+s_R一回・元compiler/control/full measured X/Y/P0を維持する。

| 点 | physical ID | q/R | 使用役割 | path数 | 新規compile |
|---|---|---|---|---:|---:|
| m0 T.1 | 53e00d8b07d1547d19a7 | 1/2 | 保存P2 freshのDEVELOPMENT、較正入力候補 | 保存4 | 0 |
| m6 T6.4 | 2a8c6718220eec59d419 | 64/128 | 保存P2 freshのDEVELOPMENT、較正入力候補 | 保存4 | 0 |
| m3 T.8 | 045843b2429bece6f29b | 8/16 | 新normal較正cohort | 仮4 | **最初に8**（X/Y） |
| m4 T1.6 | 3df0cfd3356c358aeda8 | 16/32 | 係数・予測・判定規則を固定した後の独立holdout | 仮4 | **後続8**（X/Y） |

保存P1の各2pathはdevelopment参照、P1 replayは再現性参照のまま。P2 freshへpoolせず、新holdoutへ読み替えない。保存P2価格/残差の新平均・fitは未実施で、次回の科学許可へ明示的に含める。Dの旧巨大m12 T2、C2の40/16wrapper、LD2取得は本案に含めない。

4pathという数は、既存P2の取得/paired実装を再利用するための**開発pilot仮定**である。標準誤差・CI・rare捕捉・価格精度を保証しない。件数の確定には、ユーザー/Bが必要なPR価格精度・意思決定marginと対象law/信頼水準を決め、有限cohort/rareの扱いを事前に選ぶ必要がある。現状は「初回8＋後続8」という仮の有限scopeであり、十分なnの算定結果ではない。必要nが大きければ別途範囲を決め直し、自動増数しない。

## 既存実装と凍結順序

1. Cの本受入索引から元P2 freshの個別C/S、同path/axis/digest/cohortを読む。既存`price_model_validation`又はC1 multiresponse fitの(1,q,qR)処理と、`rpe_mean_price_model`の`STRUCTURAL_PLUS_RESIDUAL`経路を再利用する。新frameworkを作らない。
2. **m3取得前に**固定3較正セル・m4 holdout、式`E[C]候補 = 保存E_I[S] + θ0 + θq q + θL qR`、X/Y paired応答、cohort/seed、SOURCE、誤差指標・判定規則を固定する。X/Yの値が等しくても別応答を保持する。正規化Bは価格へ掛けない。
3. Dがm3のnormal pathを元samplerで取得し、同pathのS/C・new ZZ/scalar/profile/axis receiptを保存する。既存m0/m6を再compileしない。3セルの識別可能性・rankは既存検査で確認する。
4. 明示許可された範囲でCが保存2セル＋m3を較正し、係数・m4のX/Y予測をcommit/pushで固定する。P1/replayによる追加補正やholdout混入はしない。
5. **凍結公開後だけ**Dが別cohort/seedのm4 holdoutを取得し、Cへ渡す。既存B/D consumerでSOURCEと独立性を照合して、固定済み規則を適用する。

これは固定κのschedule上の検証。qとRが連動するので独立したR最適化や任意κへの移送を保証しない。m4はm0/m6の間の移送点で、m7～m12の長q/R外挿は別の未資格domainである。

## law・誤差・実行前に残る判断

構造値はideal finite-K law I。Dのnormal cohortは元PCG64/NumPy CDF sampler Qで、SOURCE生成対応はd72dc4e6を再利用する。I-targetの混合推定には`E_Q[C−S]−E_I[C−S]`が残り、Q-targetには`E_I[S]−E_Q[S]`が残る。生成SOURCE一致やnominal event probabilityを、距離0・固定seed IID証明に読み替えない。effective categorical law/長trajectory距離はUNKNOWN。

保存のideal E[S]区間、有限cohortの標本誤差、係数/移送誤差、law systematic、希少order2、XY/cross-round共有誤差を分ける。order2未観測を0寄与とせず、必要なforced/conditional検証はnormal cohortと別unit・別承認scopeにする。今回それを追加取得へ自動展開しない。

実行前にユーザーが決める事項：

- 問いを有限cohortの開発予測検証に限るか、理想全law E[C]の必要精度まで要求するか。後者は本16wrapperだけで保証できない。
- PR用の許容誤差ε_PR、観測値/予測値/区間等の分母、X/Y・cohortばらつき・同時性・rareの扱い、PASS/FAIL/UNKNOWN規則。**決定論PFの2%や旧pilotの2%/95%は未採択で、流用しない。** 結果を見て変更しない。
- 仮n4を初回developmentとして採用するか、要求精度に基づく別の有限nを先に設計するか。cohort名・元trial_seed規則からのseed・除外/独立性metadataを固定する。
- 許可する科学call：保存P2個別C/Sの新しい集計と較正、新m3 normal4pathのsampling/paired S/C8wrapper、係数/予測固定、凍結後m4 normal4path/8wrapperと限定holdout判定。法則距離/rare/追加round/N/Gは別scope。

## STOPと到達できる結論

SOURCE/physical/scalar/law/profile/axis不一致、欠測、取得gate不成立、rank不成立なら部分結果を保存しUNKNOWN/STOP。holdout許容差を超えた場合は事前規則に従いFAIL/STOP。閾値緩和、refit、追加path、別law/lowering、他roundへの自動retryはしない。

成功しても「指定cohort/固定domainでの予測検証を通過した」という限定証拠。全law平均価格、長q/R、正式N/G/winnerを自動的に資格しない。Bの固定配分のQ3モデル適合は参照できるが、今回の価格/残差実行の承認や十分N資格ではない。

**引渡し順：C metadata受入（完了）→ユーザーによる規則/n/scope採択と科学許可→D m3取得→C較正・係数/m4予測凍結→D独立m4取得→限定判定・mandatory STOP。** D1/D2/C2の完了待ちは置かない。
