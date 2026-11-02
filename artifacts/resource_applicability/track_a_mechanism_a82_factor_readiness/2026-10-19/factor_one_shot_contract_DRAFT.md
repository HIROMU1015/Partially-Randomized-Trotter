# A82比較説明因子専用one-shot契約案（実行未承認）

SOURCE `06c94745f50345b2d93714760608d9d11b83e784`。固定contract SHA `fe6300e3320bff41c4131873fbdc8b5cbe048013bf2b1060469eaa50b6c36ea0`、input binding SHA `5696bfcdfacf6be2bdd1b89f17e504c11629b649cf8766e61b03b26a7931c50a`。
正本は専用local audit内のA82_factor_one_shot_contract_DRAFT.jsonとA82_factor_input_binding_DRAFT.json。正本には実入力・未使用出力先・runtimeのpathとSHAを固定し、この公開候補文書にはprivate絶対pathや完全hash一覧を転記しない。

対象はA82_OLD_PRの1比較、H8 linear1Å/STO-3G/tol-only DF28/T=.8/P=0、通常signed/factor1、ε=.05、PR B2_p7_q2_2nd_r4_K2対PF B0I_p11_q1_2nd、旧confirmation union n32。独立n48とは別cohort。

前段182d7044…で正常完了したA82.PRと、同deltaにexact copyされた既存PFを読む。元core32と説明archiveはID・status・SHAで継承を確認するだけで、数値式を再評価しない。SOURCE a8961c45…のPR検証自体も再実行しない。

## 許可を求める新規計算

cosine/sineの各軸について、既存31比較と同一定義・同じbinary64演算順序の7因子と対応log因子だけを生成する。

|因子名|既存定義（p=保存PR、d=保存PF）|
|---|---|
|nominal_normalization_squared|exp(2×(ell_nom,p−ell_nom,d))|
|normalization_margin_squared|exp(2×(ell_margin,p−ell_margin,d))|
|normalization_upper_squared|exp(2×(ell_upper,p−ell_upper,d))|
|headroom_squared|exp(2×(log h_d−log h_p))|
|alpha_log|exp(log(log(2/alpha_p))−log(log(2/alpha_d)))|
|continuous_shot|exp(log_Nstar,p−log_Nstar,d)|
|rounding_factor|exp(log_kappa,p−log_kappa,d)|

保存側のnormalization、signed bias/headroom、Nstar、kappa、保存shot、ceilを再計算しない。費用・shot総数比・有効単価比・SE・ランキング・統計を新計算しない。既存mechanism_explanation.pyのexplain()はside再評価を含むため起動せず、その因子blockだけを新adapterへ移した。人工テストでblockのASTと演算順序、人工結果のexact一致を確認した。元3科学fileはb6716396…とbyte一致。

## grant・実行・終了

今回のgrant案はapproved/approved_by_user/runtime_authorizationがすべてfalse、有効grant発行0。実CLIは契約・科学入力の読込前に拒否された。旧side検証grantはscopeが異なり再使用できない。実行には今回のSOURCE、契約SHA、binding SHAに結合した別のユーザー明示承認receiptと新grantが必要。

実行前後にSOURCE5 file/Git blob・入力14件・Python/runtime68 file・元core/説明/delta・契約・新grant/receipt・新marker/ledgerを照合する。未使用outputへ排他的なone-shot reservationとworker claimを作り、親はworkerのnative exit・wait/reap・出力SHA/schemaを検査する。科学factor passは1回、対象軸は2つ。他31比較とPR/PF sideの式呼出しは0。

因子は各軸7個の正の有限値、log因子は有限値を要求する。必要な保存fieldの不足、SOURCE/input/runtime/control不一致、gate不通過、非有限/表現不能出力、native異常はfail-closed STOP。retry/refund/自動次stageなし。型やthreshold/toleranceを調整して結果を通過させない。

出力はA82_factor_delta.json、inheritance_receipt.json、worker_result.json、native_wait.json、terminal.json。新control3件はONE_SHOT.json、ledger.json、WORKER_CLAIM.json。正常時は計8 file。失敗時はSTOP.jsonも保存。既存ファイルは上書きしない。

今回の状態は説明side64/64、比較因子31/32のまま。まだ32/32完了とは報告しない。数値実行承認後、PASSした保存deltaだけを[統合計画](integration_plan_AFTER_PASS.md)に従い別記録へ加える。Figure 2数値/cohort変更、最終freeze、詳細科学レビューは今回のscope外。
