# H4/H6高次PF PR本探索 run01：取得・技術終了監査

科学SOURCE `5a448d94ea389ada6e2b2755de6a215ed2034679`。承認済みの新規EXPLORATION単価取得だけを実行した。

対象は直鎖1 Å/STO-3G、tol-only `1e-8`、H4 rank10/prefix3・H6 rank19/prefix5、保存Hamiltonian/状態、T=0.8/P=0、S2/Yoshida4/m5_best、q=1/2/4/8、K4/outer32。既存compiler/profile・native control・検証閾値を維持した。

| 系 | 状態 | 完了trajectory | 受理compile軸 | native exit/reap |
| --- | --- | ---: | ---: | --- |
| H4 | ACQUISITION_AND_END_AUDIT_PASS_MANDATORY_STOP | 793/793 | 1586/1586 | 0/True |
| H6 | ACQUISITION_AND_END_AUDIT_PASS_MANDATORY_STOP | 607/607 | 1214/1214 | 0/True |

両軸native受入とbyte完全復元を別々に監査した。途中STOP・欠測をPASSへ読み替えない。

[公開技術summary](../../artifacts/resource_applicability/track_a_h46_pr_exploration_run01/2026-10-21/run_summary.json)に、各契約・結果SHA・archive容量・wall/RAM/scratch・終了状態を保存した。peakは定期観測値で厳密な上界ではない。

MCSE目標2%・分散倍率4・群別下限33は保存pilotに基づく設計で、未観測S4・母平均費用・分散の保証ではない。pilot・独立確認との混合はしない。

新しい分散/MCSE評価、shot計算、費用再会計、順位、独立確認、AD03採用は実施していない。既存結果・grant・ledger・marker・STOPを保全した。

**mandatory STOP。** 次の担当：ユーザーが取得結果を確認し、新しい探索標本の限定統計評価を別途承認するか判断する。Codexは承認前に統計・会計へ進まない。

## 文書保存処理の復旧

両系の科学取得と終了監査が完了した後、公開資料commitの検査が、更新した索引に元から存在する内部絶対パスに反応して停止した。元の`PUBLIC_TECHNICAL_PROJECTION_ONLY` receiptは保持し、新規成果物全体と既存文書の追加行を検査して文書保存だけを復旧した。封印済み科学SOURCE、raw、契約、grant/ledger/marker/STOP、結果は変更せず、科学retryは0件。H2の独立した取得・監査結果も変更していない。

[構造検証](../../artifacts/resource_applicability/track_a_h46_pr_exploration_run01/2026-10-21/structure_checks.json)は、既存の契約・終了監査・終了witness・公開転記の対応を確認した作者自身の検査であり、独立科学レビューや新しい統計解析ではない。
