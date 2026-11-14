# C受入：PR-DF LD1 m4独立holdout

2026-11-15 JST。**軸別receipt 8/8件受入・保留0、軸別比較2/2件受入・保留0。** 保存結果の読取・metadata照合のみ。平均・誤差・予測・残差の再計算、fit、sampling/compile、CI、N/Gは0。

正本は[ADMISSION.json](ADMISSION.json)。`candidate_rows`のX/Y各行から、旧CのALIGNMENT行、観測前のexact予測、D2の保存比較、4pathの元axis receiptへ戻れる。既存の全候補対応表・凍結JSONは変更せず、m4の差分索引として追加した。

## 固定SOURCEと読取手順

- Cの旧候補索引：`c99d511e2422b8f250490b11054f754a81e536c5:docs/research/pr_df_ld1_q3_price_admission_20261115/ADMISSION.json`。その`alignment`から`0b4b2d941…:…/ALIGNMENT.json`の元m4 X/Y行へ進む。
- 観測前予測：`3cac53da1a4a8b2228245593c74c1ff9ec1f5212:docs/research/pr_df_ld1_residual_freeze_20261115/run01/FROZEN_MODELS.json#models[axis].m4_prediction.predicted_RZ`。比較用の分子・分母を直接照合し、再予測していない。元HANDOFFは`890755f0a…`。
- D2最終結果：`063f2b3b6ea271a179a469be91252fa4e5a7d096:docs/research/d_pr_df_ld1_m4_holdout_results_20261115/{C_HANDOFF,COMPARISON}.json`。元科学実行SOURCEは`7bdfc6f629aa74a5c1347abf84ad6486c669c9d8`。
- 元receipt公開：`3042e6e66ad693ed514274334adecd3d1b42dc02:docs/research/d_pr_df_ld1_m4_holdout_results_20261115/run01/receipts/p0..p3/`。各C受入行の`original_record.axis_receipt/original_receipt/structural_receipt/path_result/gate`に完全locator・元hash・生成時SOURCEを保持した。

consumerは本`candidate_rows`を`(configuration_id,physical_id,m,axis)`で既存表へ対応付ける。`comparison_admission`は本JSONの比較受入へ、`receipt_admissions`は4件の受入へ進むselector。各比較受入の`comparison`はD2原比較の固定locatorで、`saved_exact_comparison`はP/O/誤差の原分子・分母のcopyである。生成時の旧価格configuration/canonical IDと、Cの元/total decoder IDを別々に保持し、IDを付け替えない。

## 対象・照合範囲

physical `3df0cfd3356c358aeda8`、m4/T1.6/q16/R32、H4 DF10/LD1/S2/Δ.1/K2。cohortは凍結時どおり`D_PR_DF_LD1_M4_HOLDOUT_NORMAL_20261115_V1`、fresh index0～3。各pathのX/Yは共有event列の別測定軸で、8独立pathとは数えない。

元H/state/snapshot・preparation、fixed-grid/κ/law/signed構造・fragment順、scalar c+s_R一回、新ZZ、元control・測定付きfull wrapper/P0・Qiskit1.3.0/rz,sx,x,cx/opt1/seed17を固定metadataで照合した。cohortと4×16の元seed表、same-path S/C・event digest・軸別fingerprint・compiler・receipt bytes/hash、D2保存の取得/identity gateを確認した。既存`price_implementation_guard.evidence`と前段`path_binding`を再利用し、guard/価格算術は変更していない。

`full_wrapper_statevector_validation=false`、`numerical_action_gate=NOT_RUN_AS_IN_ORIGINAL_P2_FRESH`を維持。取得/identity上の有効性を、新しい量子作用保証やモデルPASSへ格上げしない。

## 保存済みの比較（X/Y個別）

| axis | P exact | O exact | P−O exact | 符号付き相対差 exact |
|---|---|---|---|---|
| X | 217164876899444503/1683627180032 | 259385/2 | −1188941146855657/1683627180032 | −1188941146855657/218353818046300160 |
| Y | 217164876899444503/1683627180032 | 259385/2 | −1188941146855657/1683627180032 | −1188941146855657/218353818046300160 |

D2の表示値は両軸O=129692.5 RZ、P−O≈−706.178399206563 RZ、相対≈−0.544502110150211%。CはD2の保存値を転記しただけで、平均や誤差を再計算していない。絶対誤差・絶対相対差もADMISSIONの保存exact copyと原COMPARISONから参照できる。

比較状態は**`DESCRIPTIVE_COMPLETE`**、モデル判定は**`NOT_ASSESSED_NO_ADOPTED_TOLERANCE`**、許容差は**null**。原4pathのO定義、O=0の相対null、欠測時O/errors=null・n4保持を引き継ぐ。事後閾値、PF2%/旧pilot2%・95%、PASS/FAILを追加しない。凍結モデルの旧statusを上書きせず、記述的holdout結果の受入を別記した。

## 重複確認・技術検証・ロードマップ

着手時のC HEAD/公開refは`890755f0aeea3abfcdd5c849422bb198666a477e`で一致し、worktree clean。関連C結果path・公開ref・関連Python script名の確認範囲で同じC受入を認めず、新規差分だけ作成した。他担当のworktree/processには変更なし。

人工新5＋既存path/law受入回帰8＝**13 test method PASS**。[TESTS.txt](TESTS.txt)。初期の人工fixtureは同じdict/listを共有して拒否条件を意図せず先に壊していたため独立copyへ修正した。Pathの`/`を数値除算として検出したtestも比較関数内へ対象を絞った。受入実装や原科学値の変更・再計算はない。これは作者自身の技術確認であり、独立科学レビューではない。

ロードマップ差分：LD1 m4の「D2結果→C consumer受入待ち」は解消し、候補→凍結P→保存比較→元receiptが接続済み。I/Q差・小標本/rare・係数/共有不確かさ・全law E[C]/CI・長round価格・十分N/G/winner・モデル資格は未解消。ロードマップ本体は統合担当が更新する。

**mandatory STOP：commit/push・remote照合後に終了。追加取得/fit/閾値設定/科学再解析へ進まない。Full-H D7の受入・fitはD1最終引渡し後の別作業。**
