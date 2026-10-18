# 強い対照：今回閉じた点と残る不足

[入口](README.md) / [前回の不足表][PGAPS] / [前回native再利用分類][PREUSE]。新driverの追加はB3の4方式だけであり、Closed P5/CTSへの拡張は行われていない。基本数式・旧ranking・19 keysを調べ直していない。

## 1. 既存証拠だけで閉じる範囲

G9 v2の登録P5・同一3-qubit/I1 provider・direct/compiler・8ρ/22-axis policy内の既存matched比較はそのまま引用可能。Closed P5のT-primary、matched CTSのCX利点とトレードオフという保存結論は、前回の[既存結果引用][ABASE]を参照する。今回新しい差・比・順位は計算していない。

19 RZ key／18 G7由来sequenceのstrict-phase・actual adjoint・tool/options・hash条件も前回の固定分類を維持する。新driverはそれらを呼ばず、旧価格schema・補助SOURCEも変わらない。既存G9価格の引用・同identity資産のmetadata結合だけなら新compile/synthesis不要。

## 2. B3取得後にも残る不足

| 不足 | 既存物 | 新B3で埋まらない理由／必要時の最小作業 |
|---|---|---|
| Closed P5の同等全trial古典費用 | 既存`P5Closed.sample`、DP/group法、形式確認 | driverのARMS/Group snapshotは旧4方式専用。新scheduleと同等audit・angle・保存への接続は別途必要 |
| CTSの同等全trial古典費用 | 既存Pauli収集、finite event law、native会計 | production draw/監査/全N accumulator・情報取得/setup費用が未取得。G9の参照リスト会計をμとして使わない |
| 強い対照にも公平なschedule | 新B3の24-constructor・X/Y共有・2pass・保持案 | FullのGreen/F3処理を対照へ強制せず、必要なaudit証拠基準と実課金を揃える。片側だけ無料cache/弱いselectorにしない |
| 単一policyでのN/総native | G7旧NとG9別policyの保存値 | 共通policyを新採択しない限り直接統合不可。旧G9内部比較の引用で足りればP2不要 |
| 新C1の具体provider/価格 | G9固定provider/event価格、旧C1 schema | B3は物理CQを選定しない。event/primitive粒度、ID registry、phase/adjoint、error0/2ε、compiler/precisionのbindingが別途必要 |
| control/direction/word/angle別縮約 | G7 label合計係数、G9 event rows | 新価格の粒度が保存summaryを超える場合のみ追加B2/P2を検討。既存G9同scopeの値を引用するなら再縮約不要 |
| depth | 保存T/CX/1Q、gate列 | 保存depthなし。gate数から推測しない。研究判断に不要なら取得しない |
| C2総費用 | prep/readoutの旧共通非負parameter | 実state preparation/readout/backend、E2E estimator等は未取得。未知を0にして閉じない |

旧[PCROSS][PCROSS]・[PGAPS][PGAPS]で確定した不足であり、今回4方式の接続が具体化しても強い対照までの監査完了にはならない。新adapter/価格pool/契約を今回作って補わない。

## 3. 新native取得が必要になる条件

新Q/encoding/control/ancilla、RZ angle/precision、gate set/compiler/合法最適化、または必要な価格座標が変わり、同じcontextを支える保存証拠がない場合。その差分が研究判断に不可欠と採択されたときだけ、対象key/価格を最小化して取得を検討する。

metadata binding不足を理由に全native再compile、同じ19 keyの再合成、旧数学構成の再検証を要求しない。depthも新しい導出/会計は別scopeで、今回は実施しない。G9 providerを新C1に正式採択する権限も本監査にはない。

## 4. 最小の判断分岐

既存G9比較で判断できるなら追加取得なしでよい。4方式の古典実費が必要なら、F01技術修正・正式採択・外部admission・別途B3実行承認が先。強い対照までの古典費用が判断を変える場合だけ、ユーザーがP3の必要性を判断する。

P2/P3・新C1価格・C2を不足の数だけ自動推奨しない。B3が完了しても一般的資源優位や研究GOを自動判定しない。**mandatory STOP。**

<!-- FIXED_REFERENCES -->

[ABASE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ff0bed24ba8fdb361a2890646ef84347ae37d457/docs/research/track_b_g7_g9_strong_comparator_audit_20261019/existing_strong_baselines.md
[PCROSS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/568bf486805dfbd2fd4a13ef5a25771351981ce2/docs/research/track_b_g7_g9_p1_public_crosswalk_20261019/b3_c1_g9_identity_crosswalk.md
[PGAPS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/568bf486805dfbd2fd4a13ef5a25771351981ce2/docs/research/track_b_g7_g9_p1_public_crosswalk_20261019/classical_strong_comparator_gaps.md
[PREUSE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/568bf486805dfbd2fd4a13ef5a25771351981ce2/docs/research/track_b_g7_g9_p1_public_crosswalk_20261019/native_evidence_reuse_matrix.md
