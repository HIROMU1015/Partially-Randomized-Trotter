# Decision Handoff：P1先行照合の到達点

[入口](README.md) / [native再利用](native_evidence_reuse_matrix.md) / [古典費用不足](classical_strong_comparator_gaps.md) / [P1-final差分](p1_final_delta_checklist.md)。状態：**`P1_PRELIMINARY_PUBLIC_PREPARATION_CROSSWALK_COMPLETE`**。

## Q1. 公開準備とG9 v2のidentity照合はどこまで完了したか

固定`c8cbeb18...`の契約候補・補助SOURCE・3 schemas・再利用表と、`ff0bed24...`の監査／必要なG9固定fieldを静的照合した。数学的P5 anchor、G7の旧4方式/lawと保存N継承、Layer A/B、audits、cache、価格key・粒度・control/phase・費用座標の対応と不足を確定した。[identity表](b3_c1_g9_identity_crosswalk.md)

**将来driverの実装・採択schedule・物理providerは未照合。実入力correctness・profiling/native取得も未実施。** 公開準備の人工48 PASSを今回のテストや実G7成功としない。旧G7/G9基本監査、形式構成、native取得、先行研究調査を繰り返していない。

## Q2. G9価格・cache・matched比較を再利用する条件

- **既存結果の引用**：G9の固定P5・3-qubit I1/direct・22-axis/8ρ/resource policy内ならそのまま利用可。Closed P5のT-primary、CTSのCX利点とトレードオフは前回引用資料を継承する。[ABASE][ABASE]
- **RZ資産**：同じfull angle key/scale/ε、strict phase、tool/options・sequence SHA、W込みactual adjointなら19取得列は再利用候補。18列のG7来歴も確認済み。旧wrapper費用は移植せず、新合成は不要。[G9KEY][G9KEY] [G9MC][G9MC]
- **新C1との結合**：具体provider/実装/最適化/価格scopeが一致し、旧event価格と新primitive schemaのbindingが閉じること。G9 whole-event価格は完成した新primitive poolではない。depthは欠測のまま、state/backendもnullを維持する。

今回G9を正式providerに採択していない。同じ保存証拠のmetadata bindingや引用だけなら、新compileを必要作業にしない。

## Q3. Closed P5・CTSの公平な古典比較に足りないもの

Closed P5には既存sample API・DP/group処理、CTSには既存Pauli収集・finite digital event lawがある。しかし新B3 adapterのARMSは4方式限定で、P5Closedのgroup表現／CTSのPauli・identity eventはそのまま非対応。CTSのproduction draw、両方式の等価audit/angle/保存・全N費用、setup/cache/X/Y共有scheduleが不足する。

B3の4方式計測が完成してもこの不足は埋まらない。G9の参照/native会計CPUを母期待μやproduction費用へ転用しない。固定workload記述費用と期待費用の統計保証のどちらを問うかも採択が必要。[古典費用表](classical_strong_comparator_gaps.md)

## Q4. Codex A公開後に必要な追加確認

新driverの公開固定commitと`c8cbeb18...`からの差分だけを確認する。D01〜D10はSOURCE/policy、schedule、X/Y/cache共有、angle、call/STOP、provider/price binding、費用scope、採択/承認/保全に限定した。[チェックリスト](p1_final_delta_checklist.md)

これは未公開driverの完成を推定するものでも、将来チェックの自動実行承認でもない。今回Codex Aのlive worktree/index/lock/process/未commit成果を調べていない。

## Q5. 新計算・compileなしで判断に渡せること

既存G9の強い対照比較、19 RZ資産の再利用条件、schema結合の不足、追加古典費用が必要となるscope、将来確認の最小差分を渡せる。**Fullの一般資源優位、実CPU優位、QPE/RPE・C2総費用、独立新規性は認定しない。**

## 次の作業を増やさないための分岐

| 問い | 最小の扱い |
|---|---|
| G9既存matched結論を引用したいだけ | P0で閉じる。P2/P3/compile不要 |
| 将来B3/C1公開版と結合可能か | 別途指示後、CodexがD01〜D10の変更差分だけ静的照合 |
| 旧literal価格と同粒度か | 保存係数を引用可能。新価格による期待費用評価は別途承認 |
| role/direction/word/angle価格・whole-event cancellation・非加算depthが旧summaryを超えるか | 追加B2縮約が必要となり得る。旧G9 event rowsで既に閉じるなら再取得不要。今回は粒度差の特定まで |
| G7/G9を単一policyの新N/総量にしたいか | bias/confidence/resource配分・axesを採択するP2が別途必要。G7の3ρをCTSへ勝手に拡張しない |
| 強い対照の古典費用も研究判断に必要か | 必要性をユーザーが判断した場合だけP3の同等入口・production/保存と計測を別承認で検討 |
| 新provider/precision/compiler/座標の価格が保存証拠を超えるか | その差分が問いに必要な場合のみ新native取得を検討。metadata不足だけで全compileや全19 key再合成を推奨しない |
| End-to-End総費用を比較したいか | C2の実state/prep/readout/backend・費用規則が不足。未知を0にして閉じない |

**ユーザーが採択する必要がある選択**は、費用の問い（記述的workload／期待値）、層・setup/cache/output schedule、angle/error allocation、provider/情報アクセス、direct/helper/最適化・価格粒度、単一policyの必要性、C2を除く主張scopeである。本監査はこれらを確定しない。採択と数値解析・profiling・compileの実行承認は別。

最初の担当は**ユーザー**：今回の条件付き再利用・不足で研究判断できるか、将来公開後のP1-final照合を依頼するかを決める。依頼された場合のみ**Codex**が公開固定差分を確認する。新計算や重要レビューの開始へ自動的に進まない。

## 保全・STOP

旧G9 v1失敗、v2結果/25-check記録・別承認・消費済みmarker、G7/G8/F2/F3/W2b/W2c-A、全既存研究状態は不変。親tracked/root原本、直前の自分の公開監査checkout、既知consumed ledgerを開始/終了で照合し、基点全blobを保全した。他worktree全体やCodex A private状態を監査したとは主張しない。

成果物は新規文書・管理JSONのみ。新科学計算・保存値再解析・実装・科学SOURCE import/実行/変更・profiling・sampling・compile/synthesis・科学的採択・authorization/grant/marker/claimは0件。[管理検査](verification.json)と[公開照合](publication_receipt.json)完了後、**mandatory STOP**。

<!-- FIXED_REFERENCES -->

[ABASE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ff0bed24ba8fdb361a2890646ef84347ae37d457/docs/research/track_b_g7_g9_strong_comparator_audit_20261019/existing_strong_baselines.md
[G9KEY]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0ef2b92738750a3c0d187743dd4b7c9b927db802/artifacts/track_b_g9_p5_native_preparation/2026-10-10/synthesis_inventory_v1.json
[G9MC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0ef2b92738750a3c0d187743dd4b7c9b927db802/artifacts/track_b_g9_v2_api_boundary_preparation/2026-10-10/contract_v2.json
