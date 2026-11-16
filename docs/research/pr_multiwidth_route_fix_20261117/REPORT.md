# PR複数幅探索：候補とQ3実行先の接続修正

**R06を止めたPauli LD4の未登録routeと、DF LD6の不適合template選択を修正した。** 保存入力を元の共通workerに結び付け、両候補ともmetadata専用dry-checkがPASSした。人工・回帰72件PASS。新規H4信号・sampling・compile・fit・実Q3はすべて0件。R06の停止状態と原結果は変更せず、探索は再起動していない。

## 原因と変更

R06の最終停止は2026-11-17 04:21 JSTの`V2_RESULT_REQUIRES_REVIEW`。Pauliは`NO_ROUTE_ADAPTER`、DFは`UNKNOWN_RESULT`だった。元runは`../h4_pr_multiwidth_optimization_20261117/run01/`。公開e0f7838b09bc09412ec764d80686cb9578e44d01は稼働中snapshotで、最終停止を示す記録ではない。[STOP_CONTEXT.json](STOP_CONTEXT.json)に今回読み取った停止metadataと元fileのhashを保存した。

|対象|原因|修正|
|---|---|---|
|Pauli LD4、`8bf6eb88…`|catalogに価格・信号があるが、roundの9構成用実行先一覧に含まれない|configuration、packet、case、canonical binding、target、adopted evaluationを照合し、既存`pauli_b_common_entry`へ候補専用に登録|
|DF LD6、`5bebe691…`|`prdf_d2_split`の`configuration_ids=*`が対象外の構成も受け付け、template検索で例外|template内の実際のIDを確認。保存候補自身のcase/bindingから既存`prdf_d2_common_entry`へ登録|
|Pauli信号準備|保存LD1/2/3/5のindex/context locatorをcommit/path型のoverrideとして送付|元の固定A entryが持つ保存descriptorを使う。派生prefixはcommit/path envelopeを必須とし、不明なSOURCEは拒否|
|Pauli依存file|固定A entryからimportする2 scriptが当初の配備から欠落|既に配備済みの同一SOURCEの2 scriptも既存の起動前SOURCE一致検査へ追加|

`pr_candidate_completion.py`の既存`install()`を拡張し、新規準備候補と保存catalog候補で`common_route()`を共用した。別構成の信号・価格・bindingは補わない。既に有効なrouteは変更せず、旧job keyと再利用判定を保持する。

実行先がまだない候補は`execution_needs`と既存の`CONFIGURATION_INPUTS_MISSING`へ返す。保存適合点は保持し、その候補の新規投入だけを保留して他候補を進める。入力要求が残れば幅driverは`WAITING_FOR_INPUTS`とし、探索範囲を尽くしたと誤表示しない。V2の候補順位・shot探索・価格計算の式は維持し、legacy選択関数も実行可能性guard以外は元SOURCEとAST一致を検査した。数値gate失敗、曖昧な稼働job、保存不能のUNKNOWNを停止する条件は維持する。

## 検証

- [TEST_GROUP_0.json](TEST_GROUP_0.json)：38件、[TEST_GROUP_1.json](TEST_GROUP_1.json)：34件、すべてPASS。実装担当自身の検査であり、独立レビューではない。
- 同じ追加11件は本番native Python 3.11.0rc1でもPASS（[NATIVE_TESTS.json](NATIVE_TESTS.json)）。検査はcommit前の作業treeで実施し、検査したcodeのblobは[TECHNICAL_SOURCE.json](TECHNICAL_SOURCE.json)で公開版に照合する。CIや外部再現の証拠ではない。
- [追加テスト](../../../tests/test_pr_execution_route_binding.py)：wildcard template不一致、SOURCE/target/binding不一致拒否、descriptor形式、依存欠落の起動前拒否、保存最安点保持、他候補継続、通常の複数幅CLI→共通queue→人工応答→再開0起動。
- 既存の複数幅準備、準備とQ3の重なり、queue/予算、prefix-cache設定、通常起動・再開、UNKNOWN/STOP保持の人工回帰を再利用した。
- [SAVED_BINDING_CHECK.json](SAVED_BINDING_CHECK.json)：固定catalog fee61d272の停止2候補を再結合。修正前は両方ともconfigurationをカバーしない。元のconfiguration/packet/case locatorとbindingを維持し、価格・MSEは再計算していない。
- 本番と同じadapterの`--mode dry-check`で[Pauli](metadata_preflight/PR_PAULI/check01/DRY_CHECK.json)・[DF](metadata_preflight/PR_DF/check01/DRY_CHECK.json)ともPASS。native環境はPython 3.11.0rc1、DF Acceleratorの元AST guardもPASS。stageは`JOB_START → CASE_BUILT → METADATA_PASS → JOB_END`で、`SCIENTIFIC_ROUTE_START`は0件。

共通concurrent queue、DP prefix-cache、SOURCEに基づく結果再利用を維持した。実H4の新規計算での性能や完走は未検証で、速度倍率は測定していない。

## 保全と残る条件

停止済み実行checkoutのローカルc8bcb3a65を基点とする独立branch `codex/pr-multiwidth-route-fix-20261117`で作業した。元の実行worktree、固定round05、R06の入力・raw・価格・STOP・失敗jobは編集していない。修正PASSを科学的成功や自動retryの許可へ読み替えていない。

初回pushは、その基点の未公開auto-commit履歴に約118 MBの`events_PRIVATE.json`が含まれていたためGitHubに拒否された。raw・履歴・branchは保全し、公開済みe0f7838b09bc09412ec764d80686cb9578e44d01から技術差分だけを`codex/pr-multiwidth-route-fix-public-20261117`へ複写して公開する。両基点の`src/scripts/tests/configs`には差分がなく、公開版の修正codeは検証したblobと一致する。履歴書換え・force push・raw削除は行っていない。

一部DF準備には`Conflicting saved means: select explicit saved inventory`もある。固定DF entryのinventory統合で検出された競合で、今回その数値・出所の選択や検査緩和はしていない。該当候補は保留し、復旧時には対象の保存inventoryを明示して整合を確認する必要がある。対象全域の入力が揃ったことや、未評価幅の最適性は主張しない。

次の復旧は[HANDOFF](HANDOFF.md)の条件に従う。**本成果は技術修正の公開で停止する。**
