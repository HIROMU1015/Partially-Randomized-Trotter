# 共有価格storeの保全と保存検証の再利用

2026-11-18。Codex sub-agent `price_store_acceptance` の技術照合。独立科学レビューではない。
新規実信号・実compile・実Q3・fit・価格資格変更は0件。

正本はRomuTailの `.rpe-central/price-validation/models.sqlite3`。
[読み取り監査](READONLY_AUDIT.json) では13モデル（保存状態 `INDEPENDENT_MEASURED` 10、`PENDING` 3）。
旧server2 snapshotは独立実測9、待機3、未完了1であり、DF7の保存受入とparser metadataがローカルで進んでいた。
この10件は以前の保存受入済み状態であり、今回新たに独立検証した件数ではない。
全モデルのID・凍結identityは旧snapshotと一致し、今回確認した実storeのjob/owner/attempt/external jobは0件。
既存の待機3件は選定済み代表点を保全し、compile予算0のまま取得していない。

live storeはSQLite read-only接続から専用snapshotへbackupした。snapshot本体と作業コピーは
`.rpe-central/validation/price-store-integration-20261118/` に保持し、Gitへ公開しない。
[旧snapshotの反復再取込](REIMPORT_COPY_CHECK.json) はこの**コピーだけ**へ2回行った。
13件retained・新規0件、全recordのbyte値とローカルの前進したDF7状態は不変だった。この初期監査・コピー検査でのlive storeへの書込みは0件。

c4の取込helperは1件の競合で全取込をabortしていたため、新しい要求の「該当項目だけHOLD」に合わせて最小拡張した。
競合するjob/owner/attempt/凍結要求は元recordをそのまま残してreportへHOLDを返し、無関係なモデルの取込を続ける。
部分成果や進んだ状態を古いsnapshotで上書きせず、同じモデルから別jobを作らない。
返す `states` は取込先の実状態、`source_states` は元snapshotの状態として区別する。
[人工・保存metadata検査](HANDOVER_TESTS.log) は9件PASS。保存DF7の凍結係数不一致拒否、
外部attemptの競合HOLD、UNKNOWN/部分成果の保全を含む。科学workerは呼ばない。
通常の `Service.observe` も、凍結identityが一致しないモデルだけを保留し、無関係なモデルを引き続き受け入れる。
保留中のモデルで新規dispatchはしないが、既存の凍結ownerによる同じjobの回収は維持する。
[価格検証の既存suiteと追加検査](PRICE_ACCEPTANCE_TESTS.log) は49件中48 PASS・1 SKIP。
SKIPはこのhostに旧local0 deploymentがないための旧bootstrap検査であり、Windows通常入口は最終配置で別途確認する。

[LD2共有store再利用](SHARED_SAVED_LD2.json) は、実store snapshotのコピーと通常の
`run_rpe_method_optimizer.make_executor` / `Service.observe` を使用した。
異なるrun・shot配分を表す人工contextから同じ保存LD2モデルを参照し、共有13モデルと各runのapplicationを公開できることを確認した。
optimizerは人工代替、compiler dispatchと保存証拠の再scanは明示的に禁止して実行した。
既存recordとSOURCEは不変で、新規compile要求・Q3・信号は0件。
人工価格contextはCや通常の科学結果へ公開していない。

再現コマンド（保存snapshotを読み、内部で一時コピーを作る）：

```bash
python3 -B docs/research/rpe_price_release_integration_20261118/store/check_shared_saved_ld2.py \
  --code-root /ABS/FIXED_INTEGRATED_CHECKOUT \
  --source-store /ABS/READONLY_SAVED_STORE.sqlite3 \
  --output /ABS/TECHNICAL_RESULT.json
```

最初の実装中検査は `implementation_worktree_changed=true` として記録した。
最終固定版の検収では同じ入口を固定checkoutへ向け、結果を区別する。

共有storeのLD2は、m5・m6・m9の保存観測・予測・元receiptが既存Cと一致する一方、q/Rだけがnullだった。
旧snapshotはm9以前のため復元せず、[コピーでの前進補完](SAVED_POINT_METADATA_COPY.json) を検査した。
固定 `c1f95e080` PriceRulesの `evidence_points` と各40桁commitの元receipt `request.m/q/R` を照合すると、
m5=(32,33)、m6=(64,66)、m9=(512,526) のみを補完できた。現在の適用先 `points` は参照しない。
`price_validation.enrich_saved_evidence_metadata=true` のときだけ通常observerが欠損q/Rを補完する。
非null値、観測、予測、係数、parser版、資格、owner、attempt、部分成果には触れず、根拠不足・不一致はそのモデルだけHOLDにする。
この検査は専用コピーの13モデルが既存Cの科学metadataと一致し、元snapshotの全record byteが不変であることを確認した。
live storeの補完はこの検査では実施していない。

再現は次のとおり。結果の `copy_only_updates` は実storeへ自動適用しない。

```bash
python3 -B docs/research/rpe_price_release_integration_20261118/store/check_saved_point_metadata.py \
  --code-root /ABS/FIXED_INTEGRATED_CHECKOUT \
  --source-store /ABS/READONLY_SAVED_STORE.sqlite3 \
  --publication-repo /ABS/EXISTING_C_CHECKOUT \
  --output /ABS/TECHNICAL_METADATA_RESULT.json
```

[既存Cから保存receiptへの参照](C_SAVED_MODEL_CHAIN.json) はC `ca5905e79` の13モデル・11 run、
固定receipt selector 42件を確認した。LD2の全receipt URLは既存HTMLにも存在する。
Windows実閲覧ファイルの以前の検収はc4の `LOCAL0_MAP_CHECK.json` を出所付きで再利用し、
今回のWindows実ファイル確認を行ったとは扱わない。最終配置での通常同期結果は上位REPORTに記録する。

## 最終固定版での確認と欠損metadataの前進補完

最終固定SOURCE `e917165ba1f36b7627614cd25e88b5108d9b20b9` のclean checkout
`.rpe-central/releases/e917165ba-price-fleet` と、SERVICEと同じ `.venv311/bin/python` で検査した。
[補完前の通常入口LD2再利用](final-e917165ba/SHARED_SAVED_LD2_CLEAN_BEFORE.json) と
[固定版でのコピー検査](final-e917165ba/SAVED_POINT_METADATA_COPY_CLEAN.json) はPASS。
どちらも `implementation_worktree_changed=false`、新規compile・Q3・信号は0件。

主担当の明示した作業範囲に従い、[liveのonline backup](final-e917165ba/ONLINE_BACKUP.json) を取得後、
コピー検査と同じ固定SOURCE・receipt照合を行って[欠損6項目だけをliveへ補完](final-e917165ba/LIVE_METADATA_FILL.json)した。
`ModelStore.change` の単一transaction内で、現在rowがbackup時のrowと完全一致し、
各pointのq/R以外の全fieldのdigestと固定receiptのrequestが一致することを照合した。
変更したのはPAULI_LD2のm5・m6・m9のnull q/Rだけ。非null値の上書き、snapshot復元、再取込は行っていない。
係数・観測・予測・価格資格・status・owner・attempt・job・部分成果をすべて保全し、残る12モデルはrecord byteも不変。
モデル13件（保存実測10・待機3）、job IDなしの状態は補完前後で同じ。再適用で追加変更は0件。
前後のSQLite backupは専用validation領域に保全し、公開JSONには実pathとhashを記録する。

[補完後の通常入口LD2再利用](final-e917165ba/SHARED_SAVED_LD2_CLEAN_AFTER.json) もPASS。
2つのrun・shot配分contextで保存recordをそのまま再利用し、新規証拠scan・compile要求・Q3・信号は0件。
通常 `Service.export` から出た共有sidecarも、[採用済みCと全13モデルを照合](final-e917165ba/NORMAL_SHARED_EXPORT_CHAIN.json)した。
保存receipt参照は延べ48件・異なるselector42件のまま。人工モデル・人工評価結果はこのexportに含めず、適用情報は0件。
このmetadata補完は新たな独立科学検証として数えない。科学実行と本番探索は0件で、mandatory STOPを維持する。

元のc4検収の出所は
[`8ba82646e` REPORT](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8ba82646eae8a013b58a956bb10995c3c9451f86/docs/research/h4_price_validation_local_controller_20261118/REPORT.md)。
保存LD2通常method CLI、価格worker a18の2要求×稼働3host metadata、既存C表示の検収は再実装せず再利用する。
本統合で変更していないworker/runtimeの実科学処理を一律に再実行しない。
