# H6 DBA：限定並列検証

[報告・範囲・限界・技術STOP](../../../../docs/research/track_a_h6_dba_20261018.md)。

- `plan_v1.json`：D2軸・B2群32軸・A16 signalの固定範囲。
- `input_binding_v1.json`：準備時点のHamiltonian/state/DF/compiler・保存予測。
- `source_closures_v1.json`：各stageの実行SOURCE・入力SHA・SOURCE closure、現在bytes照合。
- `technical_validation_v1.json`：人工18件、作者自身の検証。
- `A_result.json.gz`：正式PR16構成のsignal・bias・normalization・経験的適格性。
- `A_execution_receipt.json`：SOURCE、SHA、native exit0、最終gateとSTOP。
- `A_eligibility_counts.json`：通常/非相殺を別々にした保存判定の件数。
- `execution_snapshot_v1.json`：D/B稼働中・A終了の時点記録。後続完了状態はstage終端receiptを優先する。
- `completion/`：観測器が全stageのnative終端確認後に書く公開可能な結果投影。作成前は未完了。

private絶対path・有効grant・raw array・runtime/NPZ・cacheは収録しない。
追加sampling/compileの自動起動・費用統合再会計はしない。
