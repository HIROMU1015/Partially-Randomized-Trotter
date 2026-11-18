# PR幅探索の空き枠への投入

通常の `scripts/run_rpe_schedule_search.py` で、先行Q3の採用を待つ間にも後着の準備済み候補を評価できる。適用範囲は明示した複数幅のPR探索。PR-PauliとPR-DFは一つのQ3キューを使う。

- Q3の待ち行列があれば、現行どおり次の判断に必要なジョブを先読みより優先する。Q3 workerの終了、待機予約の解除、候補の準備完了で、割当側へ通知する。
- 候補は、保存済みの信号・価格・構成を通常の検査と登録に通してから選択に追加する。処理中のQ3、候補の同一性、登録済みのrouteはそのまま引き継ぐ。先行workerの終了を確認してからその結果を取り込む。
- 準備完了通知を受けたV2は、既存ルールで次の候補を選ぶ。Q3枠が空いていれば選択結果を待っていた方式も再選択できる。片方の方式が先に候補を使い切った場合も、新しい準備済み候補が届けば再選択する。既定のlookaheadと方式別launch上限は引き続き適用する。
- 待機Q3への割当を優先してから、次の候補準備を開始する。PauliとDFで独立した準備は並行する。同方式の候補は共通の保存信号・価格を再利用できるよう順番に処理する。開始済みの準備は完了まで回収する。

新しい技術SOURCEと出力先を固定するとき、通常のroundに次を指定する。`methods`、`science_approved`、資源profile、探索条件は対象の固定入力に従う。`live_ready_refill` を指定しない既存roundは従来の経路で動く。同じrunの再開は同じ設定を読む。

```json
{
  "overlap_ready_batch": true,
  "live_ready_refill": true,
  "executor": {"scheduler": "concurrent"},
  "search_overrides": {"rule": "METHOD_MINIMUM_PRIORITY_V2"}
}
```

`executor` と `search_overrides` は既存値に追記する。独立候補を待ち中に先読みするには既存の `lookahead > 1` を指定する。選択済みの仕事をどれだけ起動できるかはQ3枠と残りlaunch予算に従う。

旧R08は `cbeb88cfce0590045e9de3d85ffb8339d45a5a01` までの公開版を起点とした固定SOURCEのrun。今回そのworktreeや状態ファイルは編集せず、新branchに成果を固定した。新SOURCEを稼働runへ適用する場合は、既存の実行所有権を確認してから出力を分け、完了job・準備証拠・残予算・STOPを引き継ぐ。

人工テストは通常CLI入口、実キュー、人工workerを通す。従来の挙動では先行workerの終了後に後着候補が投入される。新設定では先行workerが枠を保持したまま、後着候補が投入されることを開始・終了イベントで確認する。DFの選択が一旦終わった場合も、Pauliが走っている間に復帰する。同じrunの再開で追加launchは0、各方式の有限上限は2件、両方式の確認済み最安は通常経路と一致した。実H4の信号取得・compile・Q3は0件。実H4の短縮倍率は未測定。

`PROGRESS.jsonl` の `Q3_WAIT_YIELD_FOR_READY` と `LAUNCH`、`DRIVER_PROGRESS.jsonl` の `LIVE_READY_INSTALLED` と `CANDIDATE_INPUTS_END` で確認できる。`tests/test_pr_live_ready_refill.py` が人工再現手順。既存キューの障害回収・予算・再開テストと併用する。

検証はcontroller環境で51件、native Pythonで新規の順序・再開テスト2件がPASS。対象は `test_pr_live_ready_refill`、幅準備・並行実行と統合、queue予算・先読み、例外回収、R08引継ぎの既存テスト。人工workerのゲートで投入順を比較したため、時間比の計測には使用しない。
