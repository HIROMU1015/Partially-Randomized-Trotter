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

## 待機候補の補充と評価順

新しい固定SOURCE・出力先で、上記の指定に `search_overrides.work_conserving_search: true` を追加する。通常の `run_rpe_schedule_search.py` を使用する。元の同名sectionの設定は併合する。この値も前回と同じroundを再開する際の固定入力に含まれる。

- 次の探索判断に必要なQ3を先に予約する。その後、独立した構成を優先して、現在評価可能な候補を補充用に保持する。待機候補から、稼働上限に加えて共有queueに次の1件まで予約する。優先Q3を待っている間にも、workerの終了通知で残りを補充する。完了済み・予約済みの同一jobは共有する。有限予算を使い切った時点と、現時点で評価可能な候補を使い切った時点で補充を止める。
- PauliとDFの準備producerは既存の各方式・各判断の上限を使う。同方式内で要求幅を交互に選び、各幅のfamilyと物理κ scheduleの既存順序を保つ。空き枠があり、優先Q3の待機がなければ、先読みが待機中でも準備を進められる。
- 通常のV2で方式内最安を更新できる配分のうち、異なるREADY構成の未投入候補を比較するときに、既存の `IMPROVEMENT_PER_TIME_V1` のスコアを使う。各構成の最安提案について、削減可能RZに完了観測の暫定適合率を掛け、互換profileでのQ3時間の推定値で割る。今回のexecutorで検証済みの完了結果を使い、観測が足りない間は従来順を使う。V2の定期的な探索枠、物理scheduleの同一性、精度判定、採用と確認済み最安の区別を維持する。
- 実行中の候補は元のrequestのまま回収する。待機候補の並びは選択時に更新する。予算が詰まったときは、優先候補のために未起動の先読み予約を譲る。再開では既存のreceiptと起動記録を照合し、候補リストを再構成する。時間の観測は新しいexecutorで蓄積する。

通常入口の人工検収で、後着READYの取込みと再開時の結果保持を確認した。executorの人工検収では、2枠の一方を優先Q3が保持中に、他方で4件目の候補まで投入した。有限上限4件かつ最大同時2件。再開の追加投入は0件。別の人工比較で、Q3時間による優先順位の変更と、同じ準備件数内での両幅選択を確認した。監視は `Q3_STANDBY_REFILL`、`LIVE_READY_INSTALLED`、`SEARCH_ADOPT.order_reason`、`PREPARATION_WAVE_START.dispatch_reason`。実H4と複数hostの速度改善は未測定。運用中の固定runへ適用するには、別の固定入力と実行引継ぎを用いる。
