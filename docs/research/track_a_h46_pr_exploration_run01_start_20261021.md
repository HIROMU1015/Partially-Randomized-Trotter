# H4/H6高次PF PR本探索 run01：並列起動

SOURCE `5a448d94ea389ada6e2b2755de6a215ed2034679`。利用者の推奨案採択・一回取得承認を独立したH4/H6契約へ結合して起動した。

観測時刻：2026-10-21T00:49:56.054979+09:00。この報告は開始snapshotであり、完了・最終native exit・全stage PASSを意味しない。

| 系 | 新trajectory予定 | compile軸予定 | 起動worker | 起動gate | 完了trajectory |
| --- | ---: | ---: | ---: | --- | ---: |
| H4 | 793 | 1586 | 8 | PASS | 0 |
| H6 | 607 | 1214 | 28 | PASS | 0 |

両系でcontextを1回共有し、native_buildへ到達した。36科学workerが稼働し、複数回の観測で自分のprocess群のCPU時間が増加した。観測時のSTOP file・結果エラーは0。各系の登録を長いq×piece順に割当てているため、最初に重い処理が進む。

[起動snapshot](../../artifacts/resource_applicability/track_a_h46_pr_exploration_run01_start/2026-10-21/start_snapshot.json)には、plan/config/closureのSHA、2146 seal/系、15879新seedの既知registryとの衝突0、fresh gateと起動workerを記録した。private seed一覧・原本・絶対pathは公開しない。

新12件と既存104件の人工・回帰検査がPASS。native演算bytecodeは変更せず、SOURCE/input全SHAを維持した候補表cacheだけを追加した。fork/readonly共有context・SHA付き入力cache・内側thread重複防止・重い処理からの割当てを再利用した。実分子での高速化倍率は未測定。

既存338708447996 byteの容量計画を保持しfresh空き容量を確認した。容量の再見積り・worktree整理・旧raw削除は0。m5/S2/Yoshida4、q1/2/4/8、outer32/K4、compiler・CT01・その他native閾値は不変。

durable controllerがチャット終了後も取得を監督し、終了後に原SHA復元、native gate、call台帳、結果SHA、exit/reapと技術資源を監査して公開技術summaryをlocal commitする。両系とも独立STOPとし、局所数値FAILでは他系の正常jobを停止しない。共通SOURCE/input/storage異常は影響する自分のstageだけを停止する。retry/top-upなし。

新しい統計/MCSE評価、正式shot政策、総費用再会計、再ランキング、独立確認、AD03採用は実施しない。pilot・確認標本を混ぜない。取得後はmandatory STOP。

次の担当：Codexが取得の終了監査・保存・報告を行う。ユーザーは終了結果を確認した後、限定統計評価を別途承認するか判断する。

追跡観測（00:51:09 JST）：H4 8 workerはwrapper_cosineへ進行し、H6 28 workerはnative_buildを継続。前回snapshotから自分のprocess群のCPU時間が増加、STOP/エラー0。完了trajectoryはまだ0。[追跡記録](../../artifacts/resource_applicability/track_a_h46_pr_exploration_run01_start/2026-10-21/followup_observation.json)。
