# H4 pilot引継ぎ：計算を継続したままチャット終了

ユーザーの「計算に時間がかかりそうであれば、一旦チャットを終了してよい」に従い、科学processを停止・再起動せずチャットを終了する。現時点のprocess状態・進捗はcontinuation.json、以後の観測はrun01/progress_observations.jsonlに保存する。

科学SOURCEは0c329b30a81accb35c212be8defa66fee0dd0f12、grantは既に消費済み。承認は入力生成と固定4方式・5精度のone-shot、保存とlocal commitまで有効。追加実行・科学retry・候補変更・追加sampling・push・詳細科学レビューは含まない。

## 次回Codexが行うこと

1. run01/STOP.json、input_run/STOP.json、run/STOP.json、各caseのstatus/FAILED、stdout/stderrを読む。科学workerを不要に停止しない。
2. 実行中なら既存processと保存進捗を観測し、同じgrantをactivate/executeし直さない。unified exec sessionは67756、読取progress monitorは7709。sessionが参照不能でもready/launch記録のPID/start identityと保存証拠を照合し、未確認をPASSにしない。
3. 全4caseが正常終了した後、`python3 -B artifacts/non_df_h4_pilot/2026-10-23/verify_saved_run.py --runtime-files --output artifacts/non_df_h4_pilot/2026-10-23/verification.json`を一度実行する。stdlib保存値検査であり、科学計算は再実行しない。失敗caseがあれば部分結果を保持し、retryせず報告する。
4. READMEの実行中表記を最終状態へ更新し、技術結果・20精度行・N/G nullの理由・C/SE・実event次数と価格join・SOURCE/grant/STOP・子孫回収・保全を記録する。research synthesis/VALIDATION_STATUS/validation_manifestと入口indexへ、旧履歴を残して今回の結果を追加する。
5. 検証・result inventory・ローカルcommitを完了しmandatory STOP。SOURCEとraw/grant/claims/ledger/marker/STOPは後から書き換えない。

SOURCE HEADが実行中のbindingなので、今回のrawと下書きは未commit。計算終了後のcommitまでが承認された残作業である。旧inactive contract・人工技術STOPは履歴としてそのまま保持する。実科学grantはrun01/にあり、CONSUMEDとSTOP後は再使用不可。

## 利用者による研究スコープの訂正

利用者はH4非DF指示が並行研究側への指示を誤ってこのチャットへ送ったものだと訂正し、結果は消さず保全すると明示した。scope_correction.jsonに原文を保存した。計算停止・再実行の指示ではないため、実行中のone-shotはそのまま継続する。

H4の結果・検証・local commitは専用H4 worktree/branchの範囲に保持し、このチャット本来の研究や資料へ混在・統合しない。上記の引継ぎにある共通研究概要・主題側indexへの統合指示は、この訂正により適用しない。旧SOURCE・raw・grant・ledger・marker・STOPは変更しない。

## 利用者のまとめ・commit指示に基づく自動後処理

completion_authorization.jsonの利用者指示に基づき、finalize_saved_run.pyを既存科学runのSTOP待ち→保存値検証→専用H4結果まとめ→local commitの範囲で起動する。Git/STOP境界の人工8検査PASS、科学call追加0。SOURCE HEADを計算中に変更しない。

後処理のログは/tmp/h4-finalize-859ce9f6.log、実commit SHAと成否は/tmp/h4-finalize-859ce9f6.receipt.json。finalizer_claim.jsonのPIDが稼働中なら、別に同じ検証やcommitを起動しない。成功後はGit log、verification.json、result_summary.json、result_inventory.jsonを確認する。失敗時はFINALIZATION_FAILED.jsonとログを保全し、科学retry・grant再発行は行わない。

この後処理には本チャット本来の研究概要・発表資料への統合やpushを含めない。完了後mandatory STOP。
