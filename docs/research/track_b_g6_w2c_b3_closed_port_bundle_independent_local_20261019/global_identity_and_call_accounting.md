# 単一global構造・call会計

直接基点 `5e862a70f2d74defe72dd39205d38c2a4f8ebeaf`。統合公開HEAD `62967269e611b61e73b70abe20e998d9c7eaaf50`、人工測定SOURCE `deade78a19f5b23859d16816d3a7bfb37765dd80`。実施2026-10-19〜20 JST。ARTIFICIAL_ONLY・未採択・実科学入口閉鎖。

4方式×A/B×X/Y＝16cell。correctnessは各16。n16は条件あたり256＋256＋256＝768、S/P1計1,536。n256は256＋4096＋4096＝8,448、計16,896。全48pair・6phaseは一つのplan/claim/coordinator/statefile/cursor/call counter/result-log budget/STOPを共有する。

正常3件ではplanned=reserved=started=completedが全件一致。FAILED/CANCELLED/UNKNOWNは0。condition-onlyは未実行条件のNOT_STARTEDを残し、Bundle成功へ昇格しない。[独立call会計](global_call_accounting_verification.json)。公開call集計とも原rawからの再導出値を照合した。

modeはglobal trial IDに含み、seed domainから除外。run/condition/arm/layer/axis/phase/pass/trial/global offsetを区別する。offsetはtask.offset＋axis offset＋index。Sのphase0..2完了barrier後にP1 phase3..5が始まる。S失敗後P1開始、重複RESERVE/COMPLETE、offset/ID衝突は拒否。

global IDsの一意性とtrace順序を確認するが、全workerが同一物理時刻に停止したことや各関数の物理entry時刻まで証明したとはしない。
