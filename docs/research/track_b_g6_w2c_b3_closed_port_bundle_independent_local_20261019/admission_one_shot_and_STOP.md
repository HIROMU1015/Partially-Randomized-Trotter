# Admission・one-shot・STOP

直接基点 `5e862a70f2d74defe72dd39205d38c2a4f8ebeaf`。統合公開HEAD `62967269e611b61e73b70abe20e998d9c7eaaf50`、人工測定SOURCE `deade78a19f5b23859d16816d3a7bfb37765dd80`。実施2026-10-19〜20 JST。ARTIFICIAL_ONLY・未採択・実科学入口閉鎖。

保存FAKE_ONLY SQLiteをread-only immutableで読み、一つのscope/claim、binding digest、SPENT/STOPPED、spent=1、保存historyとsnapshotを照合。snapshotだけを全atomic transitionの証明と扱わない。

別processの新規fake scopeで32競合winner1/loser31、二重claim/STOP後再claim拒否、commit後ACK不明でもSPENT/1の維持を確認。11個のclosed/contract/seed/variant/fake approval改変をclaim前に拒否。production_entry/ProductionFactory/ProductionBitsへの9要求も拒否。[direct controls](regressions/direct_controls.json)。

S失敗はglobal STOP、P1新規開始禁止、部分raw/未完了/creditを保持。P1失敗はS raw/receiptを保持しP1失敗/UNKNOWNを残す。追加「S失敗後P1 ENTER偽造」は不整合、「P1失敗時S receipt喪失」は証拠不足。

trusted ordered traceによりparent arrival順、STOP後ENTER禁止、既存inflight drain、START_RESERVEDとbody entryの区別、CANCEL/FAIL/UNKNOWNを検査。未ACK creditは返却しない。停止の全worker同時性やpower-loss耐久性は未確立。

fake scope/approvalは実認可ではない。実grant/authorization/claim発行・消費、旧claim再利用、STOP解除は0。
