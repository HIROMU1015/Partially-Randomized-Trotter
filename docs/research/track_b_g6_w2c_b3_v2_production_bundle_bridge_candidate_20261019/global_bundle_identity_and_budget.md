# Global identity・call/byte予算

candidate digest、source release candidate（全source pinのdigest）、scientific input、global scope/claim、run、condition、task、arm/layer/axis/phase/pass/local trial、global trial ID、global offset、seed namespace、receipt/STOPを区別する。global trial IDはrun/mode/local trial。modeを含むtask keyとglobal offsetもS/P1の衝突を防ぐ。trace.run_idはglobal run ID、manifest_bindingは別の契約digest。

n=256、16 cells=4方式×2layer×2axis。各条件correctness16×16=256、baseline16×256=4096、instrumented16×256=4096、計8448。二条件16896。candidate、独立導出plan、scheduler、static manifest、global receipt、consumerが同じ計算に到達し、人工full形状でreserved=started=completed=16896を確認した。

Global resultは64GiB=68,719,476,736B、logは256MiB=268,435,456B。一つのBudget objectを使い、S/P1へ別々のcapを付与しない。statefile retainedはplanned1B/slot、RAM shadowは別のRAMでありlog二重課金しない。累積pwrite trafficとretained bytesは別。

n256人工実測：result **15,502,271B**、log **43,190,850B**。双方reserved=observed=physical、unknown0。trace自己課金、journal、state、全pair、条件receipt、summary、diagnostic、snapshot、post-close receipt/marker、final/delivery/caller slotを含む。条件別byte内訳はtraceのconditionとshared最終化を別表にし、globalへ一度だけ足す。

FAKE_ONLY.sqlite3とcandidate_input等は試験harnessのauthority/input snapshotで、runtime result/logとは別の保存領域。公開rawには含める。実authority storageの会計/耐久性は正式契約未実装。早期crash/初期化失敗でglobal保存証拠が欠ける場合は完全なbudget監査へ格上げしない。
