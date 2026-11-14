# m3保存価格のconsumer binding

決定論PF shortlistのm3保存価格12軸をCの現行ID/測定軸へSOURCE付きで接続し、実guard/tierで12受入・0保留。元cohort未記録はnull、旧trial/receipt/fingerprintを保持。m4/m5較正24＋m6独立holdout12は元locatorを参照し計48軸の直接価格。新7＋既存5＝12技術test PASS。fit/予測/compile/MSE/N/G0、mandatory STOP。

D bcb67b2bの66 metadata checkと元trialを再利用。生成callerとwrapper同等参照SOURCEを分け、Cの既存guardを無変更で12軸へ適用した。m3はactual幅が異なる直接観測点で、較正/holdoutへ追加しない。ロードマップ本体の更新は統合担当へ委ねる。

[REPORT](../pf_m3_saved_price_admission_20261115/REPORT.md)／[HANDOFF](../pf_m3_saved_price_admission_20261115/HANDOFF.md)。
