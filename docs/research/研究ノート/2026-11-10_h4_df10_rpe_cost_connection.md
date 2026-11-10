# 2026-11-10：H4 DF10 RPE費用会計の接続準備

[技術REPORT](../h4_df10_rpe_cost_connection_20261110/REPORT.md)。利用者指示の主眼を、追加signalより保存N/Cの会計接続確認へ置いた。D1のh=.1/β=.4,.2/13round/α=.05と保存Nは変更・再計算しない。既存の全round軸別会計を再利用し、部分m3を誤表示しない薄いadapterを追加。人工・回帰14件PASS、実H4のN×Cは未実行。

PF決定論、PR探索、PR確認の3ケースをcopy/joinで準備した。ideal Nとfinite-bit sampler価格の資格差・未知幅・未知covarianceは未保証のまま。主研究base/extensionの同候補系列48行のmetadata coverageを保存し、T1.6のPR q32には直接価格があることを確認した。q16/δ=.1とは別候補なのでNを自動移植しない。DのCDF8/Strong-Reference係数や旧DF12の係数をDF10へ移さない。

次の最小案はT=.8/m3の保存RZ小計3ケースだけ。新signal/fit/compileなしで実行可能だが、明示的数値許可が必要。その他roundのN/価格資格・保証はその後に判断する。Main Claim変更、全G/H6/詳細科学レビューなし、旧SOURCE/raw/結果/STOP保全、公開照合後mandatory STOP。
