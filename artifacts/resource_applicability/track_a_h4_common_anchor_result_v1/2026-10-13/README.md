# H4固定16 anchor：native full-wrapper補完結果

SOURCE `217c971dd537b1cc7fbc13742bc0f6c263ff7f45`。linearH4/1Å/STO-3G/legacy rank12/8modes/sector36/T.8/delta.1–.8、prefix3 PR8とfullPF S2/S4×q4の8、計16固定構成。H6-derived quarter-prefix transfer形の固定panelで、H4全grid最良ではない。

signal16、264 cost group、532 compile（primary528＋ordinary4）、PR256trajectory/960occurrence/72control probeを完了。coordinator279452、全264reap・cleanup0・first STOPなし。cost wall1068.455秒（約17分48秒）。最大8worker/内部thread1/canonical hash reuse、旧production H4/共有/venv変更なし。

[summary](result_summary.json)、[532 wrapper CSV](wrapper_metrics.csv)、[signal metadata](signal_metadata.json)、[map](resource_map.json)、[metadata入力SHA](input_receipt.json)、[CSV別算術](verification.json)。新science計算を再実行しない保存解析。Nは分析上のHoeffding allocationで、実量子shotなし。P=0、経験的u、旧host compiler bit同一性未確認。

4精度.05/.01/.005/.001のPR/fullPF RZ点比は.506631/.471627/.453431/.510507（約49–55%低減）。これは固定panel内の点比較で、母平均winner・rare tail上界・H4/H6/H8の漸近scaling証明ではない。B0なし、旧M1全218grid/32標本やH6をpoolしない。S4・native/controlを合わせたが、H4 fixed rank12とH6 tol-only rank19のDF政策は同一と認定しない。

未観測次数/component・u certificate・state preparation費用・energy/RPE最終資源・新16-mode compilerの検証は別。今回grantは消費済み、予算返却/resetなし、sourceと停止履歴は保持する。
