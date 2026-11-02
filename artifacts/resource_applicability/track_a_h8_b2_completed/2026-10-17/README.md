# H8 Phase B2：登録S4全4軸PASS

H8/隣接1Å/STO-3G/保存DF28/T=.8。B0 p15q1両軸は各168,353 RZ/誤差約0.2154%、B0+I p21q8両軸は各1,852,640 RZ/誤差0%。保存3軸＋最新sine1軸を独立受入で結合した。過去runのSTOPは保持している。

`b2_full_pass_binding_v1.json` が4軸のSOURCE・実測・判定・最終gateを結合する。`execution_receipts_v1.json` は終了原receiptのprivate pathを伏せたprojection。有効grant・private NPZ/QPY・runtimeは公開しない。

`python3 -B verify_saved_result.py` は公開された保存bytesとmetadataのみを照合し、新科学計算・live probe・fit・compileを行わない。publication inventoryにある更新可能な索引は、このcommit時点の値を検査する。

Phase C未承認・未実施、全格子/厳密bias/PR母平均保証は未認定。mandatory STOP。
