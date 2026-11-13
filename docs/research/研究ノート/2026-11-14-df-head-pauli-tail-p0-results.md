# 2026-11-14：LD1/LD2のPauli P0を一回実行

ユーザーが75fb0ba1のSCIENCE_PLAN P0だけを明示承認した。既存A branchを継続使用し、固定SOURCE fd111f989のsave-only loader/CAR/M1 JW/Pauli collector接続でfresh run01を実行した。[REPORT](../df_head_pauli_tail_p0_results_20261114/REPORT.md)。既存P3/O1/DF signal/Full PF結果・STOP、他担当worktreeは変更していない。

保存H4 DF10/Linear1Å/STO3G/同state、full256とsector(2,2)/36、LD1/2の81数値gateがすべて通過。各184非identity Pauli項で個別136labelはsector外へ遷移するが、合計tail/headはsectorを保存。full operator再構成残差最大7.1102e−15。state norm/liftとH/state/snapshot hashは一致。Pauli λはLD1 3.3066052942233157、LD2 2.0550998769244275、保存DF λは2.8629570945298686/1.6030956277929231である。

saved scalarは変更せず、Pauli identityとの差はbinary64 residualとして保存。OpenFermion内部簡約とcollector imag許容を保持し、厳密同一や本番law/RMSE保証とは区別した。Qiskit↔保存CARのbit reversalを明示し、個別eventを36へ射影してunitaryと扱わない。

人工5＋既存4 PASS、保存入力1/full拡張1/辞書2。P1 mean/reference/eigh/finite normalizer/sampling/build/compile/RPE/N/C/Gは全て0。次のP1・辞書付きB canonical・C/D価格接続は未実行。公開・remote照合後mandatory STOPし、ユーザーの次の有限科学実行承認を待つ。
