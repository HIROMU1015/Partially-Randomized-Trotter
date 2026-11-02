# 2026-11-02 P3価格確認：新32本/arm完了

ユーザーの許可に基づき、科学SOURCE`1cfce5e961e3d063741a1e0656052b10c7b27392`を人工9PASS後に公開・照合して1 batch実行した。
H4/1Å/STO-3G/full256/保存DF tol-only1e-8 rank10/案A/T.8/S2/q8/r4/K4/M_B・M_C/P=0。
保存Gaussian/DF basisを再利用、独立半指数1と保存half照合PASS、新64 Full oracle PASS、128 compile成功。
64 trajectory/2048位置、旧pilotと非pool、対角化/Givens/旧価格再compile/automatic retry/新N-Gは0。

[報告](../../../artifacts/df_pauli_matched_partition_validation/2026-11-02/stage3/run04/REPORT.md)：両軸RZ平均±SEはPauli3017.9375±9.41376、DF8559.71875±150.19704。
cos/sin共分散と全個別価格を保存し、order4未観測・母平均tail未保証を明記した。
全128異なるcompiler PID、全join/exit0、wall約69.23秒。旧Stage 2/Stage 3全STOPは不変。
原QPY320を含む704 memberを可逆archiveへ保存し、公開・remote照合後停止する。
次の担当はユーザー：Stage 4条件付きN/G会計の仮定・精度範囲を判断。Codexは別指示まで開始しない。
