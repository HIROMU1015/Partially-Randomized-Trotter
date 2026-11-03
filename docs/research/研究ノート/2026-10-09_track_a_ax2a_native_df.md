# 2026-10-09 AX-2A native DF接続の追記

前段の[AX-2A準備記録](2026-10-09_track_a_ax2a_preparation.md)は変更しない。
利用者の「次の作業に進んで」を受け、native DF global二次/四次、対称directional control、
既存RTE requestの明示event replay、phase-aware vector callback、同じHadamard gate列を接続した。
diagonal global phaseもdirectional controlに残すことが、ordinary-control作用一致に必要だった。

新53 native testsと7 preflight tests、旧57 tests、合計117 local synthetic tests passed。
分子科学計算・trajectory sampling・transpile・GPUは0。結果は未commitの実装証拠に限る。
共有科学sourceと旧準備source/artifactを保持し、新たなversionの準備inventoryへ記録する。

AX-2B草案をH4 correctness8 cell、H6 structural6 cell、full-wrapper60 callに具体化した。
DF tol1e-8、gate/matvec capは提案で、採用・実割当・認可ではない。
preflightはfield存在と科学的証明を分離し、全fieldが埋まってもlaunchを許可しない。
科学runner、snapshot/sector/reference、cap enforcement、source/task/認可freezeが必要。

正本は[AX-2A native準備v2](../track_a_ax2a_native_df_preparation_v2.md)。
終端 `AX2A_NATIVE_PREPARED_AX2B_NOT_AUTHORIZED`、科学STOP。
