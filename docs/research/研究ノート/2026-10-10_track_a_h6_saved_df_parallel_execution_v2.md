# 2026-10-10 Track A H6：並列保存DF入力完成の一回認可

ユーザーのCPU増枠と「並列化機能を使ったうえで計算を開始して」を[新v2 seal](../track_a_h6_saved_df_completion_parallel_execution_seal_v2.md)へ結合。
source `f37005f01b2be38c5993d6e82df91abe9c643d21`、CPU IDs [0, 2, 5, 6]・Numba4/BLAS1、local synthetic92 passed。旧v1/親raw/政策を保全。
入力完成一回だけを認可し、結果公開後mandatory STOP。H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION。
