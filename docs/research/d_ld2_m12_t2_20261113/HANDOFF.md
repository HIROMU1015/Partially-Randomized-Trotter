# T2 → B/C：単一pathの価格と実行可能性

実行前版。最終結果はrun01/T2_PATH_RESULT.jsonとREPORTへ記載する。対象はA a9ab固定PR-LD2/m12/X,Yの同一normal fresh path（cohort D_LONG_TIME_FEASIBILITY_V1/index0）だけ。価格SOURCEは実行前に固定したcaller commitを各request/outputへ保存する。

既存D Adapterのpath quantityとC/S/D・軸別6指標・fingerprint/cache参照、event digest・seeds/law、preparation/tail/λ/scalar/fragment、configuration/physical identityをそのまま保持する。測定付きfull wrapperのCと元Accounting Sは別quantity。P0を維持する。

実行成功はn=1の技術的実行可能性の観測であり、finite-cohort平均、適格E[C]、SE/CI、rare寄与、P/Q一致を認定しない。Bのmean入力へpath Cを転記しない。C5ccのscalar binding差分は前計画HANDOFFのまま残り、本runnerはC統合を再実装せず既存Dを直接使う。

途中失敗時はrun01/WORKER_STOP、RESULT、PIPELINE_STAGES、RESOURCESとprivate/pathの完了済みファイルを保全する。Xだけ完了した場合もYを補完しない。無断retryなし。追加科学実行・G会計は行わず、最終公開照合後mandatory STOP。
