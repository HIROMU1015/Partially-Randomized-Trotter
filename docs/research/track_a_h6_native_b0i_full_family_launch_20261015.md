# H6 native B0+I 全12構成：実行承認・最終結合

利用者指示「計算を開始してもよい　何かしらで計算が止まった場合は修正まで行ってよい」を根拠に、[準備案](track_a_h6_native_b0i_full_family_readiness_20261015.md)を今回の一度の実行へ結合した。

SOURCE `731826b3bb82dba18b31989a39607fe426d34960`、準備REVIEW `ca4165438c6be0a2a819ee50b57387f9b42aad61`。旧6軸は原SOURCEのまま条件付き再利用し、新9のscreen後に適格なn件だけ最大18軸をcompileする。driver CPU63、workers [62,61,58,53,52,50]、各thread1・AS32/RSS9GiB、memory admission240.5GiB、wallnull・累積CPU86400秒、metadata CPU1800秒。番号はown-runだけのfresh空きphysical core選択であり、他jobのaffinityは変更しない。

[最終監査bundle](../../artifacts/resource_applicability/track_a_h6_native_b0i_full_family_launch_v1/2026-10-15/README.md)。全18入力bytes/SHA・SOURCE14・provider230・profile34273一致。NPZはbyte-only検査で科学array未読込。carry失敗・UNKNOWNを保持しreset/refundなし。新規の科学条件/compiler変更・新state/signal exponential action・旧case再取得なし。

今回の最終照合は主担当自身が実施し、独立reviewとは呼ばない。既存の独立source監査の対象hashは現SOURCEと一致、人工44testsを再実施せず利用した。サブエージェント0。未使用output/control/one-shotとCPU/memory/pressure/OOM/FS/inode/quotaを起動直前に再検査し、不合格なら起動しない。

このcommit時点はAPPROVED_SEALED_AWAITING_FRESH_LAUNCH。実行後は結果またはSTOP証拠を保存し、問題があれば科学条件と既定の資源・件数の範囲内で修正する。盲目的な自動retry・次stageへの進行なし。
