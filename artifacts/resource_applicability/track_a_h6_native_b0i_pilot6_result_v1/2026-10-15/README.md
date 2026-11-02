# H6 native B0+I pilot6：結果の公開用bundle

[結果報告](../../../../docs/research/track_a_h6_native_b0i_pilot6_results_20261015.md)／[正式共通ノート](../../../../docs/research/track_a_shared_research_status.md)。SOURCE `2e733d8a683d415792ce92a56c7d149b78f42716`、run04-v3、3構成6軸の保存結果のみ。

|file|内容|
|---|---|
|[axis_compiled_metrics_v1.csv](axis_compiled_metrics_v1.csv)|6軸の実compile指標・fingerprint/compiler/source・wall/CPU/RSS・native exit|
|[precision_accounting_v1.csv](precision_accounting_v1.csv)|3構成×5精度×factor1/10＝30行。既存shot会計による6metric総費用。不適格はnull|
|[comparison_to_saved_PR_v1.csv](comparison_to_saved_PR_v1.csv)|今回3構成内のB0+I最小と、固定H6 D0確認union8/n32のPRとの5精度点比較。全面family比較ではない|
|[correctness_and_B0_reference_v1.json](correctness_and_B0_reference_v1.json)|保存独立信号との一致検査、旧B0 metadata参照とmatched delta未取得の区別|
|[result_summary_v1.json](result_summary_v1.json)|条件、SOURCE/grant、結果・正しさ・native終了/資源・元file receiptと限定|
|[file_manifest_v1.json](file_manifest_v1.json)|公開file bytes/SHA。manifest自身・検証記録を除外して自己参照を避ける|
|[documentation_checks_v1.json](documentation_checks_v1.json)|保存値projection・点比較算術・file/link・source/既存worktree保全の限定検証|

P=0、経験的allowance、有限候補・標本の点推定である。B0+I全12構成24軸の取得、matched旧B0実測、母平均保証、hardware shots、独立科学reviewは含まない。PR指標は8dbdb682固定CSV由来で、新sampling/compile0。
元private runtimeは再収録せず、必要な軽量値だけをprojectionした。原STOP/carry/grantを保全。科学source・契約・venv・共有環境・他job変更0。

利用者指示によりcommitのみ。pushは最終チャットのcommandを利用者が実行する。未pushを公開済みとしない。MANDATORY_STOP、次stage未認可。
