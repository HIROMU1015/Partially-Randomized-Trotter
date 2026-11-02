# H6探索結果：592 group完了・独立確認前

[研究用結果報告](../../../../docs/research/track_a_h6_exploration_result_v1.md)が主入口です。
linear H6/1Å/STO-3G/tol-only DF19、prefix5/10/15/full19、T0.8・delta0.1–0.8。
探索592/592、signal92、B3 normalization84、wrapper1188。PR各候補8 trajectory、独立確認0。
探索点推定の最良PR/full PF RZ費用比は0.4545 / 0.3952 / 0.3669 / 0.3878（epsilon0.05/0.01/0.005/0.001）。
P=0・経験的u・探索有限標本の条件下の値。population winnerはUNDETERMINED。

- [結果要約・精度別family最良・shot/SE/資源・入力binding](result_summary_v1.json)
- [全92候補×4精度×allowance1/10のCSV](candidate_summary_v1.csv)（不適格・未取得費用は空欄）
- [RZ/CX/depth/size map・共通準備費用交点](exploration_resource_map_v1.json)
- [全1188 wrapperのmetric/seed/compiler/source一覧](wrapper_metrics_v1.csv)
- [592 groupの原task・source・path/SHA来歴](integrated_exploration_v1.json)
- [全72 PR候補のorder・希少event診断](cost_order_diagnostics_v1.json)
- [保存metadata受理・source240・92 precision再会計・成果SHA](summary_audit_v1.json)
- [整数CSVから2,072費用値を再構成した照合](report_consistency_v1.json)
- [保存JSON専用の集計script](aggregate_saved_results.py)

実行SOURCE78a39051d8b6d41c33d592ecfe172f8824adfaeb、science SOURCE5a903aea2492c2d211c1e15e7737df0d50694c12。
source/compiler/input・元task/seedを変えず、旧449＋Phase A447＋保存7＋新135を統合。
roundoff承認を保持し、旧compilerとのoutput bit同一性は未検証。
旧partialは費用標本に混ぜず、未知actual totalはnull、予約は別。grant/retry/独立確認/H8追加なし。
今回の集計はstdlibのみ、AS1GiB・内部thread1、新science/transpile0、科学array読込0。
科学runtime/既存venv/H4/共有環境への変更0、local資料のみ・agent push0。

集計scriptは同じSOURCE closure・外部raw配置が必要で、既存reportを上書きしません。
生のruntime/NPZは本資料commitに含めません。既存実行receiptは
[原完了記録](../../track_a_h6_roundoff_resume01/2026-10-12/launch_parallel_v1/exploration_complete_receipt.json)を参照。
