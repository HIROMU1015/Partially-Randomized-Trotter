# H6独立32確認：256 group完了

[結果報告](../../../../docs/research/track_a_h6_independent_confirmation_result_v1.md)が主入口です。
linear H6/1Å/STO-3G/tol-only DF19/sector400/T0.8、prefix5/10/15/full19・delta0.1–0.8。
選択8構成×独立32＝256 group/512 wrapperを取得。159 reused＋新97、探索8標本はpoolしません。
P=0・経験的uの確認点推定でPR/full PF RZ比0.459978 /0.399630 /0.372105 /0.387006、約54–63%低い。
K2/K4順位差は有限標本のばらつき内、3/8でnonzero次数・全K4でorder4未観測。母平均winner/厳密numerical certificate/最終RPE総費用は未認定。

- [結果・精度別比較・shot/SE・感度・resources・carry](result_summary_v1.json)
- [native RZ/CX/depth/size mapと固定ペアのstate-preparation交点](confirmation_resource_map_v1.json)
- [512 wrapperの整数metric/seed/compiler/event/digest CSV](wrapper_metrics_v1.csv)
- [256 groupと再利用16 baselineのsource/path/byte SHA来歴](integrated_confirmation_v1.json)
- [希少Taylor次数と保存lawからの未観測確率](rare_order_diagnostics_v1.json)
- [251 source/536費用行/完了receipt/native退出の局所監査](summary_audit_v1.json)
- [CSV・shot・shaの別算術照合](report_consistency_v1.json)
- [保存JSON限定集計script](aggregate_saved_results.py)、[CSV限定検査script](verify_report.py)

実行SOURCE87b6b9f46b7e8ba9abe132ee21478c114b6f46f7、科学SOURCE5a903aea2492c2d211c1e15e7737df0d50694c12。
新run exit0/mandatory STOP、97全reaped/cleanup0、wall16分12.793秒、peak aggregate RSS4.973911GiB。
環境・compiler・元task/seed・roundoff条件不変、旧compiler bit完全同一性は未検証。
rawruntime/NPZ/cacheを報告commitへ追加しません。未知actualはnull、予約別、返却/resetなし。
本集計はstdlib/metadataのみ・AS2GiB/内部thread1/180秒上限。新science/transpile/trajectory/grant/GPU0。
source/test・H4・既存venv・共有環境変更0、local報告commit、agent push0。

再集計には固定SOURCEと外部raw配置が必要です。生成fileを上書きしないscriptであり、rawなしのGitHub単独再実行をPASSとしません。集約CSVからの主要費用再構成は保存report内で確認できます。
