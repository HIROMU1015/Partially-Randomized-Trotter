# H8 Phase A run01：保存結果と終了監査

[実行条件・結果・限界・次Phaseの承認境界](../../../../docs/research/track_a_h8_phase_a_run01_results_20261016.md)。

- [結果要約・全10条件の適格件数](result_summary_v1.json)／[件数CSV](eligibility_counts_v1.csv)
- [追加152 signalの元JSONを保つgzip](H8_signal_complete_v1.json.gz)／[元JSON bytes/SHAとexport receipt](signal_export_receipt_v1.json)
- [1,520条件のbias・allowance・headroom・保存shot CSV gzip](signal_eligibility_bias_shots_v1.csv.gz)
- [primitive検証](primitive_validation_v1.json)／[保存preparation・reference結合](preparation_reference_binding_v1.json)
- [SOURCE/input/profile/資源・承認scope](execution_binding_v1.json)／[実行terminal・不可逆予約・実績](execution_terminal_v1.json)
- [worker35件の退出とSHA](owned_worker_receipts_v1.json)／[外部資源監視・cleanup](resource_and_cleanup_v1.json)
- [旧成果保全](preservation_receipt_v1.json)／[元科学出力のfile SHA](scientific_output_file_receipts_v1.json)
- [限定公開検査](publication_validation_v1.json)／[公開file SHA](publication_files_v1.json)

repo rootで保存結果を検査：

```bash
python3 -P -B artifacts/resource_applicability/track_a_h8_phase_a_run01/2026-10-16/verify_saved_result.py
```

stdlibのみ。元JSONのgzip同一性、候補・適格件数・worker退出・予約/実績・公開SHAを検査する。signal作用・shot式・費用を再計算しない。`prefix_19`の原receiptにないresult SHAは公開時の別証拠として明記し、原本は保全する。

原run/NPZ/全state/runtime/cache/有効grant/ledger/credentialは収録しない。Phase B1/B2/C、V2解析へは移行せずmandatory STOP。
