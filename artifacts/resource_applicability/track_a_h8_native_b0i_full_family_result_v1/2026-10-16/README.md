# H8 native B0+I 全登録候補：軽量結果と監査

[結果報告・条件・限界](../../../../docs/research/track_a_h8_native_b0i_full_family_results_20261016.md)、[H4/H6/H8の整理](../../../../docs/research/track_a_h4_h6_h8_strengthened_baseline_status_20261016.md)。

- [主比較60条件・PR paired SE](comparison_P0_v1.csv) / [family minimaとunion6へ訂正した表示](comparison_summary_v1.json)
- [新16軸の実測費用・fingerprint・wall/CPU/RSS](axis_compiled_metrics_v1.csv)
- [登録12構成×10条件の適格性120行](case_eligibility_v1.csv) / [保存shot会計120行](case_accounting_v1.json)
- [保存scalar/signal/preparation receipt12件](case_scalar_signal_receipts_v1.json) / [native正しさ8構成](case_correctness_v1.json)
- [仮想Pの60曲線](P_sensitivity_v1.json) / [厳密有理数284区間](P_envelope_segments_v1.csv)
- [旧D0確認費用の軽量projection](saved_confirmation_cost_projection_v1.json) / [元commit・input/profile来歴](provenance_v1.json)
- [終了・監視・消費履歴・元158出力SHA監査](completion_audit_v1.json) / [抽出算術の集計](extraction_summary_v1.json)
- [公開資料だけの照合script](verify_saved_projection.py) / [限定検証結果](publication_validation_v1.json) / [file bytes/SHA](file_receipts_v1.json)

```bash
python3 -P -B verify_saved_projection.py
```

stdlibのみ・read-only。保存shot×費用、候補最小値、P区間の端点と傾き、公開file hashを検査する。NPZ・科学package・runtime・venv・実workerは不要。化学計算やcompilerを外部再現した検証ではない。

原runtime/checkpoint/cache/NPZ/ledger/credentialは収録しない。元JSONの欄名`confirmed_PR_union8`はH6由来のラベルだけの残りで、入力はH8 union6/独立32。本projectionだけ`confirmed_PR_union6`へ直し、元SOURCEと結果は不変に保つ。実測値の差し替えや再計算はない。

このbundleは完了したone-shot結果の文書化。新科学計算・追加compile・新grant・新科学レビュー・原稿改訂0、subagents0、公開準備後mandatory STOP。
