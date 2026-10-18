# H4/H6 S4 counter/native互換性修正・限定再会計

**全3 Stage COMPLETE_MANDATORY_STOP**。

[最終報告・条件・結論の限界](../../../../docs/research/track_a_h46_s4_counter_native_compatibility_results_20261018.md)

- [Stage 1 patch](stage1/counter_minimal.diff)、[68 tests](stage1/tests_summary.json)、[native時間hex](stage1/stage_time_hex.json)
- [Stage 2：8整数行の完全一致](stage2/acceptance.json)、[A/B/mixed全値](stage2/corrected_theory.json)、[33block primitive照合](stage2/primitive_reuse_checks.json)
- [期待理論対保存compiled平均：84行](stage3/expectation_vs_saved.json)／[CSV](stage3/expectation_vs_saved.csv)
- [同一event比較：84行](stage3/same_event_comparisons.json)／[CSV](stage3/same_event_comparisons.csv)
- [shot込みPR/PF：360行](stage3/pf_pr_comparisons.json)／[CSV](stage3/pf_pr_comparisons.csv)
- [旧値・新値・差分の528行対応](stage3/old_new_differences.json)、[優劣方向](stage3/direction_audit.json)
- [保存shot70条件](stage3/preserved_shots.json)、[shot-weighted totals](stage3/shot_weighted_totals.json)、[集計](stage3/result_summary.json)
- [Stage SOURCE/result commit一覧](stage_commit_index.json)、[保全監査](preservation_audit.json)
- [Stage1消費済みgrant/source/input projection](stage1/audit_projection.json)、[Stage2](stage2/audit_projection.json)、[Stage3](stage3/audit_projection.json)
- 各Stageの `terminal.json` と `native_exit.json` に正常終了/STOP/waitを保存。Stage2の[2 worker退出](stage2/worker_native_exits.json)も確認。
- [保存結果checker](verify_saved_results.py)、[byte/SHA manifest](manifest.json)

元grant原本・private raw・認証情報は公開しない。公開projectionは新たな実行grantとして使えない。

```bash
python -P -B artifacts/resource_applicability/track_a_h46_s4_counter_native_compatibility/2026-10-18/verify_saved_results.py
```

checkerはstdlibによる保存metadata/byte/source検査のみで、科学runnerを再起動しない。

8行のpretranspile不一致は解消した。compiled/theory残差と、H4主条件の理論・compiled優劣方向の不一致は残る。新結果は旧結果を上書きしない。

次の担当：ユーザーが修正版表の原稿・科学的主張への反映を判断する。追加sampling/compile/最適化/別branch merge/GPT・Claude詳細レビューは開始しない。**最終mandatory STOP**。
