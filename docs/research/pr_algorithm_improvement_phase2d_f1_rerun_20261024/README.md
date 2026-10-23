# PR混在辞書F1：修正版の完了結果とSTOP

登録済みF1の正しさ検証は全件PASS。未資格intervalのfallbackは正常で、認証score・最適選択・shot数は未評価。

- [条件・4arm比較・残差・価格・実績・費用・provenance・次の担当・mandatory STOP](EXECUTION_REPORT.md)
- [機械可読要約](../../../artifacts/pr_algorithm_improvement_phase2d_f1/2026-10-24-r1/F1_RESULT_SUMMARY.json)
- [private rawの相対pathとhash](../../../artifacts/pr_algorithm_improvement_phase2d_f1/2026-10-24-r1/OUTPUT_HASH_MANIFEST.json)

SOURCEは `8192530b17d6bc4af195c734ff43a6e78c506942`。
既存runner `scripts/run_pr_dictionary_f1_saved.py`、adapter `src/trottertracks/pr_dictionary_validation/f1_saved_execution.py`、
`f1_saved_plan.py`、Iのcircuits/reference/pricingと既存trotterlibを再利用した。
管理testは `tests/management/test_pr_dictionary_f1_saved_plan.py` と `test_pr_dictionary_compile_metadata.py`。
旧20科学testはB入力を再生成するため追加実行せず、固定16familyの登録handlerで検証した。
[旧停止run](../pr_algorithm_improvement_phase2d_f1_execution_20261024/handoff.md)とgrant／rawを保全した。
**mandatory STOP。次はユーザー／GPTの判断へ戻し、F2は別承認とする。**
