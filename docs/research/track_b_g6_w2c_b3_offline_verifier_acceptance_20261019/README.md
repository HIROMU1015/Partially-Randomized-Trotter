# B3 offline検証・古典費用・高速化受入基盤

2026-10-19 JST。**ARTIFICIAL_IMPLEMENTATION_AND_VALIDATION_COMPLETE / mandatory STOP**。
既存275＋新85＝**360 tests PASS、FAIL/ERROR/SKIP各0**。受入例の385 PASSは検査行数でありtest件数とは別。fused2項目＋parallel11項目はPENDING_EVIDENCE。

直接基点は `564409487a35e15d73f1d0f0c2611824b400ab02`、基準技術SOURCEは `baebe23f2ed9620e0e2fc492f0e2b8ced3c7f342`。
branch `track-b-g6-w2c-b3-offline-verifier-acceptance-20261019`。独立worktree、新規ファイルのみ。既存科学SOURCE・Driver/Admission/Host/Readiness・R1契約・結果は変更していない。
固定資料の取得元commitと完全SHAは [source_inventory.json](source_inventory.json)。fused公開候補は資料のみ参照、未公開server成果は参照していない。

- [pair・global整合性契約](result_integrity_contract.md)／[F01](f01_failure_verification.md)
- [費用会計](classical_cost_accounting.md)／[受入matrix](integration_acceptance_matrix.md)／[parallel契約待ち](parallel_adapter_pending_contract.md)
- [人工検証](artificial_test_report.md)／[全test ID](artificial_test_results.json)／[Q1–Q8](handoff.md)
- [人工入力manifest](artificial_inputs/manifest.json)／[7種の出力例](artificial_example/verification_report.md)
- [費用比較表](artificial_example/cost_comparison.md)／[受入例](artificial_example/integration_acceptance_report.md)
- [保全](verification.json)／[公開照合](publication_receipt.json)

実装は `src/trottertracks/algorithm_codesign/g6_w2c_b3_offline_result.py`、`g6_w2c_b3_offline_cost.py`、`g6_w2c_b3_integration_acceptance.py`、`g6_w2c_b3_offline_cli.py`。
人工producer/testsは `tests/tracks/algorithm_codesign/b3_offline_artificial_fixtures.py` と `test_g6_w2c_b3_offline_verifier.py`。
本資料は新たな科学validation pathや研究仕様・結果の更新ではなく、人工データ専用の技術ツールの入口である。既存研究索引・科学status・manifestは保持した。

## 再現

repository rootから、独立7 suitesを並列実行する。logは新規owned /tmpへ保存される。

```bash
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 docs/research/track_b_g6_w2c_b3_offline_verifier_acceptance_20261019/run_artificial_tests.py
```

既存の人工入力から7出力を再生成する例。`/tmp/b3-offline-reports-new`は未存在である必要がある。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /usr/bin/python3 -m trottertracks.algorithm_codesign.g6_w2c_b3_offline_cli \
  --artificial-manifest docs/research/track_b_g6_w2c_b3_offline_verifier_acceptance_20261019/artificial_inputs/manifest.json \
  --new-output /tmp/b3-offline-reports-new --source-root . \
  --acceptance-baseline docs/research/track_b_g6_w2c_b3_r1_f01_admission_integration_20261019/technical_contract_candidate.json \
  --artificial-test-evidence docs/research/track_b_g6_w2c_b3_offline_verifier_acceptance_20261019/artificial_test_results.json
```

新規人工入力・報告の生成は `build_artificial_example.py NEW_INPUTS NEW_REPORTS`。どちらも新規directory、報告は入力directoryの外。
入力はexplicit manifestで指定しread-only、探索・append・resume・上書きはない。manifestのartificial_onlyは利用者の入力scope宣言であり、未知データが人工物であることの暗号学的証明ではない。
今回生成・検証したのはtoy mockだけ。実データ適用は自動有効化していない。

対象となる将来契約はG7 P5_general_order、p=(1/5,3/10,1/2)、x=5/7、m5、σ+1、4方式×A/B×X/Y、A6数学的enclosure τ10^-6/Kφ32/J32。
分子/geometry/basis/DF rank/L_D/δ窓は非該当。科学上の予定31,710,848 callsは今回未実行。
人工例はcorrectness16/axisとprofile18/axis、fake modules/clockの832 toy calls。実N・seed・R1契約の変更ではない。
実G7/F3、実結果解析、実authorization/grant/claim発行消費、正式採択、最終science release/digest固定、科学レビューはすべて未実施。
