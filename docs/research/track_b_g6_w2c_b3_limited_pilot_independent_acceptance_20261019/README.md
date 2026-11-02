# 限定G7 pilot準備版：独立技術受入

**技術監査完了・producer会計修正要・科学実行は閉鎖。** 公開18ケースは、正常完了1、技術的失敗13、履歴証拠不足4。別の人工trace書込故障で、statefileの予約1件に対しreport予約0件となる不整合を再現し、独立checkerが拒否した。

[Q1–Q10](handoff.md)、[machine verification](verification.json)、[18ケース](artificial_case_verification.json)、[producer不整合と最小修正要件](producer_consumer_discrepancies.md)、[log予算](log_budget_assessment.md)を参照。

|役割|固定commit|
|---|---|
|直接作業基点・旧consumer|`bf601f8a9723a61b7a05b534019f8e036a362735`|
|producer公開HEAD|`76ca3c4089b201d55da0fede08e1f77733ac38a5`|
|producer公開content|`f3ba0cd09d3c09c5d04a6b48c3d6c9da63fff867`|
|producer人工測定SOURCE|`b9cabcfc6cf304ab65fc8a540c33ce2173f6cbaa`|
|共通指示|`7eff8a33bef4ee1f29698ab10c6da46988dc8db5` の2026-10-17正式版、§8.4|

209個の選択入力をGit/public rawのbytes/SHA256で独立照合した。producer報告の684ファイル全件を今回再検証したという意味ではない。SOURCE inventoryの27項目中25項目は測定commitにも存在して一致し、2管理scriptは測定後の追加と区別した。主要SOURCE、schema、input pinsは一致。[固定照合](independent_source_verification.md)を参照。

最終独立回帰は **623 PASS＝既存540＋新83、FAIL/ERROR/SKIP各0**。固定producerの31人工testも別枠でPASSした。producerの過去696件・481件を足していない。SOURCE不整合の検出がnegative testの成功であり、producerの欠陥が消えたことにはしない。

## 実装・再現の入口

新moduleは `src/trottertracks/algorithm_codesign/g6_w2c_b3_pilot_independent_{identity,trace,adapter,cli}.py`。testは `tests/tracks/algorithm_codesign/test_g6_w2c_b3_pilot_independent.py`。旧global core、pair/F01、CPU partitionを再利用し、通常checker経路でproducer/scheduler/Admissionをimportしていない。固定server checkerの判定を最終判定に委ねていない。

Python3.10以降・標準libraryのみ。Repository rootから:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 docs/research/track_b_g6_w2c_b3_limited_pilot_independent_acceptance_20261019/run_independent_tests.py
```

新しい専用/tmpへ10 suiteのログ・ID・SHAと禁止科学module import監視を保存する。単一人工case:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m trottertracks.algorithm_codesign.g6_w2c_b3_pilot_independent_cli \
  --case docs/research/track_b_g6_w2c_b3_limited_pilot_independent_acceptance_20261019/fixed_reference/tree/docs/research/track_b_g6_w2c_b3_limited_pilot_readiness_20261019/forward_evidence/normal \
  --fixed-tree docs/research/track_b_g6_w2c_b3_limited_pilot_independent_acceptance_20261019/fixed_reference/tree \
  --legacy-reference docs/research/track_b_g6_w2c_b3_v3_offline_acceptance_20261019/fixed_reference \
  --new-output /tmp/pilot-independent-new-normal
```

入力/referenceと重なるoutputや既存outputへの上書きを拒否する。exit0はVERIFIED_COMPLETE又は分類できたTECHNICAL_FAILURE、exit2はINSUFFICIENT_EVIDENCE/不整合等。科学GOを返すCLIではない。`build_independent_evidence.py` は18件を新規case_reportsへ再読するため、保存済み出力を持つcheckoutへそのまま再実行しない。

新しいS人工projection768 fake callsを別証拠として検証した。元の18件と旧47件に混ぜていない。16,896は静的な二条件候補で、正式科学bridgeは未実装。実G7 sampling・科学SOURCE採択・実authorization/grant/claimは0。

## 次の担当

ユーザー/GPTがscope/N、比較条件、seed、coverage、CPU estimand、trace/予算/STOPを判断する。Codexの次の技術作業としては、予約会計の修正版SOURCEとtrace I/O故障検査が先に必要。その後、別指示に基づくproduction bundle bridge/adapterの実装・人工受入・公開となる。今回のSOURCEを変更したり、実科学へ自動移行したりしていない。公開・保全確認後mandatory STOP。
