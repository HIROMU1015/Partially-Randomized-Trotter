# 受領科学レビュー

## 2026-10-15 一体再配分PF：有限step一件pilotの準備・科学実行未認可

[受領済み科学レビュー原文](quantum_chemistry_pf/qc_onebody_repartition_final_audit_scientific_review_2026-10-14.md)、[実装・比較・会計・caps](../research/qc_onebody_repartition_finite_step_preparation_v1_20261015.md)、[別明示実行承認への引継ぎ](../research/qc_onebody_repartition_finite_step_execution_handoff_v1_20261015.md)、[固定契約](../../artifacts/qc_onebody_repartition_finite_step_preparation_v1/2026-10-15/contract.json)、[source freeze21files](../../artifacts/qc_onebody_repartition_finite_step_preparation_v1/2026-10-15/source_manifest.json)、[入力identity](../../artifacts/qc_onebody_repartition_finite_step_preparation_v1/2026-10-15/input_identity.json)。
独占的1-RDM応答生成speedupは非採択のまま、完了済みGPT判断に従って有限stepの物理/controlled component有用性を別scopeで判別する準備。旧監査/STOPを上書きせず同じ方法差調査を反復しない。source19d5bd74、base9e288089。既存linear H4/1.00Å/STO-3G/DF rank12・generation順12、8spin/4spatial、Nα=Nβ=2/36dim、保存de7a coefficient/state packのみ。PRのL_D探索ではなく全density deterministic。旧PR S0 byte mismatch/held-out未開封不変。
先頭pair/spin-paired mean-zero X、一方向・最大6leading候補、E0/finite-h truthをselectorへ渡さずseal。original/response/known linear3構成×S2/S4×6hで36primitive cells、scalar correction/Richardson込み69records/276cost rows。corrected original/S4/known/subdivision/extrapolationを戻し、全対照にfixed-N scalar/basis共有/fusionを許す。新10%gateは条件付き2target×2priceのcontrolled primitive/Heisenberg modelのみ、情報取得/compile/T/全QPE費用と新規性は未確定。
登録外4-mode手書き37tests PASS、header/SHA/sourceだけのpreflight。real pack coefficient/state deserialize/RDM/expectation/eig/PF/compile0、旧27再実行0。新分子/ground-state preparation0、将来truth-H eigh1/spatial15/Schur36/seed36/独立PF再合成36・expm1024、worker/thread1、RSS512MiB/output32MiB/log2MiB、wall/CPU capなし、attempt1/retry0。認可target未発行false、旧940-call/C1認可を流用しない。
417旧files/150旧manifest records/non-result fields/global description FAIL/root137paths・旧科学source/results/認可/STOPを保持。N1STOP/N2HOLD/N3CLOSED/THRIFT現template終了、Track A/B/C1–C4/第二候補を再開しない。公開・GitHub照合後Mandatory STOP、次は固定one-shotの別明示実行承認。通常source完成だけで同じGPT科学レビューを要求しない。以下は履歴。

受領済みの外部/GPTレビュー原本を保存する。受領原文とCodexによる反映・現在状態を区別する。
rootにある未commitの既存レビュー/配置整理を、このbranchへ一括移動・mergeしない。

- [表現・算法構成探索](representation_exploration/README.md)：今回受領したmixed-access H4科学レビュー。
- [研究の現在状態](../research/研究概要・現状.md)：適用済み判断・成果固定・STOP。
