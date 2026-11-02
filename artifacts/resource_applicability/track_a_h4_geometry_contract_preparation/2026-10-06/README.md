# H4 geometry contract preparation v1

`H4_GEOMETRY_CONTRACT_REQUIRES_DECISION_SCIENCE_NOT_AUTHORIZED`

2026-10-06 JST。handoff `2a80f1d5d5e5734e51d970b2b6822cd2543fd596`、
base `c2ab34fed49bb1fb104d39fe83b36858a2c92c2a`から作成した契約準備。
準備完了時点は**local・未commit**であり、後続の明示依頼により専用branch
`track-a-h4-geometry-contract-20261006`でのcommitとoriginへのnon-force pushだけを公開対象とする。
入口：[正式契約案](CONTRACT_DRAFT_v1.md)、[review依頼](REVIEW_REQUEST_v1.md)。

6距離0.70/0.80/0.90/1.10/1.40/1.60 Å、218 template/点、
1,308 signal slots、74,784 logical wrappers/actual invocation cap、CPU workers最大12を固定。
科学source/input-bound planは未seal、本計算・source port・execution authorization作成は未認可。
SCF/DF、order/solver/gates、master seed、memory/wall/outputの4判断は未解決でSTOP。

| artifact | 役割 |
|---|---|
| [scope_v1.json](scope_v1.json) / [handoff_prompt.md](handoff_prompt.md) | 公開handoff blobをbyte保存 |
| [zero_compute_plan_v1.json](zero_compute_plan_v1.json) | 条件固定とnull科学identityを分けたsymbolic plan |
| [review_decisions_v1.json](review_decisions_v1.json) | source根拠・推奨案・必要判断 |
| scope/plan/result/checkpoint/completion_ledger/numerical_circuit schema v1 | 構造schemaと未認可・STOP条件 |
| [contract_validator_v1.py](contract_validator_v1.py) | pure JSONの意味論identity/owner/digest検査 |
| [contract_tests_result_v1.json](contract_tests_result_v1.json) | 基本111 positive/mutation tests |
| [additional_contract_checks_v1.json](additional_contract_checks_v1.json) | 有効な別scope ownerを受理後、cross-scope reuseを拒否 |
| [synthetic_records_v1.json](synthetic_records_v1.json) | 架空record/ledger/circuit declaration。science evidenceではない |
| [environment_binding_audit_v1.json](environment_binding_audit_v1.json) | 既存45 dependency metadata・pluginの一致 |
| [static_source_findings_v1.json](static_source_findings_v1.json) | 生成順と汎用weight-sortの差を含むsource監査 |
| [identity_access_audit_v1.json](identity_access_audit_v1.json) / [final_audit_v1.json](final_audit_v1.json) | 公開25 files・旧247 source・保存6 JSONの不変性と0 access |
| [artifact_manifest_v1.json](artifact_manifest_v1.json) | 新規bundleと関連index/docのfile digest |

検査は既存CPU Pythonのstdlib/jsonschemaと合成JSONだけ。Qiskit/science library import、
分子snapshot・runtime/cacheアクセス、signal/sampling/build/compile、GPU、共有環境変更、
他job変更、commit/pushは準備時点で0。監査JSON・scope・handoff内の未commit/commit-push 0は
この準備段階の履歴を保持しており、今回の公開依頼によって過去のカウンタを書き換えない。
過去のsynthetic128件は比較120件（30 task × workers 1/6/12/16の4条件）と、
残り8件（axis/phase semantics 4件＋full-operator検査4件）。保存記録だけを継承し、新規transpile0。
公開準備draft・旧source/result/status/validation manifest/原稿/図/Track Bは不変。

test sourceと実行commandは結果JSONに記録。新しく再現する場合は新しいprivate directoryへ
sourceとschema/planだけをコピーする。既存resultへ上書き・automatic retryしない。
本bundleはlocal検査記録でありimmutable CI/独立外部再現ではない。
