# Phase 2D-B：限定11件の科学入力binding結果

2026-10-23 JST。**B_BINDING_COMPLETE_MANDATORY_STOP**。固定runnerの実statusは **B_COMPLETE_MANDATORY_STOP**、binding summaryは **B_INPUTS_BOUND_F1_REVIEW_REQUIRED**。全11caseが各1attemptで完了、failed0。科学実行は停止済みである。

固定P0の4 mode・2粒子・LD1・残差density-square二fragmentの合成model。分子geometry／AO basis／分子DF rankは非該当。identityはfaithful_identity_in_tail。4armのF1用lawはT=1/5、q2、delta=1/10（この一点のみ）、r2、h=1/20、K2。catalogはT=delta=1/20、q=r1、K2。

今回得たのはphysical/catalog input、SS/SP/PS/PP canonical armとexact lambda、5件のfinite-RTE scalar metadata。F1の正しさ、F2資源比較、physical bias、独立新規性・優位性の検証は行っていない。

| arm | L | exact lambda | 将来F1 E=L+L³ |
|---|---:|---|---:|
| SS | 14 | `848/5445` | 2,758 |
| SP | 58 | `7180484777623/35269733400000` | 195,170 |
| PS | 26 | `1118879908/5636255625` | 17,602 |
| PP | 51 | `182339128250602481/876205986856200000` | 132,702 |

- [実行報告](EXECUTION_REPORT.md)／[case状態](CASE_STATUS.md)／[STOP](STOP_REPORT.md)
- [入力・law要約](INPUT_AND_LAW_SUMMARY.md)／[保存済みmetadata](INPUT_LAW_SUMMARY.json)
- [SOURCE・環境provenance](SOURCE_ENVIRONMENT_PROVENANCE.md)／[台帳](PROVENANCE.json)
- [出力hash一覧](OUTPUT_HASH_MANIFEST.json)／[内部call対応](CALL_ACCOUNTING.json)
- [F1引継ぎ](F1_HANDOFF.md)／[件数descriptor](F1_COUNT_DESCRIPTORS.json)
- [handoff](handoff.md)／[公開receipt](PUBLICATION.json)／[保全照合](VERIFICATION.md)

科学raw40file（39JSON＋log）は新規private outputで保全し、ここには許可された要約とrelative path/hash/bytesだけを公開する。承認原文・grant・marker・内部path・private logsは公開しない。

科学SOURCE `3de42d3a3efe9ebc74dde8a1d0e1f1bdbadd607c` と、この結果文書の公開commitを区別する。次の担当はユーザー。F1の残るSOURCE／price／interval／環境／case契約の準備方針を判断する。F1/F2の新規科学実行はそれぞれ別の明示承認が必要。
