# 実装した受入harnessと候補の証拠

`g6_w2c_b3_integration_acceptance`は各checkにID、対象SOURCE、固定commit、contract/schema、期待・観測、判定、根拠、未解決を保存する。
判定はPASS/FAIL/PENDING_EVIDENCE/REQUIRES_CONTRACT_CHANGE/NOT_APPLICABLE。未実行や未確認をPASSにしない。

| 区分 | 実装・実行した検査 | 証拠 |
|---|---|---|
| A SOURCE/contract | Driver/Host/Admission/Guard等13完全SHA、input/policy/A6/seed/16cells/N/二pass/保存/one-shot/STOPの10投影項目、変更時の拒否 | A_PIN_* / A_CONTRACT_*、旧Readiness/Admission/Host回帰 |
| B sampling同値 | mock event、R/C/A、trial ID、bits、phase、weight、word、A/B、correctness prefix、二pass | 既存8正常ケースの固定人工reference＋新actual Stream fixture/cross verifier |
| C canonical/hash | exact int/Fraction、巨大整数、Unicode/surrogate/DEL、key collision、chain/file、failure/partial tail | canonical_checks callback API、新codec検査、旧F01＋新fault tests |
| D Admission/STOP | 未承認entry・digest/source不一致・重複scope/claim拒否、reserve後spent、host外側STOP、STOP保存失敗、retry0/resumeなし | 今回再実行した旧275件の全ID/log/SOURCE SHA |
| E 計測 | coarse/fine別reconciliation、parent/child、共有setup、両pass、null、memory/I/O、observer path/function変更 | cost fault tests、observer_check、旧Meter/FunctionObserver人工回帰 |

`canonical_checks(candidate_callable)`は人工callbackを明示的に渡すAPI。今回のcallbackは既存reference codec。固定fused SOURCEをimport・統合していない。
`identity_checks`はSOURCE bytesと契約投影を比較し、契約差をREQUIRES_CONTRACT_CHANGEとする。trace一致でもこれをPASSへ変更しない。
`observer_check`はpath/function target集合が変わると対応再設計を要求する。異なるcall treeの費用一致までは証明しない。

[受入例JSON](artificial_example/integration_acceptance_report.json)は**385 PASS／13 PENDING_EVIDENCE／FAIL0**。
385は360 runtime test行＋13pin＋10契約＋2機能statusの受入行数。360 testsという実収集件数と混同しない。
SOURCE pinを検査したことは未知candidateの実行結果を得たことではない。各runtime testは対象となる固定baseline test SOURCE/logのSHAを伴う。

fused候補 ad3117d287c4c28c9b861b3b8dffa7bfbba407e1 の以前の339 PASSは参照履歴。今回その候補を再実行した件数へ加算しない。
候補runtimeと新helper observer mappingの2項目はPENDING_EVIDENCE。正式採択には新SOURCE/pin/observer対応、必要なcandidate回帰、ユーザー決定が別に必要。
並列11項目もPENDING_EVIDENCE。実SOURCE未公開の欄はfixed_commit=null。契約変更と証拠不足は別分類であり、未公開parallelを既存R1互換と認定しない。
