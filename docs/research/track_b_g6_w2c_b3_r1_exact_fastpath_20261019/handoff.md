# Q1–Q8・次の担当

**Q1：fused canonicalは実装・一致確認できたか。** はい。driverへ最小差分接続し、元bytes/hash/phase/weight/word、Unicode/surrogate/DEL、key衝突fallback、F01失敗文書まで人工確認した。

**Q2：冪漸化式は厳密一致したか。** 人工有理数でendpoint/partial/remainder、全enclosure Evidence/hash、outward root、radius受入/失敗detailsが一致。固定F3は未変更。本番Full sessionへは未接続。

**Q3：組合せのCPU/wall改善は。** 5反復paired medianで96bitは約1.033/1.033×、256bitは約1.007/1.007×。後者rangeは1を跨ぐ。serialization単独は約1.379×。実G7/サーバー倍率ではなく、小さな全体改善の安定優位は未確立。

**Q4：回帰は何件。** 旧275＋新canonical/guard36＋angle28＝**339 PASS、FAIL/ERROR/SKIP各0**。旧F01/Admission/Host/STOP、correctness gateを保持。全ID/log/SOURCE SHAを保存。

**Q5：R1互換と必要変更は。** fusedは人工検証A候補。新driver/helper完全SHAとdependency/observer pinを新versionへ含める必要。recurrence/組合せはBで、新しい明示的F3 backend/API・Full/canonical接続・guard/evidence/source pin回帰が必要。原F3へのmonkeypatchなし、正式契約は未変更。

**Q6：最優先の推奨。** fusedのみ。fixed F3を変更せず実装でき、単独microの効果を確認。recurrenceは同値性確認済みだが本番接続と利益の再確認を別作業へ分ける。

**Q7：Linux固有の未実施項目は。** 本環境はLinux6.8でfork/proc/filesystem依存を含む全339を実行、SKIP0。server Python3.12/EPYCで新SOURCE版のexact codec・F01/host/STOP/source pin、fine profiling path、人工4構成を再確認するのが後続候補。電源断/媒体故障の完全証明や実G7 guard/runtimeは未検証。

**Q8：新科学計算・実authority/claimは。** 0件。原precision/root/Evidenceの実行は承認された人工有理数テスト・benchmarkだけ。G7 sampling、実F3 session、保存科学値再解析、native compile、新レビュー、実authorization/claim発行・消費は行っていない。

次の担当は**ユーザー**で、fused技術候補を採択してサーバー再検証・次versionのSOURCE/pin固定を指示するか判断する。
指示後の**Codex**がSOURCE照合・必要な人工回帰/性能再確認と公開固定を行う。これは提案する後続で今回未実施。
recurrence本番統合や正式science release/contract固定、実authority/claim、実G7 correctness/31,710,848 callsは別途明示承認が必要。
公開・保全確認後 **mandatory STOP**。
