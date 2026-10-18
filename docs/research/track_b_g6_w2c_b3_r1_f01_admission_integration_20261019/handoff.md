# Q1–Q8・次の担当

**Q1：統合は完了か。** 今回承認された技術範囲は完了。固定F01 driver、R1 admission、新hostの実SOURCEを接続し人工回帰した。科学実行用最終releaseは未採択。

**Q2：接続とSTOPは動くか。** with port内のpreflight→実Boundary/execute接続を実装。normal/exception、receipt直後/output前でもSTOP。STOP保存失敗がprimaryを置換せず、予約はspentのまま。

**Q3：人工回帰は何件か。** 既存69+30+36+48+26=209、新統合66、**計275 PASS、FAIL/ERROR/SKIP各0**。全ID/log/SOURCE SHAをmachine結果に記録。正常8ケースはsubcases。旧183/181の不整合は旧記録を残して新資料で整理。

**Q4：R1科学条件は維持したか。** input/policy/A6/seed/N/law/phase/weight/bit順、24 constructor/prepare3/preflight9、cache/保存/二pass/固定順、31,710,848予算、retry0/resumeなし/STOPを維持。mockの16/axis profileは人工test内だけで科学N変更ではない。

**Q5：SOURCE/branch/commit/候補は何か。** branchとdriver/admission/hostの完全SHAはtechnical_release_candidate.md、全source/test SHAと技術候補digestはintegration_manifest.json。固定公開commitはpublication_receipt.verified_content_commit。最終science release/digestはnull。

**Q6：science前のblockerは何か。** 今回のF01/host技術統合blockerは解消。高速化公開結果の確認と採否、必要なら差分再統合・回帰、最終science SOURCE/正式契約digest固定・公開preflight、trusted実approval bootstrap/store構成と別途science実行承認が未完了。実入力correctnessも未実行。

**Q7：高速化結果の後に何を照合するか。** performance_handoff.mdのF01/authority/atomic STOP、law/bit/seed、setup/cache/保存/schedule/予算への差分を確認。ユーザーが採択した別versionで全pin/host/正式契約を更新し、必要な人工回帰後に最終freezeする。

**Q8：新science/実authority/claimは0か。** はい。実G7 import/correctness/sampling/CPU profiling、保存科学値再解析、native compile/synthesis、新科学計算、実authorization/claim発行・消費はすべて0。人工temp/owned processesだけ。

次の担当は**ユーザー**で、別サーバーの高速化固定公開成果を確認し、採用・差分統合と最終固定作業を指示するか判断する。
その指示後の**Codex**は差分照合・必要な再統合/人工回帰・最終SOURCE/契約の公開固定を行う。これは提案する後続作業で今回未実施。
その後の科学実行と実authority/store/claim発行・一回消費は**ユーザーの別途明示承認**が必要。
公開・保全確認後 **mandatory STOP**。高速化採択、最終science release採択、実authority/claim、実correctness/全N/科学レビューへ進まない。
