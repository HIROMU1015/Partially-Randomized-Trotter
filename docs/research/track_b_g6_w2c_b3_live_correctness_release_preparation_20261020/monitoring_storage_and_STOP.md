# 保存・監視・STOP

Global result1GiB/log64MiB、S/P1共通cap。全512 trialの原document/sidecar、exact event/weight/child、bits、angle/query、audit/F01、worker/source/runtime、global trace/state/counter、condition/global receipt、STOP/UNKNOWN、storage close、delivery/callerを保持する。
超過はSTOPし、Evidenceを勝手に間引かない。実科学サイズは未測定。人工raw容量を実容量保証にしない。
旧Storage V2は無変更。新version worker/pairはRESERVE/ENTER/COMPLETE/F01/cleanupを保持し、新authority・wrapperへつなぐ。
S失敗でP1未開始、P1失敗でS保存、STOP後ENTER禁止、UNKNOWN credit非返却、in-flight drainを維持する。
STOP条件はSOURCE/input、authority、seed/consumption、event/科学invariant、Codec、angle/query、worker、storage、UNKNOWN、byte/call、trace/receipt/callerの不整合。

host_outerはpreflight/admission前〜全join/STOP/close/receipt/delivery/caller close。parent CPU＋全joined worker CPUを一度だけ加算する。
coarseとcached all-call/return fineは同じcorrectness calls内の診断で、追加passやsampleはない。各exclusive partitionを別々にouterへreconcileし、parent/起動/observer/未観測分を残差として明示する。両partitionを加算しない。
SOURCE/constructor/sample/bits/packet-child-law/audit/angle/query/codec/IPC/observer/storage/RSS-HWM/I-O/STOP-drain-closeの観測構造を持つ。
最終outer観測のexport自身はpostprocessingとしてwindow外を明示する。wallはglobal monotonic outerで、pair wallを加算しない。
一時host PSIは記録し監視を再確立する。実OOM/worker異常/破損/科学的不一致/UNKNOWNの保護を保つ。共有設定・他者processへ変更はない。
実科学CPU/path率/保存量/IID/数学的atan/real durabilityはNOT_ESTABLISHED。

**LIVE_RELEASE_CANDIDATE_UNAUTHORIZED。実科学未実行。mandatory STOP。次の担当：ユーザー・GPT。**
