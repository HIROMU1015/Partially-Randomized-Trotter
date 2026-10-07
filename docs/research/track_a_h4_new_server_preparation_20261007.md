# H4 新hostのGit取得・環境監査と解決案（2026-10-07）

[資料入口](../../artifacts/resource_applicability/track_a_h4_new_server_preparation/2026-10-07/README.md)。
現在は **H4_NEW_HOST_SOURCE_RECEIVED_ENVIRONMENT_DECISION_REQUIRED_STOP**。
source取得・読取監査・実装案の固定まででSTOPし、監視修正完了・production成功・起動可能とは扱わない。

## 取得・配置

Origin HIROMU1015/Partially-Randomized-Trotter。元branchはtrack-a-h4-lazy-identity-run05-20261006、
remote/取得commitは8f77bebf99c5bd15fa6419c1f58556c3bd2837a9で一致。
追加資料branch track-a-h4-new-server-handoff-20261007はc0a5b9692778878865cfa31d7df9a3bcc259f6c4。
science6d365257770e99022b91d6a38dbee49ee0077503、generation049e69919af16ad29a67a217dc7a407d6b1754a6、
contract b662dbd72e49fa713a25c716f323843e547e973bの存在・ancestorを確認。
元commitから専用branch/worktreeを作成し、code差分0の追加資料branchをfast-forwardした。
専用branchはtrack-a-h4-new-server-monitor-preparation-20261007。

- Repository: /home/AbeHiromu/projects/partially-randomized-trotter
- Checkout: /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h4-monitor-preparation-20261007
- Evidence: /home/AbeHiromu/projects/h4-handoff-evidence/20261007
- 実input/output/実行Python: 未固定。旧pathへのsymlinkなし。

GitHub HTTPS fetchで受領したため、source/資料にSSH認証やGit bundleは不要になった。
新hostのshared/system/他job設定変更なし。旧source/plan/auth/review bytesを保持した。
公開HANDOFF_STATEは利用者提示のJSONと内容一致し、run05 STOPが最新状態。

## 環境・資源

新host GPUA100-gleap02、kernel6.8.0-50-generic、64 physical cores、online/sched affinity0-63、NUMA0-7、RAM約504GiB。
旧referenceはGPUA100-gleap-01、kernel6.8.0-49-generic、128 cores、RAM約1008GiB。
Python3.12.3は一致するが旧venvs/trotter-commonは存在しない。
既存private venvのcandidate:
/home/AbeHiromu/projects/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/venv/bin/python

45 distributionsは存在し、旧freezeに対して18 versions/45 RECORD hashesが異なる。
監査対象installed source11件は全match、source19件もbyte hash/science blob/handoff blob全match。
compiler28 defaultsはAST比較、plugin61 entrypointsはmetadata比較でmatch。compiler optionsは旧15 fieldsを機械転記。
Qiskit/plugin/science import・transpile・新人工testsは行っていない。binary/wheelやproduction/compiler equivalenceは未検証。

native read-only memory observerは成功し、全可視祖先と真のv2 root policyを確認した。
user.sliceの累積oom=1は履歴であり時期/帰属は不明、観測PSI avg10は0。
native /proc allowed mask0-255とonline/sched affinity0-63は異なる観測で、CPU使用認可ではない。
12 workersは候補、allowed_cpus=[]。資源観測を予約・launch gate PASSとみなさない。
専用evidence filesystemはext4、空き約416GiB/約2.2億inodesの観測記録。user/group/project quotaはunknown。
固定output filesystemとobserver/log込みstage積算は未完了。3.5GiB/560000inodesは旧planningで、10GiB全量空きを新必須条件にしない。

## 解決案とSTOP理由

[新server指示](handoffs/h4-new-server-20261007/NEW_SERVER_CODEX_INSTRUCTIONS.md)の
「差異は機械可読に保存し、既存のprivate環境を選ぶか、解決案を提示してSTOPします」に従う。
選択肢Aは既存private venvを変更せず、新environment/compiler profileとしてbindingし直す案。
選択肢Bは旧45 versionsをreferenceとするH4専用環境の案。Bのinstallは現在認可されておらず実施しない。
Aでも旧environmentと同一とは呼ばず、差分受入れと新profile/人工equivalence evidenceが必要。
既存/shared/system環境を変更しない。

[SOURCE_FIX_PLAN](../../artifacts/resource_applicability/track_a_h4_new_server_preparation/2026-10-07/SOURCE_FIX_PLAN_v1.md)に、
ndarray/回路のstreamingと独立observer/既存workerへ移す代案、bounded tests、phase/interval/staleness記録を示した。
独立observer例AS256MiB/RSS64MiBを追加すればadmission候補120.25GiBで、追加role/resource reviewが必要。
未承認費用を旧120GiBに含まれるとみなさず、既存capsは維持して実装案でSTOPする。
source19は変更0、新人工tests0、追加transpile0（旧累積28/64）、worker/observer launch0。
5秒制限・fail-closed・own-runだけ停止は今後の必須条件で、解決済みとは宣言しない。

## 入力・予算・認可

NPZ6/generation-freeze/journals/ledger/control/logは未受領で、実file hash照合も未実施。
metadataの期待hashとbyte-identical受領manifestを別送後に確認する。入力再生成で補わず、人工testsには分子入力を使わない。
現在run05完成compile0/signal0、全old owned終了は旧HANDOFF_STATEの記録。欠けている旧runtime/controlを新hostで再証明したとは扱わない。
旧PIDのkill/旧partial result再利用なし。

carry20 actual invocations/165214360 bytes/5466.188392877579 s、残74764。
新journal carry rowは未作成で、後の明示run開始時に別途charge。driver/worker AS/RSS8GiB、headroom16GiB、monitor5秒、
10GiB/72h/74784 capsは変更しない。old approved=trueをnew-host承認へ転記しない。
新SOURCE_COMMIT/plan/input/output/Python/CPU/observer bindingは未seal、approved=false。
不足するidentityを埋めた実行commandは作らず、absolute_launch_command=nullとする。

次に環境方針と監視実装方式を確定し、source修正→限定tests→source/plan固定→新CPU認可/最終review/明示launchを進める。
今回production・入力生成・GPU query/use・taskset・共有環境/他job変更・install/upgradeは全て0。報告後STOP。
科学scopeはH4 linear neutral singlet/STO-3G/DF12、8system+ancilla1、6固定距離、T0.8、218templates、32paired、1308signals/74784logical wrappersを維持。
new source/environment結合でseedが変わる場合を隠さず、old partial/random結果と混ぜない。


## 2026-10-07 H4新host A案 source/人工検証固定・本計算未認可

[監視修正・32人工tests](track_a_h4_new_server_monitor_fix_a_20261007.md)。既存private venv不変で準備用A案採用。
旧source19は3変更/16不変、新module込み25 closure。256×256人工matrix＋9-qubitの旧/new byte/digest一致、独立observerのGIL/GC観測・I/O delay/EOF/所有・資源境界を検証。
SOURCE `b2a5ad89e8b39d72716f7ddb17d263bd0cdedb45`、production/追加transpile0、旧28/64・benchmark128保持。環境18 version差、旧45 raw-reference RECORD差保持・normalized22差を明記。入力6/freeze/runtime/control未受領。
observer AS256MiB/RSS64MiB/admission120.25GiB、容量5.5625GiB案は未承認。allowed_cpus=[]/approved=false/runtime_authorization=false/launch=null、STOP。科学成果/原稿/Track Bと旧資料を保持。
