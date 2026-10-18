# Track A H8：新PR2候補・独立n48確認 run01

2026-10-18T23:35:44.091875+09:00 の起動snapshot。**RUNNING、結果未完了**。

SOURCE：`1549b6f891a15ff48379996c34b344118ba2606c`。公開済み設計434e67a8…と保存値結果64c85f7a…を保持した独立branchで実装した。
利用者の固定2候補×48軌道・192 full-wrapper compile・必要control・限定会計の実行承認を、個別grantへ結合した。
技術seal自身のapproved/runtimeフラグはfalseを保ち、実行承認は別private grantで管理する。

| scope | PR | ε | 固定PF | 新軌道 | worker slots |
|---|---|---:|---|---:|---:|
| H8_CONFIRM_P8_Q2 | B2_p8_q2_2nd_r4_K4 | .05 | B0I_p11_q1_2nd、実測42,268 RZ/軸 |48|2|
| H8_CONFIRM_P7_Q16 | B2_p7_q16_2nd_r4_K4 | .0005 | B0I_p15_q4_4th、実測668,636 RZ/軸 |48|42|

共通条件はlinear H8/1Å/STO-3G/tol-only1e-8 DF28/16spin modes/sector4900/T=.8/P=0。
元S2・有限K4 RTE law、DF順序、full-Fock scalar一回適用、reference不回転、ordinary ancilla、symmetric-directional controlを維持。
既存preparationを復元し、SCF/DF/state/reference/signalの新規生成は0。compiler optionsも元のopt1/seed17/rz,sx,x,cxを保持。
旧profileの原file byte一致・旧compiler output完全同一性の未検証という限界を維持し、現在のPython/runtime/providerのbyteを別途sealした。

各候補の代表1軌道をfresh control4probe・event oracle1回込みで先行し、native exit0/同親wait4・両軸/phase/law/normalization gateに合格後、残47本を展開する。
現snapshotでは各代表のnative evolutionが開始済み。最初から全workerが同時稼働しているという意味ではない。
2候補のgrant/ledger/output/STOPは独立し、1候補の失敗で他候補を停止しない。

新cohortは`INDEPENDENT_UNCONDITIONAL_CONFIRMATION_N48`。固定SOURCEとseed-plan digest、候補/replicaから96masterを導出し、
旧探索/確認/pilot由来の除外receiptとA128のmaster/派生を合わせた5,000seed値、および新2候補相互との衝突を拒否した。
namespaceが異なるだけで数学的独立性を証明したとは扱わない。探索n8・旧n32・条件付き標本は合算しない。

主評価は保存usual signed bias・倍率1の軸別shotによるpaired費用平均/SD/SEと固定PF比較。
副評価は同一標本を保存された経験的非相殺shotで重み付けする記述的評価。
**p8/q2の固定PF p11/q1は非相殺条件で不適格**のため、その副評価の比較・削減率はnullとし、他PFへの自動差替えをしない。
p7/q16の固定PFは副評価でも保存費用が存在する。通常/副評価を4条件の保証へ読み替えない。
48本全ての受入後だけ会計し、途中平均で終了/延長/候補変更しない。未観測希少費用は0補完せず、点費用再現と母平均優位未確定を分ける。

新adapterの人工18検査PASS。seed衝突/旧cohort混入/欠測/paired SE/exact tie/無効grant/normalization-law/control受入、
最小人工2childのnative wait4・所有不一致拒否・own cleanupを確認。人工compileは0、旧campaignの反復なし。作者の技術検査であり独立科学reviewではない。

起動前にSOURCE9closure、private input、既存provider/runtime、未使用output/lock、CPU/memory/cgroup/OOM/pressure、
filesystem block/inodeを確認し、両startup gateはPASS。worker数はfresh CPUとRSS見積りから決め、
memory admissionはcontext/2倍観測peak/headroomの**見積りであり実使用量・上限ではない**。強制wall/AS/RSS capは追加しない。
quota toolは提供されておらず無制限とは断定しない。既存site方針を保持し、ENOSPC/EDQUOTをSTOPとして保全する。
一時的PSI/CPU競合/無関係worktree更新で強制停止せず、実native/IO/所有/科学不一致はown-run STOP・全回収する。
既存venv・共有設定・他job・H6高次PF研究の変更は0。GPU query/use0、新PF compile0、新shot最適化0。

[seal/検査/closure bundle](../../artifacts/resource_applicability/track_a_h8_new_pr_confirmation_run01/2026-10-18/README.md)。
[runner](../../scripts/h8_pr_confirmation.py)／[metadata sealer](../../scripts/prepare_h8_pr_confirmation.py)／[native witness](../../scripts/launch_h8_pr_confirmation.py)／[人工tests](../../tests/test_h8_pr_confirmation.py)。
private運用原本・有効grant・seed値・raw科学入力・実ledgerはcommitしない。

次の担当：Codexの固定runnerが各48本の完了またはfail-closed STOPまで継続し、限定結果を保存してmandatory STOPする。
新sampling追加・refit・自動retry・次stageは行わない。ユーザーによる追加操作は現時点では不要。
