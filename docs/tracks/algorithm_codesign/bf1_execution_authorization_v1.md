# BF-1 v3 final review receipt / one-shot execution authorization

2026-10-05 JST。status: `APPROVED_FOR_ONE_SOURCE_BOUND_BF1_RUN`。

**本書は利用者提示の最終reviewと、別の明示的実行指示を記録したone-shot authorization receiptである。**
最終reviewを`APPROVED_FOR_ONE_BF1_RUN`として受領した後、実行scopeを確認する問いに対して、
利用者が「はい、進めてください」と明示的に応答した。
所定pathの正式JSONと本書だけを、固定source Sの直接のauthorization-only child Aへ追加する。
`science_execution_authorized=true`、`BF1_executed=false`（結果前receipt作成時点）。

## 1. Reviewするsourceと履歴

| 項目 | 固定値 |
|---|---|
| Repository | `HIROMU1015/Partially-Randomized-Trotter` |
| B worktree | `/home/abe/Project/prt-worktrees/track-b-algorithm-codesign` |
| Branch | `track-b-algorithm-codesign` |
| Source S | `e59344a564e70d64dc3ea39d640581c72676df31` |
| Parent | `296ec7e4c025f09e4bfda96e56e08d32388b81c5` |
| Remote | `origin/track-b-algorithm-codesign`、Sとの一致を確認済み |
| Local確認 | `PASS_SOURCE_BINDING_AND_EXISTING_SYNTHETIC_REPORT` |
| 最終review verdict | `APPROVED_FOR_ONE_BF1_RUN`（利用者提示の「相当」を正規化） |
| 明示的利用者実行指示 | scope確認への応答「はい、進めてください」 |

[Source commit](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/e59344a564e70d64dc3ea39d640581c72676df31)。
今回のcommitは明示したB関連20ファイルだけ。Track Aのsource/result/contract/status、共有`src/trotterlib`は変更していない。

旧sourceの[最終review](bf1_final_source_review_296ec7e.md)は
`REVISE_NUMERICAL_ASSEMBLY_GUARD_BEFORE_AUTHORIZATION`。
Sはそのassembly guardだけを修正する
[v3 amendment](bf1_assembly_guard_revision_v3.md)を含む。
[preregistration v1](bf1_preregistration_v1.md)と
[execution gate v2](bf1_execution_gate_revision_v2.md)の研究条件を維持する。

## 2. 最終review対象

以下は全てS内のblobを対象に確認する。本書自体はSに含まれないdraftである。

1. **Assembly allowanceのsource対応。** v3 §2のoperation数、絶対値scale S、
   `gamma(65536)*sqrt(256)`のuniform budgetが、各generatorとscalarをそれぞれ覆うか。
   full-H reconstruction discrepancyだけから個別errorを推定していないか。
   identity抽出で実際に使用する`eta,U`のresidualを計上しているか。
2. **Signed finite factorとerror伝播。** v3 §3、
   [numerics.py](../../../src/trottertracks/algorithm_codesign/numerics.py)、
   [pilot.py](../../../src/trottertracks/algorithm_codesign/pilot.py)を対応させる。
   negative timeを保持し、各factorへ`E_i`を渡し、
   `e_i <= N_i e_(i-1) + u_action,i`で伝播するか。
   full targetへ`sum(E_i)+E_c+E_sum`を渡し、scalar relative phaseを別計上するか。
3. **Synthetic evidenceの範囲。** 旧counterexampleでsignal/biasを変えずguardを修正したこと、
   非可換2×2、negative time、K2/K4、uniform/個別budget、full target、actual eigensystemの検査を確認する。
   conditional forward-error modelであり、interval certificationやH4の適格性を確定したものではない。
4. **公平性と原因帰属。** O/L/Fの同一family・同一32評価・同一初期点/refinement・同一I2 accessを確認する。
   F vs Lをnovelty-relevant比較とする。post-search cross-scoreはcache-onlyで、
   追加candidate/signal/cell取得0、primary classification変更なし。
   `SEARCH_REACHABILITY_OR_BUDGET_EXPLANATION_NOT_EXCLUDED`なら設計原理の発見とは解釈しない。
5. **実行境界。** [freeze.py](../../../src/trottertracks/algorithm_codesign/freeze.py)の
   source S → single authorization-only child A、source/plan/domain/test/environment照合、
   入力操作前のauthorization拒否を確認する。一回限り、retryなし、全outcomeでmandatory STOP。

入力loaderは[science_input.py](../../../src/trottertracks/algorithm_codesign/science_input.py)の
source textをreviewする。今回、loader実行や分子NPZのresolve/stat/hash/loadは行っていない。

## 3. Sへ結び付けたlocal確認

Sの77 source blobをsource planのbyte数・SHA256と照合し、working sourceとも全件一致を確認した。
plan/domain/test report/guard report/false authorization draftもSのblobと一致する。
既存v1/v2 preparationの9ファイルはparentからbyte単位で維持され、研究契約の17項目はv2と同じ。
Python/package/thread環境はv3 planと一致した。

[限定semantic report](../../../artifacts/track_b_bf1_preparation/2026-10-05/v3/synthetic_semantic_report.json)は
**46 passed in 1.51s**。synthetic small matrixとformulaのみ。
publication時に同じtestを再実行せず、pass時のsource-content sealとSの一致を確認した。
これはlocal checksのcommitへの対応付けであり、immutable CI・外部再現ではない。
元packetの`source_commit_bound=false`は準備時点の履歴として保持する。

| Identity | 値 |
|---|---|
| Plan canonical fingerprint | `602f6c0fbefbd48499fb0c2c9348753bd72141db7970f064572f214cc0fd63df` |
| Domain raw SHA256 | `caece92bd2d9f827b0f1f17281865869420bf923273ff294fd30cabd24b51efe` |
| Semantic report raw SHA256 | `11233d732d3d2d96a31ba3ee269caf96fda52e9e2f7bf855b1e403f04e271c80` |
| Guard report canonical fingerprint | `46ad20f42b6b8e7f46aea749bf65d49837ff37f7f652519475aa5fb2411c9231` |

[Guard regression report](../../../artifacts/track_b_bf1_preparation/2026-10-05/v3/assembly_guard_regression_report.json)：
旧Suzuki5 1×1 witnessのaxis bias errorは`[3.1553663404348953e-6, 2.400019153858679e-6]`。
旧`u_signal=2.593050117669466e-6`、新`u_signal=0.0013425077736177234`。
bias自体は変わらず、新guardが両axisを覆う。
新guardはこのtoyでは保守的であり、H4のguard幅・feasibility・係数・改善率は未評価。
実行時に数値guardやcapが原因で不適格・INCONCLUSIVEとなっても、閾値や上限を緩めない。

科学入力操作、science signal、Hamiltonian生成、trajectory、circuit build/compile、GPU query/use、
full test suite実行はいずれも今回0。Track Aのruntime/cacheは流用していない。

## 4. 維持するone-shot contract

Known/developmentのH4 linear 1.00 Å、STO-3G、8 qubits、DF rank 12、
generation-prefix `L_D=3`、one-body込み4 D generators、`T=0.8`。
M2 1.30 Åを新held-outとして利用しない。

5-stage symmetric fourth-order family、O/L/F各32 coefficient evaluations、同じ16 starts +16 refinements。
`q={1,2,4,8}`、`R_bud={5,10,20,40,80}`、固定rounding allocation、K2と事前triggerによるK4。
fixed refsはnative S2、Yoshida3、Suzuki5、Morales v3 21-stage。
primary `epsilon=0.01, alpha=0.05`、BF-Cのprimary routeはfinite action ratio `<=0.95`のみ。
bridge `epsilon=0.05`では係数再探索しない。

上限は1 worker/各thread 1、CPU 8h、wall 4h、RSS 4GiB、aggregate RSS 8GiB、
output 128MiB、ideal 400 cells、finite 4000 cells、science retries 0。
全outcomeで直ちにSTOPし、BのRQ・新規性・着地点を全面再評価する。BF-2へ自動進行しない。

## 5. Review後のauthorization publication

reviewで問題があれば、science実行せずsource修正と新S固定へ戻る。
最終reviewの`APPROVED_FOR_ONE_BF1_RUN`と、別の明示的利用者実行指示が揃った場合に限り、
Sの直接の子Aへ次の新規追加pathだけをcommitする。

- 必須：`artifacts/track_b_bf1_authorization/2026-10-05/authorization_v1.json`
- 任意receipt：`docs/tracks/algorithm_codesign/bf1_execution_authorization_v1.md`（本書）

本書はAで使う所定receipt pathに置いている。
review identity・判定と別の明示的指示を記録済みであり、上記二pathだけでAを確定する。
source Sの上に別のreview-only commitを作らず、S内の既存source/packetを変更しない。
JSONへA自身のSHAを書かない。AのSHAはlaunch時にGitから読み、one-shot registryとresultへ記録する。

## 6. 受領reviewのidentityとscope

review対象Sは`e59344a564e70d64dc3ea39d640581c72676df31`。
公開依頼書はreview-only branchのcommit
`e911db6b409f9814386f31b99c6a427cf2771501`にある
[最終review依頼](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e911db6b409f9814386f31b99c6a427cf2771501/docs/tracks/algorithm_codesign/bf1_final_review_request_e59344a.md)。
review commitをscience branchへmerge/cherry-pickしない。

受領元はこの会話で利用者が提示した最終review文。
原文の判定表現は「`APPROVED_FOR_ONE_BF1_RUN` 相当」、
実行境界の表現は「これはまだ実行そのものではなく」。
このscopeを維持してreview判定を正規化し、実行指示としては扱わない。
reviewerのmodel identityは提示されていない。別会話のreviewは独立のscience replicationではない。

受領reviewはassembly guard修正、cross-score、single authorization-only child、mandatory STOPを認め、
BF-1前の研究条件追加を要求していない。H4で数値marginが大きすぎる場合はINCONCLUSIVEで停止し、
閾値・上限を緩めない。BF-Cでもsearch reachability説明が残る場合はmethod deltaを確定しない。
全outcomeでBF-1直後にSTOPし、BのRQ・新規性・着地点を全面再評価する。

## 7. 明示的実行指示とchild commit確定の順序

[所定pathの正式JSON](../../../artifacts/track_b_bf1_authorization/2026-10-05/authorization_v1.json)は、
source S、review判定、plan/domain/test identity、別の明示的指示、one-shot/STOP制約を記録する。
`explicit_user_execution_instruction="はい、進めてください"`、`science_execution_authorized=true`。
旧v1/v2/v3 preparation packetは変更していない。

確認したscopeは「source e59344a の固定契約でBF-1を一回実行し、終了後mandatory STOPする正式な実行指示」。
利用者の原文応答と確認scopeをJSONの`execution_instruction_provenance`に別々に保存する。
authorization準備時刻は`2026-10-05T10:16:58Z`。

固定Sの[verify_launch](../../../src/trottertracks/algorithm_codesign/freeze.py)は、
非空の`explicit_user_execution_instruction`を要求し、さらにauthorization JSONのworking bytesが
authorization-only child Aのcommitted bytesと一致することを要求する。
明示的指示を記録する前にAを確定して後から追記すると、この固定仕様と両立しない。

従って、sourceを変更せずに進める順序は
**review通過 → authorization準備 → 別の明示的実行指示受領・記録 → A確定 → BF-1一回**。
本receiptとJSONをAへcommit固定した後、source/input/環境のlaunch gateを照合して、一回だけ実行する。

準備時の確認は77 sealed sourceとSの一致、plan/domain/test identity、固定環境の一致。
disabled draftは`BF1_EXECUTION_NOT_AUTHORIZED`で拒否され、memory-onlyのenabled probeも
指示がnullのため`Missing separate explicit execution instruction`で拒否された。
probeはfileへ保存していない。これは明示的指示受領前の準備検査の履歴である。
本receipt作成時点ではrunner、science loader、one-shot registryのconsumeは行っていない。
科学入力操作、science signalは0（結果前）。Hamiltonian生成、trajectory、circuit/compile、GPUは
実行中も禁止する。既存development snapshotだけを一回読み、結果に関係なくmandatory STOPする。
