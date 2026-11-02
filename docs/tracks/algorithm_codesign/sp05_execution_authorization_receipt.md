# SP-0.5 一回実行authorization・明示指示receipt

source S：`65f6fcdb3dc1ad8bfccfaee6e1413336aef91184`。
source review・authorization準備の記録P：`b0fa5afefd437aa0953593af2e5981c0826363a7`。
Pは参照資料として保存し、実行HEAD／親には使用しない。
今回の実行用authorization-only commit AはSの直接の子で、変更はauthorization JSONと本receiptのみ。

branch／worktree：`track-b-sp05-one-shot-execution-20261006`、
`/home/abe/Project/prt-worktrees/track-b-sp05-one-shot-execution-20261006`。
contract SHA-256：`ee126a0fcde4aa5eff96dfb5835647deec42150c27703e86a3023164b888d09f`。
source、contract、tool identity、target、catalogue、precision、J判定、capsは不変。

以下に利用者の明示指示を原文で記録する。同じ全文をauthorization JSONの
`explicit_execution_instruction`へ記録し、UTF-8 SHA-256も保存する。
一回実行後は成功・失敗にかかわらずSTOPし、必要最小限の保存結果・監査だけをGPTへ公開する。

## 利用者の明示実行指示（原文）

<!-- BEGIN USER EXECUTION INSTRUCTION -->
source `65f6fcdb3dc1ad8bfccfaee6e1413336aef91184` の固定契約に基づき、SP-0.5 synthesis-economics gateを**一回だけ実行してください**。

実行にあたっては、以下を厳守してください。

- `b0fa5afefd437aa0953593af2e5981c0826363a7` はsource review・authorization準備の記録として保持し、実行HEADには使用しない。
- source `65f6fcdb3dc1ad8bfccfaee6e1413336aef91184` から別の実行用branch/worktreeを作る。
- source Sの**直接の子**としてauthorization-only commit Aを作る。
- Aで変更してよいのは、
  - `artifacts/track_b_sp05_economics_preparation/2026-10-06/authorization.json`
  - 必要なら `docs/tracks/algorithm_codesign/sp05_execution_authorization_receipt.md`\
    のみ。
- authorization JSONを
  - `status = APPROVED_FOR_ONE_SP05_RUN`
  - `science_execution_authorized = true`
  - `source_commit = 65f6fcdb3dc1ad8bfccfaee6e1413336aef91184`
  - `runs = 1`
  - `retries = 0`
  - `mandatory_STOP = true`\
    とし、この明示実行指示をreceiptとJSONへ正確に記録する。
- contract、source、pygridsynth identity、target、catalogue、operator error、J判定、資源上限を変更しない。
- 8 target／23 planned synthesis keys、exact π/4 catalogue一つだけを使用する。
- 第二catalogue、target追加、precision変更、threshold変更を行わない。
- registered target synthesisとSP-0.5 J評価は**一回だけ**行う。
- partial failure、timeout、memory cap、numeric inconclusiveの場合もretryしない。
- 分子入力、DF Hamiltonian、trajectory sampling、wrapper pilot、16-cell pilot、GPU処理へ進まない。

実行終了後は、結果が

- `PRIMITIVE_TRADEOFF_EXISTS`
- `NO_PRIMITIVE_TRADEOFF_IN_REGISTERED_SET`
- `INCONCLUSIVE`

のいずれであっても、**mandatory STOP**してください。

`PRIMITIVE_TRADEOFF_EXISTS`の場合も、wrapper-level pilot、新規性成立、DF-native improvement、D/R placement advantageを自動認可しないでください。

結果については、少なくとも以下を保存・照合してください。

- 全synthesis keyのsequence identity、T/T† count、error guard
- 全16 primitive row
- ordinary / controlled pair別のJ intervalとclassification
- weight second moment
- expected T count
- zero-cost baselineの扱い
- π-rational targetと±1/5 rad targetの結果
- controlled pairでのmoment penalty
- resource cap / runtime / error / numeric failure
- source・authorization・contract・tool identity
- one-shot marker
- retry=0、wrapper pilot unauthorized

結果・監査・GPT向けreview資料を必要最小限commit・pushし、branchとfull commit SHAを報告してください。

**SP-0.5終了後は次の研究方針判断をGPT側へ戻し、それ以上の科学実行を行わないでください。**
<!-- END USER EXECUTION INSTRUCTION -->
