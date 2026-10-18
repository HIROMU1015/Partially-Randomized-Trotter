# Decision Handoff：何を承認すれば実費比較できるか

[入口](README.md) / [候補契約](execution_contract_candidate.md) / [証拠対応](cost_evidence_matrix.md)。技術状態は`PREPARATION_COMPLETE_WITH_ADOPTION_AND_PROVIDER_GAPS`。

**B3/C1の補助interface・schema・guard・人工検査は整った。一方、実G7の初回接続確認と実driver固定は未実施で、現時点で「すべてそのまま実行可能」とは言わない。** B3は費用estimand/schedule/angle条件、C1は未固定の物理Q/providerが主な採択待ちである。

## 1. 何が既に利用可能か

- P5_general_orderの固定p/x/m、G7 N/digital acceptance/IR係数/有限RZ cacheを再利用できる。同じB1/B2取得やsynthesisを繰り返す必要はない。
- W2c-AのCL/Q条件と、Ordinaryとの限定logical非劣、登録P5の旧conditional期待T結論は維持する。実CPU/native全体の優位ではない。
- canonical用の有限root/law preflight witness、実保持state seal、4方式共通audit、event/zero/phase/πW検査、exclusive cost stage記録を新SOURCEへ用意した。旧F3・G7は変更していない。
- native価格key/schema/guardはcontrol役割、forward/actual-adjoint、strict phase、compiler/precision/最適化を区別する。未知座標はnull。cacheやprovider証拠を数値的に再認証したとは言わない。

## 2. 残る不足は何か

| 不足 | 技術的に準備したもの | ユーザー判断が必要な点 |
|---|---|---|
| B3費用estimand | 固定全N観測と期待費用を分離した候補 | 固定workloadの実費を記述するか、母期待費用を保証するか。後者の統計policyは未採択 |
| angle精度とschedule | 既存F3/共通enclosureへ接続する仕様 | τ/K_phi/J、層A/B、setup/cache/output共有。旧G7 error budgetを変更せず収まる条件 |
| 実G7統合 | duck-typed adapter・guard・人工fixtureで検査 | 採択後の実driver固定/初回correctnessを科学実行承認に含める。現在は未検証 |
| 物理CQ | provider/価格/actual-adjoint/phase interface | Q/encoding/sector/basis、gate set/compiler/ancilla、exact-query誤差0との互換性。便宜的Pauli化はしない |
| 新価格の粒度 | 旧係数とのmapping仕様 | role/word/depthで旧B2 summaryが不足するなら追加縮約を独立に認可 |
| End-to-End | C2不足を明示 | state/readout/backend/全N estimatorは今回対象外・未知。0にしない |

## 3. 次に必要な最小承認

**次の担当はユーザー**。まずB3について、第一候補である「旧seed・保存Nの完了した固定workloadを同じ保存/監査条件で比較し、当該環境の記述的実費として報告する」か、期待費用の統計保証まで求めるかを決める。前者でもCPU優位はまだ得られておらず、α_axisをCPU CIに転用しない。

その上で契約を採択し、**別途**「固定SOURCE/補助SOURCE/入力、層・angle・schedule、call予算、保存・STOPに基づく初回correctness＋B3 profiling」を明示承認する。Codexは実driverを固定してから実行する。今はdriver接続と科学実行を一つの未承認作業として勝手に開始しない。

C1を同時に進めるには、物理Q/providerが先に必要である。現P5は抽象Qのケースなので、native compilerを指定するだけでは不足する。物理provider/error/policyをユーザーが選び、必要なCQ compile/price取得を承認すれば、旧RZ cache・B1/B2と結合できる範囲を比較する。role/depth粒度が不足すれば、その追加B2縮約も承認scopeへ含める。**新物理誤差を既存budgetへ黙って足さない。**

## 4. 結論とSTOP

Fullが古典費用で勝つかは、追加packet/law/hash/requery/保存がtrial削減利益を超えるかを測る必要がある。native量子費用では、abstract CQの価格を物理化しなければ、旧RZ/query条件付き結論を超えられない。CPU秒・bit・memory・T/CX/depthの合成scalarや科学的GOを作っていない。

新SOURCEは補助実装/人工検査段階で、実入力correctness・profiling・compile・synthesis・backendは0件。候補は未採択、authorization/grant/marker/claim0件。既存SOURCE/raw/科学契約/消費済みclaim/STOP・親worktree・他研究を保全し、公開照合後**mandatory STOP**。
