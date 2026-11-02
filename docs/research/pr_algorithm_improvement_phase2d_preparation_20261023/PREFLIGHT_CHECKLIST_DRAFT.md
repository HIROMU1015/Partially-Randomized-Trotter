# 将来の実行前preflight設計

**今回は科学preflightを実施していない。** 下記を将来承認済みSOURCEで確認する。fixture生成やscientific importsを「preflight」と称して承認前に開始しない。

| 項目 | 将来の確認・失敗時の扱い |
|---|---|
| Authorization | 正しい段階I/B/F1/F2、明示承認、one-shot、fresh namespaceを照合。不足はSTOP |
| Git HEAD | frozen SOURCE commit、tree、dirty diff、transitive import closureを照合 |
| Input identity | fixture/catalog/arm/law/recipe/env hash、serializerを照合。候補文書hashをinput代用にしない |
| Exact fixture | c、D、d、Vの順序、Gamma convention、JW sign、zero policyを照合 |
| Register/state | system0–3／ancilla4、little endian、X0X1、full register、2粒子target、中間projectionなし |
| H/split/scalar | 同じH、LD1、R二fragment、faithful_identity_in_tail、外部cを照合 |
| Four arms | SS/SP/PS/PPのcollect規則、各L、I/u/s/v coverage、基底keyを照合 |
| Context | F1 T.2/q2/r2/K2、F2 T.2/q1/r2/K0を混用しない |
| Dependency | exact Python patch／wheel lock／Qiskit1.3.0等／BLAS、no dense fallbackを照合 |
| Compiler | opt1、seed17、basis、topology-free、approximation1、qubits_initially_zeroFalse、layout tracking |
| Seeds | 科学sampling RNGなし、config metadata seed0、compiler17。追加completion RNGがあれば事前binding必須 |
| Numerical gates | rtol0、固定tolerances、90/128digits登録、interval/ceil provider資格、bias表示範囲を照合 |
| Call inventory | L依存式を正確整数化、全case ID、logical requests／unique misses／attempts、output countsを確定 |
| Output | fresh科学namespace、atomic per-case records、ログ・保存量・hash、既存データへ書かない |
| Duplication | 自分のrun lock、atomic claim、同じcase keyの重複禁止。旧grant reuseなし |
| Grant consumption | 最終承認後だけfresh authorization発行、消費の原子性、failed attemptの保持、auto retry禁止 |
| Resource availability | 空きCPU affinity/RAM/GPU/diskと他job非干渉を確認。科学call上限と資源選択を分離 |
| Performance path | 既存timing／static bottleneckからcompileをprocess並列、outer/inner threadsを協調、immutable input共有 |
| Reference check | 最初の登録caseをreference/optimized照合に使い、未登録pilotを追加しない。失敗はSTOP |
| Failure handling | worker failureはattemptを保存し停止／安全回収。未実行caseは勝手に再開しない |
| Original preservation | base tree差分、元worktree/job／SOURCE／input／raw／レビュー／契約／grant／STOPの非変更を確認 |

process数は実memory見積もりと空きCPUに基づいて選ぶ。固定少数worker／CPU/RAM/GPU/wall-time capは置かない。nested BLAS/OpenMP/Rayon/Qiskit並列を専用processの環境内で調整し、共有venv/cache、OS/scheduler/quota、他ユーザーjobを変更しない。GPUは対応backendがある場合のみ利用判断し、使ったと偽らない。

90/128digits referenceはprocessごと独立context、cacheはfull identity、角度・phaseを含むIR単位。計算済み同一caseを検査のために無断再実行しない。checkpointは将来の自動resume許可ではない。
