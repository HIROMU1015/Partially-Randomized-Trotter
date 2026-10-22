# Codex指示：H2非DF pilotのQiskit並列実行問題を解消し、未完了2ケースを取得する

## 1. 今回の目的と実行承認

Repository：
`HIROMU1015/Partially-Randomized-Trotter`

前回、DFと通常Pauli-JW Hamiltonianに対する固定時間PR/PF資源評価を行ったが、H2 pilotの4ケース中2ケースがQiskit runtime例外で失敗した。

今回の目的は、**Qiskitの並列実行問題を技術的に解消し、未完了2ケースを取得して、H2の4方式比較を完成させること**である。

ユーザーは今回、以下を一括で明示承認する。

- 失敗原因の技術調査
- Qiskit並列実行方式の修正
- 人工回路による再現・並列検証
- 必要な既存回帰テスト
- 新規SOURCE・実行契約・grantの作成
- 失敗した2ケースに限定した新規科学実行
- C/N/Gの取得・検証・結果統合
- 技術報告・成果物保存・commit

技術修正が成立すれば、上記の各工程で個別の承認を求めずに進めてよい。

ただし、**旧run01のSTOPを解除したり、消費済みgrantでretryしたりすることは禁止する。**

今回の計算は、新しい科学実行契約に基づく、明示的に承認された別runとして実施する。

H4以降への拡大、新しい科学モデルの採択、候補探索、追加の独立確認は今回の承認に含まれない。

GPTによる詳細科学レビューも開始しない。

---

## 2. 参照する既存結果

前回のbranch：

`codex/non-df-h2-pilot-20261022`

結果commit：

`bd2f3ce9b2035d9caefac75785e6fe70a3dcbca2`

科学SOURCE：

`b5ed2c6e2399b55a0c24eb8437f842892a6703ef`

技術報告：

`docs/non_df_h2_pilot_validation.md`

科学実行記録：

`artifacts/non_df_h2_pilot/2026-10-22/`

特に次を確認すること。

- 元のexecution contract
- SOURCE・grant
- `run01/STOP.json`
- `run01/failures.json`
- `run01/inputs/snapshot.npz`
- `run01/inputs/receipt.json`
- 各caseのstatus・validation・result
- preservation・result inventory

作業前にGitHub・ローカルworktreeの最新状態を確認する。

上記commitから新しい専用branch/worktreeを作成することを基本とするが、その後に正式な修正があれば保全して取り込む。

元の科学記録はすべて読み取り専用として扱う。

---

## 3. 前回の結果

対象はH2・1.00 Å・STO-3Gの固定時間信号評価。

| 方式 | 前回の状態 |
|---|---|
| Pauli-JW PF | 完了・C/N/G受理 |
| Pauli-JW PR | Qiskit例外で失敗 |
| DF PF | Qiskit例外で失敗 |
| DF PR | 完了・C/N/G受理 |

成功済みの2ケースは再計算しない。

**今回の新しい科学計算はPauli-JW PRとDF PFの2ケースに限定する。**

元のH2入力生成・SCF・FCI・DF構築は成功済み。

同一積分から作成したPauli-JW HamiltonianとDF Hamiltonianの整合性、状態、参照信号の照合もPASSしている。

既存snapshot・receipt・hashが検証できるなら、それらを再利用し、SCFやDF生成を繰り返さない。

---

## 4. Qiskit例外の原因調査

前回の失敗は以下である。

### Pauli-JW PR

`Already mutably borrowed`

Qiskitのhigh-level synthesisおよびequivalence library関連処理中に発生。

### DF PF

`Already borrowed`

controlled RZZ構築からbasis translationに至る処理中に発生。

前回は4ケースを同一process内のThreadPoolExecutorで並列実行した。

Qiskit 1.3.0のprocess-global `SessionEquivalenceLibrary` など、共有mutable stateへの競合が疑われている。

ただし、**現時点では原因が確定したわけではない。**

最初に以下を実施する。

1. 保存tracebackと実際の呼出し経路を確認する。
2. 同時実行時に共有されるQiskit内部状態を調査する。
3. 人工回路だけを使い、borrow例外の再現条件を調べる。
4. thread並列とprocess並列の挙動を比較する。
5. 失敗の原因が共有mutable stateなのか、別の実装不具合なのかを切り分ける。

原因調査のために、前回の実分子科学runnerを再実行してはいけない。

---

## 5. 推奨する技術修正

基本方針は、**独立した科学caseごとにQiskit処理を別processへ分離すること**である。

例えば、Python subprocessやspawn方式のmultiprocessingを利用する。

各processが自身のQiskit実行状態を保持し、複数case間でmutableなequivalence libraryやcompiler objectを共有しない構成を優先する。

確認する事項：

- Qiskit内部状態のprocess間分離
- compiler設定・versionの固定
- 同一seedでの再現性
- child processへの入力の正確な引渡し
- stdout/stderr/tracebackの保存
- caseごとの終了コード
- 出力ファイル衝突の防止
- case別SOURCE・input hashの照合
- 異常終了時の適切なSTOP
- child processの確実な回収

thread並列を禁止すること自体が目的ではない。

**安全に並列実行でき、かつ既存の科学条件を変えない方式へ修正すること**が目的である。

process分離以外に、技術的により適切な解決方法が確認できれば採用して構わない。

ただし、科学条件や回路構造を変える方法は採用しない。

---

## 6. 人工並列テスト

科学計算へ進む前に、人工回路で十分な技術検証を行うこと。

特に前回不足していた、**実際の4ケース同時実行に対応する並列preflight**を追加する。

対象：

- Pauli PFに相当する回路
- Pauli PRに相当するランダム回路
- DF PFに相当するcontrolled回路
- DF PRに相当するランダム回路

重点的に確認する処理：

- PauliEvolutionGate
- controlled Pauli rotation
- controlled RZZ
- high-level synthesis
- basis translation
- Hadamard wrapper
- cosine/sine両軸
- Qiskit transpile
- RZゲート計数

人工回路の並列実行を行い、borrow例外が生じないことを確認する。

また、同じ人工入力を逐次・安全な並列方式で処理した場合に、科学的な回路意味とゲート数の整合性が維持されることを確認する。

既存の人工テストと、実分子生成を伴わない安全な回帰テストも実行する。

**人工並列テストと既存回帰がPASSするまでは、実分子の科学計算を開始しないこと。**

ここで解決できない場合は、技術的な未解決事項を記録してSTOPする。

---

## 7. 新しい科学実行契約

技術修正が成立したら、前回の固定科学条件を参照して新しい契約を作成する。

### 対象物理系

- H2
- 核間距離1.00 Å
- STO-3G
- charge 0
- spin 0
- canonical real RHF
- 2 spatial orbitals
- 4 spin orbitals
- active space / frozen coreなし

前回保存済みのHamiltonian・DF・状態を使用する。

### 時間発展

- T=0.8
- q=4
- δ=0.2
- 外側PF：S2

### 今回取得するケース

**Case A：Pauli-JW PR**

- 非identity Pauli項14個
- 係数絶対値順のprefix
- L_D=7
- r=2
- K=4
- 既存paired finite-RTE法則

**Case B：DF PF**

- tol-only DF rank3
- 全3 fragmentを保持
- L_D=3
- S2

前回と同じ演算子順序、位相規約、controlled回路、normalization、compiler条件を維持する。

### 精度と費用

- ε_sig = 0.05 / 0.01 / 0.005 / 0.001 / 0.0005
- cosine/sineに各ε_sig/√2を配分
- 各軸の失敗確率α=0.025
- 既存Hoeffding十分shot会計
- 既存の経験的numerical allowance規約
- 状態準備費用P=0
- full ordinary-controlled Hadamard wrapper
- compiled RZ数
- G=N_cos C_cos + N_sin C_sin

### compiler

- Qiskit 1.3.0
- Qiskit Nature 0.7.2
- rz/sx/x/cx
- optimization level 1
- seed17
- topologyなし

Pauli PRのMC本数は前回契約の64 paired trajectoriesを維持する。

前回のseed生成規則と計算条件を確認し、新しい実行契約に固定する。

途中で標本数を増やしたり、精度条件を変更したりしない。

---

## 8. 科学実行の進め方

技術preflightがPASSした場合は、今回の一括承認に基づき、新しい科学実行を開始してよい。

ただし次を守る。

1. 前回のrun01は永久に保全する。
2. 新規namespaceを使用する。
3. 新しいSOURCE・grant・marker・ledgerを作成する。
4. 新規runと旧runのlineageを記録する。
5. 科学実行は各caseでone-shotとする。
6. 失敗後の自動retryは禁止する。
7. 失敗を別seed・別IDで再投入して禁止条件を回避しない。

新しい実行は、前回失敗した2ケースの取得だけに限定する。

Pauli PFとDF PRの成功済み結果は、元SOURCEとresult hashを照合して読み取り専用で再利用する。

同じH2入力・同じ評価条件による結果であることを確認する。

実行方式の修正によって科学的な回路定義や比較条件が変化した場合は、旧成功結果との結合を中止して報告する。

---

## 9. 4方式の結果統合

新しい2ケースが完了したら、前回成功済みの2ケースと組み合わせ、次の4方式を比較できる表を作成する。

- Pauli-JW PF
- Pauli-JW PR
- DF PF
- DF PR

各方式について、以下を整理する。

- Hamiltonian/state hash
- PR/PFの分割条件
- signal・bias
- normalization
- cosine/sine別のC
- cosine/sine別のN
- 総費用G
- 精度適格性
- PRの有限MC標本数・SE
- 数値検証status
- 科学SOURCE・実行run

5精度についてC/N/Gの比較表を作成する。

比較可能な条件が成立すれば、既存保存値だけを使用して、4方式の総費用を分かりやすく示す補助図も作成してよい。

新しい科学計算による候補最適化は行わない。

特に、DFとPauli方式ではHamiltonianの分割単位が異なり、L_Dの値を直接対応させてはいけない。

H2の固定構成についての比較であり、一般的なPR/PF優位性やサイズスケーリングを主張しないこと。

---

## 10. 技術・科学的検証

以下を確認する。

- Hamiltonian・stateの同一性
- 保存H2入力のhash
- Pauli/DFの演算子整合性
- PF/PRの数値信号
- B0等の別方式が混入していないこと
- finite-RTEの法則・正規化
- scalar/event位相
- controlled evolution
- Hadamard wrapper
- compiled probability
- cosine/sine別shot数
- C/N/G会計
- MC標本のpaired covariance
- 未観測Taylor eventの扱い
- 旧成功2ケースとの比較可能性

数値許容差は前回の固定契約を維持する。

不合格となった検証項目を、閾値変更によってPASSにしてはいけない。

MCの名目95%区間を母平均の厳密保証として扱わない。

---

## 11. 計算資源と共有環境保護

最新版の正式共通指示：

`research_project_instructions_unified_2026-10-17.md`

を確認し、§8.4に従う。

CPU・RAM・worker数に恣意的な低い上限を設けない。

Qiskitのthread-safety問題を避けながら、独立processによる並列化などを活用して効率的に実行する。

実際に利用可能な空き資源と、共有環境への非干渉を確認する。

共有サーバー全体や他ユーザーの環境・設定・process・job・ファイル等に影響する変更・操作は禁止する。

共有venvや共有Qiskit installationを修正しない。

必要な修正は、新しいbranch/worktree内のsourceと、自分が所有する独立process・cache・出力領域に限定する。

既存の承認済み科学契約の資源条件は無断変更しない。

---

## 12. 成果物とSTOP

新規branch/worktreeに以下を保存する。

- Qiskit並列実行エラーの調査記録
- 原因の切り分け結果
- 修正source
- 人工並列テスト
- 既存回帰テスト
- 新しいexecution contract
- 新SOURCE・grant・marker・STOP
- 新しい2ケースのC/N/G
- 各種numerical evidence
- 4方式の結果統合表
- 必要に応じた補助図
- provenanceとhash
- README・技術報告

既存の科学raw結果・SOURCE・grant・ledger・marker・STOP・図・スライド・論文資料は変更しない。

### 完了報告

次を報告すること。

1. Qiskit例外の原因と修正方法
2. 人工並列テスト・既存回帰の結果
3. 新規科学実行した2ケースの状態
4. 4方式比較が完成したか
5. 各方式のC/N/Gと数値検証結果
6. 旧runとの対応・差異
7. 未解決の問題と科学的限界
8. SOURCE・branch・commit・push状態

新しい専用branchへローカルcommitまで実施する。

push・公開は別途承認がない限り実施しない。

**技術修正・承認済みの新科学実行・結果統合・検証まで完了したらmandatory STOPすること。**

失敗した場合は、その時点までの証拠を保存し、自動retryや次stageへの移行を行わない。

GPTの詳細科学レビューは開始しない。