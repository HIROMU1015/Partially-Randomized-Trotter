# Codex指示：非DF PR/PF基盤の技術的堅牢化とH4実分子pilot準備

## 1. 目的

Repository：`HIROMU1015/Partially-Randomized-Trotter`

H2の実分子pilotでは、以下4方式の固定時間信号資源評価が完了した。

- Pauli-JW PF
- Pauli-JW PR
- DF-native PF
- DF-native PR

今回の目的は、H4以降の実分子検証へ進む前に、次の2点を技術的に整備することである。

1. **Qiskit独立プロセス実行基盤の耐障害性を向上させる。**
2. **PRの希少Taylor eventが、有限MCによる回路費用推定に与える未解決事項を整理する。**

その後、H4の実分子pilotに必要な実行契約・固定条件・技術preflightを準備する。

既存のDF・Pauli実装、科学結果、SOURCE、grant、marker、ledger、STOPは保全する。

## 2. 参照するH2完了結果

Branch：

`codex/non-df-h2-process-recovery-20261023`

結果commit：

`c8990de7721545faff88b636108034dedb80a463`

科学SOURCE：

`7a6c4263d96153ecbe721dcfb6941d0d2684b3b0`

主要資料：

- `docs/non_df_h2_process_recovery_validation.md`
- `artifacts/non_df_h2_process_recovery/2026-10-23/comparison.md`
- `artifacts/non_df_h2_process_recovery/2026-10-23/comparison.json`
- `artifacts/non_df_h2_process_recovery/2026-10-23/STOP.json`
- `src/trotterlib/isolated_qiskit_execution.py`
- `src/trotterlib/fixed_signal_evaluation.py`
- `src/trotterlib/pauli_signal_adapter.py`

作業前に最新のbranch・worktree・GitHubを確認すること。

既存の承認済み修正を失わないよう、新しい専用branch/worktreeで作業する。

## 3. 現在確認できている実装状況

H2・1.00 Å・STO-3G、T=0.8、q=4、S2、PR r=2/K=4で4方式×5精度の結果を取得済み。

ε_sig=0.05での総RZ費用は以下。

| 方式 | G [RZ] |
|---|---:|
| Pauli PF | 54,215,700 |
| Pauli PR | 14,813,895.8125 |
| DF PF | 56,989,854 |
| DF PR | 29,038,120.96875 |

C/N/Gの会計と実施した数値検証はPASSしている。

また、Qiskitのthread並列で起きたborrow例外は、独立プロセス化によって解消した。

ただし、以下は未解決である。

- 子プロセスの準備待ち・終了待ちの耐障害性
- 異常停止・ハング時の安全な回収処理
- Pauli PRの希少Taylor eventによる平均compiled費用への影響
- DF PRの未観測Taylor event
- より大きな実分子での技術検証

人工テストのPASSを、実分子での一般的な実装保証とは扱わない。

## 4. 作業A：Qiskit実行基盤の堅牢化

まず `isolated_qiskit_execution.py` と関連runnerを調査する。

現在の実装では、子プロセスがready状態になるまでの待機と、実行終了までの待機について、明示的な停止監視が十分ではない可能性がある。

以下を確認・改善すること。

### A-1. プロセス管理

- 子プロセスの起動失敗
- ready signalが発行されない場合
- ready前の異常終了
- ready後のハング
- 子プロセスの非ゼロ終了
- 親プロセスの例外
- ユーザーによる中断
- 出力ファイル・markerの衝突
- stdout/stderrの保存失敗
- 子プロセス・子孫プロセスの回収漏れ

各ケースについて、所有するプロセスだけを安全に管理・終了できるようにする。

ただし、科学計算の実行時間には規模による違いがある。

**恣意的な短いwall-time上限を新設し、正常な科学計算を不必要に停止させてはいけない。**

待機監視・異常検出・タイムアウト・安全終了の方針は、既存実行契約と実行規模に応じて設定できる構造にする。

タイムアウト等が発生した場合は、実行を自動retryせず、原因と未完了状態を記録してSTOPする。

### A-2. Qiskitの並列実行

現在採用しているcase別独立Python process方式を維持する。

- Qiskitのprocess-global mutable stateを共有しない。
- thread並列への安易な差し戻しをしない。
- compiler設定とseedを維持する。
- case別のSOURCE・入力hash・出力を分離する。
- 同一科学caseが重複起動されないようにする。
- 正常終了時にすべての子プロセスが回収されていることを確認する。

### A-3. 人工テスト

正常系に加えて、意図的に失敗する人工workerを使用して次を検証する。

- readyを返さないworker
- 起動直後に例外を返すworker
- 非ゼロexit
- 長時間応答しないworker
- 親processの中断
- 複数workerの一部だけが失敗
- 出力先の競合
- 失敗時のSTOP・証拠保存
- 重複実行の拒否

共有サーバーの他ユーザーや別jobへ影響しない、人工processだけで検証する。

## 5. 作業B：希少Taylor eventの保存証拠診断

既存H2のPauli PR・DF PRについて、保存済みのRTE法則とMC結果を確認する。

### B-1. Pauli PR

前回の64 paired trajectoryでは、合計512 event位置について、

- order 0：510
- order 2：2
- order 4：0

が観測された。

order 4の1 event当たりの法則確率は約 `3.5227e-7` であり、未観測は不自然ではない。

ただし、**未観測であることは平均compiled費用への寄与が0であることを意味しない。**

以下を整理する。

- 各Taylor次数の法則確率
- 観測回数
- MC標本全体に占める各次数の観測状況
- 保存されたtrajectory費用との対応
- 名目MC標準誤差・95%区間が評価している範囲
- 希少eventのcompiled費用で未取得の情報
- 現在の結果から保証できない内容

### B-2. DF PR

同様に、DF PRの未観測order 2/4について、保存済み法則・event記録・費用の対応を確認する。

Pauli PRとDF PRではTail法則や回路構造が異なるため、同じ次数のevent費用が同じになるとは仮定しない。

### B-3. 許可範囲

今回は**保存証拠の読み取り・既存値の照合・技術的な診断整理**を実施する。

新しいMC sampling、回路compile、希少eventの母平均費用の新規数値評価、科学的な再ランキングは行わない。

既存の名目区間を厳密な母平均保証へ格上げしない。

不足する情報がある場合は、その情報を取得するための実行契約案を作成する。

ただし、実際の追加科学計算は別の明示実行承認を必要とする。

## 6. 作業C：H4実分子pilotの準備

作業A・Bの結果を踏まえ、H4の実分子pilotへ拡張するための実行契約案を作成する。

### 対象系

初期案：

- linear H4
- 隣接距離1.00 Å
- STO-3G
- charge 0、spin 0
- active space / frozen coreなし
- 通常Pauli-JW Hamiltonian
- tol-only DF Hamiltonian

H2と同様に、同一MO積分からPauli版とDF版を生成する。

状態と参照Hamiltonianの対応を固定し、演算子表現間の一致を検証する設計とする。

### 比較方式

- Pauli PF
- Pauli PR
- DF PF
- DF PR

外側S2を基本とする。

固定時間信号に対するC/N/Gを評価する。

H2のpilotと同様に、Full Hadamard wrapper、cosine/sine両軸、Hoeffding shot会計を扱う。

ただし、H4のPauli項数・DF fragment数・prefix・候補構成はH2と異なる。

**H2のL_Dや候補条件を、そのままH4へ機械的に移植しないこと。**

### 固定すべき条件

以下を具体化する。

1. Hamiltonianと積分生成規約
2. DF分解条件
3. 状態・参照信号の生成条件
4. Pauli項の順序とprefix
5. DF fragmentの順序とprefix
6. PF/PRの分割条件
7. T/q/r/K
8. RTEの法則・正規化
9. 信号精度と失敗確率
10. numerical allowance
11. MC標本数・seed・不確かさの設計
12. compiler・制御回路仕様
13. 物理・回路・数値検証の許容差
14. SOURCE・grant・one-shot・STOP
15. 実行資源・並列処理・証拠保存

H4ではPauli項数の増加により回路規模が大きくなるため、case別process並列と、科学的意味を変えない重複計算排除を検討する。

MCの本数についても、H2の64本を無条件に十分とみなさない。

ただし、今回の段階で勝手に新しい科学的な標本精度保証や推定器を採択しない。

### 今回のH4実行範囲

H4の新規科学計算については、必要な実行契約・SOURCE計画・検証gateを準備する。

新規PySCF/SCF/FCI/DF生成、実分子signal計算、MC sampling、compileなどの科学実行は、H4実行範囲に対する別途明示承認まで開始しない。

**H4実行時には、各工程をまとめた一括承認で進められる契約案とし、不要な個別承認の往復を発生させないこと。**

## 7. 技術検証

次を確認する。

- Qiskit process分離の正常動作
- ハング・異常終了の安全処理
- 科学caseの重複起動防止
- 子プロセスの完全回収
- 全ログとSTOPの保存
- 元H2の科学結果との後方互換性
- 既存DF版の回帰
- 非DF版の回帰
- Pauli PRのRTE法則・位相の人工検証
- C/N/Gの保存schema
- SOURCE・hash・provenance
- 既存科学結果・grant・marker・STOPの不変性

人工テストや保存値検査を優先する。

内部で新規実分子計算を開始する既存テストは、今回の技術検証から除外する。

テスト対象・PASS件数・skip・除外理由を報告する。

## 8. 実行性能と共有環境保護

最新版正式共通指示：

`research_project_instructions_unified_2026-10-17.md`

を実際に確認し、特に§8.4を適用する。

CPU・RAM・worker数に恣意的な低い上限を設けない。

重い処理では、利用可能な資源に応じて十分な並列化・高速化を検討する。

ただし、Qiskitの共有mutable stateに起因する競合を再導入しない。

共有環境全体、他ユーザー、他jobへ影響する設定・process・ファイル・cache操作は絶対に禁止する。

変更は自分が所有する分離されたworktree・process・出力領域に限定する。

既存の承認済み固定科学契約の資源条件も無断変更しない。

## 9. 成果物

新しい専用branch/worktreeで作業し、次を保存する。

- Qiskit実行基盤の技術修正
- 追加人工テスト
- 異常終了・待機監視の検証結果
- H2の保存希少event診断資料
- H4実分子pilotの実行契約案
- H4の固定候補・精度・MC設計案
- 実行前技術gate
- SOURCE・provenance
- テスト結果
- 保全検査
- STOP記録
- 技術報告Markdown

詳細な科学レビュー記録は作成しない。

既存の研究結果・図・スライドは変更しない。

## 10. 完了報告とmandatory STOP

以下を報告すること。

1. Qiskit実行基盤の修正内容
2. 新しい異常系テストと結果
3. 子プロセスの待機・終了・回収に関する保証範囲
4. Pauli PRの希少event診断結果
5. DF PRの希少event診断結果
6. 母平均費用の評価で未解決の事項
7. H4実分子pilotの準備状況
8. H4実行に必要な科学計算・固定条件
9. 既存DF/非DF回帰テストの結果
10. SOURCE・branch・commit・push状態
11. mandatory STOPの状態

必要な技術実装・人工検証・保存証拠診断・H4実行準備まで完了すること。

ローカルcommitまで実施してよい。

push・公開は別途明示承認がある場合のみ実施する。

**今回の承認範囲を完了したらmandatory STOPすること。**

新しい科学モデルの採択、H4の実分子科学計算、未観測eventの追加sampling、旧失敗科学runのretry、GPT詳細科学レビューへ自動進行しないこと。