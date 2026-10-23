# Codex指示：H4非DF/DFのC/L誤差会計・literal wrapper保証転送・既存40軸接続

Repository：`HIROMU1015/Partially-Randomized-Trotter`

基点branch：`codex/non-df-h4-local-diagnostic-20261023`

基点commit：
`de94417e7e1a0b984b93b81aba673b5ef94fdf98`

L技術・科学SOURCE：
`eb195bf2dedd0aed9d80ad7f84948b59ea53d854`

C科学SOURCE：
`e80d7510880f8db993b777c627bb8e95b05a070d`

元H4科学SOURCE：
`0c329b30a81accb35c212be8defa66fee0dd0f12`

## 1. 目的と承認範囲

Lの限定数値診断が完了しました。

- 128 bit固定
- Forward 148件、inverse 148件
- Basis 11件
- Component support 324件
- 全631論理call完了
- PF一様basis伝播上界：約3.69×10⁻¹³
- PR一様basis伝播上界：約4.03×10⁻¹³
- 人工・回帰148件PASS
- Native終了・回収PASS
- Mandatory STOP済み

一方で、以下が未解決です。

1. C/Lの誤差被覆の重複：`COMBINATION_BLOCKED`
2. Literal wrapperへの保証転送
3. B_savedによる測定スケールへの転送
4. 旧compiled program全体の保証：`MISSING_EVIDENCE`
5. 正式certificate受理とN/G会計

今回は、次の3工程を一括で実施してください。

**A. C/L誤差会計の精密な接続**

既存Cの`circuit_semantic_allowance`とLの保証付き局所誤差上界について、SOURCE上の定義・primitive・保証範囲・重複・未被覆を特定する。

**B. Literal wrapperへの保証転送**

DF-nativeのdiagonal angle、controlled decomposition、phase accumulation、basis、B_saved scaleについて、既存SOURCEとL証拠から保証できる範囲を実装・検証する。

**C. 既存40軸への接続設計**

既存Cの40軸数値包含を再利用するためのproof adapter・誤差会計・gate接続を設計し、既存証拠で閉じる部分と追加の数値証拠が必要な部分を区別する。

### 今回の承認

以下を一括承認します。

- GitHub資料・SOURCE・保存結果の調査
- 既存誤差会計の技術的・数学的な対応関係の確認
- 必要なproof adapter・verifierの実装
- 記号的・解析的な証明規則の実装
- 人工・mock・回帰テスト
- 既存保存証拠のSHA・構造・scope照合
- 追加検証が必要な場合の最小実行契約案
- 新技術SOURCE固定
- 成果物保存・終了監査・commit・mandatory STOP

**新規科学計算、実H4数値再解析、正式certificate受理、N/G再会計は承認しません。**

承認済みの技術作業は、細かな中間確認なしで完了まで進めてください。

## 2. 最新共通運用指示

正式な共通運用指示として、

`research_project_instructions_unified_2026-10-23.md`

を適用してください。

特に§8.4と§10を遵守してください。

### 既存コードの再利用

作業開始前に、過去の実装・検証で使用した以下の資産を確認してください。

- Cの保証付き数値計算コード
- Lの局所誤差・basis合成コード
- C/Lのcoverage ledger
- H4のproof adapter・certificate gate checker
- Phase/control/basisの記号的checker
- 既存の人工・回帰テスト
- SOURCE・SHA・grant・ledger・STOP監査コード

既存実装が要求仕様に適合する場合、再利用・必要最小限の拡張を優先してください。

重複する計算・検証・監査ロジックを不要に再実装しないでください。

既存コードを流用する際は、入力・出力、誤差の定義、数値精度、SOURCE、依存関係、再現性の整合性を確認してください。

新規実装が必要な場合は、理由と既存実装との差分を記録してください。

## 3. 固定資料

最初にGitHubの固定commitと必要資料を取得・照合してください。

主資料：

- `docs/non_df_h4_local_diagnostic_20261023.md`
- `artifacts/non_df_h4_local_diagnostic/2026-10-23/`
- `result_summary.json`
- `coverage_ledger.json`
- `audit02/local_gate_bounds.csv`
- `audit02/basis_bounds.csv`
- `audit02/uniform_basis_transfer.csv`
- `audit02/component_support.csv`

Cの資料：

- `docs/non_df_h4_c_reference_evidence_20261023.md`
- `artifacts/non_df_h4_c_reference_evidence/2026-10-23/`
- 既存40軸の数値包含・normalization・SOURCE・実行契約

Certificate gateの資料：

- `docs/non_df_h4_proof_binding_20261023.md`
- `artifacts/non_df_h4_proof_binding/2026-10-23/`
- `gate_bindings.json`
- `guarantee_relations.json`
- 関連するcertificate gate schemaとchecker

Kの資料：

- `docs/non_df_h4_kl_production_20261023.md`
- `artifacts/non_df_h4_kl_production/2026-10-23/K01/`

既存の固定SOURCE、raw結果、grant、ledger、marker、STOPを変更しないでください。

---

## 4. 作業A：C/L誤差会計の重複解消に向けた検証

### A-1. Primitive coverageの追跡

Cの保存誤差会計をSOURCEから確認してください。

重点対象：

- `bias_distance`
- `circuit_semantic_allowance`
- `basis_reuse`
- `phase_accumulation`
- `pi_rounding`

Lの保存保証：

- `local_forward`
- `local_inverse`
- `basis_reuse`
- Basis/control branchの合成誤差
- PF/PRのuniform operator bound

各primitiveについて、以下を記録してください。

1. 数学的な定義
2. 実装SOURCE・関数・コード位置
3. 元の誤差が対象とする演算子・状態・信号
4. Cが使用した誤差上界または経験的推定の性質
5. Lが保証する誤差範囲
6. 両者の重複・非重複
7. 正式な結合に不足する命題

**単に同じ名前のprimitiveが存在することと、同じ数学的誤差が包含されていることを区別してください。**

Cのsemantic allowanceが経験的・heuristicな性質を持つ場合、Lの厳密上界によって自動的にcertificateへ格上げしてはいけません。

### A-2. 二重計上のない結合方法

次の方法を比較してください。

- Cの既存上界がLの対象誤差を包含していることを証明する方法
- Primitiveを互いに重ならない項へ分解する方法
- CとLを同じ誤差分解から導くjoint enclosure
- 上記が成立しない場合に、誤差会計を分離したまま保持する方法

どの方法についても、数学的な十分条件・必要な証拠・成立しない条件を明示してください。

**未証明の減算・相殺を行わないでください。**

また、既存C totalとL uniform boundの単純加算を、証明なしに正式な結合方式として採用しないでください。

保守的な上界の候補を考える場合も、各項が対象演算子間の差を本当に包含するか確認してください。

### A-3. 結合判定

新しいcoverage verifierを実装する場合は、少なくとも次を区別してください。

- `PROVED_DISJOINT`
- `PROVED_COVERED`
- `OVERLAP_UNRESOLVED`
- `MISSING_PROOF`
- `NOT_APPLICABLE`

これらは必要に応じた内部判定です。既存schemaを変更する場合は後方互換性を維持してください。

**実証拠について条件が証明できない限り、従来の`COMBINATION_BLOCKED`を保持してください。**

今回は既存40軸のtotal errorを数値的に再集計しません。

## 5. 作業B：Literal wrapperへの保証転送

Lの保証済みscopeは、理想局所gateと保存encoded matrixの差、およびbasis/operatorの一様誤差伝播です。

この結果をliteral constructed wrapperへ転送するための証拠を整理・実装してください。

### B-1. Diagonal angle・phase

以下を確認してください。

- DF-nativeのdiagonal evolution
- Rotation angleの生成
- 保存binary64と理想的な角度の対応
- πの丸めと誤差
- Phase accumulation
- Global phaseとrelative ancilla phase
- Signed identity / Z / ZZの扱い
- PRのorder 0/2/4
- Finite RTE componentの位相・符号

特に、**system単体ではglobal phaseであっても、controlled evolutionではancilla間のrelative phaseになり得る**ことを保持してください。

位相を落とした等価性をcontrolled wrapperの証明へ流用しないでください。

### B-2. Controlled decomposition

次を確認してください。

1. Control=0 branch
2. Control=1 branch
3. DF basisのforward/inverse
4. Diagonal/control部分の構成
5. Constant・identity termのphase
6. Generic controlled gateの定義
7. Literal builderとencoded modelの対応

Lのforward/inverse上界をどこまで転送できるか整理してください。

丸められたDF行列のunitarityを、理想gateのunitarityと同一視してはいけません。

### B-3. Hadamard wrapper

Cosine/sineそれぞれについて、

- Ancilla初期化
- H
- Controlled evolution
- Cosine/sineのphase操作
- 最終H
- Binary measurement
- `p0−p1`とRe/Imの関係

を確認してください。

数式上の理想wrapperと、SOURCEが構築するliteral wrapperを別のscopeとして扱ってください。

保存された経験的なprobability差だけで、数学的な一様誤差上界を証明したことにはしないでください。

### B-4. B_saved scale

既存C/normalizationの保存結果を利用し、次の対応を整理してください。

- Nominal normalization B
- Saved B
- Saved `log_B_upper`
- Implemented sampling law
- Binary readoutの±B_saved支持集合
- Encoded signalと測定期待値の関係
- Lのoperator誤差をmeasurement errorへ転送するための条件

**Binary readoutが±B_savedに収まることと、encoded event演算子のnormが保証されていることを混同しないでください。**

既存のIID uniform-word等の条件付き統計仮定を維持してください。

PCG64の無条件独立性証明を、新しい必須条件として追加しないでください。

今回は新たなB_scaled数値上界を計算しません。

## 6. 作業C：既存40軸への接続設計

固定対象は従来の4方式×5精度×cosine/sineの40軸です。

Cの40軸は64 bitで数値設計目標を達成していますが、正式certificateは未受理です。

今回は既存の40軸を再計算せず、証拠を接続する構造だけを完成させてください。

### C-1. 証拠graph

各軸について、次を対応付けてください。

1. Cの保存数値包含
2. Reference
3. Signed bias
4. Normalization
5. Encoded event
6. Lの局所gate/basis上界
7. C/Lの誤差被覆
8. Phase/control/diagonal
9. Literal Hadamard wrapper
10. B_saved scale
11. Compiler transfer
12. Combined error budget
13. Conditional statistics
14. Certificate acceptance

既存の40軸×14 gate schema・proof adapterを優先して再利用してください。

### C-2. 充足性の分類

各gateについて以下を区別してください。

- 既存証拠だけで成立が証明できる
- 保存証拠はあるが照合・証明が未完了
- 新しい技術的実装・人工検証が必要
- 新しい保証付き数値計算が必要
- 旧program欠落などにより証拠が不足
- 固定条件への不適合が証明されている

SOURCE・入力・結果・SHA・数学的命題・残存blockerを対応させてください。

保存フラグやSHAの一致だけから科学的な証明を成立させないでください。

### C-3. 追加検証の最小化

新しい数値証拠が必要な場合は、対象を最小化した実行契約案を作成してください。

少なくとも以下を具体化してください。

- 未解決の数学的命題
- 対象となるgate・basis・phase・event
- 再利用するC/L結果
- 必要な精度と保証方法
- 科学call予算
- SOURCE・runtime・入力SHA
- 受理条件
- Grant/ledger/STOP
- Native終了監査

**40軸のfull-H reference、law mean、bias、Lの148/148/11/324を再計算することを既定案にしないでください。**

今回は追加の数値計算を実行しません。

## 7. K：旧compiler保証との境界

旧postcompiled programとpass traceが未取得であり、compiler全体は`MISSING_EVIDENCE`です。

今回はC/L結合とliteral wrapperへの転送を優先してください。

旧compilerについては、

- 新たに利用可能となった既存証拠
- SOURCE-levelで確定した限定的な恒等式
- 未証明のcompiler pass
- Program本体・trace欠落

を整理するだけとします。

旧compilerの全証明を今回の必須完了条件にしないでください。

**新しいcompileや新program系列への切替えは行わないでください。**

旧programの保証を維持できない見通しが強まった場合は、研究の評価対象・保証方針に影響する事項として別途報告してください。

研究方針の変更をCodexが独断で決定してはいけません。

## 8. 人工・回帰・技術検証

既存のT・binding・Lの148件を再利用し、必要な人工・mock・記号的テストを追加してください。

重点検証：

- Primitiveの重複・欠落検出
- 未証明の減算・相殺の拒否
- 二重計上の検出
- Phaseの符号
- Controlled branchの対応
- Global/relative phase
- Basisの作用順序
- Diagonal angleの規則
- Hadamard wrapperのcosine/sine
- B_saved scaleのscope
- Encoded/literal/compiledの混同拒否
- 別case・別軸への証拠流用拒否
- SOURCE/SHA不一致拒否
- 未証明時のfail-closed
- 未承認の数値実行・正式受理・N/G会計の拒否

人工検証は既存の技術作業として実施して構いません。

ただし、実H4入力を使った新しい行列・確率・数値上界・結合誤差の評価を、人工テストの名目で実行してはいけません。

技術的に証明可能な部分は実装・検証まで進め、実H4の数値証明が不足する部分は未解決のまま保持してください。

## 9. 実行境界・保全

今回は以下を禁止します。

- 新規SCF/FCI・DF生成
- 状態最適化
- 実H4の数値計算・数値再解析
- Reference/law mean/biasの再計算
- Lの数値診断の再実行
- Sampling・RNG replay
- 新規compile
- 新しい40軸総誤差の数値計算
- B・allowance・guard・lawの無断変更
- 正式certificate受理
- N/G再会計
- 既存科学runのretry/resume
- 消費済みgrantの再利用
- 詳細GPT科学レビュー
- 最終scientific evidence freeze

元H4、C、T、proof binding、K、Lの固定SOURCE・raw・grant・ledger・marker・STOPを保全してください。

既存Lのmtime監査失敗履歴も保持し、新しい時刻順序証明が得られたことにしないでください。

今回の作業では新しい科学grantを発行しません。

実行権限は無効のままとし、技術的なSOURCE固定と科学実行承認を区別してください。

## 10. 計算資源

最新版共通運用指示§8.4を適用してください。

CPU・RAM・worker数に恣意的な低い上限を設けないでください。

重い処理では既存処理の再利用、重複処理の削減、必要な並列化・高速化を行ってください。

ただし、共有環境全体や他ユーザーに影響する変更・操作は絶対禁止です。

自分の作業領域・process・cacheだけを対象とし、既存の固定契約・科学条件・SOURCEを変更しないでください。

## 11. 成果物

新しい専用branchに、以下を保存してください。

1. C/L primitive coverageのSOURCE追跡報告
2. 重複・非重複・未被覆のproof obligation一覧
3. 二重計上を避ける結合方法の技術設計
4. Literal wrapper保証転送の実装・検証
5. Diagonal/control/phase/B_savedの証拠対応表
6. 40軸×14 gateへの接続matrix
7. 未解決gate・blocker一覧
8. 必要な場合の最小追加数値検証契約案
9. 人工・回帰テスト結果
10. 新技術SOURCE・runtime・SHA・保全監査
11. README・引継ぎ資料・mandatory STOP記録

成果物では、次を明確に区別してください。

- 数学的に証明できたこと
- 静的SOURCE・SHAの照合ができたこと
- 人工テストで検証したこと
- 実H4で数値検証されていないこと
- 証拠が不足していること

既存の科学結果・certificate gateの判定を無断で書き換えないでください。

## 12. 完了・報告・STOP

専用branchへlocal commitし、有効なrepository運用規則に従って公開を試みてください。

Pushに成功した場合はremote SHAを確認してください。認証不足などで失敗した場合は、結果を保全し、原因と必要な操作を報告してください。

最終報告では、以下を示してください。

1. C/Lの誤差重複の具体的な原因
2. 重複を解消できた範囲と未解決部分
3. Literal wrapperへ保証を転送できた範囲
4. Diagonal/control/phase/B_savedの残存証拠
5. 40軸のcertificate gate充足状況
6. 新規数値計算が必要な最小対象
7. 旧compiler保証との関係
8. テスト結果・SOURCE・commit・公開状態
9. 次の担当・作業・必要な承認

**今回承認したSOURCE調査、C/L結合方法の設計、literal wrapperの技術実装・人工検証、40軸接続設計、成果物保存、終了監査まで一括で実施してください。**

ただし、以下の区別を維持します。

- 技術的に証明できた範囲は明確に記録する。
- 実H4で未証明の条件は未証明のまま残す。
- C/Lの結合が成立しなければ`COMBINATION_BLOCKED`を維持する。
- 旧compilerは証拠が揃うまで`MISSING_EVIDENCE`を維持する。
- 正式certificate受理とN/G再会計へ進まない。

**完了後はmandatory STOPしてください。**