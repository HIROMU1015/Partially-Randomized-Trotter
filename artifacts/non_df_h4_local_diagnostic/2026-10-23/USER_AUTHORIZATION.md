# Codex指示：H4非DF/DFの局所数値診断L・one-shot科学実行・終了監査

Repository：`HIROMU1015/Partially-Randomized-Trotter`

基点branch：`codex/non-df-h4-kl-production-20261023`

基点commit：
`80147890fa91ad8270ce7643ab671727b3db5de4`

L技術SOURCE：
`eb195bf2dedd0aed9d80ad7f84948b59ea53d854`

C科学SOURCE：
`e80d7510880f8db993b777c627bb8e95b05a070d`

元H4科学SOURCE：
`0c329b30a81accb35c212be8defa66fee0dd0f12`

## 1. 目的と明示的な実行承認

前段階では、KのSOURCE-level保証調査と、Lのproduction実装・人工検証が完了しました。

- 人工・回帰148件PASS
- L production runner実装・SOURCE固定済み
- 128 bit固定
- 148種類の局所gateと11基底の検証対象を固定
- 324 component supportのcoverage対象を固定
- C/L誤差の重複は未解決
- 旧compiled programの保証は`MISSING_EVIDENCE`

**今回は、固定契約に基づくLの実H4数値診断を明示的に承認します。**

対象は、DF-nativeの局所gate・basis・event familyに関する保証付き数値証拠の取得です。

以下を一括で実施してください。

1. 固定SOURCE・runtime・入力・実行契約の最終確認
2. 新しいone-shot grantの発行・消費
3. 固定範囲内のL数値診断
4. 局所誤差上界・basis合成・support coverageの結果保存
5. 人工・回帰・数値・構造検証
6. SOURCE・SHA・scientific call・native終了・保全監査
7. 専用commitへの結果保存
8. mandatory STOP

**固定範囲内では、途中の細かな承認確認は不要です。**

ただし、未承認の計算追加、科学条件変更、正式certificate受理、N/G再会計には進まないでください。

## 2. 固定資料・実装の確認

まずGitHub上の固定commitと以下の資料を実際に取得し、内容・SHAを確認してください。

- `docs/non_df_h4_kl_production_20261023.md`
- `artifacts/non_df_h4_kl_production/2026-10-23/SOURCE.json`
- `artifacts/non_df_h4_kl_production/2026-10-23/seal_verification.json`
- `artifacts/non_df_h4_kl_production/2026-10-23/L_execution_contract_DRAFT.json`
- `artifacts/non_df_h4_kl_production/2026-10-23/coverage_ledger.json`
- `artifacts/non_df_h4_kl_production/2026-10-23/static_input_check.json`

実装：

- `src/trotterlib/h4_local_enclosure.py`
- `src/trotterlib/h4_local_control.py`
- `scripts/run_h4_local_enclosure.py`

関連するC productionコード、証拠adapter、証拠schema、既存テストも確認してください。

**最新版の研究共通運用指示（2026-10-23版）の§8.4・§10を適用してください。**

既存実装・関数・テスト・実行パイプラインの再利用を優先し、不要な再実装を避けてください。

封印済みL SOURCEと入力・runtimeが一致する場合、科学的なアルゴリズムや実装を変更せず、そのSOURCEで実行してください。

不一致がある場合は勝手に再封印・条件変更して続行せず、証拠を保存してSTOPしてください。

## 3. 今回の科学実行範囲

対象は固定済みのH4非DF/DF pilotのうち、**DF-nativeの局所gate・basisに関するL診断のみ**です。

既存の最大予算：

| 対象 | 最大実行数 |
|---|---:|
| Forward局所gate包含 | 148 |
| Inverse局所gate包含 | 148 |
| Basis合成・一様誤差伝播 | 11 |
| Symbolic component support確認 | 324 |

### 数値条件

- Precision：128 bit固定
- Arb/Acbによる保証付き区間演算
- 保存binary64パラメータのexactな取込み
- 外向き包含
- 保存descriptor・matrix・registry・SHAとの一致
- 既存の演算順序・位相・control規則を維持

128 bitで保証が成立しない場合、**96/192/256 bit等へ勝手に変更しないでください。**

未証明・nonfinite・入力不一致・異常終了は記録し、fail-closedでSTOPしてください。

## 4. 実施するL数値診断

### A. 局所gate

148種類の局所gateについて、保存forward/inverseと対応する理想的なRZ・XXPlusYY演算子を比較してください。

- 理想gateの保証付き構成
- 保存forwardとの誤差包含
- 保存inverseとの誤差包含
- Frobenius norm等による演算子norm上界
- 位相・符号・qubit順序・inverseの整合性
- 各gateのdescriptor・SHA・誤差区間の記録

理想gateのunitarityと、丸められた保存行列のunitarityを混同しないでください。

### B. 11基底の合成

保存されたordered basisについて、局所誤差を保証付きで合成してください。

- Forward/inverseの順序
- Basis変換・逆変換
- Control=0/1の両branch
- Global phase・relative phase
- 誤差のtelescoping bound
- 基底合成に対する一様上界

演算順序や位相を変更せず、保存済みモデルに対する保証を取得してください。

### C. PR component support

保存された324 componentについて、固定event familyをカバーしていることを確認してください。

- component ID
- basis対応
- sign・identity・phase
- qubit support
- order 0/2/4
- 固定r=4、q=8
- deterministic S2との組合せ

未観測eventを、過去の有限MC標本だけで網羅済みと扱わないでください。

**全324 supportに対する検証と、過去の観測event集合は区別してください。**

## 5. Cの保存数値証拠との関係

既存Cでは40軸すべてが64 bitで数値設計目標を満たしています。

今回は、その40軸を再計算しません。

特に、既存Cの誤差会計には以下が含まれています。

- `bias_distance`
- `circuit_semantic_allowance`
- basis reuse
- phase accumulation
- pi rounding

Lで取得する局所・basis誤差と、Cの既存semantic allowanceには重複が存在します。

**今回のL結果を、Cの保存totalへ単純加算してはいけません。**

以下を維持してください。

- Lの独立した局所誤差上界は保存する。
- 既存Cのtotal・allowance・biasは変更しない。
- 既存coverage ledgerを更新する場合も、未証明の重複解消を仮定しない。
- C/Lの結合が証明できなければ`COMBINATION_BLOCKED`を維持する。
- 新しい40軸の総誤差上界を作成しない。
- 正式certificate受理へ進まない。

Lの結果からC/Lの重複解消方法を具体化できる場合は、**後続作業案として報告するだけ**にしてください。

## 6. 実行制御とgrant

実行前に固定SOURCE・runtime・入力SHA・契約の一致を確認してください。

既存L契約に対する今回の明示承認を記録し、新しい未使用namespaceとL専用one-shot grantを準備してください。

既存の消費済みgrantは再利用しないでください。

次を維持します。

- 実行前SOURCE・入力・runtime照合
- 新grantの一回だけの有効化
- Call前の非返却予約ledger
- 重複launch・重複claim拒否
- 予約・実行数の監査
- native exit・worker回収
- 異常時のpartial保存
- 自動retry禁止
- STOP後の再実行拒否

前回のH4科学runでは元supervisorの終了証拠が欠けたため、今回も既存のowned native witnessによってworkerの終了記録を確実に保存してください。

ただし、新しい正常終了記録を元H4 runの欠落証拠へ転用してはいけません。

## 7. 計算資源と高速化

最新版共通運用指示の§8.4を適用してください。

- CPU・RAM・worker数に恣意的な低い上限を設けない。
- 実際の空き資源を確認し、効率的に実行する。
- 148種類の重複除去済み局所gateと保存basis情報を再利用する。
- 計算上有効なら独立処理の並列化を利用する。
- Arb/Acb precision contextの危険なthread共有を避ける。
- 科学的演算順序・精度・結果の再現性を変更しない。
- 共有環境全体や他ユーザーに影響する変更・操作は絶対禁止。
- 実際の資源不足や干渉の危険があれば、自分のjobだけを適切に制御する。

元の固定契約の資源条件も無断変更しないでください。

## 8. 禁止事項

今回承認していない作業：

- SCF/FCI
- 新DF生成
- 状態最適化
- Full-H reference再計算
- Law mean再計算
- Bias再計算
- Sampling・RNG replay
- 新規compile
- 旧compiled programの再生成
- Cの40軸再計算
- N/G再会計
- PF/PR候補の再選択
- Allowance・guard・B・lawの変更
- 正式certificate受理A
- 新しいcompiled program系列への切替え
- 詳細科学レビュー・最終evidence freeze

元H4科学SOURCE・結果・grant・ledger・marker・STOP、Cの保存数値証拠、K調査結果を変更しないでください。

旧compiler全体の保証は`MISSING_EVIDENCE`のまま維持します。

## 9. 数値・構造・終了監査

実行終了後、以下を確認してください。

1. 148種類のforward結果
2. 148種類のinverse結果
3. 11基底の合成結果
4. 324 component supportのcoverage
5. 各結果の有限性・区間包含・固定precision
6. SOURCE・入力・runtime SHA
7. Descriptorと保存registryの一致
8. 結果の欠落・重複・不一致
9. 科学call予約と実行記録
10. native exit・worker回収・STOP
11. 旧SOURCE・raw・grant・ledger・marker・STOPの保全
12. Cの40軸結果・旧N/Gの不変性

**技術的な成功、局所数値保証の成立、C/L結合の成立、正式certificate受理を区別してください。**

一部の局所gateで保証が成立しなかった場合は、その対象と原因を保存し、未解決を未解決のまま報告してください。

自動retryや追加precisionによる救済は行わないでください。

## 10. 成果物

専用結果領域に以下を保存してください。

- 148 forward / 148 inverseの局所保証結果
- 11基底の合成誤差結果
- 324 supportのcoverage結果
- 誤差区間・保証状態・SHA付きCSV/JSON
- C/L coverage ledgerと残存問題
- 実行SOURCE・入力・runtime・grant・call ledger
- native終了・worker回収・STOP
- 数値・人工・回帰・構造検証記録
- 結果報告書・README・引継ぎ資料
- mandatory STOP記録

元の成果物は上書きせず、新しい専用結果として保存してください。

成果物は専用commitに保存し、公開については有効な承認・repository運用規則を確認してください。

承認範囲内でpush可能なら実施してremote SHAを確認してください。認証等で失敗した場合は、保存済み成果物を変更せず原因と必要な操作を報告してください。

## 11. 完了報告

最終報告では次を簡潔に示してください。

1. 実行した148/148/11/324の結果
2. 各局所誤差上界・basis合成の保証状態
3. 128 bitで保証が成立した範囲
4. 未証明・未対応・失敗条件
5. C/Lの重複と結合可能性
6. Literal wrapperへの保証転送に残る課題
7. 旧compilerの未証明状態
8. 科学call予算・native exit・worker回収
9. SOURCE・SHA・保全監査
10. Commit・push・公開状態
11. 次に必要な作業と承認

**今回承認したLの限定数値診断について、実行準備・one-shot科学実行・結果検証・保存・終了監査まで一括で進めてください。**

途中の細かな確認は不要です。

ただし、**Lの成功をもって正式certificateを受理したり、N/G再会計へ進んだりしないでください。**

完了後は**mandatory STOP**してください。