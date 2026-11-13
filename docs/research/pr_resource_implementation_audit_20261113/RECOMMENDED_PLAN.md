# 改善計画（変更する価値があるものだけ）

[IMPROVEMENT_CANDIDATES](IMPROVEMENT_CANDIDATES.md)から、研究目的への寄与が明確なものを選んだ。いずれも**未実装・未承認**であり、今回は自動で実施しない。

担当は候補として書いたもので、ユーザーが割り当てを決める。

| 順 | 改善 | 担当候補 | 承認の種類 |
|---:|---|---|---|
| R1 | 制御ZZ回転の実装形の検証と、価格定義の判断（候補1） | 検証はD（またはC）、判断はユーザー | 小回路のcompile・statevector検証の実行承認。採用時は価格定義変更の科学判断と再取得の承認 |
| R2 | `candidate_id`のmemo化（候補3） | B（共通API所有者）→A/Cは固定参照を更新 | 技術変更（科学実行なし） |
| R3 | RPE分岐選択の閉形式化とS3統計部品（候補2、6） | A | 技術変更＋人工同値性test。S3の科学実行は別承認 |
| R4 | 固定SOURCE loaderの事前検査helper（候補4） | B（共通）、A/C/Dは新規codeで使用 | 技術変更 |
| R5 | 共有moduleの所有・固定方針と既存FAILの解消（候補5） | B（Cの同意を得て） | 技術変更。固定結果の参照commitは変えない |
| R6 | D inventoryのcache（候補10） | D | 技術変更（任意、低優先） |

## R1：制御ZZ回転（最優先の判断事項）
- **範囲**：
  1. 小回路（ancilla＋2 system qubit）で、`RZZGate(2θ).control(1)`と`cx·crz·cx`のstatevector/unitary同値性（ancilla位相を含む）を、数角度・両符号で確認する。
  2. 同じ小回路を固定profile（Qiskit 1.3.0、rz/sx/x/cx、opt1、seed17）でcompileし、RZ/CX数を比較する。最大8回路。
  3. 同値の場合、`df_rte_qiskit`の新しい版（新SOURCE）で、tail生成だけを置き換える案を作る。provider `c31ef424`は変更しない。
- **判断**：
  - 研究の価格定義をどちらの回路にするかを、ユーザーが決める。旧価格（C1、long-time、Full-H、K1対応）は旧SOURCEのまま保全する。
  - 採用する場合は、PR・Full-H RTEのcompiled価格と構造Sの対応を新SOURCEで取り直す範囲を、別に計画する。PFは影響を受けない。
- **関連**：過去の研究C監査（`research_c_rustiq_interface_audit_20261024`、Qiskit同梱`synth_pauli_network_rustiq`）とは、回路合成の改善として統合して判断できる。

## R2：`candidate_id`のmemo化
- **範囲**：`pf_pr_rpe_pipeline.candidate_id`が使う`cid`関数を、一度だけ構築してcacheする。
  - 入力file・関数名・出力は変えない。
  - 必要なら、固定blobのSHA照合を初回に1回だけ行う。
- **検証**：
  - B一覧のtest（106件）と、A/C/Dの関連test。
  - cacheの有無で、人工candidate集合のIDが一致すること。
  - 実測の目安：`check_rpe_new_evidence_intake`が75 s→16.5 s。
- **注意**：各担当の固定結果が参照するcommitは変えず、新commitとして配布する。

## R3：RPE分岐選択の閉形式化（S3の前提）
- **範囲**：
  - `_vectorized_reconstruction`と同じ入出力・同じ距離式・同じambiguous判定（1e−12）を持つ、O(1)/roundの新関数を追加する。既存関数は参照実装として残す。
  - S3の未解決率の区間は`scipy.stats.binomtest(...).proportion_ci`を使い、方式を事前に固定する。
- **検証（人工のみ）**：次の条件で、分岐・推定・ambiguousの完全一致を確認する。
  - 乱数位相（M≤12）
  - wrap境界と半分の間隔のtie
  - 小qの全探索
  - 実測の目安：M=12で180倍、memoryはO(反復)。
- **承認**：S3の科学実行（試行数、δ、η、未解決の扱い）は別承認。MSE区間の方式（Hoeffding、分散利用型）は、その時にユーザーとAが選ぶ（候補7）。

## R4：固定SOURCE loaderの事前検査
- **範囲**：小さな共通helperを作る。
  - 固定blobから関数を抽出してcacheする。
  - `symtable`/`ast`で自由変数を列挙し、scopeに無ければexec前にエラーにする。
  - `NodeTransformer`の一致件数を、期待件数と照合する。
- **適用**：新しく書くcodeから使う。実行済みの固定SOURCE・A `h4_pr_rmse_completion`は変更しない。Aが次の版を作る時に置き換える。
- **検証**：S0a run01型の欠落（`ast`の未提供）を事前に検出すること。既存loaderと同じ関数出力になること。

## R5：共有moduleの所有と固定
- **範囲**：
  - `pf_pr_rpe_pipeline.py`、`full_baseline_v2_requests.py`、`mean_compiled_price.py`の正本をBとし、A/C/DはB commitを固定参照する。
  - D統合版`full_baseline_v2_requests`（e7dca5f7）は、B公開版と同一のV2 IDを生成することを確認済み。次のB統合で採用を検討する。
  - Cのbyte固定test（`4cb61ef4`）は、固定blobとの比較へ直してB tree上の誤検出FAILを解消する（Cの同意が必要）。
- **検証**：全担当のtest。既存固定結果のmetadata照合（再計算なし）。

## R6：D inventoryのcache（任意）
- **範囲**：`saved_grid`とselection IDsのcache、endpointのindex化。
- **効果の目安**：`coverage()`の14.6 sのうち約11 sの削減（推定）。
- **検証**：COVERAGE/RECEIPTSの出力が完全に一致すること。

## 実施しないもの（現状維持）

次は、利益が小さいか科学リスクが大きいため変更しない。
- 物理核（eigh/spectral power、finite RTE、expm_multiply、Decimal ceiling）
- compiler pass構成
- `same()`とidentityの埋め込み
- git helperの統合
- 構造S counterの複製（C1凍結）
- pydantic/jsonschema・numba・threadpoolctl・hypothesisの導入
- C K1区間演算の置換（python-flintは任意の独立照合に限る）
