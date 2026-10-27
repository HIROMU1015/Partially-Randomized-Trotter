# P2依存closureとGit履歴

summary `b6c899698645a4ccdaa65c0c5e8d706fd9d1cc2b` はP2 `04f4cd9cdc8e348a9f0a182ead9cd07f1ee69696` の祖先。merge-baseはsummary、直線7 commit。
P2の直接parentは `00958ae1945c780718b2deef07d5dab1096a5cd8`。先行6と最後1の差を同じ採用単位にしない。

|commit|変更file数|元commitの役割（件数はmetadata）|
|---|---:|---|
|`f7757c4320ba34dd36a64de4250c29a69bf38390`|4|docs: audit H6 Hybrid/RTE search coverage and propose limited extension|
|`bcc2eb6b201e1434bf47560775f5afb3e14d3ca8`|16|feat: prepare fixed H6 Hybrid fairness adapter and disabled execution plan|
|`887b69e449cb3c378286f09d0d7626599850fb67`|1|feat: freeze H6 fixed-four Phase B workers using saved kernels|
|`af891fd45805454c6592bbee5441266f14235ffd`|1|feat: freeze H6 paired statistical design and conditional comparison|
|`f1b3dd8627a75555362939f360e0ddcdf0da5acf`|9|chore: preserve running H6 fixed-four continuation and result checks|
|`00958ae1945c780718b2deef07d5dab1096a5cd8`|37|results: complete fixed-four H6 Hybrid fairness comparison|
|`04f4cd9cdc8e348a9f0a182ead9cd07f1ee69696`|18|Add Pauli-JW study pipeline with explicit existing-core bindings|

最初はH6公平性資料、次に既存H6 adapter/準備test、科学SOURCEとworker、解析、継続/保全、最後に結果handoff。
元の科学worker/解析scriptを新開発の起動対象へしない。今回そのruntimeを確認/実行していない。
最後のP2 commitだけが18ファイル差分で、旧kernel/runner/initializer/SOURCE/result/STOP編集は0。

## 選定コードclosure

P2が要求するPauliHamiltonian/RTE compiler/H4 PR/H4 mean/H6 extension/pure-RTE/fixed-account/atomic writerから、
ASTの局所Import/ImportFrom（条件付・関数内を含む）とliteral `A.extract` のcode参照を辿った。
限定57 code path。summary比較では**56同blob、異なる既存blob0、欠損1**。
これは新全branch監査でも、数値依存全体・optional SDK・raw/archiveを復元した証明でもない。
legacy root initializerの別closureや動的非literal/外部backendは未完全検証で、実環境は依存不足のまま。

欠損は `src/trotterlib/h6_hybrid_fairness_extension.py` だけ：

- 追加元：`00958ae1945c780718b2deef07d5dab1096a5cd8:src/trotterlib/h6_hybrid_fairness_extension.py`
- Git blob：`0f09c36f1b191abba75e39f56746a1828f4419f8`、5,567 bytes
- 初出：readiness `bcc2eb6b201e1434bf47560775f5afb3e14d3ca8`。以降科学SOURCE/P2 baseで同moduleを保持。
- 同pathへbytesそのまま選択追加。新law/mean/seed/account kernelを生成しない。
- fixed-four signature/config、保存入力identity、元lawのprivate extraction、cohort seed、trajectory/mean/accountは元bodyのまま。

主な既存同blob依存：pauli_signal_adapter.py、rte.py、rte_compiled_cost.py、h4_conditional_cng.py、
h4_pauli_pr_search.py、h4_pauli_pr_mean.py、pauli_rte_event_extension.py、h6_rte_fairness.py、
fixed_signal_evaluation.py、fixed_signal_accounting.py、pauli_vector_tight_enclosure.py、h4_d2_pauli_control.py、
rte_event_audit.py、parallel_validation_executor.py、scripts/pauli_h2_h6_study.py。
これらは再移植/再実装不要。`PauliSignalAdapter`とwrapper/central controlは既存trajectory内で呼ばれ、新wrapperを重ねない。
generic fixed_signal_shotsを研究固有のguard/exact-integer会計へ置換しない。

### 既存コード内の明示的抽出参照

|呼出し元|既存code対象|
|---|---|
|`src/trotterlib/pauli_rte_event_extension.py`|`src/trotterlib/h4_pauli_pr_search.py::law_descriptor`|
|`src/trotterlib/pauli_rte_event_extension.py`|`src/trotterlib/h4_pauli_pr_search.py::trajectory_seed`|
|`src/trotterlib/pauli_rte_event_extension.py`|`scripts/pauli_h2_h6_study.py::seeds`|
|`src/trotterlib/h6_rte_fairness.py`|`src/trotterlib/h4_pauli_b0_comparison.py::account`|
|`src/trotterlib/h4_pauli_pr_search.py`|`src/trotterlib/h4_ball_evidence.py::cdf_binary64`|
|`src/trotterlib/h4_pauli_pr_search.py`|`src/trotterlib/h4_ball_evidence.py::grid_masses`|
|`src/trotterlib/h6_hybrid_fairness_extension.py`|`src/trotterlib/h4_pauli_pr_search.py::law_descriptor`|
|`src/trotterlib/h6_hybrid_fairness_extension.py`|`scripts/pauli_h2_h6_study.py::seeds`|

H6追加fileのimport時にも `A.extract` が元law/seedのfunction ASTを組み立てる。対象は同checkoutの既存codeで、科学実行や別worktree依存ではない。
これを新しいalias loaderで代替しない。実dependencyが足りない状態で既存moduleを別名ロードしてtest PASSにしない。
既存bridgeの動的backend/alias処理は科学kernelの歴史的実装として保持するが、今回起動していない。
H4 pure-RTE専用のartifact内mean code参照など、今回接続しない経路のデータ/動的closureは完全確認の対象外。

## import・証拠・codeの区別

summary/base/P2でtrotterlibとtrottertracksのinitializer blobは同一。
通常の既存package importは旧eager closureを通り`matplotlib`不足で停止。新bindingが明示要求しなくても旧DF/phase moduleが間接importされる挙動は残る。
外部SDKの必要性を隠さず、task専用mockを実science providerと扱わない。

H6のSOURCE.jsonの科学commitは `887b69e449cb3c378286f09d0d7626599850fb67` で、結果handoff tipや新P2 commitとは別。
原SOURCE/STOP/resultのlocatorがGitHub固定refへ辿れることと、summary checkoutにその結果filesがあることは別。
H6の後続result/SOURCE/STOPを新たにsummaryの同名pathへコピーしない。元固定ref参照だけで保全する。

全7 commit mergeは科学worker・計画・新結果・途中状態を一括採用してしまう。
次回必要なのはP2採用13 path＋不足H6 code1 pathであり、先行6 commit全体・その研究成果コピーは不要。
