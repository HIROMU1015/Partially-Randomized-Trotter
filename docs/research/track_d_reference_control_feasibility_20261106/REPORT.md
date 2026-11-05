# Track D：Reference・Control削減の技術的成立性調査

2026-11-06 JST。技術調査の到達点は **真空保持DF pathに残る制御削減機構を特定、正味full interrogation費用は未判定**。科学的GO、研究方向の正式採択・棄却、既存STOP解除はしていない。

## 1. 読み順と成果物

| 成果物 | 内容 |
|---|---|
| [CONTROL_BASELINE_AUDIT.md](CONTROL_BASELINE_AUDIT.md) | 達成済みcontrol最適化、SOURCE版、実装境界、保存価格の種類 |
| [REFERENCE_APPLICABILITY.md](REFERENCE_APPLICABILITY.md) | 原著gadgetの分枝、DF/Pauli pathごとのreference条件と位相 |
| [SIGNAL_PRESERVATION_MAP.md](SIGNAL_PRESERVATION_MAP.md) | X/Yの条件付き分布、coherent mean、weighted estimator、必要Nの保存条件 |
| [FULL_COST_COMPARISON_MAP.md](FULL_COST_COMPARISON_MAP.md) | 残る削減対象と追加操作、保存full-wrapper価格、全RPEへの接続 |
| [TECHNICAL_GAPS_AND_NEXT_CHECKS.md](TECHNICAL_GAPS_AND_NEXT_CHECKS.md) | D-1〜D-4、静的Gap、別許可を要する最小検証、次担当 |
| 本REPORT | 実施範囲、参照資料、確定事項、公開・STOP境界 |

## 2. 実施範囲と固定資料

専用branch：`codex/track-d-reference-control-feasibility-20261106`。親commit：
[e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8)。既存repositoryの分離worktreeで6文書だけを追加する。mainや他Trackのbranch、既存のdirty/untrackedファイルは変更しない。

指定 [PROJECT_MAP](../../../PROJECT_MAP.md) を入口として適用AGENTS、`research_project_instructions_unified_2026-10-25.md`の§8.1/8.2/8.4/10/10.3を実際に確認した。共通指示2026-10-25は元workspaceにあるlocal資料で、親commitにはないため、公開SOURCEに存在するかのようなリンクは作らず、原文も転載しない。利用者がchatに与えた現行AGENTS指示も適用した。

[研究概要・現状](../研究概要・現状.md)を全文読み、[研究目的](../研究目的・研究課題.md)、[VALIDATION_STATUS](../../../VALIDATION_STATUS.md)、[validation manifest](../../../artifacts/validation_manifest.json)の関連状態を確認した。固定親の概要は旧時点の記述で、後続完了/STOPを指定統合資料・個別結果から優先して追った。local dirty版の長い概要を固定親の本文と同一視していない。

| 指定固定資料 | 実際に使った範囲 |
|---|---|
| [3環境統合35714770](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/3571477093441ac4852166fe829f8ccc5db5ec4f/docs/research/pr_algorithm_improvement_three_environment_integration_20261105) | DUPLICATION_AND_REUSE_MAPのDF-CONTROL、MD-CONTROL、RTE-BASE、SHARING、RPE-COST等と関連固定資産。台帳schema・IDX-CIR-01/02、Track A/研究C等の関連行・SOURCE役割 |
| [文献調査e8b19ff6](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/docs/research/pr_rpe_end_to_end_literature_survey_20261106) | RESEARCH_OPPORTUNITIES O5、PRIOR_ART_COMPARISON、END_TO_END_COST_ANALYSIS_MAP、SOURCES。全面的な再監査はしない |
| [main固定546bca7e](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/546bca7ed2cece59b714b986a264d7b0b0871a01) | GitHub APIでmain SHAを確認。選択した共通5 library、native v5、counterのGit blobを過去固定版と限定照合 |

行列、Hamiltonian、state/reference、回路を生成していない。科学moduleをimportせず、SOURCE本文、Git bytes、既存test本文、保存JSONのmetadata/既存値だけを読んだ。runner、test suite、RNG replay、simulation、sampling、compile/transpile、保存値の数値再解析・再rankingは実施していない。重要科学レビュー、新定理、正式研究契約の作成もしていない。

## 3. 確認した一次文献

| 文献・版 | 直接確認した箇所と本調査での用途 |
|---|---|
| Roweほか、[Split-Evolution QPE v2](https://arxiv.org/abs/2604.14921v2)、2026-06-10 | PDF §II Figs.1–4/Eqs.1–13：2層CSWAP、operator順序、共通固有基底、vacuum/reference、準備oracleとの違い。§III Eqs.14–27/Tables I–III：対照のdiagonal-only control、追加register/cat、§IVの費用scope。全文全節の独立reviewではない |
| Simon・Love、[Halving the Cost of Controlled Time-Evolution v1](https://arxiv.org/abs/2511.13855v1)、2025-11-17 | HTML/PDF §II–III、Figs.2–3：対称PFのordinary control構成、directionalと逆順undo、higher-order適用。native既存構成との限定対応 |
| Güntherほか、[PR v2](https://arxiv.org/abs/2503.05647v2)、2026-07-10 | PDF §IV/V Figs.6–7、App.Aのcoherent mean、App.A.4 A43–A46、App.B冒頭のRPE、App.E19–E24：normalization、det/random、正負独立RTE、全round会計。原著の決定論側削減を比較に保持 |

SE/PRの指定HTMLは取得できなかったためPDFへ切り替えた。PDFと抽出textは一時領域でのみ用い、公開成果物に複製しない。書誌・版はarXiv一次ページで照合。原著の費用改善率・FeMoco実績を本研究の改善率へ転記していない。追加の広い先行研究探索は行わず、指定文献と既存調査を再利用した。

## 4. 確認したSOURCE・既存結果

| 対象 | 確認深度・用途 |
|---|---|
| [df_rte_qiskit.py](../../../src/trotterlib/df_rte_qiskit.py)、[df_rte_circuit.py](../../../src/trotterlib/df_rte_circuit.py)、[df_rte_tail.py](../../../src/trotterlib/df_rte_tail.py)、[rte.py](../../../src/trotterlib/rte.py) | controlのbuild分枝、basis/phase/order、finite event/lawとnormalization、DF I/Z/ZZとregistryの関連定義。全関数を独立監査した意味ではない |
| [df_partial_s2.py](../../../src/trotterlib/df_partial_s2.py)、[repeated](../../../src/trotterlib/df_partial_s2_repeated.py)、[ops](../../../src/trotterlib/df_trotter/ops.py) | D block、actual basis inverse、Gaussian producer、global/relative phase、boundary fusion |
| [ax2a_control_plan.py](../../../src/trottertracks/resource_applicability/ax2a_control_plan.py)、[native v5](../../../src/trottertracks/resource_applicability/ax2b_native_df_v5.py)、[AX2a native](../../../src/trottertracks/resource_applicability/ax2a_native_df.py) | symmetric構成、S2/S4、中央tail維持、scalar、native wrapper。SOURCE版の相違を保持 |
| [legacy wrapper](../../../src/trotterlib/rpe_hadamard_interrogation.py)、[counter](../../../src/trotterlib/native_df_gate_model.py)、[resource accounting](../../../src/trotterlib/rpe_resource_accounting.py) | X/Y・bit mapping、固定loweringとA/Bの相殺境界、raw半径とshot接続 |
| [typed mixed circuits](../../../src/trottertracks/pr_dictionary_validation/circuits.py)、[gates](../../../src/trottertracks/pr_dictionary_validation/gates.py) | Pauli parity、actual adjoint、support/full sharing、fusion、phase保持compile。mixed wrapperはfixtureのoccupied X準備を含む実装なので、価格をno-prep exact-groundへそのまま移さない |
| `tests/test_df_rte_qiskit_builder.py`、`test_df_partial_s2_repeated.py`、`test_rpe_hadamard_interrogation.py`、`test_native_df_gate_model.py`、`test_native_df_qiskit_connection.py`、対応support script | test本文と既存公開receiptを静的参照。control0/1、order/sign、basis/phase、実inverse、Y順序の既存coverage。新規実行なし |
| [AX-1b identity接続](../track_a_ax1b_identity_compatibility_fix_20261009.md)、[AX-2a準備](../track_a_ax2a_technical_preparation_v1.md)、[H4 v5結果](../track_a_ax2b_h4_pilot_v5_result.md)、[limited v2](../track_a_ax2b_h4_limited_execution_v2.md)、[native理論](../track_a_native_df_theoretical_gate_model_20261016.md)、[人工Qiskit接続](../track_a_native_df_qiskit_connection_validation_20261016.md)、[理論評価](../track_a_native_df_theory_evaluation_20261016.md) | 各scopeの関連節・SOURCE/結果metadata。既存wrapper実測、primitive/明示分解/期待価格、後続欠測の違いを確認。旧reviewを再開催していない |

H4 v5の個別2 cost recordを転記した条件は費用地図§3に全て記載した。SOURCE全体は未commit byte freezeで、base commitだけでは不足する。今回の3部品hash一致から全run SOURCEを新たに認定しない。統合台帳でSOURCEがUNKNOWNの古い行を現mainで補完していない。

mixed `gates.py` は固定F2版と同blob、`circuits.py` は現版のprovider-support接口追加で別blobだったため、9468c708の本文と差分も直接確認した。既存のcontrol/adjoint/fusion機構の共通性と、SOURCE全体の同一性を分けた。

## 5. 技術的結論と未解決事項

既存の強いbaselineは、Gaussianを非制御にし、中央diagonalだけを制御し、既存basis共有・D境界fusionを持つ。nativeはさらに対称PFのD halfの任意角負担を削減している。これらを未実装の改善候補へ戻さない。適用可能な最適化を一つの実測pipelineで全て合成済みとも主張しない。

Referenceの追加候補は、そこで残る中央RTEのCRZ/CRZZとCZ、directionalのancilla接続を除く交換である。物理的なnumber-conserving GaussianでconjugateされたZ/ZZとscalarからなるpath族は真空rayを保存し、event metadataから位相を追跡できる。一般Pauli/custom basis/mixed pathにはその条件を仮定できない。

SE原著Eq.2の分枝内積を照合すると、full Vωのreference保存と既知phase補正は、fresh-state一shotのX/Y条件付き分布保存の十分条件になる。この条件照合を、共通固有基底が必要なcoherent反復QPEの保証や新しい一般定理とはしない。sampling law/weights/normalization/trial規約を保てば同じNとdecoderへ接続可能だが、実装精度・noise・post-selectionを含む保証は未確認である。

追加register、2層CSWAP、routing、phase合成、cutで失うfusionを払った正味利益は、新しい同条件full-wrapper compileなしに確定できない。現在の位置は **DFの限定族でD-1の残余機構、既存最適化と重なる部分はD-2、一般真空適用はD-3、正味費用はD-4**。新規性や総費用優越の正式判定は行っていない。

## 6. 文書検査・公開・STOP

公開差分はこのdirectoryの6 Markdownだけとし、既存SOURCE/input/raw/result/grant/ledger/marker/archive/STOPを変更しない。既存索引・研究正本・manifestは更新しない。新しい科学結果、実行path、正式研究判断の変更ではなく、今回指示が公開物を6文書に限定するためである。

文書検査は、6件の存在、UTF-8、内部リンク先、固定SOURCE/refの存在、出典と条件の明示、変更path限定、`git diff --check`を対象にする。科学testsを文書検査のために動かさない。commit/push後、専用remote refのSHAと6文書の取得本文・Git blobを照合し、最終chatに公開commitと照合結果を示す。自分自身のcommit SHAを本文へ埋め込む循環更新はしない。

**mandatory STOP：Track Dの技術調査と6文書の公開取得照合が完了した時点で停止。** 有望さにかかわらずbuilder実装、科学計算、compile、詳細科学レビューへ進まない。

次の担当はユーザー／GPT。DF真空保持pathに限定した接続を継続するか、既知法の応用実装として扱うか、必要な研究判断を選ぶ。継続を選ぶ場合に限り、Codexによる最小人工gadget correctnessと一件の同条件full-wrapper compileを別許可の候補とする。重要科学レビューはさらに独立した許可事項。自動開始する次段階はない。
