# Track D：最小検証用DF pathの静的照合

2026-11-06 JST。基点 [c3cd64755944be4aad2e032affd1ed0308cf5743](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/c3cd64755944be4aad2e032affd1ed0308cf5743)。今回の追加は本Markdownだけ。SOURCE本文・保存metadataの読取と文書検査を行い、新実装、回路build、科学moduleのimport、simulation、sampling/RNG replay、compile、保存値の再会計は行っていない。正式共通指示2026-10-25 §8.1/8.2/8.4/10.3と既存契約・STOPを維持する。

## 1. 選定した1件

**既存人工fixture `native-partial-closed:B`** を、Referenceの初回correctness・単一path価格比較の技術対象に選ぶ。これは研究方向の正式採択ではない。

| 項目 | 固定された既存定義 |
|---|---|
| pathのSOURCE | [test_native_partial_whole_trajectory_matches_closed_model](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d2431b8c2a385af20a3d996cf8074b7a826a6e6b/tests/test_native_df_qiskit_connection.py#L283) の `model="B"`。model Aは別の共有設定であり今回の選定対象ではない |
| 人工入力 | [support script](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d2431b8c2a385af20a3d996cf8074b7a826a6e6b/scripts/native_df_qiskit_validation_support.py#L169) の `artificial_inputs()` と `rte_inputs(same_basis=True, identity=False)`。2 system modes、occupation bit p=mode p、number-conserving orbital frames u/v |
| deterministic blocks | one-body eigenvalues (.7, −.2)、DF squared η=(.4, −.9)、λ=−.6。framesと全数値は同helperを正本とする。分子geometry/AO basis/分子DF rank/L_Dは非該当 |
| outer条件 | T=.8、q=2、scalar c=.17、partial S2。half時間はSOURCEの `T/q/2` のまま |
| 中央ordered word | `sequence=(events[0],events[1])`。既存列挙順から、K=0、τ=.4、`Zplus`（frame a、Z0、sign +）→`ZZminus`（同frame a、Z0Z1、sign −）。両outer stepに同じwordを渡す。PRNGなし |
| 保存結果 | [connection_validation_v1.json](../../../artifacts/resource_applicability/track_a_native_df_qiskit_validation/2026-10-16/connection_validation_v1.json) の `observations[label="native-partial-closed:B"]` と既存test transcript。元statusは `QISKIT_CONNECTION_SYNTHETIC_PASS`、terminalは `MANDATORY_STOP` |

選定理由は、全入力を既存SOURCEから特定でき、scalar・正負rotation・D/RTE・basis共有を含み、既存人工参照を再利用できること。**このfixtureのτとouter時間は回路接続検査の入力で、単一Hamiltonianのτ=λ_R δ/rを満たす科学scheduleとして認定されていない。** 全PR law、元Hのenergy推定、必要shot数や性能の検証対象にはしない。

前回引用したH4 `H4_B2_K2_rep0` も確認したが、保存event列だけではdeterministic blocks・scalarを含むfull pathを再構成できず、指定NPZは本checkoutにない。今回これを生成・再sampleして埋めない。上記人工fixtureならその欠落を持ち込まずに最小検証を準備できる。

## 2. SOURCEと保存結果の結合

保存記録はbase `1ea42fbf…`、別native参照 [87b6b9f46b7e8ba9abe132ee21478c114b6f46f7](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/87b6b9f46b7e8ba9abe132ee21478c114b6f46f7) を明示する。baseだけを全実行SOURCEとしない。

今回、保存 `source_sha256` のcounter、2 test files、support script、runnerの5件について、読取中のbytesとd2431b8cのbytesが記録hashに一致することを照合した。`loaded_source_closure` はnative v5/control planを87b6b9f4に、共通libraryをbase/reference両版に結ぶ。この2 native filesと `df_partial_s2.py`、`df_rte_qiskit.py`、`df_trotter/ops.py`、`rte.py` の現bytesもclosureの記録hashと一致した。[前回baseline監査](CONTROL_BASELINE_AUDIT.md)の同blob確認も再利用する。保存receiptのGit blobは `ad99d94d1124450645256b0c45fd87da0ab47c31`。これは来歴の確認で、新たな数値correctnessの認定・test再実行ではない。

## 3. 真空・phase・identity

`U_j(t)`を上記number-conserving frameでconjugateしたone-body/squared number block、`E_+`/`E_-`を選定wordのrotationとする。既存の時間順を行列順に書けば

\[
S=U_0(t)U_1(t)E_-E_+U_1(t)U_0(t),\quad
t=T/(2q),\qquad V=e^{-icT}S^2.
\]

真空r=|00〉についての照合は次のとおり。

| 要素 | SOURCEとの対応・必要条件 |
|---|---|
| Gaussian | `U_to_qiskit_ops_jw_givens` は正方orbital unitaryのnumber-conserving変換。RZ/XXPlusYYのcomplete basisは真空rayを保持する。実basisのglobal phaseを捨てず、逆順actual inverseとのsandwichで相殺する |
| deterministic | [ops.py](../../../src/trotterlib/df_trotter/ops.py) `one_body_diagonal_primitives`／`df_squared_diagonal_primitives` はnumber演算子の発展。真空作用は1。内部 `primitives.global_phase` はRZ/RZZ分解に必要なので削除しない |
| RTE | [rte.py](../../../src/trotterlib/rte.py) の列挙はcomponent順、`unsigned_rotation_angle` はsignを含む。φ=atan(.4)としてE+の真空位相は−φ、E−は+φ。K=0なのでorder phase=+1、productはない |
| scalar | [native v5](../../../src/trottertracks/resource_applicability/ax2b_native_df_v5.py) `build_partial_native` は最後に `P(-c*T)` を1回置く。対応する物理Vのscalarは上式の通り |
| identity | `identity=False`なのでtail identity eventは存在しない。このfixtureに分子preparationの `extracted_identity_coefficient` を補作しない。primitive内のidentity成分、末尾scalar、tail identityの三つを区別する |

従って既知の恒等式をこのwordへ対応させると、真空位相は **χ=−cT（mod 2π）**。φの対消去を含む構造式であり、新しい数値phaseを評価していない。Referenceのancilla補正は **P(χ)**。無制御側でscalarがglobal phaseとして消えても、補正から省かない。

最小cutは二つのcomplete outer stepの間：`B=S`、`A=e^{-icT}S`。元pathからの分割であり、独立random halvesを作らない。A†は実回路 `.inverse()` によりscalarとgate順を含めて得る。[前回の分枝照合](REFERENCE_APPLICABILITY.md#2-今回の一shot照合)を適用し、fresh ψ⊗rの終端X/Y分布を比較する。system出力状態やgadget全体を直接制御と同じunitaryとは要求しない。

## 4. 既存最適化の適用境界

| 機構 | 選定fixtureへの適用・限界 |
|---|---|
| diagonal-only | event builderはGaussianを非制御とし、中央CRZ/CRZZと必要phaseだけを制御。native D blocksも同じ範囲を制御する。全Gaussian制御との比較は不要 |
| symmetric-directional | nativeのD forward=uncontrolled、reverse=directional、中央tail=ordinaryで適用可能。ancilla0で逆順Dがundoされる。K0 word全体を任意にdirectional化した方式ではない |
| RTE basis共有 | model Bでは同frame aの2 eventを同じoccurrence内で共有。nativeはouter間でtailを別composeし、共有scopeを閉じる。過去の保存Bが全PFを融合済みという意味ではない |
| deterministic fusion | legacy repeatedの同block half融合と、counter `assemble(model="B")` の隣接basis取消は再利用候補。native PF自体はこの全相殺をemitしない。ordinary/directionalの異なる中央modeを同じordinary halfとして合算しない |
| Reference cut | 同じ共有を各lane内に与える。cutを跨ぐD境界の相殺・rotation融合は失われ得るので、費用から落とさない |

初回の残余対象は **中央RTEのCRZ/CRZZとD directionalのancilla接続**。このpathにCZ productはなく、product削減や高次event、explicit identityの優位を検証したとはしない。wordはsame-basis/K0であり、一般の異basis random wordを代表する性能標本でもない。

## 5. 次に必要な最小変更と再利用

実装は今回は行わない。将来許可された場合も既存SOURCEを改変せず、新しい限定adapter/testだけを追加する案とする。

| 必要な差分 | 再利用する既存処理 |
|---|---|
| 固定fixture取得、explicit wordの再構成 | 上記support helpersと`RTEEvent`。新sampler、分子I/O、保存結果の再最適化は不要 |
| 無制御S/A/Bを組む小adapter | event builderの `controlled=False`、既存D primitive producerとbasis inverse。native `build_partial_native` はcontrol専用なので無制御builderとして偽装しない |
| SE型gadgetと読出し | A†/Bを2 laneへ配置、前後2層CSWAP、P(χ)、H/(Sdg)/H/measure。2 system＋2 reference＋ancilla。cat、途中reset、post-selectionなし |
| 公平な共有・fusion接続 | `assemble(model="B")`、既存block schedule・legacy境界条件。tailを省いたprefix価格だけでfull pathを代用しない。新しい最適化探索は不要 |
| correctness harness | 既存`connection`/`same_unitary`、Fock minor oracle、scalar/phase/adjoint testsを再利用。ただし既存wrapperは単一systemのancilla-last型なので、Referenceのmetadata・register配置は小adapterで明示し、旧型へ偽装しない |

## 6. 次の実行条件案とSTOP

以下は**未承認の提案**。旧receiptのSTOPやgrantを再使用しない。

**人工correctness**：同じV、同じχ、両axis・bit mappingを固定する。rの保存、actual adjoint、scalar付き直接controlの一致と、Referenceの誘導X/Y測定演算子を元system上で比較する。これなら特定のψだけの偶然一致を避け、人工ground state生成も不要。既存人工operator検査の絶対許容差 `3e-12` を出発点とし、phase fittingや失敗後の閾値緩和はしない。補正P省略を負対照にする。K>0/explicit identity/general Pauliは初回対象外と明記する。これは新build・数値operator評価を含むため別の実行許可が必要。

**full-wrapper比較**：同一pathについて①diagonal-only ordinary、②symmetric-directional、③Referenceの3方式、X/Yの6 wrapperを対象案とする。①②には既存basis共有と適用可能な隣接fusionの機会を与え、その新たに接続した構成を旧保存結果と区別する。③は同じ処理をlane内へ与え、cut損失を含める。初期ψの準備は全方式で除外し、CSWAP2層、補正P、追加reference qubits・読出しは計上する。

compiler案は既存F2のphase保持経路 `gates.compiled_wrapper` と `CompilerProfile`（Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、approximation_degree=1、qubits_initially_zero=False、backend/couplingなし）を全armで共通利用。新しい適用先なのでcompile前後の誘導測定演算子も照合する。既存の `transpile_and_measure_cost` はこのphase保持前処理を持たず、名称だけで同一profileとしない。RZ/CX/depth/size/qubitsを別々に記録し、T/物理routing/必要N/全RPE費用は評価しない。前処理・routingなしという限界も残す。

必要承認は、**この固定人工fixtureの限定adapter実装＋correctness数値検査と、PASS時だけの上記6 wrapper compile**。一括した固定scopeとして許可可能であり、各段で承認を取り直す仕組みは作らない。実装SOURCE・入力定義・compiler・比較scopeは実行前に固定する。重要科学レビューや研究採択の許可とは独立である。

correctness不一致、phase/入力/SOURCE不明、対応しないfusion、compiler非互換ならその時点でSTOPし、別path・別条件・閾値変更・retryへ自動移行しない。PASSしても指定比較と結果公開照合後にmandatory STOP。正味削減がなくても恣意的なcut/word探索へ拡張しない。旧人工fixtureの命令数guardは既存実装条件として保持し、新規jobに人工的なCPU/RAM/wall上限を追加せず、実資源と共有環境非干渉に従う。

**今回の終了条件**：本書だけを専用branchでcommit/pushし、remote SHA・取得本文・blobを照合してmandatory STOP。次の担当はユーザー／GPTによる上記固定scopeの実装・科学実行許可の判断。Codexは許可が与えられた範囲だけを後続実施し、現時点では計算へ進まない。
