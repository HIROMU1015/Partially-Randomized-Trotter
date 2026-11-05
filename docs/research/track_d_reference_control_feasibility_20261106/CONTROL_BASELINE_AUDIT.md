# Track D：controlled回路の既存baseline

2026-11-06 JST。静的SOURCE照合。実行範囲と文献一覧は [REPORT](REPORT.md)、追加費用は [FULL_COST_COMPARISON_MAP](FULL_COST_COMPARISON_MAP.md)。新build・compile・test実行はしていない。

## 1. 比較の出発点

Gaussian basis changeの全gateを制御する回路は主baselineにしない。既存DFはbasisを非制御とし、中央diagonalと必要なscalar/event phaseを制御する。さらにnative系列は対称PFの前半をuncontrolled、逆順後半をdirectionalにして、同じordinary-controlled演算子を実装する。中央RTEはこの最適化後にもordinary-controlledのまま残る。

ただし、legacyの境界fusion、native directional、理論モデルBのGaussian相殺は、全てが一つの実測pipelineで同時に実現済みという意味ではない。比較には適用可能な各既存構成と同じcompiler最適化機会を与え、構成間の価格を足し引きして架空の実測baselineを作らない。

## 2. 版と役割

リンク先は全て `HIROMU1015/Partially-Randomized-Trotter`。調査branchの親は `e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8`。

| 固定版 | 役割・今回確認した範囲 |
|---|---|
| [main 546bca7e](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/546bca7ed2cece59b714b986a264d7b0b0871a01) | GitHub branch APIでもこのSHAを確認。収集された現mainのimplementation snapshot。過去runの実行SOURCEではない |
| [2978e2fe](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/2978e2fea672b7a1ff20cac74269ec9a610159dc/src/trotterlib) | 統合再利用mapのDF-CONTROL固定snapshot。legacy event/repeated builderを読む入口 |
| [87b6b9f4 native v5](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/87b6b9f46b7e8ba9abe132ee21478c114b6f46f7/src/trottertracks/resource_applicability/ax2b_native_df_v5.py) | native providerの固定SOURCE。人工Qiskit接続検証が実際に読み込んだ別SOURCE。H4 pilot全体のSOURCE commitと同一視しない |
| [d2431b8c counter](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d2431b8c2a385af20a3d996cf8074b7a826a6e6b/src/trotterlib/native_df_gate_model.py) | zero-RZ訂正後の明示分解counter。compiled priceではない |
| [9468c708 mixed SOURCE](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/9468c7089727f5648d9f59cddbaebe0a42f5f0a0/src/trottertracks/pr_dictionary_validation) | F2のphase保持修正版。typed DF/Pauli router、actual adjoint、X/Y wrapper、compileの再利用候補。合成fixtureの結果を分子へ移さない |

次の5ファイルは、main、調査親、2978e2fe、87b6b9f4でGit blobが同一と照合した：`df_rte_qiskit.py`、`df_partial_s2.py`、`df_partial_s2_repeated.py`、`rte.py`、`rpe_hadamard_interrogation.py`。native v5はmain・調査親・87b6b9f4で同blob。counterはmain・調査親・d2431b8cで同blob。この限定した同一性をSOURCE closure全体や科学条件の一致へ拡張しない。

mixedの `gates.py` はmain・調査親・9468c708で同blob。一方 `circuits.py` は9468c708と現snapshotで異なるため固定版本文にも戻った。現版は `provider.reference_support` の接口を追加している。今回引用した中央control、actual adjoint、run共有、D境界fusion、wrapperの機構は固定F2版にもあるが、custom providerの実行条件まで同一とはしない。

## 3. 実装されている操作

以下の相対SOURCEリンクは調査親に存在する本文。主要関数と分枝を直接読んだ。

| 対象 | 実装・条件 | 確認できる範囲／残る削減 |
|---|---|---|
| Gaussian basis | [event builder](../../../src/trotterlib/df_rte_qiskit.py) `_append_basis` は逆順＋各gateの `.inverse()`、中央演算、forwardの順。制御は中央だけ。`control_diagonal_only=False` を拒否 | basis制御除去は達成済み。Referenceの新利益として再計上しない |
| 中央rotation | 同 `_append_central_application`：Zは `RZ(2 angle).control(1)`、ZZは `RZZ(2 angle).control(1)` | 非対称random pathのCRZ/CRZZは残る。Referenceでは通常rotationへ置換し得る |
| 中央product | 同関数：support各qubitへのCZ。非制御側はZ。負係数productの符号は別phase | control由来のentangling操作は候補。ただしZ/CZはCliffordであり任意角T削減と同一でない |
| identity/event phase | `_identity_application_phase` と `_event_phase_angle` を集約。controlledではancilla `PhaseGate`、uncontrolledでは `circuit.global_phase` | scalarを捨てる置換は不可。Referenceでも既知参照位相へ戻す必要がある |
| basis共有 | 同 `_build`：同basis ID/hashの隣接applicationを共有、event境界を越えてflatten。identityでactive basisを閉じる。明示 `DFRTEBasisPlan` 対応 | full/support選択とrun共有は既存。異basisをsupportだけで同一視しない |
| deterministic DF | [partial S2](../../../src/trotterlib/df_partial_s2.py) と [ops](../../../src/trotterlib/df_trotter/ops.py)：one-body/DF squaredのdiagonal primitivesを同じphase規約で使う | Gaussianはここでも非制御。native directionalの適用先 |
| half-step融合 | [repeated](../../../src/trotterlib/df_partial_s2_repeated.py) `_build_boundary_optimized`：step境界の同じblock 0のhalfをfull timeへ融合。constant/extracted identity phaseも集約 | この具体的fusionは達成済み。任意D/RTE境界の全basis共有を実装済みとはしない |
| native対称制御 | [control plan](../../../src/trottertracks/resource_applicability/ax2a_control_plan.py)、[v5](../../../src/trottertracks/resource_applicability/ax2b_native_df_v5.py) | forward uncontrolled／reverse directional、global S2・Yoshida S4。中央tailはordinary control。tailをfinite mean gateで代用しない |
| X/Y wrapper | [legacy wrapper](../../../src/trotterlib/rpe_hadamard_interrogation.py)、native `build_native_hadamard_wrapper` | H → already-controlled evolution → Y軸のみSdg → H → ancilla測定、bit 0/1を+1/−1。再controlled化しない |
| ancilla | legacy/nativeはsystem n qubitsの直後に制御ancilla 1個 | Referenceは別途n-qubit registerが必要。既存controlに常にn個の補助qubitが必要という比較は誤り |
| mixed/Pauli | [circuits](../../../src/trottertracks/pr_dictionary_validation/circuits.py) `append_central`、`build_event_sequence`、`actual_adjoint` | Pauliの基底変換・parity計算は非制御、中央rotation/productを制御。run共有・D境界fusion、実回路 `.inverse()` を既に持つ |

`df_rte_circuit.py` はtyped request・basis plan・出力metadata、`df_rte_tail.py` はDFのI/Z/ZZ分解とbasis registry、`rte.py` はfinite law・order/sign・normalizationとproduct→rotation順を定義する。registryは任意runtime gate列を受けられるため、型名DFだけで真空保存を保証しない。[Reference条件](REFERENCE_APPLICABILITY.md)の限定が必要。

## 4. 対称controlledの二つの意味

[Simon–Love v1](https://arxiv.org/abs/2511.13855v1) §III/Fig.3は、対称PFから **diag(I,S)** を作る構成。これとnativeのforward uncontrolled／reverse directionalは同じ消去機構に対応する。DFでは対角Z/ZZのparityを通じてdirectionalを作れる。basisが非制御であることだけでは、中央CRZの任意角rotation倍増は消えない。

一方、[PR v2](https://arxiv.org/abs/2503.05647v2) App.A.4 Eqs.A43–A46は **diag(S†,S)** によりHadamard信号の時間を倍にする利用法も述べる。PRでは正負のRTEを独立に展開し、決定論部分だけを削減する。独立random halvesの採用は、今回固定した個々のpath/lawの単なるcompile変更ではない。原著E21にその決定論側削減が含まれる点も比較から落とさない。

任意ordered random pathで全gateの角度だけを反転しても、一般にactual adjointにはならない。native partial構成が使うのは **既知の逆順D half** と **ancilla0で恒等となる中央controlled tail** であり、random pathそのもののpalindromeは仮定しない。

## 5. 価格証拠の種類とSOURCE上の注意

| 保存資産 | 価格の意味 | 制限 |
|---|---|---|
| [AX-1b](../track_a_ax1b_identity_compatibility_fix_20261009.md) | 保存full-wrapperのモデル・compiler identity接続 | 同じwrapper semanticsやモデル存在はnative新回路の実測価格を意味しない |
| [AX-2B H4 v5報告](../track_a_ax2b_h4_pilot_v5_result.md) | measured full Hadamard wrapperの固定trajectory compile結果。ordinaryとdirectional、両軸 | small local evidence。N/全RPE/要求精度資格なし。後続limited execution v2にwrapper欠測があっても旧v5を未実行へ戻さない |
| [native A/Bモデル](../track_a_native_df_theoretical_gate_model_20261016.md) | 固定loweringのprimitive/期待count。Bは明示Gaussian相殺 | native builderのPF raw列はBをemitしない。T/実compiler最適価格ではない |
| [Qiskit接続](../track_a_native_df_qiskit_connection_validation_20261016.md) | 人工例でraw順序、global phase、固定lowering、ordinary/directionalの一致 | 過去tests・receiptを読むだけ。今回PASSを再実行していない。分子Referenceの検証でない |
| [研究Cなどのnominal/固定word](../pr_rpe_end_to_end_literature_survey_20261106/PRIOR_ART_COMPARISON.md) | 元条件付きのprimitive/native/固定word価格 | own-DF ground、全law資格、要求精度・Nの違いを保持し、元H groundの全費用へ移さない |

H4 v5はbase `b2e1bf65e21893b6c617223b42313623d3186f12` に未commit SOURCEを加え、[実行前manifest](../../../artifacts/resource_applicability/track_a_ax2b_h4_fingerprint_preparation/2026-10-09/h4_preparation_manifest_v5.json)の `local_uncommitted_byte_hashes` で凍結した。baseにはnative v5自体がなく、baseを実行SOURCE全体と呼べない。今回、manifest内の `df_rte_qiskit.py`、`ax2a_control_plan.py`、`ax2b_native_df_v5.py` の3 hashと87b6b9f4の対応bytesを照合し一致した。これは3部品の来歴を結ぶ確認で、全157 SOURCEの再認定ではない。公開carrierは現main、run SOURCEは元manifest closureという区別を保持する。

## 6. 再利用と残る対象

既存のphase保持event、basis registry、actual adjoint、wrapper、native control plan、公開済み固定結果を再利用する。mixedの [gates.py](../../../src/trottertracks/pr_dictionary_validation/gates.py) はnested gate定義のphase欠落を避けるcompile前loweringも持つが、別の科学SOURCE・条件への適合は未確認。

残る具体的対象は、中央random RTEの制御rotation・制御productと、対称制御でも残るancillaへのparity接続である。Gaussian制御除去、通常のW†W相殺、対称PFのrotation削減をReference固有効果へ数えない。Referenceがその残余費より安いかは未測定。
