# 既存builder・DF head接続・A/C HANDOFF

本書は技術設計で、new Pauli lowering/modelの採択や実価格取得ではない。[REPORT](REPORT.md)／[価格・有限案](PRICE_AND_VALIDATION_PLAN.md)。

## 再利用する関数とSOURCE

| 役割 | 既存file / API | SOURCEと適合条件 |
|---|---|---|
| generic K0/2 event | `src/trotterlib/rte.py: RTEComponent, finite_rte_distribution, _make_event, sample_rte_events` | P3 native9c4006bfとD基点のfileがbyte一致。今回は人工の明示indexだけで_make_eventを使い、samplingしない |
| Pauli rotation/event | `pauli_signal_adapter.append_pauli_evolution/append_pauli_event` | 同9c4006bf/現D一致。非identity単一IXYZ label、Qiskit little endian、実係数符号、元application_sequence |
| efficient controlled Pauli | `h4_d2_pauli_control.central_control_body/central_control_tokens/_rotation` | 同9c4006bf/現D一致。default LieTrotterのsingle PauliEvolutionGate＋明示X/Y/Z productのみ。all-I・任意Gaussian・複数PauliをまとめたHamiltonian gate等は拒否 |
| P3の結合例 | `matched_partition_stage3.residual_sequence/trajectory_parts`、後続 `matched_partition_price_mean` | P3 native9c4006bf/確認run04 SOURCE1cfce5e9。保存Gaussian halfにtailを挟む既存経路。元q8/r4/K4のrunnerを新candidateへそのまま起動しない |
| O1-M1比較経路 | `o1_m1_resource_bridge.build_controlled_trajectory` | library8a13539e。Pauli eventをto_gate().control(1)で制御、scalarも元step位置。D2と別の回路実装 |
| DF headと外部tail | `resource_applicability/ax2b_native_df_v5.build_partial_native` | provider c31/現D byte一致。既存ordered deterministic blockとn+1-wire controlled tail列を渡す。tailへ制御を二重付与しない |
| wrapper | 同nativeの`build_native_hadamard_wrapper`、又はP3の`fixed_signal_evaluation.hadamard_wrapper(already_controlled=True)` | P3 wrapper fileも現D一致。cosine/+Im sine、ancilla n、最終measurement。current nativeへ接続する場合はnative wrapperを使う |
| compile/metrics/cache | `rte_compiled_cost.transpile_and_measure_cost/TranspiledCircuitCostCache`、D fast fingerprint | 固定Qiskit1.3・basis/opt1/seed17を明示。新しいcompilerを作らない。cacheのlaw/quantity資格は別に保持 |

## Pauli eventの作用素と制御

labelの右端がqubit0。`P_j`は符号を含まないIXYZ string、係数符号は別fieldとする。rotationは `exp(-i sign(c_j) * theta * P_j)`。既存eventの `theta=atan(tau/(k+1))` で、τの符号も保持する。D側でsignを二重適用しない。

`append_pauli_event`は元application_sequenceどおりにproductを先に適用し、rotationを最後に適用する。各負productの−1と `event.phase=(-1)^(k/2)` はsource bodyのglobal phaseへ加わる。非可換productの順を並べ替えたりPauli積を位相なしでcollectしたりしない。

D2のsupport weight wに対する回路は、XならH、YならSdg→HでZ basisへ移し、非identity supportの最後をtargetとして他supportからCXを集め、中央に **CRZ(2 sign(c) theta)**、CXを逆順に戻し、局所basisを戻す。parityのCXは2(w−1)個だが、これは**未compileの構造数**でありcompiled CX/RZ単価ではない。

basisとparityは非制御のまま、central CRZだけを制御する。control=0では前後が相殺し、control=1では元rotationになる。product X/Y/ZはCX/CY/CZへ写し、source全体のglobal phase phiを `P(phi)` on ancillaとして一度だけ適用する。これはrelative phaseなので捨てられない。all-I scalarはこのrotation経路へ入れずAのscalarとして分離する。

「DF基底回転がなくなる」とは、一般のGaussian orbital basisをeventごとに往復する必要をPauli label用の局所Cliffordへ置き換えられるという意味である。長いsupportのparity、局所X/Y基底、K2 product、head境界は残る。

## DF head＋tailの最小接続

1. Aが固定した物理headのordered DF blocksと、別のPauli tail descriptorを受け取る。headをPauli化し直さない。
2. 元samplerのevent単位を保ち、signed piece/outerごとのsystem-only Pauli wordを既存append_pauli_eventで作る。D2はそのtail wordだけへ適用する。
3. `build_partial_native(blocks,...,controlled_tails=...)`へ渡す。ordinaryでは両head半分をordinary control、現symmetric_directionalでは最初をuncontrolled・後半をdirectionalとしてlow branchを相殺する。今回この両経路を人工行列で確認した。
4. 複数signed pieceは元scheduleのliteral `T*w/q`・allocation・順序を維持し、一pieceのnative呼出しはscalar=0、元carrierと同じ位置で最終total-T scalarを一回付ける設計を優先する。local段階のscalarと外側を二重に加算しない。
5. native full measured X/Y wrapper→既存canonical fingerprint/cache→元compilerへ接続する。tail D2のorder/product phaseと外側Hamiltonian scalarは別の量で、それぞれ一回だけである。

現在のD carrier composeはDF tail builderを内部で固定生成する。最小差分案はtail factory/representationの明示dispatchを持つopt-in接続で、default DF経路は不変とする。元native APIのcontrolled_tailsを使い、新しいhead emitter/compile pipelineを複製しない。P3の保存Gaussian一block headを、未確認の一般DF-prefix headへ単に再ラベルしない。

`candidate_price_adapter.NativeRuntime`はprepare後のDF component/basis/λを検査するため、Pauli辞書名だけの変更では使えない。新contextにはhead preparationとtail representationを別に保持し、tail用のA descriptor hash、ordered labels/coefficients、scalar/source、law/λ/K、q/R/signed allocation、wire/endian、native/lowering/wrapper/compiler SOURCEをkeyへ含める。event数だけをkeyにしない。

PauliHamiltonian.from_termsはcollect・並べ替え・係数検査をする。Aの固定辞書をD側で作り直す用途には使わず、既存event関数へAが固定したorderとlabelsを直接結ぶ。DF用Accounting/E[S]もPauliへ暗黙適用しない。初回はdirect Cを優先し、Sが必要なら既存counterへの適合を別の人工同値性で確認する。

## Aから必要なもの（未取得を推測しない）

- 元H/target/sector又はfull-Hilbert-space・qubit/JW mapping、状態/入力SOURCEとhead/tail分割の物理的成立性。
- ordered DF headのSOURCE/hash、Pauli dictionaryのordered ID/label/実係数/sign/hash、identity s_R・Hamiltonian constant・total scalarの数量と元binary64加算規則。
- λ_Rと正規化/finite-K/event lawのsource-bound descriptor、R/q/κ・signed weights/allocation/time hex、component/wordの合法support。Dは新実λを計算しない。
- 比較するDF辞書との対応範囲と、同じtarget/headを持つ候補のcanonical/physical identity。sector専用同一性を別encodingのfull circuit同一性に読み替えない。

調査中に公開されたA75fb0のPauli技術仕様を確認した。Aはoperator/dictionary段階と限定mean段階を分離し、LD1/m0 normal1 pathの価格接続を後続案にしている。実辞書/λ/新canonical候補はまだ未取得である。legacy physical CIDは辞書を含まないため、Rまで同じならDF/Pauliで衝突し得る。B管理のcanonical dictionary identityと完全source/contextで区別し、Dで独自の物理ID生成器を作らない。

同仕様は、tailの合計がsectorを保存しても個々のPauli eventが保存するとは限らないことを整理している。Dのnative eventはfull2^n空間で作り、36次元の射影Pauliをunitary gateとして使わない。state lift/sector同一性と新λはAの証拠を受け取る。scalarは保存s_Rの元operandと抽出差の扱いもAに従い、Dでsector traceや新Pauli scalarへ自動置換しない。BのPauli候補N/riskも未取得。Dの人工head接続PASSはこれらの物理的成立性を証明しない。

generic event primitiveのK0人工PASSは、Aの本番providerでK0を採用した証拠ではない。A75fb0では現evaluateのK2/4/6制限とK0追加の別判断を明示している。最初の実接続案は元K2を維持する。

## Cの比較guardへ必要な最小拡張案

C3a8b37fのguardは、現時点でDF旧/newの2 routeだけを `_rte_source` に登録し、同policyのPR/Full-Hに同じrte_rotation_loweringを要求する。`by_method['PR']`一つではDF-tail PRとPauli-tail PRの異なる合法routeを同時に表せない。Pauli D2は現在のguardでは**未対応として拒否される**のが妥当で、DF新ZZのIDを貼って通さない。

既存guardをC側でrepresentation/実装routeごとに拡張する案とする。共通なのは費用quantity・compiler/control semantics・measured XY/P0・target/scalarの契約。route固有のnative/lowering IDは別でよい。Pauli側は既存 `D2_candidate_central_rotation_control` policyとappend_event/D2/DF head/wrapperの固定SOURCE、activation/receipt側implementation_evidenceで束縛する。新しいIDをこの調査で正式採択したとはしない。

同じmetadataに見せるためのguard緩和はしない。DF側の旧汎用ZZを新DF実装と混ぜるS1拒否は保持し、明示されたPauli routeとの公平な比較だけを追加する。Cの実装をD branchへ重複作成・無断mergeしない。対応前の正式G入力はUNKNOWN。
