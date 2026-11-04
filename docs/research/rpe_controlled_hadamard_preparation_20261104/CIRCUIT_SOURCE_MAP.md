# Controlled RTE・Hadamard：SOURCEと演算子の対応

2026-11-04 JST / LOCAL / Codex C。静的調査・人工検証・限定計画だけを実施。
基点は `f4905fff592351d33ba07b927892fb9532c99711`、専用branchは
`codex/rpe-controlled-hadamard-preparation-20261104`、別worktreeで作業した。
旧production SOURCEを変更していない。参照ファイルのbyte identityは
[SOURCE_BINDINGS.json](SOURCE_BINDINGS.json)に記録する。

対象条件はlinear H4／隣接1.00 Å／STO-3G／保存legacy DF12、8 system qubit、
保存physical state・4電子sector70、Partial `L_D=3, δ=.02`、
`m=0..17, q=2^m, r∈{1,2,4,8,16,32}, K=2`、
`extract_identity_phase, coefficient_atol=1e-12`、保存N/α/β/ε。
以下の式はSOURCEの意味の整理であり、この条件の新数値評価ではない。

## 三つの量を分離する

列vectorへ右から作用させる。決定論blockのhalf evolutionを
`D_j=B_j exp(-i δ h_j/2) B_j†` とすると、回路append順のforwardは
`D_0,D_1,...,D_(d-1)`、行列は `F=D_(d-1)...D_0`。
reverseは同じ正時間halfを逆順にappendし、`R=D_0...D_(d-1)`となる。
**RはFの逆行列でもadjointでもない。** 以下ではconstant/extracted identity位相をFに含める。

1. PF・理想tailの演算子：`U_PF=R exp(-i δ H_R) F`。
   finite RTEの平均・単一trajectoryと同一ではない。
2. trajectory `T` の実装対象：
   `U_j(T)=R E_(j,r-1)...E_(j,0) F`、
   `V_T=U_(q-1)(T)...U_0(T)`。
   保存stateのnormalized rayに対し `z_T=ψ†V_Tψ/(ψ†ψ)`。
   ordinary-controlled Hadamardの条件付き平均はRe z_T / +Im z_T。
3. categorical平均：独立な各位置のchoiceを仮定して
   `ν=E_T[z_T]=ψ†(R M^r F)^q ψ/(ψ†ψ)`。
   `M=E[E_event]`。これは平均**振幅**演算子であり、
   random-unitary channel `E[VρV†]`、`E[|z_T|²]`や単一trajectoryのq乗ではない。

イベントを1回引いてq回使い回すと一般に `E[U_T^q] != E[U_T]^q`。
非可換性があっても、独立な位置の有限和を順序を保持して展開すれば上の積は成立する。
PCG64の固定列を数学的IIDと認定したことにはならない。

## SOURCE対応表

リンクは本branchの変更していないproduction SOURCEへ向ける。

| 構成 | 関数・入口 | 演算子・注意点 |
|---|---|---|
| DF split・抽出 | [prepare_df_partial_s2](../../../src/trotterlib/df_partial_s2.py#L328)、[extract_df_diagonal_tail](../../../src/trotterlib/df_rte_tail.py#L563) | deterministic one-body correction＋保存DF prefix。tailのIを抽出しconstantと区別。cutoff既定値は0なので明示1e-12が必要 |
| F/R | [step builder](../../../src/trotterlib/df_partial_s2.py#L816) `_append_block/_build_parts` | basis B†→diagonal evolution→B。Fだけにstep scalar位相。Rはblock逆順 |
| diagonal scalar/RZ/RZZ | [ops.py](../../../src/trotterlib/df_trotter/ops.py#L226) | one-body `sum ε_j n_j`、fragment `λ(sum η_j n_j)^2`。Pauli化によるscalarも含み、制御時はancilla PhaseGate |
| component | [component_dense_operator](../../../src/trotterlib/df_rte_tail.py#L907)、[RTEComponent](../../../src/trotterlib/rte.py#L93) | `P_l=s_l B_l Z_support B_l†`, `s_l=±1`。`coefficient_abs`はprobability/λに使い、unitaryへ再乗算しない |
| event生成 | [_make_event](../../../src/trotterlib/rte.py#L1594) | 入力indexはrotation first、products next。`selected_component_ids/application_sequence`はproducts first、rotation last。取り違え禁止 |
| event参照 | [event_unitary](../../../src/trotterlib/rte.py#L1751)、[_apply_event_to_state](../../../src/trotterlib/rpe_hadamard_failure_validation.py#L358) | `E=(-1)^(n/2) exp(-i θ_n P_rot) P_n...P_1`, `θ_n=atan(τ/(n+1))`。productをappend順に左乗算 |
| controlled event | [QiskitDFRTEEventCircuitBuilder](../../../src/trotterlib/df_rte_qiskit.py#L162) | `_build/_append_central_application`。Z/ZZ productはCZ列、負productはπ位相、rotationは`RZ/RZZ(2*s_rot*θ).control(1)` |
| r event | [DFRTEEventSequenceCircuitRequest](../../../src/trotterlib/df_rte_circuit.py#L455) | occurrenceにr個を指定。同じeventのr乗と一般に異なる。隣接basis取消は登録ID/hash/planを照合 |
| q step | [DFPartialS2RepeatedRequest / builder](../../../src/trotterlib/df_partial_s2_repeated.py#L114) | step-major、`U_(q-1)...U_0`。全qのoccurrenceを保持するため構築量はqに線形。power-gateへ自動圧縮しない |
| boundary最適化 | [repeated builder](../../../src/trotterlib/df_partial_s2_repeated.py#L575) | constant/Iをq倍して1回だけ適用。隣接する同じD_0 halfだけfull δへ融合。他block/eventの順序は動かさない |
| support policy | [apply_run_threshold_policy](../../../src/trotterlib/research_direction_full_scope.py)、[sequence policy](../../../src/trotterlib/research_direction_sequence_policy.py) | `support_run_le_1`。保存training fingerprint `9dcdc4a28866dc223afb0ea212df645a3ed4a28e5c6dad6ddfc0ec1fcb28ba9e`。supportを保つcolumnsの同値性が必要 |
| 条件付き参照 | [_trajectory_signal](../../../src/trotterlib/rpe_hadamard_failure_validation.py#L374)、[B trajectory_state](../rpe_fixed_request_measurement_b_20261103/run_fixed_request_b.py#L86) | F→各event→Rをq回。前者はscalarだけ返し、B companionはbranch照合用vectorも返す |
| Hadamard wrapper | [interrogation](../../../src/trotterlib/rpe_hadamard_interrogation.py#L305) | H→controlled V→H→ZでRe。sineはcontrolled V後にS†→Hで+Im。bit0=+1、bit1=−1 |
| 長q wrapper入口 | [benchmark builder](../../../src/trotterlib/rpe_hadamard_compiled_cost_benchmark.py#L163) | 通常wrapperはq=1,2,4限定。benchmark入口は明示maximum以下の正の2冪。metadata受理は長q回路の検証ではない |
| categorical law | [mean_adapter](../rpe_m17_rigorous_mean_enclosure_preparation_20261103/mean_adapter.py#L13) | probability64→NumPy CDF64→uniform53格子のexact bucket mass。metadata event_probabilityと区別 |
| 非列挙平均 | [event_average](../rpe_m17_rigorous_mean_enclosure_preparation_20261103/mean_adapter.py#L163)、[round_adapter](../rpe_18round_ideal_enclosure_preparation_20261104/round_adapter.py#L67) | `A=Σ p_l P_l`、`M=Σ_n w_n (-1)^(n/2)(cos θ_n I-i sin θ_n A) A^n`、`S=R M^r F`。既存ball powerを再利用 |

## Control・phaseの意味

ordinary controlは `C(U)=|0><0|⊗I+|1><1|⊗U`。
`(I⊗B) C(D) (I⊗B†)=C(BDB†)`なので、basis自体を制御する必要はない。
control=0でbasisが相殺すること、同じbasis inverseとwire配置を使うことが条件。
support completionは対象supportに同じ共役作用を与える必要がある。

`RZ(2sθ)=exp(-i sθ Z)`。SOURCEの`unsigned_rotation_angle`という名前の値は
`coefficient_sign*rotation_angle`であり、非負を意味しない。
正θの符号を逆にするとsineが変わり、全体位相の任意alignmentでは救済できない。
このRZ定義とcontrolのphase差は[Qiskit 1.3 RZ](https://quantum.cloud.ibm.com/docs/en/api/qiskit/1.3/qiskit.circuit.library.RZGate)、
[CRZ](https://quantum.cloud.ibm.com/docs/en/api/qiskit/1.3/qiskit.circuit.library.CRZGate)でも確認した。

| 位相の出所 | raw構築 | boundary_optimized | 二重計上を避ける規則 |
|---|---|---|---|
| H constant / tailから抽出したI | 各Fに1回 | `-qδ(c_const+c_I)`を1回 | built Fに含めた場合、参照でさらに掛けない |
| deterministic diagonal primitivesのscalar | 各block内 | 融合したblockの対応scalar | constant/Iとは別。消さない |
| `event.phase=(-1)^(n/2)` | occurrence内の位相和 | 同じoccurrence内 | signed Pに含まれないので1回必要 |
| product coefficient sign | unsigned Z/CZ実装なら負ごとπ | 同左 | signed Pを使う理論参照へ再びproduct-signを掛けない |
| rotation coefficient sign | θへ吸収 | 同左 | product-sign位相へ加えない |
| faithful identity event | productは±I、rotationはexp(-isθ) | 同左 | 本H4固定scopeはextract policy。人工対照だけで検査 |
| B_K normalization | probability lawの側 | 同左 | trajectory unitary／categorical平均に追加B除算しない |

controlledの場合、上表のscalar/global phaseはancilla-relative PhaseGateになる。
`C(e^{iφ}U)`と`C(U)`は一般に測定平均が異なる。
Qiskitのglobal phaseの扱いは[QuantumCircuit 1.3](https://eu-de.quantum.cloud.ibm.com/docs/en/api/qiskit/1.3/qiskit.circuit.QuantumCircuit)とも照合した。
π64の加算丸めと厳密±1との差は別問題で、既存`builder_phase_bound(q,r)`のscopeを維持する。
そのboundはK2/extracted identity用であり、人工K4のPASSをboundの拡張には使わない。

さらに`boundary_optimized`はscalarを `-δ*q*(c_const+c_I)` と1回で計算するが、
`half_operators`は保存された2個のstep phaseを別々に掛けたFをq乗する。
両者は実数算術では同じでもbinary64評価順が異なる。
`builder_phase_bound(q,r)`はevent/productのπ64加算を対象とし、このscalar再結合差を明示的には含めない。
block0のhalf融合とbasis取消についても、exact algebraの同値性とfloat gate実行の誤差を分ける。
未評価の差を0と置いたり、m17の既存scopeをboundary最適化全体の保証へ拡大したりしない。

## Endianと独立参照

Qiskitではqubit0が下位bitである（[公式bit ordering](https://quantum.cloud.ibm.com/docs/en/guides/bit-ordering)）。
system0..7、ancilla8ならarrayの前256振幅がcontrol0、後256がcontrol1。
gateのqargs `[ancilla,*support]` と、最終full matrixのblock順を混同しない。
OpenFermion sectorとの対応には既存`_qiskit_to_openfermion_sector_permutation`を使い、
`embed_sector`の`indices[permutation]`へ保存stateを配置する。
normalized rayとraw vdotの差を暗黙に消さない。

候補Bはcontrolled出力から理論signalを作っていないが、理論側F/Rにも既存uncontrolled builderを使う。
共通のモデル誤りを完全に排除する独立証明ではない。
今回の人工参照はoccupation-energyのexpでF/Rを構成し、production diagonal primitive生成を使わず照合した。
basisのruntime operationsとDF preparationは共通なので、この共通依存も残る。

既存Bの証拠範囲は[VALIDATION_GAPS](VALIDATION_GAPS.md)、人工実行は
[ARTIFICIAL_TEST_RESULTS](ARTIFICIAL_TEST_RESULTS.md)、次の未承認計画は
[SCIENTIFIC_VALIDATION_PLAN](SCIENTIFIC_VALIDATION_PLAN.md)を参照。
