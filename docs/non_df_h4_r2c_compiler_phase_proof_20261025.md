# H4 Pauli PR全law R2-C：compiler・位相の限定証明

2026-10-25。判定 **PROOF_INCOMPLETE**。現行opt1を一様保証するための対象を、有限のgate lowering、native 1Q run再合成、位相更新へ絞れた。control・全orderの演算順序・逆ゲート消去と、条件付きの有限回数上界を証明した。Euler角の特異点にも発散しない演算子上界を導いた。ただし、実際のbinary64再合成・parameter binding・remainderまで覆う数値誤差証明は閉じていない。全lawのρ上界・保存予算達成は未確定のまま。

基点は `codex/non-df-h4-central-r2-design-20261025` / `b297571d6949bb8b39152f08589c3a3b9c829ba6`。GitHubから同commitを取得して存在を確認した。今回のbranchは `codex/non-df-h4-r2c-compiler-phase-proof-20261025`。

固定条件：linear H4、隣接1.00 Å、STO-3G、Pauli-JW184非identity項、prefix27/tail157、S2/T=.8/q8/r4/K4、system8＋ancilla8。Qiskit **1.3.0**、basis `rz/sx/x/cx`、opt1、seed17。backend/coupling/initial_layout指定なし。この設定を変更していない。

## 1. 調査方法と根拠

旧[C・event law・D2・Arbへの対応](non_df_h4_central_r2_design_20261025.md)と、その[source_evidence.json](../artifacts/non_df_h4_central_r2_design/2026-10-25/source_evidence.json)を再利用した。科学C SOURCE `e80d7510880f8db993b777c627bb8e95b05a070d`、D2 SOURCE `74515d8eb4f073c40fc783681a3775dbca89e9b1`、6局所Arb結果を再評価していない。

Qiskit公式1.3.0タグのRust SOURCEを実際にGitHubから取得した。既存環境のPython preset/dispatcherも静的に読み、[公式SOURCE一覧・Git blob SHA](../artifacts/non_df_h4_r2c_compiler_phase/2026-10-25/upstream_sources.json)と[照合結果](../artifacts/non_df_h4_r2c_compiler_phase/2026-10-25/verification.json)を保存する。Rust全文7ファイル、Pythonのequivalence library、phase関連抜粋は同namespaceの`upstream/`、Apache-2.0ライセンスも併記する。

| 対象 | 実装上の根拠 | この調査で確定した範囲 |
|---|---|---|
| C→入力 | `h4_arb_pr0_preparation.py:build_plan/_selected_unitary_enclosure`、`pauli_signal_adapter.py:append_pauli_event/trajectory`、`h4_d2_pauli_control.py:_tokens/_rotation` | exact operand、chronology、sign/phaseの所有 |
| circuit phase | [circuit_data.rs](https://github.com/Qiskit/qiskit/blob/1.3.0/crates/circuit/src/circuit_data.rs#L848) | float setterは`rem_euclid(2π_f)` |
| DAG phase | [dag_circuit.rs](https://github.com/Qiskit/qiskit/blob/1.3.0/crates/circuit/src/dag_circuit.rs#L6310)、同L821/L7012 | float加算→setter→remainder。symbolic phaseはPython式処理 |
| restricted preset | `generate_preset_pass_manager.py:L318`、`level1.py`、`builtin_plugins.py:L109/L535`、`common.py:generate_translation_passmanager` | Targetなし、layout/routingなし、init消去→translation→opt1反復 |
| lowering | [basis_search.rs](https://github.com/Qiskit/qiskit/blob/1.3.0/crates/accelerate/src/basis/basis_translator/basis_search.rs)、[compose_transforms.rs](https://github.com/Qiskit/qiskit/blob/1.3.0/crates/accelerate/src/basis/basis_translator/compose_transforms.rs)、[translator](https://github.com/Qiskit/qiskit/blob/1.3.0/crates/accelerate/src/basis/basis_translator/mod.rs#L591) | gate数をcostとする探索、parameter bind/numericとphase追加 |
| actual 1Q backend | [euler_one_qubit_decomposer.rs](https://github.com/Qiskit/qiskit/blob/1.3.0/crates/accelerate/src/euler_one_qubit_decomposer.rs#L1076) | Complex64積、ZSXX再合成、gate数による採否 |
| 行列・消去 | [gate_matrix.rs](https://github.com/Qiskit/qiskit/blob/1.3.0/crates/circuit/src/gate_matrix.rs)、[inverse_cancellation.rs](https://github.com/Qiskit/qiskit/blob/1.3.0/crates/accelerate/src/inverse_cancellation.rs) | RZ行列のcomplex exp、SXのdyadic値、parameterなしの逆演算 |

タグSOURCEとインストール済みwheelのversionを対応させた静的調査であり、wheelを同SOURCEから再ビルドして一致証明したものではない。今回新compileは0。全入力の定理を特定runtimeへ受理する際には、このSOURCEモデルと実backendの対応も前提として残る。

## 2. 全orderで閉じる記号的対応

イベントeのC作用は、productを発生順に右から作用させる

\[
 E_e=(-1)^{k/2}\exp(-i\hat a_k P_{j_*})P_{j_k}\cdots P_{j_1},\qquad k\in\{0,2,4\}.
\]

Pauliの負符号はそのまま演算子に含める。k=0は空積、k=2のorder phaseは−1、k=4は＋1。productは交換しない。4個のproductでも同じ帰納法が成立し、未観測order4を除外する必要はない。

D2でbasis変換A（XはH、YはSdg→H）とparity CX列Fを使うと、`CRZ(θ)`の両側をAFで共役する作用は

\[
 |0\rangle\!\langle0|_a\otimes I+|1\rangle\!\langle1|_a\otimes e^{-i(\theta/2)P}.
\]

control=0ではbasis/parityとその逆が打ち消し合い、control=1ではPauli回転となる。`reversed(label)`とwire tupleを用いるので、Qiskit little-endian配置も保存する。controlled X/Y/Zを発生順に合成すればcontrolled productとなる。全gateが同じancilla8をcontrolに持つため、別event間でもこの積の恒等式を帰納的に適用できる。

負product・order phase・native scalarの**body global phase** gは最後のancilla `P(g)`で一度だけ移す。これはcontrol後のrelative phaseなので消去できない。wrapperはcosineがH→control→H、sineは最後のH前にSdgを置くため **+Im**。同一raw ψ、ν=ψ†ψを共通分母とし、ψや出力を再正規化しない。これらは全orderの記号的証明で、6本の局所結果からの外挿ではない。

## 3. η：C→central入力のoperandと位相

### 3.1 回転角の誤差を置く位置

retainedのC角は `α_j=exact(c_j) exact(δ)/2`。入力は `θ_j=fl(2*c_j*(δ/2))`、有効角`β_j=exact(θ_j)/2`。元の積の丸め順序を変えない。倍・半分がexactかつ非overflowの範囲なら、u=2^-53、λ=2^-1075として

\[
 |\beta_j-\alpha_j|\le \{u|2\alpha_j|+\lambda\}/2.
\]

これはbinary64 round-to-nearestの条件付き一般式。今回保存27 operandへ代入・数値評価していない。subnormal/overflow・丸めモードも無条件に除外していない。

eventの`a_k`はCも入力も同じ保存binary64値。D2の符号と2倍がexactな有限域なら、その辺の回転差は0。新たなatan近似誤差は加えない。Cでnative scalar operandは既に元binary64積を保持しているため、true Hamiltonian constantとその積の誤差を再度加算しない。

`||exp(-iαP)−exp(-iβP)||≤min(2,|α−β|)`とunitary積のtelescopingより、phase差d_gが得られたとき

\[
 \eta\le\min\{2,\;2q\sum_{j=1}^{L}|\beta_j-\alpha_j|+d_g\}. \tag{A}
\]

product自体はexactで、その負符号の近似はd_gだけが所有する。

### 3.2 位相setterの実際の経路とwinding

`QuantumCircuit.global_phase`はCircuitDataへ委譲し、Rust setterは`angle.rem_euclid(2.*PI)`。DAGも同じremainder処理で、float phase追加は普通のf64加算である。従って「加算だけのγ_n bound」では不足する。

p_f=2π_f。更新入力a_t、丸め後和y_t、setter出力g_tについて、整数winding n_tと残差を

\[
 A_t=y_t-(g_{t-1}+a_t),\quad R_t=g_t-(y_t-n_t p_f)
\]

と**定義**する。真の理想加算a_t*に対し、h_t=h_(t−1)+a_t*−n_t 2πと持ち上げれば厳密に

\[
 g_t-h_t=(g_{t-1}-h_{t-1})+(a_t-a_t^*)+A_t+R_t-n_t(p_f-2\pi). \tag{B}
\]

よって

\[
 d_g\le \min\{2,\sum_t(|a_t-a_t^*|+|A_t|+|R_t|+|n_t||p_f-2\pi|)\}. \tag{C}
\]

最後は位相差から`|e^{ig}−e^{ih}|`への上界。π_f、`np.angle(-1+0j)`の値、native scalarのexact sum、加算、remainder、windingをそれぞれ区別する。Bの符号を無視して単にmod 2πの同値だけ主張しない。

setter呼出回数は最大`Kqr+qr+q`（負product更新・各event phase更新・各step constant更新）。位相を加える項のうち非zeroなπ系は`N_-+N_2≤4qr`。0位相のsetterも回数上は除外しない。`|a_t|≤A`という域を別途証明すれば、g∈[0,p_f]と丸め上界から|n_t|にも有限上界を置ける。

**未解決**：実platformの`rem_euclid`/fmodの誤差定理とπ/angle endpoint定数の包含をSOURCEへ結ぶこと。R_tを0や1ulpと決めていない。tiny negativeに正のmodulusを足すと、丸めにより結果がp_fそのものになり得るため、常に`0≤g<p_f`とする証明は採用しない。人工テストはこの端点と恒等式Bを検査するが、Rust/libm全域の誤差証明ではない。

## 4. κ：今回到達するcompilerの限定経路

### 4.1 到達するstageとgate

`_skip_target`はbackendもcouplingもない今回の呼出でtrue。Targetは自動生成されず、layout/routing stageもない。従ってqubit配置の最適化を仮定する必要はない。

init opt1はparameter-free gateのInverseCancellation。入力族はH/S/Sdg/CX/CRZ/CY/CZ/Pと末尾measurementで、UnitaryGate・3q gate・制御フロー・calibration・symbolic入力を持たない。translationのUnitarySynthesisは対象unitaryがなく、HighLevelSynthesisは標準gateをequivalence libraryへ残す。BasisTranslatorがnative化する。

opt1の反復本体は`Optimize1qGatesDecomposition → InverseCancellation → GatesInBasis → 必要時translation → Depth/Size FixedPoint`。opt2/3のConsolidateBlocks等を今回の証明対象へ加えない。末尾measurementのwire/classical対応を変えるpassはこのrestricted経路にない。

### 4.2 Lowering：恒等式と実際のbindingを区別

equivalence libraryには次の短いnative到達経路が存在する（右側は実行順）。

| 入力 | exact π・exact算術での恒等式 | native gate数の候補上限 |
|---|---|---:|
| CRZ(θ) | RZ(θ/2)_t, CX, RZ(−θ/2)_t, CX | 4 |
| CY | Sdg_t, CX, S_t | 3 |
| CZ | H_t, CX, H_t | 7 |
| P(φ) | `e^{iφ/2} RZ(φ)` | 1 |
| S/Sdg | `e^{±iπ/4} RZ(±π/2)` | 1 |
| H | `e^{iπ/4} RZ(π/2) SX RZ(π/2)` | 3 |

H→S,SX,Sのlibrary phaseは−π/4であり、SをRZへ変えた後のnet phaseは＋π/4になる。これを取り違えない。CRZのθ/2は非underflow域でexactだが、symbolic bindingの実処理をexact算術とは呼ばない。

Rust `basis_search`は各native gate cost=1、rule内のcost和によるDijkstra探索。従って正常なcost探索・非overflow・固定標準libraryを前提とすれば、採用分解のgate数は上表の候補以下に抑えられる。ただし**同costの経路もある**。CRZ→RZZ→CX/RZと直接CRZ分解は双方4で、特定の一方だけが必ず選ばれるとは証明していない。

`compose_transforms`は置換DAGを作り、`assign_parameters_from_mapping`で式を合成する。`replace_node`は`ParameterExpression.bind/assign → numeric → Param`を経て角度・global phaseを加える。非parameter gateはtarget DAG phaseを直接追加し、parameter gateはsymbolic phaseをbindingする。選択経路・tieの全てについて、この実算術とphase処理を追う必要がある。**未解決**：到達する同cost rule閉包の確定、numeric/SymEngineの変換誤差、π定数とphaseの丸めの一様bound。表の記号的恒等式だけでその数値辺を閉じない。

### 4.3 Native後のgate数・run長・反復数

native prefixのgate数をN0とする（measurement除外）。1Q backendの`target=None`時の`old_error/new_error`は、名前と異なり**gate数**を比較する値で、operator誤差ではない。native runはoutside_basis=false、ZSXX出力もRZ/SX/Xだけ。採用条件はgate数の厳密な減少（ゼロgate化も含む）。InverseCancellationも削除のみなのでnative集合を保ち、追加translation条件は成立しない。

従って正常・有限な数値処理を前提に、採用1Q rewrite数≤N0、任意run長≤N0、optimization反復≤N0+1。各反復のrunはdisjointなので行列積の試行回数≤N0(N0+1)。これはSOURCEからの構造証明で、RZ費用が単調減少するとの仮定や保存256本の最大値を使わない。

全lawに共通する粗いnative数は、n system qubits、L retained数として

\[
 N_0\le (2qL+qr)(10n+2)+7nKqr+7+\mathbf1_{a=\sin}. \tag{D}
\]

一回のPauli rotationの各wire basis対≤8 gate、parity≤2(n−1) CX、CRZ≤4、product≤n個のcontrolled Pauli各≤7、aggregate Pと両H・sine Sdgを加えたもの。前項のcost探索条件付き。実compiled費用や母平均tailを評価した結果ではない。phase書込回数は置換数に、初回loweringの有限rule内phase数とDAG変換の回数を足して数える必要があり、N0だけで全phase処理を数えたことにはしない。

inverse cancellationにはparameter比較の`1e-10`があるが、今回のinverse listはparameter-free。nativeではX/CX、initではH/H、S/Sdg等のexact逆演算に限られ、このtoleranceを誤差allowanceへ持ち込まない。異なる非可換gateを無断に交換して消す処理でもない。

## 5. Euler数値再合成：閉じた補題と残る数値証明

Rustは各runの行列をComplex64で左乗算する。SX/Xはdyadic行列、RZは`exp(±iθ/2)`。計算されたMから、δ=arg det M、a=|M00|、b=|M10|、χ1=arg M11、χ2=arg M10を使い

\[
 \theta=2\operatorname{atan2}(b,a),\quad\phi=\chi_1+\chi_2-\delta,\quad\lambda=\chi_1-\chi_2,\quad\gamma=\delta/2
\]

を求める（L942）。u1x phaseは`γ−(θ+φ+λ)/2`（L979）。ZSXXの`circuit_psx_gen`（L276）と`fnz`（L544）へ渡す。既定`ANGLE_ZERO_EPSILON=1e-12`は**angle簡約閾値**で、libmや行列積の誤差証明ではない。

### 5.1 特異点で発散しないexact Euler補題

Uを2×2 unitary、`||M−U||₂≤ε<1`とする。上式を**exact実数演算**で評価して作った`V=e^{iγ}RZ(φ)RY(θ)RZ(λ)`は、zero entryのargに任意の位相を選んでも

\[
 \|V-U\|_2\le\min\{2,24\epsilon+6\epsilon^2\}. \tag{E}
\]

証明：r=√(a²+b²)とすれば|r−1|≤ε、r>0。d=detM/|detM|、d_U=detUとおく。`|detM−detU|≤2ε+ε²`より`D=|d−d_U|≤4ε+2ε²`。Vの第一列は `(d e^{-iχ1}a/r, M10/r)`。

`|a/r−|M11||≤|1−r|+||M00|−|M11||≤3ε`とunitary恒等式`U00=d_U conj(U11)`より、第一成分の差≤4ε+D、第二成分の差≤2ε。第一列の2-norm差は≤6ε+D。第二列はunitaryのdet位相で第一列から決まり、その差≤6ε+2D。行列のoperator norm≤Frobenius norm≤両列差の和≤12ε+3DでEを得る。M11=0でも第一項の大きさはa/rで同じ不等式が成立する。

これで「argの微分がzero entryで発散するから一様証明が不可能」という障害は解消する。Eは定理の一部であって、実際のRustがexact Eulerを計算したとする証明ではない。

### 5.2 Actual backendへ接続する未解決項

1. **行列形成ε(l)**：RZ complex exp・複素積/和の実算術・run長lから`||M−U_run||≤ε(l)<1`を証明する。丸めモード・subnormal・libm・実backendを含む。任意の`γ_l`やtrig ulpを仮定しない。
2. **角度計算の差**：detの丸め、complex abs/arg/atan2、φ/λ/phase加減算をEのexact式との差へ接続する。zero entryで個別角の相対誤差を要求せず、演算子の差を使う。
3. **簡約分岐とmodulo**：θ≈0/π/2/π、X shortcut、RZ省略、`mod_2pi`の`(angle+π_f).rem_euclid(2π_f)−π_f`と＋π付近を−πへsnapする分岐を覆う。例えば理想RYの角をθ0へ変更する辺は差≤|θ−θ0|/2だが、同時のu1x phase変更・λ/φ集約も所有する必要がある。1e-12を一律roundoff allowanceへ読み替えない。
4. **位相**：`fnz`はwrapped角をRZへ置き、その半分をsequence.global_phaseへ足す。`P(x)=e^{ix/2}RZ(x)`全体でperiodicityを保つ。RZ角だけをmod 2πにしてよいとは言わない。sequence phaseからDAGへの加算、DAG→circuit変換もB型のwinding誤差を含める。

一回のaccepted rewriteについて、これらからphase込みの`||V_new−U_run||≤χ(l)`を証明できれば、unitary積のtelescopingで

\[
 \kappa_a\le\min\{2,\;\kappa_{lower,a}+\sum_{accepted\ s}\chi(l_s)+\kappa_{DAG\ phase,a}\}. \tag{F}
\]

inverse cancellationの項はexactで0。χにDAG phase更新を含めるなら最後の項を別加算しない。棄却rewriteの計算誤差は回路へ入らず、ただし有限性/終了の前提は必要。Arb bridgeのgiven-program評価誤差をχやκへ流用しない。

## 6. 全law上界への接続・実用性

同一norm/axis配置で、A・Fが全allowed trajectoryに成立するときだけ

\[
 |m_{out,i,a}-m_{C,i,a}|\le\min(2,\eta+2\kappa_a),\qquad
 \rho_{PR,a}\le B_{saved}\min(2,\eta+2\kappa_a). \tag{G}
\]

body→信号のηは係数1、generic compiled wrapper→Z期待値のκは係数2。B_saved=`1449450801590761/1125899906842624`は一度だけ。C bias/normalization/numerical allowanceやC/L semantic項を追加誤差として二重計上しない。

保存予算はcosine `19792741170690161541/7631344236317346665332736`（約2.5936e-6）、sine `102330934580088554157/30525376945269386661330944`（約3.3523e-6）。値を変更せず、Gの数値評価も今回は行っていない。

| 項目 | 判定 | 残る条件 |
|---|---|---|
| order0/2/4のcontrol・順序・signed product | 記号的に閉じた | 浮動operand/phaseは別項 |
| retained丸めとphase windingの分解 | 恒等式・条件付き上界 | 保存operandのscalar包含、R_t/π/angleの根拠 |
| native集合・消去・有限rewrite/run長 | restricted経路で構造的に閉じた | 正常有限処理、lowering/costの条件 |
| exact Eulerの特異点 | Eで閉じた | 実際のbinary64手順への接続 |
| loweringの数値意味・actual χ(l) | **未証明** | 上記の特定箇所だけが対象 |
| 全law ρ・予算達成 | **UNDETERMINED / null** | η/κの一様保証と承認済みscalar評価 |

**PROOF_PATH_SUPPORTEDはまだ付けない**。実数恒等式からactual opt1への未証明辺が残るためである。**PROOF_ROUTE_BLOCKEDとも判定しない**。SOURCEとEにより次の対象を有限のscalar/2×2 kernelへ具体化でき、現時点で不可能性や過大な保証上界を証明したわけではない。6局所値が小さいことも、Gの予算達成を予測・保証する根拠には使わない。

## 7. 人工検証・再利用・保全

新規[人工補題テスト](../tests/test_h4_r2c_proof_lemmas.py)3件＋既存`test_h4_r2_static_design.py`7件、**10 PASS**。既存CのPauli/Arb primitivesとprivate importを再利用した。2qubitのH native恒等式でglobal phase符号、CRZのcontrol/wire/角度、synthetic winding境界、exact Eulerのθ≈0/π近傍、既存2〜3qubit全order非可換law/符号/+Im/norm/B一回を検査した。人工Euler検査は2×2演算子に対する補題検査で、H4 compilerの全域保証にはならない。

初回新人工テストのFLINT Fraction引数変換1件を修正し、全10件を再実行してPASS。これは科学runの失敗・retryではない。主担当自身の検査であり、別担当の独立レビューとは呼ばない。

実H4 state作用/Arb/scalar新上界評価/compile/sampling/RNG replay/Hamiltonian/state生成/N_G/新科学grantはすべて0。SOURCE/raw/結果/grant/ledger/marker/STOPは変更しない。基点との差分と既存引用SHAで確認する。[verification.json](../artifacts/non_df_h4_r2c_compiler_phase/2026-10-25/verification.json)はこの作業の技術検査だけを記録し、元H4 native終了UNKNOWNを代替しない。旧N/G null、C/L・正式受理false、PR母平均費用tailの不足も保持する。

運用指示はユーザー提示pathの最新版`research_project_instructions_unified_2026-10-25.md`を取得し§8.4/10/10.3を確認した。今回§10.3の版不一致はない。初期探索時の未発見を未解決扱いにはしない。共有環境/他ユーザーへの変更、恣意的資源制限、承認証明用の仕組みの新設はない。

## 8. 最小の次作業と停止境界

1. **Codex：限定した数値証明の継続**。到達lowering/tieの閉包とphase式を静的に固定し、actual Complex64行列形成・Euler/snap/remainderの一様誤差補題を証明する。人工scalar/2×2検査までならH4状態作用は不要。今回のSTOP後に進めるには、この継続scopeをユーザーが選ぶ必要がある。
2. **ユーザー：証明継続か実装family再設計かを判断**。証明が大幅に拡大するなら、解析可能なtemplate loweringへ変える研究判断も選択肢。ただし現時点では代替案の提示だけで、compilerを変更しない。新familyの価格・局所証拠は旧opt1と別に扱う。
3. **証明後に限りユーザーが別承認／Codexがscalar包含**。保存retained丸めと全域phase/κ式の評価範囲・precision・call・STOPを固定して承認する。今回そのH4 scalar評価も実行しない。既存R2の次契約は無効のまま保全し、新科学grantを発行しない。

必要最小の次段階は上記scalar/2×2証明であり、新H4状態作用・追加order4 trajectory・標本拡大を既定の工程にはしない。Law-weighted化はuniformの保守性が実際に支配すると証明できた場合、実装family変更は現行証明が現実的でないと判断した場合だけ再設計する。どちらも今回未実施。

commit/push・GitHub remote SHA照合の結果は[publication_status.json](../artifacts/non_df_h4_r2c_compiler_phase/2026-10-25/publication_status.json)に記録する。保存・公開確認後 **mandatory STOP**。scalar科学評価、追加科学計算、compiler変更、正式certificate受理へ自動移行しない。

## 公開確認結果

証明commit `0c7a2eac3c4e80f4d7d4eb9743d81e3d7ac20f2b` をlocal保存した。pushはHTTPS Username取得不能でexit128。GitHub接続でも新branchは検索結果0、同commitはAPI422 `No commit found`。remote SHAはnull/未確認で、公開済みとは扱わない。[失敗記録](../artifacts/non_df_h4_r2c_compiler_phase/2026-10-25/push_attempt01.txt)を保全し、公開状態の追記もlocal commitする。認証失敗が判明している同pushを反復しない。

認証のある環境での再実行：

```bash
git -C /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/non-df-h4-r2c-compiler-phase-proof-20261025 push -u origin codex/non-df-h4-r2c-compiler-phase-proof-20261025
```

成果をlocal保全してmandatory STOP。科学処理へは進まない。
