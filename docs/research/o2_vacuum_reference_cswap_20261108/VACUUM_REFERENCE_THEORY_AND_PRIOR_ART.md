# 真空参照CSWAP：pathwise信号と既知法への対応

2026-11-08。対象は理想演算と、今回作成した人工1–3 system qubitsのみ。実H4の新評価ではない。技術分類は KNOWN_SE_QPE_APPLICATION と、限定した PR_SPECIFIC_APPLICATION_POTENTIAL。独立した新機構は確認していない。

## 先行研究と比較順序

既存文献調査 e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8 の O5、O2統合 d28164b5d51e0ca5a0ade74ded8ff176b443dab8、非同値性選別1bf88b2、共通basis静的調査ffcdd1fを再利用した。M/D/Eや39 stageの再調査・再実装は行わない。

最重要対照は [Rowe et al., SE-QPE v2](https://arxiv.org/pdf/2604.14921v2)。§II Eq.(2)は二度のCSWAPと二registerの非制御演算、Eq.(3)–(6)は共通固有基底を用いる複数blockの保証、Eq.(8)–(9)は真空位相を扱う。同節のfanoutと§IIIのDF資源比較も確認した。非厳密固有stateを許容する条件は維持する。論文の分子・T資源改善率を本PRへ移植しない。

関連する [Tazi et al., Shallow Quantum Scalar Products with Phase Information](https://arxiv.org/abs/2411.19072) のabstractはzero/one-control scalar-product測定の既知性を示す。全方式の最適性を主張する資料にはしない。Compute–uncomputeには制御付きstate準備・逆演算の別アクセスが必要で、今回それを無料oracleとして追加しない。

実装前に、(i)既存diagonal-only controlled PR、(ii)同じsplit・位相・fanoutを持つ既知SE、(iii)候補を比較した。候補は(ii)と同一だが、(i)の中央対角回転の制御費を除く経路と二laneの並列化が残るため、許可された人工実装・compileへ進んだ。

## 単回実験の導出

Qiskitの回路順にRを先、Lを後とし、U=LRと置く。真空vが両因子の固有stateで

\[
L|v\rangle=e^{i\ell}|v\rangle,\qquad R|v\rangle=e^{ir}|v\rangle
\]

とする。角度はradian。信号ancillaをa、targetをT、真空referenceをVとする。CSWAP(a=1)、TへのL†とVへのR、CSWAPの順で

\[
\frac{e^{ir}|0\rangle L^\dagger|\psi\rangle+
e^{-i\ell}|1\rangle R|\psi\rangle}{\sqrt2}\otimes|v\rangle
\]

となる。最後に a へ P(ell+r)=diag(1,exp(i(ell+r))) を作用させると

\[
\frac{e^{ir}}{\sqrt2}
\bigl(|0\rangle L^\dagger|\psi\rangle+|1\rangle R|\psi\rangle\bigr)
\otimes|v\rangle .
\]

したがってZ_aの読出し前のcoherenceから

\[
\langle X_a\rangle+i\langle Y_a\rangle
=\langle L^\dagger\psi|R\psi\rangle
=\langle\psi|LR|\psi\rangle.
\]

L/Rの可換性も、psiがUの固有stateであることも不要。元Hの基底stateという研究上の仮定を変更せず適用できる。真空referenceは最後に分離してvへ戻るが、targetは元psiへ戻るとは限らない。全gateをcontrolledにした回路そのものとのunitary同一性は主張しない。

XはH→Z読出し、YはSdg→H→Z読出し、bit0→+1、bit1→−1。Z_omega=<psi|U_omega|psi>に対して

\[
p_{\omega,X}(0)=(1+\Re Z_\omega)/2,\quad
p_{\omega,Y}(0)=(1+\Im Z_\omega)/2.
\]

±1のbinary分布はこの平均で一意に決まるため、各pathの平均だけでなく各軸の分布が一致する。元のactual有限Qを変更せず平均すると、同じ有限複素target sum Q Z_omegaを得る。原法にpath重みがあればそれも保持する。本gadgetから新たな重み・postselection・成功数正規化を導入しない。

理想binary測定ではM2=1、座標variance=1−mu²が原法と同じ。追加のvariance削減や必要shot削減は生じない。利益の候補は一shotの量子回路費用。

## RPEとcoherent QPEを分ける

各round・各shotで新しい同じtarget stateと真空を準備し、同じlaw・独立性で取得するRPEなら、上記の分布保存を各実験へ適用できる。元のPF/RTE近似、round別bias、phase unwrap、energy精度保証は引き続き別条件。新gadgetの数値誤差も別途必要で、今回全roundの計算やRPE再認証はしない。

真空部分を除いたgadgetは、共通global phaseを除き

\[
S_k=\operatorname{diag}(L_k^\dagger,R_k)
=(I\otimes L_k^\dagger)\,C(U_k).
\]

共有targetに複数のS_kを続けると、余分なL_k†が次のblockの入力を変える。例えば、すべての先行L_k†が後続U_jと可換なら、共通target演算を全controlled blockの後ろへ移せる。この十分条件の下でphase-registerのreduced densityは保存される。全splitが同じ固有基底を共有する条件も十分。必要条件とは主張しない。

単回の非可換性許容から、複数blockの条件成立は導かれない。人工反例は、2 system qubitsで U=exp(i .23)RZ_0(.71)、L1=XXPlusYY(pi/2,0)、R1=L1†U、L2=I、R2=U²、psi=|01>。Uの厳密固有stateかつU/U²でも、共有targetのcoherent QPEでは逆Fourier読出し分布が異なる。これは各blockをfresh psiで測る分布保存と矛盾しない。数値はTEST_RESULTSの反例欄に保存。

## PR SOURCEの静的接続

既存SOURCEのファイル・関数を実際に確認し、保存分子payloadは読まず、次の対応を整理した。

| SOURCE | 接続と保持すべき事項 |
|---|---|
| src/trotterlib/rte.py: RTEEvent / unsigned_rotation_angle | rotation符号はcoefficient_signと結合。application順序を維持 |
| df_rte_qiskit.py: DF RTE builder、_event_phase_angle、_append_application系 | Gaussian逆→Z/ZZ/rotation→Gaussian順。product負符号のpiとevent phaseを別途保持 |
| df_partial_s2.py / df_partial_s2_repeated.py: buildとscalar phase処理 | deterministic半step・RTE・反復順を変えず、PhaseGateとglobal phaseを正しく対応 |
| df_rte_tail.py: extraction_component_circuit_specs / describe_basis_change_operations | support、選択basis、係数符号のSOURCE。新たな抽出や集計はしない |
| research_direction_sequence_policy.py: make_run_threshold_basis_plan / source_basis_run_lengths | selected policy・run境界を保持。新policy選択なし |
| research_direction_structure_pilot.py: _build_from_operations / build_conjugated_pauli_sequence | 人工対照にdiagonal_only controlと既存shared-basis fusionを再利用 |
| rpe_hadamard_interrogation.py | X/Yとbit規約を再利用 |
| src/trottertracks/algorithm_codesign/a1_engine_connection.py | 既存engineのphase/ordering契約を参照するだけ。旧認証を新gadgetへ移植しない |

真空はJWの全0。RZ(theta)、RZZ(theta)はともに真空位相−theta/2、XXPlusYYは|00>を位相0で保つ。number-preserving Gaussian basisのRZ/XXPlusYY列も真空固有stateを保つ。Z、P、S/Sdg、T/Tdg、SWAPは真空位相0。circuit.global_phaseは加算する。inverseは順序を逆にし位相を反転、負時間も実際の角度から取得する。単なるparticle-conservingという名称や元Hの真空energyでは代用しない。

元Hadamardのancilla PhaseGateは相対位相で、非制御laneでのglobal phaseへ移す際に消してはならない。scalar/identity/eventの寄与を含む実際のL/R gate列からell/rを取得し、最後のPで戻す。selected fullの演算子を変えない。

splitは既存application/blockの境界に限定する。人工例は先半R・後半Lで固定し、split探索・角度分割最適化を行っていない。実PRでは境界fusionをまたぐ追加損失もfull比較へ含める。

## 実装・数値保証の範囲

新adapter vacuum_reference_cswap.py は1–3 system qubitsの人工入力だけを許し、既存gate checkerとbit mappingを利用する。gatewise真空保存を証明できないgate、未知oracle、誤ったmappingを拒否する。これは十分条件検査で、途中で真空を破り最後に回復する回路を不可能と判定する理論ではない。状態・Born・RNG・科学loader・compileはadapterに含めない。

実装の角度ledgerはbinary64値を正確な有理数として加算する。ただしQiskitのglobal phaseのmod 2pi表現、inverse生成、Pへのfloat変換、gate生成/作用、readoutは新しい数値保証を持たない。人工比較の許容差8e−12、compile監査2e−10は小規模binary64検算用であり、全support一様上界ではない。位相fitting・許容差緩和はしなかった。

将来は元state誤差に加え、真空初期化・phase ledger/補正・L†/R生成・CSWAP/fanout・compile/routing・Born/readout/gridの誤差を整理する。例えばreference初期化のtrace distance D_refなら、理想回路後の座標差は2 D_ref以下。Pの角度差Deltaによる単回signal差は|Delta|以下。他の未知項は0にしない。旧exact Fraction engineの認証値をそのまま新回路の認証値とはしない。

postselectionなしの理想実装は全attemptを1回測定しretry0。hardware error detectionを採る場合のacceptance・bias・追加測定費はUNKNOWNであり、現在の推定器へ自動追加しない。
