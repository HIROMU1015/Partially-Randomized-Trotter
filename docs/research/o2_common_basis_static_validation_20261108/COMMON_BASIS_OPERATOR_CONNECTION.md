# 共通basisへの接続：理想演算・保存gate・旧群判定

2026-11-08 JST／SERVER1 GPUA100-gleap-01。静的SOURCE読解と紙上の式だけ。実H4 metadata、保存matrix/state、LAW/catalogの新評価・群再生成・人工test・回路構築は行わない。

## 固定資料と実際に照合したSOURCE

正本SOURCEは 39d839063c53d949342af4a0541b4865bdd054ed。本作業のbase 1bf88b251fd09caa0a227972ee0f1b56c5888e6aでも次の5ファイルが同じGit blobであることを照合した。関数は実行していない。

| SOURCE | 固定blob | 今回の確認 |
|---|---|---|
| [m_label_average.py][M] | ce16edb0fe154f8e504a500d80ba7c249a349e78 | SlotLabel、_row_key、qualify_group、partition、readout/instrument |
| [m_n0_header_binding.py][BIND] | 5e779c2c93760a181f96881515e8009667b21801 | _basis、_half_blocks、bind_n0_headers、summarize_binding |
| [research_direction_sequence_policy.py][POLICY] | 9de1738bb799474dfbabb65da2601e2d6edffac8 | source basis再構成照合、support登録・ID、run policy |
| [research_direction_structure_pilot.py][STRUCT] | 6e1e98e4667c7170e53032c44da4ebb34af92f1f | column completion、inverse/forward、diagonal_only |
| [df_trotter/ops.py][OPS] | f950b0f7950042229de60ecd728f17638bbd265b | Gaussian変換、転置、JW bit reverse |

関連RTE/Hadamard、exact engine、元Mの人工記録、[旧M metadata結果][MRESULT]、[O2統合][SYNTH]、[最新baseline規則][CHECK]を必要箇所だけ再利用した。39 stageの再監査や旧12/10/3人工testの再実行はしない。

## 1. Literal群判定の意味

_row_keyは context/slot/policy/law/system_order、selected_basis_id、A/B/Vのliteral key、kind、rotation φ、scalar/event phaseを群署名へ入れる。support/signは妥当性を検査するが署名へ入れず、共通frame内で変わる対角関数として扱う。actual_massはactual lawとの一致を検査し、群質量とconditional qに使う。

original_basis_idはSlotLabelのreference-only fieldで署名に使わない。binderの別recordにはoriginal_basis_hashも残るが、それ自体は演算子証明ではない。register_support_restricted_basesのkeyは(source_basis_id,source_basis_hash,support)、selected IDは元ID＋support＋wp06b-v1、hashは実operation列を表す。

したがって同じ元basisでも異なるsupportは異なるselected ID/operation列となり得る。現判定は「元basisへ各labelを演算子的に戻せるか」を判定せず、渡されたselected A/B/Vがliteralに同じという**十分条件**を使う。共通frameへ書き換えられる一般条件より強い。

旧結果は保存linear H4/1.00Å/STO-3G/legacy DF12/8system/L_D3/m0q1/δ=.02/r1K2/atol0、N1 actual Q/selected full、n=0唯一rotation slotにおける正232 label・232 singleton・非自明群0である。この正しい限定結果を変更しない。今回新しい群の存在・件数を保存入力から取得していない。

**異なるsupportのlabel同士が同じoperatorだという主張ではない。** 各labelのselected表現を、そのlabel自身の対角作用のまま元共通basisへ置き換えられる条件を調べる。

## 2. Column保存が理想的に保存する演算子

一粒子unitaryをO、Fock空間のnumber保存GaussianをG(O)=Γ(O)とし、

\[
G(O)a_p^\dagger G(O)^\dagger=\sum_a O_{ap}a_a^\dagger
\]

を採用する。従って G(O)n_pG(O)†=Σ_ab O_ap O_bp* a_a†a_b。**正確なunitary** OとO_Sがp∈Sの列を同じに保つなら、各n_p、Z_p=I−2n_p、その積Z_p Z_q、supportだけの任意の対角関数の共役作用が一致する。列ごとのphase差だけでもrank-one projectorは同じだが、SOURCEが意図する条件は列のcopyである。

別の書き方ではR=O†O_Sがsupport上の標準基底を固定し、Γ(R)がそのnumber作用と可換なので、G(O_S)D_SG(O_S)†=G(O)D_SG(O)†。complement内を回すRが非自明でも成立する。これは回路literal一致を要求しない紙上の十分条件で、実H4の等価群を新発見したという意味ではない。

support_restricted_unitary_completionは入力unitarityをmax-entryでatol内と検査し、指定1/2列をそのままcopyする。残りは標準基底から2回再直交化し、completion残差を10 atol内で検査する。これは浮動小数処理の意図・実装上の条件であり、出力が数学的に正確なunitaryである証明ではない。atol=1e-10をsignal biasへ移植しない。

## 3. Gaussian/JWとSOURCEの作用方向

SOURCEのU_to_qiskit_ops_jwは BogoliubovTransform(O.T)を呼ぶ。既存環境のqiskit_nature/second_q/circuit/library/bogoliubov_transform.pyを読取り確認したところ、入力Wの行jは b_j†=Σ_k W_jk a_k†、かつ回路G a_j† G†=b_j†を定義する。O.Tを渡すことで上のcolumn規約になる。[公式SOURCE/規約](https://qiskit-community.github.io/qiskit-nature/_modules/qiskit_nature/second_q/circuit/library/bogoliubov_transform.html)。確認したinstalled SOURCEはqiskit-nature0.7.2、同ファイルSHA256 b3956e8469dfe22ed57a5bfe6ce341f34dfda6a22bef7082855371c157b181f2。上の公開公式ページは0.8系で、実依存の規約はinstalled本文でも確認した。全依存環境を新たに認証したものではない。dense fallbackも a† log(O) a とJW/OpenFermion→Qiskit bit reverseを使う。実保存依存を再実行して規約を検証したわけではない。

同じJW wire/mode順でn_p=(I−Z_p)/2。Qiskit system bitは z=Σ_p bit_p 2^p、signed Pauliは σ_l(-1)^(Σ_{p∈S_l}bit_p)。一粒子の列index、mode、Z wireを勝手に交換しない。

_build_from_operationsとdf_rte_qiskitは、時間順に G_S†→D_l→G_S を適用するため、演算子は **G_S D_l G_S†**。diagonal_onlyではbasisは非制御、中央Dだけ制御し、inactive branchはG_S G_S†=I。scalarはcontrolledではancilla-relative PhaseGateとして残す。

共通frameを V_f=G(O_f)† と置くと理想full pathは

\[
U_l=e^{i\chi_l}A\,G(O_f)D_lG(O_f)^\dagger B
    =e^{i\chi_l}A V_f^\dagger D_lV_fB .
\]

**このV_fは一粒子matrix O_f自身ではない。** binderもselected operationsのadjointをMのVへ渡す。_half_blocksはBへD blocksをforward順、Aへreverse順で並べ、各blockのhalf-time diagonal/global phaseを保持する。reverse順はinverseではなく、一般にA=B†とは仮定できない。

RTEEvent.unsigned_rotation_angleという名前でも、値はrotation.coefficient_sign×rotation_angle。中央RZ/RZZのgate角はその2倍なので D_l=exp(−iφ σ_l P_{S_l})。n=0のφはatan(τ/1)、event phaseは+1。signを二重適用しない。constant/identity phase、block内scalar、event phaseはχ_l又はA/B内に元SOURCEどおり残す。basis全体のglobal phaseは共役で相殺するが、full evolutionの相対phaseは捨てない。

## 4. 共通frameの複素平均・readout

一つの共通contextと互いに排他的な候補群fを仮定し、w_f=Σ_{l∈f}Q(l,context)、q_l=Q(l|f,context)、Σq_l=1を**元actual Q**から定義する。重みを係数の理想値で置換しない。

\[
d_f(z)=\sum_{l\in f}q_l e^{i\chi_l}D_l(z),\quad
x_z=\langle z|V_f A^\dagger|\psi\rangle,\quad
y_z=\langle z|V_f B|\psi\rangle .
\]

旧Mと同じbranch準備 (|0⟩V_f A†ψ+|1⟩V_f Bψ)/√2 に対し、

\[
Z_f=\sum_z x_z^*d_f(z)y_z,\qquad
p_f(z)=(|x_z|^2+|y_z|^2)/2 .
\]

d=a+ibなら、system readout zの後でancillaの O_X(z)=aX−bY、O_Y(z)=bX+aYを測ればRe/Im Z_fを保存する。m(z)=|d|、α_X=−arg d、α_Y=π/2−arg dの軸へ回し、bit値±1にmを掛ける。RZ(−α)→H、α=π/2はSdg→Hという既存規約と一致する。m=0は結果0が既知だが、**失敗・中断で未観測のデータを0へ補完する規則ではない**。

既存M rotation readoutは共通φ/χ、r(z)=Σq_l σ_l P_{S_l}(z)に対する d=e^{iχ}(cosφ−i sinφ r)の特殊形。productはrの符号をdecoderへ残す等価な表現を使う。一般のlabel別phase/angleを含むdへ拡張するinstrumentを今回実装・資格認定していない。

A≠B†でもbranch式は成立する。ただしancilla-free測定を自動適用しない。A=B†なら両branchは同じV_f Bψとなり、system測定だけでRe/Im d(z)を返す強い既知groupingが可能。保存H4でその条件を新たに判定していない。

全targetは Σ_f w_f Z_f＋未変更の群外寄与。identity labelや他order/slotを都合よく除外・再正規化しない。zero-grid labelは寄与0でも元記録へ残す。今回q_l、w_f、d(z)、p_f(z)の実数値や新群を一つも算出していない。

## 5. 理想同値と有限targetの境界

| 層 | 言えること | 言えないこと |
|---|---|---|
| 正確なO/O_S・Γ | support列保存なら共役Z/ZZ・同角回転が一致 | actual saved gateの一致 |
| 保存角を厳密値とした理想gate列 | 各basis列はunitary、中央conjugated Pはinvolution | 別々に合成したselected/full列の演算子が自動で一致 |
| binary64の局所gate matrix | 保存値とhashで対象を固定できる | matrixが数学的に厳密unitary、逆列との積が厳密I |
| exact Fraction engine | 与えた丸めmatrix上の加算/積/Born比をexactに処理 | Gaussian合成誤差やmodel差の修復、新readoutの保証 |

元finite targetを、actual Qと保存角の理想selected unitaryで定義したF_selとする。旧engineの丸めmatrixによるnormalized Born計算は、それを近似する認証経路であり、丸めた非unitary matrix積を無条件に別のunitary targetと呼ばない。共通full gate列を使うF_comとの一致又は差の上界が必要である。

正確なHermitian involution P1/P2に同じθを使う場合

\[
e^{-i\theta P_1}-e^{-i\theta P_2}
=-i\sin\theta(P_1-P_2),\quad
\|\Delta U\|=|\sin\theta|\|P_1-P_2\|.
\]

θは中央RZ/RZZのgate角の半分。非unitary丸めmatrixやinvolution未確認の近似Qにこの式をそのまま使わない。angle/phase差があれば別項（例えば2|sin(Δθ/2)|、2|sin(Δχ/2)|）を加える。

紙上の保守的な取得経路として、正確なunitary近傍O_S/O_fのsupport列差δ_pと、各Γから理想保存basis列へのoperator誤差η_S/η_fが証明できれば、CARの||a(u)||=||u||から

\[
\epsilon_P\le4\sum_{p\in S}\delta_p+2(\eta_S+\eta_f)
\]

を使える。number差を二つのa†a項へ分解しZ=I−2n、ZZの差をtelescopingする保守的な式で、tightさは主張しない。**そのような正確unitary近傍・δ・ηを今回取得していない。** max-entry toleranceや同じoriginal IDをこれらの値へ代入しない。

一slotだけの置換なら b_frame≤Σ_f w_f Σ_l q_l ||U_l^sel−U_l^com||、他の同一unitary A/Bによる左右作用はnormを増やさない。一般の複数置換はpathごとのtelescopingと元Qで会計するが、今回別slotへ拡張しない。新bias allowanceは未採択。

さらに新branch gate、system/ancilla joint Born、状態norm、q/dの古典丸め、z依存軸/weight、bit-grid、decoder/output誤差が必要。dの誤差は理想observableで||δa X−δb Y||=|δd|と整理できるが、これはactual gate/angle生成誤差の認証ではない。旧selected-engine certificateが新列・新readoutまで覆うとはしない。実H4へのfinite-target保証は**UNKNOWN**である。

[M]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/39d839063c53d949342af4a0541b4865bdd054ed/src/trottertracks/algorithm_codesign/m_label_average.py
[BIND]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/39d839063c53d949342af4a0541b4865bdd054ed/src/trottertracks/algorithm_codesign/m_n0_header_binding.py
[POLICY]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/39d839063c53d949342af4a0541b4865bdd054ed/src/trotterlib/research_direction_sequence_policy.py
[STRUCT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/39d839063c53d949342af4a0541b4865bdd054ed/src/trotterlib/research_direction_structure_pilot.py
[OPS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/39d839063c53d949342af4a0541b4865bdd054ed/src/trotterlib/df_trotter/ops.py
[MRESULT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e98589ae129d3b81ad428c947d2f4c88623caa9b/docs/research/o2_m_n0_metadata_result_20261108/REPORT.md
[SYNTH]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d28164b5d51e0ca5a0ade74ded8ff176b443dab8/docs/research/o2_research_results_integration_20261108/O2_RESEARCH_SUMMARY.md
[CHECK]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1bf88b251fd09caa0a227972ee0f1b56c5888e6a/docs/research/o2_distinct_mechanism_screening_20261108/EQUIVALENCE_AND_BASELINE_CHECK.md
