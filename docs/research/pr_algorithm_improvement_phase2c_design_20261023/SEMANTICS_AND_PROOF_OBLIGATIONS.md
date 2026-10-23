# 意味論と証明・検証の未解決条件

全項目は設計上の要求であり、今回新しい非自明な証明・数値検証を実行していない。SOURCEにある一般的なRTEの性質と、新しいmixed production adapterの認証は別である。

## 1. 同一Hamiltonianと同一task

保存入力のH_ref、物理sector、register順、状態、scalar、energy reference、DF rank／切捨て方針を固定する。元modelとの `ΔH_input` は保存し、辞書変換の `ΔH_dictionary`、PF splitting誤差、有限Taylor打切り、sampling丸め、回路合成、統計誤差を分ける。

本案は同じfragmentを再表記し、D/R帰属を変更しない。P展開で現れるIも既存identity policyに従う。残差内scalarを外へ移すと真のHが同じでも有限Taylor構成は変わり得るので、新規の移動は禁止する。`λ=0`は認証してからdeterministic-onlyへ送る。数値的に小さいことだけで0にしない。

共通左frameを替えるだけでは `(VB_j)†(VB_i)=B_j†B_i`。この操作だけで相対Gaussian移動が安くなるとはしない。今回のP展開は物理registerを維持した辞書変更であり、状態を都合よく回した比較ではない。

S側のnumber-conserving frame atomと、物理Pauli atomの個別symmetryは同じとは限らない。PのsumがRを表すことは、各Pが粒子数sectorを保存することを意味しない。本案ではfull registerで操作し、初期状態とtargetのsectorを固定する。中間postselectionを行わない。sectorだけに射影したPauliをunitaryと扱う実装は不適格であり、必要なら当該P候補を除外する。物理register全体を使うことによるsimulation／encoding費用も別途数える。

## 2. 順序、符号、制御

signed involutionを `P_i=sgn(a_i)Q_i` と置く。時間順に記録されたproductsを `i_1,...,i_n`、最後のchildをjとすると、eventのsystem operatorは

\[
U_\omega=(-1)^{n/2}\exp(-i\theta_n P_j)
P_{i_n}\cdots P_{i_1}.
\]

行列積は時間順と逆になる。basisを揃えるためにこの列をsortしない。identity、負product、paired位相、scalarはsystemだけのglobal phaseとして捨てず、Hadamard ancillaのrelative phaseに保つ。例えばcontrolled `-I` はancilla位相であり、空回路とは異なる。

frame-conjugated atom `Q=B Z_S B†` は時間順 `B†, central, B`。Pauli branchでも局所basis／parityを非制御でcompute/uncomputeし、中央と位相を制御する。inverseはactual adjointで作る。RZ角度の符号だけを変えてgate列の順序を維持する操作は、一般のadjointではない。

SOURCEの `event.unsigned_rotation_angle` はcoefficient signを織り込むaccessorである。名称から符号を無視した値だと解釈しない。新adapterがsignを二重適用しないよう、一度だけ物理回転へ反映する仕様を固定する。

## 3. 保証の階層

| 層 | 接続したい対象 | 現在の状態 |
|---|---|---|
| 理想係数代数 | 全候補の `Σ a_i Q_i = R` | exact変換の設計要求。新inputでの検証なし |
| 理想finite-RTE | `B E[U]=T_(K+1)(-ihR)`、全体は固定Dを挟む構成 | 一般既存RTEを適用する条件。mixed接続は未検証 |
| 実controlled回路 | `|0><0|⊗I + |1><1|⊗U_ω` | phaseとorderを含む新routerの認証が必要 |
| 有限bit | 係数、確率、atan／角度、normalizer、seed law | 丸め・zero・誤差集約の仕様が必要 |
| 合成・信号 | 近似gate、prep/readout、複素mean、target signal | 別予算。RPE全round・energy保証は今回の範囲外 |
| 統計 | fresh IID bounded Hadamard outcomesの複素mean | 既存Hoeffding型policyの適用条件を明示。古典cost trajectory再利用と量子shotを混同しない |

first moment一致はquantum channel一致ではない。有限Taylor polynomialもexact exponentialではない。λ依存の粗いRTE boundが変わることと、理想の固定R polynomialが変わることを区別する。

## 4. PROOF_OBLIGATION一覧

| ID | 満たすべき条件 | 最小の確認方法・失敗時 |
|---|---|---|
| PO-01 | 保存factorからS/P両辞書への係数代数、fermion/JW符号、scalar、sectorが同じH_refを表す | 静的導出と将来の小系全operator／sector検査。任意dense traceを大系の構成法にしない。不合格Pを除外 |
| PO-02 | typed metadata、basis key/hash、support、sign、identity、由来collectが整合 | 新adapterの意味論検査。Z/ZZ制限へ長いPauliを偽装しない |
| PO-03 | mixed involutionへ既存paired lawを接続し、order/phaseを含む有限corrected meanを保つ | 既存一般条件への形式的接続と小系finite event検査。新しい非自明な保証は別作業 |
| PO-04 | actual controlled unitaryのorder、adjoint、ancilla相対位相が正しい | 正／負Pauli・identity・異basis・n=0/2、X/Yのoperator probe。system unitaryだけで合格にしない |
| PO-05 | 価格状態遷移が**定義したliteral plan**の期待countと一致し、必要workが記録される | 小poolの有限列挙対照。actual compiled Cとの一致や上界は要求していない |
| PO-06 | coefficient/probability/angle/Bの有限bit誤差が明示予算内 | zero不確定の扱い、丸めlaw、精度とgate誤差の合成。未評価をΔH=0としない |
| PO-07 | 共通signal allowanceとIID bounded条件の下でN式が適用できる | 既存統計policyとの接続。finite meanの統計保証をenergy保証へ昇格しない |
| PO-08 | 同情報・cache・precision・control/fusionを与えた近接対照と比較できる | 構成表とF1。既存poolへ同じtyped interfaceを与え、独立性未確定を維持 |
| PO-09 | 各primitiveと実行Hilbert空間が整合し、中間sector投影でmeanを変えない | full register上のcontrolled operator検査とsector外成分の扱い。sector符号化では全atomの不変性が必要 |

PO-01～09は全て `OPEN_NOT_EXECUTED`。価格DPの漸化式を書いたこと、RTEの既存関数を見つけたこと、文書verifierが通ることは、いずれもこれらの充足証拠ではない。

## 5. 既存関数を再利用できる範囲

`RTEComponent`と`RTETailLike`、`finite_rte_distribution`、`make_rte_config`、eventのapplication sequence／phaseは既存部品。構造的なI・Z/ZZの抽出条件は `df_rte_tail.py` の参考にするが、DF専用metadataを新辞書へ偽装しない。`normalize_involutory_tail`のdense演算子検証は小系参照用に限る。

`df_rte_qiskit.py`の非制御basis、中央制御、隣接同basis相殺、phase集約と、`df_partial_s2_repeated.py`のhalf-step融合を維持する。任意長Pauliのcontrolled実装は既存toy `apply_pauli`に参考があるが、production routerとそのprecision保証は新規作業である。今回それを実装していない。

固定出典：`C-SAMPLER`、`C-TAIL`、`C-CONSTRUCT`、`C-SHOTS`、`P2-BUILD`、`P2-REPEAT`、`EV-0028`。[固定identityと範囲](SOURCE_EVIDENCE_INDEX.json)。
