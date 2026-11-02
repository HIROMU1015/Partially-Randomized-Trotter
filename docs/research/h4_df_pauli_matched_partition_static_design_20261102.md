# P3：H4 DF/Pauli-JW matched Hamiltonian分割の静的調査

2026-11-02 JST。**静的SOURCE・保存metadataの調査のみ。分割案・科学条件は未採択、P3科学実行は未許可・未実行。**
対応する[実行準備・再利用表](h4_df_pauli_matched_partition_execution_readiness_20261102.md)と一組で読む。

## 1. 目的と現在の到達点

研究の中心は、DF Hamiltonianに対する部分ランダム化PFのリソース評価である。Pauli-JWは比較対象である。
全体の到達点は[研究全体地図][LAND]・[claim/evidence対応表][GAPS]を基準とし、P3の元の未確定事項は[検証設計§6][DESIGN]・[準備資料P3節][READY]を引き継ぐ。

[P1][P1]はH6 Pauli-JWの保持78項がI/Z可換であり、残存biasのスケールが保持/tailの外側対称分割に対応すると報告した。
[P2][P2]は同じ研究系の固定3trajectoryで、保持回路の大きな単独価格と結合時の非加法性を確認した。
これらはPauli内の診断であり、H4 DFとのmatched分割、DFのGaussian費用、表現単独の因果寄与を検証していない。
P1の`STOP_UNRESOLVED_COMPONENTS`、P2の`STOP_P2_COMPLETE_NO_AUTOMATIC_FOLLOWUP`を保持する。

今回の静的結論は、**同じfull Hの保存数値診断と共通raw stateはあるが、新しい保持T・残差Vの個別一致は未検証**ということである。
案Aは既存DFの一体保持構造に近い。案Bは補正一体演算子を残差辞書に加える経路が必要になる。
以下の恒等式・実装上の推論と、既存数値証拠を区別する。新しいoperator値・一致誤差・費用倍率は計算していない。

## 2. 固定H4共通入力と照合した原本

基準commitは`4591b8d75a2a6a55c74a51225f0c61648023c8ee`、入力生成SOURCEは`0c329b30a81accb35c212be8defa66fee0dd0f12`。
[入力生成receipt][REC]、[Pauli Cモデル][PM]、[DF Cモデル][DM]、[旧比較の統合資料][SYN]を実際に読んだ。

| 条件 | 保存内容・適用範囲 |
|---|---|
| 分子 | linear H4、z=0/1/2/3 Å、隣接1.00 Å、STO-3G、中性singlet |
| 軌道・qubit | real orthonormal spatial MO 4、interleaved spin orbital 8、8 system qubits |
| Pauli | 184非identity項＋constant、係数cutoff 0、descending absolute coefficient→label順 |
| DF | `df_rank=None`、`df_tol=1e-8`で返却rank 10。rank10を強制した生成ではない |
| 共通入力 | 同じ未threshold MO一体積分・chemist ERI・核反発。再SCF/FCI/DF生成をしない |
| state | 両Cモデルに同一256要素のcomplex raw配列、同一state_hash |
| sector | Nα=Nβ=2の保存36次元index。full Fock空間は256次元 |
| 旧比較条件 | T=.8、S2、q8/r4/K4、step_time=.1。Pauli prefix27とDF prefix1＋一体保持は異なる分割 |
| compiler境界 | Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、full cosine/+Im sine wrapper、P=0。新しい分割の価格は未取得 |

receiptの`gates.DF_full_H_agreement.error=8.462237051574415e-15`、tolerance `1e-9`、PASSは**過去の数値診断の引用**である。
SOURCEではfull 256次元行列のspectral normを比較した。[build_h4_inputs][INPUT]。
数学的な厳密同一性や、新しいT/V別の数値一致を証明した結果ではない。
`df_truncation_value=0.0`も保存浮動小数値であり、丸め誤差を含む全Hの厳密等号へ格上げしない。

今回の読取hash照合は以下。[元snapshot][SNAP]はbytesのSHAとNPY headerの名前・shape・dtypeだけを確認し、配列を数値operatorへ再構成していない。

| 原本 | SHA256 |
|---|---|
| receipt.json | `6a2304db448845aef6f86da06a7bffd5d5c708273964e7e797c3d8ec71fcb6bf` |
| snapshot.npz | `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50` |
| 元pauli.json | `49e2f0b86cba9933eee20a03a05934d950caafb5044401bca50d3aa79a518175` |
| pauli_jw_model.json | `9ae8973b457cab26bb685a8fac9887dbc5e3be5db9059355822893d26cae2aa4` |
| df_native_model.json | `cc60f6dc3de885974cfe74ef822e49380edb6b9ca44777bf12e9654256cd322a` |

snapshot/pauli.jsonのSHAはreceipt記載値と一致。receiptおよび入力生成SOURCEはGitHub接続でも取得した。
両Cモデルのraw配列を比較し、保存binary64対をlittle-endian complex bytesへpackして既存array_hash規約と照合した。
共通state_hashは`8339b1a8b0bec35ae07688e667fde7ba602d56c1182523d58e0889ea114e8b80`。
これは配列・hashの照合であり、norm・energy・signalの再評価ではない。
receiptの独立DF ground state `df_state_hash=bfdc4948…`は別の診断配列であり、共通Cモデルのstateと取り違えない。

NPY headerには`one_mo(4,4)`、`eri_chemist(4,4,4,4)`、`spin_one(8,8)`、`spin_two(8,8,8,8)`、
`df_one_body(8,8)`、`df_lambdas(10)`、`df_g_matrices(10,8,8)`、stateと両full matrix、36 indexが存在する。
bare/effの将来の復元に必要な保存tensorはあるが、今回その差を数値計算していない。
legacy H4 DF12、長時間研究の別H4 DF10、H6 weighted-Hermitian completionはこのanchorへjoinしない。

## 3. Hamiltonian生成・normal-orderingのSOURCE対応

### 3.1 積分・spin・JW規約

[h4_molecular_inputs.py][INPUT]の`build_h4_inputs`は同じMO係数から`one=mo.T @ hcore @ mo`とPySCF `ao2mo.restore(1,...)`のchemist ERIを作る。
[PauliHamiltonian.from_spatial_integrals][PAULI]は`eri.transpose(0,2,3,1)`を用い、spin indexを`2*p+alpha(0)/beta(1)`とする。
一体は`spin_one[a::2,a::2]=one`、二体は`spin_two[a::2,b::2,b::2,a::2]=two/2`。
従って`InteractionOperator`へ渡す二体tensorには既に1/2が含まれる。さらに1/2を掛けない。
DF生成は`spin_lift_unthresholded`で同じspin tensorを作り、`two_body=spin_two/2`を渡す。ここでは当該helperの返すspin_twoはまだ1/2を含まない。
独立legacy照合に使った`spinorb_from_spatial`にはhidden cutoffがあり、production DFの経路とは区別している。

Pauli生成は`InteractionOperator→get_fermion_operator→jordan_wigner→from_openfermion`。
後者はqubit qをlabelの`n-1-q`に置き、Qiskit little-endianの`q[n-1]...q[0]`へ変換する。
独立OpenFermion行列との照合には`reverse_bits`を使う。MOはこの保存順序を保持し、軌道の再並べ替えや別SCFの符号/gaugeを導入しない。

ここでのbare二体入力は、`W[P,Q,R,S] a†_P a†_Q a_R a_S`というnormal-ordered quartic形式である。
入力生成側に追加の`normal_ordered()`呼出しはない。square表示への並べ替えは次の依存SOURCEで行われる。
一般的な量子化学の慣習からindex・符号を補ったものではない。[DF生成][DFH]、[OpenFermion低rank SOURCE][OFLR]。

### 3.2 one_body_correctionの生成式

元input launchが指す専用runtimeを読取調査し、OpenFermion 1.6.1を確認した。
`openfermion/circuits/low_rank.py`は公式v1.6.1のGit blob `330c4556ddd36b8885ceafce73afe8b456a1bd0f`と一致した。
SHA256は`de74fb802550417788362845515ed9e56a180e4ff85ab21ab43ba8445156dbbe`。
これは今回読めた依存SOURCEの同一性であり、全runtimeの履歴を新たに監査・認定したものではない。

同ファイル`get_chemist_two_body_coefficients`は、入力Wを`transpose([0,3,1,2])`し、spin_basis=Trueならalpha/alpha/beta/beta sliceを取る。
spatial indexでその返却tensorをCと書くと、コード上の関係は

\[
C_{pqrs}=W_{2p,\,2r+1,\,2s+1,\,2q},\qquad
D_{2p+\sigma,\,2s+\tau}=-\delta_{\sigma\tau}\sum_q C_{pqqs}.
\]

Dが`one_body_correction`である。上式のWは**1/2を含めて実際に渡したtensor**を意味する。
`low_rank_two_body_decomposition`はこのDと、Cの固有分解に由来するλ/Gを返す。
[df_hamiltonian_from_integrals][DFH]は`one_body = input_one_body + D`とし、constantは変更せず、最後にHermitian化する。
第4返却値`truncation_value`は誤差診断metadataであり、Hamiltonianのscalar補正として足していない。

**名前の注意**：[df_hamiltonian_to_model][PART]の`DFModel.one_body_correction`には、Dだけでなく`DFHamiltonian.one_body`全体、すなわちh_effが入る。
このfield名を根拠にh_effをDと同一視してはならない。

### 3.3 fragmentの定義・順序

[DFHamiltonian][DFH]の定義は`constant + Σ h_eff[p,q] a†_p a_q + Σ_l λ_l A_l²`、`A_l=Σ G_l[p,q] a†_p a_q`である。
依存SOURCEはspatial固有vectorを`kron(reshape(vector), eye(2))`としてspinへ戻し、fragment生成時に重み`abs(λ)*(sum(abs(G)))²`で降順にする。
projectの`rank_df_fragments`は別のproxy `abs(λ)*||G||_F²`でprefixを選ぶ。これをRTE辞書展開後のexact coefficient L1 normと同一視しない。
`build_df_blocks`は保存λ/Gの順で、各Gを`descending_abs`の固有値順へ対角化してGaussian basisを作る。
辞書内順序・fragment original_index・basis hashは別々に保存する。[PART][PART]、[Gaussian/diagonal primitives][OPS]。

## 4. T_bare、T_eff、V_normal、V_square

記号を`c_N`=核反発、`T_bare=Σ h_spin a†a`、`D̂=Σ D a†a`、`T_eff=T_bare+D̂`、`V_square=Σ λ_l A_l²`とする。
元のnormal-ordered quarticをV_normalとする。CARによる並べ替えと**打切りのない正確なfactorization**なら

\[
H=c_N I+T_{\rm bare}+V_{\rm normal}
 =c_N I+T_{\rm eff}+V_{\rm square},\qquad
V_{\rm normal}=\widehat D+V_{\rm square}.
\]

この恒等式は、bare二体をsquareへ並べ替える際の一体収縮Dを落とさないという意味である。
V_squareはbare quarticだけではなく、fermionic normal-orderingで一体成分を生み、JW/occupation表示ではidentityも含む。
今回の保存DFにはtol、浮動小数、Hermitian化があるため、保存値について無条件の厳密等号とはしない。
将来は`df_one_body−spin_one`とDの生成規約を対応させ、TとVを個別に比較する。差は今回計算していない。

| 分割候補 | 共通保持T | 共通残差V | 既存実装との距離・解釈 |
|---|---|---|---|
| 案A | T_eff | V_square | DF一体保持＋二体fragment L_D0に近い。Pauli側に同じTとsquare残差の明示経路が必要。「bare二体だけのrandom化」とは呼ばない |
| 案B | T_bare | V_normal | DF残差へD̂を加える必要。単にDF one_bodyをbareへ置換するとD̂が欠落する。物理的bare一体/二体の分割を問えるが、辞書の追加が必要 |

[B3 SOURCEのnative_context/context][B3SRC]はfragment prefix0でもone_body blockが1つ残ることを明示検査する。
従って案AはB3に近い**構造**である。既存B3の別snapshot・長時間・law・価格を今回のH4値へ転用する意味ではない。
Pauli L_D0は全非identity項をrandom化するため、B3と同一方式ではない。[B3公開成果][B3]。
**案A/Bの採用は未決定。両方の実行も今回許可・採択していない。**

## 5. scalar・identity・phaseの会計

保存Pauli constantは`-0.33147781341681026`、保存DF constant/核反発は`2.29310124732`である。[PM][PM]、[DM][DM]。
この相違は、それぞれconstant fieldへ移した内容が違うことを示す。Pauli側は電子演算子のJW identityを含む。
それだけを不一致として差し引いたり、DFへPauli constantをそのまま加えたりしない。
quartic→density productのD生成は一体補正であり、依存SOURCEに別のscalar返却はない。JW identity抽出とは別段階である。

[exact_df_diagonal_coefficients][TAIL]は、`λ(Ση_i n_i)²`をI/Z/ZZへ展開する。
`s=Ση_i`、`n_i=(I−Z_i)/2`から、コードと同じ代数式は

\[
\alpha_I=\frac\lambda4(s^2+\sum_i\eta_i^2),\quad
\alpha_{Z_i}=-\frac\lambda2\eta_i s,\quad
\alpha_{Z_iZ_j}=\frac\lambda2\eta_i\eta_j\ (i<j).
\]

これは式の確認であって保存ηから新しい係数を計算した結果ではない。
`extract_df_diagonal_tail`はthreshold後にidentityを抽出し、`extract_identity_phase`ならrandom辞書から除きdeterministic scalarへ移す。
`prepare_df_partial_s2`の既定はこのpolicy・coefficient_atol=0である。
異なるbasisの同じZ supportは同一global operatorとは限らず、componentをsupport名だけで統合しない。

将来のM_B/M_Cには、次のどちらかの**共通bookkeeping**を実行前に選ぶ。これは分割A/Bとは別のscalar表記の選択である。

| 表記 | 保持block | scalarとrandom残差 | 二重計上防止 |
|---|---|---|---|
| nativeに近い表記 | traceを含む物理T全体 | `v_I=Tr_full(V)/256`、scalar=`c_N+v_I`、random=`V−v_I I` | TのJW identityはGaussian blockのphase内に残す。Pauli/DFで同じv_Iを対応付ける |
| 全trace抽出表記 | `T−t_I I` | scalar=`c_N+t_I+v_I`、random=`V−v_I I` | Gaussian blockが本来持つt_I phaseを明示的に補償する。外側scalarとの二重加算禁止 |

traceはfull256次元のidentity係数を定義する記号であり、sector平均へ置換しない。今回trace数値は計算していない。
M_AはTを保持せず、元Pauli constant全体を一度だけ保持する。M_A/B/Cでphaseの置き場所が違っても、full Hのscalarを一度だけ実装する必要がある。
案Bの残差にはD̂のidentityもあるため、square由来identityだけを抽出して完了とはしない。

[one_body_diagonal_primitives][OPS]は一体blockの`−τ Σeps/2`に対応するglobal phaseを保つ。
[DF S2 builder][S2]はconstant/extracted identityを一回、controlledならancilla PhaseGateとして実装する。
[DF event builder][EVBUILD]は負product/order phaseを集約し、actual adjoint（逆順＋各gate.inverse）を使う。
Pauliの[D2制御][D2]もbody phaseをancilla相対phaseへ移す。controlled化後に無関係なglobal phaseとして捨ててはならない。

## 6. M_A/M_B/M_Cの定義と比較可能性

| arm | 保持・残差 | 再利用できる骨格 | 比較で判別できること／残る交絡 |
|---|---|---|---|
| M_A | 元Pauli全非identity純RTE、constant保持 | Pauli L_D0 law/event/central制御 | M_Bとの差は保持追加に加え、tail辞書・law・B・bias・回路構造が変わる |
| M_B | 選択Tを保持、選択Vをglobal Pauli辞書でrandom化 | 明示Pauli残差＋共通保持adapterが必要 | 共通物理分割を持つPauli実装。旧prefix27で代用不可 |
| M_C | M_Bと同じT/V、DF由来のrotated Z/ZZ辞書でrandom化 | 案Aはfragment全tail、案BはD̂のrotated Z追加 | M_Bとの差は辞書・λ・probability・angle・B・Gaussian/control・compiler応答の総合差 |

保存Pauli/DFのfull Hは既存数値診断の範囲で対応しており、bitwiseまたは数学的厳密等号ではない。どちらを共通targetにするか、既存差をallowance/転送で扱うかを実行前に選ぶ。新分割に合わせて旧full Hを黙って置換しない。
full H一致、T一致、V一致はそれぞれ独立に確認する。さらにscalar移管後のT/block・V/random辞書の一致も確認する。
同じ保持operatorと、同じ保持**入力program**は別条件。Gaussian保持blockを両armへ同じ命令列で挿入できれば、孤立保持価格を揃えられる。
それでもtailとの全体compile応答が異なるので、Full差から保持価格を機械的に引いて表現単独効果とはしない。
共通保持programが作れなければ、保持実装差を含む比較として表示する。[P2の限定解釈][P2]。

## 7. state・reference・sector・理想平均

共通raw配列φに対し、物理信号は`<φ|U|φ>/ν`、νは同じbinary64 raw値のexact squared normとする。
exact νは[既存Arb結果のraw_norm_exact][NORM]に保存され、[bridge][BRIDGE]はinput/outputを再スケールせずνで一度だけ割る。
今回νを再計算していない。両C結果のstate_norm包含は同一だが、reference包含はPauli/DF各表現のfull Hに対する別結果である。[Pauli結果][PR]、[DF結果][DR]。
共通rayだけでreference/allowance/ρ保証政策まで一致したとはしない。Stage 1で選んだtargetと保存referenceの転送が必要なら、その評価は将来の別科学scopeである。
cosineはRe、sineはSdg→Hで+Im。B補正はraw meanへ一度、ν補正も一度。attenuationと近似biasを混同しない。

spin liftとDFの`G_l=spatial⊗I_spin`という数学的構造から、正確なT_bare/T_effおよびV_normal/V_squareの**和演算子**はNα/Nβを保存する。
保存丸め・新しい辞書/実装での対応は未検証。さらに個別Pauli stringは一般に粒子数を保存せず、Gaussian basis途中や個々のrotated Z componentもspin sector保存を一律に仮定できない。
和Vの不変性から、全event trajectoryやgate途中の36次元閉性は導けない。
名目meanをVの多項式として評価する場合と、実装law/個別回路を作用させる場合を分ける。無断の射影・各gate後のsector切出しをしない。
既存[finite_rte_signal_validation][FINITE]にもbasis変換・leakageの検査があるが、新しい分割の検査結果ではない。

条件付きのcorrectness期待値として、同じV、q/r/K、同じTのexact half block・scalar、同じpaired cutoffなら、
補正済み名目meanは`P_(K+1)(−i δ V/r)^r`をtailに使う同じ演算子となる。
[finite_taylor_operator/finite_rte_corrected_operator][RTE]と[P1のpaired_polynomial_action][POLY]がこの関係を実装している。
従ってM_B/Cの名目補正mean一致を将来の検査にできる。一方、λ・未補正meanのattenuation/B・event angle・有限grid law・実compiled作用は同じとは限らない。
これは代数上の条件付き関係で、今回のH4新辞書で成立を数値確認した結果や中心claim採択ではない。

## 8. 確認済み事項と不足、停止

| 証拠の種類 | 今回の確認 | 未確認・将来の科学評価 |
|---|---|---|
| SOURCE/metadata | 積分index、1/2、spin/JW順、D吸収、scalar/identity・actual adjointの実装 | 新T/V descriptorの生成と数値operator化 |
| 保存数値証拠 | full-H PASSを引用、state配列/hash・snapshot bytes一致 | T/V別一致、辞書和・scalar一致、sector leakage |
| 代数 | CARとsource規約に基づくD移管、I/Z/ZZ式、条件付き名目mean関係 | 丸め・truncation・encoded basisを含む新実装の精度 |
| 回路/費用 | 既存central/Gaussian/wrapperの実装を読取 | 新分割のbuild/compile、実測RZ/CX/depth、law別N/C/G |

研究上残る交絡は、物理分割、辞書の集約・norm、law/normalization、basis/control実装、compilerの結合応答、state/reference/precision政策、候補選択・費用標本である。
新分割での価格やshot負担をP1/P2・旧D1/D2・別snapshotのB3から移さない。

成果物はこの文書と実行準備文書のみ。科学SOURCE・新数値結果・adapter・runner・testsは作成していない。
旧SOURCE/input/raw/result/grant/ledger/marker/STOP、main/元研究branch、他job・共有環境を変更しない。
正式共通運用指示2026-10-25の§8.1/8.4/10を適用し、承認証明・新grant基盤を作らない。
**GitHub公開・remote SHA/全文照合後mandatory STOP。次はユーザー/GPTがA/B/両方、共通保持block、最小Stage scopeを選ぶ。Codexは新しい明示許可なしに科学実行しない。**

[LAND]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/docs/research/pr_repository_scientific_landscape_20261101.md
[GAPS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/docs/research/pr_repository_claim_evidence_gap_matrix_20261101.md
[DESIGN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6f0759df7c1df7fd6d835922b0246753a8a98d6c/docs/research/pauli_pure_rte_mechanism_validation_design_20261101.md
[READY]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6f0759df7c1df7fd6d835922b0246753a8a98d6c/docs/research/pauli_pure_rte_mechanism_execution_readiness_20261101.md
[REC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/non_df_h4_pilot/2026-10-23/run01/input_run/input_generation/data/receipt.json
[PM]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/non_df_h4_c_reference_evidence/2026-10-23/run01/pauli_jw_model.json
[DM]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/non_df_h4_c_reference_evidence/2026-10-23/run01/df_native_model.json
[PR]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/non_df_h4_c_reference_evidence/2026-10-23/run01/pauli_jw_result.json
[DR]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/non_df_h4_c_reference_evidence/2026-10-23/run01/df_native_result.json
[SYN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/non_df_h4_comparison_synthesis_20261025.md
[INPUT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0c329b30a81accb35c212be8defa66fee0dd0f12/src/trotterlib/h4_molecular_inputs.py
[DFH]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0c329b30a81accb35c212be8defa66fee0dd0f12/src/trotterlib/df_hamiltonian.py
[PAULI]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0c329b30a81accb35c212be8defa66fee0dd0f12/src/trotterlib/pauli_signal_adapter.py
[OFLR]: https://github.com/quantumlib/OpenFermion/blob/v1.6.1/src/openfermion/circuits/low_rank.py
[PART]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0c329b30a81accb35c212be8defa66fee0dd0f12/src/trotterlib/df_partial_randomized_pf.py
[OPS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0c329b30a81accb35c212be8defa66fee0dd0f12/src/trotterlib/df_trotter/ops.py
[TAIL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0c329b30a81accb35c212be8defa66fee0dd0f12/src/trotterlib/df_rte_tail.py
[S2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0c329b30a81accb35c212be8defa66fee0dd0f12/src/trotterlib/df_partial_s2.py
[EVBUILD]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0c329b30a81accb35c212be8defa66fee0dd0f12/src/trotterlib/df_rte_qiskit.py
[B3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ebe9e2aa2c83cd71a6da0a885b78f32274bfb6ed/artifacts/resource_applicability/track_a_h46_long_time_b3/2026-10-26/REPORT.md
[B3SRC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/638e0a989b4471a075549ccb58617b1030e6748f/scripts/h46_b3.py
[P1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bfb4ca75437a9b3605a8f57b5f82100615dd034f/artifacts/pauli_pure_rte_mechanism_validation/2026-11-01/bias/REPORT.md
[P2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c9073274daee69c9b69d8db3dc332bf6132c2259/artifacts/pauli_pure_rte_mechanism_validation/2026-11-01/cost/REPORT.md
[D2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/11e50e262f95630b14fa4070ad30894ba3f64b74/src/trotterlib/h4_d2_pauli_control.py
[NORM]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/non_df_h4_central_arb_pf_cosine/2026-10-25/run01/signal_64.json
[BRIDGE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/h4_arb_bridge.py
[FINITE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0c329b30a81accb35c212be8defa66fee0dd0f12/src/trotterlib/finite_rte_signal_validation.py
[RTE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0c329b30a81accb35c212be8defa66fee0dd0f12/src/trotterlib/rte.py
[POLY]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a0a3aff915f98456733d6bf09ed641a1be423628/src/trotterlib/pauli_signal_adapter.py
[SNAP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/non_df_h4_pilot/2026-10-23/run01/input_run/input_generation/data/snapshot.npz
