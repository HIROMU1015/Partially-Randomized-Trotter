# P3 Stage 2：H4 matched有限RTE信号・normalizationの科学契約案

2026-11-02 JST。**非実行の契約案。科学実行は未承認。**
本資料と[実行準備表](h4_p3_stage2_execution_readiness_20261102.md)の公開・全文照合後にmandatory STOPする。
今回の科学callは0。新operator、signal、B、law、回路、SOURCEを作成していない。
以下の数式は将来の実行定義であり、数値結果ではない。

## 1. 引継ぎ・初期tuple

基点はStage 1結果`5708e37080738c4ebf74590d799a3741246483b2`、その科学SOURCEは
`da13170e593e6d845b014e3f110691be10e29d43`。
[REPORT][S1REPORT]、[34項目の比較][S1CMP]、[独立構築経路][S1PATH]、
[入力来歴][S1INPUT]、[run01 STOP][S1STOP]を継承する。
H/T/Vのspectral差`7.588648017242457e-15` / `2.886535547987025e-15` /
`1.2843889125727285e-14`は保存値の引用である。operatorのengineering一致を、
新しい信号差・B・native回路・費用の一致保証へ移さない。
Stage 1の入力decoder preflight停止と、修正SOURCEからの初回operator batchは別記録として保全する。

| 項目 | 初期Stage 2契約案の固定条件 |
|---|---|
| 系 | Linear H4、隣接1.00 Å、STO-3G、中性singlet、system 8 qubits |
| 入力 | 保存共通H4入力、tol-only `1e-8`の保存DF rank10。再SCF/FCI/DF分解なし |
| 分割 | 案A：保持`T_eff`、残差`V_square`。残差identityは共通scalarへ移管 |
| 時間 | `T=0.8`、S2、`q=8`、`r=4` / outer step、`K=4`（最大偶数event次数4、多項式5次） |
| 時間operand | `T=Fraction.from_float(0.8)`、`δ=T/8`、`h=δ/4`。同じoperandを両armで共有 |
| 比較 | `M_B`：共通T＋Pauli残差辞書、`M_C`：共通T＋DF残差辞書のみ |
| 空間・state | full256次元、同じ保存raw φ。36次元への射影・再正規化なし |
| 保持発展 | 共通の数値的exact exponential。T内部のPF近似なし、回路合成なし |

`qr=32`は各nominal meanに含まれる独立なshort-step lawの位置数であり、trajectory標本数ではない。
残差をS2の両側に分けて64位置にしない。保持half-stepは各outer stepの両側に置く。
このtupleを大域最適とは主張しない。M_A純RTE、案B、Stage 3/4、P1Nは新計算の対象外。

## 2. 共通operator・scalarの所有

数学的には

\[
 H=c_NI+T_{\rm eff}+V_\square,
 \quad v_I=\operatorname{Tr}_{256}(V_\square)/256,
 \quad V_0=V_\square-v_II,
 \quad c=c_N+v_I.
\]

共通保持TにはそのJW identityも残す。nuclear scalarと残差identityだけを外側phaseへ一度移す。
Pauliの保存full-H constantを、nuclear scalarへ追加しない。

数値実行ではDFを研究のanchorとし、保存`df_one_body`のnative occupation作用を`T_*`、
保存10組の`λ_l,G_l`から直接作る`Σ_l λ_l A_l²`を`V_*`とする。
この直接構築は後述のrotated Z/ZZ辞書和から独立させる。
共通数値scalarには[Stage 1保存値][S1CMP]の

| 所有 | 引用するbinary64値 |
|---|---:|
| `c_N` | `2.29310124732` |
| 共通`v̄`：`scalar_identity.DF_square_total` | `4.679666433618258` |
| Pauli residualの保存identity（照合用、共通scalarには不使用） | `4.679666433618263` |
| DF Tの保存identity（T内に残す） | `-7.30424549435507` |
| Pauli full-H constant（追加しない） | `-0.33147781341681026` |

を用いる。`V_0*=V_*−v̄I`、`c_*=c_N+v̄`が初期実行の共通数値targetである。
二つの保存scalar operandのexact rational和をphase入力の基準とし、binary64への丸めも記録する。
本準備ではscalarの新しい加算値・traceを計算していない。

数学的trace係数`v_I`と保存数値binding`v̄`は区別する。
各辞書の抽出identityを`v_X`とし、`v_X−v̄`、辞書和と`V_0*`の差を将来測定・保存する。
各armだけに別のscalarを使う、identityをランダム成分に戻す、あるいは差を黙って捨てることは禁止する。
微小なidentity差・basis丸めは§7で信号へ伝播するbinding uncertaintyに含める。
差がoperator整合条件を満たさなければSTOPし、別規約へ自動変更しない。

## 3. 独立な残差辞書

### M_B：Pauli辞書

Stage 1のPauli経路と同じ元quartic tensorから`V_normal−D̂`をJW展開する。
`W`は既に二体の`1/2`を含み、`T_eff=T_bare+D̂`、`V_square=V_normal−D̂`である。
DFの`λ,G`やDF辞書和をPauli残差生成の代用にしない。
`PauliHamiltonian.from_openfermion/from_terms`を使い、同じglobal Pauli labelを統合後にidentityを分離し、
cutoff0（exact zeroのみ除去）、既存のabs係数順・label tie順で保存する。

\[
 V_0^P=\sum_{j:P_j\ne I}a_j^P P_j,
 \qquad P_j^\dagger=P_j,\ P_j^2=I.
\]

spin順はinterleaved、JWはlittle-endian、labelは`q[n−1]…q0`。
複素係数の虚部を無条件に落とさず、Hermiticityと変換丸めの検査・誤差記録を行う。
全Hのabs-prefixを保持Tの代用にしない。

### M_C：DF辞書

保存fragment生成順`l=0,…,9`を保持し、各8×8 Hermitian `G_l`を
`G_l=U_l diag(η_li) U_l†`と対角化する。これは保存factorのbasis決定であり、再DF factorizationではない。
`diag_hermitian(sort='descending_abs', assume_hermitian=True)`をHermiticity確認後に使う。
自動symmetrization・QR修復は行わない。固有basis・順序・固有値を将来の科学descriptorとして保存する。

\[
 n_{li}=\sum_{pq}U_{l,pi}U^*_{l,qi}a_p^\dagger a_q,
 \quad Z_{li}=I-2n_{li},\quad s_l=\sum_i\eta_{li},
\]
\[
 \lambda_l A_l^2=\alpha_l I+\sum_i b_{li}Z_{li}
                       +\sum_{i<j}d_{lij}Z_{li}Z_{lj},
\]
\[
 \alpha_l=\frac{\lambda_l}{4}(s_l^2+\sum_i\eta_{li}^2),\quad
 b_{li}=-\frac{\lambda_l}{2}s_l\eta_{li},\quad
 d_{lij}=\frac{\lambda_l}{2}\eta_{li}\eta_{lj}.
\]

`exact_df_diagonal_coefficients`でI/Z/ZZ係数を得て、全`α_l I`を辞書から除く。
成分IDは`(fragment,basis,Z/ZZ support)`で区別し、異なるbasisの同じsupportを統合しない。
fragment rank/proxy weightを係数L1 normに置き換えない。cutoff0とし、basis内は既存係数関数の生成順を使う。
full occupation作用でprojector/Z/ZZを作り、Stage 1と同じ`reverse_bits(8)`でnative配列をQiskit順へ移す。
Gaussian circuitは作らない。
数学的なinvolution性と数値basisの直交・固有残差を区別し、後者の影響をbinding uncertaintyへ含める。

両経路でそれぞれ作った`V_0^X`を共通`V_0*`と照合する。
共通operatorを辞書和へコピーして一致を得ることはしない。

## 4. Nominal lawとnormalization

各`X∈{P,DF}`について、符号をinvolutionへ含める：

\[
 \lambda_R^X=\sum_j|a_j^X|,\quad p_j^X=|a_j^X|/\lambda_R^X,
 \quad S_j^X=\operatorname{sgn}(a_j^X)P_j^X,
 \quad W_X=\sum_jp_j^XS_j^X=V_0^X/\lambda_R^X.
\]

係数のbinary64 operandをexact値とみなす和がλの基準であり、`fsum`の丸めとの差も記録する。
`a_X=λ_R^X h`、`k∈{0,2,4}`について

\[
 w_k^X=\frac{a_X^k}{k!}\sqrt{1+(a_X/(k+1))^2},\quad
 \beta_X=\sum_kw_k^X,\quad \pi_k^X=w_k^X/\beta_X,
 \quad\theta_k^X=\arctan(a_X/(k+1)).
\]

一eventではk個のproduct成分と1個のrotation成分を同じ`p_j`から独立に選ぶ数学的lawを使う。
SOURCEの作用順はproducts、rotationで、phaseは`(-1)^(k/2)`である。
標本を生成せず、その理想平均をmoment恒等式で評価する：

\[
 Q_X=\sum_{k=0,2,4}\pi_k^X(-1)^{k/2}
       (\cos\theta_k^X I-i\sin\theta_k^X W_X)W_X^k,
 \qquad \beta_XQ_X=P_5(-ihV_0^X).
\]

`cosθ=1/√(1+x²), sinθ=x/√(1+x²)`を使えばbinary64角度bindingを導入せず計算できる。
係数確率、event次数確率、short-step β、全体Bを別fieldで保存する。

\[
 B_{\rm total}^X=\beta_X^{qr}=\beta_X^{32},\qquad
 \log B_{\rm total}^X=32\log\beta_X.
\]

Pauliはglobal Pauli辞書の係数和、DFはbasisを区別したrotated Z/ZZ辞書の係数和を使うため、
同じ物理残差でもλ・β・Bが異なり得る。これらの数値は今回未計算。
`finite_rte_distribution.exact_finite_distribution`というfield名は、浮動小数点計算のrigorous certificateを意味しない。

## 5. 信号、共通reference、signed bias

### State・ν・保存reference

保存raw state hashは`8339b1a8b0bec35ae07688e667fde7ba602d56c1182523d58e0889ea114e8b80`。
256個のcomplex128成分をsigned zeroも含めbit-copyで読み、再生成・再最適化・再正規化しない。
保存[exact norm][NU]の`raw_norm_exact`をそのまま用いる：

```text
ν = 139984046386112801203518524734647603751956277672653015785565228589451076440261666494512381492855534829/139984046386112763159840142535527767382602843577165595931249318810236991948760059086304843329475444736
```

同じraw stateとの対応は[保存input binding][NUBIND]と[inputs][NUINPUT]で確認する。
全て`Z(U)=φ†Uφ/ν`、実部cosine・虚部`+Im` sineとする。既にphysical化された値を再びνで割らない。

共通referenceにはSOURCE `e80d7510880f8db993b777c627bb8e95b05a070d`の
[保存DF C結果][DFREF]の`reference`を選ぶ。Pauli C結果は[別表現の参考証拠][PREF]として残す。
`h4_ball_evidence.CandidateWorker._tier`と`signal`を確認すると、保存`reference`は
**raw φ内積でありν未除算**である。将来、次の保存raw区間の両端をexact νで一度だけ割り、physical共通reference区間とする。

| raw区間 | 下端 | 上端 |
|---|---|---|
| Re | `-11924112533004168869/73786976294838206464` | `-11924112533004168379/73786976294838206464` |
| Im | `9102140382757569469/9223372036854775808` | `2275535095689392383/2305843009213693952` |

保存計算は64bit Arbの`TARGET_MET`で、正式科学受理はfalseのまま。保存区間自体の保証と新計算のengineering誤差を混同しない。
今回は区間の新しい除算値・midpointを計算していない。新full-H exponentialは初期Stage 2にも含めない。

保存DF Cのexact-operand full Hを`H_C`、保存DF full matrixを`H_s`、新共通構築を
`H_*=c_NI+T_*+V_*`とする。保存DF結果の`operator_snapshot_residual_upper.upper`は

```text
17902112760494771817/649037107316853453566312041152512
```

である。将来`||H_*−H_s||₂`と構築丸め評価を加え、`H_C`へのtransfer uncertaintyを記録する。
Hermitian作用素についてのDuhamel評価`|ΔZ|≤|T| ||ΔH||₂`を接続するが、
新しい点推定norm＋engineering allowanceをrigorous上界と呼ばない。
Stage 1のPauli/DF H差を0と置かず、Pauli辞書の`V_0^P−V_0*`も別途会計する。

### Exact split・独立finite mean

`E_T=exp(−iδT_*/2)`、`E_V=exp(−iδV_0*)`を共有・cacheし、

\[
 U_{\rm split}=e^{-ic_*T}(E_TE_VE_T)^q,\quad
 U_{\rm nom}^X=e^{-ic_*T}[E_T\{P_5(-ihV_0^X)\}^r E_T]^q,
\]
\[
 U_{\rm raw}^X=e^{-ic_*T}[E_TQ_X^rE_T]^q,\quad
 Z_{\rm nom}^X=B_{\rm total}^X Z_{\rm raw}^X.
\]

Pauliは辞書からのvector paired recurrence、DFは独立DF辞書和からのdense finite operatorで構成する。
Pauli helperの`step_time`にはδを渡す（関数内でrで割る）。DF configも`evolution_time=δ`、
`rte_steps=4`、`dimensionless_step_time=λ_R h`とし、hをさらにrで割る誤接続を避ける。
さらに`V_0*`からの直接多項式作用`Z_nom*`を共通operator対照として1回取得する。
`Z_nom*`を両armへコピーしない。raw meanもmoment式で独立評価し、`Z_nom/B`だけで定義して照合を自明にしない。
corrected recurrenceの`normalization=1`を守り、補正済み値へBを再び掛けない。
全体phaseは共通scalarを一度、内積のν補正も一度。事後phase alignmentはしない。

理想的に`V_0^P=V_0^DF=V_0*`なら、上の有限5次多項式により両armの補正平均は一致する。
これは辞書L1 normやBの一致を必要としない数学的恒等式である。
数値実装での一致は独立な辞書構築と§7の許容不確かさ内で初めて確認できる。

\[
 \Delta_{\rm outer}=Z_{\rm split}-Z_{\rm ref},\quad
 \Delta_{\rm finite}^{X}=Z_{\rm nom}^{X}-Z_{\rm split}.
\]

referenceを区間として保持し、表示用midpointとの差だけでacceptしない。
Re/Im別の符号付き差・不確かさを保存する。数値binding差はfinite差に混入し得るため別欄にも示す。
絶対値の加算を一意な因果寄与率とせず、raw attenuationと補正後finite biasも区別する。

## 6. 初期scopeと後続scope

初期実行案は1つのtuple、2つの独立辞書、共通operator bundle1組、複素信号6個
（split、nom P、nom DF、nom *、raw P、raw DF）とnormalization bundle2組だけとする。
共通exponential2個とその同一精度のspectral照合、保存Gの8×8対角化10回を含む。
科学計算としての件数・再利用境界は[実行準備表§4](h4_p3_stage2_execution_readiness_20261102.md#4-将来の最小科学実行範囲と件数)に定義する。

初期scopeに53-bit CDF grid、固定binary64 event角度、implemented mean、trajectory、RNG replayは含めない。
将来の別scopeでは方式ごとに`Δ_grid^X=Z_implemented^X−Z_nom^X`を扱う。
ideal moment式を再利用することと、保存CDF/angleによる実効lawの評価は別処理である。
lawの包含、native circuit、実compiled残差ρ、shot/N/C/Gもそれぞれ別問題とする。

Stage 3ではM_B/M_Cに同じDF-native Gaussian一体blockを使う方針を維持する。
数学的Gaussian分解は二次Hamiltonianのexact evolutionを表現できるが、
数値basis、角度binding、gate lowering、compilerの誤差まで0とはしない。
Stage 2ではT内部PF biasを導入せず、Stage 3でその数値/native作用を共通`E_T`へ照合する計画に留める。
今回も初期Stage 2もGaussian回路をbuild/compileせず、dense exponentialの価格をnative価格へ転用しない。

## 7. 数値精度・correctness・STOP

### 固定精度と誤差会計

初期matrix/vectorはcomplex128、full256。SciPy `expm`とHermitian spectral分解による照合を共通T/Vに1回ずつ行う。
主計算はStage 1のNumPy1.26.4/SciPy1.14.1/OpenFermion1.6.1系を再利用候補とし、
実際のproject専用runtimeとversionを将来SOURCE固定時に記録する。bitwise環境同一性を新たな要件にはしない。
scalar λはencoded係数のexact rational和も保持し、β/B/log Bの独立照合だけ標準`decimal`50桁で行う。
`finite_rte_distribution`のbinary64評価と同じ3項の式を比較し、差と丸めを残す。
decimal値もexact λ/δから計算し、最終的なbinary64変換の差をraw mean/B closureへ伝播する。
これはscalarの診断精度であり、full signalの高精度包含計算ではない。新しいArb依存環境を初期実行の必須条件にしない。

Stage 1のoperator比較政策`τ_A=16γ_d max(1,||A||₂,||B||₂)`、
`γ_m=mε/(1−mε)`、`ε=2^-52`を新辞書和・Hermiticity・basis再構成の整合検査に適用する。
dimensionは対象matrixのd（8または256）。PASSは差`≤τ_A/4`、FAILは`>4τ_A`、その間はUNRESOLVED。
**このoperator政策をsignalの固定許容差として使わない。** 過去の`1e-10`も採用しない。

信号には以下を分けた、演算に応じるengineering uncertaintyを付ける。

1. 保存reference区間幅とexact νによる変換。`H_C↔H_*`のtransferを別欄で追加する。
2. 辞書sum、identity、DF eigen/orthogonality残差を含むoperator binding。
3. exponential、有限多項式、r/q反復、phase、内積、ν除算の丸め・計算法の差。
4. λ/β/Bのscalar丸め。raw meanの補正照合にはBの不確かさも伝播する。

将来の最小実装はnorm評価と次の少数の式で誤差を集約し、library内部の全call監査frameworkは作らない。
complex行列積の丸め目安を`g(A,B)=γ_(8d) || |A||B| ||_F`、
誤差伝播を`e(AB)=||A||₂ e_B+||B||₂ e_A+e_A e_B+g(A,B)`とし、
和・scalar積・matvec・dotにも同じγと絶対値和の評価を使う。
主計算とspectral exponentialのFrobenius差、固有再構成残差`r_A`、直交残差`r_U`から
`e_exp=16[difference+|t|r_A+r_U(2+r_U)+γ_(8d)(1+|t| ||A||₂)]`をengineering目安とする。
Hermitian条件の丸め残差もこの会計に含め、勝手にsymmetrizeしない。

辞書binding差ε_Vに対して、`R=max(||−ihV_0^X||₂,||−ihV_0*||₂)`、
`L_5(R)=Σ_(j=1)^5 R^(j−1)/(j−1)!`を用い、短時間多項式差の目安を`h L_5(R) ε_V`とする。
その後は実際の非unitary polynomial normを使ってr/qの積誤差を伝播する。
`r M^(r−1)`、`q M^(q−1)`の増幅を省略せず、多項式をunitaryと仮定しない。
raw φの内積へ戻してνで割るまで伝播し、Re/Imごとの`u_num`、`u_bind`、`u_ref/transfer`を出す。
identity defectは辞書binding差の内訳であり、同じ差を二度加算しない。
outer差にはsplit数値誤差＋reference幅＋Hamiltonian transfer、finite差にはnom/split数値誤差＋辞書bindingを付ける。
finite差へ無関係なreference幅を足さず、B closureにはraw/nom数値誤差とBの不確かさを付ける。
これらは検算可能な診断目安であり、厳密包含や主要論文claimの保証には使わない。

### 判定と終了

| 条件 | 初期実行の判定・処理 |
|---|---|
| 保存入力/hash、raw state/ν対応、SOURCEに不一致 | 科学処理へ進まずSTOP。保存原本を修復名目で書き換えない |
| scalar/D符号、1/2、辞書和、basis/Hermiticityがoperator整合検査で非PASS | 後続信号を実行せずSTOP。原因を記録 |
| `nom P−nom DF`、各`nom X−nom *`、`B_X raw X−nom X` | 各比較の数値＋binding不確かさの和U以内でPASS、U超〜4UでUNRESOLVED、4U超でFAIL |
| closureの非PASS、非有限値、予想外のλ=0、安定な誤差会計が作れない | 対応する後続処理を停止。別r/K/solver/state/referenceへ自動変更しない |
| outer/finiteのReまたはIm差がその不確かさ以下 | その成分はUNRESOLVED。0確定・原因解明済みにしない。既定の同tuple計算を超える再計算はしない |
| 数値closureはPASSだがbias成分が未解像 | 結果を`STOP_STAGE2_UNRESOLVED_COMPONENTS`として終了。許可なしに精度昇格しない |
| 全予定比較が解像・整合 | `STOP_STAGE2_NOMINAL_DIAGNOSTIC_COMPLETE_NO_AUTOMATIC_FOLLOWUP`で終了 |

PASSは工学的な整合確認に限定する。主張できるのは固定tupleのnominal平均・正規化・signed差とその解像状態まで。
資源優位性、全law実装保証、実ρ=0、総N/G、M_Aとの新比較、主要claim採択は支持しない。

## 8. 承認境界と今回のmandatory STOP

科学条件・数学定義・数値binding・6信号/2normalizationの最小scopeは本案で具体化できる。
数値的な適合・分解能は未測定であり、実行成功を前提にしない。
ユーザーがこの案の科学scopeと精度政策を明示許可した後に限り、Codexが最小接続・人工correctness検査・
新しい科学SOURCE固定・同tupleの限定実行を行う。SOURCEの具体的SHAはその実装後にのみ決まる。
承認証明書、approval packet、専用grant発行基盤は作らない。
通常の実装・debugは許可範囲内でまとめて行い、科学的な条件変更は別の明示許可を要する。

今回は文書2件を公開・照合して**mandatory STOP**。
Stage 1の`STOP_STAGE1_PASS_NO_AUTOMATIC_FOLLOWUP`、P1/P2、元SOURCE/input/raw/result/旧STOPを保全する。
詳細科学レビュー・正式採択・論文改訂にも移行しない。

[S1REPORT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5708e37080738c4ebf74590d799a3741246483b2/artifacts/df_pauli_matched_partition_validation/2026-11-02/stage1/REPORT.md
[S1CMP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5708e37080738c4ebf74590d799a3741246483b2/artifacts/df_pauli_matched_partition_validation/2026-11-02/stage1/run01/operator_comparisons.json
[S1PATH]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5708e37080738c4ebf74590d799a3741246483b2/artifacts/df_pauli_matched_partition_validation/2026-11-02/stage1/run01/construction_paths.json
[S1INPUT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5708e37080738c4ebf74590d799a3741246483b2/artifacts/df_pauli_matched_partition_validation/2026-11-02/stage1/run01/source_and_inputs.json
[S1STOP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5708e37080738c4ebf74590d799a3741246483b2/artifacts/df_pauli_matched_partition_validation/2026-11-02/stage1/run01/STOP.json
[DFREF]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/non_df_h4_c_reference_evidence/2026-10-23/run01/df_native_result.json
[PREF]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/non_df_h4_c_reference_evidence/2026-10-23/run01/pauli_jw_result.json
[NU]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/non_df_h4_central_arb_pf_cosine/2026-10-25/run01/signal_64.json
[NUBIND]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/non_df_h4_central_arb_pf_cosine/2026-10-25/input_binding.json
[NUINPUT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/non_df_h4_central_arb_pf_cosine/2026-10-25/inputs.json
