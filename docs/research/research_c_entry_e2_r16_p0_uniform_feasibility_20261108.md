# 研究C：ENTRY E2/r16のWhole-P0一様保証・Trajectory Target成立性

2026-11-08 JST。**CASE C：targetを変えない誤差連鎖と一部の解析的評価を具体化したが、全word正常終了・実Rust一様誤差は未成立。**
既存P0を変更せず、数値入力を得るだけで全law保証を完成できる段階には達していない。
新しい実H4計算・保存科学データの数値再解析・回路build/compile・Arb包囲・RPE/価格計算・重要GPT科学レビューは0。
科学実行可能な契約やcall数を作った扱いにせず、mandatory STOPする。

今回の進展は次の三点である。

1. Compiler比較を**同じ保存raw/compiled gate列の厳密な意味**の間に置き、literal CARとの差を別脚へ置くことで、
   nonunitaryなliteral Uをunitary composerへ入れず、元reference targetを維持する接続を示した。
2. Literal ZZがHermitian/involutionでない場合も扱えるGram-defect→event norm→IID categoricalの式を導出した。
3. Euler角自体の不連続性を避け、厳密算術の再構成mapを全near-unitary領域で抑えるlemmaを導出した。
   これはRustのf64/libm・分岐簡略化・phase処理を包囲した結果ではない。

## 1. 固定対象・範囲・既存証拠

起点は[前段報告](research_c_entry_e2_r16_all_law_readiness_20261108.md)、commit
`10269db6b4ad27707dcfe624dd487508bab733a8`。そのmetadata adapterと106人工/回帰を再利用した。
直鎖H4・隣接1.00 Å/STO-3G、ENTRY `H4_100_C0_ENTRY_g003_budget64`（strength .03/64更新）、number CDF8、
元順LD2/F0,F1 prefix、`tail_linear_Z_to_existing_kappa_keep_frame_identity_v1:LD2`。
Partial ordered S2/s1/r16/K0、δ=h=.1、m0..12、Nα=Nβ=2/sector36、system256/control込み512、
own-DF ground oracle/準備費除外。CA=.00159、DF reserve1/8、βtotal=π/6、βstat=π/12、αtotal=.01/axis=.01/26。
P0/Qiskit1.3.0/opt1/seed17/rz,sx,x,cx/approximation_degree1/qubits_initially_zero=false、
MPFR160 declared phase、phase fitなし、actual raw/compiled inverse、head/X/Y、fenced serialを維持。

- 科学SOURCE：`cfc64b70861a6482dddd042da8a110abebd6ad7a`。
- code SOURCE SHA256：`6bd88bc86bd1da2876f87b8ae67be509b15727f01517cea9c7219f91c350c4cc`。
- J/U NPZ SHA256：`0e1119dfb6be9b75fb5e592b7a65c76cda61e348fd7cad65d7f104bf997929f5`。
- J/U/κ/scalar/物理入力と再配置の詳細hashは前段の[readiness](../../artifacts/research_c_entry_e2_all_law/2026-11-08/technical/readiness.json)をそのまま使う。
- 条件付き総RZ **75,621,395,471** は上記条件のideal finite-RTE/exact-native laneと保存価格proxyの引用。
  今回の保証取得済み価格ではなく、再計算・保証付きshotsへの接続は行わない。

旧IID-grid Stage A 38-call、B2/Arb384 27-call、4固定word 86-callは実行済みとして扱う。
旧candidate `a8813aca…`・再配置前r32の数値を今回へ流用せず、旧包囲・pilotを再実行しない。
Arb192の固定回路/polar/exact-H targetも新しい正式targetとして採択しない。
既存結果の資格は前段報告と後続[B1/B2統合](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3eccab64a2672813e479c113c2485219f8479885/docs/research/research_c_b1_b2_reference_bound_integration_20261103.md)を維持する。

## 2. 実際に確認した固定Qiskit/P0 source

公式tag1.3.0に対応するcommitは **`896d1c8b72888a52a46081a6d1f4de8c56db8aea`**。
別versionのsourceを代用していない。保存Rust本文と公式固定commit、主要installed Python本文と公式固定commitの一致を確認した。
Source/wheelの再現build認証やlibmの全域誤差保証を取得したという意味ではない。
具体的なpath/hash/比較結果は[technical.json](../../artifacts/research_c_entry_e2_p0_feasibility/2026-11-08/technical/technical.json)に記録する。

| source | 調べた処理と結果 |
|---|---|
| [builtin_plugins.py](https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/qiskit/transpiler/preset_passmanagers/builtin_plugins.py#L535) | opt1はOptimize1q→InverseCancellation→basis条件付き再translation→Depth/Size FixedPointのDoWhile |
| [flow_controllers.py](https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/qiskit/passmanager/flow_controllers.py#L89) | 既定max_iteration=1000。1000回目の後も条件がtrueならPassManagerError |
| [optimize_1q_decomposition.py](https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/qiskit/transpiler/passes/optimization/optimize_1q_decomposition.py#L216) | 実runはRustを呼ぶ。Pythonの未使用helperだけから数値挙動を推定しない |
| [Euler Rust](https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/crates/accelerate/src/euler_one_qubit_decomposer.rs#L1076) | collect_1q_runs、StandardGate matrix、左積fold、ZYZ→U1X params、ZSXX、accepted置換とDAG phase加算 |
| [gate_matrix.rs](https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/crates/circuit/src/gate_matrix.rs#L314) | RZはcomplex exp(±iθ/2)、SX/Xはdyadic。数学的RZとRust評価matrixは別 |
| [basis_search.rs](https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/crates/accelerate/src/basis/basis_translator/basis_search.rs#L96) | native cost1、rule costは加法。Uの5-node witnessを使う構文上限の根拠 |
| [inverse_cancellation.rs](https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/crates/accelerate/src/inverse_cancellation.rs) | gate_eqはparam比較を持つが、当P0の登録cancel群はparameter-free。任意RZを近似削除する経路ではない |
| FixedPoint、UnrollCustomDefinitions、BasisTranslator、HLS、UnitarySynthesis | 固定点は連続2回の整数depth/size一致。Official定義をbasis+uへ展開。今回grammarにopaque unitary/HLS plugin/control flowはないという適用前提 |
| QuantumCircuit.inverse、RZ/SX/X/CX gate inverse、phase setter | actual gate列を逆順にして各gate inverse、global phaseを反転。Phase normalizationの誤差は別に残る |

Repo側は`pr_dictionary_validation.gates.compiled_wrapper`の空EquivalenceLibraryによる事前unrollと
`c1_h6_compiler_precision.compiled_wrapper`のMPFR marker/bind/castを確認した。
MarkerはRust内部phaseや角度を高精度へ変更しない。最終marker係数/未束縛parameter/layout検査にも例外経路がある。
新しいsource管理基盤は作らず、既存の保存Rustと固定公式URL、installed source hashを用いた。

## 3. Trajectory targetを変えない誤差連鎖

全比較でword w、符号・atom順・angle bits・head順・scalar/identity・同じcandidateを固定する。
以下の「厳密」は数学的gate/CAR定義の意味であり、その演算子を今回数値生成したという意味ではない。

| 記号/段階 | 空間・定義 | exactness/位相/逆の条件 |
|---|---|---|
| 抽象saved-atom仕様 | Canonical CARによるZ_a Z_b、signed係数、frame、K0 angleを指定するschema | Frameを厳密なcanonical rotationと解釈するには厳密unitary Vが必要。保存Uから一意にそのVが決まるとは扱わない |
| L_j：literal atom | 保存binary64 Uをexact dyadic数とし、D_a=f†(u_a)f(u_a)、Z_a=I−2D_a、P=±Z_a Z_b、E=cI−isP | U*U=Iを仮定しない。Pの順序を保持。非Hermitian/非involutory/nonunitaryの可能性あり |
| T_L(w)：literal trajectory | 同じhead/scalarとordered16 event、full256のliteral CAR map、controlled diag(I,T_L) | Sector36はその制限。Meanの数学的referenceとのbridgeは別脚。Headのnorm/Hermiticityも要入力 |
| R_math,m / R_num,m | Exact-dyadic saved-generator finite-RTE数学referenceと保存binary64 corrected reference/attenuation | Sector36の旧B2はこの二脚のみ。新E2のB/Uはnull |
| A(w)：raw gate列の厳密意味 | 実builderが出力するfloat角をexact realとして解釈したGaussian/controlled diagonal/phaseの全gate列、full512 | 各gateがunitary。Uに近い別polar matrixへDFを変更する定義ではない。raw numeric Operator評価誤差とは別 |
| C(w)：compiled gate列の厳密意味 | 固定P0出力rz/sx/x/cxとdeclared phaseの数学的演算子、full512 | 正常出力と有限angle/phaseが先に必要。通常global phaseもphase-sensitive normで保持 |
| A_minus/C_minus | **同じ実現word**から実際に構築する.inverse() gate列の厳密意味 | C_minus=C†を無条件に代入しない。Gate inverseとphase setterの誤差εinvを評価する |
| F_axis(w_1,…,w_n) | Ancilla head H、n=2^m個のwhole kernel、X/Y tail、状態準備を除く測定回路 | 各occurrence freshかsame-word repeatかを区別。全出力leakage・full512を保持 |

したがってforwardの比較は

`||C(w) − controlled(T_L(w))|| <= eps_P0(w) + eps_raw_literal(w)`、
`eps_P0(w) >= ||C(w)−A(w)||`、`eps_raw_literal(w) >= ||A(w)−controlled(T_L(w))||`。

**Compiler composerはunitaryなC/A脚にのみ使う。** Literal脚にはnorm growthを用い、距離2のcapを当てない。
Actual inverseについては

`||C_minus−controlled(T_L)†|| <= eps_inv + eps_P0 + eps_raw_literal`。

ここでadjointのnorm不変性を使うが、`T_L†=T_L^{-1}`とは主張しない。
Actual physical circuitのinverse-returnと、nonunitary referenceの`T_L† T_L`をidentityへ置くことは別である。
Control1のscalar位相は測定可能なrelative phase。Gaussianのvacuum global phaseはsandwichで相殺するものと
head relative transitionの解析補正を区別し、phase fitをしない。

### 有限primitiveの評価を全wordへ使える範囲

Raw builderのbasis9・headのdirected relative4等は固定有限入力で、全wordの数値再構成なしでも
「同じprimitiveに対する上界を全出現に適用する」経路がある。Tailは同一basisでのみsharing、head内relative融合、
head/tail跨ぎrelative融合なし。これは**raw→literal脚**の構造であり、P0出力のeventwise分解ではない。

例えばnative Gaussianのexact single-particle map V_gとliteral Uの距離d、Gram defect εを得られれば、
CARの`||f(v)||=||v||`より各density差は`(1+sqrt(1+ε))d`以下。
Z差はその2倍、ordered pair差はさらに`1+ν`倍（ν=1+2ε）という同じtargetの摂動評価を使える。
Actual emitterのtransfer評価、headの数値固有分解、relative積、phase・angle生成丸めを含む必要があり、
保存single-particle max-entry allowanceだけをそのままfull-Fock certificateとはしない。
どのprimitive入力も今回decode/build/evaluateしていないので新上界値はnull。

## 4. Literal Uをunitaryとしないnorm-growthとIID比較

以下は全Fock CARの解析的十分条件で、E2保存UのGram誤差を新計算した結果ではない。
Spin liftしたVの列をv_aとし、`ε >= ||V†V−I||_2`。

`D_a=f†(v_a)f(v_a)`なら`D_a²=||v_a||²D_a`、`||D_a||=||v_a||²`。
したがって

`ν = 1+2ε >= ||Z_a||`、`d_Z=4ε(1+ε) >= ||Z_a²−I||`。

異なるa,bのordered pair `P=±Z_a Z_b`について

`d_P=d_Z(1+ν²) >= ||P†P−I||`、
`h_P=8ε(1+ε) >= ||P−P†||`。

後式は`[Z_a,Z_b]=4[D_a,D_b]`とCARから
`||[D_a,D_b]|| <= 2 |<v_a,v_b>| ||v_a||||v_b|| <=2ε(1+ε)`。
符号はnormを変えない。このPをHermitianと仮定する必要がない。

`E=cI−isP`、実数c=cosθ、s=sinθなら

`E†E−I = s²(P†P−I) + i c s(P†−P)`。

`||E|| <= sqrt(1+s²d_P+|cs|h_P) <= β := 1+(s²d_P+|cs|h_P)/2`。
Angleを未評価のままなら|s|,|c|≤1の保守的値を使える。ただし今回ε自体がnullなのでβもnull。
Norm-growthが大きすぎれば形式的には正しくてもRPEには使えない。

同じliteral trajectory mapをp/q両方が使い、両法則でevent・coherent occurrenceが独立で、
各kernelのhead norm積がh≥1、event normがβ≥1、event TVがvなら、N_e=16·2^m、n=2^mとして

`| E_p z_L − E_q z_L | <= 2 min(1,N_e v) h^n β^N_e`。

証明はIID couplingで一つでもeventが違う確率≤min(1,N_e v)、各wordのmatrix element絶対値≤h^nβ^N_e。
Mismatch時の二つの絶対値の和を使う。**Nonunitaryなので全体を2でcapしない**。
h=β=1なら従来の`min(2,2N_e v)`へ戻る。実PCG64のjoint lawをIIDと証明する式ではない。
新helperは大きなexact powerを展開せず、prefactor/head/eventの因数形を返す。

別の方法は同じraw gate-map上でp/qを比較し、そのunitary couplingと
raw-mean→literal-meanのgapを一度だけ加えること。どちらを採っても比較targetと誤差partitionを揃え、
raw→literal項と既存bridge項へ同じgapを二重に含めない。
この二経路の選択・新数値評価を今回正式採択した扱いにはしない。

## 5. 合法grammar・P0の終了性で具体化できたこと

### 構文とnode上限

Tail F2..F7、8 modeのZZ support最大6×28=168位置。実非zero catalog/確率は新生成していない。
K0は16 event、各eventが一つのsigned ZZ rotation。Basis runsは1..16、tail Gaussian最大32出現。
各half-headは3 diagonalと、先頭/末尾basis＋2 relative Gaussian。両head計8 Gaussian、合計最大40。
Head diagonalは計6、scalar/identity Pは既存zero-time head P等を含め保守的10とする。
Actual inverseはこの出力gate列を反転する別対象で、同じK0 wordを再抽選しない。

既存official symbolic templateを再利用したSOURCE条件付き上限：

- Gaussian一出現：RZ≤8、XXPlusYY≤28。
- XXPlusYY：u7/rz6/cx2/sx1。u→nativeは3RZ+2SXの5-node witness以下。
- Controlled RZ：rz2/cx2。Controlled RZZ：u18/rz2/cx14。P：u1。
- K0 central16回：RZ≤2/CX≤4を各eventへ保守的計上。

よってN_1q≤62,994、N_CX≤4,752、**N≤67,746**。
これは構文上限の新しい整数評価で、実H4回路生成・gate価格測定ではない。
Official definitionsの全parameter適用、固定equivalence graph/native cost1/5-node witness、固定emitter grammarが前提。
旧r16/s4/h1/40の67,731や旧r32の107,522を流用したものではない。

### Accepted rewriteの証明と、正常終了に足りない条件

Native basis閉包、target=None、error_mapなし、ZSXXのみ、calibrationなしなら、Rustのold_errorは(len,len)、
new_errorも(len_new,len_new)。Acceptedはbasis外強制・strict shorter・identity扱いのいずれか。
Basis内ではstrict shorterに帰着し、identity扱いもlen_new=0で短縮する。
ZSXX出力≤5、取消は削除のみ。したがって既存`monotone_budget`の

`accepted rewrites R<=N`、`accepted visits ΣL<=N+5R<=6N`、`max L<=N`

が条件付きで使える。Rewritten RZの再入力もこの会計に含む。Rejected candidateはgate列を変えないので誤差を加算しない。
CX削除で後のiterationに別runが結合するため、初回run partitionだけに固定してはいけない。

しかしこれは**1000回以内の正常返却**の証明ではない。変化があるiterationでsizeが最低1減るというだけでは、
無guardでの保守的上限はN+1=67,747で、1000より大きい。
実際のcontrollerを使う人工mockでは、strict decreaseする1001段の処理でも1000回guardで例外になる。
この例は推論の不足を示すものであり、E2 wordで1001回かかる反例を発見したという意味ではない。

| 正常終了の必要条件 | 到達点 |
|---|---|
| Unroll/translationが有限・全gateに定義あり | 通常grammarのofficial有限定義とbasis reachabilityはsourceで追える。Opaque/HLS/別targetを追加すれば適用外 |
| 各run処理がfinite値を返す | Matrix取得unwrap、DAG不変量、arg/abs/atan2/mod、phase式の有限性の全domain証明は未完 |
| Optimizationが1000回以内にFixedPoint | Strict size decreaseは成立条件を明示できたが、それだけでは不足。E2 grammar固有のround縮減証明が未取得 |
| 通常gate列を返す | Finite angles/phases、basis閉包、marker係数1/未束縛parameterなし、layout同一の全word条件が必要 |
| 資源・一般例外 | OOM等を数学的回路保証へ混ぜない。許容入力上の数値/logic例外の排除を別に要する |

当sourceではactive ZSXXはsqrt(det)を使わず、det.arg、二成分absのatan2、各entry.argを使う。
他Euler basisのsqrt/divisionを当P0のblockerと誤記しない。det非zeroでもc/dが0近傍になることはあり、
atan2(0,0)・arg(0)の規約とそのphase-sensitive影響を評価する必要がある。

## 6. 既存composerを確認し、厳密Euler部分を縮めた

### 既存式の成立前提

`ε_P0 <= ε_translation + 6N p/(1−Np) + N ζ(D_N) + ε_phase`、`Np<1`。

`p=(1+a)(1+16γ3)−1+28h0(1+u)^3`、`u=2^-53`、`γ3=3u/(1−3u)`、`h0=2^-1075`。
aはnative nodeの**operator norm**による一様matrix評価誤差。Entry residualや数ULP観測ではない。
SX/XはdyadicだがRZのcomplex expと、half angle subnormal等のaは未取得。
逐次matmulは実Rustの`other @ operator`、各complex entryの4実積と3加算のpathに対応する既存保守的rateを再利用した。
IEEE RN/gradual underflow/finite intermediates/no reassociation/separateまたは正しく丸めたFMAというモデルは条件であり、
installed wheel全体について今回証明したわけではない。

`η_L=Lp/(1−Lp)`ならrunのMは、厳密なRZ/SX/X product Gに対して`||M−G||<=η_L`。
Gのdetは{1,i,−1,−i}。det Mは各coset周辺、`||M†M−I||<=2η_L+η_L²`のtubeに入る。
Gは**現在のgate列**なので、以前のrewriteが出した角度も含む。Global DAG phaseは別会計。
これは個別Euler角の小領域や有限captured boxではなく、全run長と全wordに対する条件付き領域である。

ζは同じD_N全体に対する`M→実際に出力したphase付きV`。
全branches、atan2/arg cuts、θ≈0/π/2/π、RZ省略/SX/X選択、mod端点を覆う必要がある。
Translationは初期unroll/parameter/official rule、phaseは各DAG update/normalizationとMPFR bind/castの未包含分。
ζ内のlocal phase丸めをphase項へもう一度入れない。

### 新lemma：角度を連続と仮定せずexact Euler mapを評価

M=[[a,b],[c,d]]、あるunitary Gに対して`||M−G||<=η<1/4`とする。
r=sqrt(|a|²+|c|²)、t=phase(det M)、u_c=phase(c)、u_d=phase(d)。
0のphaseは任意の単位複素数でよい。r≥1−η>0、det M≠0。
Rustの`params_zyz_inner`の**厳密算術**版を通常ZYZ意味へ戻したmapは

```
F(M) = [[ (|a|/r) t/u_d,           -( |c|/r) t/u_c           ],
        [ (|c|/r) u_c,              ( |a|/r) u_d              ]]
```

ここで単位phasorの逆数はその共役、すなわち`1/u=conj(u)`である。
F(M)はunitaryでphase-sensitive（phase fitなし）。θ、φ、λ個別のarg cutを跨いでもこの式は適用できる。

証明のため、first column u=(a,c)/r、v=(-conj(c),conj(a))/r、Q=[u,tv]を置く。
Second column x=(b,d)=αu+βv。`|α|<=κ=(2η+η²)/(1−η)`。
`|r−1|<=η`、`||β|−1|<=η`（QR residual長はσmin M以上かつcolumn norm以下）。
よって`||M−Q||<=2η+κ`。Qのd0=t conj(a)/rに対し`|d−d0|<=η+κ`。
Weighted phase inequality `|A phase(d)−d0|<=2|d−d0|`（A=|d0|）はd=0でも成立する。
F−Qで変わるのは対角2成分で同じ絶対値なので`||F−Q||<=2(η+κ)`。
したがって

`||F(M)−M|| <=4η+3κ <= B(η):=6η/(1−η)+8η+2η²`。

最後の緩い式はη<1/4で成立し、unitary Fと`||M||<=1+η`から2+ηとのminも使える。
これで**near-unitaryからexact Euler再構成まで**は一様に抑えられる。
`Np<1`に加え、このlemmaには`η_N<1/4`、すなわち`Np<1/5`が必要。

既存ΣL≤6Nを使えば、`k(η_N)=6/(1−η_N)+8+2η_N`として

`Σaccepted [η_L+B(η_L)] <= (1+k(η_N)) 6Np/(1−Np)`。

これは既存product＋Nζをそのまま重ねる追加項ではなく、ζをexact再構成と実f64差へ**分割する場合の代替会計**。
ζ全体を得た場合にはこの部分を再加算しない。新helperは解析的部分だけ返し、whole_P0_boundはnullのまま。

### なお必要な実Rust部分

実際の出力V_f64とexact F(M)の距離ωが残る。`ζ<=B(η_N)+ω`という経路はあるが、ωを得ていない。

- det/abs/arg/atan2の丸め・libm model。Arg角の絶対誤差のLipschitz定数は0近傍で発散するので、
  上記のような**振幅付きphasor/operator評価**が必要。
- `circuit_psx_gen`のzero SX/one SX/general、π近傍の書換え、flip、X選択、RZ drop、mod端点。
  理想恒等式の確認と、f64 threshold/parameter arithmeticの全branch誤差は別。
- `ANGLE_ZERO_EPSILON=1e-12`はsimplification判定値。ζそのものに代入しない。
- Determinant phaseとglobal phaseは普通の角度moduloでは落とせない。P gateの2πとRZの2πの違いを保つ。
- Local f64 phase、DAGのrem_euclid、MPFR markerへの加算、finalbind/cast/setterのどこを各項へ入れるかを固定する。
  MPFRという名前だけで全pathを正確としない。

上記は既存P0を維持した追加静的解析で閉じる**可能性**がある。
ωとguard内正常終了の全domain根拠が得られたとはいえず、単にH4を数値評価して定数を埋める段階でもない。
全compiler/PRで不可能という結論ではない。

## 7. Full512・13roundへの接続と有用性

C/A unitary compiler脚の小さなbound、raw→literal脚、実inverse、head/X/Y shellの各上界がそろえば、
既存`c1_h4_full_synthesis.propagate`の両側norm-growthや`ColumnStream/repeat_bound`のscope条件へ接続できる。
Kernelのdeterministic両half-headはkernel内、shells['head']はancilla H準備。二重計上しない。
Native全512をsector72列へ暗黙縮小せず、縮小を選ぶならtarget不変性と全leakage出力を別に示す。

13round packetのsynthesis/categorical/reference bridge/reference uncertainty/actual streamは未取得のまま。
Raw complex単位の追加項にattenuationやr2^mをもう一度掛けない。
PF/corrected RTEは既存A_m倍、DFはenergy段階だけ、RPE統計は別枠。
Literal/reference二脚の旧B2包囲からnative trajectoryのunitarityや全wordを推論しない。

最低限の有用性条件は`δ_m/A_m < sin(βtotal−βstat)=sin(π/12)`のような全round位相余裕。
これはsynthesisだけが小さいという判定ではない。A_m、PF/RTE等を含め、実際の保存policyの条件で検証する必要がある。
今回保存値を新たに再解析せず、RPE PASS/保証shots/保証付き75,621,395,471 RZは表示しない。
IID-gridの数値十分条件が成立してもactual PCG64 joint stream・actual CAは自動的に解決しない。

## 8. 最小実装・人工検証

追加は[c_entry_e2_p0_feasibility.py](../../src/trottertracks/c_entry_e2_p0_feasibility.py)と
[人工test](../../tests/management/test_research_c_entry_e2_p0_feasibility.py)。科学runnerは作らない。
既存`c_pr_aware_df_c2a.monotone_budget/symbolic_templates/compiler_coverage/trace_artificial`、
`c_pr_aware_df_zsxx_uniform.run_bound/product_rate/uniform_whole_bound`、前段candidate bindingを呼ぶ。
これら既存moduleは旧固定3eccab64版とbyte同一で、共有sourceを変更していない。
新規部分は上記二つのscalar lemma・factored coupling・その最小routingで、compiler/Arb/RPEの再実装ではない。

人工・既存回帰 **187件PASS（新46＋既存141、前段106件を含む）**。
Poles/cosets/relative phase、非直交literal Uと非Hermitian ZZ、norm-growth、実controller guard、target/space/coverage/inverse誤用、
double counting、null保持、supplied schemaからのRPE誤昇格拒否、cross-event 1Q run、actual phase/inverse/X-Yを確認した。
全domainの根拠は上記解析と未解決条件であり、人工sampleの最大値を証明へ昇格していない。

Arb/flint importを遮断した選択suiteのみ実行。最初のguard実行はSymPyの任意backend探索も捕捉し170PASS/17停止したため、
自分のtest processだけ`SYMPY_GROUND_TYPES=python`で再確認した。新Arb計算は0、共有venv/設定変更なし。
5 warningsは旧Almost_optimal_grouping.pyのescape SyntaxWarning。Science/source修復による再実行ではない。
[technical.json](../../artifacts/research_c_entry_e2_p0_feasibility/2026-11-08/technical/technical.json)にtests/source/structural値/STOPを集約し、
管理ファイルを増やさない。Local人工testであり、独立レビュー・immutable CI・実H4保証ではない。

## 9. 次の最小作業・科学実行へ進める境界

**次の担当はCodexの追加静的解析、必要な保証前提・別政策を選ぶ判断はGPT/ユーザー。**
現P0継続なら、優先度は(1)E2 grammarで1000回以内となるiteration縮減の証明または有限abstract-state証明、
(2)上記F(M)に対する実Rust ωの全branch/wrapped phaseを含むoperator誤差モデル、の二つ。
角度の連続性を仮定した方法ではなく、小振幅側を重み付きで処理することに情報価値がある。
新H4 wordを増やす実行や、旧Arb384/固定word再実行を次の最小作業とはしない。

| 段階 | 今回閉じた部分 | 進める前提・未取得入力 |
|---|---|---|
| 構造的方法 | Gate-list compiler脚とliteral脚の分割、Gram/norm-growth、exact Euler lemma、node/visit上限 | 1000回内成功・全f64 image/phase modelは未成立 |
| E2固有の数値入力 | 保存hash/source/ground/κ/scalar/結果bindingを再利用 | a、ω、translation、phase、Gram defect、raw/literal/head/inverse/shell boundはnull |
| Reference/IID | 同じliteral targetならnonunitary couplingの経路あり | 今回p/q table、head norm、B/U、保存kernel等の入力取得とSOURCE固定が必要 |
| Actual/RPE | 既存誤差分割を保持 | PCG64 joint stream、全13roundの有用性・保証水準は別残件 |

保証方法が未成立なので、新科学entry/family/call上限は**null/なし**。
最小科学契約を完成したとはしない。方法成立後に限り、同じE2一候補/固定r16・13roundを対象に、
保存bytesで再利用できるprimitiveを先に照合し、新評価が必要なfinite primitive・一様定数・reference二脚だけをSOURCEへ固定する。
入力欠損/target差/未包囲branch/phase drift/guard/数値失敗でSTOP、固有fresh outputへone-shot、retry/resumeなし。
その時の計算domainとinstrumentationが決まる前に旧38/27/86callを転用しない。

Eventwise fence/barrier・別synthesisを将来選ぶならP0内部の全word依存を弱め得るが、別compiler政策になる。
現在のcompiled価格がどれだけ変わるかは未計算で、変化率を捏造しない。
保存総RZの数値を新compilerの価格として使えず、再校正が必要になる。
今回はcompilerもpolar/exact-H等の正式targetも変更しない。

## 10. 保全・公開・STOP

独立branch `codex/research-c-entry-e2-r16-p0-uniform-feasibility-20261108`、
worktree `/tmp/research-c-entry-e2-r16-p0-feasibility-20261108`。
正式共通指示の§8.1/8.2/8.4/10/10.3とAGENTSを実際に確認した。
旧SOURCE/input/raw/結果/契約/grant/ledger/marker/STOP、元dirty checkout、前段の106件/CASE B記録を保持する。
追加内容は技術進展CASE Cであり、前段を「解決済み」に書き換えない。

新科学計算・保存科学値の数値再解析・H4 build/compile・Arb包囲・RPE/価格・retry/resume・重要レビューは0。
概要・dated note・manifest・入口を更新し、commit/公開後にremote SHA/tree/本文/主要SOURCEを照合する。
**Mandatory STOP。次の科学判断・新科学実行を自動開始しない。**
