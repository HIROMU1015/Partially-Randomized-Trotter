# A-1：H6純RTE／Hybridの全law Native接続誤差に関する限定技術調査

2026-11-02 JST。直接の基点は[A調査][A]のcommit `0e1c7aae386986f6c1bb9483548671655447f516`。
独立worktree `h6-native-bridge-scalar-phase-audit-20261102`、branch `codex/h6-native-bridge-scalar-phase-audit-20261102`で実施した。

**保持角度項は、確認したoperand域とbinary64の丸め条件の下で、任意trajectoryに共通の式へ整理できる。H32/H64で共有し、R64では空和となる。**
**位相側は更新順序と有限回数を特定できたが、actual remainder・windingの誤差を閉じる根拠が不足しており、全law Native誤差の数値上界は未取得である。**
既存Aの未確定という結論を変更せず、局所engineering結果を外挿しない。Compiler誤差や価格tailは本調査の対象外。

## 1. 固定対象・許可範囲・来歴

linear H6、隣接1.00 Å、STO-3G、中性singlet。保存Pauli-JW918非identity項＋constant、system12/full4096。
T=.8/S2/q8/K2、P=0、central-rotation control、Full13qubit/ancilla wire12、cosine／+Im sine。
保存raw stateと既存exact νを維持。保存compiler条件はQiskit1.3.0、opt1、seed17、rz/sx/x/cx、追加backend/coupling/layout/routing指定なし。DF rankは適用外。

| ID | Candidate | L_D | r | event位置数 |
|---|---|---:|---:|---:|
| R64 | H6_p000_q8_r64_K2 | 0 | 64 | 512 |
| H32 | PR_p0078_q8_r32_K2 | 78 | 32 | 256 |
| H64 | PR_p0078_q8_r64_K2 | 78 | 64 | 512 |

K2のsupportはorder0/2だけ。保存lawの53-bit実効CDFとbinary64 event角度を使う前提を維持し、名目lawへ戻さない。
新しい角度差・位相差・上界・確率・bias・shot・N/G・価格の数値評価、状態/行列作用、sampling/RNG replay、QPY decode/再生成、compile、人工数値testsはすべて0。
SOURCEのテキスト/AST、保存metadataのhash・値の一致・sign/exponent bit分類、既存数理関係の適用条件だけを調べた。
新runner・library・誤差framework・compiler patch・承認証明機構は作成していない。独立詳細科学レビューや正式証明の完成とは呼ばない。

| 資料 | 固定版と利用範囲 |
|---|---|
| A調査 | [全law転送の既存調査][A]、上記直接基点。NativeとCompilerの不足を切り分ける |
| 局所診断 | [REPORT][LOCAL]／[STOP][LOCALSTOP]、結果`71432728ee0140fb3c1f8bf30ecf51b2faba86ec`、科学SOURCE`9d392de7c8c6f8ae35faba9aeb1c8c6f6a215fef`。3 trajectory/15作用/6比較は引用のみ |
| 元H6証拠調査 | [調査書][INV]／[INVENTORY][INVJSON]、`b3bab573cf17b9dd8ddc1dca5f1a6a6c935ef6d1` |
| R64 | 元SOURCE`e755ecf581600192fd2745d7066e410cebd933d3`、結果`10cb3e5cca7834ebf75301bd91bd093ca51d4a6b` |
| H32/H64 | 元SOURCE`887b69e449cb3c378286f09d0d7626599850fb67`、結果`00958ae1945c780718b2deef07d5dab1096a5cd8` |
| 既存R2/R2-C | [転送設計](../non_df_h4_central_r2_design_20261025.md)／[限定証明](../non_df_h4_r2c_compiler_phase_proof_20261025.md)。記号的control恒等式、unitary積のtelescoping、phase windingの恒等式を再利用。R2-CのPROOF_INCOMPLETEは維持 |

主なコード参照：元Hybrid SOURCEの[Pauli append/trajectory][SRCAPP]、[central control][SRCD2]、[event生成][SRCEVENT]、[law exporter][SRCLAW]。
現在の[保持・phase descriptor](../../src/trotterlib/pauli_mechanism_cost.py)、[selected作用](../../src/trotterlib/h6_selected_implementation.py)、[target平均](../../src/trotterlib/pauli_vector_tight_enclosure.py)、
[保存Rust phase setter](../../artifacts/non_df_h4_r2c_compiler_phase/2026-10-25/upstream/circuit_data.rs)も読取した。これらの科学処理は呼び出していない。

添付指示の局所科学SOURCE文字列には`…aba9eb1…`と1文字欠落があった。[保存source_and_inputs.json][LOCALINPUT]と実在Git commitが一致する上表の`…aba9aeb1…`を参照した。別SOURCEへの差替えや再実行はしていない。

## 2. Target／Nativeの演算子規約

P_j=s_j σ_j、s_j∈{−1,+1}、σ_jは保存labelのHermitian Pauli wordとする。
event eはproductを発生順に作用し、rotationを最後に作用する。

\[
E_e=(-1)^{k/2}\exp(-i\widehat\phi_k P_{j_R})P_{j_k}\cdots P_{j_1},\qquad k\in\{0,2\}.
\]

`_make_event`は最初の抽選IDをrotation、残りをproductsとして保存し、`application_sequence=products→rotation`とする。
異なるPauli wordを交換・積約しない。Pauli同士の非可換積に含まれる位相を別のπ項として再追加しない。
各outer stepの適用順は保持forward→r events→保持reverse→保存scalar `exp(+i s)`。
sは`scalar_step_binary64`をexact dyadicと解釈した値で、q回適用する。raw stateは再正規化せず、νは共通の分母。

U_C(ω)はこの数学的selected-target unitary。
V(ω)は、**central controlが実際にemitしたCRZ/Pのbinary64 operandをexact実数と解釈したときの、control=1側の有効system block**である。
central化前のPauliEvolutionGate bodyと区別する。元bodyにexactなc×timeが格納されても、D2が倍角をfloatで形成する際に差が入り得るためである。
理想gate定義に対するこの意味論と、Statevector/libmが信号を数値評価するときの丸め誤差は別である。
局所`engineering_signal_uncertainty`をDの一様上界にしない。

## 3. 保持回転角のSOURCE対応

### 3.1 実際のoperand順序

| 段階 | SOURCE | operandの意味 |
|---|---|---|
| δ保存 | `h4_pauli_pr_search.law_descriptor` | `cfg.T/cfg.q`をbinary64として保存。固定3 lawは`step_time=0.1` |
| half-step | `PauliSignalAdapter.trajectory` | `self.step_time/2`を先に形成し、各保持PauliEvolutionGateのtimeへ渡す |
| c保存/復元 | `append_pauli_evolution`、Qiskit `SparsePauliOp.from_list/to_list`、D2 `_real` | real binary64→complex128の実部→real float。単一の符号なしlabel、虚部0という条件を保つ |
| central倍角 | D2 `_tokens` | `_real(2 * coefficient * time)`。Pythonの左結合なので**`(2*c)*time`**。`2*(c*time)`へ書き換えない |
| target保持角 | `pauli_vector_tight_enclosure.evaluate`、`h6_selected_implementation.selected_action` | exact(c)×exact(δ)/2。floatで先に積を丸めない |
| P2 descriptor | `pauli_mechanism_cost.source_tokens` | 同じ`float(2*c*t)`を復元する。回路生成/作用や全域誤差証明の代用ではない |

RNをbinary64 round-to-nearest, ties-to-even、exを保存floatのexact実数とすると、一般に

\[
h=\mathrm{RN}(\delta/2),\quad d_j=\mathrm{RN}(2c_j),\quad
\theta_j=\mathrm{RN}(d_jh),\quad
\alpha_j=\mathrm{ex}(c_j)\mathrm{ex}(\delta)/2,\quad
\beta_j=\mathrm{ex}(\theta_j)/2.
\]

half/doubleがexactで有限、積がnormal範囲、変換がbit-preservingであれば、θ_jは`RN(ex(c_j)ex(δ))`に一致する。
このとき、u=2^-53に対する既存の通常の乗算丸め式から
`|β_j−α_j| ≤ u |α_j|`を条件付きで使える。今回は保存cやδをこの式へ代入評価していない。
RN・非FMAの演算順・normal域を外す場合は、一般式と絶対underflow項へ戻す。全保持角度差を0とはしない。

### 3.2 保存metadataだけで確認した入力域

元model SHA256 `6c66a2eeb4d99cf5bbd9137bbcb067275536be3201d5cc047225266dc5f58681`が保存bindingと一致した。
元archiveから3 law memberだけを選択読取し、part/archive/member hashを照合した。QPYや価格標本は読んでいない。

| member | 保存SHA256 |
|---|---|
| H6/batch01/cases/PR_p0078_q8_r32_K2/spec.json | `8f7fdc7427588db16cb1f8ecd5800f2148e1122d1ab1a6923110bac5e0fff364` |
| H6/batch01/cases/PR_p0078_q8_r64_K2/spec.json | `c1df4c492df4f74c5bc9311bf107ea799da687b93f3b1ec01af1a1553bf8f578` |
| cases/H6_p000_q8_r64_K2/law.json | `5f83761e9633d7b854481e6a1f83bdc58f2d82b521c7a78269abdba65e8f715d` |

H32/H64のretained78 label・coefficientのbinary64 bitsはmodelの同じprefixと一致し、R64は空である。
78係数はすべてnonzero normal、符号を除くunbiased exponentが十分内側の帯`−100<e<100`にあることを**bit分類だけ**で確認した。
δと6個の保存event角度もnonzero normalで同じ帯にあり、δは正である。
この粗い入力域とδ=.1の既存operandから、通常のbinary64演算モデルではcの2倍、δの半分、両者の積はoverflow/subnormal/underflow域から離れる。
入力に±0、NaN、Infはない。これは角度差の算出ではなくoperandの型・域の技術確認であり、丸めモードの実行時認証ではない。
保持labelはI/Zだけだが、誤差式に可換性を仮定する必要はなく、保持回路を結合・再合成しない。

H32/H64はq・δ・78係数・演算順が共通なので、将来の保持差78個の評価とその保持項を共有できる。
各係数は各outer stepのforward/reverseで一回ずつ使われ、和はj=1,…,L_D、係数2q。
R64の保持項は構造的な空和である。ただしscalar sは存在し、Native bridge全体が0になるわけではない。

### 3.3 Event角度

targetは保存角度φhat_kを使い、nativeも`event.rotation_angle`とrotation componentの符号±1を使う。
`(2*(±1))*φhat_k`は上の有限normal域でexactな符号/2倍変換なので、制御CRZのhalf-angleは同じ±φhat_kに対応する。
保存角度から理想atanへの誤差を新規追加しない。6角度は候補別に異なり、H32/H64の保持項共有をevent角度の共有へ広げない。
実効lawと生成algorithmが同じ固定角度を使うことは元SOURCEで確認し、全trajectoryの角度を再生成して調べる必要はない。

## 4. Scalar／product／order位相の所有

保存3 lawのsは共通の正値`0.032484153606273984`、元constantは負値`−0.3248415360627398`。既存値を引用しただけで積・差を再計算していない。
targetはsを既に丸め済みのexact operandとして使う。真のHamiltonian constant×時間への近似誤差は既存target/biasの辺に属し、本bridgeへ再追加しない。

| 位相/作用 | Selected-target | Native元生成と制御 | 回数と依存性 |
|---|---|---|---|
| constant scalar | 各step `exp(+i s)` | `trajectory`末尾の`global_phase -= constant*step_time`。乗算→減算→setter | q回。共通sだが、その直前のgはevent列に依存 |
| 負product | 各signed P_jの−1 | product全wordをappend後、符号が負なら`global_phase += math.pi` | 負product一個につき一回。wire数やYの個数だけ重複加算しない |
| rotationの負符号 | exp内のsigned P_j | central角度のcoefficient±1 | 別のπ位相に移さない |
| order0 | +1 | `np.angle(complex(+1))`を加算してsetter | eventごと一回。0加算もsetter呼出から除外しない |
| order2 | −1 | `np.angle(complex(−1))`を加算してsetter | order2ごと一回。負product位相とは別 |
| Pauli product内部 | 順序を保った行列積 | controlled X/Y/Zを元の時間順・wireでemit | 非可換積を並べ替えたり、別のcommutation位相を追加したりしない |
| body集約g | 上記符号とscalarを数学的に持つ | D2 `_tokens`で取り出し、最後のancilla `P(g)`に一回だけ移す | bodyではglobal、control後はrelative phase |
| Full wrapper global phase | 測定observableに影響しない | central回路は上記P(g)をgateとして持ち、wrapperへcompose | P(g)をFull全体global phaseとして消去しない |

元`law_descriptor`のscalarは`float(-constant*step_time)`、元`trajectory`は`g -= constant*step_time`である。
有限normal域・対称RN・同じoperandという条件では、`−RN(constant*δ)=RN((−constant)*δ)=s`となり、scalar increment自体は共有できる。
このbindingが未確認の実装へ広げるなら、差`|a_scalar−s|`をq回の項として残す。
共通scalarだからevent列依存の逐次加算丸めまで相殺すると扱うことはできない。

`_make_event`の`complex((-1)**(k//2))`は実部±1・虚部+0の形で、NumPy1.26.4の`angle`は`arctan2(imag,real)`へ渡す。
したがって非負のendpoint +0/+πを返す通常のatan2規約が適用候補になる。
もし−1−0jへ入力規約が変われば−πになり得る。数学的な−1は同じでもmodulus定数差・加算順の評価を省略しない。
actual endpointの定数誤差は未評価で、SOURCEがatan2を呼ぶことだけでulp保証を付けない。
§3のnormal入力確認は、phase還元後の微小値・±0やその処理まで除外した証拠ではない。位相側のsigned zero・境界は別に保持する。

## 5. Python／Rust remainderとwinding

### 5.1 二つの経路を同一視しない

| 経路 | 実装上の処理 | ここで認定しないこと |
|---|---|---|
| 元Native | Pythonの`g += a`または`g -= constant*δ`→`QuantumCircuit.global_phase` setter→Rust `CircuitData.set_global_phase`の`angle.rem_euclid(2.*PI)` | Python `%`で復元したgとの全域bit一致や誤差0 |
| P2 descriptor | `phase=(phase+float(value))%(2*math.pi)` | 3 selected QPYの一致を全trajectoryへ外挿すること |
| CPython float `%` | 3.11.0rc1 `float_rem`はC `fmod`後に除数と符号を合わせ、zeroは`copysign(0, divisor)` | C fmodの実platform誤差上界、Rustとの同一backend |

元Qiskit1.3.0 Rust setterは保存SOURCEのblob `63368f240e5ffbc1f5c92809e7c74bcf2711f120`と照合した。
CPython側は保存runtime版に対応する[公式3.11.0rc1 SOURCE][CPY]を読取した（blob `47d308b8eb3702b03d760509a07d53452e96b2c7`）。
`%`で重要なのは`float_rem`であり、`divmod`の商を補正する処理を誤差評価へ混ぜない。
[Pythonの浮動remainder説明][PYREM]と[Rustのrem_euclid説明][RSREM]にも、負の微小入力では丸め後にmodulusそのものになり得るという注意がある。
Rustの現行説明はhistorical wheelを生成したRust/libmの誤差証明ではなく、今回その対応を新しく認定していない。

Nativeのmodulus p_RはRustのbinary64 π定数を2倍したoperand、P2のp_PはPython `math.pi`の2倍である。
いずれも数学的2πとは区別する。同じ十進表示やSOURCEの式の類似だけでp_R=p_Pやp_R=2πとしない。
以降p_fはNativeのactual modulusをexact実数と解釈した値。

### 5.2 固定H6で狭められる入力域と、残る前提

保存s>0、product更新は+π_f、order更新は通常のendpoint規約下で0または+π_f。
よって、gが非負のremainder域に戻ること、両π定数の正値性/大きさ、finite RNが成り立てば、更新前後の入力yも非負である。
一般の負scalar入力や任意の負angle更新まで調べる必要はない。負の微小入力でのmodulus端点例はR2-Cの一般的注意として保全する。
これは非負域だけでactual remainderの誤差を証明したという意味ではない。

さらに`0≤g≤p_f`かつ`0≤a≤π_f<p_f`と加算丸めの余裕を確認できれば、`0≤y<2p_f`へ限定できる。
数学的な還元整数はn∈{0,1}となる。境界g=p_fも排除せず、y=p_fではn=1を含める。
yがp_f近傍を跨ぐと保存gはほぼp_fからほぼ0へ跳ぶため、代表値の差`|g−g'|`だけで誤差を測らない。
π_fと数学的πの差、windingの違いを残した円周上の距離で評価する。

十分条件の一例として、上記域でactual remainderが正しいfmod/reductionを返すことが示せれば、
y<p_fではy自体、p_f≤y<2p_fではSterbenz条件下のexact差y−p_fが表現可能となる。
ただし**現在のSOURCEだけでは、そのactual remainder条件の誤差定理との接続が閉じていない**。
R_t=0、1ulp、任意のγ係数を実測なし・証明なしで採用しない。scalar点数の追加評価だけでも全域のこの不足は埋まらない。

### 5.3 再利用するwinding恒等式

R2-Cの記号を維持し、tを**実際のNative phase更新順**とする。
g_tは保存floatをexact実数とした値、a_tは実際の増分、a_t*は対応する理想増分（s、π、0）。
y_tは加減算の丸め後operand。整数n_tを取り、

\[
A_t=y_t-(g_{t-1}+a_t),\quad R_t=g_t-(y_t-n_t p_f),\quad
h_t=h_{t-1}+a_t^*-n_t2\pi
\]

と定義すれば、

\[
g_t-h_t=(g_{t-1}-h_{t-1})+(a_t-a_t^*)+A_t+R_t-n_t(p_f-2\pi)
\]

はexactな恒等式である。n_tの選択とR_tの評価を整合させ、NativeとPython descriptorで同じnを強制しない。
initial phaseがexact0、phase取得/転送で値が変わらないという条件の下で、

\[
d_{phase}(\omega)=|e^{ig_T}-e^{ig_*(\omega)}|
\le\min\left(2,\sum_{t=1}^{T(\omega)}\left[|a_t-a_t^*|+|A_t|+|R_t|+|n_t|\,|p_f-2\pi|\right]\right).
\]

ここでg_*(ω)=q s+π(N_-(ω)+N_2(ω))。N_-は負productの個数、N_2はorder2の個数。
本式のg_*はexact sumであり、floatで`q*s`を新計算した値ではない。initial/transferが変わる実装にはその誤差も別に必要。
g取得の`_real`とancilla `P(g)`は今回のreal float規約では値の転送であり、追加のπやscalar適用ではない。

K2ではN_-≤2N_2、N_2≤qrなので、N_-+N_2≤3qr。
元SOURCEの加減算更新回数は、負product N_-回＋全event qr回＋scalar q回、したがって`T(ω)≤3qr+q`。
これは命令構造からの有限回数で、新しい確率・誤差数値・価格上界ではない。
constructor等の初期0設定はinitial条件で扱い、0のevent更新を回数から省かない。
全経路で使えるa/y/gの域、加算残差、remainder残差、整数windingを保証できれば、この有限和をuniform化できる。
それらが未確定の現段階ではd_phaseの数値上界を出さない。

## 6. 制御意味論・actual adjoint・Full測定

既存R2/R2-Cのcontrol恒等式はsystem qubit数や特定の標本に依存しない。
D2 `_rotation`はXにH、YにSdg→H、active wireのparity CX、central CRZ、parityの逆順、basisの逆を明示emitする。
H/CXは自己adjoint、Yのbasis undoはH→Sである。数値的に逆行列を再合成せず、理想gate定義ではactual inverseとなる。
これによりcontrol=0ではI、control=1では`exp(−i θ_j σ_j/2)`となる。
Little endianの`reversed(label)`と保存wire順を保つ。

controlled productは同じancillaをcontrolとするCX/CY/CZの時間順積である。
保持reverseは**同じ符号の角度を保持順の逆に並べたS2後半**で、forward保持unitaryのadjointではない。
eventの順序・角度を逆転してinverseと扱う操作もない。P3 Gaussianのadjoint検証をここへ持ち込まない。

数学的なgate対応、basis/parityのactual inverse、±符号の所有はSOURCEと既存代数で確認できる。
一方、binary64 angle bindingとphase setterがtargetのexact operandへどの誤差で接続するかは§3/5の条件が必要。
保存3 trajectoryのQPY整合はこの全域の数値条件を代替しない。

Native Fullは同じcontrol blockに対し、cosineはH→control→H、sineはH→control→Sdg→H。
測定はancilla Z、後者が+Imとなる。m_C,aは`ψ†U_Cψ/ν`のReまたはIm、m_native,aは同じraw state/νに対応する期待値。
`|m_native,a−m_C,a|≤D(ω)`の係数は1。B_savedは全law信号へ転送するとき一回だけ使う。
bodyのglobal phaseはcontrol後のrelative phaseとして観測される。Full全体のglobal phaseだけはobservableに影響しない。
事後phase alignmentは行わない。

## 7. 全lawへ適用できる条件付き式

§3のevent角度一致と§6の制御意味論が成立するとき、unitary積のtelescopingから既存式

\[
D(\omega)=\|V(\omega)-U_C(\omega)\|_2
\le\min\left(2,2q\sum_{j=1}^{L_D}|\beta_j-\alpha_j|+d_{phase}(\omega)\right)
\]

を使える。保持角度のoperator誤差はnorm不変な積の前後に挿入されるため、event列や保持との可換性に依存しない。
Hybridの共通保持上界をR_retとし、phaseが全ωでd_phase≤d_*と保証できれば、

\[
\rho_{native,a}=\left|B_{saved}\mathbb E_p[m_{native,a}-m_{C,a}]\right|
\le B_{saved}\mathbb E_p[D(\omega)]
\le B_{saved}\min(2,R_{ret}+d_*).
\]

R64ではR_ret=0。H32/H64ではR_retを共有できるが、r・event角度・B・位相更新列は同一ではない。
phaseはr別のuniform envelopeにできても、候補別lawとBを無断で同一視しない。
uniform式自体は確率の再計算を要さず、law-weighted化する場合だけ保存実効質量/IID仮定を正しく接続する。
scalar、負product、order位相を§5のd_phaseへ所有させた場合、それらを別のevent誤差として二重加算しない。
既存bias/targetの数値allowanceも新Native誤差の代用品や追加のscalar誤差にはしない。

## 8. 要因別の到達点

| 誤差要因 | 既存の数学的根拠 | SOURCEで確認可能な事項 | 未解決条件 | 次の作業 |
|---|---|---|---|---|
| 保持角度形成 | Pauli回転のLipschitz差、unitary積telescoping、normal乗算RN | `(2*c)*half`の順、共通prefix78、q/δ、operand bit域 | 実丸めモデルの適用、78差の数値値は未取得 | 必要時に共通78 pairだけexact有理数で別承認評価 |
| Event角度形成 | ±符号と2倍のexact変換、CRZ half-angle | 既存binary64角度、k=0/2、normal域、同じ生成式 | 別runtime/異なるoperandには再適用不可 | 今回新角度差は不要。既存bindingを維持 |
| Scalar位相 | exp(+is)のq回積、RNの符号対称性 | 元はproduct後の減算、保存s共通正値、targetもs所有 | actual scalar binding/加算残差の上界 | 必要時に共通scalar binding1組を別承認確認。full-H誤差へ戻さない |
| Product/order位相 | signed Pauli積、(−1)^(k/2) | product後π、event後np.angle、時間順、K2回数 | atan2 endpoint定数の実装根拠/誤差値 | endpointをscalar証拠へ接続。新trajectory不要 |
| Remainder/winding | 既存exact恒等式、円周距離、条件付きSterbenz域 | Rust setterとPython `%`の別経路、有限更新回数、非負域の候補 | actual remainder誤差、modulus差、境界を含むwindingの全域根拠 | 最優先の残条件。有限点probeで全域証明としない |
| Controlled相対位相 | diag(I,V)、basis/parityのactual inverse、+Im wrapper | 明示gate/wire/order、P(g)一回、Full globalとの区別 | operand/phase数値辺は上記が所有 | 既存代数を再利用し、無用な回路生成なし |
| 全law転送 | `rho_native≤B E D`、uniform上界 | 共通保持項、有限K2 family、固定B/ν規約 | phase uniform上界が未取得 | 条件未確定を維持。compiled保証/N/Gへ自動接続しない |

数学的恒等式、明示条件付きの上界、metadataで確認した条件、未実施scalar数値評価、actual実装の未証明条件を別の状態として扱う。
保存inputと各関数の対応は確認できたが、Native全law予算達成や実装込みの低費用順位は認定しない。

## 9. 必要な場合だけ選ぶ次の限定scope

現状の仮想ρ=0・保存価格標本による結果を報告するだけなら、追加科学計算は必須でない。
Native保証をさらに求める場合でも、まずactual remainderの実装根拠を閉じる必要性と負荷をユーザーが判断する。
今回の公開後に自動で証明・scalar評価へ移行しない。

| 候補scope | 対象・件数・精度の案 | 完了/STOPと承認境界 |
|---|---|---|
| 保持operandだけのscalar評価 | 78個の共通cとδ/half/θ binding、各`|β−α|`一組ずつ。exact Fractionで一括、Hybrid2候補で再評価しない。R64は対象なし | 新科学的数値評価なので別の明示許可が必要。固定SOURCE/hash/operand/丸め順不一致・未対応域ならSTOP。状態作用/compile0、phase上界は得たと扱わない |
| phase binding/定数だけのscalar確認 | 共通scalar binding1組、math.pi／Rust modulus operandと2つのorder endpoint。πを含む定数評価はArb128bit等を新scopeで固定 | SOURCE/actual backendとの対応を先に固定。未解像時はSTOPし、自動精度retryしない。有限点の確認をremainder全域証明にしない |
| uniform phase式の限定評価 | remainder/windingの補題が閉じた後だけ、r32/r64の2種類の更新回数式をscalar包含。個別trajectory列挙なし | 必要な実装条件・誤差式・precision/call/STOPを固定して別承認。条件が閉じなければ数値実行せず未確定STOP |

いずれも案であり実行未承認・未実施。新H6状態/行列作用、sampling、QPY、compile、N/G、価格解析を追加しない。
既存scalar/Pauli primitivesとarchive読取を適合範囲で再利用し、必要以上のframeworkを作らない。
Compiler側のopt1 1Q再合成、lowering、accepted rewrite、event境界融合、全law compiledρはA調査の未解決事項として維持する。
Native上界が得られてもCompiled保証にはならず、N/Gの順位感度も別scopeである。

## 10. 技術照合・非干渉・保全・mandatory STOP

元2科学SOURCEのappend/trajectory、D2 `_real/_tokens/_rotation`、event、law exporter、wrapper、tight targetの該当関数と、
P2で抽出されたcontrol emit部分をASTで22比較し一致した。phase setterの保存upstream Git blob、元model hash、3 law memberのhash・保持prefixのbitsを照合した。
旧テストの`test_h4_r2_static_design.py`、`test_h4_r2c_proof_lemmas.py`、`test_h6_selected_implementation.py`は対象と限界を読取し、今回は一件も実行していない。
通常の文書/SOURCE整合性確認を行い、科学的なPASS・独立レビューへ格上げしない。

Bの価格tail調査は別worktree/branchであり、結果待ちや資料変更をA-1の停止理由にせず、そのファイル・process・価格・標本には書き込まない。
A-1で価格再集計・tail寄与・母期待価格・費用順位を評価した件数は0。Bの未実行/実行状態や結論を代行して報告しない。
共有設定・libraryや他worktreeを変更しない。元P1/P2、局所診断、R2/R2-C、P3、AのSOURCE/input/raw/QPY/結果/grant/ledger/marker/STOPを保全する。
旧manifest entriesは保持し、本非実行調査の参照だけを追加する。新科学コード・runnerはない。

文書整合性の技術照合はPASS：変更対象は調査書・ノート・索引/status/manifestの7ファイルだけ、旧manifest210 entriesの内容不変、
本調査書/新ノートのlocal link7件有効、既存A・局所STOP・R2-Cのbytes不変、`git diff --check`異常なし。
このPASSは全law誤差の認定ではない。文書は上記独立branchへ公開し、公開後のSHA/本文照合結果は完了報告に示す。

現在の`STOP_H6_SELECTED_IMPLEMENTATION_LOCAL_DIAGNOSTIC_COMPLETE_NO_AUTOMATIC_FOLLOWUP`を維持する。
本A-1も文書commit/push・remote SHA/公開本文照合後に**mandatory STOP**。
全law Native/Compiledの数値上界、保存許容幅所属、actual_N/Gは未取得のまま。
次の担当：**ユーザー**が、現状の条件付き成果を使うか、§9の限定scopeが必要か判断する。
**Codex**は新しい具体的指示・必要な数値実行許可がある場合だけ、その固定scopeを進める。

[A]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0e1c7aae386986f6c1bb9483548671655447f516/docs/research/h6_whole_law_transfer_feasibility_20261102.md
[LOCAL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/71432728ee0140fb3c1f8bf30ecf51b2faba86ec/artifacts/h6_selected_implementation/2026-11-02/run01/REPORT.md
[LOCALSTOP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/71432728ee0140fb3c1f8bf30ecf51b2faba86ec/artifacts/h6_selected_implementation/2026-11-02/run01/STOP.json
[LOCALINPUT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/71432728ee0140fb3c1f8bf30ecf51b2faba86ec/artifacts/h6_selected_implementation/2026-11-02/run01/source_and_inputs.json
[INV]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b3bab573cf17b9dd8ddc1dca5f1a6a6c935ef6d1/docs/research/h6_pure_rte_hybrid_implementation_evidence_inventory_20261102.md
[INVJSON]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b3bab573cf17b9dd8ddc1dca5f1a6a6c935ef6d1/artifacts/h6_pure_rte_hybrid_evidence_audit/2026-11-02/INVENTORY.json
[CPY]: https://github.com/python/cpython/blob/v3.11.0rc1/Objects/floatobject.c#L637
[PYREM]: https://docs.python.org/3.11/reference/expressions.html#binary-arithmetic-operations
[RSREM]: https://doc.rust-lang.org/std/primitive.f64.html#method.rem_euclid
[SRCAPP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/887b69e449cb3c378286f09d0d7626599850fb67/src/trotterlib/pauli_signal_adapter.py
[SRCD2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/887b69e449cb3c378286f09d0d7626599850fb67/src/trotterlib/h4_d2_pauli_control.py
[SRCEVENT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/887b69e449cb3c378286f09d0d7626599850fb67/src/trotterlib/rte.py
[SRCLAW]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/887b69e449cb3c378286f09d0d7626599850fb67/src/trotterlib/h4_pauli_pr_search.py
