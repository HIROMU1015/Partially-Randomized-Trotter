# O2：実装前の非同値性による候補選別

2026-11-08 JST／SERVER1 GPUA100-gleap-01。baseは案E結果 f085b56566d60d9cb5df221f352a2b52ac51ea88。**DISTINCT_MECHANISM_SUPPORTED は0件。** 三方向を具体的な操作・情報取得条件へ落とし、既知法への還元と未解決のアクセスを分けた。新しいinstrumentや科学契約を発行しない。O2全体の不可能性・終了判断ではない。

## 共通target・比較規則

\[
F=\sum_\omega Q(\omega)U_\omega,\quad
Z=\operatorname{Tr}(\rho F),\quad
H_X=(F+F^\dagger)/2,\quad H_Y=(F-F^\dagger)/(2i).
\]

Qは元のactual finite-grid lawで、各Uにはselected full policy、scalar/identity/event phase、演算順を含む。純粋状態ρ=|ψ⟩⟨ψ|で元targetに一致する。平均channel・|Z|²・success条件付き平均を代替targetにしない。各座標μ=Tr(ρH_a)に対し、通常Hadamard bit X_ω∈{±1}、E[X_ω|ω]=μ_ωを参照する。X/Yは別の有料取得である。

以下は記号的設計だけ。H4/他分子のgeometry、basis、DF rank、L_D、δ窓は今回の新計算には非該当。actual Qを仮定した式を、実N1入力への評価済み適格性と取り違えない。

## 1. Normalizer表を作らずに取得する二段推定

### 操作・入力・出力

元Qの古典samplerと、既知metadataから取得する正score a(ω)∈[a_min,1]を仮定する。scoreの取得価格・上限・厳密な実効coin確率は無料としない。未知state・Born平均は入力にしない。

提案操作は、(i) M本のfresh Q metadataで A=E_Q[a] の独立不偏推定 A_hat を作り、(ii) Q候補を確率aで受理するrejectionから N本を取得、各受理pathだけ通常Hadamardを実行し、(iii) T=A_hat N^-1 Σ X_i/a(ω_i)を返すこと。二段の乱数・測定・選択データを独立にする。これは今回実行する擬似コードではない。

accepted lawはq(ω)=Q(ω)a(ω)/A。Qを変更した際のweightを戻すため

\[
E_q[X/a]=\mu/A,\qquad E[T]=\mu.
\]

### 数学的利益と負担

v=Var_Q(a)、G=E_Q(1/a)と置くと、独立・固定M/N・制限なしrejectionモデルで

\[
\operatorname{Var}(T)=
\frac{A^2+v/M}{N}\left(\frac{G}{A}-\frac{\mu^2}{A^2}\right)
+\frac{v\mu^2}{MA^2}.
\]

これは本作業の記号的導出で、原資料のPR性能定理ではない。Mを固定してNだけ増やすと、一般にnormalizer校正の項が残る。|T|≤1/a_min、A_hat≤1なのでrangeも元bitより広がり得る。μやvを無料で知ってMを決められるとはしない。

score取得平均費c_metaと量子測定費C_qを分離すると、古典生成はおおむね M c_meta＋(N/A)c_meta、量子費はN E_q[C_q]。全path compileをscore取得に使うなら拒否候補にもその取得費を払う。有限word capで完了条件に条件付けた出力へ上の独立モデルの式を自動適用しない。finite scope・停止時の統計の扱いは未解決で、実行案は発行しない。

53-bit coinではaは実際に実装するinteger threshold/2^53を正本にする。理想score/supを丸めてから旧qやweightを使ってはならない。positive Q support上でa=0となる丸めは拒否し、支持を落とさない。

### 還元判定

**REDUCES_TO_EXISTING_METHOD。** このlawとweightはrejection＋importance samplingそのもの。同じscore・独立normalizer推定・拒否費をbaselineにも与えれば、操作・outcome・費用をそのまま再現できる。正確なnormalizer表を持つbaselineより弱い情報で動く場合はあるが、既知rejection/ランダムnormalizer baselineに対する独立機構ではない。

[Schmon–Doucet–Deligiannidis](https://arxiv.org/pdf/1903.00939)の§3はBernoulli raceと受理率の不偏推定を扱う。この既知取得原理との重複を確認した。同論文のparticle-filter性能をPRへ移植せず、独立二段式以外の相関した推定器に上のvariance式を一般化しない。新code不要で、この方向の形式的な変種を増やさない。

## 2. Prefixの有限状態summaryからsamplingとreturnを同時生成

### 具体的な操作

長さLの生成path j_1…j_Lについて、安価なsummary更新 s_{t+1}=T_t(s_t,j) があり、(a) 元actual conditional Q_t(j|prefix)がq_t(j|s_t)で厳密に表せ、(b) terminal s_Lから元phase付きU_ωを等価なV_sとして生成でき、(c) full-wrapperの正score g(s_L)が取得できる、と仮定する。

terminal値 h_L(s)=g(s)から

\[
h_t(s)=\sum_j q_t(j|s)h_{t+1}(T_t(s,j))
\]

を計算し、生成時の遷移を

\[
q_t^*(j|s)=q_t(j|s)h_{t+1}(T_t(s,j))/h_t(s)
\]

にする。telescopingにより path law は Q(ω)g(s_L)/h_0(s_0)。最後にV_sを生成し、weight h_0/g(s_L)でHadamard座標を補正する。元targetを保存するには、operator endpointの等価性だけでなく全phaseと実取得lawを保持し、g=0で支持を消してはならない。

有限状態数d、alphabet bなら、この再帰自体はO(Ldb) arithmeticで計算できる。ただしd、exact算術bit費、summary/table構築、terminal full価格の取得が安いことはこの式から従わない。label sign/phaseをstateに含めず係数へ移す場合は、複素係数・絶対値normalization・decoderへの対応も別途要る。

### 同値部分・未解決部分

再帰は既知のweighted path dynamic programming、samplingは終端scoreで傾けた既知conditional IS。全てのUがterminal同値類で一致するなら、returnの短縮は同じpushforwardのReturn Aggregationと同じである。[Mohriのweighted automata](https://cs.nyu.edu/~mohri/pub/hwa.pdf) §1–2との重複を確認した。compile/fusionにも同じ等価returnを認める。

**DISTINCTNESS_UNRESOLVED。** 未解決なのは再帰の名称や実装でなく、既存Return/Closed/Parentでは安く取得できないfull価格・operator/law整合summaryを、別の具体的構造から安く構築できるか。現在そのsummary・閉じた遷移・計算量差を特定していない。Tとgを新oracleとして無料入力にするだけでは非同値性を支持しない。

現ReturnKernelは局所return queryを持つが、この一般summaryやwhole-price normalizerを提供しない。逆に、既存baselineへ同じT/q/gを与えれば同じ再帰と短returnを使える。**現在のコードに関数がないことだけを、アルゴリズムの非同値性にしない。**

q*は解析上の条件付き法則であり、53-bit実装でそのまま厳密取得できるとは限らない。exact integer rejection等を別途接続するか、実効finite-bit lawでweightを戻す。q*を丸めたままh_0/gのweightを使わない。

利益の条件は、全取得・生成・finite-confidence費を含めた C_N V_N<C_B V_B と追加の有限誤差保証。単なる全path表のstate名変更、全reduced wordをstateへ保持する方法、Mの実label再分割は新しい安価なアクセスではない。Pauli/Clifford等の既知閉包なら既知group演算・強いcompilerも同じ表を使える。実selected basis/連続角/非加法価格へのcompact閉包はUNKNOWN。

## 3. Labelを消した座標observableの測定

### 操作と還元

Qと明示U記述からH_aを構成し、もし H_a=R diag(λ) R† を取得できれば、R†を作用して計算基底zを測りλ_zを返す。E[Y]=Tr(ρH_a)、M2=Tr(ρH_a²)、range=λ_max−λ_min。同じ回転R・spectrum記述・費用をbaselineへ渡せば、そのまま既知直接測定となる。Rの生成をLCU/block encoding/QPEで行う場合も、その既知アクセスと準備・逆回路・反射・測定費を全て戻す。

**KNOWN_METHOD_APPLICATION。** 具体化できた操作は既知observable測定であり、同じ入力からRを安く得る新構造は特定していない。形式的にHを書けることを安価な量子oracleと見なさない。

さらに、stateに依存しない一copyの全状態不偏POVMをE_y、実outcomeをyとすると、ΣE_y=I、ΣyE_y=H_a。次の基本恒等式が成り立つ。

\[
B-H_a^2=\sum_y(yI-H_a)E_y(yI-H_a)\succeq0,\quad
B=\sum_y y^2 E_y .
\]

従ってTr(ρB)≥Tr(ρH_a²)。outcomeを三値/多値に変えるだけで、同observableの理想spectral測定より二次momentを下げる説明にはならない。[D’Ariano–Giovannetti–Perinotti](https://arxiv.org/abs/quant-ph/0510187)との既知observable推定の重複を確認した。この一copy恒等式の導出は y²項−2H²＋H²の展開だけで、量子simulation不要。

ただしspectral測定の回路費が無料とは言わない。新しい**安価な実装アクセス**があれば資源上の差はあり得るため、この不等式を全アルゴリズム・状態固有条件・coherent queryのNO-GOへ拡張しない。既知sector内だけの全状態不偏性なら、その空間へ制限したHで同じ議論。特定groundの真期待値をoracleに使うことは許さない。

## 早期除外した他の方向

copy再利用/二copyの重なり測定は、一般にTr(ρUρV†)等を取得しても線形Zを返さず、状態を戻す逆準備も無料でない。元targetへ戻す具体的操作・追加費を示せず NO_SUPPORTED_IMPROVEMENT_PATH とした。標準amplitude estimationを名前だけ替える方向も、coherent sampler/逆state/反射を戻せば既知法の適用で、新しい取得構造を特定していない。

既存M/D/Eの別条件・別pool・新特徴量を救済候補にしていない。三方向とも実装前に止めた。新アルゴリズムcode・人工代数test・量子simulation・科学計算・旧科学値再解析は0。
