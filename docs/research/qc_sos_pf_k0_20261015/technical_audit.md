# SOS-PF候補A：K0技術事前検証

2026-10-15 JST。区分：文献・紙上数学・静的source監査。新科学数値なし。

**K0-1〜K0-5の監査を完了し、全体を `UNDETERMINED_DATA_REQUIRED` とする。`K0_PROTOTYPE_CANDIDATE` は付与しない。** PSDによるSOS gap型の漸近係数上界と、Hermitian一体平方の回路接続は成立する。一方、独立した有利な分割生成規則、化学で有用な有限step certificate、同情報の強対照に対する総費用改善は確認できていない。これはSOS-PF全般の不可能性・正式棄却ではない。**K1未認可、正式採択なし、Mandatory STOP。**

## 1. 研究目的と入力の固定

対象は元の有限基底電子Hamiltonianのground energyを、指定精度・成功確率で低費用に推定する問題である。厳密な量子ground stateを利用できる既存前提を保持する。古典E0、全RDM、励起resolvent、厳密gapを無料の設計入力にはしない。状態準備法の開発へ戻さない。

基準は研究状態commit `78b50c515d9df9556f261e74ca7c57af461b7199`、M案K0原固定commit `155336b07ad0d5e6e9c6cad35736e9412060a299`。現在のroot HEADは別commitで多数の既存変更を含むため、研究状態・source参照はlocal Git objectを指定して読む。今回GitHubから独立再取得したとは主張しない。[入力・固定資料のidentity](provenance.json)。

ユーザーの[今回の指示](received_instruction.txt)と[指定GPTレビュー原文](supplied_gpt_review.md)をbyte保存した。Claude `レビューouter.md` 原本は今回未取得であり、GPTレビュー経由の引用をClaude全文の独立監査としない。損益分岐式等も取得済みの式だけを検討する。

最初に `PROJECT_MAP.md`、rootの `docs/research/研究概要・現状.md` を読み、指定固定commitの目的文書・採用decision・M監査・統合引継ぎ・厳密ground前提レビューの関連節に戻った。root概要は旧時点の記述を含むため、量子化学系列の現在状態には指定された採用decisionを用いる。`VALIDATION_STATUS.md` とglobal manifestは履歴・状態確認の読取りのみ。保存科学数値の再集計・再ランキング・再認証は行っていない。

第一候補の現固定構成CLOSED、H-E1現定義終了、M-a/M-b STOP、小改変系列PAUSED、旧D HOLD、他Track・N1/N2/N3等の既存状態を保持する。今回のK0許可は過去のSTOPの解除ではない。block-encoding/qubitizationを新たな実行対象にしない。

## 2. K0-1：先行研究と新規性

一次本文の実際に読んだ節・版・取得限界は[文献registry](literature_registry.json)に記録する。以下は全文証明の再現・網羅的初出性調査ではない。

| ID・一次資料 | 確認箇所と既知の内容 | 今回への含意 |
|---|---|---|
| L1 [Hejaziほか、2412.16811v1](https://arxiv.org/html/2412.16811v1) | §I Eq.9、§II Eqs.10–13、§IV。任意次数PFの固有値摂動、低エネルギー評価、特別な二分割公式 | ground bias特化とPSD低エネルギーの接続は既知。§IVは局所性・有界degree・一定数の可解群を仮定 |
| L2 [Hejaziほか、2402.10362v1](https://arxiv.org/html/2402.10362v1) | §I–III、補足§VIIIの漏洩条件。低エネルギー射影下の交換子誤差 | PSD分割で通常normを制限normへ置換する方針は既知。fermion拡張にもmode degree等の費用依存がある |
| L3 [Mizuta–Kuwahara、2504.20746v1 PDF](https://arxiv.org/pdf/2504.20746v1) | 主定理1 Eqs.2–5,13–18、補足S2定理S1 Eq.S5 | projected nested commutatorの低エネルギー一次依存は明示的に既知。任意の密なDF平方への有利な定数の転用は未成立 |
| L4 [Yi–Crosson、npj QI 8,37](https://www.nature.com/articles/s41534-022-00548-w) | 二分割のS1/S2等スペクトル、定理2 Eqs.23–27、Discussion | 固有値・overlap・gapを分ける解析は既知。非縮退branch条件をSOSで自動解消できない |
| L5 [Lowほか、2502.15882v1](https://arxiv.org/html/2502.15882v1) | §II Eq.12、§III Eqs.13–23、§IV入口 | 化学SOS、Gram SDP、DFTHC、下界とnormalizationの交換は既知。spectral amplification資源をPFの資源へ移せない |
| L6 [Kingほか、2505.01528v1](https://arxiv.org/html/2505.01528v1) | §I–II、termwise SA Eqs.3–4、定理4/5/7の入出力条件 | 単純PSD化と低energy advantageの競合、取得情報・overlapの会計は既知。PF固有値定理ではない |
| L7 [Rubin–Low–DePrince、2602.05069v1](https://arxiv.org/html/2602.05069v1) | §II Eqs.2–12、§III.3 Eqs.63–80、§IV末 | weighted SOS、symmetry ideal、v2RDM双対、spin-free Gramは既知。一般SOS→Hermitian平方PFの費用保存までは与えない |
| L8 [Casaresほか、2606.30741v1](https://arxiv.org/html/2606.30741v1) | §III、§V.2–3、Appendix D。SPRINT、near-integrability、processing、回路価格 | 分割とPF次数・費用の共設計が既知。論文のleading-error推定を有限stepの厳密certificateとしない |
| L9 [Kang–Su、2510.06316v3](https://arxiv.org/html/2510.06316v3) | §1.6、§6.1–3 Eqs.245–264、Cor.27/29 | **SOSをPFでシミュレートし、化学のHermitian一体平方と固定粒子数normを使う構成まで既知**。Hamiltonian block encoding/QSVTのaccessと補助qubitを使うので、直接の軌道回転PFと同一ではない |

L9は受領レビューの主要リストにない近接先行研究である。これにより「SOSとPFの組合せ」や「化学の平方fragmentをPFで使う」だけを研究空白とは扱えない。ただしL9のアクセスモデルは異なり、SOS gap比例のground固有値certificateが本候補と同値だとは認定していない。対照範囲の既存除外も勝手に解除しない。

L3のHTML表示の本文日付はv1 PDFと異なったため、主条件と補足定理S1はv1 PDFでも照合した。一般PSDを全supportの項として扱えば同定理の粗い特殊化は可能だが、局所性の良いサイズ依存は失われる。「非局所なのでSOS gap一次依存の原理がすべて未踏」という推論はしない。

**新規性の切分け**：PSD分散補題、BCH・固有値摂動、SOS/BLISS生成、平方の軌道回転指数化は既存数学の直接利用である。残り得るのは、(i)同じ入力価格で有利なPF-native分割を生成する具体則、(ii)非局所化学に有用な明示定数の有限step保証、(iii)強い既存PFとの費用差を伴う統合構成。ただし今回、これらの独立寄与を確定する構成は得ていない。名称未発見も、既知部品から導出可能という事実も、全体の新規性の肯定／否定証明にはしない。

## 3. K0-2：基本補題と二分割の独立導出

### 3.1 共通の数学的契約

有限次元の対象sectorを各fragmentが保存するとし、すべてのnormをそのsectorで取る。

\[
H'=H-E_{\rm SOS}I=\sum_{j=1}^M F_j,\quad 0\preceq F_j\preceq f_jI,
\quad H'|0\rangle=E_g|0\rangle.
\]

`E_g` はSOS下界からのoffsetであり、励起状態へのspectral gapではない。`f_j` は厳密normに限らず、検証済みの古典的上界でよい。`phi_j=<F_j>`、`sigma_j²=<F_j²>−phi_j²` とする。和の恒等式と演算子順序から

\[
\sum_j\phi_j=E_g,\quad
0\leq\sigma_j^2\leq f_j\phi_j-\phi_j^2\leq f_j\phi_j,
\quad \sum_j\sigma_j^2\leq f_{\max}E_g.
\tag{1}
\]

非負性と `F_j²≤f_j F_j` だけの帰結であり、新規定理としない。`E_g=0` なら全 `F_j|0>=0`。従って任意h・任意順序のPFがground上で厳密になる。ただし位相の読み出し・alias問題まで消えるとは言わない。

### 3.2 対称BCHの規約と係数

積は表示どおりの行列積で定義する。

\[
S_{ABA}(h)=e^{-ihA/2}e^{-ihB}e^{-ihA/2},\qquad
\log S_{ABA}=-ih(H'+h^2D+O(h^4)).
\]

一般のX,Yで、次数3までの対称BCHは

\[
\log(e^{X/2}e^Ye^{X/2})
=X+Y-\frac1{24}[X,[X,Y]]+\frac1{12}[Y,[Y,X]]+O(5).
\]

`X=−ihA, Y=−ihB` では三次の積に `(-i)³=i` が付くため、

\[
\boxed{D=\frac1{24}[A,[A,B]]-\frac1{12}[B,[B,A]].}
\tag{2}
\]

`a=<A(H'−E_g)A>` と置くと、

\[
\langle[A,[A,B]]\rangle=-2a.
\]

`Q=I−|0><0|` とすれば `QB|0>=−QA|0>`、従って
`<B(H'−E_g)B>=a` で、もう一つの二重交換子も `−2a` となる。

\[
\boxed{d=\langle D\rangle=\frac a{12}\geq0.}
\tag{3}
\]

`H'−E_g I` はgroundに対してPSDなので非負性はPSD分割でなくても成立する。孤立branchなら、これが固有値のh²係数である。PSDを使う大きさの評価は

\[
0\leq d\leq\frac{W}{12}\min(\sigma_A^2,\sigma_B^2)
\leq\frac{W}{12}\min(f_A\phi_A,f_B\phi_B)
\leq\frac{W}{12}\frac{f_Af_B}{f_A+f_B}E_g,
\tag{4}
\]

ここで `W=||H'−E_g I||≤||H'||≤f_A+f_B`。最後は `phi_A+phi_B=E_g` と二直線の交点による上界。分母ゼロならA=B=0でd=0。`W f_max E_g/12` も安全だが弱い。**二分割の正符号を多fragment一般へ移さない。**

縮退groundの場合、任意に選んだgroundベクトルの対角期待値だけでは個々のbranch係数を指定できない。まず `P_0DP_0` をground部分空間内で対角化する必要がある。二分割ではこの圧縮は `P_0 A(H'−E_g)A P_0/12` としてPSD。縮退部分空間へのoverlapと特定ベクトルへのoverlapも別である。

## 4. 多fragment：係数辞書と古典的に評価可能な定数

### 4.1 S2の順序を明示する

\[
S_2(h)=e^{-ihF_1/2}\cdots e^{-ihF_{M-1}/2}e^{-ihF_M}
e^{-ihF_{M-1}/2}\cdots e^{-ihF_1/2}.
\]

`R_j=Σ_{k>j}F_k` とおく。外側の対称BCHを一つずつ適用すると

\[
D=\sum_{j=1}^{M-1}\left\{\frac1{24}[F_j,[F_j,R_j]]
-\frac1{12}[R_j,[R_j,F_j]]\right\}.
\tag{5}
\]

内部S2のDはこの次数では加算され、外側との交差は次の次数に入る。従ってraw辞書は

\[
\kappa_{jjk}\mathrel{+}=1/24\ (k>j),\qquad
\kappa_{klj}\mathrel{+}=-1/12\ (k,l>j).
\tag{6}
\]

raw項数は `Σ(r+r²)=(M³−M)/3`、`Σ|κ|=M(M−1)(4M+1)/144`。これは相殺前の辞書であり、独立Lie monomialの数ではない。可換性、同一添字、Jacobi、係数相殺を利用する場合は同じ規約で両対照へ与える。

### 4.2 分散から非負行列上界へ

`delta F_j=F_j−phi_j I`、`w_j=||delta F_j||≤f_j`。四つの語を両端のベクトルnormで評価すると

\[
|\langle[F_a,[F_b,F_c]]\rangle|
\leq2(\sigma_a\sigma_c w_b+\sigma_a\sigma_b w_c)
\leq2f_b\sqrt{f_af_c}\sqrt{\phi_a\phi_c}
+2f_c\sqrt{f_af_b}\sqrt{\phi_a\phi_b}.
\tag{7}
\]

`x_j=√phi_j` とし、ゼロから始める実対称非負行列 `B_SOS` に各raw辞書項ごとに

\[
B_{\rm SOS}\mathrel{+}=|\kappa_{abc}|\left[
f_b\sqrt{f_af_c}(e_ae_c^T+e_ce_a^T)
+f_c\sqrt{f_af_b}(e_ae_b^T+e_be_a^T)\right]
\tag{8}
\]

を加える。同一添字の対角項もこの式どおり二回加える。すると

\[
\boxed{|d|\leq x^TB_{\rm SOS}x\leq K_{\rm SOS}E_g,
\qquad K_{\rm SOS}:=\max_a\sum_b(B_{\rm SOS})_{ab}.}
\tag{9}
\]

実対称非負行列の最大絶対行和がspectral normの上界であり、**固有値計算もgroundのphi取得も不要**。係数・norm上界だけからO(M³)演算、行和を直接加算すればO(M)保存量で評価可能。密行列を作るならO(M²)保存量。これは静的計算量で、今回実行した数値集計ではない。

fragment増加に対して定数が自動で小さくならない。`f_j≤f` なら、Eq.6前半の任意行和は `(M−1)f²/8` 以下。後半で行mに寄与する係数は

\[
\frac{f^2}{12}\left\{(M-m)^2+3\sum_{j<m}(M-j)\right\}
\leq\frac{M(M-1)f^2}{8}.
\]

従って安全な一様上界は

\[
K_{\rm SOS}\leq (M^2-1)f^2/8.\tag{10}
\]

これは精密なサイズ則や最適定数ではない。細分化でfも変化するのでMだけでは比較できない。化学の密な非局所平方を局所定理へ移すと、局所support・extensiveness・群数の悪化も価格に戻す。回転後に対角であることと、全fragmentが共通の局所basisを持つことは異なる。

### 4.3 実用情報と証拠精度

認証済み `L≤E_0≤U` があれば `G=U−L` を `E_g` の上界として使用できる。Lは**実際に用いるPF-native PSD分割**の下界でなければならない。一般SOSのLを、変換後に悪化した下界の代わりに使わない。

浮動小数SDP出力は、そのまま厳密証拠にならない。Gram PSD、normal-order後の係数一致、sector ideal、丸め残差を検証する。例えば残差演算子norm≤rhoなら、Lをrhoだけ下げて別誤差を計上できるが、負のGram成分を切るだけでは元Hとの一致は保たれない。Uも検証可能な変分上界が必要。これらの生成・検証費用と証拠のbit長は未測定である。

## 5. h⁴、補正後残差、有限stepとoverlap

### 5.1 漸近展開の成立範囲

孤立・非縮退ground、gap `g_0=E_1−E_0>0`、原点から続くanalytic log branchで

\[
H_{\rm eff}=H'+h^2D+h^4T+O(h^6),\quad
E(h)=E_g+h^2d+h^4\{\langle T\rangle-\langle DQ(H'-E_g)^{-1}QD\rangle\}+O(h^6).
\tag{11}
\]

`D=Σ_w c_w W_w` を長さ3の語、`T=Σ_v t_v W_v` を長さ5の語に展開する。norm上界fから

\[
N_D=\sum_w|c_w|\prod_{j\in w}f_j,\quad
B_D=\sum_w|c_w|\left(\prod_{j\in w\setminus\{\mathrm{last}\}}f_j\right)\sqrt{f_{\mathrm{last}}},
\]
\[
A_T=\sum_v|t_v|\sqrt{f_{\mathrm{first}}f_{\mathrm{last}}}
\prod_{j\in v\setminus\{\mathrm{first,last}\}}f_j
\tag{12}
\]

が計算可能な粗い定数となる。積は**出現位置**ごとに取り、同じfragmentの重複も含む。`||F_j|0>||≤√(f_j E_g)` を右端に使って

\[
\|D\|\leq N_D,\quad\|QD|0\rangle\|\leq B_D\sqrt{E_g},\quad
|\langle T\rangle|\leq A_TE_g.
\]

したがってh⁴係数の絶対値は `E_g(A_T+B_D²/g_0)` 以下、overlap損失の先頭係数は `E_g B_D²/g_0²` 以下。前者には二項の相殺があるので、下界改善から実際の残差減少を結論しない。Tの完全な係数辞書を今回数値生成していない。Eq.12はその辞書を与えた場合の明示評価規則である。

### 5.2 有限hへの条件付き接続を明示する

「剰余を何とか評価すればよい」で終わらないよう、**粗いが明示的な十分条件**を紙上で与える。新規性・有用性は主張しない。

`Lambda=Σ f_j>0`、`R=log(3/2)/Lambda`、`M_R=log(2)/R`。複素zで `|z|≤R` なら、指数積のnorm級数から

\[
\|S_2(z)-I\|\leq e^{|z|\Lambda}-1\leq1/2,
\quad\|\log S_2(z)\|\leq\log2.
\]

ゆえに `H_eff(z)=i log(S_2(z))/z` は原点の可除特異点を除いてanalytic。対称性でevenとなる。円上の上界M_RとCauchy係数評価により、実 `|h|<R` で

\[
\|H_{\rm eff}(h)-H'-h^2D\|\leq
r_4(h):=M_R\frac{(|h|/R)^4}{1-(|h|/R)^2}.
\tag{13}
\]

`Lambda=0` は全fragmentゼロなので別途厳密。known scalar `E_SOS` の位相は最後に復元し、このlog半径へ無用に混ぜない。

さらに**検証済みのgap下界** `g≤g_0` と `G≥E_g` を与え、

\[
v(h)=h^2N_D+r_4(h)<g/2,\quad
b(h)=h^2B_D\sqrt G+r_4(h)
\tag{14}
\]

とする。Weyl評価で最下固有値は孤立したまま。P=|0><0|とQのSchur補元を使うと、Q blockと対象固有値の距離は `g−2v` 以上で、

\[
\boxed{|E(h)-E_g-h^2d|\leq r_4(h)+\frac{b(h)^2}{g-2v(h)},}
\tag{15}
\]
\[
\boxed{|E(h)-E_g|\leq h^2K_{\rm SOS}G+r_4(h)+\frac{b(h)^2}{g-2v(h)}.}
\tag{16}
\]

同じblock方程式から `s=b/(g−2v)` とすれば

\[
1-|\langle0|0(h)\rangle|^2\leq\frac{s^2}{1+s^2}.
\tag{17}
\]

導出の要点：`<0|H_eff|0>=E_g+h²d+<R4>`、off-diagonal blockのnorm≤b。Schur項は非負で上界 `b²/(g−2v)`、固有vectorのQ成分とP成分の比はs以下。全h区間でEq.14が成り立てばbranchの連続追跡も閉じる。縮退の場合はPをground全空間へ広げ、scalar式をそのまま使用しない。

この証明は非局所の有限次元にも適用するが、半径が `1/Σf` で狭く、r4はSOS gapに比例しない。gap下界も元Hの係数だけから安価に得られるとは限らない。**従って有限hの条件付き保証自体は書けるが、化学で有用な保証付きstep削減は未成立**。既存の鋭いcommutator剰余や低エネルギー定理より優れているとも言わない。Eq.15–17は標準analytic評価と摂動の紙上接続であり、数値検証済みの新定理とラベルしない。

### 5.3 scalar補正とQPE/RPEの境界

`|d_tilde−d|≤eta_d` を別途保証できれば、補正 `E_corr(h)=E(h)−h² d_tilde` の有限誤差はEq.15右辺に `h² eta_d` を加えたものとなる。scalar補正はeigenvectorを変えず、overlap改善を生まない。dの取得・精度・測定失敗確率は別価格。厳密量子ground stateがあってもdの古典値は無料でない。

長い問い合わせは同じhのkernelを反復するものとし、`S_2(h)^r=exp(−irh H_eff(h))`。エネルギーbiasと総時間を掛けた位相bias、読み出しの統計誤差、合成誤差を分離する。異なるroundでhを変えるなら異なるH_effになり、単一固有位相モデルを再確認する必要がある。alias解消には元のエネルギー窓・既知scalar復元・位相unwrappingが必要であり、principal logの局所存在だけで全roundの成功率を保証しない。

## 6. K0-3：PF-native SOSの三層

### 6.1 A：既存DFのPSD化は安価な必須対照

既知の表示を `H=cI+Q+Σ_l lambda_l L_l²`、`L_l=a†g_l a`、`g_l=g_l†` とする。
非負lambdaのDFでは平方項は既にPSD。Qの対象粒子数sectorにおける下限qを用いれば

\[
H=(c+q)I+(Q-qI)+\sum_l\lambda_l L_l^2.\tag{18}
\]

一体Qは軌道固有値の小さいものからeta個の和が固定N最低値。固定N_alpha/N_betaならspin blockごとに和を取る。これは紙上の構成法であり、今回一体固有値すら計算していない。負lambdaやsigned CDFでは自動PSDを仮定せず、既知の区間上界u²を使って `|lambda|(u²I−L²)` 等へshiftする。一般density Jも独立にPSD証拠が必要。

重要なのは**既存fragmentの定数shiftだけなら回路と実biasは不変**という厳密恒等式である。元fragment `H_j` に対して `F_j=H_j−alpha_j I` なら、各fragmentの総時間係数が1のPFで

\[
e^{-ih\sum_j\alpha_j}\ S_2(h;\{F_j\})=S_2(h;\{H_j\}).\tag{19}
\]

交換子D、T、元Hへ復元した固有位相誤差・overlapは同一。改善し得るのは**同じ回路の保証とstep選択**である。さらに `Σalpha_j=0` のshiftならE_SOSすら不変で、phiやfを変えただけになる。PSD下限を超えた任意shiftは許されない。

### 6.2 B：構造化PF-SOS

固定frameのdensity平方を使う平方完成は

\[
\lambda L^2=\lambda(L-bI)^2+2\lambda b L-\lambda b^2I.
\tag{20}
\]

全平方のbを選ぶと、残余一体 `Q_b=Q+Σ2lambda_l b_l L_l`、下界は `c−Σlambda_l b_l²+min_sector Q_b`。**平方の中心移動には補償一体項が必要**で、下界は単調改善しない。定数shiftだけの場合と違いPF分割は変わり得るが、この操作は既知の平方完成・一体再配分の範囲にある。

固定frame・少数Gram block等を使えばfragment数・回路形を制約できる。しかし良好なgapが存在する保証や、同情報の既知法を超える選択規則は未特定。これを閉じた一体再配分系列や旧Dの再開理由にはしない。bの探索・Gram因子回転・最適化は実施していない。

### 6.3 C：Hermitian一体平方とGram実化

Hermitianかつnumber-conservingな一体演算子の実basisを `E_mu` とする。Iを含めてもよい。実対称Gram `S≥0` で係数一致が成り立てば、実因子分解 `S=VV^T` から

\[
X_a=\sum_\mu V_{\mu a}E_\mu=X_a^\dagger,
\quad\sum_{\mu\nu}S_{\mu\nu}E_\mu E_\nu=\sum_aX_a^2,
\quad M_{\rm square}=\operatorname{rank}S.\tag{21}
\]

複素Hermitian Gram `G=S+iT≥0` ではSは実対称PSD、Tは実反対称で、

\[
\sum_{\mu\nu}G_{\mu\nu}E_\mu E_\nu
=\sum_{\mu\nu}S_{\mu\nu}E_\mu E_\nu
+i\sum_{\mu<\nu}T_{\mu\nu}[E_\mu,E_\nu].\tag{22}
\]

最後の項は消えず、number-conserving一体basisではHermitian一体残余になる。`[a†ua,a†va]=a†[u,v]a` を使用した。Gの因子 `C=A+iB` から

\[
S=\operatorname{Re}G=AA^T+BB^T,\quad
\operatorname{rank}S\leq\min(m,2\operatorname{rank}G).\tag{23}
\]

例えば `G=[[1,i],[-i,1]]` はrank1だがReG=Iはrank2。**複素rank以下という一般的fragment会計は誤り**。実平方はrankS個、残余一体のPSD化を追加すれば原則さらに一fragment必要。残余が `Q_res`、そのsector下限がq_resなら、下界を `E_SOS+q_res` に変更して `Q_res−q_res I` を加える。q_res<0ならgapは悪化する。実化だけで元の下界・rank・実biasを保存したとはしない。

Eq.23はHermitian一体basisでのGramに対する命題である。一般のpair creation/annihilation、非Hermitianbasis、高次SOSへ、そのまま適用しない。`O=A+iB` の `O†O=A²+B²+i[A,B]` は常に成立するが、A/Bがnumber-conserving一体でなければ今回のPF grammarへ閉じない。元のspin-free／full DQG SOSで得られたgapの良さをこの限定クラスへ移さない。

### 6.4 厳密同値性とsector保存

`H−E_SOS I=ΣX²+Q+J_ideal` は係数をCARでnormal orderして確認する。sectorで `P J_ideal P=0` でも、各fragmentがsectorを保存しなければ `P exp(−ihF)P=exp(−ihPFP)` ではない。固定Nなら全軌道回転・平方がNを保存する。固定N_alpha/N_betaやS²を使うときはさらに各fragmentの保存則を確認する。単に全Hがspin対称という条件では足りない。一般のideal `f r+r f†` も、圧縮でゼロなのか、sectorを消滅させるのかを分ける。

DF/CDF切捨て・係数丸めを使えば近似対象はH_tildeである。元Hへのnorm誤差rhoはground energyで少なくとも別予算rhoを要する。hを小さくしてもこの誤差は消えない。今回新しい近似を採用していない。

## 7. 回路接続と実装の確認範囲

`X=cI+a†xa`、`x=x†` はGaussian rotationで `c+Σ_p epsilon_p n_p` へ対角化できる。

\[
(c+\sum_p\epsilon_p n_p)^2
=c^2I+\sum_p(2c\epsilon_p+\epsilon_p^2)n_p
+2\sum_{p<q}\epsilon_p\epsilon_q n_pn_q.\tag{24}
\]

したがって `e^(−ihX²)` は軌道回転・対角n/nn位相・逆回転で実装可能。複素Hermitian xにも対応するが、実直交回転だけの価格をそのまま使わない。controlled版ではbasis回転は無制御のまま対角部分を制御でき、scalarはancillaの相対位相として必要に応じてまとめられる。known scalarを古典復元するframeも全対照へ同様に許す。

固定commitの `src/trotterlib/df_trotter/model.py` のDFBlock/OneBodyGaussianBlock、`ops.py` の `df_squared_diagonal_primitives`、`append_diagonal_primitives`、`apply_df_block`、`df_partial_s2.py` の制御phase・構造会計を静的に確認した。sourceをimport・実行していない。

既存lower-level grammarにはRZ/RZZ・相対位相がある。一方DFBlockはeta/lam形式でc付き平方を直接受け取るAPIではない。Eq.24の一体係数を同じframeで合成するadapter・同値性検査は未実装。sourceの基底変換の積順もそのまま尊重する。Gaussian decompositionには小系dense fallbackがあり、利用可能な本番Givens経路を確認せず「scalable実装済み」とは言えない。

n modesでgeneric平方一fragmentは最大n個のRZ、n(n−1)/2個のRZZ、O(n²) Gaussian gatesを要する。S2は単純には2M−1回のfragment指数化、隣接stepや同basis融合で価格は変わる。exact-zeroや特殊角はsourceが省略し得る。**fragment数とgate数、angle slot数とT数は同一ではない。**

## 8. K0-4：費用レバーと強い対照

### 8.1 比較する二軸

| 比較 | 固定するもの | 変えるもの | 言える利益 |
|---|---|---|---|
| I：保証法 | H、PSD分割、順序、回路、入力情報 | 通常／sector commutator、既知低energy、Eq.9と有限hの補足 | 同じ回路の認証可能hの拡大。実biasの変更ではない |
| II：分割生成 | H、成功率、保証法、許可情報・費用単位 | A/B/CのPSD分割 | 実bias、overlap、fragment数・単価も変わり得る |

必須対照はDF/CDF+S2、PSD化DF+同じ鋭い評価、CS2、S4。可解groupの条件が満たせる場合にはnear-integrable/processed PF、Hejazi custom公式も含める。二分割公式の片側に、多fragmentのexact evolutionを無料で埋め込まない。純共役processorは固有値を変えないので、eigenvalue改善分として数えない。一般SOSを使う既知方法は先行性・情報条件の対照だが、block-encoding/QSVTを今回自動実装しない。

### 8.2 記号的損益分岐

同じ有効次数p、同じ誤差・overlap・読出し規約、**有限hで有効な** `|bias|≤c_p h^p` がある場合だけ、

\[
N_{\rm step}\sim T_{\rm interrogation}(c_p/\varepsilon_{\rm PF})^{1/p}.
\tag{25}
\]

Heisenberg型で `T_interrogation∝1/epsilon_E`、同割合のPF予算なら量子費用proxyは `c_step c_p^(1/p) epsilon_E^(−1−1/p)`。S2では平方根則になる。実際には剰余・alias・overlap・合成精度が最大hを制約する。

Claude式の `M mean_c_SOS/(L mean_c_DF)` を、fusion・control・合成精度を含むstep単価比qへ置き換え、`alpha=c2_SOS/c2_strong`、追加反復倍率chiとすれば

\[
\mathrm{quantum\ cost\ ratio}\approx\chi q\sqrt\alpha.
\tag{26}
\]

追加量子情報取得の正規化価格rho_Qがあれば右辺に加える。古典SDP時間・メモリを量子gate数へ無定義に加算しない。古典・量子のresource vectorまたは明示換算価格を使う。参考の2倍改善なら `chi q sqrt(alpha)+rho_Q≤1/2` が必要だが、**採択gateではない**。q>1ならgapだけを一桁改善しても十分とは限らない。Eq.26へ漸近の `K_SOS G` を代入しただけでは保証付き総費用比にならない。

### 8.3 価格の支配項

| 要素 | 改善が残るための条件 | 現状 |
|---|---|---|
| SOS gap | PF-native化・残余shift後にもGが十分小さい | 文献の一般SOS値では代用不可、未測定 |
| K_SOS・norm | M、幅、順序、非可換coupling増を含めた積K Gが改善 | 古典的な評価規則は得たが実値なし |
| 有限h | r4とgap条件がleading gainを打ち消さない | 条件付き粗い十分式のみ、有用性未確定 |
| 回路単価 | Givens、RZ/RZZ、control、fusionを含むqが回収可能 | 静的構成可能、compileなし |
| FT synthesis | 全問い合わせで合成誤差予算を満たすbit精度 | gate数増で一角あたり精度も厳しくなり得る |
| 古典生成 | Gram構築・SDP・因子化・証拠検証を含め許容価格 | 新規SDPなし、実時間なし |
| overlap・readout | PF branchへの成功率下界を同条件で確保 | 元の厳密ground前提だけでは自動1にならない |

Hermitian一体basisの次元m=O(n²)ではGram保存O(n⁴)、dense因子化O(n⁶)、M平方の軌道対角化O(M n³)。一般rankがO(n²)なら平方phase群はO(M n²)で増える可能性がある。これらは最悪の静的演算量であり、DF rankの実際のscalingやSDP全費用を確定したものではない。rank制約・native費用目的は一般に非凸となり得る。既知SOS SDPを解くだけで最安PF分割を得るとは言えない。

また、認証済み古典区間 `[L,U]` が `U−L≤2epsilon_E` なら、中点だけでenergy精度を満たせる。将来の比較では**古典certificate単独**も考慮する。極小gapを得る古典作業が既に問題を解いているのに、それを無料入力とした量子優位は主張しない。

**費用判定**：保証改善と分割変更による誤差改善という機構は存在する。前者の大部分は既知理論を用いる同回路の評価であり、後者に有利な独立生成則は未特定。静的監査から `NO_COST_LEVER` という一般否定はできないが、具体的な正の総費用利益も確認できない。

## 9. K0-5：結果分類、K1の情報価値、停止

| 項目 | 今回の判定 | 残った問題 |
|---|---|---|
| K0-1 先行性 | 基本部品は `REDUCES_TO_KNOWN`。全体の新規性は未確定 | L9も含む既知法と区別できる生成則・有用な保証が必要 |
| K0-2 数学 | 漸近係数・明示K・条件付き有限h接続は成立。実用保証は `MATHEMATICALLY_UNCLOSED` | 化学で有利な剰余・gap情報・有効h領域・証拠取得価格 |
| K0-3 表現 | AとHermitian一体平方の構成可能性は支持 | 一般SOSのgap/rank保存は否定。B/Cの有利な具体生成は未確定 |
| K0-4 費用 | `UNDETERMINED_DATA_REQUIRED` | step単価、実bias、利用可能certificateを同情報で比較する証拠なし |
| K0-5 次段設計 | 条件付き草案を作成、実行しない | 生成則か保証研究を主題にするかのGPT科学判断が先 |

全体状態は `UNDETERMINED_DATA_REQUIRED`。これは「直ちに数値を取ればよい」という推奨ではない。**未特定の構成原理は数値不足とは別問題**で、一般SOSの既知SDPをとりあえず実行しても、新しい構成の価値を判定できない。K1を検討するなら[草案](k1_design_draft.md)の開始前条件を先に満たす。

研究価値が残る結果は、同じ情報・精度・成功率で、(a)新たな生成則による分割が既知PSD-DF/強いPFを上回る、または(b)既存の強い解析が扱えない有用な有限h領域を取得可能なcertificateで保証すること。そのどちらを成果と認めるかはGPT側の科学判断を要する。補題の再確認、同回路のscalar shift、一般SOS gapだけの改善は独立算法の成功としない。

停止理由はK0範囲を完了したことと、中心的な生成差・実用保証・費用の根拠が未確定なことである。資料不足だけによる科学的否定、全SOS-PFの不能性、正式研究終了は宣言しない。

## 10. 成果と実施境界

成果は本監査、[文献registry](literature_registry.json)、[method status](method_status.json)、[GPT引継ぎ](gpt_handoff.md)、[条件付きK1草案](k1_design_draft.md)、[provenance](provenance.json)、入力原文と保全記録。root既存trackedファイルのbyte保全をmetadataとして検査する。これは科学数値の再検証ではない。

新分子生成、SDP求解、一体を含む固有値計算、期待値/RDM計算、保存数値再集計、PFシミュレーション、optimizer/certificate探索、compile、T合成、resource benchmarkは**すべて0**。紙上の式・計算量・損益分岐は実験結果ではない。外部論文はHTMLの関連節を取得して読み、全PDFのbyte保存・全付録の再証明はしていない。Claude原本は未取得。

既存研究の決定・validation manifest・source/results/認可/STOP・索引は変更していない。今回は監査資料の新規追加だけを許可されたscopeと解釈し、既存規範文書へ新採択を反映する一般運用は適用しない。公開済み／CI固定済みの証拠とも言わない。

**K0 COMPLETE / MANDATORY STOP ACTIVE。K1、新候補採択、追加科学計算、候補B/C、旧D、commit/pushは開始しない。** 次は[GPT引継ぎ](gpt_handoff.md)に基づく科学判断である。
